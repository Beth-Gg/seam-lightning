import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import litserve as ls
from fastapi import HTTPException

from .services.diarization_service import DiarizationService
from .services.pipeline import AudioTooLongError, SeamAudioPipeline
from .services.preprocessor import AudioPreprocessor
from .services.transcriber import TranscriptionEngine

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("app.litserve")


class SeamAudioLitAPI(ls.LitAPI):
    def __init__(self):
        super().__init__(
            api_path="/predict",
            max_batch_size=int(os.getenv("LITSERVE_MAX_BATCH_SIZE", "2")),
            batch_timeout=float(os.getenv("LITSERVE_BATCH_TIMEOUT", "0.25")),
        )
        self.max_upload_bytes = int(os.getenv("MAX_AUDIO_UPLOAD_MB", "500")) * 1024 * 1024

    def setup(self, device):
        serving_device = "cuda" if str(device).startswith("cuda") else "cpu"
        logger.info("Starting Seam AI audio pipeline on %s", serving_device)

        transcriber = TranscriptionEngine(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            device=serving_device,
        )

        diarizer = None
        if os.getenv("HF_TOKEN"):
            diarizer = DiarizationService()
        else:
            logger.warning("HF_TOKEN is not set; diarization/embeddings disabled.")

        self.pipeline = SeamAudioPipeline(
            preprocessor=AudioPreprocessor(),
            transcriber=transcriber,
            diarizer=diarizer,
            max_audio_seconds=float(os.getenv("MAX_AUDIO_SECONDS", "3600")),
            default_match_score_threshold=float(os.getenv("DIARIZATION_PINECONE_MIN_SCORE", "0.65")),
        )

    def decode_request(self, request) -> Dict[str, Any]:
        upload = request.get("file") or request.get("audio_file") or request.get("audio")
        if upload is None:
            raise HTTPException(status_code=400, detail="Upload an audio file using form field 'file'.")

        project_id = request.get("project_id")
        meeting_id = request.get("meeting_id")
        if project_id in (None, ""):
            raise HTTPException(status_code=400, detail="Form field 'project_id' is required.")
        if meeting_id in (None, ""):
            raise HTTPException(status_code=400, detail="Form field 'meeting_id' is required.")

        suffix = Path(getattr(upload, "filename", "") or "audio").suffix or ".wav"
        raw_file = tempfile.NamedTemporaryFile(prefix="seam-audio-", suffix=suffix, delete=False)
        bytes_written = 0

        try:
            with raw_file:
                while chunk := upload.file.read(1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > self.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Audio upload is too large.")
                    raw_file.write(chunk)

            return {
                "audio_path": raw_file.name,
                "project_id": str(project_id).strip(),
                "meeting_id": str(meeting_id).strip(),
                "match_score_threshold": self._optional_float(
                    request.get("match_score_threshold") or request.get("distance_threshold")
                ),
            }
        except Exception:
            Path(raw_file.name).unlink(missing_ok=True)
            raise
        finally:
            upload.file.close()

    def batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return inputs

    def predict(self, inputs: Union[Dict[str, Any], List[Dict[str, Any]]]):
        if isinstance(inputs, list):
            return [self._predict_one(item) for item in inputs]
        return self._predict_one(inputs)

    def unbatch(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return outputs

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "transcript": output["transcript"],
            "speaker_segments": output["segments"],
            "metadata": output["metadata"],
        }

    def _predict_one(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = payload["audio_path"]
        try:
            return self.pipeline.run(
                audio_path,
                project_id=payload["project_id"],
                meeting_id=payload["meeting_id"],
                match_score_threshold=payload.get("match_score_threshold"),
            )
        except AudioTooLongError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Audio pipeline failed")
            raise HTTPException(status_code=500, detail="Audio pipeline failed.") from exc
        finally:
            Path(audio_path).unlink(missing_ok=True)

    def _optional_float(self, value):
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="match_score_threshold must be a number.",
            ) from exc


if __name__ == "__main__":
    api = SeamAudioLitAPI()
    devices = os.getenv("LITSERVE_DEVICES", "auto")
    server = ls.LitServer(
        api,
        accelerator="auto",
        devices=devices if devices == "auto" else int(devices),
        workers_per_device=int(os.getenv("LITSERVE_WORKERS_PER_DEVICE", "1")),
        timeout=float(os.getenv("LITSERVE_TIMEOUT_SECONDS", "7200")),
    )
    server.run(host=os.getenv("LITSERVE_HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
