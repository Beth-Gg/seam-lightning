import asyncio
import concurrent.futures
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf

from ..schemas import Segment
from .diarization_service import DiarizationService
from .preprocessor import AudioPreprocessor
from .transcriber import TranscriptionEngine
from ..utils.segment_tools import clean_segments

logger = logging.getLogger("app.audio.pipeline")


class AudioTooLongError(ValueError):
    """Raised when an audio file exceeds the configured processing limit."""


class SeamAudioPipeline:
    def __init__(
        self,
        preprocessor: Optional[AudioPreprocessor] = None,
        transcriber: Optional[TranscriptionEngine] = None,
        diarizer: Optional[DiarizationService] = None,
        max_audio_seconds: Optional[float] = None,
        default_match_score_threshold: float = 0.65,
    ):
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.transcriber = transcriber or TranscriptionEngine()
        self.diarizer = diarizer
        self.max_audio_seconds = max_audio_seconds or float(os.getenv("MAX_AUDIO_SECONDS", "3600"))
        self.default_match_score_threshold = float(
            os.getenv("DIARIZATION_PINECONE_MIN_SCORE", str(default_match_score_threshold))
        )

        if self.diarizer is None and os.getenv("HF_TOKEN"):
            self.diarizer = DiarizationService()

    def run(
        self,
        audio_path: str,
        *,
        project_id: str,
        meeting_id: str,
        match_score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        processed_audio_path: Optional[str] = None

        try:
            self._validate_duration(audio_path)
            processed_audio_path = self.preprocessor.process(audio_path)
            self._validate_duration(processed_audio_path)

            # Amharic + English mix: let Whisper auto-detect per segment (language=None).
            transcription = self.transcriber.transcribe(processed_audio_path, language=None)
            transcription_segments = self._segments_from_transcription(transcription.get("segments", []))
            cleaned_segments = clean_segments(transcription_segments)

            diarization_enabled = self.diarizer is not None
            if diarization_enabled:
                threshold = (
                    match_score_threshold
                    if match_score_threshold is not None
                    else self.default_match_score_threshold
                )
                diarized_output = self._run_async(
                    self.diarizer.process_audio_segments(
                        processed_audio_path,
                        cleaned_segments,
                        project_id=project_id,
                        match_score_threshold=threshold,
                    )
                )
                final_segments = diarized_output.segments
            else:
                final_segments = cleaned_segments

            transcript = self._format_transcript(final_segments) or transcription.get("full_text", "")

            return {
                "transcript": transcript,
                "segments": [self._serialize_segment(segment) for segment in final_segments],
                "metadata": {
                    **transcription.get("metadata", {}),
                    "diarization_enabled": diarization_enabled,
                    "project_id": project_id,
                    "meeting_id": meeting_id,
                },
            }
        finally:
            if processed_audio_path and processed_audio_path != audio_path:
                AudioPreprocessor.cleanup(processed_audio_path)

    def _validate_duration(self, audio_path: str) -> None:
        duration = self._get_duration_seconds(audio_path)
        if duration is not None and duration > self.max_audio_seconds:
            raise AudioTooLongError(
                f"Audio duration {duration:.1f}s exceeds limit of {self.max_audio_seconds:.1f}s."
            )

    def _get_duration_seconds(self, audio_path: str) -> Optional[float]:
        try:
            return float(sf.info(audio_path).duration)
        except Exception:
            return self._probe_duration_with_ffprobe(audio_path)

    def _probe_duration_with_ffprobe(self, audio_path: str) -> Optional[float]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(Path(audio_path)),
        ]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            return float(completed.stdout.strip())
        except Exception as exc:
            logger.warning("Could not determine duration for %s: %s", audio_path, exc)
            return None

    def _segments_from_transcription(self, segments: List[Dict[str, Any]]) -> List[Segment]:
        return [Segment.model_validate(segment) for segment in segments]

    def _format_transcript(self, segments: List[Segment]) -> str:
        if self.diarizer:
            return "\n".join(f"{segment.speaker}: {segment.text}" for segment in segments if segment.text).strip()
        return " ".join(segment.text for segment in segments if segment.text).strip()

    def _serialize_segment(self, segment: Segment) -> Dict[str, Any]:
        return segment.model_dump() if hasattr(segment, "model_dump") else dict(segment)

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coroutine)).result()
