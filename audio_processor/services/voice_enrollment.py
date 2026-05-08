import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from fastapi import UploadFile

from .pinecone_store import PineconeSpeakerStore
from .preprocessor import AudioPreprocessor
from .speaker_embedding import SpeakerEmbeddingExtractor

logger = logging.getLogger(__name__)


class VoiceEnrollmentService:
    """Embed enrollment audio and upsert into Pinecone under namespace ``project_id``."""

    def __init__(self) -> None:
        self.store = PineconeSpeakerStore()
        self.preprocessor = AudioPreprocessor()

    async def enroll_user_voice(self, file: UploadFile, speaker_id: str, project_id: str) -> bool:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.error("HF_TOKEN is not set; cannot compute speaker embedding.")
            return False
        if not self.store.enabled:
            logger.error("Pinecone is not configured.")
            return False

        suffix = Path(file.filename or "enroll.wav").suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(prefix="enroll-", suffix=suffix, delete=False)
        tmp_path = tmp.name
        standardized_path: Optional[str] = None
        try:
            tmp.close()
            contents = await file.read()
            Path(tmp_path).write_bytes(contents)

            standardized_path = self.preprocessor.process(tmp_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            extractor = SpeakerEmbeddingExtractor(hf_token, device)
            waveform, sr = extractor.embed_file_mono(standardized_path)
            embedding = extractor.embed_mono_chunk(waveform, sr)
            vector_id = str(speaker_id).strip()
            namespace = str(project_id).strip()

            self.store.upsert_speaker(
                namespace=namespace,
                vector_id=vector_id,
                vector=embedding.astype(float).tolist(),
                metadata={
                    "speaker_id": vector_id,
                    "speaker_label": vector_id,
                },
            )
            logger.info("Enrolled speaker %s in Pinecone namespace %s", vector_id, namespace)
            return True
        except Exception:
            logger.exception("Voice enrollment failed")
            return False
        finally:
            if standardized_path:
                AudioPreprocessor.cleanup(standardized_path)
            Path(tmp_path).unlink(missing_ok=True)

