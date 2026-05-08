import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import logging
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from ..schemas import DiarizationOutput, Segment
from .pinecone_store import PineconeSpeakerStore
from .speaker_embedding import SpeakerEmbeddingExtractor

logger = logging.getLogger("Diarizer")


class DiarizationService:
    """
    Labels each transcript segment by comparing segment audio embeddings to
    enrolled speakers in Pinecone (namespace = project_id).
    """

    def __init__(self) -> None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("HF_TOKEN not found in environment variables.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing diarization (embedding + Pinecone) on %s...", self.device)

        self.extractor = SpeakerEmbeddingExtractor(hf_token, self.device)
        self.store = PineconeSpeakerStore()
        self.default_min_score = float(os.getenv("DIARIZATION_PINECONE_MIN_SCORE", "0.65"))

    async def process_audio_segments(
        self,
        audio_path: str,
        segments: List[Segment],
        project_id: str,
        match_score_threshold: Optional[float] = None,
    ) -> DiarizationOutput:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        min_score = self.default_min_score if match_score_threshold is None else float(match_score_threshold)

        audio_data, samplerate = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)

        pinecone_ready = self.store.enabled
        if not pinecone_ready:
            logger.warning("Pinecone unavailable; all segments labeled Unknown.")

        logger.info("Embedding %s segments for speaker matching (namespace=%s)...", len(segments), project_id)

        final_segments: List[Segment] = []
        for seg in segments:
            new_seg = seg.model_copy()
            try:
                start_sample = int(seg.start * samplerate)
                end_sample = int(seg.end * samplerate)
                chunk = audio_data[start_sample:end_sample]
                if chunk.size < int(0.05 * samplerate):
                    new_seg.speaker = "Unknown"
                    final_segments.append(new_seg)
                    continue

                emb = self.extractor.embed_mono_chunk(chunk, samplerate)
                vec = emb.astype(float).tolist()

                if pinecone_ready:
                    resolved = self.store.query_best_speaker(
                        namespace=project_id,
                        vector=vec,
                        top_k=1,
                        min_score=min_score,
                    )
                    new_seg.speaker = resolved[0] if resolved else "Unknown"
                else:
                    new_seg.speaker = "Unknown"
            except Exception as exc:
                logger.warning("Skipping speaker match at %.2fs: %s", seg.start, exc)
                new_seg.speaker = "Unknown"
            final_segments.append(new_seg)

        return DiarizationOutput(segments=final_segments)
