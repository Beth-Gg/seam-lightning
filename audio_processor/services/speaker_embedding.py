import torchaudio

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: None  # type: ignore[attr-defined]

import logging
from typing import Tuple

import numpy as np
import torch

from pyannote.audio import Inference, Model

logger = logging.getLogger(__name__)


class SpeakerEmbeddingExtractor:
    """Shared pyannote/embedding extractor for diarization and voice enrollment."""

    def __init__(self, hf_token: str, device: torch.device):
        if not hf_token:
            raise RuntimeError("HF_TOKEN is required for speaker embeddings.")
        self.device = device
        logger.info("Loading pyannote/embedding on %s...", device)
        try:
            self.model = Model.from_pretrained("pyannote/embedding", token=hf_token)
        except TypeError:
            self.model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
        self.model.to(device)
        self.inference = Inference(self.model, window="whole", device=device)

    def embed_mono_chunk(self, mono_audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """mono_audio: 1-D float waveform."""
        tensor_chunk = torch.from_numpy(mono_audio.astype(np.float32)).unsqueeze(0)
        emb = self.inference({"waveform": tensor_chunk, "sample_rate": sample_rate})
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    def embed_file_mono(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load file as mono float32; returns (waveform, sample_rate)."""
        import soundfile as sf

        audio_data, samplerate = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        return audio_data.astype(np.float32), int(samplerate)
