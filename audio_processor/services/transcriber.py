import torch
import logging
from faster_whisper import WhisperModel
from typing import Dict, List, Any

# Set up structured logging
logger = logging.getLogger("app.audio.whisper")
#set model_size to large_v3 on prod or later or even fetch it from .env
class TranscriptionEngine:
    def __init__(self, model_size: str = "small", device: str | None = None):
        """
        Production-grade Whisper wrapper using CTranslate2.
        - large-v3: Best for multilingual (Amharic + English)
        - turbo: Faster, slightly lower accuracy
        """
        # Auto-detect GPU if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # compute_type: "float16" for GPU (fast), "int8" for CPU (efficient)
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        logger.info(f"Loading Whisper model: {model_size} on {self.device}...")
        self.model_size = model_size
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)

    def transcribe(self, audio_path: str, language: str | None = None) -> Dict[str, Any]:
        """
        Transcribes preprocessed audio and returns word-level timestamps.
        
        Args:
            audio_path: Path to the 16kHz Mono WAV file.
            language: ISO code (e.g. am, en) or None for automatic detection (mixed Amharic/English).
        """
        logger.info(f"Starting transcription for: {audio_path}")

        # beam_size=5 is a good balance between accuracy and speed
        # word_timestamps=True is essential for speaker diarization alignment
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        formatted_segments = []
        full_text = []

        for segment in segments:
            segment_data = {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                # "words": [
                #     {"start": w.start, "end": w.end, "word": w.word} 
                #     for w in (segment.words or [])
                # ]
            }
            formatted_segments.append(segment_data)
            full_text.append(segment.text.strip())

        logger.info(f"Transcription complete. Detected language: {info.language} ({info.language_probability:.2f})")

        return {
            "metadata": {
                "language": info.language,
                "duration": info.duration,
                "model": f"faster-whisper-{self.model_size}",
            },
            "segments": formatted_segments,
            "full_text": " ".join(full_text)
        }

# Example Integration:
# engine = TranscriptionEngine()
# result = engine.transcribe("meeting.standardized.wav")
# print(result['segments'])