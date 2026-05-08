import subprocess
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app.audio.preprocessor")
class AudioPreprocessor:
    def __init__(self, output_format="wav", sample_rate=16000, channels=1):
        """
        Standardizes audio for ML pipelines.
        Target: 16kHz, Mono, 16-bit PCM WAV.
        """
        self.output_format = output_format
        self.sample_rate = sample_rate
        self.channels = channels

    def process(self, input_path: str) -> str:
        """
        Converts WebM, MP4, or WAV to a standardized format.
        Applies loudness normalization and high-pass filtering.
        """
        input_file = Path(input_path)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path = input_file.with_suffix(f".standardized.{self.output_format}")

        # FFmpeg command explanation:
        # -ar 16000: Sets sample rate to 16kHz
        # -ac 1: Downmixes to Mono
        # -af: Audio filters (Highpass to remove rumble + Loudnorm for volume consistency)
        command = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            "-af", (
                "highpass=f=100,"        # Remove low rumble
                "afftdn,"                # Reduce background hiss
                "loudnorm,"              # Equalize voices
                "silenceremove=1:0:-50dB" # Remove long dead air
            ),
            "-c:a", "pcm_s16le",
            str(output_path)
        ]

        try:
            logger.info(f"Preprocessing audio: {input_file.name}")
            subprocess.run(command, check=True, capture_output=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError("Failed to preprocess audio.") from e

    @staticmethod
    def cleanup(file_path: str):
        """Removes temporary processed files."""
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")

# Example Usage:
# processor = AudioPreprocessor()
# clean_audio = processor.process("meeting.m4a")