from typing import List
from pydantic import BaseModel, Field

class Segment(BaseModel):
    start: float
    end: float
    speaker: str = "SPEAKER_00"
    text: str = ""

class TranscriptionOutput(BaseModel):
    transcript: str = ""
    segments: List[Segment] = Field(default_factory=list)

class DiarizationOutput(BaseModel):
    segments: List[Segment] = Field(default_factory=list)

# Internal contract for the transcription service
class TranscriptSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str

class TranscriptionResult(BaseModel):
    transcript: str
    segments: List[TranscriptSegment] = Field(default_factory=list)
    diarization_enabled: bool = False