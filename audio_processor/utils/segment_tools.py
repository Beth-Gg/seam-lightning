from typing import List

from ..schemas import Segment

def clean_segments(segments: List[Segment], min_duration: float = 0.8) -> List[Segment]:
    """
    Consolidates transcription segments to improve diarization accuracy.
    Preserves the 'Identity Noise' prevention logic.
    """
    if not segments:
        return []

    cleaned: List[Segment] = []
    
    for seg in segments:
        duration = seg.end - seg.start
        
        if duration < min_duration and cleaned:
            last_seg = cleaned[-1]
            # Extend last segment to cover this one
            last_seg.end = seg.end
            last_seg.text = f"{last_seg.text} {seg.text}".strip()
        else:
            # Add a copy to avoid mutating the original input list
            cleaned.append(seg.model_copy())
            
    return cleaned