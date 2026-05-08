import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

import httpx


class LightningAudioClient:
    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.endpoint = endpoint or os.getenv("LIGHTNING_AUDIO_ENDPOINT", "http://localhost:8000/predict")
        self.timeout = httpx.Timeout(timeout_seconds or float(os.getenv("LIGHTNING_AUDIO_TIMEOUT_SECONDS", "7200")))

    async def process_path(
        self,
        audio_path: Path,
        project_id: str,
        meeting_id: str,
        match_score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        with audio_path.open("rb") as audio_file:
            return await self.process_stream(
                audio_file,
                filename=audio_path.name,
                project_id=project_id,
                meeting_id=meeting_id,
                match_score_threshold=match_score_threshold,
            )

    async def process_stream(
        self,
        audio_file: BinaryIO,
        filename: str,
        project_id: str,
        meeting_id: str,
        match_score_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "project_id": project_id,
            "meeting_id": meeting_id,
        }
        if match_score_threshold is not None:
            data["match_score_threshold"] = str(match_score_threshold)

        files = {"file": (filename, audio_file, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint, data=data, files=files)
            response.raise_for_status()
            return response.json()
