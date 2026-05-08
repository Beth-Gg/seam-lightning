from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .services.voice_enrollment import VoiceEnrollmentService

router = APIRouter()


def get_voice_enrollment_service() -> VoiceEnrollmentService:
    return VoiceEnrollmentService()


@router.post("/enroll-voice")
async def enroll_voice(
    project_id: str = Form(...),
    speaker_id: str = Form(...),
    file: UploadFile = File(...),
):
    service = get_voice_enrollment_service()
    success = await service.enroll_user_voice(file, speaker_id, project_id)
    if not success:
        raise HTTPException(status_code=500, detail="Voice enrollment failed.")
    return {"status": "success", "project_id": project_id, "speaker_id": speaker_id}
