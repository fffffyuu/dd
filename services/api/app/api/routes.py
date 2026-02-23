from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.plant import HealthCheckResponse, PlantIdentifyResponse
from app.services.inference import InferenceService
from app.services.preprocessing import preprocess_image

router = APIRouter()
inference_service = InferenceService()


@router.get("/health", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")


@router.post("/identify", response_model=PlantIdentifyResponse)
async def identify_plant(image: UploadFile = File(...)) -> PlantIdentifyResponse:
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    raw = await image.read()
    try:
        processed = preprocess_image(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return inference_service.identify(processed.tensor)
