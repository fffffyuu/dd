from typing import List, Optional
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    scientific_name: str
    common_name: str
    confidence: float = Field(ge=0, le=1)


class CareProfile(BaseModel):
    watering: str
    sunlight: str
    soil: str
    temperature: str


class DiseaseReport(BaseModel):
    likely_disease: str
    confidence: float = Field(ge=0, le=1)
    diagnosis: str
    treatment: str


class PlantIdentifyResponse(BaseModel):
    top_predictions: List[Prediction]
    selected: Prediction
    description: str
    care: CareProfile
    disease_report: Optional[DiseaseReport] = None
    processing_ms: float


class HealthCheckResponse(BaseModel):
    status: str
