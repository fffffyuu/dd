from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.schemas.plant import Prediction, CareProfile, DiseaseReport, PlantIdentifyResponse
from app.services.catalog import get_metadata


CLASS_INDEX = ["fiddle_leaf_fig", "snake_plant", "money_plant"]


class InferenceService:
    def __init__(self) -> None:
        self.session = self._build_session(settings.model_path)
        self.disease_session = self._build_session(settings.disease_model_path)

    @staticmethod
    def _build_session(model_path: str) -> ort.InferenceSession | None:
        if not Path(model_path).exists():
            return None
        providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(model_path, providers=providers)

    def identify(self, tensor: np.ndarray) -> PlantIdentifyResponse:
        started = time.perf_counter()
        probs = self._predict_probs(tensor)

        top_idx = np.argsort(-probs)[:3]
        predictions: list[Prediction] = []
        for idx in top_idx:
            key = CLASS_INDEX[idx]
            meta = get_metadata(key)
            predictions.append(
                Prediction(
                    scientific_name=meta.scientific_name,
                    common_name=meta.common_name,
                    confidence=float(probs[idx]),
                )
            )

        best_key = CLASS_INDEX[top_idx[0]]
        best_meta = get_metadata(best_key)

        disease_report = self._predict_disease(tensor)
        elapsed = (time.perf_counter() - started) * 1000

        return PlantIdentifyResponse(
            top_predictions=predictions,
            selected=predictions[0],
            description=best_meta.description,
            care=CareProfile(
                watering=best_meta.watering,
                sunlight=best_meta.sunlight,
                soil=best_meta.soil,
                temperature=best_meta.temperature,
            ),
            disease_report=disease_report,
            processing_ms=round(elapsed, 2),
        )

    def _predict_probs(self, tensor: np.ndarray) -> np.ndarray:
        if self.session is None:
            return np.array([0.1, 0.2, 0.7], dtype=np.float32)

        input_name = self.session.get_inputs()[0].name
        logits: Any = self.session.run(None, {input_name: tensor})[0][0]
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _predict_disease(self, tensor: np.ndarray) -> DiseaseReport:
        if self.disease_session is None:
            return DiseaseReport(
                likely_disease="healthy",
                confidence=0.88,
                diagnosis="No significant disease markers found.",
                treatment="Maintain airflow and avoid overwatering.",
            )

        input_name = self.disease_session.get_inputs()[0].name
        logits = self.disease_session.run(None, {input_name: tensor})[0][0]
        classes = ["healthy", "powdery_mildew", "leaf_spot"]
        idx = int(np.argmax(logits))
        conf = float(np.exp(logits[idx]) / np.exp(logits).sum())
        disease = classes[idx]

        advice = {
            "healthy": ("No disease symptoms detected.", "Continue current care cycle."),
            "powdery_mildew": (
                "Likely fungal powdery mildew infection.",
                "Isolate plant, reduce humidity, and apply sulfur-based fungicide.",
            ),
            "leaf_spot": (
                "Spotted lesions indicate potential bacterial/fungal leaf spot.",
                "Prune infected leaves and use copper fungicide.",
            ),
        }
        diagnosis, treatment = advice[disease]
        return DiseaseReport(
            likely_disease=disease,
            confidence=conf,
            diagnosis=diagnosis,
            treatment=treatment,
        )
