from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ProcessedImage:
    tensor: np.ndarray
    masked_preview: np.ndarray


def preprocess_image(raw_bytes: bytes, image_size: int = 224) -> ProcessedImage:
    np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = _foreground_mask(rgb)
    segmented = cv2.bitwise_and(rgb, rgb, mask=mask)

    resized = cv2.resize(segmented, (image_size, image_size), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

    return ProcessedImage(tensor=tensor.astype(np.float32), masked_preview=segmented)


def _foreground_mask(rgb_image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 25, 25], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    raw_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    if cleaned.mean() < 10:
        # fallback to saliency-style center prior if plant color mask is weak
        h, w = cleaned.shape
        fallback = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(fallback, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 255, -1)
        return fallback
    return cleaned
