"""Single-image inference with saved per-category PatchCore models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.dataset import preprocess_image
from src.models.patchcore import PatchCore


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Accept grayscale, RGB or RGBA arrays and return HxWx3 uint8 RGB."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[..., :3]
    return image


@dataclass
class Prediction:
    category: str
    score: float
    threshold: float
    is_anomalous: bool
    anomaly_map: np.ndarray   # S x S float32
    image: np.ndarray         # S x S x 3 uint8, the resized input


class AnomalyPredictor:
    """Loads one PatchCore model per category lazily and applies the category threshold."""

    def __init__(self, models_dir: str | Path, thresholds_path: str | Path,
                 rule: str = "p99", device: str = "cpu") -> None:
        self.models_dir = Path(models_dir)
        self.thresholds = json.loads(Path(thresholds_path).read_text())
        self.rule, self.device = rule, device
        self._models: dict[str, PatchCore] = {}

    @property
    def categories(self) -> list[str]:
        return sorted(self.thresholds)

    def _model(self, category: str) -> PatchCore:
        if category not in self._models:
            self._models[category] = PatchCore.load(self.models_dir / f"model_{category}.pt", device=self.device)
        return self._models[category]

    @torch.no_grad()
    def predict(self, image: np.ndarray, category: str, threshold_scale: float = 1.0) -> Prediction:
        if category not in self.thresholds:
            raise ValueError(f"Unknown category {category!r}; choose from {self.categories}")
        image = to_rgb(image)
        model = self._model(category)
        size = model.cfg["image_size"]
        amap = model.predict(preprocess_image(image, size).unsqueeze(0))[0].numpy()
        threshold = self.thresholds[category][self.rule] * threshold_scale
        score = float(amap.max())
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return Prediction(category, score, threshold, score > threshold, amap, resized)


def render_overlay(pred: Prediction, alpha: float = 0.5) -> np.ndarray:
    """Heatmap over the image. The colour scale runs from 0 to 2x the threshold, so the same
    colour means the same thing across images of a category."""
    norm = np.clip(pred.anomaly_map / (2.0 * pred.threshold), 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return cv2.addWeighted(pred.image, 1 - alpha, heat, alpha, 0)