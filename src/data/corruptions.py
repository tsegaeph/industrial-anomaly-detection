"""Test-time image corruptions for the robustness experiment (applied to the resized uint8 image)."""
from __future__ import annotations

import functools

import cv2
import numpy as np


def gaussian_noise(img: np.ndarray, level: float, seed: int = 0) -> np.ndarray:
    """Additive Gaussian noise; level = standard deviation on the 0-255 scale."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0.0, level, img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def brightness(img: np.ndarray, level: float, seed: int = 0) -> np.ndarray:
    """Multiplicative brightness change; level 1.0 = unchanged."""
    return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, level: float, seed: int = 0) -> np.ndarray:
    """Gaussian blur; level = sigma in pixels of the resized image."""
    return cv2.GaussianBlur(img, (0, 0), level)


CORRUPTIONS = {"noise": gaussian_noise, "brightness": brightness, "blur": blur}

# (kind, level). "clean" is the uncorrupted reference, which must reproduce the normal results.
ROBUSTNESS_SETTINGS: list[tuple[str, float]] = [
    ("clean", 0.0),
    ("noise", 10), ("noise", 25), ("noise", 50),
    ("brightness", 0.6), ("brightness", 0.8), ("brightness", 1.2), ("brightness", 1.4),
    ("blur", 1.0), ("blur", 2.0),
]


def make_corruption(kind: str, level: float):
    """Return a callable (img, seed=...) -> img for the given corruption and level."""
    return functools.partial(CORRUPTIONS[kind], level=level)