"""MVTec AD indexing and image/mask loading utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

CATEGORIES: tuple[str, ...] = (
    "bottle", "capsule", "carpet", "metal_nut", "screw", "toothbrush", "zipper",
)


def build_index(root: str | Path, categories: Iterable[str] = CATEGORIES) -> pd.DataFrame:
    """Scan the MVTec AD folder tree and return one row per image.

    Columns: category, split, defect_type, label (0 normal / 1 defective),
    image_path, mask_path (None for normal images).
    Raises if a category/split is missing, a defective image has no mask,
    or any training image is not 'good' (would indicate leakage).
    """
    root = Path(root)
    rows: list[dict] = []
    for cat in categories:
        cat_dir = root / cat
        for split in ("train", "test"):
            split_dir = cat_dir / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing directory: {split_dir}")
            for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                defect_type = defect_dir.name
                is_good = defect_type == "good"
                if split == "train" and not is_good:
                    raise ValueError(f"Non-normal data in train split: {defect_dir}")
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if split == "test" and not is_good:
                        mask_path = cat_dir / "ground_truth" / defect_type / f"{img_path.stem}_mask.png"
                        if not mask_path.is_file():
                            raise FileNotFoundError(f"Missing mask for {img_path}")
                    rows.append(
                        {
                            "category": cat,
                            "split": split,
                            "defect_type": defect_type,
                            "label": 0 if is_good else 1,
                            "image_path": str(img_path),
                            "mask_path": str(mask_path) if mask_path else None,
                        }
                    )
    df = pd.DataFrame(rows)
    logger.info("Indexed %d images across %d categories", len(df), df["category"].nunique())
    return df


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as HxWx3 uint8 RGB (grayscale is replicated to 3 channels)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_mask(path: str | Path | None, shape_hw: tuple[int, int]) -> np.ndarray:
    """Load a ground-truth mask as a boolean HxW array (any non-zero pixel = defect).

    If path is None (normal image), returns an all-False mask of shape_hw.
    """
    if path is None:
        return np.zeros(shape_hw, dtype=bool)
    with Image.open(path) as im:
        mask = np.asarray(im.convert("L")) > 0
    if mask.shape != shape_hw:
        raise ValueError(f"Mask shape {mask.shape} != image shape {shape_hw}: {path}")
    return mask