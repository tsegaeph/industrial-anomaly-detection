"""Dataset inspection figures."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.mvtec import load_image, load_mask


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint defect pixels red and draw the defect outline."""
    out = image.astype(np.float32).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    out[mask] = 0.5 * out[mask] + 0.5 * red
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 2)
    return out


def plot_category(df: pd.DataFrame, category: str, n_defect_types: int = 2,
                  seed: int = 42, save_dir: Path | None = None) -> None:
    """One row: normal test image, then (image | mask | overlay) for n defect types."""
    rng = np.random.default_rng(seed)
    cat = df[(df.category == category) & (df.split == "test")]
    normal = cat[cat.label == 0].sample(1, random_state=seed).iloc[0]
    types = sorted(cat[cat.label == 1].defect_type.unique())
    chosen = list(rng.choice(types, size=min(n_defect_types, len(types)), replace=False))

    ncols = 1 + 3 * len(chosen)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.2))
    img = load_image(normal.image_path)
    axes[0].imshow(img); axes[0].set_title("normal", fontsize=9)

    col = 1
    for dt in chosen:
        row = cat[cat.defect_type == dt].sample(1, random_state=seed).iloc[0]
        img = load_image(row.image_path)
        mask = load_mask(row.mask_path, img.shape[:2])
        axes[col].imshow(img);                     axes[col].set_title(dt, fontsize=9)
        axes[col + 1].imshow(mask, cmap="gray");   axes[col + 1].set_title("mask", fontsize=9)
        axes[col + 2].imshow(make_overlay(img, mask)); axes[col + 2].set_title("overlay", fontsize=9)
        col += 3
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{category}  ({img.shape[1]}x{img.shape[0]})", fontsize=11)
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"visual_{category}.png", dpi=120, bbox_inches="tight")
    plt.show()


def defect_area_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of image area covered by defect, per category and defect type."""
    rows = []
    for r in df[df.label == 1].itertuples():
        mask = load_mask(r.mask_path, load_image(r.image_path).shape[:2])
        rows.append({"category": r.category, "defect_type": r.defect_type, "area_frac": mask.mean()})
    stats = (pd.DataFrame(rows)
             .groupby(["category", "defect_type"])["area_frac"]
             .agg(n="count", min="min", median="median", max="max")
             .reset_index())
    return stats