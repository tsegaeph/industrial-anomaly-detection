"""Heatmap figures from saved experiment outputs."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset import preprocess_mask
from src.data.mvtec import load_image, load_mask
from src.evaluation.analysis import load_maps
from src.visualization.inspection import make_overlay


def show_cases(exp_dir: str | Path, df: pd.DataFrame, category: str, image_paths: list[str],
               save_path: str | Path | None = None) -> None:
    """One row per image: image | ground truth | anomaly heatmap | heatmap over image.

    The colour scale is shared across the category (0 to the 99.9th percentile of all its maps),
    so colours are comparable between rows.
    """
    d = load_maps(exp_dir, category)
    maps, size = d["maps"], d["maps"].shape[-1]
    pos = {str(p): i for i, p in enumerate(d["image_paths"])}
    vmax = float(np.percentile(maps.astype(np.float32), 99.9))
    lookup = df.set_index("image_path")

    fig, axes = plt.subplots(len(image_paths), 4, figsize=(11, 2.8 * len(image_paths)), squeeze=False)
    for r, p in enumerate(image_paths):
        row = lookup.loc[p]
        raw = load_image(p)
        img = cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)
        mask_path = row.mask_path if isinstance(row.mask_path, str) else None
        mask = preprocess_mask(load_mask(mask_path, raw.shape[:2]), size)[0].numpy() > 0.5
        amap = maps[pos[p]].astype(np.float32)

        axes[r, 0].imshow(img);                      axes[r, 0].set_title(f"{row.defect_type}", fontsize=9)
        axes[r, 1].imshow(make_overlay(img, mask));  axes[r, 1].set_title("ground truth", fontsize=9)
        axes[r, 2].imshow(amap, cmap="jet", vmin=0, vmax=vmax)
        axes[r, 2].set_title(f"score {amap.max():.1f}", fontsize=9)
        axes[r, 3].imshow(img); axes[r, 3].imshow(amap, cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        axes[r, 3].set_title("overlay", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(category, fontsize=11)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.show()