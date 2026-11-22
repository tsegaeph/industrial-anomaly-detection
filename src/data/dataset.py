"""PyTorch dataset and preprocessing for MVTec AD."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.mvtec import load_image, load_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# A corruption maps (resized uint8 HxWx3 image, seed=int) -> uint8 image of the same shape.
Corruption = Callable[..., np.ndarray]


def preprocess_image(image: np.ndarray, image_size: int, corruption: Corruption | None = None,
                     seed: int = 0) -> torch.Tensor:
    """HxWx3 uint8 RGB -> normalized 3 x S x S float tensor (full image, no crop).

    An optional corruption (robustness experiments only) is applied to the resized uint8 image.
    """
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if corruption is not None:
        resized = corruption(resized, seed=seed)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def preprocess_mask(mask: np.ndarray, image_size: int) -> torch.Tensor:
    """HxW bool -> 1 x S x S float tensor in {0, 1}.

    Area interpolation then a 0.5 threshold keeps thin defects better than nearest-neighbor.
    """
    resized = cv2.resize(mask.astype(np.float32), (image_size, image_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((resized >= 0.5).astype(np.float32)).unsqueeze(0)


class MVTecDataset(Dataset):
    """Dataset over rows of the index DataFrame produced by `build_index`.

    Each item is a dict: image (3,S,S), mask (1,S,S), label (0/1), category,
    defect_type, image_path.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 256, corruption: Corruption | None = None) -> None:
        if len(df) == 0:
            raise ValueError("Empty DataFrame passed to MVTecDataset")
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = load_image(row.image_path)
        mask = load_mask(row.mask_path, image.shape[:2])
        return {
            "image": preprocess_image(image, self.image_size, self.corruption, seed=idx),
            "mask": preprocess_mask(mask, self.image_size),
            "label": int(row.label),
            "category": row.category,
            "defect_type": row.defect_type,
            "image_path": row.image_path,
        }


def get_split(df: pd.DataFrame, category: str, split: str) -> pd.DataFrame:
    """Rows of one category and split. Train is guaranteed to contain only normal images."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    out = df[(df.category == category) & (df.split == split)]
    if split == "train" and (out.label != 0).any():
        raise ValueError("Defective image found in train split (leakage)")
    return out.reset_index(drop=True)


def subsample_normals(train_df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Seeded random subset of n normal training images (few-shot experiment)."""
    if n >= len(train_df):
        return train_df
    return train_df.sample(n=n, random_state=seed).reset_index(drop=True)


def make_loader(df: pd.DataFrame, image_size: int = 256, batch_size: int = 16, shuffle: bool = False,
                num_workers: int = 2, corruption: Corruption | None = None) -> DataLoader:
    """DataLoader over a split. Colab has few CPU cores, so keep num_workers small."""
    return DataLoader(MVTecDataset(df, image_size, corruption), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())