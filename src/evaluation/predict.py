"""Run a model over a dataloader and collect maps, masks and labels as numpy arrays."""
from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader


def collect_predictions(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (maps N×S×S float32, masks N×S×S bool, labels N int). Image score = maps.max over H,W."""
    maps, masks, labels = [], [], []
    for batch in loader:
        maps.append(model.predict(batch["image"]).numpy())
        masks.append(batch["mask"][:, 0].numpy() > 0.5)
        labels.append(batch["label"].numpy())
    return np.concatenate(maps), np.concatenate(masks), np.concatenate(labels)