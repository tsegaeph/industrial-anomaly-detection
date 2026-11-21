"""Image- and pixel-level anomaly detection metrics."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def image_scores(maps: np.ndarray) -> np.ndarray:
    """Image-level anomaly score = maximum of the (smoothed) anomaly map."""
    return maps.reshape(len(maps), -1).max(axis=1)


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3, n_bins: int = 5000) -> float:
    """Area under the per-region-overlap curve for FPR in [0, fpr_limit], normalised to [0, 1].

    PRO at a threshold = mean over connected ground-truth regions of the fraction of the
    region's pixels flagged. FPR is measured over all ground-truth-normal pixels.
    Implemented with score histograms (n_bins thresholds between min and max score).
    """
    edges = np.linspace(float(maps.min()), float(maps.max()), n_bins + 1)

    def count_ge(values: np.ndarray) -> np.ndarray:
        """Number of values >= each threshold edges[:-1]."""
        hist, _ = np.histogram(values, bins=edges)
        return hist[::-1].cumsum()[::-1]

    normal_pixels = maps[~masks]
    fpr = count_ge(normal_pixels) / normal_pixels.size

    pro_sum = np.zeros(n_bins, dtype=np.float64)
    n_regions = 0
    for m, s in zip(masks, maps):
        if not m.any():
            continue
        n_comp, comp = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        for c in range(1, n_comp):
            region = comp == c
            pro_sum += count_ge(s[region]) / region.sum()
            n_regions += 1
    if n_regions == 0:
        return float("nan")
    pro = pro_sum / n_regions

    # Add the threshold above the maximum (FPR=0, PRO=0) and order by increasing FPR.
    fpr = np.append(fpr, 0.0)[::-1]
    pro = np.append(pro, 0.0)[::-1]
    keep = fpr <= fpr_limit
    x = np.append(fpr[keep], fpr_limit)
    y = np.append(pro[keep], np.interp(fpr_limit, fpr, pro))
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0)
    return float(area / fpr_limit)


def evaluate_category(maps: np.ndarray, masks: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """All metrics for one category. maps N×H×W float, masks N×H×W bool, labels N in {0,1}."""
    if len(np.unique(labels)) < 2:
        raise ValueError("Test set must contain both normal and defective images")
    scores = image_scores(maps)
    flat_masks, flat_maps = masks.reshape(-1), maps.reshape(-1)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(flat_masks, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_masks, flat_maps)),
        "aupro": aupro(maps, masks),
    }