"""Image-level decision thresholds from held-out normal scores, and thresholded metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.evaluation.analysis import load_maps
from src.evaluation.metrics import image_scores


def threshold_rules(heldout_scores: np.ndarray) -> dict[str, float]:
    """Rules that use normal data only. All are fixed in advance."""
    s = np.asarray(heldout_scores, dtype=np.float64)
    std = s.std(ddof=1) if len(s) > 1 else 0.0
    return {"max_heldout": float(s.max()), "p99": float(np.percentile(s, 99)),
            "mean_plus_3std": float(s.mean() + 3.0 * std)}


def classification_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Image is flagged defective when score > threshold."""
    pred = scores > threshold
    tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum()); tn = int((~pred & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fp / (fp + tn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def oracle_best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """Best F1 over all thresholds, chosen on the test set itself. An upper bound, NOT a real result."""
    p, r, _ = precision_recall_curve(labels, scores)
    return float(np.max(2 * p * r / np.clip(p + r, 1e-12, None)))


def evaluate_thresholds(exp_dir: str | Path, categories,
                        calib_file: str = "calibration_heldout_scores.json") -> pd.DataFrame:
    """One row per (category, rule). Reads saved test maps and the given calibration scores file."""
    exp_dir = Path(exp_dir)
    calib = json.loads((exp_dir / calib_file).read_text())
    rows = []
    for cat in categories:
        d = load_maps(exp_dir, cat)
        scores, labels = image_scores(d["maps"].astype(np.float32)), d["labels"]
        oracle = oracle_best_f1(scores, labels)
        for rule, thr in threshold_rules(np.array(calib[cat]["scores"])).items():
            rows.append({"category": cat, "rule": rule, "threshold": thr,
                         **classification_metrics(scores, labels, thr),
                         "oracle_f1": oracle, "n_heldout": calib[cat]["n_heldout"]})
    return pd.DataFrame(rows)