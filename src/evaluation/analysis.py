"""Post-hoc analysis of saved experiment outputs (maps_<category>.npz)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_maps(exp_dir: str | Path, category: str) -> dict[str, np.ndarray]:
    """Load saved maps (float16), labels, image paths, defect types. Our own files, so pickle is safe."""
    z = np.load(Path(exp_dir) / f"maps_{category}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _scores(maps: np.ndarray) -> np.ndarray:
    return maps.astype(np.float32).reshape(len(maps), -1).max(axis=1)


def per_defect_type_auroc(exp_dir: str | Path, categories) -> pd.DataFrame:
    """Image AUROC of each defect type against ALL normal test images of its category.

    Computed from float16-stored maps, so values can differ slightly from results.csv.
    """
    rows = []
    for cat in categories:
        d = load_maps(exp_dir, cat)
        scores, labels, types = _scores(d["maps"]), d["labels"], d["defect_types"].astype(str)
        normal = scores[labels == 0]
        for t in sorted(set(types[labels == 1])):
            s = scores[types == t]
            y = np.r_[np.zeros(len(normal)), np.ones(len(s))]
            rows.append({"category": cat, "defect_type": t, "n": len(s),
                         "image_auroc": roc_auc_score(y, np.r_[normal, s])})
    return pd.DataFrame(rows)


def ranked_errors(exp_dir: str | Path, category: str, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Threshold-free error analysis: the k lowest-scoring defective images (likely misses)
    and the k highest-scoring normal images (likely false alarms)."""
    d = load_maps(exp_dir, category)
    t = pd.DataFrame({"image_path": d["image_paths"].astype(str), "defect_type": d["defect_types"].astype(str),
                      "label": d["labels"], "score": _scores(d["maps"])})
    missed = t[t.label == 1].nsmallest(k, "score").reset_index(drop=True)
    alarms = t[t.label == 0].nlargest(k, "score").reset_index(drop=True)
    return missed, alarms