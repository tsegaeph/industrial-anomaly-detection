"""Smoke test of an exported demo folder: one normal and one defective test image per category."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--space-dir", required=True)
ap.add_argument("--data-root", required=True)
args = ap.parse_args()

space = Path(args.space_dir).resolve()
sys.path.insert(0, str(space))              # use ONLY the exported copy of src/
import torch  # noqa: E402
from src.inference.predictor import AnomalyPredictor  # noqa: E402

print("CPU threads:", torch.get_num_threads())
predictor = AnomalyPredictor(space / "models", space / "thresholds.json", rule="p99")
for cat in predictor.categories:
    test = Path(args.data_root) / cat / "test"
    defect = sorted(d.name for d in test.iterdir() if d.is_dir() and d.name != "good")[0]
    for sub in ("good", defect):
        img = np.asarray(Image.open(sorted((test / sub).glob("*.png"))[0]))
        predictor.predict(img, cat)         # first call includes model loading
        t0 = time.time()
        pred = predictor.predict(img, cat)
        ms = 1000 * (time.time() - t0)
        print(f"{cat:11s} {sub:18s} score {pred.score:.3f} thr {pred.threshold:.3f} "
              f"{'DEFECTIVE' if pred.is_anomalous else 'normal   '} {ms:6.0f} ms")