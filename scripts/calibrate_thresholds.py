"""Score held-out normal training images with a model fitted on the remaining ones.

The recorded scores are used to choose an image-level decision threshold without touching test data.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_split, make_loader  # noqa: E402
from src.data.mvtec import CATEGORIES, build_index  # noqa: E402
from src.evaluation.metrics import image_scores  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402

logger = logging.getLogger("calibrate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    size, bs, seed = cfg["image_size"], cfg["batch_size"], cfg["seed"]
    df = build_index(args.data_root)
    result = {}
    for cat in CATEGORIES:
        train_df = get_split(df, cat, "train")
        held = train_df.sample(frac=args.holdout_frac, random_state=seed)
        fit_df = train_df.drop(held.index).reset_index(drop=True)
        held = held.reset_index(drop=True)
        model = PatchCore(**cfg["params"], image_size=size, seed=seed)
        model.fit(make_loader(fit_df, size, bs))
        maps = np.concatenate([model.predict(b["image"]).numpy() for b in make_loader(held, size, bs)])
        scores = image_scores(maps)
        result[cat] = {"n_fit": len(fit_df), "n_heldout": len(held), "scores": scores.tolist()}
        logger.info("[%s] held-out normal scores: min %.3f mean %.3f max %.3f", cat, scores.min(), scores.mean(), scores.max())
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration_heldout_scores.json").write_text(json.dumps(result, indent=2))
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()