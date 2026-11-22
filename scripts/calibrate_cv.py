"""K-fold out-of-fold normal scores: every normal training image is scored by a model that never saw it."""
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
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_split, make_loader  # noqa: E402
from src.data.mvtec import CATEGORIES, build_index  # noqa: E402
from src.evaluation.metrics import image_scores  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402

logger = logging.getLogger("calibrate_cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    size, bs, seed = cfg["image_size"], cfg["batch_size"], cfg["seed"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    f = out / "calibration_cv_scores.json"
    result = json.loads(f.read_text()) if f.exists() else {}
    df = build_index(args.data_root)

    for cat in CATEGORIES:
        if cat in result:
            logger.info("[%s] already done, skipping", cat)
            continue
        train_df = get_split(df, cat, "train")
        scores = np.zeros(len(train_df))
        for k, (fit_idx, held_idx) in enumerate(KFold(args.folds, shuffle=True, random_state=seed).split(train_df)):
            model = PatchCore(**cfg["params"], image_size=size, seed=seed)
            model.fit(make_loader(train_df.iloc[fit_idx].reset_index(drop=True), size, bs))
            held = train_df.iloc[held_idx].reset_index(drop=True)
            maps = np.concatenate([model.predict(b["image"]).numpy() for b in make_loader(held, size, bs)])
            scores[held_idx] = image_scores(maps)
            del model; gc.collect(); torch.cuda.empty_cache()
            logger.info("[%s] fold %d/%d done", cat, k + 1, args.folds)
        result[cat] = {"n_fit": int(len(train_df) * (args.folds - 1) / args.folds),
                       "n_heldout": len(train_df), "scores": scores.tolist()}
        logger.info("[%s] OOF normal scores: min %.3f mean %.3f p99 %.3f max %.3f", cat, scores.min(),
                    scores.mean(), np.percentile(scores, 99), scores.max())
        f.write_text(json.dumps(result, indent=2))   # saved after every category (disconnect-safe)


if __name__ == "__main__":
    main()