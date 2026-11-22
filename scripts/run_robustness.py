"""Evaluate saved PatchCore models on corrupted test images (noise, brightness, blur).

Example:
    python scripts/run_robustness.py --models-dir <exp_dir_with_model_*.pt> --data-root <mvtec_root> \
        --out-dir <reports/metrics/robustness>
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.corruptions import ROBUSTNESS_SETTINGS, make_corruption  # noqa: E402
from src.data.dataset import get_split, make_loader  # noqa: E402
from src.data.mvtec import CATEGORIES, build_index  # noqa: E402
from src.evaluation.metrics import evaluate_category  # noqa: E402
from src.evaluation.predict import collect_predictions  # noqa: E402
from src.models.patchcore import PatchCore  # noqa: E402

logger = logging.getLogger("run_robustness")
METRICS = ["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = build_index(args.data_root)
    cats = args.categories or list(CATEGORIES)

    for cat in cats:
        f = out / f"robustness_{cat}.csv"
        if f.exists():
            logger.info("[%s] found saved results, skipping", cat)
            continue
        model = PatchCore.load(Path(args.models_dir) / f"model_{cat}.pt")
        size = model.cfg["image_size"]
        test_df = get_split(df, cat, "test")
        rows = []
        for kind, level in ROBUSTNESS_SETTINGS:
            corruption = None if kind == "clean" else make_corruption(kind, level)
            maps, masks, labels = collect_predictions(
                model, make_loader(test_df, size, args.batch_size, corruption=corruption))
            m = evaluate_category(maps, masks, labels)
            rows.append({"category": cat, "corruption": kind, "level": level, **m})
            logger.info("[%s] %s %s: image_auroc=%.4f aupro=%.4f", cat, kind, level, m["image_auroc"], m["aupro"])
        pd.DataFrame(rows).to_csv(f, index=False)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.concat([pd.read_csv(out / f"robustness_{c}.csv") for c in cats], ignore_index=True)
    table.to_csv(out / "robustness_all.csv", index=False)
    macro = table.groupby(["corruption", "level"], sort=False)[METRICS].mean()
    macro.to_csv(out / "robustness_macro.csv")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(macro)


if __name__ == "__main__":
    main()