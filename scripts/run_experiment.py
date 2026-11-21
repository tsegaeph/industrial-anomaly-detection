"""Run one experiment: per category, fit on normal train images and evaluate on the test split.

Example:
    python scripts/run_experiment.py --config configs/padim_r18_256.yaml \
        --data-root /content/industrial-anomaly-detection/datasets/mvtec_anomaly_detection \
        --out-dir /content/drive/MyDrive/industrial-anomaly-detection/experiments
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_split, make_loader, subsample_normals  # noqa: E402
from src.data.mvtec import CATEGORIES, build_index  # noqa: E402
from src.evaluation.metrics import evaluate_category  # noqa: E402
from src.evaluation.predict import collect_predictions  # noqa: E402

logger = logging.getLogger("run_experiment")
METRIC_COLS = ["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict):
    kwargs = dict(cfg.get("params", {}), image_size=cfg["image_size"], seed=cfg["seed"])
    if cfg["model"] == "padim":
        from src.models.padim import PaDiM
        return PaDiM(**kwargs)
    if cfg["model"] == "patchcore":
        from src.models.patchcore import PatchCore
        return PatchCore(**kwargs)
    raise ValueError(f"Unknown model: {cfg['model']!r}")


def run_category(cfg: dict, df: pd.DataFrame, cat: str, out: Path) -> dict:
    train_df = get_split(df, cat, "train")
    if cfg.get("n_train"):
        train_df = subsample_normals(train_df, int(cfg["n_train"]), cfg["seed"])
    test_df = get_split(df, cat, "test")
    size, bs = cfg["image_size"], cfg["batch_size"]

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model = build_model(cfg)

    t0 = time.time()
    model.fit(make_loader(train_df, size, bs))
    fit_s = time.time() - t0

    t0 = time.time()
    maps, masks, labels = collect_predictions(model, make_loader(test_df, size, bs))
    pred_s = time.time() - t0

    row = {"category": cat, "n_train": len(train_df), "n_test": len(test_df)}
    row.update(evaluate_category(maps, masks, labels))
    row.update(fit_s=fit_s, ms_per_img_e2e=1000 * pred_s / len(test_df),   # includes image loading
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else float("nan"))

    np.savez_compressed(out / f"maps_{cat}.npz", maps=maps.astype(np.float16), labels=labels,
                        image_paths=test_df.image_path.to_numpy(), defect_types=test_df.defect_type.to_numpy())
    if cfg.get("save_model"):
        model.save(out / f"model_{cat}.pt")
    (out / f"metrics_{cat}.json").write_text(json.dumps(row, indent=2))

    del model, maps, masks
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return row


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply KEY=VALUE overrides; dotted keys address nested dicts (e.g. params.coreset_ratio=0.01)."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--categories", nargs="+", default=None, help="override categories in config")
    ap.add_argument("--force", action="store_true", help="recompute even if results exist")
    ap.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set name=x image_size=384 params.coreset_ratio=0.01")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    base_name = cfg["name"]
    cfg = apply_overrides(cfg, args.overrides)
    if args.overrides and cfg["name"] == base_name:
        raise SystemExit("When using --set, also set name=<new_name> so results are not mixed up.")

    out = Path(args.out_dir) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))   # the config actually used
    set_seed(cfg["seed"])

    df = build_index(args.data_root)
    categories = args.categories or cfg.get("categories") or list(CATEGORIES)

    rows = []
    for cat in categories:
        cached = out / f"metrics_{cat}.json"
        if cached.exists() and (out / f"maps_{cat}.npz").exists() and not args.force:
            logger.info("[%s] found saved results, skipping", cat)
            rows.append(json.loads(cached.read_text()))
            continue
        logger.info("[%s] fitting and evaluating...", cat)
        rows.append(run_category(cfg, df, cat, out))
        logger.info("[%s] %s", cat, {k: round(v, 4) for k, v in rows[-1].items() if k in METRIC_COLS})

    table = pd.DataFrame(rows)
    macro = {"category": "MACRO_MEAN", **table[METRIC_COLS + ["fit_s", "ms_per_img_e2e", "peak_gpu_gb"]].mean().to_dict()}
    table = pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
    table.to_csv(out / "results.csv", index=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(table.to_string(index=False))
    logger.info("Saved to %s", out)

if __name__ == "__main__":
    main()