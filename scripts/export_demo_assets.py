"""Assemble a self-contained Hugging Face Space folder: app, minimal source, models, thresholds."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.mvtec import CATEGORIES  # noqa: E402
from src.evaluation.thresholds import threshold_rules  # noqa: E402

SRC_FILES = ["src/data/mvtec.py", "src/data/dataset.py", "src/models/common.py",
             "src/models/patchcore.py", "src/inference/predictor.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, help="folder with model_*.pt and the calibration file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--calib-file", default="calibration_cv_scores.json")
    args = ap.parse_args()

    exp, out = Path(args.exp_dir), Path(args.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "models").mkdir(parents=True)

    for f in SRC_FILES:
        dst = out / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, dst)
    for pkg in {Path(f).parent for f in SRC_FILES} | {Path("src")}:
        (out / pkg / "__init__.py").touch()

    calib = json.loads((exp / args.calib_file).read_text())
    thresholds = {c: threshold_rules(np.array(calib[c]["scores"])) for c in CATEGORIES}
    (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    for c in CATEGORIES:
        src = exp / f"model_{c}.pt"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy(src, out / "models" / src.name)

    shutil.copy(REPO / "demo/app.py", out / "app.py")
    shutil.copy(REPO / "demo/requirements.txt", out / "requirements.txt")
    shutil.copy(REPO / "demo/README.md", out / "README.md")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Space folder ready at {out} ({size:.0f} MB)")
    print(json.dumps({c: round(v["p99"], 3) for c, v in thresholds.items()}, indent=2))


if __name__ == "__main__":
    main()