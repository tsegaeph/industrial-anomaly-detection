"""Summary plots built from saved experiment results (no dataset or GPU needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATS = ["bottle", "capsule", "carpet", "metal_nut", "screw", "toothbrush", "zipper"]


def _save(fig, save_path) -> None:
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_robustness(macro_csv: str | Path, save_path=None) -> None:
    """Macro metrics vs corruption level; the clean result is drawn at the no-corruption position."""
    t = pd.read_csv(macro_csv)
    clean = t[t.corruption == "clean"].iloc[0]
    clean_x = {"noise": 0.0, "brightness": 1.0, "blur": 0.0}
    xlabel = {"noise": "noise std (0-255 scale)", "brightness": "brightness factor", "blur": "blur sigma (px)"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, kind in zip(axes, ("noise", "brightness", "blur")):
        g = t[t.corruption == kind].copy()
        g = pd.concat([g, pd.DataFrame([{**clean.to_dict(), "level": clean_x[kind]}])]).sort_values("level")
        for m, label in (("image_auroc", "image AUROC"), ("pixel_ap", "pixel AP"), ("aupro", "AUPRO")):
            ax.plot(g.level, g[m], marker="o", label=label)
        ax.set_xlabel(xlabel[kind]); ax.set_title(kind); ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro mean over 7 categories"); axes[0].legend(fontsize=8)
    _save(fig, save_path)


def plot_fewshot(exp_root: str | Path, save_path=None, ns=(5, 10, 25, 50), seeds=(0, 1, 2),
                 full_name="patchcore_wrn50_256") -> None:
    """Macro metrics vs number of normal training images (mean +- std over seeds)."""
    exp_root = Path(exp_root)

    def macro(name):
        t = pd.read_csv(exp_root / name / "results.csv")
        return t[t.category == "MACRO_MEAN"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    full = macro(full_name)
    for m, label in (("image_auroc", "image AUROC"), ("aupro", "AUPRO")):
        vals = np.array([[macro(f"patchcore_n{n}_s{s}")[m] for s in seeds] for n in ns])
        ax.errorbar(ns, vals.mean(1), yerr=vals.std(1, ddof=1), marker="o", capsize=3, label=label)
        ax.axhline(full[m], ls="--", lw=1, color=ax.lines[-1].get_color(), alpha=0.6)
    ax.set_xscale("log"); ax.set_xticks(list(ns)); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("normal training images per category (dashed = all)")
    ax.set_ylabel("macro mean"); ax.grid(alpha=0.3); ax.legend()
    _save(fig, save_path)


def plot_model_comparison(exp_root: str | Path, names: dict[str, str], save_path=None) -> None:
    """Grouped bars per category for several experiments. names maps label -> experiment folder."""
    exp_root = Path(exp_root)
    metrics = (("image_auroc", "image AUROC"), ("pixel_ap", "pixel AP"), ("aupro", "AUPRO"))
    tables = {lab: pd.read_csv(exp_root / n / "results.csv").set_index("category") for lab, n in names.items()}
    rows = CATS + ["MACRO_MEAN"]
    w = 0.8 / len(tables)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=False)
    for ax, (m, title) in zip(axes, metrics):
        for i, (lab, t) in enumerate(tables.items()):
            ax.bar(np.arange(len(rows)) + i * w, t.loc[rows, m], w, label=lab)
        ax.set_xticks(np.arange(len(rows)) + w * (len(tables) - 1) / 2)
        ax.set_xticklabels([r.replace("MACRO_MEAN", "macro") for r in rows], rotation=45, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    _save(fig, save_path)