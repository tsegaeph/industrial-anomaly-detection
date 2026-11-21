"""PatchCore: coreset memory bank of locally aware patch features (Roth et al., CVPR 2022).

Simplifications vs. the reference implementation: concatenated layer features are reduced to
`target_dim` with one channel-wise adaptive average pool, and the image score is the maximum of
the smoothed anomaly map (no re-weighting).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.common import ResNetFeatureExtractor, gaussian_blur

logger = logging.getLogger(__name__)


@torch.no_grad()
def greedy_coreset(feats: torch.Tensor, n_select: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Greedy k-center selection on a random projection. Returns indices of selected rows."""
    n = feats.shape[0]
    if n_select >= n:
        return torch.arange(n, device=feats.device)
    gen = torch.Generator().manual_seed(seed)
    proj = (torch.randn(feats.shape[1], proj_dim, generator=gen) / proj_dim ** 0.5).to(feats.device)
    z = feats @ proj
    zn = (z * z).sum(1)
    sel = torch.empty(n_select, dtype=torch.long, device=feats.device)
    idx = torch.randint(n, (1,), generator=gen).to(feats.device)[0]
    min_d = torch.full((n,), float("inf"), device=feats.device)
    for i in range(n_select):
        sel[i] = idx
        d = zn - 2.0 * (z @ z[idx]) + zn[idx]      # squared distance to the newest center
        min_d = torch.minimum(min_d, d)
        idx = torch.argmax(min_d)
    return sel


class PatchCore:
    def __init__(self, backbone: str = "wide_resnet50_2", layers: tuple[str, ...] = ("layer2", "layer3"),
                 image_size: int = 256, coreset_ratio: float = 0.1, target_dim: int = 1024,
                 sigma: float = 4.0, seed: int = 0, device: str | None = None) -> None:
        self.cfg = dict(backbone=backbone, layers=list(layers), image_size=image_size,
                        coreset_ratio=coreset_ratio, target_dim=target_dim, sigma=sigma, seed=seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = ResNetFeatureExtractor(backbone, tuple(layers)).to(self.device)
        self.memory: torch.Tensor | None = None   # (M, target_dim)
        self.grid: tuple[int, int] | None = None

    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B*h*w, target_dim) locally aggregated patch features."""
        feats = [F.avg_pool2d(f, 3, stride=1, padding=1) for f in self.extractor(images.to(self.device, non_blocking=True))]
        size = feats[0].shape[-2:]
        feats = [feats[0]] + [F.interpolate(f, size=size, mode="bilinear", align_corners=False) for f in feats[1:]]
        emb = torch.cat(feats, dim=1)                                   # (B, C, h, w)
        self.grid = (int(size[0]), int(size[1]))
        emb = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])         # (B*h*w, C)
        return F.adaptive_avg_pool1d(emb.unsqueeze(1), self.cfg["target_dim"]).squeeze(1)

    @torch.no_grad()
    def fit(self, loader: DataLoader) -> "PatchCore":
        feats = torch.cat([self._embed(b["image"]) for b in loader])
        n_keep = max(1, int(round(self.cfg["coreset_ratio"] * feats.shape[0])))
        sel = greedy_coreset(feats, n_keep, seed=self.cfg["seed"])
        self.memory = feats[sel].contiguous()
        logger.info("PatchCore: %d patches -> memory bank of %d (%.1f MB)",
                    feats.shape[0], self.memory.shape[0], self.memory.numel() * 4 / 1e6)
        return self

    @torch.no_grad()
    def _nn_dist(self, q: torch.Tensor) -> torch.Tensor:
        """L2 distance from each query patch to its nearest memory patch (chunked matmul)."""
        m = self.memory
        mn = (m * m).sum(1)
        step = max(256, int(2e8 // m.shape[0]))     # keep each distance block <= ~800 MB
        out = []
        for s in range(0, q.shape[0], step):
            qc = q[s:s + step]
            d2 = (qc * qc).sum(1, keepdim=True) - 2.0 * (qc @ m.T) + mn[None]
            out.append(d2.min(1).values.clamp_min(0).sqrt())
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,H,W) anomaly maps on CPU (higher = more anomalous)."""
        if self.memory is None:
            raise RuntimeError("Call fit() or load() first")
        scores = self._nn_dist(self._embed(images))
        maps = scores.view(images.shape[0], 1, *self.grid)
        size = self.cfg["image_size"]
        maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)
        return gaussian_blur(maps, self.cfg["sigma"])[:, 0].cpu()

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "memory": self.memory.cpu(), "grid": self.grid}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PatchCore":
        state = torch.load(path, map_location="cpu")
        model = cls(device=device, **state["cfg"])
        model.memory = state["memory"].to(model.device)
        model.grid = tuple(state["grid"])
        return model