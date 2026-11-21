"""PaDiM: per-position Gaussian model of pretrained CNN features (Defard et al., 2020)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.common import ResNetFeatureExtractor, gaussian_blur

logger = logging.getLogger(__name__)


class PaDiM:
    """Fit on normal images only; predict pixel-level anomaly maps.

    Args:
        backbone: torchvision ResNet name.
        d_reduced: number of randomly selected channels kept from the concatenated features.
        image_size: input side length (also the anomaly-map size).
        eps: diagonal regularisation added to every covariance matrix.
        sigma: Gaussian smoothing of the final map (in pixels of the output map).
        seed: seed for the channel selection.
    """

    def __init__(self, backbone: str = "resnet18", d_reduced: int = 100, image_size: int = 256,
                 eps: float = 0.01, sigma: float = 4.0, seed: int = 0,
                 device: str | None = None) -> None:
        self.cfg = dict(backbone=backbone, d_reduced=d_reduced, image_size=image_size,
                        eps=eps, sigma=sigma, seed=seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = ResNetFeatureExtractor(backbone).to(self.device)
        self.idx: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None      # (d, S)
        self.inv_cov: torch.Tensor | None = None   # (S, d, d)
        self.grid: tuple[int, int] | None = None

    # ---- features -------------------------------------------------------
    @torch.no_grad()
    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,d,S): multi-layer features at layer-1 resolution, channel-subsampled."""
        feats = self.extractor(images.to(self.device, non_blocking=True))
        size = feats[0].shape[-2:]
        emb = torch.cat([feats[0]] + [F.interpolate(f, size=size, mode="nearest") for f in feats[1:]], dim=1)
        if self.idx is None:
            g = torch.Generator().manual_seed(self.cfg["seed"])
            self.idx = torch.randperm(emb.shape[1], generator=g)[: self.cfg["d_reduced"]].to(self.device)
        self.grid = (int(size[0]), int(size[1]))
        return emb[:, self.idx].flatten(2)

    # ---- fit ------------------------------------------------------------
    @torch.no_grad()
    def fit(self, loader: DataLoader, chunk: int = 512) -> "PaDiM":
        """Estimate mean and inverse covariance per position from normal training images."""
        chunks = [self._embed(b["image"]) for b in loader]
        X = torch.cat(chunks)                                  # (N, d, S)
        N, d, S = X.shape
        if N < 2:
            raise ValueError("Need at least 2 training images to estimate covariance")
        self.mean = X.mean(0)
        Xc = X - self.mean
        cov = torch.empty(S, d, d, device=self.device)
        for s in range(0, S, chunk):
            xs = Xc[:, :, s:s + chunk]
            cov[s:s + chunk] = torch.einsum("ndc,nec->cde", xs, xs) / (N - 1)
        cov += self.cfg["eps"] * torch.eye(d, device=self.device)
        self.inv_cov = torch.linalg.inv(cov.double().cpu()).float().to(self.device)
        logger.info("PaDiM fitted on %d images: d=%d, positions=%d", N, d, S)
        return self

    # ---- predict --------------------------------------------------------
    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) -> (B,H,W) anomaly maps on CPU (higher = more anomalous)."""
        if self.mean is None:
            raise RuntimeError("Call fit() or load() first")
        emb = self._embed(images)                                   # (B,d,S)
        diff = emb - self.mean
        tmp = torch.einsum("bds,sde->bse", diff, self.inv_cov)
        maha = (tmp * diff.permute(0, 2, 1)).sum(-1).clamp_min(0).sqrt()   # (B,S)
        maps = maha.view(-1, 1, *self.grid)
        size = self.cfg["image_size"]
        maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)
        return gaussian_blur(maps, self.cfg["sigma"])[:, 0].cpu()

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg, "idx": self.idx.cpu(), "mean": self.mean.cpu(),
                    "inv_cov": self.inv_cov.cpu(), "grid": self.grid}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PaDiM":
        state = torch.load(path, map_location="cpu")
        model = cls(device=device, **state["cfg"])
        model.idx = state["idx"].to(model.device)
        model.mean = state["mean"].to(model.device)
        model.inv_cov = state["inv_cov"].to(model.device)
        model.grid = tuple(state["grid"])
        return model