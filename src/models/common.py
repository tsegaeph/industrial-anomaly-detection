"""Shared building blocks: frozen ResNet feature extractor and Gaussian blur."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNetFeatureExtractor(nn.Module):
    """Frozen ImageNet-pretrained torchvision ResNet returning intermediate feature maps."""

    def __init__(self, backbone: str = "resnet18", layers: tuple[str, ...] = ("layer1", "layer2", "layer3")) -> None:
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="IMAGENET1K_V1")
        self.layers = tuple(layers)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        names = ("layer1", "layer2", "layer3", "layer4")
        last = max(names.index(n) for n in self.layers)
        self.blocks = nn.ModuleDict({n: getattr(net, n) for n in names[: last + 1]})
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "ResNetFeatureExtractor":
        return super().train(False)  # always eval: BatchNorm must use ImageNet statistics

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        out = []
        for name, block in self.blocks.items():
            x = block(x)
            if name in self.layers:
                out.append(x)
        return out


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Separable Gaussian blur on (B,1,H,W) maps with reflect padding (kernel radius = 4*sigma)."""
    radius = int(4.0 * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, device=maps.device, dtype=maps.dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    maps = F.pad(maps, (radius, radius, radius, radius), mode="reflect")
    maps = F.conv2d(maps, k.view(1, 1, 1, -1))
    return F.conv2d(maps, k.view(1, 1, -1, 1))