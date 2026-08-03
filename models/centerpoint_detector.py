"""Pillar-based CenterPoint detector for Waymo LiDAR.

This is a distinct CenterPoint-style backend: PointPillars feature encoding,
a multi-scale BEV encoder, and an anchor-free center heatmap / box regression
head.  It shares the proven Waymo voxelization, target creation, decode, and
proxy-evaluation interface with PointPillars while using a different BEV trunk.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_detector
from models.base_detector import BaseDetector
from models.pointpillars_detector import (
    PillarFeatureNet,
    PillarScatter,
    PointPillarsDetector,
    PointPillarsHead,
    _ConvBnReLU,
)


class CenterPointBEVBackbone(nn.Module):
    """Multi-scale BEV backbone with an FPN-style high-resolution fusion."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(_ConvBnReLU(in_channels, 64), _ConvBnReLU(64, 64))
        self.down1 = nn.Sequential(_ConvBnReLU(64, 96, s=2), _ConvBnReLU(96, 96), _ConvBnReLU(96, 96))
        self.down2 = nn.Sequential(_ConvBnReLU(96, 128, s=2), _ConvBnReLU(128, 128), _ConvBnReLU(128, 128))
        self.lateral1 = nn.Conv2d(96, 128, 1, bias=False)
        self.lateral2 = nn.Conv2d(128, 128, 1, bias=False)
        self.refine = nn.Sequential(_ConvBnReLU(128, 128), _ConvBnReLU(128, 128))

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        base = self.stem(bev)                         # H × W
        scale2 = self.down1(base)                      # H/2 × W/2
        scale4 = self.down2(scale2)                    # H/4 × W/4
        scale4_up = F.interpolate(self.lateral2(scale4), size=scale2.shape[-2:],
                                  mode='bilinear', align_corners=False)
        return self.refine(self.lateral1(scale2) + scale4_up)  # H/2 × W/2


class CenterPointPillars(nn.Module):
    """Pillar encoder + multi-scale BEV backbone + anchor-free center head."""

    def __init__(self, num_classes: int = 3, pfn_out: int = 64):
        super().__init__()
        self.pfn = PillarFeatureNet(9, pfn_out)
        self.scatter = PillarScatter(pfn_out)
        self.backbone = CenterPointBEVBackbone(pfn_out)
        self.head = PointPillarsHead(128, num_classes)

    def forward(self, pillars: torch.Tensor, coords: torch.Tensor, batch_size: int,
                grid_shape: tuple[int, int]) -> dict[str, torch.Tensor]:
        features = self.pfn(pillars, pillars.shape[1])
        bev = self.scatter(features, coords, batch_size, grid_shape)
        return self.head(self.backbone(bev))


@register_detector('CenterPointDetector')
class CenterPointDetector(PointPillarsDetector):
    """CenterPoint-style alternative to PointPillars for 3-D Waymo boxes.

    It intentionally inherits the voxelizer, center targets, loss, decoder,
    and collate function from :class:`PointPillarsDetector`, so checkpointing
    and the existing LiDAR evaluation command work without a separate trainer.
    """

    def build_model(self, cfg: dict, device: torch.device) -> nn.Module:
        model = CenterPointPillars(
            num_classes=cfg['model'].get('num_classes', 3),
            pfn_out=cfg['model'].get('pfn_out', 64),
        )
        weights = cfg.get('resume_weights')
        if weights and Path(weights).exists():
            checkpoint = torch.load(weights, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and isinstance(checkpoint.get('model'), nn.Module):
                model = checkpoint['model']
                print(f'Resumed from segment {checkpoint.get("seg", "?")}')
            else:
                model.load_state_dict(checkpoint)
        return model.to(device)
