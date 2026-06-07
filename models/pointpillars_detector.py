"""
pointpillars_detector.py — PointPillars 3-D object detector for Waymo LiDAR.

Architecture (Lang et al., 2019 + CenterPoint-style head):
  1. Voxelization     — scatter raw (N, 4) XYZI points into (P, T, 9) pillars
  2. PillarFeatureNet — PointNet-style MLP + max-pool → (P, 64)
  3. PillarScatter    — sparse → dense (B, 64, H, W) BEV pseudo-image
  4. BEV Backbone     — two-stage 2-D CNN (64→128), returns (B, 128, H/2, W/2)
  5. Detection Head   — per-class heatmap + shared box regression
  6. Loss             — Gaussian focal loss (heatmap) + L1 (box offsets)
  7. Inference        — heatmap peak detection + box decoding

Registered as 'PointPillarsDetector'.  YAML config key: model.type.

Default Colab T4 settings (configs/pointpillars.yaml):
  Detection range:  ±50 m XY, -3…3 m Z
  Pillar size:      0.25 m × 0.25 m  →  400×400 BEV grid
  Max pillars:      6 000
  Max pts/pillar:   20
  Classes:          vehicle, pedestrian, cyclist  (3)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_detector
from models.base_detector import BaseDetector


# ── Utility functions ─────────────────────────────────────────────────────────


def _gaussian_2d(radius: int) -> np.ndarray:
    """Square Gaussian kernel of given radius (diameter = 2*radius+1)."""
    diameter = 2 * radius + 1
    sigma    = diameter / 6.0
    y, x     = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g        = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g.astype(np.float32)


def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, radius: int):
    """Stamp a Gaussian blob (max-blended) onto heatmap in-place."""
    H, W = heatmap.shape
    g    = _gaussian_2d(radius)
    d    = 2 * radius + 1

    x0 = max(cx - radius, 0);  x1 = min(cx + radius + 1, W)
    y0 = max(cy - radius, 0);  y1 = min(cy + radius + 1, H)
    gx0 = x0 - (cx - radius);  gx1 = gx0 + (x1 - x0)
    gy0 = y0 - (cy - radius);  gy1 = gy0 + (y1 - y0)

    if x1 > x0 and y1 > y0 and gx1 > gx0 and gy1 > gy0:
        np.maximum(heatmap[y0:y1, x0:x1], g[gy0:gy1, gx0:gx1],
                   out=heatmap[y0:y1, x0:x1])


def _gaussian_focal_loss(pred: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
    """CenterNet / CenterPoint Gaussian focal loss.

    pred, target: (B, C, H, W), target in [0, 1].
    """
    pos  = (target == 1.0).float()
    neg  = (target < 1.0).float()

    loss_pos = -torch.log(pred.clamp(min=1e-6)) * (1 - pred) ** 2 * pos
    loss_neg = (-torch.log((1 - pred).clamp(min=1e-6))
                * pred ** 2
                * (1 - target) ** 4
                * neg)

    num_pos = pos.sum().clamp(min=1)
    return (loss_pos.sum() + loss_neg.sum()) / num_pos


def _voxelize(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    voxel_size: tuple[float, float],
    max_pillars: int,
    max_pts: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Pillarize an (N, 4) XYZI point cloud.

    Returns:
        pillars: (P, max_pts, 9) float32 — 9 features per point:
                 [x, y, z, intensity,  x-xc, y-yc, z-zc,  x-xp, y-yp]
                 where (xc,yc,zc) = mean of pillar's points,
                       (xp,yp)   = pillar centre in BEV.
        coords:  (P, 3) int32 — [batch_idx=0, y_grid, x_grid].
        grid_shape: (H, W) of the BEV grid.
    """
    W = round((x_range[1] - x_range[0]) / voxel_size[0])
    H = round((y_range[1] - y_range[0]) / voxel_size[1])

    # Filter to detection range
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] < x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] < y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    )
    pts = points[mask]

    if len(pts) == 0:
        return (np.zeros((0, max_pts, 9), np.float32),
                np.zeros((0, 3), np.int32),
                (H, W))

    # Grid indices
    xi = np.clip(((pts[:, 0] - x_range[0]) / voxel_size[0]).astype(np.int32), 0, W - 1)
    yi = np.clip(((pts[:, 1] - y_range[0]) / voxel_size[1]).astype(np.int32), 0, H - 1)

    # Sort by pillar linear index for grouping
    lin = (yi * W + xi).astype(np.int64)
    order = np.argsort(lin, kind='stable')
    pts = pts[order];  xi = xi[order];  yi = yi[order];  lin = lin[order]

    # Pillar boundaries
    bounds = np.concatenate([[0], np.where(np.diff(lin))[0] + 1, [len(pts)]])
    num_raw = len(bounds) - 1

    # If too many pillars, keep those with the most points
    if num_raw > max_pillars:
        sizes   = np.diff(bounds)
        keep    = np.argsort(sizes)[-max_pillars:]
        new_pts, new_xi, new_yi = [], [], []
        for k in sorted(keep):
            s, e = bounds[k], bounds[k + 1]
            new_pts.append(pts[s:e]); new_xi.append(xi[s:e]); new_yi.append(yi[s:e])
        pts = np.concatenate(new_pts)
        xi  = np.concatenate(new_xi)
        yi  = np.concatenate(new_yi)
        lin = (yi * W + xi).astype(np.int64)
        order   = np.argsort(lin, kind='stable')
        pts = pts[order]; xi = xi[order]; yi = yi[order]; lin = lin[order]
        bounds = np.concatenate([[0], np.where(np.diff(lin))[0] + 1, [len(pts)]])

    num_pillars = len(bounds) - 1
    pillars = np.zeros((num_pillars, max_pts, 9), np.float32)
    coords  = np.zeros((num_pillars, 3), np.int32)

    for p in range(num_pillars):
        s = bounds[p]
        e = min(bounds[p + 1], s + max_pts)
        n = e - s

        p_pts = pts[s:e]               # (n, 4)
        xc, yc, zc = p_pts[:, :3].mean(axis=0)
        xp = x_range[0] + (xi[s] + 0.5) * voxel_size[0]
        yp = y_range[0] + (yi[s] + 0.5) * voxel_size[1]

        pillars[p, :n, :4] = p_pts
        pillars[p, :n, 4]  = p_pts[:, 0] - xc
        pillars[p, :n, 5]  = p_pts[:, 1] - yc
        pillars[p, :n, 6]  = p_pts[:, 2] - zc
        pillars[p, :n, 7]  = p_pts[:, 0] - xp
        pillars[p, :n, 8]  = p_pts[:, 1] - yp

        coords[p] = [0, int(yi[s]), int(xi[s])]

    return pillars, coords, (H, W)


def _make_heatmap_targets(
    gt_boxes: np.ndarray,    # (M, 7) [cx, cy, cz, dx, dy, dz, heading]
    gt_labels: np.ndarray,   # (M,) int
    grid_shape: tuple[int, int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    voxel_size: tuple[float, float],
    num_classes: int,
    anchor_sizes: dict[int, tuple],   # class_id → (mean_dx_m, mean_dy_m)
    stride: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Build GT tensors for the center-based head.

    The detection head operates on a strided feature map (grid_shape // stride).
    For each GT box we stamp a Gaussian on the heatmap and record regression
    targets at the grid cell closest to the box center.

    Returns (all on the strided grid H' × W'):
        heatmap:  (C, H', W') float32
        offset:   (2, H', W') float32   — sub-cell offset [dx, dy] ∈ (–0.5, 0.5)
        z_center: (1, H', W') float32
        log_dim:  (3, H', W') float32   — log(dx, dy, dz)
        heading:  (2, H', W') float32   — sin, cos of heading angle
        reg_mask: (H', W') bool          — True where a GT center lands
        box_idx:  (H', W') int32         — which GT box (for logging)
    """
    H, W = grid_shape
    Hs   = H // stride
    Ws   = W // stride

    heatmap  = np.zeros((num_classes, Hs, Ws), np.float32)
    offset   = np.zeros((2, Hs, Ws), np.float32)
    z_center = np.zeros((1, Hs, Ws), np.float32)
    log_dim  = np.zeros((3, Hs, Ws), np.float32)
    heading  = np.zeros((2, Hs, Ws), np.float32)
    reg_mask = np.zeros((Hs, Ws), bool)

    for b_i, (box, cls) in enumerate(zip(gt_boxes, gt_labels)):
        cx, cy, cz, dx, dy, dz, h_angle = box
        cls = int(cls)

        # Map box center to strided grid
        gx = (cx - x_range[0]) / voxel_size[0] / stride
        gy = (cy - y_range[0]) / voxel_size[1] / stride

        gxi, gyi = int(gx), int(gy)
        if not (0 <= gxi < Ws and 0 <= gyi < Hs):
            continue

        # Gaussian radius proportional to box footprint
        mean_dx, mean_dy = anchor_sizes.get(cls, (4.0, 2.0))
        radius = max(1, round(((mean_dx / voxel_size[0]) / stride +
                               (mean_dy / voxel_size[1]) / stride) / 4))

        _draw_gaussian(heatmap[cls], gxi, gyi, radius)

        # Regression targets at the center cell
        offset[:, gyi, gxi]   = [gx - gxi, gy - gyi]
        z_center[0, gyi, gxi] = cz
        log_dim[:, gyi, gxi]  = [np.log(max(dx, 0.1)),
                                  np.log(max(dy, 0.1)),
                                  np.log(max(dz, 0.1))]
        heading[:, gyi, gxi]  = [np.sin(h_angle), np.cos(h_angle)]
        reg_mask[gyi, gxi]    = True

    return heatmap, offset, z_center, log_dim, heading, reg_mask, np.array([0])


# ── Neural network modules ────────────────────────────────────────────────────


class PillarFeatureNet(nn.Module):
    """PointNet-style MLP + max-pool over points in each pillar.

    Input:  (P*T, 9)   where P = num_pillars, T = max_pts_per_pillar
    Output: (P, D)     D = out_channels (default 64)
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 64):
        super().__init__()
        self.out_channels = out_channels
        self.linear  = nn.Linear(in_channels, out_channels, bias=False)
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act     = nn.ReLU(inplace=True)

    def forward(self, pillars: torch.Tensor, num_pts: int) -> torch.Tensor:
        """
        Args:
            pillars: (P, T, 9) float32
            num_pts: T (max points per pillar, constant per batch)
        Returns:
            (P, D) float32
        """
        P, T, C = pillars.shape
        x = pillars.reshape(P * T, C)
        x = self.act(self.bn(self.linear(x)))
        x = x.reshape(P, T, self.out_channels)
        x = x.max(dim=1)[0]   # (P, D)
        return x


class PillarScatter(nn.Module):
    """Scatter per-pillar features to a dense BEV pseudo-image."""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(
        self,
        pillar_features: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int,
        grid_shape: tuple[int, int],
    ) -> torch.Tensor:
        """
        Args:
            pillar_features: (total_P, D)
            coords:          (total_P, 3) int — [batch, y, x]
            batch_size:      B
            grid_shape:      (H, W)

        Returns:
            (B, D, H, W) float32
        """
        H, W  = grid_shape
        D     = self.num_features
        bev   = torch.zeros(batch_size, D, H, W,
                            dtype=pillar_features.dtype,
                            device=pillar_features.device)
        b_idx = coords[:, 0].long()
        y_idx = coords[:, 1].long()
        x_idx = coords[:, 2].long()
        bev[b_idx, :, y_idx, x_idx] = pillar_features
        return bev


class _ConvBnReLU(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class PointPillarsBEVBackbone(nn.Module):
    """Two-stage 2-D backbone operating on the 400×400 BEV pseudo-image.

    Stage 1: 64→64,  stride 1, 3 conv layers  →  400×400
    Stage 2: 64→128, stride 2, 4 conv layers  →  200×200
    Output:  (B, 128, 200, 200)
    """

    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.stage1 = nn.Sequential(
            _ConvBnReLU(in_channels, 64, s=1),
            _ConvBnReLU(64, 64),
            _ConvBnReLU(64, 64),
        )
        self.stage2 = nn.Sequential(
            _ConvBnReLU(64, 128, s=2),
            _ConvBnReLU(128, 128),
            _ConvBnReLU(128, 128),
            _ConvBnReLU(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)   # (B, 64,  H,   W)
        x = self.stage2(x)   # (B, 128, H/2, W/2)
        return x


class PointPillarsHead(nn.Module):
    """Center-based detection head.

    Outputs for each spatial location:
      heatmap   — (C,)     probability of each class center
      offset    — (2,)     sub-pixel offset (ox, oy) ∈ (–1, 1)
      z_center  — (1,)     absolute Z of box center
      log_dim   — (3,)     log(width, length, height)
      heading   — (2,)     (sin θ, cos θ)
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        mid = in_channels
        self.heatmap_head = nn.Sequential(
            _ConvBnReLU(in_channels, mid),
            nn.Conv2d(mid, num_classes, 1),
        )
        self.offset_head  = nn.Sequential(
            _ConvBnReLU(in_channels, mid),
            nn.Conv2d(mid, 2, 1),
        )
        self.z_head       = nn.Sequential(
            _ConvBnReLU(in_channels, mid),
            nn.Conv2d(mid, 1, 1),
        )
        self.dim_head     = nn.Sequential(
            _ConvBnReLU(in_channels, mid),
            nn.Conv2d(mid, 3, 1),
        )
        self.heading_head = nn.Sequential(
            _ConvBnReLU(in_channels, mid),
            nn.Conv2d(mid, 2, 1),
        )

        # Initialise heatmap head bias for focal loss stability
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            'heatmap':  self.heatmap_head(x).sigmoid(),
            'offset':   self.offset_head(x).tanh(),
            'z_center': self.z_head(x),
            'log_dim':  self.dim_head(x),
            'heading':  self.heading_head(x),
        }


class PointPillars(nn.Module):
    """Full PointPillars model (PFN + scatter + backbone + head)."""

    def __init__(self, num_classes: int = 3, pfn_out: int = 64):
        super().__init__()
        self.pfn      = PillarFeatureNet(in_channels=9, out_channels=pfn_out)
        self.scatter  = PillarScatter(num_features=pfn_out)
        self.backbone = PointPillarsBEVBackbone(in_channels=pfn_out)
        self.head     = PointPillarsHead(in_channels=128, num_classes=num_classes)

    def forward(
        self,
        pillars:    torch.Tensor,    # (total_P, T, 9)
        coords:     torch.Tensor,    # (total_P, 3)
        batch_size: int,
        grid_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        feat  = self.pfn(pillars, pillars.shape[1])     # (total_P, 64)
        bev   = self.scatter(feat, coords, batch_size, grid_shape)
        bev   = self.backbone(bev)
        return self.head(bev)


# ── Detector implementation ───────────────────────────────────────────────────


@register_detector('PointPillarsDetector')
class PointPillarsDetector(BaseDetector):
    """PointPillars 3-D object detector for Waymo LiDAR.

    Backbone: custom PointPillars (PFN → scatter → 2-D CNN → center head).
    Switchable by replacing this class with any other BaseDetector subclass.

    YAML reference:
        model:
          type: PointPillarsDetector
          num_classes: 3
          pfn_out: 64

    Data format produced by WaymoLiDARDataset:
        points:    (N, 4) float32  [x, y, z, intensity]
        gt_boxes:  (M, 7) float32  [cx, cy, cz, dx, dy, dz, heading]
        gt_labels: (M,)   int32
    """

    # Mean box sizes for Gaussian radius computation (metres)
    _ANCHOR_SIZES: dict[int, tuple[float, float]] = {
        0: (4.5, 2.0),    # vehicle
        1: (0.9, 0.8),    # pedestrian
        2: (1.8, 0.6),    # cyclist
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ── Private helpers ───────────────────────────────────────────────────────

    def _vox_cfg(self, cfg: dict) -> dict:
        vc = cfg['voxel']
        return {
            'x_range':              tuple(vc['x_range']),
            'y_range':              tuple(vc['y_range']),
            'z_range':              tuple(vc['z_range']),
            'voxel_size':           tuple(vc['voxel_size']),
            'max_pillars':          vc['max_pillars'],
            'max_points_per_pillar': vc['max_points_per_pillar'],
        }

    def _pillarize_batch(
        self, points_list: list[np.ndarray], cfg: dict
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        """Voxelize all samples in a batch and combine with offset batch indices."""
        vc        = self._vox_cfg(cfg)
        all_pil   = []
        all_coord = []
        grid_shape = None

        for b_i, pts in enumerate(points_list):
            pil, coord, gs = _voxelize(
                pts,
                vc['x_range'], vc['y_range'], vc['z_range'],
                vc['voxel_size'],
                vc['max_pillars'],
                vc['max_points_per_pillar'],
            )
            if grid_shape is None:
                grid_shape = gs
            if len(pil):
                coord[:, 0] = b_i    # set batch index
                all_pil.append(pil)
                all_coord.append(coord)

        if all_pil:
            pillars = torch.from_numpy(np.concatenate(all_pil,  axis=0))
            coords  = torch.from_numpy(np.concatenate(all_coord, axis=0))
        else:
            pillars = torch.zeros(0, vc['max_points_per_pillar'], 9)
            coords  = torch.zeros(0, 3, dtype=torch.int32)

        return pillars, coords, grid_shape or (400, 400)

    # ── BaseDetector ─────────────────────────────────────────────────────────

    def build_model(self, cfg: dict, device: torch.device) -> nn.Module:
        from pathlib import Path
        num_classes = cfg['model'].get('num_classes', 3)
        pfn_out     = cfg['model'].get('pfn_out', 64)
        model       = PointPillars(num_classes=num_classes, pfn_out=pfn_out)

        weights = cfg.get('resume_weights')
        if weights and Path(weights).exists():
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), nn.Module):
                model = ckpt['model']
                print(f'Resumed from segment {ckpt.get("seg", "?")}')
            else:
                model.load_state_dict(torch.load(weights, map_location=device))

        return model.to(device)

    def loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        pillars    = batch['pillars']
        coords     = batch['coords']
        grid_shape = batch['grid_shape']
        batch_size = batch['batch_size']

        preds = model(pillars, coords, batch_size, grid_shape)

        # GT tensors — already on device
        hm_gt   = batch['heatmap']
        off_gt  = batch['offset']
        z_gt    = batch['z_center']
        dim_gt  = batch['log_dim']
        hd_gt   = batch['heading']
        mask    = batch['reg_mask']    # (B, H', W') bool

        # Heatmap focal loss
        loss_hm = _gaussian_focal_loss(preds['heatmap'], hm_gt)

        # Box regression — only on positive positions
        if mask.any():
            loss_off = F.l1_loss(preds['offset'][mask.unsqueeze(1).expand_as(preds['offset'])],
                                 off_gt[mask.unsqueeze(1).expand_as(off_gt)])
            loss_z   = F.l1_loss(preds['z_center'][mask.unsqueeze(1).expand_as(preds['z_center'])],
                                 z_gt[mask.unsqueeze(1).expand_as(z_gt)])
            loss_dim = F.l1_loss(preds['log_dim'][mask.unsqueeze(1).expand_as(preds['log_dim'])],
                                 dim_gt[mask.unsqueeze(1).expand_as(dim_gt)])
            loss_hd  = F.l1_loss(preds['heading'][mask.unsqueeze(1).expand_as(preds['heading'])],
                                  hd_gt[mask.unsqueeze(1).expand_as(hd_gt)])
            return loss_hm + loss_off + loss_z + loss_dim + loss_hd
        else:
            # No positive GT in batch — only heatmap loss; still has grad_fn
            return loss_hm

    def predict(
        self,
        model: nn.Module,
        points: np.ndarray,
        cfg: dict,
    ) -> list[dict]:
        """Run inference on a single (N, 4) XYZI point cloud.

        Args:
            model:  PointPillars nn.Module in .eval() mode.
            points: (N, 4) float32 XYZI array from load_lidar_points_xyzi().
            cfg:    Full config dict.

        Returns:
            List of {'box': [cx,cy,cz,dx,dy,dz,heading], 'label': int, 'score': float}.
        """
        device    = next(model.parameters()).device
        vc        = self._vox_cfg(cfg)
        num_classes = cfg['model'].get('num_classes', 3)
        score_thresh = cfg['eval'].get('score_thresh', 0.2)
        nms_iou      = cfg['eval'].get('nms_iou', 0.2)
        voxel_size   = vc['voxel_size']
        x_range      = vc['x_range']
        y_range      = vc['y_range']
        stride       = 2   # backbone reduces BEV by 2×

        pil, coord, grid_shape = _voxelize(
            points,
            vc['x_range'], vc['y_range'], vc['z_range'],
            voxel_size,
            vc['max_pillars'],
            vc['max_points_per_pillar'],
        )

        if len(pil) == 0:
            return []

        pillars_t = torch.from_numpy(pil).to(device)
        coords_t  = torch.from_numpy(coord).to(device)

        with torch.no_grad():
            preds = model(pillars_t, coords_t, 1, grid_shape)

        hm  = preds['heatmap'][0]    # (C, H', W')
        off = preds['offset'][0]
        z   = preds['z_center'][0]
        dim = preds['log_dim'][0]
        hd  = preds['heading'][0]

        Hs  = grid_shape[0] // stride
        Ws  = grid_shape[1] // stride

        detections = []
        for cls in range(num_classes):
            cls_hm = hm[cls]                              # (H', W')
            scores = cls_hm.cpu().numpy().flatten()
            idxs   = np.where(scores >= score_thresh)[0]
            for flat_idx in idxs:
                gy = flat_idx // Ws
                gx = flat_idx %  Ws
                score = float(scores[flat_idx])

                ox  = float(off[0, gy, gx])
                oy  = float(off[1, gy, gx])
                cx  = x_range[0] + ((gx + ox) * stride + 0.5) * voxel_size[0]
                cy  = y_range[0] + ((gy + oy) * stride + 0.5) * voxel_size[1]
                cz  = float(z[0, gy, gx])
                dx  = float(np.exp(dim[0, gy, gx].item()))
                dy  = float(np.exp(dim[1, gy, gx].item()))
                dz  = float(np.exp(dim[2, gy, gx].item()))
                s   = float(hd[0, gy, gx])
                c   = float(hd[1, gy, gx])
                heading = float(np.arctan2(s, c))

                detections.append({
                    'box':   [cx, cy, cz, dx, dy, dz, heading],
                    'label': cls,
                    'score': score,
                })

        return detections

    def collate_fn(self, batch: list) -> dict:
        """Collate LiDAR samples into a batch dict for loss().

        Input items: (points (N,4), gt_boxes (M,7), gt_labels (M,)) from
                     WaymoLiDARDataset.__getitem__().
        """
        points_list, boxes_list, labels_list = zip(*batch)
        B    = len(points_list)
        cfg  = self.cfg

        # Voxelize (CPU, numpy)
        pillars, coords, grid_shape = self._pillarize_batch(
            [np.asarray(p) for p in points_list], cfg
        )

        # Build GT heatmap targets
        vc         = self._vox_cfg(cfg)
        num_cls    = cfg['model'].get('num_classes', 3)
        stride     = 2
        Hs         = grid_shape[0] // stride
        Ws         = grid_shape[1] // stride

        hm_batch   = np.zeros((B, num_cls, Hs, Ws), np.float32)
        off_batch  = np.zeros((B, 2, Hs, Ws), np.float32)
        z_batch    = np.zeros((B, 1, Hs, Ws), np.float32)
        dim_batch  = np.zeros((B, 3, Hs, Ws), np.float32)
        hd_batch   = np.zeros((B, 2, Hs, Ws), np.float32)
        mask_batch = np.zeros((B, Hs, Ws), bool)

        for b_i in range(B):
            boxes  = np.asarray(boxes_list[b_i], dtype=np.float32)
            labels = np.asarray(labels_list[b_i], dtype=np.int32)
            if len(boxes):
                hm, off, z, dim, hd, mask, _ = _make_heatmap_targets(
                    boxes, labels, grid_shape,
                    vc['x_range'], vc['y_range'], vc['voxel_size'],
                    num_cls, self._ANCHOR_SIZES, stride=stride,
                )
                hm_batch[b_i]   = hm
                off_batch[b_i]  = off
                z_batch[b_i]    = z
                dim_batch[b_i]  = dim
                hd_batch[b_i]   = hd
                mask_batch[b_i] = mask

        return {
            'pillars':    pillars.float(),
            'coords':     coords.long(),
            'grid_shape': grid_shape,
            'batch_size': B,
            'heatmap':    torch.from_numpy(hm_batch),
            'offset':     torch.from_numpy(off_batch),
            'z_center':   torch.from_numpy(z_batch),
            'log_dim':    torch.from_numpy(dim_batch),
            'heading':    torch.from_numpy(hd_batch),
            'reg_mask':   torch.from_numpy(mask_batch),
        }
