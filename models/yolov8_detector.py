"""
yolov8_detector.py — YOLOv8 detector implementing BaseDetector.

Wraps the Ultralytics YOLOv8 model family (nano → xlarge) with a custom
training loop that feeds Waymo GCS data directly without using Ultralytics'
built-in data pipeline (which requires files on disk).

Registered as 'YOLOv8Detector'.  Reference from a YAML config:

    model:
      type: YOLOv8Detector
      backbone: n          # n | s | m | l | x
      num_classes: 4
      class_names: [vehicle, pedestrian, cyclist, sign]

Backbone VRAM budget (approximate, batch=8, img=640):
    n (nano)   ~4 GB  — Colab free T4
    s (small)  ~6 GB  — Colab Pro T4 / A100
    m (medium) ~9 GB  — A100
    l / x      ~14+ GB — A100 / H100
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models import register_detector
from models.base_detector import BaseDetector


@register_detector('YOLOv8Detector')
class YOLOv8Detector(BaseDetector):
    """YOLOv8-family 2-D object detector for Waymo camera images.

    Uses v8DetectionLoss for training (same loss as the Ultralytics training
    loop) and YOLO.predict() for inference so NMS and coordinate rescaling
    are handled internally.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    # ── BaseDetector ─────────────────────────────────────────────────────────

    def build_model(self, cfg: dict, device: torch.device) -> nn.Module:
        """Load YOLOv8 weights and prepare for training.

        Load priority (first match wins):
          1. cfg['resume_weights'] — a toolkit .pt checkpoint or Ultralytics .pt
          2. Pretrained yolov8{backbone}.pt from Ultralytics (auto-downloaded)
        """
        try:
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
            from ultralytics.utils import DEFAULT_CFG
        except ImportError as exc:
            raise ImportError(
                'ultralytics not installed. '
                'Run: pip install ultralytics'
            ) from exc

        backbone = cfg['model'].get('backbone', 'n')
        weights  = cfg.get('resume_weights') or f'yolov8{backbone}.pt'

        path = Path(weights)
        if path.exists():
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            # Toolkit checkpoint format: {'model': nn.Module, 'optimizer': ..., 'seg': int}
            if isinstance(ckpt, dict) and isinstance(ckpt.get('model'), nn.Module):
                nn_model = ckpt['model'].to(device)
                print(f'Resumed from segment {ckpt.get("seg", "?")}')
            else:
                # Ultralytics format
                nn_model = YOLO(weights).model.to(device)
        else:
            # Fresh pretrained weights (auto-downloaded from Ultralytics)
            nn_model = YOLO(weights).model.to(device)

        # v8DetectionLoss reads model.args as an IterableSimpleNamespace.
        # When loaded outside the Ultralytics training loop, args may be a
        # plain dict — replace it with the default config namespace.
        nn_model.args = get_cfg(DEFAULT_CFG)
        return nn_model

    def loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """v8DetectionLoss forward pass."""
        from ultralytics.utils.loss import v8DetectionLoss

        loss_fn = v8DetectionLoss(model)
        preds   = model(batch['img'])
        loss, _ = loss_fn(preds, batch)
        return loss

    def predict(
        self,
        model: nn.Module,
        img_bgr: np.ndarray,
        cfg: dict,
    ) -> list[dict]:
        """Inference via YOLO.predict() — handles NMS + coordinate rescaling.

        Returns list of {'box': [x1,y1,x2,y2], 'label': int, 'score': float}.
        """
        from ultralytics import YOLO

        eval_cfg = cfg.get('eval', {})
        conf     = eval_cfg.get('conf', 0.25)
        iou      = eval_cfg.get('iou',  0.45)
        img_size = cfg['data'].get('img_size', 640)
        backbone = cfg['model'].get('backbone', 'n')

        # Wrap the raw nn.Module in a YOLO shell for inference
        yolo = YOLO(f'yolov8{backbone}.pt')
        yolo.model = model.eval()

        results    = yolo.predict(img_bgr, conf=conf, iou=iou,
                                  imgsz=img_size, verbose=False)
        detections: list[dict] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                detections.append({
                    'box':   r.boxes.xyxy[i].cpu().tolist(),
                    'label': int(r.boxes.cls[i]),
                    'score': float(r.boxes.conf[i]),
                })
        return detections

    def collate_fn(self, batch: list) -> dict:
        """Produce the batch dict expected by v8DetectionLoss.

        Input items: (img_tensor (3,H,W) float32, labels (N,5) float32)
        Labels format: [class_id, cx, cy, w, h] normalised to [0,1].
        """
        imgs, labels_list = zip(*batch)
        imgs = torch.stack(imgs)   # (B, 3, H, W)

        cls_all, bboxes_all, bidx_all = [], [], []
        for i, lbl in enumerate(labels_list):
            if len(lbl):
                cls_all.append(lbl[:, 0])
                bboxes_all.append(lbl[:, 1:])
                bidx_all.append(torch.full((len(lbl),), float(i)))

        return {
            'img':       imgs,
            'cls':       torch.cat(cls_all)    if cls_all    else torch.zeros(0),
            'bboxes':    torch.cat(bboxes_all) if bboxes_all else torch.zeros((0, 4)),
            'batch_idx': torch.cat(bidx_all)   if bidx_all   else torch.zeros(0),
        }
