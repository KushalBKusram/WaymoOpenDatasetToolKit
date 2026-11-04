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

import cv2
import numpy as np
import torch
import torch.nn as nn

from models import register_detector
from models.base_detector import BaseDetector
from modules.runtime import resolve_image_hw


def load_with_new_head(model_cls, yaml_name: str, pretrained: nn.Module, cfg: dict,
                       in_channels: int = 3) -> nn.Module:
    """Build ``model_cls`` for our classes and copy every matching pretrained tensor.

    The COCO classification head (80 classes) does not match ours, so it keeps
    its fresh initialisation; the backbone and box branches are transferred.
    Extra input channels start from the mean of the pretrained RGB filters.
    """
    num_classes = int(cfg['model'].get('num_classes', 4))
    names = cfg['model'].get('class_names') or [str(i) for i in range(num_classes)]
    if in_channels == 3 and getattr(pretrained, 'nc', None) == num_classes:
        model = pretrained
    else:
        model = model_cls(yaml_name, ch=in_channels, nc=num_classes, verbose=False)
        model.load(pretrained, verbose=False)
        if in_channels != 3:
            old = pretrained.model[0].conv.weight.data
            new = model.model[0].conv.weight.data
            new[:, :3] = old
            new[:, 3:] = old.mean(dim=1, keepdim=True)
    model.names = {index: name for index, name in enumerate(names)}
    model.nc = num_classes   # the Ultralytics trainer sets this; RT-DETR's loss reads it
    return model


@register_detector('YOLOv8Detector')
class YOLOv8Detector(BaseDetector):
    """YOLOv8-family 2-D object detector for Waymo camera images.

    Uses v8DetectionLoss for training (same loss as the Ultralytics training
    loop) and YOLO.predict() for inference so NMS and coordinate rescaling
    are handled internally.
    """

    input_channels = 3

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _default_weights(self, cfg: dict) -> str:
        return f"yolov8{cfg['model'].get('backbone', 'n')}.pt"

    def _model_spec(self, cfg: dict):
        """(Ultralytics loader class, model class, model yaml) for this family."""
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        return YOLO, DetectionModel, f"yolov8{cfg['model'].get('backbone', 'n')}.yaml"

    def _model_from_state(self, cfg: dict, ckpt: dict, device: torch.device) -> nn.Module:
        """Rebuild the architecture from the config and load exported weights into it."""
        _, model_cls, yaml_name = self._model_spec(cfg)
        num_classes = int(cfg['model'].get('num_classes', 4))
        names = cfg['model'].get('class_names') or [str(i) for i in range(num_classes)]
        model = model_cls(yaml_name, ch=self.input_channels, nc=num_classes, verbose=False)
        # The trained head can be wider than a fresh build for this class count (the COCO-sized
        # hidden width is kept when the 80-class head is replaced), and the default width differs
        # between Ultralytics versions, so size the classification branch from the saved tensors.
        detect = model.model[-1]
        width = ckpt['state_dict'][f'model.{len(model.model) - 1}.cv3.0.0.conv.weight'].shape[0]
        if detect.cv3[0][0].conv.weight.shape[0] != width:
            from ultralytics.nn.modules import Conv
            detect.cv3 = nn.ModuleList(
                nn.Sequential(Conv(branch[0].conv.in_channels, width, 3), Conv(width, width, 3),
                              nn.Conv2d(width, num_classes, 1))
                for branch in detect.cv3)
        # Match the activations the weights were trained with, and Ultralytics' BatchNorm settings.
        modules = dict(model.named_modules())
        for name, kind in ckpt.get('activations', {}).items():
            if name in modules and type(modules[name].act).__name__ != kind:
                modules[name].act = getattr(nn, kind)()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-3, 0.03
        model.load_state_dict(ckpt['state_dict'])
        model.names = {index: name for index, name in enumerate(names)}
        model.nc = num_classes
        print(f'Loaded exported weights ({ckpt.get("source_run") or "unnamed run"})')
        return model.float().to(device)

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

        loader, model_cls, yaml_name = self._model_spec(cfg)
        weights  = cfg.get('resume_weights') or self._default_weights(cfg)

        nn_model = None
        path = Path(weights)
        if path.exists():
            try:
                # Exported weights (scripts/export_weights.py) hold only tensors and metadata.
                ckpt = torch.load(weights, map_location=device, weights_only=True)
            except Exception:
                ckpt = torch.load(weights, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and ckpt.get('format') == 'waymo-toolkit-weights-v1':
                nn_model = self._model_from_state(cfg, ckpt, device)
                ckpt = None
            # Toolkit checkpoints carry an explicit segment number. Ultralytics
            # checkpoints also contain a ``model`` module, so checking that field
            # alone would incorrectly retain their half-precision export weights.
            if isinstance(ckpt, dict) and 'seg' in ckpt and isinstance(ckpt.get('model'), nn.Module):
                nn_model = ckpt['model'].float().to(device)
                print(f'Resumed from segment {ckpt["seg"]}')
        if nn_model is None:
            # Pretrained Ultralytics weights (auto-downloaded if absent). The COCO
            # head has 80 classes, so rebuild a head for our classes and keep the
            # backbone and box branches.
            pretrained = loader(weights).model.float()
            nn_model = load_with_new_head(
                model_cls, yaml_name, pretrained, cfg, self.input_channels,
            ).to(device)

        # v8DetectionLoss reads model.args as an IterableSimpleNamespace.
        # When loaded outside the Ultralytics training loop, args may be a
        # plain dict — replace it with the default config namespace.
        nn_model.args = get_cfg(DEFAULT_CFG)
        # Ultralytics checkpoints are commonly exported with parameters frozen;
        # its Trainer unfreezes them before optimization, so do the same here.
        for parameter in nn_model.parameters():
            parameter.requires_grad_(True)
        return nn_model

    def loss(self, model: nn.Module, batch: dict) -> torch.Tensor:
        """v8DetectionLoss forward pass."""
        from ultralytics.utils.loss import v8DetectionLoss

        loss_fn = v8DetectionLoss(model)
        preds   = model(batch['img'])
        loss_components, _ = loss_fn(preds, batch)
        # Current Ultralytics returns [box, cls, dfl] components; its Trainer
        # backpropagates their sum. Expose one scalar through this adapter.
        return loss_components.sum()

    def predict(
        self,
        model: nn.Module,
        img_bgr: np.ndarray,
        cfg: dict,
    ) -> list[dict]:
        """Inference via YOLO.predict() — handles NMS + coordinate rescaling.

        Returns list of {'box': [x1,y1,x2,y2], 'label': int, 'score': float}.
        """
        eval_cfg = cfg.get('eval', {})
        conf     = eval_cfg.get('conf', 0.25)
        iou      = eval_cfg.get('iou',  0.45)
        net_h, net_w = resolve_image_hw(cfg['data'].get('img_size', 640))

        # Wrap the raw nn.Module in an Ultralytics shell once; rebuilding it per
        # frame reloads weights from disk and dominates measured latency.
        shells = self.__dict__.setdefault('_yolo_shells', {})
        yolo = shells.get(id(model))
        if yolo is None:
            loader = self._model_spec(cfg)[0]
            yolo = loader(self._default_weights(cfg))
            yolo.model = model.eval()
            shells[id(model)] = yolo

        # Training resizes every image straight to the network size (labels are
        # normalised), so do the same here instead of letting predict() letterbox.
        # Boxes come back in network-input pixels and are scaled to the original frame.
        height, width = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (net_w, net_h))
        results    = yolo.predict(resized, conf=conf, iou=iou,
                                  imgsz=[net_h, net_w], verbose=False)
        scale = (width / net_w, height / net_h, width / net_w, height / net_h)
        detections: list[dict] = []
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            for i in range(len(r.boxes)):
                box = r.boxes.xyxy[i].cpu().tolist()
                detections.append({
                    'box':   [coord * factor for coord, factor in zip(box, scale)],
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
