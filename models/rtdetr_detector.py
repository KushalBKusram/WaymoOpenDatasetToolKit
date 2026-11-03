"""RT-DETR (Ultralytics): a pretrained transformer detector for Waymo camera images.

Shares the YOLOv8 adapter's batch format and inference path. The Ultralytics
checkpoint is COCO-pretrained; the classification head is rebuilt for our
classes and the rest of the network is fine-tuned.

    model:
      type: RTDETRDetector
      variant: l            # l | x
      num_classes: 4
"""

from __future__ import annotations

import torch

from models import register_detector
from models.yolov8_detector import YOLOv8Detector


@register_detector('RTDETRDetector')
class RTDETRDetector(YOLOv8Detector):
    def _default_weights(self, cfg: dict) -> str:
        return f"rtdetr-{cfg['model'].get('variant', 'l')}.pt"

    def _model_spec(self, cfg: dict):
        from ultralytics import RTDETR
        from ultralytics.nn.tasks import RTDETRDetectionModel
        return RTDETR, RTDETRDetectionModel, f"rtdetr-{cfg['model'].get('variant', 'l')}.yaml"

    def loss(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        return model.loss(batch)[0]
