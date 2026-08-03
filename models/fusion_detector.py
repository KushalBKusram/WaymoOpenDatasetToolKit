"""Calibrated RGB--LiDAR early-fusion detector.

The model consumes five channels: RGB plus sparse projected LiDAR depth and
reflectance.  It predicts normal 2-D boxes in the selected camera.  This is a
useful, inspectable fusion baseline; it is not a full multi-view 3-D fusion
network.
"""

from __future__ import annotations

import numpy as np
import torch

from models import register_detector
from models.transformer_detector import TransformerDetector
from modules.fusion import build_fusion_tensor


@register_detector('CameraLiDARFusionDetector')
class CameraLiDARFusionDetector(TransformerDetector):
    """Early RGB + calibrated LiDAR-raster fusion using object queries."""

    input_channels = 5

    def predict(self, model: torch.nn.Module, img_bgr: np.ndarray, cfg: dict,
                *, lidar_points: np.ndarray | None = None, camera_calibration=None,
                **_: object) -> list[dict]:
        if lidar_points is None or camera_calibration is None:
            raise ValueError(
                'CameraLiDARFusionDetector.predict requires lidar_points and '
                'camera_calibration for the same timestamp and camera.'
            )
        image_size = int(cfg['data'].get('img_size', 512))
        fusion = build_fusion_tensor(
            img_bgr, lidar_points, camera_calibration, image_size,
            max_depth=float(cfg['data'].get('fusion_max_depth', 75.0)),
        )
        device = next(model.parameters()).device
        with torch.no_grad():
            output = model(fusion.unsqueeze(0).to(device))
        probabilities = output['logits'][0].softmax(-1)[:, :-1]
        scores, labels = probabilities.max(dim=-1)
        threshold = float(cfg.get('eval', {}).get('conf', 0.25))
        valid = scores >= threshold
        from models.transformer_detector import _cxcywh_to_xyxy, _nms
        boxes = _cxcywh_to_xyxy(output['boxes'][0][valid]).clamp(0, 1)
        scores, labels = scores[valid], labels[valid]
        if not len(boxes):
            return []
        height, width = img_bgr.shape[:2]
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        kept: list[int] = []
        iou_threshold = float(cfg.get('eval', {}).get('iou', 0.45))
        for label in labels.unique():
            indices = torch.where(labels == label)[0]
            kept.extend(indices[_nms(boxes[indices], scores[indices], iou_threshold)].tolist())
        return [{'box': boxes[index].detach().cpu().tolist(), 'label': int(labels[index]),
                 'score': float(scores[index])} for index in kept]
