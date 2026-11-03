"""YOLOv8 with calibrated LiDAR depth/intensity as two extra input channels.

Same network as the camera-only YOLOv8 detector except for the first
convolution, so a comparison between the two isolates the LiDAR input. The
extra channels start from the mean of the pretrained RGB filters.
"""

from __future__ import annotations

import numpy as np
import torch

from models import register_detector
from models.yolov8_detector import YOLOv8Detector
from modules.fusion import build_fusion_tensor


@register_detector('YOLOFusionDetector')
class YOLOFusionDetector(YOLOv8Detector):
    input_channels = 5

    def predict(self, model: torch.nn.Module, img_bgr: np.ndarray, cfg: dict,
                *, lidar_points: np.ndarray | None = None, camera_calibration=None,
                **_: object) -> list[dict]:
        if lidar_points is None or camera_calibration is None:
            raise ValueError(
                'YOLOFusionDetector.predict requires lidar_points and camera_calibration '
                'for the same timestamp and camera.'
            )
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # older Ultralytics releases kept it in ops
            from ultralytics.utils.ops import non_max_suppression

        eval_cfg = cfg.get('eval', {})
        size = int(cfg['data'].get('img_size', 512))
        tensor = build_fusion_tensor(
            img_bgr, lidar_points, camera_calibration, size,
            max_depth=float(cfg['data'].get('fusion_max_depth', 75.0)),
        )
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            output = model(tensor.unsqueeze(0).to(device))
        predictions = output[0] if isinstance(output, (list, tuple)) else output
        kept = non_max_suppression(
            predictions, conf_thres=eval_cfg.get('conf', 0.25), iou_thres=eval_cfg.get('iou', 0.45),
        )[0]
        height, width = img_bgr.shape[:2]
        scale = (width / size, height / size, width / size, height / size)
        return [{'box': [coord * factor for coord, factor in zip(row[:4], scale)],
                 'label': int(row[5]), 'score': float(row[4])} for row in kept.tolist()]
