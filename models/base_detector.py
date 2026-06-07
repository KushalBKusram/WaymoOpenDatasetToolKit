"""
base_detector.py — Abstract base class for all object detectors.

Every detector plugged into train.py must implement four methods:

    build_model(cfg, device) → nn.Module
        Load / initialise weights; return the model ready for .train().

    loss(model, batch) → Tensor
        Run a forward pass and return the scalar loss tensor.
        Return a Tensor with grad_fn is None when the batch has no
        positive targets (training loop skips the parameter update).

    predict(model, img_bgr, cfg) → list[dict]
        Run inference with NMS on a single BGR image; return a list of
        {'box': [x1,y1,x2,y2], 'label': int, 'score': float} dicts.

    collate_fn(batch) → dict
        Collate a list of (img_tensor, labels_tensor) dataset items into
        whatever dict format the detector's loss function expects.

To register a new detector:
    1. Subclass BaseDetector in models/<name>_detector.py
    2. Decorate the class with @register_detector('YourTypeName')
    3. Import the module in models/__init__.py so the decorator fires
    4. Create configs/<name>.yaml with  model.type: YourTypeName

The models/__init__.py auto-imports all detector modules, so step 3 only
requires adding one line there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BaseDetector(ABC):
    """Abstract interface for pluggable object detection backbones.

    Subclass this to add a new detector family (RT-DETR, PointPillars, …).
    All config values are passed as a plain dict loaded from a YAML file so
    the interface stays free of library-specific types.
    """

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abstractmethod
    def build_model(
        self,
        cfg: dict,
        device: torch.device,
    ) -> nn.Module:
        """Initialise and return the model with weights loaded.

        Implementations should handle three cases:
          1. Fresh pretrained weights from the upstream library.
          2. Resuming from a toolkit checkpoint (our own .pt format).
          3. Custom weight path supplied via cfg['resume_weights'].

        Args:
            cfg:    Full config dict loaded from YAML.
            device: Target compute device.

        Returns:
            nn.Module — not yet set to .train() mode; the caller does that.
        """

    @abstractmethod
    def loss(
        self,
        model: nn.Module,
        batch: dict,
    ) -> torch.Tensor:
        """Forward pass + loss computation on one batch.

        Args:
            model: The nn.Module returned by build_model(), already on device
                   and in .train() mode.
            batch: Dict produced by collate_fn(), already moved to device.

        Returns:
            Scalar loss Tensor.  If grad_fn is None (batch has no positive
            targets), the training loop will skip the parameter update.
        """

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        img_bgr: np.ndarray,
        cfg: dict,
    ) -> list[dict]:
        """Run inference on a single BGR image and return detections.

        Args:
            model:   The nn.Module in .eval() mode.
            img_bgr: (H, W, 3) uint8 BGR numpy array.
            cfg:     Full config dict — implementations should read
                     cfg['eval']['conf'] and cfg['eval']['iou'].

        Returns:
            List of detection dicts:
                {'box': [x1, y1, x2, y2], 'label': int, 'score': float}
            Empty list if no detections pass the confidence threshold.
        """

    @abstractmethod
    def collate_fn(self, batch: list) -> dict:
        """Collate dataset items into a batch dict for loss().

        This method is passed directly as the DataLoader collate_fn.

        Args:
            batch: List of (img_tensor, labels_tensor) tuples from
                   WaymoGCSDataset.__getitem__().
                   - img_tensor:    (3, H, W) float32, values in [0, 1]
                   - labels_tensor: (N, 5) float32 — [class_id, cx, cy, w, h]
                                    normalised to [0, 1]; empty if no boxes.

        Returns:
            Dict with at minimum an 'img' key (B, 3, H, W) float32 tensor.
            Additional keys depend on what loss() expects.
        """
