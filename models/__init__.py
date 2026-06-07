"""
models/__init__.py — Detector registry.

Usage in train.py:
    from models import build_detector
    detector = build_detector(cfg)   # cfg['model']['type'] selects the class

Adding a new detector (three steps):
    1. Create models/<name>_detector.py with a BaseDetector subclass
    2. Decorate the class: @register_detector('YourTypeName')
    3. Add an import line at the bottom of this file so the decorator runs

The registry is populated by the decorator at import time; no manual mapping
table needed.
"""

from __future__ import annotations

from .base_detector import BaseDetector

_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_detector(name: str):
    """Class decorator — register a BaseDetector subclass under a string key.

    Example::

        @register_detector('YOLOv8Detector')
        class YOLOv8Detector(BaseDetector):
            ...

    The name must match cfg['model']['type'] in the YAML config.
    """
    def decorator(cls: type[BaseDetector]) -> type[BaseDetector]:
        if name in _REGISTRY:
            raise ValueError(
                f"Detector {name!r} is already registered. "
                "Choose a unique name or remove the duplicate."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator


def build_detector(cfg: dict) -> BaseDetector:
    """Instantiate a detector from a config dict.

    Args:
        cfg: Full YAML config dict.  cfg['model']['type'] must match a
             name passed to @register_detector.

    Returns:
        Instantiated BaseDetector subclass, initialised with cfg.

    Raises:
        KeyError: If the type name is not in the registry.
    """
    model_type = cfg['model']['type']
    if model_type not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown detector type: {model_type!r}. "
            f"Registered types: {available}.\n"
            f"Check the 'model.type' field in your YAML config."
        )
    return _REGISTRY[model_type](cfg)


def list_detectors() -> list[str]:
    """Return sorted list of all registered detector type names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Auto-import detector modules so their @register_detector decorators fire.
# Add one line here for each new detector family.
# ---------------------------------------------------------------------------
from . import yolov8_detector        # noqa: F401, E402
from . import pointpillars_detector  # noqa: F401, E402
