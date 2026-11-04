"""Test setup: make the repository importable and stub the heavy, network-bound dependencies.

The tests only exercise pure-Python logic (tracking, scoring, manifests, configs), so torch, the
detector registry and the Waymo GCS reader are replaced by empty stand-ins when they are not
needed. If torch is installed the real one is used.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


try:
    import torch  # noqa: F401
except ImportError:
    _stub("torch", manual_seed=lambda seed: None,
          cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda seed: None))

_stub("models", build_detector=lambda cfg: None)
_stub("modules.waymo_open_dataset", YOLO_CLASS_MAP={1: 0, 2: 1, 4: 2, 3: 3}, ToolKit=object, _C_BOX="[CameraBoxComponent]")
