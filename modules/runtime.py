"""Runtime helpers shared by training and evaluation entry points."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_torch_device(requested: str = "auto") -> torch.device:
    """Select an available PyTorch device, including Apple Silicon MPS."""
    choice = requested.lower()
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": mps_available}
    if choice == "auto":
        choice = "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if choice not in available:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if not available[choice]:
        raise RuntimeError(f"Requested device {choice!r} is not available on this machine.")
    return torch.device(choice)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for comparable experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
