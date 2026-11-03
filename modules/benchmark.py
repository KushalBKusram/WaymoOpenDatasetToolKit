"""Frozen benchmark split manifest and deterministic frame sampling."""

from __future__ import annotations

import json
from pathlib import Path


def load_split_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    overlap = set(manifest["train"]) & set(manifest["eval"])
    if overlap:
        raise ValueError(f"Train and eval segments overlap in {path}: {sorted(overlap)[:3]}")
    return manifest


def stride_timestamps(timestamps: list, stride: int = 1) -> list:
    """Keep every ``stride``-th timestamp, starting from the first."""
    if stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {stride}")
    return list(timestamps[::stride])
