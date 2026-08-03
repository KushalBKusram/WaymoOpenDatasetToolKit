"""Small, dependency-light helpers for reproducible training/evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunArtifacts:
    """Persist configs, metrics, and visual-review records for one run."""

    def __init__(self, run_dir: str | Path, cfg: dict[str, Any] | None = None):
        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.path / "metrics.json"
        if cfg is not None:
            self.write_json("config.json", cfg)

    def write_json(self, name: str, data: Any) -> Path:
        path = self.path / name
        path.write_text(json.dumps(data, indent=2, default=str) + "\n")
        return path

    def read_metrics(self) -> dict[str, Any]:
        if not self.metrics_path.exists():
            return {"schema_version": 1, "events": []}
        return json.loads(self.metrics_path.read_text())

    def record(self, event_type: str, **values: Any) -> None:
        data = self.read_metrics()
        data.setdefault("schema_version", 1)
        data.setdefault("events", []).append(
            {"type": event_type, "timestamp": utc_now(), **values}
        )
        self.write_json("metrics.json", data)
