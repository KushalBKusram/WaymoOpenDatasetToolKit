"""On-demand local Parquet cache for a single Waymo segment workflow."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable

import gcsfs
import numpy as np


GCS_BUCKET = "waymo_open_dataset_v_2_0_0"
WORKFLOW_COMPONENTS = {
    "Camera Frames": ("camera_image", "camera_box"),
    "LiDAR Frames": ("lidar", "lidar_calibration", "lidar_box"),
    "Segment Analysis": ("lidar_box",),
}
ProgressCallback = Callable[[str, int, int, int, int], None]


class SegmentCache:
    """Download only the Parquet components needed for a chosen workflow."""

    def __init__(self, root: str | Path = ".waymo_cache"):
        self.root = Path(root)

    def path_for(self, split: str, component: str, context_name: str) -> Path:
        return self.root / split / component / f"{context_name}.parquet"

    def has_component(self, split: str, component: str, context_name: str) -> bool:
        return self.path_for(split, component, context_name).is_file()


    def files_for_segment(self, split: str, context_name: str) -> list[Path]:
        """Return every cached file belonging to one exact segment."""
        split_root = self.root / split
        files = list(split_root.glob(f"*/{context_name}.parquet"))
        files.extend((split_root / "frame_indexes").glob(f"{context_name}_*.json"))
        decoded_root = split_root / "decoded_lidar_frames" / context_name
        if decoded_root.is_dir():
            files.extend(path for path in decoded_root.rglob("*") if path.is_file())
        return sorted(path for path in files if path.is_file())

    def segment_cache_size(self, split: str, context_name: str) -> int:
        return sum(path.stat().st_size for path in self.files_for_segment(split, context_name))

    def remove_segment(self, split: str, context_name: str) -> int:
        """Remove only this segment's cached files; return the removed byte count."""
        files = self.files_for_segment(split, context_name)
        removed_bytes = sum(path.stat().st_size for path in files)
        for path in files:
            path.unlink()
        decoded_root = self.root / split / "decoded_lidar_frames" / context_name
        if decoded_root.is_dir():
            shutil.rmtree(decoded_root)
        return removed_bytes

    def timeline_path(self, split: str, context_name: str, component: str = "lidar") -> Path:
        return self.root / split / "frame_indexes" / f"{context_name}_{component}.json"

    def load_timeline(self, split: str, context_name: str, component: str = "lidar") -> list[int] | None:
        path = self.timeline_path(split, context_name, component)
        if not path.is_file():
            return None
        return [int(timestamp) for timestamp in json.loads(path.read_text())]

    def save_timeline(self, split: str, context_name: str, timestamps: list[int], component: str = "lidar") -> Path:
        path = self.timeline_path(split, context_name, component)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(timestamps) + "\n")
        return path

    def lidar_frame_path(self, split: str, context_name: str, frame_index: int, top_lidar_only: bool) -> Path:
        mode = "top" if top_lidar_only else "all"
        return self.root / split / "decoded_lidar_frames" / context_name / mode / f"{frame_index:06d}.npz"

    def load_lidar_frame(self, split: str, context_name: str, frame_index: int, top_lidar_only: bool) -> list[np.ndarray] | None:
        path = self.lidar_frame_path(split, context_name, frame_index, top_lidar_only)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            return [archive[name] for name in sorted(archive.files)]

    def save_lidar_frame(self, split: str, context_name: str, frame_index: int, top_lidar_only: bool, points: list[np.ndarray]) -> Path:
        path = self.lidar_frame_path(split, context_name, frame_index, top_lidar_only)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.part")
        payload = {f"points_{index:02d}": array for index, array in enumerate(points)}
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def download_workflow(
        self,
        split: str,
        context_name: str,
        workflow: str,
        progress: ProgressCallback | None = None,
    ) -> list[Path]:
        components = WORKFLOW_COMPONENTS[workflow]
        filesystem = gcsfs.GCSFileSystem()
        paths: list[Path] = []
        for index, component in enumerate(components, start=1):
            paths.append(self._download_component(
                filesystem, split, context_name, component, index, len(components), progress,
            ))
        return paths

    def _download_component(
        self,
        filesystem: gcsfs.GCSFileSystem,
        split: str,
        context_name: str,
        component: str,
        index: int,
        count: int,
        progress: ProgressCallback | None,
    ) -> Path:
        destination = self.path_for(split, component, context_name)
        if destination.is_file():
            size = destination.stat().st_size
            if progress:
                progress(component, size, size, index, count)
            return destination
        remote = f"{GCS_BUCKET}/{split}/{component}/{context_name}.parquet"
        total = int(filesystem.info(remote).get("size", 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.part")
        downloaded = 0
        try:
            with filesystem.open(remote, "rb") as source, temporary.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
                    downloaded += len(chunk)
                    if progress:
                        progress(component, downloaded, total, index, count)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination
