"""Appearance embeddings for tracking: ImageNet ResNet-18 features of detection crops.

No training: each detection box is cropped (with a little padding), resized to 128 x 128, passed through a
pretrained ResNet-18 and global-average-pooled to a 512-d vector, then L2-normalised. Cosine distance between
two such vectors is the appearance distance used by `modules.tracking.SortTracker`.

Embeddings are cached per detector run (`attach_embeddings`) so tracker settings can be tuned without
re-embedding.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

CROP_SIZE = 128
PADDING = 0.1                      # fraction of the box size added on each side before cropping
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_boxes(image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
    """BGR image + xyxy boxes -> (N, 3, CROP_SIZE, CROP_SIZE) float32 normalised RGB crops."""
    import cv2

    height, width = image.shape[:2]
    crops = np.zeros((len(boxes), 3, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        pad_x, pad_y = PADDING * (x2 - x1), PADDING * (y2 - y1)
        left, top = int(max(np.floor(x1 - pad_x), 0)), int(max(np.floor(y1 - pad_y), 0))
        right, bottom = int(min(np.ceil(x2 + pad_x), width)), int(min(np.ceil(y2 + pad_y), height))
        right, bottom = max(right, left + 2), max(bottom, top + 2)                 # never an empty crop
        patch = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB)
        patch = cv2.resize(patch, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        crops[index] = ((patch.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    return crops


class ResNetEmbedder:
    """ImageNet ResNet-18 (torchvision) used as a generic appearance encoder."""

    def __init__(self, device: str = "auto", batch_size: int = 128):
        import torch
        import torchvision

        from modules.runtime import resolve_torch_device

        self.torch = torch
        self.device = resolve_torch_device(device)
        self.batch_size = batch_size
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    def embed(self, image: np.ndarray, boxes: list[list[float]]) -> np.ndarray:
        """(N, 512) float32 unit vectors for the given boxes of one BGR image."""
        if not boxes:
            return np.zeros((0, 512), dtype=np.float32)
        crops = crop_boxes(image, boxes)
        out = []
        with self.torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = self.torch.from_numpy(crops[start:start + self.batch_size]).to(self.device)
                features = self.model(batch).float().cpu().numpy()
                out.append(features)
        features = np.concatenate(out, axis=0)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, 1e-12)


def attach_embeddings(toolkit, frames: dict, camera: int, run_dir: Path, device: str = "auto") -> float:
    """Add an 'embedding' to every detection in `frames` (as returned by track_eval.collect).

    Embeddings are cached in `<run_dir>/tracking/embeddings.npz`, one array per segment (all detections of
    the segment in order) plus the per-frame detection counts; a segment is re-embedded if the cached counts
    no longer match the detections. Returns the seconds spent embedding (0 when everything was cached).
    """
    cache_path = Path(run_dir) / "tracking" / "embeddings.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    embedder, spent, dirty = None, 0.0, False
    for segment, segment_frames in frames.items():
        counts = np.array([len(dets) for _, dets in segment_frames], dtype=np.int64)
        cached = cache.get(f"{segment}|emb")
        if cached is None or not np.array_equal(cache.get(f"{segment}|counts"), counts):
            if embedder is None:
                embedder = ResNetEmbedder(device)
            started = time.time()
            toolkit.assign_segment(segment)
            timestamps = sorted(int(t) for t in toolkit.get_timestamps())
            pieces = []
            for timestamp, (_, dets) in zip(timestamps, segment_frames):
                image = toolkit.load_camera_frame(timestamp, camera) if dets else None
                pieces.append(embedder.embed(image, [d["box"] for d in dets]) if dets else np.zeros((0, 512), np.float32))
            cached = np.concatenate(pieces, axis=0).astype(np.float16)
            cache[f"{segment}|emb"], cache[f"{segment}|counts"] = cached, counts
            spent += time.time() - started
            dirty = True
        offset = 0
        for _, dets in segment_frames:
            for det in dets:
                det["embedding"] = cached[offset].astype(np.float32)
                offset += 1
    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **cache)
    return spent
