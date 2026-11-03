"""
train.py — Config-driven training script for Waymo object detection.

The model backbone and all hyperparameters live in a YAML config file.
New detector families (LiDAR, pose, …) plug in via the models/ registry
without touching this file.

Usage examples:

  # Standard training run (nano backbone, free Colab T4)
  python train.py --config configs/yolov8n.yaml

  # With Drive directory for checkpoints
  python train.py --config configs/yolov8n.yaml \\
                  --drive-dir /content/drive/MyDrive/waymo

  # Override segment count or batch for a quick test
  python train.py --config configs/yolov8n.yaml --total-segs 5 --batch 4

  # Training resumes automatically when latest.pt exists in drive-dir
  python train.py --config configs/yolov8n.yaml \\
                  --drive-dir /content/drive/MyDrive/waymo

Prerequisites (Colab):
  from google.colab import auth; auth.authenticate_user()
  from google.colab import drive; drive.mount('/content/drive')
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

# Share batches between workers through shared-memory files, not file descriptors:
# a new DataLoader per segment otherwise exhausts the open-file limit on long runs.
torch.multiprocessing.set_sharing_strategy('file_system')

from models import build_detector
from modules.benchmark import load_split_manifest, stride_timestamps
from modules.run_artifacts import RunArtifacts
from modules.fusion import build_fusion_tensor
from modules.runtime import resolve_torch_device, set_global_seed
from modules.waymo_open_dataset import (
    LIDAR_DET_CLASS_MAP,
    YOLO_CLASS_MAP,
    ToolKit,
    _C_BOX,
    _L_BOX,
)


# ── Dataset ───────────────────────────────────────────────────────────────────


class WaymoGCSDataset(Dataset):
    """Streams camera images + 2-D box labels from one GCS Parquet segment.

    No files are written to disk.  Each __getitem__ decodes a JPEG stored
    as bytes in the Parquet column directly into a float32 tensor.

    Args:
        toolkit:     ToolKit instance with an active segment assigned.
        imgsz:       Square image size to resize to.
        cameras:     Camera IDs to include.
        max_frames:  Optional limit on timestamps per segment (for smoke tests).
        frame_stride: Keep every Nth timestamp (applied before max_frames).
    """

    def __init__(
        self,
        toolkit: ToolKit,
        imgsz: int = 640,
        cameras: tuple = (1, 2, 3, 4, 5),
        max_frames: int | None = None,
        frame_stride: int = 1,
    ):
        self.toolkit = toolkit
        self.imgsz   = imgsz

        cam_df       = toolkit._read_cached('camera_image')
        self.box_df  = toolkit._read_cached('camera_box')
        timestamps = stride_timestamps(
            sorted(cam_df['key.frame_timestamp_micros'].unique().tolist()), frame_stride
        )
        if max_frames is not None:
            timestamps = timestamps[:max_frames]
        selected_timestamps = set(timestamps)
        self.index = [
            (row['key.frame_timestamp_micros'], row['key.camera_name'])
            for _, row in cam_df.iterrows()
            if row['key.camera_name'] in cameras and row['key.frame_timestamp_micros'] in selected_timestamps
        ]
        labeled_keys = set(zip(
            self.box_df['key.frame_timestamp_micros'], self.box_df['key.camera_name']
        ))
        labeled = sum((ts, cam) in labeled_keys for ts, cam in self.index)
        print(f'   label coverage: {labeled}/{len(self.index)} frames have boxes')

    def __len__(self) -> int:
        return len(self.index)

    def _load_image_and_labels(self, i: int):
        """Load an unscaled BGR image and normalized 2-D labels."""
        ts, cam = self.index[i]
        image = self.toolkit.load_camera_frame(ts, cam)
        h0, w0 = image.shape[:2]
        boxes_df = self.box_df[
            (self.box_df['key.frame_timestamp_micros'] == ts) &
            (self.box_df['key.camera_name'] == cam)
        ]
        labels = []
        for _, row in boxes_df.iterrows():
            type_int = int(row[f'{_C_BOX}.type'])
            if type_int not in YOLO_CLASS_MAP:
                continue
            labels.append([
                YOLO_CLASS_MAP[type_int],
                min(max(float(row[f'{_C_BOX}.box.center.x']) / w0, 0.0), 1.0),
                min(max(float(row[f'{_C_BOX}.box.center.y']) / h0, 0.0), 1.0),
                min(float(row[f'{_C_BOX}.box.size.x']) / w0, 1.0),
                min(float(row[f'{_C_BOX}.box.size.y']) / h0, 1.0),
            ])
        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        return image, label_tensor

    def __getitem__(self, i: int):
        image, labels = self._load_image_and_labels(i)
        image = cv2.resize(image, (self.imgsz, self.imgsz))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, labels


class WaymoCameraLiDARFusionDataset(WaymoGCSDataset):
    """Camera samples augmented with a calibrated sparse LiDAR depth raster."""

    def __init__(self, toolkit: ToolKit, imgsz: int = 512, cameras: tuple = (1,),
                 top_lidar_only: bool = True, max_depth: float = 75.0,
                 max_frames: int | None = None, frame_stride: int = 1):
        super().__init__(toolkit, imgsz=imgsz, cameras=cameras,
                         max_frames=max_frames, frame_stride=frame_stride)
        self.top_lidar_only = top_lidar_only
        self.max_depth = max_depth
        self.calibrations = {camera: toolkit.load_camera_calibration(camera) for camera in cameras}

    def __getitem__(self, i: int):
        image, labels = self._load_image_and_labels(i)
        timestamp, camera = self.index[i]
        point_sets = self.toolkit.load_lidar_points_xyzi(timestamp, top_only=self.top_lidar_only)
        if self.top_lidar_only and point_sets:
            point_sets = [max(point_sets, key=len)]
        points = np.concatenate(point_sets, axis=0).astype(np.float32) if point_sets else np.zeros((0, 4), np.float32)
        return build_fusion_tensor(image, points, self.calibrations[camera], self.imgsz, self.max_depth), labels


# ── LiDAR Dataset ────────────────────────────────────────────────────────────


class WaymoLiDARDataset(Dataset):
    """Streams LiDAR point clouds + 3-D box annotations from one GCS segment.

    No files are written to disk.  Each __getitem__ returns:
      - points:    (N, 4) float32  [x, y, z, intensity]  all lasers concatenated
      - gt_boxes:  (M, 7) float32  [cx, cy, cz, dx, dy, dz, heading]
      - gt_labels: (M,)   int32    0=vehicle, 1=pedestrian, 2=cyclist

    The collate_fn provided by PointPillarsDetector (or any registered LiDAR
    detector) accepts a list of these tuples and builds the pillars + GT tensors.

    Args:
        toolkit:  ToolKit instance with an active segment assigned.
    """

    def __init__(self, toolkit: ToolKit, top_lidar_only: bool = True,
                 max_frames: int | None = None, frame_stride: int = 1,
                 min_gt_points: int = 0):
        self.toolkit        = toolkit
        self.top_lidar_only = top_lidar_only
        self.min_gt_points  = min_gt_points   # boxes with fewer LiDAR points cannot be detected
        toolkit._read_cached('lidar_box')   # load in the parent so DataLoader workers never touch Dask
        self.timestamps     = stride_timestamps(toolkit.get_timestamps(), frame_stride)
        if max_frames is not None:
            self.timestamps = self.timestamps[:max_frames]
        print(f'   {len(self.timestamps)} LiDAR frames in segment'
              + ('  (TOP lidar only)' if top_lidar_only else ''))

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, i: int):
        ts = self.timestamps[i]

        # ── Points ──────────────────────────────────────────────────────────
        pts_list = self.toolkit.load_lidar_points_xyzi(ts, top_only=self.top_lidar_only)   # list[(N_l, 4)]
        if pts_list:
            if self.top_lidar_only:
                # TOP lidar (laser 1) always has far more returns than the 4
                # single-beam side lasers; keeping only the largest laser array
                # cuts per-frame memory ~4–5× with no meaningful accuracy cost.
                pts_list = [max(pts_list, key=len)]
            points = np.concatenate(pts_list, axis=0).astype(np.float32)
        else:
            points = np.zeros((0, 4), dtype=np.float32)

        # ── GT boxes ────────────────────────────────────────────────────────
        boxes_df = self.toolkit.load_lidar_boxes(ts)
        gt_boxes  = []
        gt_labels = []

        for _, row in boxes_df.iterrows():
            type_int = int(row[f'{_L_BOX}.type'])
            if type_int not in LIDAR_DET_CLASS_MAP:
                continue
            points_col = 'num_top_lidar_points_in_box' if self.top_lidar_only else 'num_lidar_points_in_box'
            if int(row[f'{_L_BOX}.{points_col}']) < self.min_gt_points:
                continue
            cx  = float(row[f'{_L_BOX}.box.center.x'])
            cy  = float(row[f'{_L_BOX}.box.center.y'])
            cz  = float(row[f'{_L_BOX}.box.center.z'])
            dx  = float(row[f'{_L_BOX}.box.size.x'])     # length (forward extent)
            dy  = float(row[f'{_L_BOX}.box.size.y'])     # width  (lateral extent)
            dz  = float(row[f'{_L_BOX}.box.size.z'])
            hdg = float(row[f'{_L_BOX}.box.heading'])
            gt_boxes.append([cx, cy, cz, dx, dy, dz, hdg])
            gt_labels.append(LIDAR_DET_CLASS_MAP[type_int])

        gt_boxes  = (
            np.array(gt_boxes,  dtype=np.float32)
            if gt_boxes else np.zeros((0, 7), dtype=np.float32)
        )
        gt_labels = (
            np.array(gt_labels, dtype=np.int32)
            if gt_labels else np.zeros((0,), dtype=np.int32)
        )

        return points, gt_boxes, gt_labels


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(nn_model, optimizer, seg_num: int, path: Path):
    """Save model + optimizer state in toolkit checkpoint format."""
    torch.save(
        {
            'model':     nn_model,            # full nn.Module (not state_dict)
            'optimizer': optimizer.state_dict(),
            'seg':       seg_num,
        },
        path,
    )


# ── Progress tracker ──────────────────────────────────────────────────────────


class ProgressTracker:
    """JSON file recording trained and pending segment names.

    Stored on Drive so training resumes cleanly after a Colab disconnect.
    """

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            data = json.loads(path.read_text())
            self.trained = data.get('trained', [])
            self.pending = data.get('pending', [])
            self.pass_no = data.get('pass', 1)
            self.segments = data.get('segments', self.trained + self.pending)
        else:
            self.trained = []
            self.pending = []
            self.pass_no = 1
            self.segments = []

    def initialise(self, all_segments: list):
        self.segments = list(all_segments)
        done = set(self.trained)
        self.pending = [s for s in all_segments if s not in done]
        self._write()

    def next_pass(self, seed: int):
        """Start another pass over every segment, in a seeded shuffled order."""
        self.pass_no += 1
        order = list(self.segments)
        random.Random(seed + self.pass_no).shuffle(order)
        self.trained, self.pending = [], order
        self._write()

    def mark_done(self, seg: str):
        if seg in self.pending:
            self.pending.remove(seg)
        if seg not in self.trained:
            self.trained.append(seg)
        self._write()
        print(
            f'   Progress: {len(self.trained)} done, '
            f'{len(self.pending)} pending  [{self.path.name}]'
        )

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {'trained': self.trained, 'pending': self.pending,
                 'pass': self.pass_no, 'segments': self.segments},
                indent=2,
            )
        )


# ── Training ──────────────────────────────────────────────────────────────────


def train(cfg: dict, drive_dir: Path, resume: bool = False):
    train_cfg = cfg['train']
    device    = resolve_torch_device(train_cfg.get('device', 'auto'))
    set_global_seed(int(train_cfg.get('seed', 42)))
    ckpt_dir  = drive_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_file = drive_dir / 'progress.json'
    artifacts = RunArtifacts(drive_dir, cfg)
    latest_ckpt   = ckpt_dir  / 'latest.pt'

    # Resume is explicit so a new experiment cannot silently reuse old weights.
    if latest_ckpt.exists():
        if not resume:
            raise FileExistsError(
                f'{latest_ckpt} already exists. Use a new --drive-dir or pass --resume intentionally.'
            )
        cfg['resume_weights'] = str(latest_ckpt)
        print(f'Resuming from : {latest_ckpt}')

    detector = build_detector(cfg)
    nn_model = detector.build_model(cfg, device)
    nn_model.train()

    optimizer = torch.optim.AdamW(
        nn_model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )
    if resume and latest_ckpt.exists():
        saved = torch.load(latest_ckpt, map_location=device, weights_only=False)
        if isinstance(saved, dict) and saved.get('optimizer'):
            optimizer.load_state_dict(saved['optimizer'])
            print('Restored optimizer state')
        del saved

    task = cfg.get('task', 'camera_2d_detection')
    backbone_info = (f'backbone={cfg["model"]["backbone"]}'
                     if 'backbone' in cfg['model'] else
                     f'pfn_out={cfg["model"].get("pfn_out", 64)}')

    print(f'Device        : {device}')
    print(f'Config        : {cfg["name"]}')
    print(f'Task          : {task}')
    print(f'Detector      : {cfg["model"]["type"]}  ({backbone_info})')
    print(f'Epochs/seg    : {train_cfg["epochs_per_seg"]}')
    print(f'Batch size    : {train_cfg["batch_size"]}')
    print(f'LR            : {train_cfg["lr"]}')
    print(f'Run dir       : {drive_dir}')
    print(f'Seed          : {train_cfg.get("seed", 42)}')
    artifacts.record('run_started', task=task, device=str(device), seed=int(train_cfg.get('seed', 42)))

    # ── Progress ──────────────────────────────────────────────────────────────
    tracker = ProgressTracker(progress_file)
    toolkit = ToolKit(split=cfg['data']['split'])

    passes = int(train_cfg.get('passes', 1))
    if not tracker.segments:
        manifest_path = cfg['data'].get('manifest')
        if manifest_path:
            print(f'\nUsing frozen training segments from {manifest_path} ...')
            all_segs = load_split_manifest(manifest_path)['train']
        else:
            print('\nFetching segment list from GCS ...')
            all_segs = toolkit.list_segments()
        total_segs = train_cfg.get('total_segs')
        tracker.initialise(
            all_segs if total_segs is None else all_segs[:total_segs]
        )
        print(f'{len(tracker.pending)} segments queued.')

    print(f'\n{len(tracker.trained)} done, {len(tracker.pending)} remaining.\n')

    # ── Segment loop ──────────────────────────────────────────────────────────
    # Dataset class is selected by cfg['task']; collate_fn always from detector.
    cameras  = tuple(cfg['data'].get('cameras', [1, 2, 3, 4, 5]))
    img_size = cfg['data'].get('img_size', 640)
    stride   = cfg['data'].get('frame_stride', 1)

    global_epoch = ((tracker.pass_no - 1) * len(tracker.segments)
                    + len(tracker.trained)) * train_cfg['epochs_per_seg']
    while tracker.pending or tracker.pass_no < passes:
        if not tracker.pending:
            tracker.next_pass(int(train_cfg.get('seed', 42)))
            print(f'\n=== Pass {tracker.pass_no}/{passes} ===')
            continue
        seg = tracker.pending[0]
        seg_num = len(tracker.trained) + 1
        unit    = (tracker.pass_no - 1) * len(tracker.segments) + seg_num
        total   = len(tracker.trained) + len(tracker.pending)
        print(f'\n── Pass {tracker.pass_no}/{passes}  Segment [{seg_num}/{total}] ─────────────')
        if train_cfg.get('lr_schedule') == 'cosine':
            # Piecewise-constant per segment visit, so it resumes exactly from `unit`.
            warmup = int(train_cfg.get('warmup_units', 0))
            if warmup and unit <= warmup:
                factor = unit / warmup
            else:
                factor = 0.5 * (1 + math.cos(math.pi * (unit - 1) / max(passes * len(tracker.segments), 1)))
            for group in optimizer.param_groups:
                group['lr'] = train_cfg['lr'] * max(factor, 0.01)
        print(f'   {seg[:72]}')

        toolkit.assign_segment(seg)
        if task == 'camera_lidar_fusion_2d_detection':
            dataset = WaymoCameraLiDARFusionDataset(
                toolkit, imgsz=img_size, cameras=cameras,
                top_lidar_only=cfg['data'].get('top_lidar_only', True),
                max_depth=cfg['data'].get('fusion_max_depth', 75.0),
                max_frames=cfg['data'].get('max_frames_per_seg'),
                frame_stride=stride,
            )
        elif task == 'lidar_3d_detection':
            dataset = WaymoLiDARDataset(
                toolkit,
                top_lidar_only=cfg['data'].get('top_lidar_only', True),
                max_frames=cfg['data'].get('max_frames_per_seg'),
                frame_stride=stride,
                min_gt_points=int(cfg['data'].get('min_gt_points', 0)),
            )
        else:
            dataset = WaymoGCSDataset(
                toolkit, imgsz=img_size, cameras=cameras,
                max_frames=cfg['data'].get('max_frames_per_seg'),
                frame_stride=stride,
            )
        loader  = DataLoader(
            dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            collate_fn=detector.collate_fn,
            num_workers=int(cfg['data'].get('num_workers', 0)),
            pin_memory=(device.type == 'cuda'),
        )
        print(f'   {len(dataset)} samples  ({len(loader)} batches)')

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(1, train_cfg['epochs_per_seg'] + 1):
            global_epoch += 1
            total_loss = 0.0
            skipped    = 0

            for batch in loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss = detector.loss(nn_model, batch)
                if loss.grad_fn is None:
                    skipped += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    nn_model.parameters(), train_cfg['grad_clip']
                )
                optimizer.step()
                total_loss += loss.item()

            trained_batches = len(loader) - skipped
            if trained_batches == 0:
                raise RuntimeError(
                    "No gradient-bearing batches were produced. Training was stopped before "
                    "recording a misleading checkpoint; inspect the model/loss configuration."
                )
            avg = total_loss / trained_batches
            print(
                f'   epoch {epoch}/{train_cfg["epochs_per_seg"]}  '
                f'loss={avg:.4f}  '
                f'({trained_batches}/{len(loader)} batches had labels)'
            )
            artifacts.record(
                'train_epoch', global_epoch=global_epoch, segment=seg,
                segment_number=seg_num, epoch=epoch, loss=avg,
                trained_batches=trained_batches, total_batches=len(loader),
            )

        tracker.mark_done(seg)

        # ── Free segment memory ───────────────────────────────────────────────
        # The toolkit caches the full lidar Parquet (~1 GB) for the segment.
        # Explicitly releasing dataset + loader allows Python to reclaim that
        # memory before the next segment is assigned and its Parquet is loaded.
        del dataset, loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        # ── Save checkpoint ───────────────────────────────────────────────────
        if unit % train_cfg['save_every'] == 0 or not tracker.pending:
            ckpt_path = ckpt_dir / f'seg_{unit:04d}.pt'
            save_checkpoint(nn_model, optimizer, unit, ckpt_path)
            save_checkpoint(nn_model, optimizer, unit, latest_ckpt)
            print(f'   Checkpoint → {ckpt_path}')

    print(f'\nTraining complete.  Latest weights: {latest_ckpt}')


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train a Waymo object detector from a YAML config file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--config',
        required=True,
        metavar='YAML',
        help='Path to the detector config YAML (e.g. configs/yolov8n.yaml).',
    )
    p.add_argument(
        '--drive-dir',
        default=None,
        metavar='DIR',
        help='Directory for checkpoints + progress.json. Defaults to runs/<config name>.',
    )
    # ── Optional per-run overrides (take precedence over YAML) ───────────────
    p.add_argument(
        '--total-segs',
        type=int,
        default=None,
        metavar='N',
        help='Override train.total_segs from the YAML.',
    )
    p.add_argument(
        '--batch',
        type=int,
        default=None,
        metavar='N',
        help='Override train.batch_size from the YAML.',
    )
    p.add_argument(
        '--max-frames',
        type=int,
        default=None,
        metavar='N',
        help='Limit timestamps per segment for any backend (useful for fast local tests).',
    )
    p.add_argument(
        '--epochs-per-seg',
        type=int,
        default=None,
        metavar='N',
        help='Override train.epochs_per_seg from the YAML.',
    )
    p.add_argument(
        '--passes',
        type=int,
        default=None,
        metavar='N',
        help='Override train.passes from the YAML (passes over all training segments).',
    )
    p.add_argument(
        '--frame-stride',
        type=int,
        default=None,
        metavar='N',
        help='Override data.frame_stride from the YAML (1 = every frame).',
    )
    p.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override train.lr from the YAML.',
    )
    p.add_argument(
        '--device',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default=None,
        help='Override train.device (default: auto selects CUDA, then MPS, then CPU).',
    )
    p.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoints/latest.pt in the selected run directory.',
    )
    p.add_argument(
        '--smoke-test',
        action='store_true',
        help='Run one segment, one epoch, the FRONT camera, and ten timestamps by default.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides — these take precedence over YAML values.
    if args.total_segs is not None:
        cfg['train']['total_segs'] = args.total_segs
    if args.batch is not None:
        cfg['train']['batch_size'] = args.batch
    if args.passes is not None:
        cfg['train']['passes'] = args.passes
    if args.epochs_per_seg is not None:
        cfg['train']['epochs_per_seg'] = args.epochs_per_seg
    if args.frame_stride is not None:
        cfg['data']['frame_stride'] = args.frame_stride
    if args.lr is not None:
        cfg['train']['lr'] = args.lr
    if args.device is not None:
        cfg['train']['device'] = args.device
    if args.smoke_test:
        cfg['name'] = f"{cfg['name']}-smoke"
        cfg['train'].update({'total_segs': 1, 'epochs_per_seg': 1, 'batch_size': min(cfg['train']['batch_size'], 2)})
        cfg['data'].update({'cameras': [1], 'max_frames_per_seg': 10})
    if args.max_frames is not None:
        cfg['data']['max_frames_per_seg'] = args.max_frames

    run_dir = Path(args.drive_dir) if args.drive_dir else Path('./runs') / cfg['name']
    train(cfg, run_dir, resume=args.resume)
