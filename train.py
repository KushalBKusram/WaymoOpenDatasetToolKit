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

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from models import build_detector
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
        toolkit:  ToolKit instance with an active segment assigned.
        imgsz:    Square image size to resize to.
        cameras:  Camera IDs to include.
    """

    def __init__(
        self,
        toolkit: ToolKit,
        imgsz: int = 640,
        cameras: tuple = (1, 2, 3, 4, 5),
    ):
        self.toolkit = toolkit
        self.imgsz   = imgsz

        cam_df       = toolkit._read_cached('camera_image')
        self.box_df  = toolkit._read_cached('camera_box')

        self.index = [
            (row['key.frame_timestamp_micros'], row['key.camera_name'])
            for _, row in cam_df.iterrows()
            if row['key.camera_name'] in cameras
        ]

        labeled = sum(
            1 for ts, cam in self.index
            if len(self.box_df[
                (self.box_df['key.frame_timestamp_micros'] == ts) &
                (self.box_df['key.camera_name'] == cam)
            ]) > 0
        )
        print(f'   label coverage: {labeled}/{len(self.index)} frames have boxes')

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ts, cam = self.index[i]

        img   = self.toolkit.load_camera_frame(ts, cam)   # H×W×3 BGR
        h0, w0 = img.shape[:2]

        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes_df = self.box_df[
            (self.box_df['key.frame_timestamp_micros'] == ts) &
            (self.box_df['key.camera_name'] == cam)
        ]
        labels = []
        for _, row in boxes_df.iterrows():
            type_int = int(row[f'{_C_BOX}.type'])
            if type_int not in YOLO_CLASS_MAP:
                continue
            cx = float(row[f'{_C_BOX}.box.center.x']) / w0
            cy = float(row[f'{_C_BOX}.box.center.y']) / h0
            bw = float(row[f'{_C_BOX}.box.size.x'])   / w0
            bh = float(row[f'{_C_BOX}.box.size.y'])   / h0
            labels.append([
                YOLO_CLASS_MAP[type_int],
                min(max(cx, 0.0), 1.0),
                min(max(cy, 0.0), 1.0),
                min(bw, 1.0),
                min(bh, 1.0),
            ])

        labels = (
            torch.tensor(labels, dtype=torch.float32)
            if labels
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        return img, labels


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

    def __init__(self, toolkit: ToolKit):
        self.toolkit = toolkit
        self.timestamps = toolkit.get_timestamps()
        print(f'   {len(self.timestamps)} LiDAR frames in segment')

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, i: int):
        ts = self.timestamps[i]

        # ── Points: concatenate all lasers into (N, 4) ──────────────────────
        pts_list = self.toolkit.load_lidar_points_xyzi(ts)   # list[(N_l, 4)]
        if pts_list:
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
        else:
            self.trained = []
            self.pending = []

    def initialise(self, all_segments: list):
        done = set(self.trained)
        self.pending = [s for s in all_segments if s not in done]
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
                {'trained': self.trained, 'pending': self.pending},
                indent=2,
            )
        )


# ── Training ──────────────────────────────────────────────────────────────────


def train(cfg: dict, drive_dir: Path):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir  = drive_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_file = drive_dir / 'progress.json'
    latest_ckpt   = ckpt_dir  / 'latest.pt'

    # Pass latest checkpoint path into cfg so detector.build_model can find it
    if latest_ckpt.exists():
        cfg['resume_weights'] = str(latest_ckpt)
        print(f'Resuming from : {latest_ckpt}')

    detector = build_detector(cfg)
    nn_model = detector.build_model(cfg, device)
    nn_model.train()

    train_cfg = cfg['train']
    optimizer = torch.optim.AdamW(
        nn_model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['weight_decay'],
    )

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
    print(f'Drive dir     : {drive_dir}')

    # ── Progress ──────────────────────────────────────────────────────────────
    tracker = ProgressTracker(progress_file)
    toolkit = ToolKit(split=cfg['data']['split'])

    if not tracker.pending:
        print('\nFetching segment list from GCS ...')
        all_segs   = toolkit.list_segments()
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

    for seg in list(tracker.pending):
        seg_num = len(tracker.trained) + 1
        total   = len(tracker.trained) + len(tracker.pending)
        print(f'\n── Segment [{seg_num}/{total}] ──────────────────────────────')
        print(f'   {seg[:72]}')

        toolkit.assign_segment(seg)
        if task == 'lidar_3d_detection':
            dataset = WaymoLiDARDataset(toolkit)
        else:
            dataset = WaymoGCSDataset(toolkit, imgsz=img_size, cameras=cameras)
        loader  = DataLoader(
            dataset,
            batch_size=train_cfg['batch_size'],
            shuffle=True,
            collate_fn=detector.collate_fn,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
        )
        print(f'   {len(dataset)} samples  ({len(loader)} batches)')

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(1, train_cfg['epochs_per_seg'] + 1):
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
            avg = total_loss / max(trained_batches, 1)
            print(
                f'   epoch {epoch}/{train_cfg["epochs_per_seg"]}  '
                f'loss={avg:.4f}  '
                f'({trained_batches}/{len(loader)} batches had labels)'
            )

        tracker.mark_done(seg)

        # ── Save checkpoint ───────────────────────────────────────────────────
        if seg_num % train_cfg['save_every'] == 0 or not tracker.pending:
            ckpt_path = ckpt_dir / f'seg_{seg_num:04d}.pt'
            save_checkpoint(nn_model, optimizer, seg_num, ckpt_path)
            save_checkpoint(nn_model, optimizer, seg_num, latest_ckpt)
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
        default='./runs/waymo',
        metavar='DIR',
        help='Directory for checkpoints + progress.json (default: ./runs/waymo).',
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
        '--lr',
        type=float,
        default=None,
        help='Override train.lr from the YAML.',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides — these take precedence over YAML values
    if args.total_segs is not None:
        cfg['train']['total_segs'] = args.total_segs
    if args.batch is not None:
        cfg['train']['batch_size'] = args.batch
    if args.lr is not None:
        cfg['train']['lr'] = args.lr

    train(cfg, Path(args.drive_dir))
