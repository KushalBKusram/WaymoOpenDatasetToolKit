"""
train.py — Train YOLOv8 on Waymo data streamed directly from GCS.

Images are read from GCS Parquet files into memory — no disk export required.
A JSON progress file tracks which segments are done so training resumes
cleanly after a Colab disconnect. Checkpoints are saved to Google Drive.

Usage examples:

  # Train on all segments, save artefacts to Drive
  python train.py --drive-dir /content/drive/MyDrive/waymo

  # Larger model, more epochs per segment
  python train.py --drive-dir /content/drive/MyDrive/waymo --model s --epochs-per-seg 3

  # Resume (auto-detects latest.pt and progress.json on Drive)
  python train.py --drive-dir /content/drive/MyDrive/waymo

  # Custom weights to start from
  python train.py --drive-dir /content/drive/MyDrive/waymo \\
                  --weights /content/drive/MyDrive/waymo/checkpoints/latest.pt

Prerequisite:
  pip install ultralytics
  # On Colab:
  #   from google.colab import auth; auth.authenticate_user()
  #   from google.colab import drive; drive.mount('/content/drive')
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules.waymo_open_dataset import (
    YOLO_CLASS_MAP, YOLO_CLASS_NAMES, ToolKit, _C_BOX,
)


# ── Dataset ───────────────────────────────────────────────────────────────


class WaymoGCSDataset(Dataset):
    """Streams camera images + 2-D box labels from one GCS Parquet segment.

    No files are written to disk. Each __getitem__ decodes a JPEG stored
    as bytes in the Parquet column directly into a float32 tensor.

    Args:
        toolkit:  Initialised ToolKit instance with a segment assigned.
        imgsz:    Square image size to resize to (default: 640).
        cameras:  Camera IDs to include (default: all 5).
    """

    def __init__(
        self,
        toolkit: ToolKit,
        imgsz: int = 640,
        cameras: tuple = (1, 2, 3, 4, 5),
    ):
        self.toolkit = toolkit
        self.imgsz = imgsz

        # Pre-cache both components up front to avoid per-item GCS round trips
        cam_df = toolkit._read_cached('camera_image')
        self.box_df = toolkit._read_cached('camera_box')

        self.index = [
            (row['key.frame_timestamp_micros'], row['key.camera_name'])
            for _, row in cam_df.iterrows()
            if row['key.camera_name'] in cameras
        ]

        # Diagnostic: report label coverage for this segment
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

        img = self.toolkit.load_camera_frame(ts, cam)   # H x W x 3 BGR
        h0, w0 = img.shape[:2]

        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Use pre-cached box_df instead of calling load_camera_boxes()
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
            bw = float(row[f'{_C_BOX}.box.size.x']) / w0
            bh = float(row[f'{_C_BOX}.box.size.y']) / h0
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


def collate_fn(batch):
    """Collate into the dict format expected by v8DetectionLoss."""
    imgs, labels_list = zip(*batch)
    imgs = torch.stack(imgs)   # (B, 3, H, W)

    cls_all, bboxes_all, bidx_all = [], [], []
    for i, lbl in enumerate(labels_list):
        if len(lbl):
            cls_all.append(lbl[:, 0])
            bboxes_all.append(lbl[:, 1:])
            bidx_all.append(torch.full((len(lbl),), float(i)))

    return {
        'img':       imgs,
        'cls':       torch.cat(cls_all)    if cls_all    else torch.zeros(0),
        'bboxes':    torch.cat(bboxes_all) if bboxes_all else torch.zeros((0, 4)),
        'batch_idx': torch.cat(bidx_all)   if bidx_all   else torch.zeros(0),
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────


def save_checkpoint(nn_model, optimizer, seg_num, path):
    """Save in a format that can be reloaded with both YOLO() and torch.load."""
    torch.save(
        {
            'model': copy.deepcopy(nn_model),
            'optimizer': optimizer.state_dict(),
            'seg': seg_num,
        },
        path,
    )


def load_nn_model(weights, model_size, device, get_cfg, DEFAULT_CFG, YOLO):
    """Load model weights; handles both Ultralytics .pt and our own checkpoints."""
    path = Path(weights)
    if path.exists():
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        # Our saved checkpoints store the full nn.Module under 'model'
        if isinstance(ckpt, dict) and isinstance(
            ckpt.get('model'), torch.nn.Module
        ):
            nn_model = ckpt['model'].to(device)
            print(f'Resumed from segment {ckpt.get("seg", "?")}')
            return nn_model
    # Fall back to Ultralytics loader (pretrained yolov8n.pt etc.)
    yolo = YOLO(weights)
    return yolo.model.to(device)


# ── Progress tracker ──────────────────────────────────────────────────────


class ProgressTracker:
    """JSON file that records trained and pending segment names.

    Stored on Drive so it survives Colab session restarts.
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
        """Populate pending from all_segments, skipping already trained."""
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
            f'Progress: {len(self.trained)} done, '
            f'{len(self.pending)} pending  [{self.path}]'
        )

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {'trained': self.trained, 'pending': self.pending},
                indent=2,
            )
        )


# ── Training ──────────────────────────────────────────────────────────────


def train(args):
    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        from ultralytics.utils.loss import v8DetectionLoss
    except ImportError:
        print(
            'Error: ultralytics not installed. Run: pip install ultralytics',
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device        : {device}')

    drive_dir = Path(args.drive_dir)
    ckpt_dir = drive_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    progress_file = drive_dir / 'progress.json'
    latest_ckpt = ckpt_dir / 'latest.pt'

    # ── Load model ────────────────────────────────────────────────────────
    if args.weights:
        weights = args.weights
    elif latest_ckpt.exists():
        weights = str(latest_ckpt)
        print(f'Resuming from : {weights}')
    else:
        weights = f'yolov8{args.model}.pt'

    nn_model = load_nn_model(
        weights, args.model, device, get_cfg, DEFAULT_CFG, YOLO
    )
    nn_model.train()

    # v8DetectionLoss reads model.args as an IterableSimpleNamespace.
    # When loaded outside the Ultralytics training loop, args may be a plain
    # dict — replace it with the default cfg namespace to fix attribute access.
    nn_model.args = get_cfg(DEFAULT_CFG)

    loss_fn = v8DetectionLoss(nn_model)
    optimizer = torch.optim.AdamW(
        nn_model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    print(f'Model         : {weights}')
    print(f'Epochs/seg    : {args.epochs_per_seg}')
    print(f'Batch         : {args.batch}')
    print(f'Save every    : {args.save_every} segments')
    print(f'Drive dir     : {drive_dir}')

    # ── Progress ──────────────────────────────────────────────────────────
    tracker = ProgressTracker(progress_file)
    toolkit = ToolKit(split='training')

    if not tracker.pending:
        print('\nFetching segment list from GCS ...')
        all_segs = toolkit.list_segments()
        tracker.initialise(
            all_segs if args.total_segs is None else all_segs[:args.total_segs]
        )
        print(f'{len(tracker.pending)} segments queued '
              f'({"all" if args.total_segs is None else args.total_segs} requested).')

    print(
        f'\n{len(tracker.trained)} done, '
        f'{len(tracker.pending)} remaining.\n'
    )

    # ── Segment loop ──────────────────────────────────────────────────────
    for seg in list(tracker.pending):
        total = len(tracker.trained) + len(tracker.pending)
        seg_num = len(tracker.trained) + 1
        print(f'\n── Segment [{seg_num}/{total}] ──────────────────────────')
        print(f'   {seg[:72]}')

        toolkit.assign_segment(seg)
        dataset = WaymoGCSDataset(toolkit, imgsz=args.img_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
        )
        print(f'   {len(dataset)} samples  ({len(loader)} batches)')

        # ── Epoch loop ────────────────────────────────────────────────────
        for epoch in range(1, args.epochs_per_seg + 1):
            total_loss = 0.0
            skipped = 0
            for batch in loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                preds = nn_model(batch['img'])
                loss, _ = loss_fn(preds, batch)
                if loss.grad_fn is None:
                    skipped += 1
                    continue   # no positive targets in batch; skip update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(nn_model.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item()

            trained_batches = len(loader) - skipped
            avg = total_loss / max(trained_batches, 1)
            print(
                f'   epoch {epoch}/{args.epochs_per_seg}  '
                f'loss={avg:.4f}  '
                f'({trained_batches}/{len(loader)} batches had labels)'
            )

        tracker.mark_done(seg)

        # ── Save checkpoint ───────────────────────────────────────────────
        if seg_num % args.save_every == 0 or not tracker.pending:
            ckpt_path = ckpt_dir / f'seg_{seg_num:04d}.pt'
            save_checkpoint(nn_model, optimizer, seg_num, ckpt_path)
            save_checkpoint(nn_model, optimizer, seg_num, latest_ckpt)
            print(f'   Checkpoint → {ckpt_path}')

    print(f'\nTraining complete.  Latest weights: {latest_ckpt}')


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train YOLOv8 on Waymo data streamed from GCS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--drive-dir',
        default='./runs/waymo',
        metavar='DIR',
        help='Drive directory for checkpoints + progress.json '
             '(default: ./runs/waymo)',
    )
    p.add_argument(
        '--model',
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size to start from (default: n)',
    )
    p.add_argument(
        '--weights',
        default=None,
        metavar='PATH',
        help='Explicit checkpoint to load (overrides auto-resume)',
    )
    p.add_argument(
        '--total-segs',
        type=int,
        default=None,
        metavar='N',
        help='Number of training segments to use (default: all)',
    )
    p.add_argument(
        '--epochs-per-seg',
        type=int,
        default=2,
        metavar='N',
        help='Epochs to train on each segment (default: 2)',
    )
    p.add_argument(
        '--img-size',
        type=int,
        default=640,
        help='Input image size (default: 640)',
    )
    p.add_argument(
        '--batch',
        type=int,
        default=8,
        help='Batch size (default: 8)',
    )
    p.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)',
    )
    p.add_argument(
        '--save-every',
        type=int,
        default=5,
        metavar='N',
        help='Save checkpoint every N segments (default: 5)',
    )
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
