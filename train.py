"""
train.py — Export Waymo YOLO dataset and train a YOLOv8 model.

Usage examples:

  # Export 5 train + 2 val segments, then train yolov8n for 50 epochs
  python train.py --export

  # Skip export (dataset already on disk), train with a larger model
  python train.py --model m --epochs 100

  # Custom dataset and output directories
  python train.py --export --yolo-dir /content/yolo_dataset \\
                  --project-dir /content/drive/MyDrive/waymo/runs

  # Export only, no training
  python train.py --export --no-train

Prerequisite:
  pip install ultralytics
  gcloud auth application-default login   # or google.colab.auth on Colab
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Waymo YOLOv8 training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Export flags ──────────────────────────────────────────────────────
    p.add_argument(
        '--export',
        action='store_true',
        help='Stream segments from GCS and export YOLO dataset to --yolo-dir',
    )
    p.add_argument(
        '--yolo-dir',
        default='./yolo_dataset',
        metavar='DIR',
        help='Root of the YOLO dataset directory (default: ./yolo_dataset)',
    )
    p.add_argument(
        '--train-segs',
        type=int,
        default=5,
        metavar='N',
        help='Number of training segments to export (default: 5)',
    )
    p.add_argument(
        '--val-segs',
        type=int,
        default=2,
        metavar='N',
        help='Number of validation segments to export (default: 2)',
    )

    # ── Training flags ────────────────────────────────────────────────────
    p.add_argument(
        '--no-train',
        action='store_true',
        help='Skip training (export only)',
    )
    p.add_argument(
        '--model',
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv8 model size (default: n)',
    )
    p.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)',
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
        default=16,
        help='Batch size (default: 16)',
    )
    p.add_argument(
        '--project-dir',
        default='./runs/waymo',
        metavar='DIR',
        help='Directory for training artefacts (default: ./runs/waymo)',
    )
    p.add_argument(
        '--run-name',
        default=None,
        metavar='NAME',
        help='Training run name (default: yolov8<model>_waymo)',
    )
    p.add_argument(
        '--weights',
        default=None,
        metavar='PATH',
        help='Resume from checkpoint (default: pretrained yolov8<model>.pt)',
    )

    return p.parse_args()


def export(args):
    from modules.waymo_open_dataset import ToolKit

    yolo_dir = args.yolo_dir

    print(f'Exporting {args.train_segs} training segment(s) → {yolo_dir}')
    toolkit = ToolKit(split='training')
    segments = toolkit.list_segments()
    for i, seg in enumerate(segments[:args.train_segs]):
        print(f'  [{i+1}/{args.train_segs}] {seg[:60]}', end=' ... ', flush=True)
        toolkit.assign_segment(seg)
        toolkit.export_yolo(output_dir=yolo_dir, yolo_split='train')
        print('done')

    print(f'\nExporting {args.val_segs} validation segment(s) → {yolo_dir}')
    toolkit_val = ToolKit(split='validation')
    val_segments = toolkit_val.list_segments()
    for i, seg in enumerate(val_segments[:args.val_segs]):
        print(f'  [{i+1}/{args.val_segs}] {seg[:60]}', end=' ... ', flush=True)
        toolkit_val.assign_segment(seg)
        toolkit_val.export_yolo(output_dir=yolo_dir, yolo_split='val')
        print('done')

    print('\nExport complete.')


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            'Error: ultralytics not installed. Run: pip install ultralytics',
            file=sys.stderr,
        )
        sys.exit(1)

    yaml_path = Path(args.yolo_dir) / 'dataset.yaml'
    if not yaml_path.exists():
        print(
            f'Error: dataset.yaml not found at {yaml_path}. '
            'Run with --export first.',
            file=sys.stderr,
        )
        sys.exit(1)

    weights = args.weights or f'yolov8{args.model}.pt'
    run_name = args.run_name or f'yolov8{args.model}_waymo'

    print(f'\nTraining  : {weights}')
    print(f'Data      : {yaml_path}')
    print(f'Epochs    : {args.epochs}')
    print(f'Batch     : {args.batch}')
    print(f'Image size: {args.img_size}')
    print(f'Output    : {args.project_dir}/{run_name}')

    model = YOLO(weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        project=args.project_dir,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best = Path(args.project_dir) / run_name / 'weights' / 'best.pt'
    print(f'\nTraining complete. Best weights: {best}')


def main():
    args = parse_args()

    if not args.export and args.no_train:
        print('Nothing to do: pass --export, --no-train, or both.', file=sys.stderr)
        sys.exit(1)

    if args.export:
        export(args)

    if not args.no_train:
        train(args)


if __name__ == '__main__':
    main()
