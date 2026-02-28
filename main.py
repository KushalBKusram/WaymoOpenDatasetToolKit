"""
main.py — Waymo Open Dataset Toolkit v2 entry point.

Usage examples:

  # List the first 10 segments in the training split
  python main.py --list

  # Extract camera images + LiDAR labels for one segment
  python main.py --segment 10023947602400723454_1120_000_1140_000

  # Extract everything (camera, lidar labels, lidar point clouds)
  python main.py --segment 10023947602400723454_1120_000_1140_000 --all

  # Use validation split and a custom output directory
  python main.py --split validation --save-dir /tmp/waymo_out \\
                 --segment 10023947602400723454_1120_000_1140_000

Prerequisite:
  gcloud auth application-default login
"""

import argparse
import sys

from modules.WaymoOpenDataset import ToolKit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Waymo Open Dataset v2 Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        '--split',
        default='training',
        choices=['training', 'validation', 'testing'],
        help='Dataset split to use (default: training)',
    )
    p.add_argument(
        '--save-dir',
        default='./output',
        metavar='DIR',
        help='Root directory for extracted files (default: ./output)',
    )
    p.add_argument(
        '--list',
        action='store_true',
        help='List available segment names and exit',
    )
    p.add_argument(
        '--segment',
        metavar='CONTEXT_NAME',
        help='Segment context name to process',
    )
    p.add_argument(
        '--all',
        action='store_true',
        help='Also extract LiDAR point clouds (slow; requires more memory)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    toolkit = ToolKit(split=args.split, save_dir=args.save_dir)

    # --list: print available segments and exit
    if args.list:
        print(f'Listing segments for split="{args.split}" ...')
        segments = toolkit.list_segments()
        print(f'Found {len(segments)} segment(s). First 10:')
        for seg in segments[:10]:
            print(f'  {seg}')
        return

    # --segment: extract data
    if not args.segment:
        print('Error: provide --segment <context_name> or --list.', file=sys.stderr)
        sys.exit(1)

    toolkit.assign_segment(args.segment)
    print(f'Segment : {args.segment}')
    print(f'Split   : {args.split}')
    print(f'Output  : {args.save_dir}')

    print('\n[1/3] Extracting camera images and 2-D labels ...')
    toolkit.extract_camera_images()
    print('      Done.')

    print('[2/3] Extracting LiDAR 3-D box labels ...')
    toolkit.extract_lidar_labels()
    print('      Done.')

    if args.all:
        print('[3/3] Extracting LiDAR point clouds (range image → xyz) ...')
        toolkit.extract_lidar_points()
        print('      Done.')
    else:
        print('[3/3] Skipped LiDAR point cloud extraction (pass --all to enable).')

    print('\nAll done. Output written to:', args.save_dir)


if __name__ == '__main__':
    main()
