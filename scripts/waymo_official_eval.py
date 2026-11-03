"""Score LiDAR detections with the official Waymo Open Dataset 3-D detection metric.

Needs Linux x86-64 and TensorFlow (the official package ships a compiled TF op, so it
cannot run on macOS). Produce the input on any machine with:

  python evaluate.py --config configs/centerpoint.yaml --run-dir runs/centerpoint-v1.2-s42 \
      --dump-detections runs/centerpoint-v1.2-s42/detections.npz

then, in an environment with `waymo-open-dataset-tf-2-12-0` installed:

  python scripts/waymo_official_eval.py runs/centerpoint-v1.2-s42/detections.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Toolkit class id -> Waymo label type (1 vehicle, 2 pedestrian, 4 cyclist).
WAYMO_TYPE = np.array([1, 2, 4], np.uint8)   # the TF op takes uint8 types and difficulties
MIN_L1_POINTS = 5


def load_dump(path: Path):
    data = np.load(path)
    gt, pred = data["gt"], data["pred"]
    difficulty = gt[:, 8].astype(np.uint8)
    unset = difficulty == 0     # v2 validation boxes carry no difficulty; derive it from point counts
    difficulty[unset] = np.where(gt[unset, 9] >= MIN_L1_POINTS, 1, 2)
    return {
        "ground_truth_frame_id": data["gt_frame"],
        "ground_truth_bbox": gt[:, :7].astype(np.float32),
        "ground_truth_type": WAYMO_TYPE[gt[:, 7].astype(int)],
        "ground_truth_difficulty": difficulty,
    }, {
        "prediction_frame_id": data["pred_frame"],
        "prediction_bbox": pred[:, :7].astype(np.float32),
        "prediction_type": WAYMO_TYPE[pred[:, 7].astype(int)],
        "prediction_score": pred[:, 8].astype(np.float32),
        "prediction_overlap_nlz": np.zeros(len(pred), bool),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("detections", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Default: <input name>_official_metrics.json beside the input.")
    args = parser.parse_args()

    import tensorflow as tf
    from waymo_open_dataset.metrics.python import wod_detection_evaluator

    ground_truth, predictions = load_dump(args.detections)
    ground_truth = {k: tf.convert_to_tensor(v) for k, v in ground_truth.items()}
    predictions = {k: tf.convert_to_tensor(v) for k, v in predictions.items()}
    evaluator = wod_detection_evaluator.WODDetectionEvaluator()
    evaluator.update_state(ground_truth, predictions)
    result = evaluator.evaluate()

    from waymo_open_dataset.protos import breakdown_pb2

    generator_names = breakdown_pb2.Breakdown.GeneratorId.Name
    object_types = ("VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST")   # OBJECT_TYPE shards are 0-based indices
    level_names = {1: "LEVEL_1", 2: "LEVEL_2"}
    rows = []
    for index, (generator, shard, difficulty) in enumerate(result.breakdown.numpy()):
        generator_name = generator_names(int(generator))
        rows.append({
            "breakdown": generator_name,
            "shard": object_types[int(shard)] if generator_name == "OBJECT_TYPE" else int(shard),
            "difficulty": level_names.get(int(difficulty), int(difficulty)),
            "ap": float(result.average_precision[index]),
            "apH": float(result.average_precision_ha_weighted[index]),
        })
    out = args.out or args.detections.with_name(args.detections.stem + "_official_metrics.json")
    out.write_text(json.dumps({"source": str(args.detections), "rows": rows}, indent=2) + "\n")
    for row in rows:
        if row["breakdown"] == "OBJECT_TYPE":
            print(f'{row["shard"]:<16}{row["difficulty"]:<9} AP {row["ap"]:.4f}   APH {row["apH"]:.4f}')


if __name__ == "__main__":
    main()
