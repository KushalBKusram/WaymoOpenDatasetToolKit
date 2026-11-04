# Reproducibility audit

Audited 2026-11-03 from a fresh `git clone` of commit `f02e588` into an empty directory, with a new
virtual environment and no local `.waymo_cache/`. Machine: Apple Silicon Mac, Python 3.12.13, device `mps`
(the reference runs used CUDA on an RTX 4080 SUPER). Google Cloud ADC was set up beforehand with
`gcloud auth application-default login`.

## What was reproduced

| Step | Command | Result |
|---|---|---|
| Install | `pip install -r requirements.txt` | Works, 38 s with a warm pip cache. Resolved versions are in `docs/requirements-tested.txt`. |
| Smoke training | `python train.py --config configs/yolov8n.yaml --smoke-test --max-frames 5 --drive-dir ./runs/smoke` | Completes (about 6 min, almost all of it first-time data download). Pretrained weights are fetched automatically. |
| Evaluation of a trained model | `python evaluate.py --config configs/yolov8n_hires_100.yaml --run-dir runs/yolov8n_hires_100-v1.2-s42` | mAP 0.15669 vs 0.15661 recorded (CUDA); AP50 0.2564 vs 0.2563; AP75 0.1640 vs 0.1640; 1,967 frames. |
| Tracking | `python scripts/track_eval.py --config configs/yolov8n_hires_100.yaml --run-dir runs/yolov8n_hires_100-v1.2-s42 --score-thresh 0.25 --iou-threshold 0.2 --max-age 20 --min-hits 3` | MOTA 0.0537, IDF1 0.3265, 650 vehicle ID switches: identical to the recorded run. |

The evaluation used the original checkpoint copied into the clean clone, so it checks the evaluation and
tracking code and data path, not training. The 0.0001 mAP difference is the CUDA versus MPS numeric
difference.

## What was not reproduced

- **Training from scratch.** Checkpoints are not in the repository (`runs/` and `*.pt` are git-ignored,
  and none are published). A full v1.3 run is 3 passes over 100 segments; on the RTX 4080 SUPER training
  took about 20 minutes, and the first run is dominated by downloading about 27 GB of camera data.
  Training is seeded (seed 42) but GPU nondeterminism means a retrained model will not match to
  the last digit; the protocol treats differences below about 0.01 mAP as noise.
- **LiDAR training and the official Waymo evaluator.** The official evaluator needs Linux x86-64 with
  TensorFlow 2.13 and was not run here. LiDAR training needs a decoded-LiDAR cache and was not re-run.
- **RT-DETR-L and the fusion models.** Not re-run.

## Gaps found, and what to know

1. **Requirements are unpinned.** A clean install today resolves to numpy 2.5.4, torch 2.15.0 and
   ultralytics 8.4.201 (`docs/requirements-tested.txt`). The pipeline works with these, but future releases
   may not (the tracker already avoids `motmetrics` internals that break on numpy 2). Use
   `pip install -r docs/requirements-tested.txt` to get the audited versions (it installs cleanly in a fresh
   Python 3.12 environment and the packages import; the full pipeline was run on the unpinned install above).
2. **README recommends Python 3.11, but 3.12 was tested.** Both should work; only 3.12 and 3.10 (the
   official evaluator) have been exercised recently.
3. **The model architecture can depend on the Ultralytics version.** The training machine reported
   Ultralytics 8.4.198 before the reference runs. A fresh YOLOv8n build in Ultralytics 8.4.115 does not match
   the trained models: the first conv of its SPPF block has no activation (the trained model has SiLU) and
   its classification head is narrower (64 against 80 hidden channels). The weights therefore cannot be loaded
   into a fresh 8.4.115 build without adjustment. Loading a pickled training checkpoint is unaffected, because
   the saved model object carries its own structure. Exported weights (`scripts/export_weights.py`) record each
   layer's activation and the loader sizes the head from the saved tensors; the released weights load with
   an exact match on 8.4.115 (the version installed here) and produce the reference mAP. I did not build the
   model under 8.4.198 directly, so which version change caused the difference is unconfirmed. To retrain,
   use the Ultralytics version of the original runs, since a different version may change the starting
   architecture and the result.
4. **Credentials are required.** Every step above needs read access to the public Waymo bucket through
   ADC; there is no offline path without downloading the segments first.
5. **The protocol did not say how to re-evaluate a reference run.** The commands in the table above now
   serve that purpose, and the protocol's section 8 links here. Tracking commands are also in
   `docs/tracking_baseline.md`.
