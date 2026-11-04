"""Export a toolkit training checkpoint as a weights-only file for distribution.

Training checkpoints (`checkpoints/latest.pt`) pickle the whole model object and the
optimizer state, so loading one runs arbitrary pickle code. The exported file holds only
tensors and plain metadata and loads with `torch.load(path, weights_only=True)`.

  python scripts/export_weights.py --config configs/yolov8n_hires_100.yaml \
      --run-dir runs/yolov8n_hires_100-v1.2-s42 --out yolov8n_hires_100-v1.2-s42.weights.pt

Point `resume_weights` (or `evaluate.py`'s checkpoint) at the exported file; the YOLO
detectors rebuild the architecture from the config and load the tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

FORMAT = "waymo-toolkit-weights-v1"


def export(checkpoint: Path, cfg: dict, out: Path, source_run: str = "", seed: int | None = None,
           trained_with: str = "") -> dict:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)   # trusted local training checkpoint
    model = saved["model"]
    state = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
    payload = {
        "format": FORMAT,
        "state_dict": state,
        "model": {key: cfg["model"][key] for key in ("type", "backbone", "num_classes", "class_names") if key in cfg["model"]},
        "input_channels": int(model.model[0].conv.weight.shape[1]),
        "img_size": cfg["data"].get("img_size"),
        # Layer activations differ between Ultralytics versions (for example SPPF.cv1), so record them.
        "activations": {name: type(m.act).__name__ for name, m in model.named_modules()
                        if isinstance(getattr(m, "act", None), torch.nn.Module)},
        "trained_with_ultralytics": trained_with,
        "source_run": source_run,
        "seed": seed,
        "segments_trained": int(saved.get("seg", -1)),
    }
    torch.save(payload, out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trained-with", default="", help="Ultralytics version the checkpoint was trained with.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(args.run_dir)
    payload = export(run_dir / "checkpoints" / "latest.pt", cfg, args.out, run_dir.name, cfg.get("train", {}).get("seed"), args.trained_with)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    check = torch.load(args.out, map_location="cpu", weights_only=True)    # must load without unpickling code
    print(json.dumps({"out": str(args.out), "bytes": args.out.stat().st_size, "sha256": digest,
                      "tensors": len(check["state_dict"]), "format": check["format"],
                      "model": payload["model"], "img_size": payload["img_size"]}, indent=2))


if __name__ == "__main__":
    main()
