"""Round trip for scripts/export_weights.py: exported weights rebuild an identical YOLO model."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from export_weights import FORMAT, export  # noqa: E402
from models.yolov8_detector import YOLOv8Detector  # noqa: E402

CFG = {
    "model": {"type": "YOLOv8Detector", "backbone": "n", "num_classes": 4,
              "class_names": ["vehicle", "pedestrian", "cyclist", "sign"]},
    "data": {"img_size": [64, 96]},
    "train": {"seed": 1},
}


def test_exported_weights_load_without_pickle_and_match(tmp_path):
    model = DetectionModel("yolov8n.yaml", ch=3, nc=4, verbose=False)
    for p in model.parameters():                       # make the weights non-default
        p.data.normal_(0, 0.1)
    checkpoint, out = tmp_path / "latest.pt", tmp_path / "weights.pt"
    torch.save({"model": model, "seg": 7}, checkpoint)

    payload = export(checkpoint, CFG, out, source_run="unit-test", seed=1, trained_with="x")
    assert payload["format"] == FORMAT and payload["img_size"] == [64, 96]

    loaded = torch.load(out, map_location="cpu", weights_only=True)      # no pickled code needed
    assert loaded["format"] == FORMAT and set(loaded["state_dict"]) == set(model.state_dict())

    cfg = dict(CFG, resume_weights=str(out))
    rebuilt = YOLOv8Detector(cfg).build_model(cfg, torch.device("cpu")).eval()
    model.eval()
    x = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        expected, actual = model(x)[0], rebuilt(x)[0]
    assert torch.allclose(expected, actual, atol=1e-5)
    assert rebuilt.names[2] == "cyclist"
