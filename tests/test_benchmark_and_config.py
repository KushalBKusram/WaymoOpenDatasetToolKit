import json
from pathlib import Path

import pytest
import yaml

from modules.benchmark import load_split_manifest, stride_timestamps
from modules.runtime import resolve_image_hw

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_frozen_manifests():
    small = load_split_manifest(CONFIGS / "benchmark_splits.json")
    large = load_split_manifest(CONFIGS / "benchmark_splits_100.json")
    assert (small["id"], len(small["train"]), len(small["eval"])) == ("v1", 20, 10)
    assert (large["id"], len(large["train"]), len(large["eval"])) == ("v1-train100", 100, 10)
    assert large["eval"] == small["eval"]
    assert large["train"][:20] == small["train"]                 # the 20-segment set is a prefix of the 100
    assert len(set(large["train"])) == 100


def test_manifest_with_overlap_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"id": "bad", "train": ["a", "b"], "eval": ["b"]}))
    with pytest.raises(ValueError, match="overlap"):
        load_split_manifest(path)


def test_stride_timestamps():
    assert stride_timestamps([1, 2, 3, 4, 5], 2) == [1, 3, 5]
    assert stride_timestamps([1, 2, 3]) == [1, 2, 3]
    with pytest.raises(ValueError):
        stride_timestamps([1, 2, 3], 0)


def test_resolve_image_hw():
    assert resolve_image_hw(512) == (512, 512)
    assert resolve_image_hw([640, 960]) == (640, 960)
    assert resolve_image_hw((640, 960)) == (640, 960)
    for bad in ([1, 2, 3], "abc", None):
        with pytest.raises((ValueError, TypeError)):
            resolve_image_hw(bad)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_config_has_required_sections(path):
    cfg = yaml.safe_load(path.read_text())
    assert isinstance(cfg.get("task"), str)
    assert isinstance(cfg["model"]["type"], str)
    assert "data" in cfg and "train" in cfg
