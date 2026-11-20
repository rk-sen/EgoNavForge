"""Lightweight registry / config tests (no GPU)."""
from pathlib import Path

from ego_nav_forge.config import load_config
from ego_nav_forge.registry import registered
import ego_nav_forge.modules  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def test_modules_registered():
    names = set(registered())
    for req in (
        "ParsedRootLoader",
        "Pi3XRecon",
        "LeRobotExport",
        "FixedGripperMask",
        "YOLOPersonFilter",
        "MoGeDepth",
        "MoGeScale",
    ):
        assert req in names


def test_shared_workflow_two_datasets():
    umi = load_config(ROOT / "configs/datasets/umi_single_video_imu.yaml")
    te = load_config(ROOT / "configs/datasets/threeeyes_three_video_imu.yaml")

    def types(cfg):
        return [(b["stage"], m["type"]) for b in cfg["pipeline"] for m in b["modules"]]

    assert types(umi) == types(te)
    assert "Pi3XRecon" in {t for _, t in types(umi)}
    assert umi["_stages"]["episode_start"][0]["stream_files"]["forward_img"] == "rgb.mp4"
    assert te["_stages"]["episode_start"][0]["stream_files"]["forward_img"] == "bleftvid.mp4"
    assert "left_img" in te["_stages"]["episode_start"][0]["stream_files"]
    assert umi["_stages"]["chunk"][0]["image_key"] == te["_stages"]["chunk"][0]["image_key"]


def test_no_backends_pipeline_surface():
    """User must not see a dump of legacy scripts as the API."""
    assert not (ROOT / "backends" / "pipeline").exists()
