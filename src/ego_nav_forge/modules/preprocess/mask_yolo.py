"""Preprocess modules: configure mask / YOLO for the recon backend."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import PreprocessModule
from ego_nav_forge.paths import project_root
from ego_nav_forge.registry import register


def _resolve_asset(p: str | Path) -> Path:
    path = Path(p).expanduser()
    if path.is_file():
        return path.resolve()
    cand = project_root() / path
    if cand.is_file():
        return cand.resolve()
    return path


@register("FixedGripperMask")
class FixedGripperMask(PreprocessModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.image_key = cfg.get("image_key", "forward_img")
        self.mask_path = _resolve_asset(cfg.get("mask_path", "assets/mask_first_frame.png"))
        self.paint_black = bool(cfg.get("paint_black", True))
        self.hard_exclude = bool(cfg.get("hard_exclude", True))

    def on_frame(self, ctx: EpisodeContext, frame_index: int) -> None:
        # Applied inside Pi3 prepare; record once.
        if frame_index not in (-1, 0):
            return
        ctx.extras["mask"] = {
            "image_key": self.image_key,
            "mask_path": str(self.mask_path),
            "paint_black": self.paint_black,
            "hard_exclude": self.hard_exclude,
            "enabled": self.mask_path.is_file(),
        }
        if frame_index == -1 or frame_index == 0:
            print(
                f"  [FixedGripperMask] enabled={self.mask_path.is_file()} path={self.mask_path}",
                flush=True,
            )


@register("YOLOPersonFilter")
class YOLOPersonFilter(PreprocessModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.image_key = cfg.get("image_key", "forward_img")
        self.conf = float(cfg.get("conf", 0.55))
        self.weights = _resolve_asset(cfg.get("weights", "assets/yolov8s.pt"))

    def on_frame(self, ctx: EpisodeContext, frame_index: int) -> None:
        if frame_index not in (-1, 0):
            return
        ctx.extras["yolo"] = {
            "image_key": self.image_key,
            "conf": self.conf,
            "weights": str(self.weights),
            "enabled": True,
        }
        print(f"  [YOLOPersonFilter] conf={self.conf} weights={self.weights}", flush=True)
