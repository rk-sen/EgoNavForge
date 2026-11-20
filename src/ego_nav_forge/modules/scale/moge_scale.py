"""Optional MoGe scale module (after_pose).

When Pi3XRecon already wrote metric poses, this only validates/loads them.
For a future split (raw poses + external depth → scale), implement the math here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import Module
from ego_nav_forge.registry import register


@register("MoGeScale")
class MoGeScale(Module):
    stage = "after_pose"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.depth_key = cfg.get("depth_key", "moge")

    def on_after_pose(self, ctx: EpisodeContext) -> None:
        if ctx.poses is None and ctx.suite_dir is not None:
            p = Path(ctx.suite_dir) / "rgb_moge" / "pi3_poses_metric.npy"
            if p.exists():
                ctx.poses = np.load(p)
        if ctx.poses is None:
            raise RuntimeError("MoGeScale: no poses on context")
        ctx.extras["scale"] = {"depth_key": self.depth_key, "n": int(len(ctx.poses))}
        print(f"  [MoGeScale] ok n={len(ctx.poses)} depth_key={self.depth_key}", flush=True)
