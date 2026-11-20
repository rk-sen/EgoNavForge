"""Example: register a custom trajectory module (MMLab-style).

Usage in YAML (chunk stage):
  - type: MyIdentityPose
    enabled: true
    image_key: forward_img

You do NOT copy the whole workflow — only swap the trajectory `type`.
"""
from __future__ import annotations

import numpy as np

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import TrajectoryModule
from ego_nav_forge.registry import register


@register("MyIdentityPose")
class MyIdentityPose(TrajectoryModule):
    """Toy pose: N identity cameras (for wiring tests only)."""

    def on_chunk(self, ctx: EpisodeContext, chunk_index: int, span: tuple[int, int]) -> None:
        if chunk_index > 0:
            return
        n = int(ctx.meta.get("n_frames") or 10)
        poses = np.repeat(np.eye(4)[None, ...], n, axis=0)
        out = ctx.work_dir / "suite_toy" / "rgb_moge"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "pi3_poses_metric.npy", poses)
        ctx.suite_dir = out.parent
        ctx.poses = poses
        print(f"  [MyIdentityPose] wrote {n} identity poses → {out}", flush=True)
