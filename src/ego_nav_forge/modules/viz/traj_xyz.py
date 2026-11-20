"""Quick XY / z / yaw visualization."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import VizModule
from ego_nav_forge.paths import ensure_backend_path
from ego_nav_forge.registry import register


@register("TrajXYZViz")
class TrajXYZViz(VizModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.enabled_flag = bool(cfg.get("enabled", True))

    def on_finalize(self, ctx: EpisodeContext) -> None:
        if not self.enabled_flag:
            return
        ensure_backend_path()
        from export_lerobot_xy_yaw import gravity_align_poses

        if ctx.suite_dir is None:
            return
        poses = np.load(Path(ctx.suite_dir) / "rgb_moge" / "pi3_poses_metric.npy")
        poses_g, _ = gravity_align_poses(poses)
        xy = poses_g[:, :2, 3]
        z = poses_g[:, 2, 3]
        z = z - np.median(z[: max(5, len(z) // 10)])
        out = Path(ctx.work_dir.parent.parent) / "viz_xyz"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"{ctx.episode_id}_xyz.png"
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(xy[:, 0], xy[:, 1], "-o", ms=2)
        ax[0].set_aspect("equal", adjustable="datalim")
        ax[0].set_title(f"{ctx.episode_id} XY")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(z, color="#E4572E")
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_title("camera height (rel)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        print(f"  [TrajXYZViz] {fp}", flush=True)
