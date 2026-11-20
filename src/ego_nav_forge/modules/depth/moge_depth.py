"""Optional MoGe depth module (after_pose).

Default workflow folds MoGe into Pi3XRecon(rgb_moge). Register this separately
when you want depth maps as an explicit stream for multi-scale comparison.
"""
from __future__ import annotations

from typing import Any

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import Module
from ego_nav_forge.registry import register


@register("MoGeDepth")
class MoGeDepth(Module):
    stage = "after_pose"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.image_key = cfg.get("image_key", "forward_img")
        self.name = cfg.get("name", "moge")

    def on_after_pose(self, ctx: EpisodeContext) -> None:
        ctx.extras.setdefault("depth", {})[self.name] = {
            "image_key": self.image_key,
            "backend": "moge",
            "status": "declared",
            "note": "Default scale is inside Pi3XRecon; enable this + MoGeScale for explicit split.",
        }
        print(f"  [MoGeDepth] declared key={self.name}", flush=True)
