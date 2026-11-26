"""Pi3X chunked reconstruction backend (pose-aware Sim3 stitch)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import TrajectoryModule
from ego_nav_forge.paths import default_ckpt, default_moge, ensure_backend_path
from ego_nav_forge.registry import register


@register("Pi3XRecon")
class Pi3XRecon(TrajectoryModule):
    """Trajectory module: chunk Pi3 + pose-aware Sim3 stitch + MoGe metric scale.

    Reads a single `image_key` from ctx.streams (set by the loader). Multi-view
    datasets still use this module — only the loader's stream_files / pack change.
    Implementation currently delegates to backends/pipeline (legacy scripts);
    that is private to this class, not a user-facing workflow.
    """
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.image_key = cfg.get("image_key", "forward_img")
        self.chunk_size = int(cfg.get("chunk_size", 160))
        self.overlap = int(cfg.get("overlap", 40))
        self.stitch = cfg.get("stitch", "pose_join_R_clamped_s")
        ckpt = cfg.get("checkpoint")
        self.checkpoint = Path(ckpt).expanduser() if ckpt else default_ckpt()
        if not self.checkpoint.is_absolute():
            from ego_nav_forge.paths import project_root

            cand = project_root() / self.checkpoint
            if cand.is_file():
                self.checkpoint = cand
        if not self.checkpoint.is_file():
            self.checkpoint = default_ckpt()
        moge = cfg.get("moge_pretrained")
        self.moge_pretrained = str(moge) if moge else str(default_moge())
        self.device = cfg.get("device", "cuda")
        self.force_recon = bool(cfg.get("force_recon", False))
        self.prefix = cfg.get("prefix", "enf_")
        self.scale_mode = cfg.get("scale_mode", "moge")  # used if MoGeScale module absent
        # optional: look for existing suites from prior runs (faster smoke)
        self.reuse_suite_roots = [Path(p) for p in (cfg.get("reuse_suite_roots") or [])]
        self._last_episode: str | None = None
    def on_chunk(self, ctx: EpisodeContext, chunk_index: int, span: tuple[int, int]) -> None:
        # Backend handles all chunks in one call; only run once per episode.
        if self._last_episode == ctx.episode_id and chunk_index != -1:
            return
        if chunk_index > 0:
            return
        self._run_full(ctx)
        self._last_episode = ctx.episode_id
    def _run_full(self, ctx: EpisodeContext) -> None:
        prepared = Path(ctx.meta["prepared_dir"])
        # parent of prepared is episode work; suite goes under work_dir/suite
        suite_root = ctx.work_dir.parent.parent / "suite"
        suite_root.mkdir(parents=True, exist_ok=True)
        suite_dir = suite_root / f"{self.prefix}{ctx.episode_id}"
        poses_path = suite_dir / "rgb_moge" / "pi3_poses_metric.npy"

        if not self.force_recon:
            # Try reuse before importing the heavy Pi3 stack (rosbags/torch/…).
            candidates = [suite_dir]
            for root in self.reuse_suite_roots:
                candidates.append(root / f"{self.prefix}{ctx.episode_id}")
                # also allow bare episode id folder names used by older runs
                candidates.append(root / ctx.episode_id)
            for cand in candidates:
                p = cand / "rgb_moge" / "pi3_poses_metric.npy"
                if p.exists():
                    print(f"  [Pi3XRecon] reuse poses {p}", flush=True)
                    ctx.suite_dir = cand
                    ctx.poses = np.load(p)
                    ctx.meta["recon_reused"] = True
                    return

        ensure_backend_path()
        import run_realdog_parsed_fullseq as recon_mod

        # Build a 1-episode parsed-root pointing at prepared/
        parsed_root = ctx.work_dir / "_parsed_root"
        parsed_root.mkdir(parents=True, exist_ok=True)
        link = parsed_root / ctx.episode_id
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(prepared.resolve())

        mask_cfg = ctx.extras.get("mask") or {}
        yolo_cfg = ctx.extras.get("yolo") or {}
        image_stride = int(ctx.meta.get("image_stride", 2))

        argv = [
            "--parsed-root",
            str(parsed_root),
            "--out-root",
            str(suite_root),
            "--viz-out",
            str(ctx.work_dir.parent.parent / "viz_unused"),
            "--modes",
            "rgb_moge",
            "--prefix",
            self.prefix,
            "--only",
            ctx.episode_id,
            "--max-seqs",
            "1",
            "--image-stride",
            str(image_stride),
            "--chunk-size",
            str(self.chunk_size),
            "--overlap",
            str(self.overlap),
            "--ckpt",
            str(self.checkpoint),
            "--moge-pretrained",
            self.moge_pretrained,
            "--device",
            self.device,
            "--skip-viz",
            "--force-recon",
        ]
        if mask_cfg.get("enabled"):
            argv += ["--global-mask", str(mask_cfg["mask_path"])]
        if yolo_cfg.get("enabled"):
            argv += [
                "--yolo-person",
                "--yolo-conf",
                str(yolo_cfg.get("conf", 0.55)),
            ]
            w = yolo_cfg.get("weights")
            if w:
                argv += ["--yolo-weights", str(w)]

        print(
            f"  [Pi3XRecon] image_key={self.image_key} chunk={self.chunk_size}/{self.overlap} "
            f"stitch={self.stitch}",
            flush=True,
        )
        sys.argv = ["run_realdog_parsed_fullseq.py"] + argv
        recon_mod.main()

        if not poses_path.exists():
            raise FileNotFoundError(f"Pi3 recon failed: missing {poses_path}")
        ctx.suite_dir = suite_dir
        ctx.poses = np.load(poses_path)
        ctx.meta["recon_reused"] = False
        print(f"  [Pi3XRecon] poses n={len(ctx.poses)} → {poses_path}", flush=True)
