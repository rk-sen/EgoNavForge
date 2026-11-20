"""Load episodes from a parsed root with named stream mapping."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import IngestModule
from ego_nav_forge.registry import register


@register("ParsedRootLoader")
class ParsedRootLoader(IngestModule):
    """episode_start: discover episodes under root, build streams + prep rgb for Pi3.

    Config keys
    -----------
    root: parsed dataset root
    episodes: optional list of folder names (else all matching)
    episode_prefix: optional name prefix filter (e.g. seeker_)
    stream_files: map stream_name → relative filename inside episode dir
    pack: list of stream names for export
    image_stride: int
    recon_image_key: which image stream Pi3 reads (copied/symlinked as rgb.mp4)
    imu_key: optional stream name for IMU (file or directory)
    skip_delete_token: if set, drop episodes whose search_object equals this
    ann_path: optional action_annotations.json for delete filter / later export
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.root = Path(cfg["root"]).expanduser().resolve()
        self.episodes_filter = cfg.get("episodes")
        self.episode_prefix = cfg.get("episode_prefix")
        self.stream_files: dict[str, str] = dict(cfg.get("stream_files") or {})
        self.pack: list[str] = list(cfg.get("pack") or list(self.stream_files.keys()))
        self.image_stride = int(cfg.get("image_stride", 2))
        self.recon_image_key = cfg.get("recon_image_key", "forward_img")
        self.ann_path = Path(cfg["ann_path"]).expanduser() if cfg.get("ann_path") else None
        self.skip_delete_token = cfg.get("skip_delete_token")
        self._delete: set[str] = set()
        if self.ann_path and self.ann_path.exists() and self.skip_delete_token:
            ann = json.loads(self.ann_path.read_text(encoding="utf-8"))
            for name, info in (ann.get("samples") or {}).items():
                if (info.get("search_object") or "").strip() == self.skip_delete_token:
                    self._delete.add(name)

    def list_episodes(self) -> list[Path]:
        eps = sorted(
            p
            for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
        )
        if self.episode_prefix:
            eps = [p for p in eps if p.name.startswith(self.episode_prefix)]
        # must have recon video
        recon_file = self.stream_files.get(self.recon_image_key)
        if recon_file:
            eps = [p for p in eps if (p / recon_file).exists()]
        if self.episodes_filter:
            want = set(self.episodes_filter)
            eps = [p for p in eps if p.name in want]
        if self._delete:
            eps = [p for p in eps if p.name not in self._delete]
        return eps

    def on_episode_start(self, ctx: EpisodeContext) -> None:
        streams: dict[str, Any] = {}
        for key, rel in self.stream_files.items():
            path = ctx.episode_dir / rel
            streams[key] = str(path.resolve()) if path.exists() else None
        ctx.streams = streams
        ctx.pack = list(self.pack)
        ctx.meta["image_stride"] = self.image_stride
        ctx.meta["recon_image_key"] = self.recon_image_key
        ctx.meta["loader"] = "ParsedRootLoader"
        ctx.meta["root"] = str(self.root)
        if self.ann_path and self.ann_path.exists():
            ctx.meta["ann_path"] = str(self.ann_path.resolve())

        # Prepare a Pi3-friendly input dir: rgb.mp4 → recon stream
        prep = ctx.work_dir / "prepared"
        prep.mkdir(parents=True, exist_ok=True)
        src_img = streams.get(self.recon_image_key)
        if not src_img:
            raise FileNotFoundError(
                f"recon stream '{self.recon_image_key}' missing under {ctx.episode_dir}"
            )
        rgb = prep / "rgb.mp4"
        if rgb.exists() or rgb.is_symlink():
            rgb.unlink()
        rgb.symlink_to(Path(src_img).resolve())

        # optional: symlink imu / slam for meta consumers
        imu_rel = self.stream_files.get("imu")
        if imu_rel:
            imu_src = ctx.episode_dir / imu_rel
            if imu_src.exists():
                dst = prep / Path(imu_rel).name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if imu_src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    # directory symlink
                    dst.symlink_to(imu_src.resolve())
                else:
                    dst.symlink_to(imu_src.resolve())

        # copy slam dir if present (umi)
        slam = ctx.episode_dir / "slam"
        if slam.is_dir():
            dst = prep / "slam"
            if dst.exists() or dst.is_symlink():
                if dst.is_symlink():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            dst.symlink_to(slam.resolve())

        ctx.meta["prepared_dir"] = str(prep)
        (prep / "ego_nav_forge_meta.json").write_text(
            json.dumps(
                {
                    "episode_id": ctx.episode_id,
                    "source": str(ctx.episode_dir),
                    "streams": streams,
                    "pack": ctx.pack,
                    "recon_image_key": self.recon_image_key,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"  [ParsedRootLoader] streams={list(k for k,v in streams.items() if v)} "
            f"pack={ctx.pack} prepared={prep}",
            flush=True,
        )


@register("DictStreamLoader")
class DictStreamLoader(IngestModule):
    """Single-episode absolute stream paths (framework doc example)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.streams_cfg = dict(cfg.get("streams") or {})
        self.pack = list(cfg.get("pack") or list(self.streams_cfg.keys()))
        self.image_stride = int(cfg.get("image_stride", 2))
        self.episode_id = cfg.get("episode_id", "episode_0")

    def list_episodes(self) -> list[Path]:
        # synthesize a fake episode dir = first image parent
        for v in self.streams_cfg.values():
            p = Path(v)
            if p.exists():
                return [p.parent if p.is_file() else p]
        return [Path(".").resolve()]

    def on_episode_start(self, ctx: EpisodeContext) -> None:
        ctx.streams = {k: str(Path(v).resolve()) for k, v in self.streams_cfg.items()}
        ctx.pack = list(self.pack)
        ctx.meta["image_stride"] = self.image_stride
        ctx.meta["loader"] = "DictStreamLoader"
