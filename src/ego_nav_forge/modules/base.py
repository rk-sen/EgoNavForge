"""Module base classes.

Users / YAML only see `type:` registered classes. Heavy algorithms live *inside*
these classes (optionally calling private backends). Workflows are YAML stage
lists over these types — not copies of scripts under backends/.
"""
from __future__ import annotations

from typing import Any

from ego_nav_forge.context import EpisodeContext


class Module:
    """Big lifecycle unit. Override the stage hook(s) you need."""

    type_name: str = "Module"
    stage: str | None = None  # optional hint: episode_start|frame|chunk|...

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = dict(cfg)
        self.enabled = bool(cfg.get("enabled", True))

    def on_episode_start(self, ctx: EpisodeContext) -> None:
        pass

    def on_frame(self, ctx: EpisodeContext, frame_index: int) -> None:
        pass

    def on_chunk(self, ctx: EpisodeContext, chunk_index: int, span: tuple[int, int]) -> None:
        pass

    def on_after_pose(self, ctx: EpisodeContext) -> None:
        pass

    def on_accumulate(self, ctx: EpisodeContext) -> None:
        pass

    def on_finalize(self, ctx: EpisodeContext) -> None:
        pass


class IngestModule(Module):
    """episode_start: build named `streams` + default `pack`."""

    stage = "episode_start"

    def list_episodes(self) -> list:
        raise NotImplementedError


class PreprocessModule(Module):
    """frame: mask / dynamic filter on a chosen image stream."""

    stage = "frame"


class TrajectoryModule(Module):
    """chunk (+ internal scale): produce metric poses from configured image_key.

    Multi-camera layouts still use one trajectory module; only `image_key` /
    loader `stream_files` change. Export `pack` is independent.
    """

    stage = "chunk"


class ExportModule(Module):
    """finalize: write dataset from poses + pack streams."""

    stage = "finalize"


class VizModule(Module):
    stage = "finalize"
