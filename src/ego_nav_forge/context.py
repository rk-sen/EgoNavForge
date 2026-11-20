"""Episode-scoped context: named streams + intermediate results."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeContext:
    """Mutable bag passed through lifecycle stages for one episode."""

    episode_id: str
    episode_dir: Path
    work_dir: Path
    streams: dict[str, Any] = field(default_factory=dict)
    pack: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    # filled by modules
    frames: list[Any] = field(default_factory=list)
    frames_meta: list[dict[str, Any]] = field(default_factory=list)
    poses: Any = None  # np.ndarray (N,4,4) metric after scale
    suite_dir: Path | None = None
    export_meta: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def get_stream(self, key: str) -> Any:
        if key not in self.streams:
            raise KeyError(f"stream '{key}' not in context; have={list(self.streams)}")
        return self.streams[key]
