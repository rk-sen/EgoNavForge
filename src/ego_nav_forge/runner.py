"""Lifecycle runner: episode_start → frame → chunk → after_pose → accumulate → finalize."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ego_nav_forge.config import STAGES, load_config, resolve_path
from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.registry import create
import ego_nav_forge.modules  # noqa: F401  — register all


class Runner:
    def __init__(self, cfg: dict[str, Any], config_path: Path | None = None):
        self.cfg = cfg
        self.config_path = config_path
        self.work_dir = Path(cfg["work_dir"]).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.modules_by_stage: dict[str, list] = {}
        for stage in STAGES:
            self.modules_by_stage[stage] = [
                create(m["type"], m) for m in cfg.get("_stages", {}).get(stage, [])
            ]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Runner":
        path = Path(path)
        cfg = load_config(path)
        # resolve relative work_dir against config file parent unless absolute
        wd = Path(cfg["work_dir"])
        if not wd.is_absolute():
            cfg["work_dir"] = str((path.parent / wd).resolve())
        return cls(cfg, config_path=path)

    def _discover_episodes(self) -> list[Path]:
        """Ask the first episode_start loader for the episode list."""
        for m in self.modules_by_stage["episode_start"]:
            if hasattr(m, "list_episodes"):
                return list(m.list_episodes())
        raise RuntimeError("No loader module with list_episodes() in episode_start")

    def run(self, only: list[str] | None = None, max_eps: int = 0) -> dict[str, Any]:
        episodes = self._discover_episodes()
        if only:
            want = set(only)
            episodes = [e for e in episodes if e.name in want]
        if max_eps > 0:
            episodes = episodes[:max_eps]

        summary = {"exp_name": self.cfg.get("exp_name"), "work_dir": str(self.work_dir), "episodes": []}
        status_path = self.work_dir / "STATUS.jsonl"
        print(
            f"[EgoNavForge] exp={self.cfg.get('exp_name')} n={len(episodes)} work_dir={self.work_dir}",
            flush=True,
        )
        for i, ep_dir in enumerate(episodes):
            print(f"\n===== [{i}/{len(episodes)-1}] {ep_dir.name} =====", flush=True)
            ctx = EpisodeContext(
                episode_id=ep_dir.name,
                episode_dir=ep_dir,
                work_dir=self.work_dir / "episodes" / ep_dir.name,
            )
            ctx.work_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._run_episode(ctx)
                rec = {"ok": True, "episode": ep_dir.name, "suite": str(ctx.suite_dir), "export": ctx.export_meta}
                summary["episodes"].append(rec)
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                print(f"[{ep_dir.name}] OK", flush=True)
            except Exception as e:
                rec = {"ok": False, "episode": ep_dir.name, "error": str(e)}
                summary["episodes"].append(rec)
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{ep_dir.name}] FAIL: {e}", flush=True)
                raise

        (self.work_dir / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"[EgoNavForge] DONE → {self.work_dir}", flush=True)
        return summary

    def _run_episode(self, ctx: EpisodeContext) -> None:
        # episode_start
        for m in self.modules_by_stage["episode_start"]:
            m.on_episode_start(ctx)

        n_frames = int(ctx.meta.get("n_frames") or len(ctx.frames_meta) or 0)
        # frame stage
        frame_mods = self.modules_by_stage["frame"]
        if frame_mods and n_frames > 0:
            for fi in range(n_frames):
                for m in frame_mods:
                    m.on_frame(ctx, fi)
        elif frame_mods:
            # modules may prepare globally when n_frames unknown yet
            for m in frame_mods:
                m.on_frame(ctx, -1)

        # chunk stage — modules may schedule internally when chunk_index < 0
        chunk_mods = self.modules_by_stage["chunk"]
        chunks = ctx.meta.get("chunks")  # optional list[(i0,i1)]
        if chunk_mods:
            if chunks:
                for ci, span in enumerate(chunks):
                    for m in chunk_mods:
                        m.on_chunk(ctx, ci, tuple(span))
            else:
                for m in chunk_mods:
                    m.on_chunk(ctx, -1, (0, n_frames))

        for m in self.modules_by_stage["after_pose"]:
            m.on_after_pose(ctx)
        for m in self.modules_by_stage["accumulate"]:
            m.on_accumulate(ctx)
        for m in self.modules_by_stage["finalize"]:
            m.on_finalize(ctx)
