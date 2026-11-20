"""Finalize: video + Pi3 metric poses + intrinsics → Blender indoor twin (v37 agent)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import ExportModule
from ego_nav_forge.paths import project_root
from ego_nav_forge.registry import register


def indoor_twin_root() -> Path:
    env = os.environ.get("EGO_NAV_FORGE_TWIN_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (project_root() / "backends" / "indoor_twin_agent").resolve()


def default_blender() -> Path:
    env = os.environ.get("BLENDER")
    if env and Path(env).exists():
        return Path(env)
    for c in (
        "/mnt/luozhaoyi/tools/blender-4.5.14-linux-x64/blender",
        "/usr/local/bin/blender",
        shutil.which("blender") or "",
    ):
        if c and Path(c).is_file():
            return Path(c)
    return Path("blender")


def poses_to_trajectory_json(
    poses: np.ndarray,
    *,
    image_stride: int = 1,
    source: str = "ego_nav_forge Pi3XRecon",
    max_poses: int = 0,
) -> dict[str, Any]:
    """Serialize (N,4,4) OpenCV c2w poses into indoor_twin_agent trajectory JSON."""
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N,4,4), got {arr.shape}")
    if max_poses and len(arr) > max_poses:
        idx = np.linspace(0, len(arr) - 1, max_poses).round().astype(int)
        arr = arr[idx]
    xyz = arr[:, :3, 3]
    diffs = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.array([0.0])
    poses_list = [[[float(x) for x in row] for row in mat] for mat in arr]
    key_frames = []
    for i, mat in enumerate(arr):
        key_frames.append(
            {
                "video_frame_approx": int(i * max(image_stride, 1)),
                "pose_row": int(i),
                "position_m": [float(x) for x in mat[:3, 3]],
                "c2w_opencv_4x4": [[float(x) for x in row] for row in mat],
            }
        )
    return {
        "source": source,
        "units": "meters",
        "convention": {
            "poses": "OpenCV camera-to-world (X right, Y down, Z forward)",
            "recommended": "Use poses[] as c2w; indoor_twin_agent inputs_prep gravity-aligns",
        },
        "n_poses": int(len(arr)),
        "image_stride": int(image_stride),
        "path_length_m": float(diffs.sum()),
        "position_min": [float(x) for x in xyz.min(axis=0)],
        "position_max": [float(x) for x in xyz.max(axis=0)],
        "position_extent_m": [float(x) for x in (xyz.max(axis=0) - xyz.min(axis=0))],
        "poses": poses_list,
        "key_frames": key_frames,
    }


def intrinsics_from_calib(
    calib_path: Path,
    camera_name: str = "bleft",
) -> dict[str, Any] | None:
    """Build twin intrinsics JSON from threeeyes/UMI-style calib.json."""
    if not calib_path.is_file():
        return None
    doc = json.loads(calib_path.read_text(encoding="utf-8"))
    cams = doc.get("cameras") or {}
    cam = cams.get(camera_name) or next(iter(cams.values()), None)
    if not isinstance(cam, dict):
        return None
    undist = cam.get("undistort") or {}
    K = undist.get("K_pinhole") or cam.get("K_pinhole")
    wh = undist.get("out_wh")
    if not wh:
        out_res = cam.get("out_resolution") or {}
        if "w" in out_res and "h" in out_res:
            wh = [out_res["w"], out_res["h"]]
    if K is None:
        return None
    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    W = int(wh[0]) if wh else int(round(cx * 2))
    H = int(wh[1]) if wh else int(round(cy * 2))
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "W": W,
        "H": H,
        "K_pinhole_pipeline": K,
        "pipeline_wh": [W, H],
        "camera_name": camera_name,
        "source": str(calib_path),
    }


@register("IndoorTwinScene")
class IndoorTwinScene(ExportModule):
    """Optional finalize module: scene twin via indoor_twin_agent v37.

    Needs metric poses from Pi3XRecon (or equivalent) plus the recon video and
    camera intrinsics. Writes under ``ctx.work_dir / out_subdir``.
    """

    stage = "finalize"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.image_key = cfg.get("image_key", "forward_img")
        self.out_subdir = cfg.get("out_subdir", "scene_twin")
        self.api_base = cfg.get(
            "api_base",
            os.environ.get("TWIN_API_BASE", "https://ai.zj-computility.com/maas/v1"),
        )
        self.api_key_env = cfg.get("api_key_env", "zhesuan_API_KEY")
        self.api_key = cfg.get("api_key")  # optional inline; prefer env
        self.model = cfg.get("model", os.environ.get("TWIN_MODEL", "kimi-k2.6"))
        self.preset = cfg.get("preset", "balanced")
        self.frames = int(cfg.get("frames", 60))
        self.res = int(cfg.get("res", 384))
        self.fps = int(cfg.get("fps", 5))
        self.blender = Path(cfg["blender"]).expanduser() if cfg.get("blender") else default_blender()
        self.intrinsics_path = Path(cfg["intrinsics_path"]).expanduser() if cfg.get("intrinsics_path") else None
        self.calib_camera = cfg.get("calib_camera", "bleft")
        self.video_path_override = Path(cfg["video_path"]).expanduser() if cfg.get("video_path") else None
        self.trajectory_path_override = (
            Path(cfg["trajectory_path"]).expanduser() if cfg.get("trajectory_path") else None
        )
        self.evidence_path = Path(cfg["evidence_path"]).expanduser() if cfg.get("evidence_path") else None
        # Optional: skip VLM cognition and Blender-build from an existing project dir
        # (ARCHITECTURE / SCENE_PLAN / BLUEPRINT / specs / assets already present).
        self.seed_project = Path(cfg["seed_project"]).expanduser() if cfg.get("seed_project") else None
        self.export_pointcloud = bool(cfg.get("export_pointcloud", True))
        self.export_usd = bool(cfg.get("export_usd", True))
        self.copy_person_asset = bool(cfg.get("copy_person_asset", True))
        self.max_poses = int(cfg.get("max_poses", 0))  # 0 = all; smoke can subsample
        self.check_only = bool(cfg.get("check_only", False))
        self.smoke = bool(cfg.get("smoke", False))
        self.objects = cfg.get("objects", "")
        self.force_architecture = bool(cfg.get("force_architecture", False))
        self.agent_root = Path(cfg["agent_root"]).expanduser() if cfg.get("agent_root") else indoor_twin_root()
        if self.smoke:
            self.preset = cfg.get("preset", "fast")
            self.frames = int(cfg.get("frames", 8))
            self.res = int(cfg.get("res", 256))
            self.fps = int(cfg.get("fps", 2))
            self.export_usd = bool(cfg.get("export_usd", False))
            if not self.max_poses:
                self.max_poses = int(cfg.get("max_poses", 24))

    def _resolve_video(self, ctx: EpisodeContext) -> Path:
        if self.video_path_override and self.video_path_override.is_file():
            return self.video_path_override.resolve()
        stream = ctx.streams.get(self.image_key)
        if not stream:
            raise RuntimeError(
                f"IndoorTwinScene: stream '{self.image_key}' missing; have={list(ctx.streams)}"
            )
        path = Path(stream)
        if not path.is_file():
            raise RuntimeError(f"IndoorTwinScene: video not found: {path}")
        return path.resolve()

    def _resolve_poses(self, ctx: EpisodeContext) -> np.ndarray:
        if ctx.poses is not None:
            return np.asarray(ctx.poses)
        if ctx.suite_dir is None:
            raise RuntimeError("IndoorTwinScene: no poses / suite_dir (run Pi3XRecon first)")
        poses_path = Path(ctx.suite_dir) / "rgb_moge" / "pi3_poses_metric.npy"
        if not poses_path.is_file():
            raise RuntimeError(f"IndoorTwinScene: missing {poses_path}")
        return np.load(poses_path)

    def _resolve_intrinsics(self, ctx: EpisodeContext, twin_in: Path) -> Path:
        dst = twin_in / "intrinsics.json"
        if self.intrinsics_path and self.intrinsics_path.is_file():
            shutil.copy2(self.intrinsics_path, dst)
            return dst
        # try episode calib.json
        for name in ("calib.json", "calibration.json"):
            cand = ctx.episode_dir / name
            doc = intrinsics_from_calib(cand, self.calib_camera)
            if doc:
                dst.write_text(json.dumps(doc, indent=2), encoding="utf-8")
                return dst
        # optional stream
        for key in ("intrinsics", "K", "calib"):
            if key in ctx.streams and ctx.streams[key]:
                p = Path(ctx.streams[key])
                if p.is_file():
                    shutil.copy2(p, dst)
                    return dst
        raise RuntimeError(
            "IndoorTwinScene: set intrinsics_path or provide episode calib.json "
            f"(camera={self.calib_camera})"
        )

    def _write_trajectory(self, ctx: EpisodeContext, twin_in: Path, poses: np.ndarray) -> Path:
        if self.trajectory_path_override and self.trajectory_path_override.is_file():
            dst = twin_in / "trajectory.json"
            shutil.copy2(self.trajectory_path_override, dst)
            return dst
        stride = int(ctx.meta.get("image_stride", 1))
        doc = poses_to_trajectory_json(
            poses,
            image_stride=stride,
            source=f"ego_nav_forge:{ctx.episode_id}",
            max_poses=self.max_poses,
        )
        dst = twin_in / "trajectory.json"
        dst.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return dst

    def _maybe_copy_pointcloud(self, ctx: EpisodeContext, out_dir: Path) -> Path | None:
        if not self.export_pointcloud or ctx.suite_dir is None:
            return None
        src = Path(ctx.suite_dir) / "rgb_moge" / "points_metric_textured.ply"
        if not src.is_file():
            return None
        dst = out_dir / "points_metric_textured.ply"
        shutil.copy2(src, dst)
        return dst

    def on_finalize(self, ctx: EpisodeContext) -> None:
        agent = self.agent_root
        twin_py = agent / "twin.py"
        pipeline_py = agent / "pipeline.py"
        if not twin_py.is_file() or not pipeline_py.is_file():
            raise RuntimeError(f"IndoorTwinScene: agent missing at {agent}")

        out_dir = ctx.work_dir / self.out_subdir
        twin_in = out_dir / "_inputs"
        twin_in.mkdir(parents=True, exist_ok=True)

        video = self._resolve_video(ctx)
        poses = self._resolve_poses(ctx)
        traj = self._write_trajectory(ctx, twin_in, poses)
        intr = self._resolve_intrinsics(ctx, twin_in)
        ply = self._maybe_copy_pointcloud(ctx, out_dir)

        # ensure person asset available next to agent (vendored)
        person = agent / "assets" / "person_supplied.glb"
        if self.copy_person_asset and not person.is_file():
            raise RuntimeError(
                f"IndoorTwinScene: missing human asset {person}; "
                "copy person_supplied.glb/.blend into backends/indoor_twin_agent/assets/"
            )

        meta = {
            "video": str(video),
            "trajectory": str(traj),
            "intrinsics": str(intr),
            "pointcloud": str(ply) if ply else None,
            "suite_dir": str(ctx.suite_dir) if ctx.suite_dir else None,
            "n_poses": int(len(poses)),
            "smoke": self.smoke,
            "preset": self.preset,
            "api_base": self.api_base,
            "model": self.model,
            "agent_root": str(agent),
        }
        (out_dir / "TWIN_INPUTS.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  [IndoorTwinScene] inputs → {out_dir}", flush=True)
        print(f"    video={video}", flush=True)
        print(f"    traj={traj} poses={meta['n_poses']}", flush=True)
        print(f"    K={intr}", flush=True)
        if ply:
            print(f"    ply={ply}", flush=True)

        if self.check_only:
            cmd = [
                sys.executable,
                str(twin_py),
                "--video",
                str(video),
                "--trajectory",
                str(traj),
                "--intrinsics",
                str(intr),
                "--out",
                str(out_dir),
                "--check",
            ]
            subprocess.run(cmd, check=True, cwd=str(agent))
            ctx.export_meta["scene_twin"] = {**meta, "check_only": True}
            return

        blender = self.blender
        if not Path(blender).is_file() and shutil.which(str(blender)) is None:
            raise RuntimeError(f"IndoorTwinScene: Blender not found: {blender}")

        # Fast path: seed from a finished agent project, only rebuild Blender (+ optional USD).
        if self.seed_project and self.seed_project.is_dir():
            project = out_dir / "project"
            if project.exists():
                shutil.rmtree(project)
            shutil.copytree(
                self.seed_project,
                project,
                ignore=shutil.ignore_patterns(
                    "previews", "usd", "scene.blend", "__pycache__", "*.pyc"
                ),
            )
            # Refresh inputs to the forge-exported video/traj/K for this episode.
            inputs = project / "inputs"
            inputs.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, inputs / "source_video.mp4")
            shutil.copy2(traj, inputs / "trajectory.json")
            shutil.copy2(intr, inputs / "intrinsics.json")
            # Ensure blender_lib + person asset
            lib_src = agent / "blender_lib"
            if lib_src.is_dir():
                shutil.copytree(lib_src, project / "_lib", dirs_exist_ok=True)
            assets_src = agent / "assets"
            if assets_src.is_dir():
                shutil.copytree(assets_src, project / "assets", dirs_exist_ok=True)
            build_py = agent / "indoor_twin_agent" / "blender_build.py"
            cmd = [
                str(blender),
                "-b",
                "--factory-startup",
                "--python",
                str(build_py),
                "--",
                "--project",
                str(project),
                "--frames",
                str(self.frames),
                "--res",
                str(self.res),
                "--fps",
                str(self.fps),
            ]
            print(f"  [IndoorTwinScene] SEED+BLENDER {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True)
            if self.export_usd:
                export_py = agent / "indoor_twin_agent" / "export_usd.py"
                subprocess.run(
                    [
                        str(blender),
                        "-b",
                        str(project / "scene.blend"),
                        "--python",
                        str(export_py),
                        "--",
                        "--project",
                        str(project),
                    ],
                    check=True,
                )
            scene = project / "scene.blend"
            usd = project / "usd" / "full_scene.usd"
            ctx.export_meta["scene_twin"] = {
                **meta,
                "mode": "seed_project",
                "seed_project": str(self.seed_project),
                "out_dir": str(out_dir),
                "scene_blend": str(scene) if scene.is_file() else None,
                "usd": str(usd) if usd.is_file() else None,
                "previews": str(project / "previews"),
            }
            print(f"  [IndoorTwinScene] done scene={scene.exists()} → {out_dir}", flush=True)
            return

        # Full agent pipeline (VLM cognition + Blender + optional USD)
        env = os.environ.copy()
        if self.api_key:
            env[self.api_key_env] = str(self.api_key)
        if not env.get(self.api_key_env):
            for alt in ("MOONSHOT_API_KEY", "KIMI_API_KEY", "API_KEY", "zhesuan_API_KEY"):
                if env.get(alt):
                    env[self.api_key_env] = env[alt]
                    break
        if not env.get(self.api_key_env):
            raise RuntimeError(
                f"IndoorTwinScene: set {self.api_key_env} (or api_key in YAML) for the VLM API"
            )

        cmd = [
            sys.executable,
            str(pipeline_py),
            "--video",
            str(video),
            "--trajectory",
            str(traj),
            "--intrinsics",
            str(intr),
            "--out",
            str(out_dir),
            "--preset",
            self.preset,
            "--model",
            self.model,
            "--api-base",
            self.api_base,
            "--api-key-env",
            self.api_key_env,
            "--blender",
            str(blender),
            "--frames",
            str(self.frames),
            "--res",
            str(self.res),
            "--fps",
            str(self.fps),
        ]
        if self.objects:
            cmd += ["--objects", str(self.objects)]
        if self.evidence_path and self.evidence_path.is_file():
            cmd += ["--evidence", str(self.evidence_path)]
        if self.force_architecture:
            cmd.append("--force-architecture")
        if not self.export_usd:
            cmd.append("--no-usd")

        print(f"  [IndoorTwinScene] RUN {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(agent), env=env)

        scene = out_dir / "project" / "scene.blend"
        usd = out_dir / "project" / "usd" / "full_scene.usd"
        ctx.export_meta["scene_twin"] = {
            **meta,
            "mode": "full_pipeline",
            "out_dir": str(out_dir),
            "scene_blend": str(scene) if scene.is_file() else None,
            "usd": str(usd) if usd.is_file() else None,
            "previews": str(out_dir / "project" / "previews"),
        }
        print(f"  [IndoorTwinScene] done scene={scene.exists()} → {out_dir}", flush=True)
