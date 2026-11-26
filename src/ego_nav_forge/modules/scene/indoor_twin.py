"""Finalize: video + Pi3 metric poses + intrinsics (+ optional point cloud) → Blender twin.

The vendored agent is backends/indoor_twin_agent (asset-library retrieval).
It replaces the previous v37 tool-calling pipeline. Point cloud is passed
through only when the suite actually wrote a PLY.
"""
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


def default_catalog() -> Path:
    env = os.environ.get("EGO_NAV_FORGE_ASSET_BANK")
    if env:
        return Path(env).expanduser().resolve()
    return (project_root() / "backends" / "asset_bank").resolve()


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


def video_pts(video: Path) -> np.ndarray:
    """Decode-order timestamps, zeroed at the first frame. Matches the agent."""
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    frames = json.loads(out).get("frames") or []
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in frames], dtype=float)
    if len(pts) < 2:
        raise RuntimeError(f"IndoorTwinScene: ffprobe found {len(pts)} frames in {video}")
    pts -= pts[0]
    return pts


def poses_to_trajectory_json(
    poses: np.ndarray,
    *,
    video: Path,
    image_stride: int = 1,
    source: str = "ego_nav_forge Pi3XRecon",
    max_poses: int = 0,
) -> dict[str, Any]:
    """OpenCV c2w poses as the agent's timestamped trajectory.

    The agent treats any file that contains ``key_frames`` as the old Pi3
    sample (``blender_matrix_world``). This writer uses ``convention:
    opencv_c2w`` so the agent applies its own camera flip. ``time_s`` is the
    video PTS of pose i at frame i * image_stride.
    """
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N,4,4), got {arr.shape}")
    if max_poses and len(arr) > max_poses:
        idx = np.linspace(0, len(arr) - 1, max_poses).round().astype(int)
        arr = arr[np.unique(idx)]
    pts = video_pts(video)
    stride = max(int(image_stride), 1)
    entries = []
    for i, mat in enumerate(arr):
        frame = i * stride
        if frame >= len(pts):
            break
        entries.append(
            {
                "time_s": float(pts[frame]),
                "matrix": [[float(x) for x in row] for row in mat],
            }
        )
    if len(entries) < 2:
        raise RuntimeError(
            f"IndoorTwinScene: only {len(entries)} poses land inside the video "
            f"({len(pts)} frames, stride {stride})"
        )
    times = [e["time_s"] for e in entries]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError("IndoorTwinScene: trajectory timestamps are not strictly increasing")
    xyz = np.array([[row[3] for row in e["matrix"][:3]] for e in entries], dtype=float)
    diffs = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.array([0.0])
    return {
        "source": source,
        "units": "meters",
        "convention": "opencv_c2w",
        "image_stride": stride,
        "n_poses": int(len(entries)),
        "path_length_m": float(diffs.sum()),
        "poses": entries,
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
    """Optional finalize module: asset-library indoor twin.

    Needs metric poses from Pi3XRecon plus the recon video and camera
    intrinsics. A colored PLY is forwarded when present and
    ``use_pointcloud`` is true; the agent still runs without it.
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
        self.api_key = cfg.get("api_key")
        self.model = cfg.get("model", os.environ.get("TWIN_MODEL", "kimi-k2.6"))
        self.frames = int(cfg.get("frames", 9))
        self.samples = int(cfg.get("samples", 40))
        self.scene_type = cfg.get("scene_type", "auto")
        self.blender = Path(cfg["blender"]).expanduser() if cfg.get("blender") else default_blender()
        self.intrinsics_path = Path(cfg["intrinsics_path"]).expanduser() if cfg.get("intrinsics_path") else None
        self.calib_camera = cfg.get("calib_camera", "bleft")
        self.video_path_override = Path(cfg["video_path"]).expanduser() if cfg.get("video_path") else None
        self.trajectory_path_override = (
            Path(cfg["trajectory_path"]).expanduser() if cfg.get("trajectory_path") else None
        )
        self.pointcloud_path = Path(cfg["pointcloud_path"]).expanduser() if cfg.get("pointcloud_path") else None
        self.use_pointcloud = bool(cfg.get("use_pointcloud", True))
        self.max_poses = int(cfg.get("max_poses", 0))
        self.check_only = bool(cfg.get("check_only", False))
        self.agent_root = Path(cfg["agent_root"]).expanduser() if cfg.get("agent_root") else indoor_twin_root()
        self.catalog = Path(cfg["catalog"]).expanduser() if cfg.get("catalog") else default_catalog()

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
        for name in ("calib.json", "calibration.json"):
            cand = ctx.episode_dir / name
            doc = intrinsics_from_calib(cand, self.calib_camera)
            if doc:
                dst.write_text(json.dumps(doc, indent=2), encoding="utf-8")
                return dst
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

    def _write_trajectory(self, ctx: EpisodeContext, twin_in: Path, poses: np.ndarray, video: Path) -> Path:
        dst = twin_in / "trajectory.json"
        if self.trajectory_path_override and self.trajectory_path_override.is_file():
            shutil.copy2(self.trajectory_path_override, dst)
            return dst
        stride = int(ctx.meta.get("image_stride", 1))
        doc = poses_to_trajectory_json(
            poses,
            video=video,
            image_stride=stride,
            source=f"ego_nav_forge:{ctx.episode_id}",
            max_poses=self.max_poses,
        )
        dst.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return dst

    def _resolve_pointcloud(self, ctx: EpisodeContext, twin_in: Path) -> Path | None:
        if not self.use_pointcloud:
            return None
        src = self.pointcloud_path
        if src is None and ctx.suite_dir is not None:
            cand = Path(ctx.suite_dir) / "rgb_moge" / "points_metric_textured.ply"
            if cand.is_file():
                src = cand
        if src is None or not src.is_file():
            return None
        dst = twin_in / "points_metric_textured.ply"
        shutil.copy2(src, dst)
        return dst

    def _api_env(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.api_key:
            env[self.api_key_env] = str(self.api_key)
        if not env.get(self.api_key_env):
            for alt in ("KIMI_API_KEY", "MOONSHOT_API_KEY", "API_KEY", "zhesuan_API_KEY"):
                if env.get(alt):
                    env[self.api_key_env] = env[alt]
                    break
        if env.get(self.api_key_env) and not env.get("KIMI_API_KEY"):
            env["KIMI_API_KEY"] = env[self.api_key_env]
        return env

    def on_finalize(self, ctx: EpisodeContext) -> None:
        agent = self.agent_root
        agent_py = agent / "agent.py"
        if not agent_py.is_file():
            raise RuntimeError(f"IndoorTwinScene: agent missing at {agent_py}")

        out_dir = ctx.work_dir / self.out_subdir
        twin_in = out_dir / "_inputs"
        twin_in.mkdir(parents=True, exist_ok=True)

        video = self._resolve_video(ctx)
        poses = self._resolve_poses(ctx)
        traj = self._write_trajectory(ctx, twin_in, poses, video)
        intr = self._resolve_intrinsics(ctx, twin_in)
        ply = self._resolve_pointcloud(ctx, twin_in)

        meta = {
            "video": str(video),
            "trajectory": str(traj),
            "intrinsics": str(intr),
            "pointcloud": str(ply) if ply else None,
            "suite_dir": str(ctx.suite_dir) if ctx.suite_dir else None,
            "n_poses": int(len(poses)),
            "frames": self.frames,
            "samples": self.samples,
            "scene_type": self.scene_type,
            "api_base": self.api_base,
            "model": self.model,
            "agent_root": str(agent),
            "catalog": str(self.catalog),
        }
        (out_dir / "TWIN_INPUTS.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  [IndoorTwinScene] inputs → {out_dir}", flush=True)
        print(f"    video={video}", flush=True)
        print(f"    traj={traj} poses={meta['n_poses']}", flush=True)
        print(f"    K={intr}", flush=True)
        print(f"    ply={ply or 'none'}", flush=True)

        if self.check_only:
            doc = json.loads(traj.read_text(encoding="utf-8"))
            if doc.get("convention") != "opencv_c2w" or len(doc.get("poses") or []) < 2:
                raise RuntimeError(f"IndoorTwinScene: trajectory not usable: {traj}")
            ctx.export_meta["scene_twin"] = {**meta, "check_only": True}
            print("  [IndoorTwinScene] check only, no API", flush=True)
            return

        if not (self.catalog / "index.sqlite3").is_file():
            raise RuntimeError(
                f"IndoorTwinScene: asset bank index missing at {self.catalog}; "
                "build it with backends/indoor_twin_agent/build_asset_bank.py"
            )

        blender = self.blender
        if not Path(blender).is_file() and shutil.which(str(blender)) is None:
            raise RuntimeError(f"IndoorTwinScene: Blender not found: {blender}")
        env = self._api_env()
        if not env.get("KIMI_API_KEY"):
            raise RuntimeError(
                f"IndoorTwinScene: set {self.api_key_env} (or KIMI_API_KEY) for the VLM API"
            )

        cmd = [
            sys.executable,
            str(agent_py),
            "--video",
            str(video),
            "--trajectory",
            str(traj),
            "--intrinsics",
            str(intr),
            "--catalog",
            str(self.catalog),
            "--output",
            str(out_dir),
            "--blender",
            str(blender),
            "--endpoint",
            self.api_base,
            "--model",
            self.model,
            "--scene-type",
            str(self.scene_type),
            "--frames",
            str(self.frames),
            "--samples",
            str(self.samples),
        ]
        if ply is not None:
            cmd += ["--pointcloud", str(ply)]
        print(f"  [IndoorTwinScene] RUN {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(agent), env=env)

        scene = out_dir / "Scene" / "IndoorTwin.blend"
        usd = out_dir / "Scene" / "Scene.usda"
        ctx.export_meta["scene_twin"] = {
            **meta,
            "mode": "asset_library",
            "out_dir": str(out_dir),
            "scene_blend": str(scene) if scene.is_file() else None,
            "usd": str(usd) if usd.is_file() else None,
            "previews": str(out_dir / "Scene" / "Previews"),
        }
        print(f"  [IndoorTwinScene] done scene={scene.exists()} → {out_dir}", flush=True)
