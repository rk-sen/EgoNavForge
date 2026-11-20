"""LeRobot export: gravity align, squat z, pack observations."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ego_nav_forge.context import EpisodeContext
from ego_nav_forge.modules.base import ExportModule
from ego_nav_forge.paths import ensure_backend_path
from ego_nav_forge.registry import register


def _feature_name(stream_key: str) -> str:
    if stream_key.endswith("_img"):
        return stream_key[: -len("_img")]
    return stream_key


def _resample_video(src: Path, n: int, dst: Path, fps: float) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cur = -1
    frame = None
    for want in idxs:
        if want < cur:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cur = -1
        while cur < want:
            ok, frame = cap.read()
            if not ok:
                break
            cur += 1
        if frame is None:
            break
        writer.write(frame)
    writer.release()
    cap.release()


@register("LeRobotExport")
class LeRobotExport(ExportModule):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.align = cfg.get("align", "camera_y_mean")
        self.squat_z_low_m = float(cfg.get("squat_z_low_m", -0.25))
        self.squat_z_high_m = float(cfg.get("squat_z_high_m", 0.0))
        self.write_md = bool(cfg.get("write_md", True))
        self.fps = float(cfg.get("fps", 15.0))
        self.pack_override = cfg.get("pack")  # optional; else ctx.pack
        self.search_object = cfg.get("search_object", "unknown")
        self.task_template = cfg.get(
            "task_template",
            "Please explore the venue, find a {obj}, and walk near the {obj}.",
        )
        self.ann_path = Path(cfg["ann_path"]).expanduser() if cfg.get("ann_path") else None
        self._episodes: list[dict] = []
        self._squat_index: dict | None = None

    def _ensure_squat_index(self, ctx: EpisodeContext) -> dict:
        if self._squat_index is not None:
            return self._squat_index
        ensure_backend_path()
        from squat_annotation_map import load_squat_annotation_index

        ann = self.ann_path
        if ann is None and ctx.meta.get("ann_path"):
            ann = Path(ctx.meta["ann_path"])
        if ann and Path(ann).exists():
            self._squat_index = load_squat_annotation_index(
                Path(ann).parent, glob_pat=Path(ann).name
            )
        else:
            self._squat_index = {}
        return self._squat_index

    def on_finalize(self, ctx: EpisodeContext) -> None:
        ensure_backend_path()
        from export_lerobot_xy_yaw import gravity_align_poses
        from squat_annotation_map import build_z_from_annotations

        if ctx.suite_dir is None:
            raise RuntimeError("LeRobotExport: suite_dir missing")
        poses_path = Path(ctx.suite_dir) / "rgb_moge" / "pi3_poses_metric.npy"
        poses = np.load(poses_path) if ctx.poses is None else ctx.poses
        poses_g, gmeta = gravity_align_poses(poses)
        n = len(poses_g)

        fm_path = Path(ctx.suite_dir) / "shared" / "frames_meta.json"
        frames_meta = json.loads(fm_path.read_text(encoding="utf-8")) if fm_path.exists() else []

        squat_index = self._ensure_squat_index(ctx)
        sample_ann = squat_index.get(ctx.episode_id)
        z_arr, z_meta = build_z_from_annotations(
            frames_meta, n, sample_ann, z_low_m=self.squat_z_low_m, z_high_m=self.squat_z_high_m
        )
        state = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            t = poses_g[i, :3, 3]
            fwd = poses_g[i, :3, :3] @ np.array([0.0, 0.0, 1.0])
            yaw = math.atan2(float(fwd[1]), float(fwd[0]))
            state[i] = [float(t[0]), float(t[1]), float(z_arr[i]), yaw]

        action = np.zeros_like(state)
        action[:-1] = state[1:] - state[:-1]
        action[:-1, 3] = (action[:-1, 3] + np.pi) % (2 * np.pi) - np.pi

        out_root = Path(ctx.work_dir.parent.parent) / "lerobot"
        out_root.mkdir(parents=True, exist_ok=True)
        ep_index = len(self._episodes)
        ep_name = f"episode_{ep_index:06d}"

        # stills from suite shared rgb
        shared_rgb = Path(ctx.suite_dir) / "shared" / "rgb"
        files = sorted(shared_rgb.glob("*.jpg"))[:n]
        img_dir = out_root / "images" / ep_name
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(files):
            dst = img_dir / f"frame_{i:06d}.jpg"
            if not dst.exists():
                shutil.copy2(f, dst)

        pack = list(self.pack_override or ctx.pack)
        cam_feats = []
        for key in pack:
            path = ctx.streams.get(key)
            if not path:
                continue
            p = Path(path)
            feat = _feature_name(key)
            if p.suffix.lower() in {".mp4", ".avi", ".mov"} or p.is_file() and "vid" in p.name:
                dst = (
                    out_root
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{feat}"
                    / f"{ep_name}.mp4"
                )
                if not dst.exists():
                    _resample_video(p, n, dst, fps=self.fps)
                cam_feats.append(feat)
            elif key == "imu" or p.suffix.lower() == ".json":
                # copy imu sidecar
                imu_dir = out_root / "meta" / "imu"
                imu_dir.mkdir(parents=True, exist_ok=True)
                if p.is_file():
                    shutil.copy2(p, imu_dir / f"{ep_name}.json")
                elif p.is_dir():
                    # umi slam dir → copy summary
                    summary = p / "slam_summary.json"
                    if summary.exists():
                        shutil.copy2(summary, imu_dir / f"{ep_name}_slam_summary.json")

        rows = []
        for i in range(n):
            rows.append(
                {
                    "episode_index": ep_index,
                    "frame_index": i,
                    "timestamp": i / self.fps,
                    "observation.state": state[i].astype(np.float32).tolist(),
                    "action": action[i].astype(np.float32).tolist(),
                    "task_index": 0,
                }
            )
        pq_dir = out_root / "data" / "chunk-000"
        pq_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(pq_dir / f"{ep_name}.parquet", index=False)

        # search_object from ann if present
        obj = self.search_object
        if self.ann_path and Path(self.ann_path).exists():
            ann = json.loads(Path(self.ann_path).read_text(encoding="utf-8"))
            so = ((ann.get("samples") or {}).get(ctx.episode_id) or {}).get("search_object")
            if so and str(so).strip() and str(so).strip() != "删":
                obj = str(so).strip()
            elif self.search_object:
                obj = self.search_object

        task = self.task_template.replace("{obj}", obj)
        ep_meta = {
            "episode_index": ep_index,
            "episode": ep_name,
            "source": ctx.episode_id,
            "suite": str(ctx.suite_dir),
            "length": n,
            "task": task,
            "task_index": 0,
            "search_object": obj,
            "state_layout": "xyz_yaw",
            "align": self.align,
            "squat_annotated": bool(z_meta.get("squat_annotated")),
            "z_remap": z_meta,
            "gravity": gmeta,
            "pack": pack,
            "cameras": cam_feats,
        }
        self._episodes.append(ep_meta)
        self._write_meta(out_root, task)
        ctx.export_meta = ep_meta
        print(
            f"  [LeRobotExport] {ep_name} n={n} cams={cam_feats} "
            f"squat={ep_meta['squat_annotated']} → {out_root}",
            flush=True,
        )

        if self.write_md:
            md = out_root / "DATASET.md"
            md.write_text(
                f"# EgoNavForge LeRobot\n\n"
                f"- episodes: {len(self._episodes)}\n"
                f"- frames: {sum(e['length'] for e in self._episodes)}\n"
                f"- search_object default: {self.search_object}\n"
                f"- align: {self.align}\n"
                f"- squat_z_low_m: {self.squat_z_low_m}\n",
                encoding="utf-8",
            )

    def _write_meta(self, out_root: Path, task: str) -> None:
        meta = out_root / "meta"
        meta.mkdir(parents=True, exist_ok=True)
        with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        cams = []
        for ep in self._episodes:
            for c in ep.get("cameras") or []:
                if c not in cams:
                    cams.append(c)
        features = {
            f"observation.images.{c}": {"dtype": "video"} for c in cams
        }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [4],
            "names": ["x", "y", "z", "yaw"],
        }
        features["action"] = {
            "dtype": "float32",
            "shape": [4],
            "names": ["dx", "dy", "dz", "dyaw"],
        }
        info = {
            "codebase_version": "v2.1",
            "robot_type": "ego_nav_forge",
            "total_episodes": len(self._episodes),
            "total_frames": int(sum(e["length"] for e in self._episodes)),
            "fps": self.fps,
            "features": features,
            "align": self.align,
            "framework": "EgoNavForge",
        }
        (meta / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
