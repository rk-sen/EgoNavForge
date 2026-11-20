"""Deterministic input preparation — runs BEFORE the first LLM call, no model involved.

Why this exists
---------------
Asking the model to *write* calibration.json is a category error: gravity alignment, FOV and
path geometry are arithmetic on files we already have. In the v2 run the model kept re-reading
the intrinsics because the numbers lived only in a tool result that got evicted from context.

So the runtime now derives everything derivable, writes `calibration.json`, and injects a compact
`INPUT_FACTS` block into *every* request. The model only decides what the numbers mean.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- json mining
def _walk(obj, depth=0):
    yield obj
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:64]:
            yield from _walk(v, depth + 1)


def _as_matrix(item) -> np.ndarray | None:
    if isinstance(item, dict):
        for key in ("c2w", "c2w_opencv_4x4", "c2w_4x4", "pose", "matrix", "transform", "world_matrix"):
            if key in item:
                return _as_matrix(item[key])
        return None
    try:
        a = np.asarray(item, dtype=float)
    except Exception:
        return None
    if a.shape == (16,):                 # some exporters flatten row-major
        return a.reshape(4, 4)
    if a.shape == (4, 4):
        return a
    if a.shape == (3, 4):
        return np.vstack([a, np.array([0, 0, 0, 1.0])])
    return None


def find_poses(doc: Any) -> tuple[np.ndarray | None, str]:
    """Find the longest run of 4x4 matrices anywhere in the document."""
    best, where = None, ""
    for node in _walk(doc):
        if isinstance(node, list) and len(node) >= 3:
            mats = [_as_matrix(x) for x in node]
            if sum(m is not None for m in mats) >= max(3, int(0.6 * len(node))):
                good = [m for m in mats if m is not None]
                if best is None or len(good) > len(best):
                    best, where = np.stack(good), "c2w matrices"
    return best, where


def find_positions(doc: Any) -> tuple[np.ndarray | None, str]:
    """Any 3-D position track: [{position_m|xyz|position|translation|t}, ...] or [[x,y,z], ...]."""
    cands: list[np.ndarray] = []

    def visit(node):
        if isinstance(node, list) and len(node) >= 3:
            rows = []
            for it in node:
                if isinstance(it, dict):
                    for k in ("position_m", "xyz", "position", "translation", "t", "center"):
                        if k in it:
                            try:
                                rows.append([float(x) for x in list(it[k])[:3]])
                            except Exception:
                                pass
                            break
                elif isinstance(it, (list, tuple)) and len(it) == 3 and all(
                        isinstance(x, (int, float)) for x in it):
                    rows.append([float(x) for x in it])
            if len(rows) >= max(3, int(0.6 * len(node))) and len(rows) >= 3:
                cands.append(np.asarray(rows, dtype=float))
        if isinstance(node, dict):
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(doc)
    if not cands:
        return None, ""
    best = max(cands, key=len)
    return best, f"positions ({len(best)})"


# --------------------------------------------------------------------------- geometry
def gravity_frame(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (R_worldfromaligned(3x3), up_world(3), note) so that R @ (aligned vec) = world vec.

    OpenCV c2w columns are (X right, Y down, Z forward) -> up ≈ -mean(Y).
    Blender-style c2w columns are (X right, Y forward, Z up) -> up ≈ +mean(Z).
    We test both and keep the axis that is most stable across poses.
    """
    ys = poses[:, :3, 1]
    zs = poses[:, :3, 2]
    cand_up_down = -ys.mean(axis=0)
    cand_up_blender = zs.mean(axis=0)
    for cand, label in ((cand_up_down, "opencv(-Y)"), (cand_up_blender, "blender(+Z)")):
        n = np.linalg.norm(cand)
        if n > 1e-6:
            stable = float(np.abs(ys @ (cand / n)).mean()) if "opencv" in label else float(np.abs(zs @ (cand / n)).mean())
            if stable > 0.7:
                up = cand / n
                return _rotation_to_up(up), up, label
    up = cand_up_down / (np.linalg.norm(cand_up_down) + 1e-9)
    return _rotation_to_up(up), up, "fallback(-Y)"


def _rotation_to_up(up: np.ndarray) -> np.ndarray:
    """Rotation R with R @ up = +Z, keeping X as east as possible."""
    z = up / np.linalg.norm(up)
    ref = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = ref - z * (ref @ z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    # R maps aligned axes (x,y,z) to world: R = [x y z] as columns
    return np.stack([x, y, z], axis=1)


def _yaw_pitch(forward: np.ndarray) -> tuple[float, float]:
    f = forward / (np.linalg.norm(forward) + 1e-9)
    yaw = math.degrees(math.atan2(f[1], f[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, f[2]))))
    return round(yaw, 1), round(pitch, 1)


# --------------------------------------------------------------------------- intrinsics
def parse_intrinsics(doc: Any) -> dict[str, Any]:
    K, W, H = None, None, None
    for node in _walk(doc):
        if isinstance(node, dict):
            for key in ("K", "K_pinhole_pipeline", "k", "intrinsics", "camera_matrix"):
                if key in node:
                    try:
                        K = np.asarray(node[key], dtype=float)
                    except Exception:
                        pass
            for key in ("pipeline_wh", "wh", "size", "image_size", "resolution"):
                if key in node and isinstance(node[key], (list, tuple)) and len(node[key]) == 2:
                    W, H = int(node[key][0]), int(node[key][1])
    if K is None:
        return {}
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]) if K.shape[1] > 2 else None, float(K[1, 2]) if K.shape[1] > 2 else None
    out: dict[str, Any] = {"fx": round(fx, 2), "fy": round(fy, 2), "cx": cx, "cy": cy}
    if W and H:
        out["width"], out["height"] = W, H
        out["hfov_deg"] = round(math.degrees(2 * math.atan((W / 2) / fx)), 1)
        out["vfov_deg"] = round(math.degrees(2 * math.atan((H / 2) / fy)), 1)
        out["lens_mm_36"] = round((36.0 / 2) / math.tan(math.radians(out["hfov_deg"] / 2)), 2)
    return out


# --------------------------------------------------------------------------- main
def prepare(project: Path, video_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write calibration.json + return the INPUT_FACTS dict (also saved as inputs_facts.json)."""
    project = Path(project)
    inputs = project / "inputs"
    traj_path = next(iter(sorted(inputs.glob("*traject*"))) , None)
    # find intrinsics by CONTENT, not by file name: your upload is K_pinhole_pipeline.json
    intr_path, intr_scan = None, {}
    for cand in sorted(inputs.glob("*.json")):
        try:
            parsed = parse_intrinsics(json.loads(cand.read_text()))
        except Exception:
            continue
        if parsed.get("fx"):
            if intr_path is None or "intrin" in cand.name.lower():
                intr_path, intr_scan = cand, parsed
            if "intrin" in cand.name.lower():
                break
    intr = intr_scan
    facts: dict[str, Any] = {"sources": {}, "notes": []}


    if intr:
        facts["intrinsics"] = intr
        facts["sources"]["intrinsics"] = intr_path.name
        facts["intrinsics_path"] = intr_path.name

    poses, ptype = None, ""
    positions = None
    if traj_path and traj_path.suffix == ".json":
        try:
            doc = json.loads(traj_path.read_text())
            poses, ptype = find_poses(doc)
            positions, _pt = find_positions(doc)       # always: denser track improves bbox/path
            if poses is None:
                ptype = _pt
        except Exception as exc:
            facts["notes"].append(f"trajectory unreadable: {exc}")
    if poses is not None:
        facts["sources"]["trajectory"] = traj_path.name
        R, up, convention = gravity_frame(poses)
        centres = poses[:, :3, 3]
        aligned = (R.T @ centres.T).T          # world -> gravity-aligned frame
        grounded = aligned.copy()
        floor_z = float(np.percentile(grounded[:, 2], 5))
        grounded[:, 2] -= floor_z
        # Pi3/MoGe tracks often carry a *relative* height (0.0-0.3 m) instead of an eye-level camera.
        # v4c put the camera at z≈0 because of this and every render saw the floor. Lift the whole
        # track to a plausible eye height when the vertical spread says it is not metric height.
        h95 = float(np.percentile(grounded[:, 2], 95))
        eye_offset = 0.0
        if h95 < 0.8:
            eye_offset = 1.45 - h95
            grounded[:, 2] += eye_offset
        fwd_world = poses[:, :3, 2] if "opencv" in convention else poses[:, :3, 1]
        fwds = (R.T @ fwd_world.T).T
        path = [(round(float(a), 2), round(float(b), 2)) for a, b in grounded[:, :2]]
        path_len = float(np.sum(np.linalg.norm(np.diff(grounded[:, :2], axis=0), axis=1)))
        heights = grounded[:, 2]
        facts["trajectory"] = {
            "poses": int(len(poses)), "type": ptype, "convention": convention,
            "up_world": [round(float(v), 3) for v in up],
            "path_length_m": round(path_len, 2),
            "camera_height_m": [round(float(np.percentile(heights, 5)), 2), round(float(np.percentile(heights, 95)), 2)],
            "raw_extent_m": [round(float(v), 2) for v in np.ptp(centres, axis=0)],
            "first_pose_xy": [round(float(v), 2) for v in grounded[0, :2]],
            "last_pose_xy": [round(float(v), 2) for v in grounded[-1, :2]],
        }
        step = max(1, len(poses) // 12)
        samples = []
        for i in range(0, len(poses), step):
            yaw, pitch = _yaw_pitch(fwds[i])
            samples.append({"i": int(i), "pos": [round(float(v), 2) for v in grounded[i]],
                            "yaw": yaw, "pitch": pitch})
        facts["trajectory"]["samples"] = samples
        facts["alignment"] = {"R_world_from_aligned": np.round(R, 4).tolist(), "floor_z_raw": round(floor_z, 3)}
        cal = {
            "gravity_up_source": convention,
            "rotation_world_from_aligned": np.round(R, 4).tolist(),
            "floor_z": 0.0,
            "camera_path": samples,
            "hfov_deg": intr.get("hfov_deg"),
            "lens_mm": intr.get("lens_mm_36"),
            "intrinsics_path": (intr_path.name if intr_path else None),
            "room_bbox_m": None,
            "metres_per_unit": 1.0,
            "scale_source": "trajectory claims metres; estimate only, not pixel metrology",
            "camera_eye_offset_m": round(float(eye_offset), 2),
            "notes": [],
        }
        if positions is not None and len(positions) > len(poses):
            # a denser position track gives a better room bbox than a handful of key frames
            p_aligned = (R.T @ positions.T).T
            com_path = float(np.sum(np.linalg.norm(np.diff(p_aligned[:, :2], axis=0), axis=1)))
            facts["trajectory"]["path_length_m_dense"] = round(com_path, 2)
        if positions is not None:
            p_aligned = (R.T @ positions.T).T
            ext = np.ptp(p_aligned[:, :2], axis=0)
            ext_min = p_aligned[:, :2].min(axis=0) - 2.0
            ext_max = p_aligned[:, :2].max(axis=0) + 2.0
            ceil_m = max(2.7, float(np.percentile(p_aligned[:, 2], 95) - floor_z) + 1.6)
            cal["room_bbox_m"] = [round(float(ext_min[0]), 2), round(float(ext_max[0]), 2),
                                  round(float(ext_min[1]), 2), round(float(ext_max[1]), 2),
                                  round(float(ceil_m), 2)]
            facts["trajectory"]["room_bbox_from_track_m"] = cal["room_bbox_m"]
        if cal.get("room_bbox_m") is None and len(poses):
            # No dense position track: use the gravity-aligned pose centres with a conservative
            # 2 m walking margin.  This is a fallback, not a floorplan; the LLM pass refines it.
            ext_min = grounded[:, :2].min(axis=0) - 2.0
            ext_max = grounded[:, :2].max(axis=0) + 2.0
            ceil_m = max(2.7, float(np.percentile(grounded[:, 2], 95)) + 1.6)
            cal["room_bbox_m"] = [round(float(ext_min[0]), 2), round(float(ext_max[0]), 2),
                                  round(float(ext_min[1]), 2), round(float(ext_max[1]), 2),
                                  round(float(ceil_m), 2)]
            facts["trajectory"]["room_bbox_from_track_m"] = cal["room_bbox_m"]
        (project / "calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding="utf8")
        facts["calibration_written"] = "calibration.json (motion part; scale/room still to be confirmed by the model)"
    elif positions is not None:
        facts["trajectory"] = {"poses": int(len(positions)), "type": ptype,
                               "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))), 2)}
        facts["notes"].append("only positions found (no orientations): yaw/pitch unknown")
    if video_meta:
        facts["video"] = video_meta

    video_rel = facts.get("video", {}).get("path") if isinstance(facts.get("video"), dict) else None
    video_path = project / "inputs" / Path(video_rel) if video_rel else None
    if video_path is None or not video_path.exists():
        # Video facts may lack a path when ffprobe returns metadata only. Do not
        # report ready.video=false when source_video.mp4 is physically present.
        video_path = next(iter(sorted((project / "inputs").glob("*.mp4"), key=lambda p: p.stat().st_size,
                                      reverse=True)), None)
    facts["ready"] = {
        "intrinsics": bool(intr), "trajectory": bool(poses is not None or positions is not None),
        "calibration": (project / "calibration.json").exists(),
        "video": bool(video_path and video_path.exists()),
    }
    (project / "inputs_facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1), encoding="utf8")
    return facts


def facts_block(facts: dict[str, Any], limit: int = 0) -> str:
    """No truncation: a cut-off fact block is how the model lost the intrinsics path in v3 and
    re-read the same files for 10 steps. `limit=0` means send the whole JSON."""
    text = json.dumps(facts, ensure_ascii=False, indent=1)
    if limit and len(text) > limit:
        text = text[:limit] + f"\n… (truncated to {limit} chars; full copy in inputs_facts.json)"
    return "<input_facts>\n" + text + "\n</input_facts>"
