"""Load YAML: shared workflow + thin dataset overlay (optional preset alias)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

STAGES = (
    "episode_start",
    "frame",
    "chunk",
    "after_pose",
    "accumulate",
    "finalize",
)

# dataset overlay → which module types receive which keys
_DATASET_INJECT: dict[str, tuple[str, ...]] = {
    "ParsedRootLoader": (
        "root",
        "episodes",
        "episode_prefix",
        "stream_files",
        "pack",
        "recon_image_key",
        "image_stride",
        "ann_path",
        "skip_delete_token",
    ),
    "FixedGripperMask": ("mask_path", "image_key"),
    "YOLOPersonFilter": ("image_key", "conf", "weights"),
    "Pi3XRecon": (
        "image_key",
        "prefix",
        "force_recon",
        "reuse_suite_roots",
        "chunk_size",
        "overlap",
        "checkpoint",
        "moge_pretrained",
        "device",
    ),
    "LeRobotExport": (
        "pack",
        "ann_path",
        "search_object",
        "fps",
        "squat_z_low_m",
        "align",
        "write_md",
    ),
    "IndoorTwinScene": (
        "image_key",
        "api_base",
        "api_key_env",
        "api_key",
        "model",
        "preset",
        "frames",
        "res",
        "fps",
        "blender",
        "intrinsics_path",
        "calib_camera",
        "video_path",
        "trajectory_path",
        "evidence_path",
        "seed_project",
        "export_pointcloud",
        "export_usd",
        "smoke",
        "check_only",
        "max_poses",
        "objects",
        "out_subdir",
        "agent_root",
    ),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = deepcopy(base)
    for k, v in overlay.items():
        if k in ("dataset", "workflow"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _apply_scene_twin(pipeline: list[dict], scene_twin: Any) -> None:
    """Enable/configure IndoorTwinScene from dataset.scene_twin (bool or dict)."""
    for block in pipeline:
        for mod in block.get("modules") or []:
            if mod.get("type") != "IndoorTwinScene":
                continue
            if scene_twin is False or scene_twin is None:
                mod["enabled"] = False
                return
            if scene_twin is True:
                mod["enabled"] = True
                return
            if isinstance(scene_twin, dict):
                mod["enabled"] = bool(scene_twin.get("enabled", True))
                for k, v in scene_twin.items():
                    if k == "enabled":
                        continue
                    mod[k] = deepcopy(v)
                return


def _inject_dataset(pipeline: list[dict], dataset: dict[str, Any]) -> list[dict]:
    """Fill shared workflow modules from dataset binding."""
    ds = dict(dataset)
    if "suite_prefix" in ds and "prefix" not in ds:
        ds["prefix"] = ds["suite_prefix"]
    if "recon_image_key" in ds:
        ds.setdefault("image_key", ds["recon_image_key"])

    out = deepcopy(pipeline)
    for block in out:
        for mod in block.get("modules") or []:
            for k in _DATASET_INJECT.get(mod.get("type"), ()):
                if k in ds and ds[k] is not None:
                    mod[k] = deepcopy(ds[k])
            # optional per-module enable flags from dataset
            if mod.get("type") == "LeRobotExport" and "lerobot_export" in ds:
                mod["enabled"] = bool(ds["lerobot_export"])
            if mod.get("type") == "TrajXYZViz" and "traj_viz" in ds:
                mod["enabled"] = bool(ds["traj_viz"])
    if "scene_twin" in ds:
        _apply_scene_twin(out, ds["scene_twin"])
    return out


def _normalize_stages(pipeline: list[dict]) -> dict[str, list[dict]]:
    stages: dict[str, list[dict]] = {s: [] for s in STAGES}
    for block in pipeline:
        stage = block.get("stage")
        if stage not in stages:
            raise ValueError(f"unknown stage '{stage}'")
        for m in block.get("modules") or []:
            if m.get("enabled", True) is False:
                continue
            stages[stage].append(m)
    return stages


def _resolve_layers(path: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Outermost config first → follow workflow: until a file with no workflow."""
    layers: list[tuple[Path, dict[str, Any]]] = []
    cur = path.resolve()
    seen: set[Path] = set()
    while True:
        if cur in seen:
            raise ValueError(f"workflow cycle at {cur}")
        seen.add(cur)
        raw = _read_yaml(cur)
        layers.append((cur, raw))
        if "workflow" not in raw:
            break
        nxt = Path(raw["workflow"])
        if not nxt.is_absolute():
            nxt = (cur.parent / nxt).resolve()
        cur = nxt
    return layers


def load_config(path: Path) -> dict[str, Any]:
    """Load workflow, or dataset/preset that chains via `workflow:`."""
    path = Path(path).resolve()
    layers = _resolve_layers(path)

    # merge innermost → outermost
    merged: dict[str, Any] = {}
    for _, raw in reversed(layers):
        merged = _deep_merge(merged, raw)

    # dataset: outer wins on key conflicts
    dataset: dict[str, Any] = {}
    for _, raw in reversed(layers):
        if raw.get("dataset"):
            dataset = _deep_merge(dataset, raw["dataset"])
    for _, raw in layers:  # outermost last
        if raw.get("dataset"):
            dataset = _deep_merge(dataset, raw["dataset"])

    if dataset:
        merged["pipeline"] = _inject_dataset(merged.get("pipeline") or [], dataset)
        merged["dataset"] = dataset

    # exp_name: prefer outermost non-empty, else stem
    for _, raw in layers:
        if raw.get("exp_name"):
            merged["exp_name"] = raw["exp_name"]
            break
    else:
        merged.setdefault("exp_name", path.stem)

    for _, raw in layers:
        if raw.get("work_dir"):
            merged["work_dir"] = raw["work_dir"]
            break
    merged.setdefault("work_dir", f"./work_dirs/{merged['exp_name']}")
    merged.setdefault("pipeline", [])
    if not merged["pipeline"]:
        raise ValueError(f"no pipeline after resolving workflow chain from {path}")
    merged["_stages"] = _normalize_stages(merged["pipeline"])
    merged["_config_path"] = str(path)
    return merged


def resolve_path(p: str | Path, base: Path | None = None) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    if base is None:
        return path.resolve()
    return (base / path).resolve()
