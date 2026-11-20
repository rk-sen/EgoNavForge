"""Deterministic room shell from the calibration bbox + SCENE_PLAN architecture.

v5 left shell construction to the model; each run spent 20-40 steps re-deriving the same rectangle,
and door/window counts drifted. Layout is not creative work: the walking path + bbox + the plan's
`architecture` already determine where the walls, openings and lights go. This module turns that
into one call (`build_shell`) modelled on vln_toolkit's fixed procedural room (no randomness).

Conventions (enforced by bs_qa):
  * Z-up, metres, walking surface top at z=0, thickness 0.20
  * walls are solid boxes with real holes; every hole gets a frame
  * `Window_*` / `Door_*` / `Wall_*` / `Ceiling` / `Floor` names are the ones QA looks for
"""
import json
import math
import re
import sys
from pathlib import Path

import bpy

LIB = Path(__file__).resolve().parent
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import bs_geo    # noqa: E402
import bs_mat    # noqa: E402
import bs_stage  # noqa: E402
import bs_floorplan  # noqa: E402


SHELL_PREFIXES = ("Wall", "Ceiling", "Floor", "Door", "Window", "WindowProxy", "Proxy",
                  "STAGE_Extent", "STAGE_PathGuide", "Shell", "Blind", "Curtain", "Mullion", "LampStem", "Light_Room",
                  "Light_Window", "Sun_Baseline")


def _calib(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _plan(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _n_from_text(text, default):
    m = re.search(r"([0-9]+|[一二三四五六七八九十]+)\s*(扇|个|樘|条|排)", str(text))
    if not m:
        return default
    token = m.group(1)
    zh = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return int(token) if token.isdigit() else zh.get(token, default)


def _window_spec(plan):
    """Accept structured entries first, then fall back to counting Chinese numerals in prose."""
    arch = plan.get("architecture") or {}
    wins = arch.get("windows") or []
    if wins and isinstance(wins[0], dict):
        w = wins[0]
        return {"count": int(w.get("count") or len(wins) or 3),
                "width": float(w.get("width_m") or w.get("width") or 1.2),
                "height": float(w.get("height_m") or w.get("height") or 1.4),
                "z0": float(w.get("z0_m") or 1.0),
                "blinds": bool(w.get("blinds") or w.get("blind")),
                "wall": (w.get("wall") or "E").upper()[:1]}
    if "windows" in arch and not wins:
        return {"count": 0, "width": 1.2, "height": 1.4, "z0": 1.0, "blinds": False,
                "wall": "E"}   # explicitly none
    text = " ".join(str(x) for x in wins) if wins else ""
    return {"count": _n_from_text(text, 3), "width": 1.2, "height": 1.4, "z0": 1.0,
            "blinds": "百叶" in text or "blind" in text.lower(),
            "wall": "E" if "东" in text or "east" in text.lower() else "E"}


def _door_spec(plan):
    arch = plan.get("architecture") or {}
    doors = arch.get("doors") or []
    if doors and isinstance(doors[0], dict):
        d = doors[0]
        return {"count": int(d.get("count") or 1), "width": float(d.get("width_m") or 0.95),
                "height": float(d.get("height_m") or 2.05), "wall": (d.get("wall") or "N").upper()[:1],
                "u": float(d.get("u_m") or 0.0)}
    if "doors" in arch and not doors:
        return {"count": 0, "width": 0.95, "height": 2.05, "wall": "N", "u": 0.0}
    text = " ".join(str(x) for x in doors)
    wall = "N"
    for key, val in (("北", "N"), ("南", "S"), ("东", "E"), ("西", "W")):
        if key in text:
            wall = val
    return {"count": _n_from_text(text, 1), "width": 0.95, "height": 2.05, "wall": wall, "u": 0.0}


def _clear_shell():
    for o in list(bpy.data.objects):
        if o.name.startswith(SHELL_PREFIXES):
            bpy.data.objects.remove(o, do_unlink=True)


def _mats(arch=None):
    arch = arch or {}
    floor_spec = arch.get("floor") or {"style": "rubber", "colors": [[0.235, 0.235, 0.225]], "tile_m": 0.6}
    floor = bs_mat.floor_pattern("M_Shell_Floor", floor_spec.get("style", "rubber"),
                                floor_spec.get("colors") or [(0.235, 0.235, 0.225)],
                                floor_spec.get("tile_m", 0.6))
    wall = bs_mat.principled("M_Shell_Wall", (0.80, 0.79, 0.76, 1), rough=0.72)
    ceil = bs_mat.principled("M_Shell_Ceiling", (0.86, 0.86, 0.84, 1), rough=0.85)
    frame = bs_mat.principled("M_Shell_Frame", (0.45, 0.45, 0.46, 1), rough=0.45, metal=0.6)
    glass = bs_mat.glass("M_Shell_Glass", tint=(0.88, 0.94, 0.98, 1), rough=0.05)
    proxy = bs_mat.principled("M_Proxy_Sky", (0.74, 0.82, 0.92, 1), rough=0.9)
    door = bs_mat.principled("M_Shell_Door", (0.55, 0.42, 0.30, 1), rough=0.55)
    light = bs_mat.emissive("M_Shell_Lamp", (1.0, 0.96, 0.88, 1), strength=1.6)
    return floor, wall, ceil, frame, glass, proxy, door, light


def _wall_side(name, side, b, h, t, holes, mat):
    x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
    if side == "N":
        return bs_geo.wall(name, (x0, y1), (x1, y1), h, t, holes=holes, mat=mat)
    if side == "S":
        return bs_geo.wall(name, (x1, y0), (x0, y0), h, t, holes=holes, mat=mat)
    if side == "E":
        return bs_geo.wall(name, (x1, y0), (x1, y1), h, t, holes=holes, mat=mat)
    return bs_geo.wall(name, (x0, y1), (x0, y0), h, t, holes=holes, mat=mat)


def _side_len(side, b):
    return (b[1] - b[0]) if side in "NS" else (b[3] - b[2])


def build(calib_path, plan_path, blend_path, preview_dir, res=640, render=True):
    """Open/build the V10 shell from plan.architecture (polygon footprint + all openings)."""
    cal, plan = _calib(calib_path), _plan(plan_path)
    arch = plan.get("architecture") or {}
    height = float(arch.get("ceiling_z") or arch.get("ceiling_z_m") or arch.get("height_m")
                   or cal.get("ceiling_m") or 2.7)

    # Always start from a factory-empty scene.  The old "open existing scene.blend" path
    # preserved the previous asset objects when a project was rebuilt, so duplicate barriers
    # from an older generation appeared to be piled on top of the new group.  Shell and
    # assets must be rebuilt from the current plan, never appended to stale geometry.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bs_stage.ensure_world()
    fp, _ = bs_floorplan.footprint(plan, cal)
    rb = list(bs_floorplan.bbox(fp)) + [height]
    for c in bs_stage.cameras_from_calib(cal, room_bbox=rb, room_polygon=fp):
        pass

    _clear_shell()
    # Camera poses are shell infrastructure: refresh them on every build so an existing .blend
    # never keeps a stale Cam_Showcase from the previous footprint (that is how a final render
    # ended up staring at the back of a barrier).
    fp, _ = bs_floorplan.footprint(plan, cal)
    rb = list(bs_floorplan.bbox(fp)) + [height]
    for c in bs_stage.cameras_from_calib(cal, room_bbox=rb, room_polygon=fp):
        pass
    mats = dict(zip(("floor", "wall", "ceil", "frame", "glass", "proxy", "door", "light"),
                   _mats(plan.get("architecture") or {})))
    info = bs_floorplan.build_shell(plan, cal, mats, height, proxy=True)
    bs_stage.baseline_sun()
    removed = bs_stage.clean_duplicates()
    bs_stage.save(blend_path)

    shots = []
    if render:
        shots.append(bs_stage.render_cam("Cam_Overview", Path(preview_dir) / "shell_overview.png", res,
                                         hide=("Ceiling", "Ceiling_Lamp", "INF_Ceiling")))
        for cam in ("Cam_Traj_01", "Cam_Traj_03"):
            if bpy.data.objects.get(cam):
                shots.append(bs_stage.render_cam(cam, Path(preview_dir) / f"shell_{cam}.png", res))
    out = {"ok": True, "blend": str(blend_path), "windows": info["windows"], "doors": info["doors"],
           "walls": info["walls"], "height": height, "footprint": info["footprint"],
           "footprint_source": info["source"], "bbox": [round(v, 2) for v in info["bbox"]],
           "openings": info["openings"], "duplicates_removed": removed, "shots": shots,
           "meshes": len([o for o in bpy.context.scene.objects if o.type == "MESH"]),
           "cameras": sorted(o.name for o in bpy.context.scene.objects if o.type == "CAMERA")}
    print("SHELL_JSON:" + json.dumps(out, ensure_ascii=False))
    try:
        stage = Path(preview_dir) / "_stage"
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "shell.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf8")
    except Exception as exc:
        print("SHELL_JSON_WRITE_WARN:", repr(exc))
    return out
