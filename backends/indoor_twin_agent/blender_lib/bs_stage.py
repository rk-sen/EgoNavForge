"""Stage scaffolding that the executor owns — not the model.

Why this exists (v5): in the v4.x runs the model was told to "write build_cameras.py" and never
produced a camera, so no stage could render, so nothing could be compared. Cameras and the
walking surface are *infrastructure*, not modelling work. This module creates them once:

    init()      → scene.blend with Floor (top z=0), Cam_Traj (animated along calibration.camera_path),
                  Cam_Overview (aerial, sees the whole room), baseline world + sun, duplicate cleanup
    snapshot()  → open scene.blend, clean duplicates, render overview + two trajectory poses
    audit()     → per-material wiring + per-object material assignment + duplicate report

Runs INSIDE Blender only. Absolute paths only (a relative save is how v4c ended up with two blends).
"""
import json
import math

import bs_trajcoord
import os
import shutil
import sys
from pathlib import Path

import bpy
from mathutils import Vector

LIB = Path(__file__).resolve().parent
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import bs_geo  # noqa: E402
import bs_mat   # noqa: E402


# --------------------------------------------------------------------------- helpers
def _calib(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _bbox(cal):
    b = cal.get("room_bbox_m") or [-4.0, 4.0, -4.0, 4.0, 2.8]
    return [float(v) for v in b]


def _purge():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)



def _safe_cam_pos(poly, p, margin=0.8):
    """Keep static QA cameras off the wall shell, not just inside the polygon.

    A point 5 cm inside the L-corner is technically inside the footprint but its frustum is
    swallowed by the wall mesh. Pull it toward the polygon centroid until an 0.8 m ring is
    inside as well.
    """
    if not poly:
        return p
    cx = sum(float(q[0]) for q in poly) / len(poly)
    cy = sum(float(q[1]) for q in poly) / len(poly)
    for _ in range(24):
        ok = True
        for a in range(8):
            ang = 2 * math.pi * a / 8
            if not _point_in_poly(p[0] + margin * math.cos(ang),
                                  p[1] + margin * math.sin(ang), poly):
                ok = False; break
        if ok:
            break
        p[0] = p[0] * .86 + cx * .14
        p[1] = p[1] * .86 + cy * .14
    return p

def _look_euler(loc, target):
    d = Vector(target) - Vector(loc)
    return d.to_track_quat("-Z", "Y").to_euler()


def ensure_world():
    sc = bpy.context.scene
    if not sc.world:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.22, 0.26, 0.32, 1)
        bg.inputs["Strength"].default_value = 0.70
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        sc.eevee.taa_render_samples = 16
    except Exception:
        pass
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.film_transparent = False


def baseline_sun():
    if bpy.data.objects.get("Sun_Baseline"):
        return
    light = bpy.data.lights.new("Sun_Baseline", "SUN")
    light.energy = 1.4
    light.angle = math.radians(18)
    obj = bpy.data.objects.new("Sun_Baseline", light)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = (math.radians(52), 0, math.radians(35))


def floor_from_bbox(cal, name="Floor", floor_spec=None):
    b = _bbox(cal)
    x0, x1, y0, y1, ztop = b[0], b[1], b[2], b[3], b[4]
    # Floor is plan-driven: dark rubber, yellow/blue interlocking foam, timber, tile... all data.
    spec = floor_spec or {"style": "rubber", "colors": [[0.15, 0.15, 0.14]], "tile_m": 0.6}
    mat = bs_mat.floor_pattern("M_Floor_Base", spec.get("style", "rubber"),
                               spec.get("colors") or [(0.15, 0.15, 0.14)],
                               spec.get("tile_m", 0.6))
    return bs_geo.floor_slab(name, x0, x1, y0, y1, thickness=0.2, top_z=0.0,
                            coll_name="Architecture", mat=mat)


def guide_path(cal, name="STAGE_PathGuide"):
    """Orange tube along the walking path + start/end posts. Makes the overview readable during
    S2-S3 and gives the model a physical 6 m reference; the shell stage may delete it."""
    samples = cal.get("camera_path") or []
    pts = [(float(p["pos"][0]), float(p["pos"][1]), 0.05) for p in samples if p.get("pos")]
    if len(pts) < 2:
        return None
    mat = bs_mat.emissive("M_StageGuide", (1.0, 0.45, 0.10, 1), strength=1.5)
    try:
        tube = bs_geo.tube(name, pts, radius=0.035, seg=8, coll_name="Stage", mat=mat, smooth=True)
    except Exception:
        tube = None
    for tag, p in (("Start", pts[0]), ("End", pts[-1])):
        bs_geo.box(f"{name}_{tag}", (0.16, 0.16, 0.9), loc=(p[0], p[1], 0.45), coll_name="Stage", mat=mat)
    b = _bbox(cal)
    wall_mat = bs_mat.principled("M_StageExtent", (0.42, 0.44, 0.46, 1), rough=0.8)
    x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
    for nm, size, loc in (
            ("STAGE_Extent_N", (x1 - x0, 0.12, 0.6), ((x0 + x1) / 2, y1, 0.3)),
            ("STAGE_Extent_S", (x1 - x0, 0.12, 0.6), ((x0 + x1) / 2, y0, 0.3)),
            ("STAGE_Extent_E", (0.12, y1 - y0, 0.6), (x1, (y0 + y1) / 2, 0.3)),
            ("STAGE_Extent_W", (0.12, y1 - y0, 0.6), (x0, (y0 + y1) / 2, 0.3))):
        bs_geo.box(nm, size, loc=loc, coll_name="Stage", mat=wall_mat)
    return tube


def _point_in_poly(x, y, poly):
    if not poly or len(poly) < 3:
        return True
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _pull_inside(p, poly):
    if not poly or _point_in_poly(p[0], p[1], poly):
        return p
    cx = sum(float(q[0]) for q in poly) / len(poly)
    cy = sum(float(q[1]) for q in poly) / len(poly)
    for _ in range(14):
        p[0] = p[0] * 0.72 + cx * 0.28
        p[1] = p[1] * 0.72 + cy * 0.28
        if _point_in_poly(p[0], p[1], poly):
            break
    return p


def cameras_from_calib(cal, max_traj=6, room_bbox=None, room_polygon=None, asset_centers=None):
    """Overview camera + up to 6 static trajectory cameras.

    PI3 camera_path is in a gravity-aligned frame. bs_trajcoord rotates positions and yaw into
    Blender Z-up and normalizes them into the plan footprint; using aligned x/y directly made
    cameras stare into walls. Cameras are infrastructure owned by the stage, not by the model.
    """
    made = []
    for o in [x for x in bpy.data.objects if x.type == "CAMERA"]:
        bpy.data.objects.remove(o, do_unlink=True)
    lens = float(cal.get("lens_mm") or 18.0)
    rb = list(room_bbox[:4]) if room_bbox and len(room_bbox) >= 4 else None
    samples = cal.get("camera_path") or []
    ts = bs_trajcoord.room_samples(cal, rb or _bbox(cal)[:4], room_polygon) if samples else []
    if ts:
        n = min(max_traj, len(ts))
        step = max(1, len(ts) // n)
        picked = ts[::step][:n]
        for i, q in enumerate(picked, start=1):
            pos = [float(v) for v in q["pos"]]
            _safe_cam_pos(room_polygon, pos)
            z = max(0.95, min(1.75, pos[2]))
            d = Vector(tuple(q.get("dir") or (math.cos(math.radians(q["yaw"])),
                                               math.sin(math.radians(q["yaw"])), 0.0)))
            if asset_centers and d.length > 1e-5:
                d.normalize()
                front = [c for c in asset_centers
                         if (Vector((c[0]-pos[0], c[1]-pos[1], c[2]-z)).dot(d) > 1.0)]
                seen = any(Vector((c[0]-pos[0], c[1]-pos[1], c[2]-z)).length < 8 and
                           Vector((c[0]-pos[0], c[1]-pos[1], c[2]-z)).normalized().angle(d) < math.radians(42)
                           for c in front)
                if not seen and front:
                    c = min(front, key=lambda c: math.hypot(c[0]-pos[0], c[1]-pos[1]))
                    d = Vector((c[0]-pos[0], c[1]-pos[1], c[2]-z))
            if d.length < 1e-3:
                d = Vector((1, 0, 0))
            cd = bpy.data.cameras.new(f"Cam_Traj_{i:02d}")
            cd.lens = max(10.0, lens)
            cd.sensor_width = 36.0
            cam = bpy.data.objects.new(f"Cam_Traj_{i:02d}", cd)
            bpy.context.collection.objects.link(cam)
            cam.location = (pos[0], pos[1], z)
            cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
            made.append(cam)
    b = list(room_bbox[:4]) if room_bbox and len(room_bbox) >= 4 else list(_bbox(cal))
    if len(b) >= 4:
        b = [b[0], b[1], b[2], b[3], (room_bbox[4] if room_bbox and len(room_bbox) > 4 else _bbox(cal)[4])]
    cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
    diag = max(2.0, math.hypot(b[1] - b[0], b[3] - b[2]))
    loc = (cx + 0.10 * diag, cy - 0.12 * diag, max(b[4], 0.5) + 1.15 * diag)
    sd = bpy.data.cameras.new("Cam_Showcase")
    sd.lens = float(lens or 15.1)
    sd.sensor_width = 36.0
    show = bpy.data.objects.new("Cam_Showcase", sd)
    bpy.context.collection.objects.link(show)
    show.location = (b[0] + 0.12 * (b[1] - b[0]), b[2] + 0.16 * (b[3] - b[2]), 1.35)
    show.rotation_euler = _look_euler(show.location,
                                      (b[1] - 0.18 * (b[1] - b[0]), b[3] - 0.12 * (b[3] - b[2]), 0.85))
    made.append(show)
    od = bpy.data.cameras.new("Cam_Overview")
    od.lens = 26.0
    od.sensor_width = 36.0
    over = bpy.data.objects.new("Cam_Overview", od)
    bpy.context.collection.objects.link(over)
    over.location = loc
    over.rotation_euler = _look_euler(loc, (cx, cy, 0.95))
    bpy.context.scene.camera = over
    made.append(over)
    return made

def clean_duplicates(tol=0.02, verbose=True):
    """Delete identical duplicates (v4.4 shipped 36 pairs: entire asset sets copied at same spot).

    Two meshes are duplicates when rounded dimensions match, world centres are within `tol`,
    and the name (minus .001 suffix) matches. Keeps the first, removes the rest.
    """
    removed = []
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    keep = []
    for o in meshes:
        dims = tuple(round(v, 2) for v in o.dimensions)
        base = o.name.split(".00")[0].split(".0")[0]
        dup = None
        for k in keep:
            if tuple(round(v, 2) for v in k.dimensions) != dims:
                continue
            if (k.matrix_world.translation - o.matrix_world.translation).length > tol:
                continue
            if k.name.split(".00")[0].split(".0")[0] != base:
                continue
            dup = k
            break
        if dup is not None:
            removed.append(o.name)
            bpy.data.objects.remove(o, do_unlink=True)
        else:
            keep.append(o)
    if verbose and removed:
        print(f"DUPLICATES_REMOVED {len(removed)}: {', '.join(removed[:8])}")
    return removed


def save(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(p))
    return str(p)


def render_cam(cam_name, out_path, res=640, hide=()):
    """Render one camera. `hide` temporarily disables render for the given object-name prefixes
    (the aerial overview must hide Ceiling, otherwise it photographs the roof)."""
    sc = bpy.context.scene
    cam = bpy.data.objects.get(cam_name)
    if not cam:
        return {"ok": False, "error": f"camera {cam_name} missing"}
    sc.camera = cam
    sc.render.resolution_x = int(res)
    sc.render.resolution_y = int(res * 0.75)
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sc.render.filepath = str(Path(out_path).resolve())
    restored = []
    for o in bpy.data.objects:
        if any(o.name.startswith(p) for p in hide) and not o.hide_render:
            o.hide_render = True
            restored.append(o)
    try:
        bpy.ops.render.render(write_still=True)
    finally:
        for o in restored:
            o.hide_render = False
    return {"ok": Path(out_path).exists(), "path": str(Path(out_path).resolve()), "camera": cam_name}


def init(calib_path, blend_path, preview_path, rebuild=False):
    out = {"ok": True, "cameras": [], "duplicates_removed": []}
    cal = None
    if Path(blend_path).exists() and not rebuild:
        bpy.ops.wm.open_mainfile(filepath=str(Path(blend_path).resolve()))
        out["reused"] = True
    else:
        _purge()
        ensure_world()
        baseline_sun()
        cal = _calib(calib_path)
        # Plan architecture wins over trajectory-derived calibration for room size. The trajectory
        # still supplies camera poses only; it must not define the floor rectangle.
        try:
            plan = json.loads(Path(calib_path).with_name("SCENE_PLAN.json").read_text())
            arch = plan.get("architecture") or {}
            box = arch.get("room_bbox_m")
            fp = arch.get("footprint_m") or arch.get("floorplan_m") or arch.get("outline_m")
            if isinstance(fp, list) and len(fp) >= 3:
                xs = [float(q[0]) for q in fp]; ys = [float(q[1]) for q in fp]
                box = [min(xs), max(xs), min(ys), max(ys)]
            if box and len(box) >= 4:
                cal["room_bbox_m"] = [float(v) for v in box[:4]] + [float(arch.get("ceiling_z") or arch.get("ceiling_z_m") or cal.get("ceiling_m") or 2.7)]
            if isinstance(fp, list) and len(fp) >= 3:
                cal["room_polygon"] = [[float(q[0]), float(q[1])] for q in fp]
        except Exception:
            pass
        # a place to walk/stand, not a "modelled room" — the model replaces/extends it in S3
        floor_spec = None
        try:
            floor_spec = (json.loads(Path(calib_path).with_name("SCENE_PLAN.json").read_text())
                          .get("architecture", {}).get("floor"))
        except Exception:
            pass
        floor_from_bbox(cal, floor_spec=floor_spec)
        guide_path(cal)
        out["fresh"] = True
    if cal is None:
        cal = _calib(calib_path)
    rb = cal.get("room_bbox_m")
    asset_centers = []
    try:
        plan = json.loads(Path(calib_path).with_name("SCENE_PLAN.json").read_text())
        for o in (list(plan.get("objects") or []) + list(plan.get("inferred") or [])):
            h = (list(o.get("size_m") or [.5, .5, .5]) + [.5])[2]
            for it in (o.get("instances") or []):
                asset_centers.append((float(it.get("x",0)), float(it.get("y",0)), float(h)*.55))
            p0 = o.get("place")
            if isinstance(p0, dict) and p0.get("x") is not None:
                asset_centers.append((float(p0.get("x",0)), float(p0.get("y",0)), float(h)*.55))
    except Exception:
        pass
    for cam in cameras_from_calib(cal, room_bbox=rb, room_polygon=cal.get("room_polygon"),
                                  asset_centers=asset_centers):
        out["cameras"].append(cam.name)
    out["duplicates_removed"] = clean_duplicates()
    out["blend"] = save(blend_path)
    r = render_cam("Cam_Overview", preview_path, res=640)
    out["preview"] = r
    print("STAGE_JSON:" + json.dumps(out, ensure_ascii=False))
    return out


def snapshot(blend_path, calib_path, out_dir, tag="s", res=640, traj_limit=5):
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend_path).resolve()))
    cal = _calib(calib_path)
    got = {o.name for o in bpy.context.scene.objects if o.type == "CAMERA"}
    if "Cam_Overview" not in got or len([c for c in got if c.startswith("Cam_Traj")]) < 2:
        for cam in cameras_from_calib(cal):
            pass
    removed = clean_duplicates()
    save(blend_path)
    shots = [render_cam("Cam_Overview", Path(out_dir) / f"{tag}_overview.png", res,
                        hide=("Ceiling", "Ceiling_Lamp", "INF_Ceiling"))]
    traj = sorted(c for c in {o.name for o in bpy.context.scene.objects if o.type == "CAMERA"}
                  if c.startswith("Cam_Traj"))[:max(0, int(traj_limit))]
    for i, name in enumerate(traj, start=1):
        shots.append(render_cam(name, Path(out_dir) / f"{tag}_traj{i}.png", res))
    out = {"ok": all(s.get("ok") for s in shots), "shots": shots,
           "duplicates_removed": removed,
           "cameras": sorted({o.name for o in bpy.context.scene.objects if o.type == "CAMERA"})}
    print("SNAP_JSON:" + json.dumps(out, ensure_ascii=False))
    return out


def _asset_parts(pid):
    root = bpy.data.objects.get(pid)
    if root is None:
        return None, []
    if root.type == "MESH":
        return root, [root]
    parts = []

    def walk(o):
        for c in o.children:
            if c.type == "MESH":
                parts.append(c)
            walk(c)

    walk(root)
    return root, parts


def _world_bbox(parts):
    pts = []
    for o in parts:
        for c in o.bound_box:
            pts.append(o.matrix_world @ Vector(c))
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def coverage(plan_path):
    """How many roots exist per planned id (`id` and `id__02` both count as one instance)."""
    plan = json.loads(Path(plan_path).read_text())
    items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
    names = [o.name for o in bpy.context.scene.objects if o.type == "EMPTY"]
    out = []
    for it in items:
        pid = str(it.get("id") or "")
        want = int(it.get("count") or 1)
        got = [n for n in names if n == pid or n.startswith(pid + "__") or n == pid + ".001"]
        out.append({"id": pid, "want": want, "got": len(got), "roots": sorted(got)[:12]})
    missing = [r["id"] for r in out if r["got"] == 0]
    short = [r["id"] for r in out if 0 < r["got"] < r["want"]]
    extra = [r["id"] for r in out if r["got"] > r["want"]]
    info = {"ok": not missing and not short and not extra, "objects": out,
            "missing": missing, "short": short, "extra": extra}
    print("COVERAGE_JSON:" + json.dumps(info, ensure_ascii=False))
    return info


def placement_report(plan_path):
    """Check that asset AABBs do not interpenetrate at the same height; `on` support is allowed."""
    plan = json.loads(Path(plan_path).read_text())
    items = {str(x.get("id")): x for x in list(plan.get("objects") or []) + list(plan.get("inferred") or [])}
    assets = []
    for o in bpy.context.scene.objects:
        if o.type != "EMPTY" or o.name.split("__")[0] not in items:
            continue
        parts = [c for c in o.children if c.type == "MESH"]
        if not parts:
            continue
        lo, hi = _world_bbox(parts)
        assets.append({"id": o.name, "box": [float(lo.x), float(hi.x), float(lo.y),
                                                           float(hi.y), float(lo.z), float(hi.z)],
                       "relation": o.get("v10_relation", "{}")})
        # clones keep the same v9_relation on the Empty root
    overlaps = []
    for i in range(len(assets)):
        a = assets[i]
        for j in range(i + 1, len(assets)):
            b = assets[j]
            ba, bb = a["box"], b["box"]
            ox = min(ba[1], bb[1]) - max(ba[0], bb[0])
            oy = min(ba[3], bb[3]) - max(ba[2], bb[2])
            oz = min(ba[5], bb[5]) - max(ba[4], bb[4])
            if ox > 0.03 and oy > 0.03 and oz > 0.03:
                try:
                    ra, rb = json.loads(a["relation"] or "{}"), json.loads(b["relation"] or "{}")
                except Exception:
                    ra, rb = {}, {}
                if (ra.get("target") == b["id"] and ra.get("type") == "on") or                    (rb.get("target") == a["id"] and rb.get("type") == "on"):
                    continue
                overlaps.append({"a": a["id"], "b": b["id"],
                                 "overlap_m": [round(ox, 2), round(oy, 2), round(oz, 2)]})
    # The user accepts a small proximity overlap (two 2.5 m barrier panels standing side by
    # side) but not a sofa inside a desk. Flag only penetration deeper than 15 cm on every
    # axis; record the rest as warnings instead of failing the whole build.
    # Penetration is the SMALLEST axis overlap: a 1.2 m x 0.17 m shelf/barrier contact is
    # a 17 cm deep intersection, while a 0.1 m x 0.8 m contact is only a 10 cm skin overlap.
    hard = [o for o in overlaps if min(o["overlap_m"]) >= 0.15]
    minor = [o for o in overlaps if min(o["overlap_m"]) < 0.15]
    out = {"ok": not hard, "assets": len(assets), "overlaps": hard[:20],
           "minor_overlaps": minor[:20], "policy": "hard = min axis penetration >= 0.15 m"}
    print("PLACEMENT_JSON:" + json.dumps(out, ensure_ascii=False))
    return out


def inspect_assets(blend_path, plan_ids, out_dir, res=480, samples=4):
    """One temporary camera per planned id: frame it, render it, delete the camera.

    This is the object-level check the v7 scaffold never had. It also reports whether the asset
    is a single joined mesh or a pile of loose parts.
    """
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend_path).resolve()))
    scene = bpy.context.scene
    old_cam = scene.camera
    scene.render.resolution_x = res
    scene.render.resolution_y = max(1, int(res * 0.75))
    scene.render.resolution_percentage = 100
    # Object checks are geometry checks, not beauty renders: Workbench is seconds per item on CPU.
    old_engine = scene.render.engine
    old_samples = None
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        sh = scene.display.shading
        sh.light = "STUDIO"
        sh.color_type = "MATERIAL"
        sh.show_shadows = True
        sh.show_cavity = True
    except Exception:
        pass
    if hasattr(scene, "cycles"):
        old_samples = scene.cycles.samples
        scene.cycles.samples = int(samples)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in plan_ids:
        root, parts = _asset_parts(pid)
        row = {"id": pid, "ok": False, "path": "", "meshes": len(parts)}
        if root is None or not parts:
            row["error"] = "object not found / no mesh" if root is None else "empty asset root"
            rows.append(row)
            continue
        lo, hi = _world_bbox(parts)
        size = hi - lo
        center = (lo + hi) / 2
        dist = max(size.x, size.y, size.z) * 1.9 + 1.0
        cam_data = bpy.data.cameras.new("__InspectCam")
        cam = bpy.data.objects.new("__InspectCam", cam_data)
        scene.collection.objects.link(cam)
        cam.location = center + Vector((dist * 0.85, -dist * 0.85, dist * 0.55))
        bs_geo.look_at(cam, center)
        cam_data.lens = 50
        hidden = []
        for o in scene.objects:
            if o.type == "MESH" and o not in parts:
                o.hide_render = True
                hidden.append(o)
        scene.camera = cam
        path = out / (pid.replace("/", "_") + ".png")
        scene.render.filepath = str(path)
        try:
            bpy.ops.render.render(write_still=True)
            row.update({"ok": path.exists(), "path": str(path), "meshes": len(parts),
                        "size_m": [round(float(v), 2) for v in size],
                        "joined": len(parts) == 1})
        except Exception as exc:
            row["error"] = str(exc)[:300]
        finally:
            for o in hidden:
                o.hide_render = False
            bpy.data.objects.remove(cam, do_unlink=True)
            bpy.data.cameras.remove(cam_data)
            scene.camera = old_cam
        rows.append(row)
    scene.camera = old_cam
    scene.render.engine = old_engine
    if old_samples is not None:
        scene.cycles.samples = old_samples
    save(blend_path)
    info = {"ok": all(r.get("ok") and r.get("meshes") == 1 for r in rows), "objects": rows,
            "count": len(rows)}
    print("ASSET_JSON:" + json.dumps(info, ensure_ascii=False))
    return info


def finalize(blend_path, out_dir, res=640, tag="build", traj_limit=5):
    """Save + clean duplicates + one overview + five trajectory views, in-place after a build script."""
    cleaned = clean_duplicates()
    # The orange path guide / extent boxes / exterior proxies are *debug aids*; final scene must
    # not ship them (the user rejected the trajectory-driven perimeter look).
    for o in list(bpy.data.objects):
        if any(o.name.startswith(p) for p in ("STAGE_PathGuide", "STAGE_Extent",
                                              "WindowProxy", "Proxy")):
            bpy.data.objects.remove(o, do_unlink=True)
    save(blend_path)
    shots = [render_cam("Cam_Overview", Path(out_dir) / f"{tag}_overview.png", res,
                        hide=("Ceiling", "Ceiling_Lamp", "INF_Ceiling"))]
    traj = sorted(c for c in {o.name for o in bpy.context.scene.objects if o.type == "CAMERA"}
                  if c.startswith("Cam_Traj"))[:max(0, int(traj_limit))]
    for i, name in enumerate(traj, start=1):
        shots.append(render_cam(name, Path(out_dir) / f"{tag}_traj{i}.png", res))
    out = {"ok": all(s.get("ok") for s in shots), "shots": shots,
           "duplicates_removed": len(cleaned),
           "meshes": len([o for o in bpy.context.scene.objects if o.type == "MESH"]),
           "cameras": sorted({o.name for o in bpy.context.scene.objects if o.type == "CAMERA"}),
           "objects": len(bpy.context.scene.objects)}
    print("BUILD_JSON:" + json.dumps(out, ensure_ascii=False))
    return out


def audit(blend_path):
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend_path).resolve()))
    bpy.context.view_layer.update()
    mats = []
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        n = m.node_tree.nodes.get("Principled BSDF")
        base_linked = bool(n and n.inputs["Base Color"].is_linked)
        imgs = [x for x in m.node_tree.nodes if x.type == "TEX_IMAGE" and x.image]
        users = [o.name for o in bpy.context.scene.objects if o.type == "MESH"
                 and any(s.material and s.material.name == m.name for s in o.material_slots)]
        mats.append({"name": m.name, "base_linked": base_linked, "images": len(imgs),
                     "objects": users[:8], "n_objects": len(users),
                     "base": [round(v, 2) for v in (n.inputs["Base Color"].default_value[:3] if n else [])]})
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    no_mat = [o.name for o in meshes if not any(s.material for s in o.material_slots)]
    unused = [m["name"] for m in mats if m["n_objects"] == 0]
    out = {"ok": True, "materials": mats, "meshes": len(meshes),
           "no_material": no_mat[:12], "unused_materials": unused[:12],
           "duplicates": [o.name for o in meshes if ".00" in o.name][:12]}
    print("AUDIT_JSON:" + json.dumps(out, ensure_ascii=False))
    return out


def _roots_for(pid):
    """Instance empties for a plan id (O1, O1__02, O1.001 handled)."""
    out = []
    for o in bpy.context.scene.objects:
        if o.type != "EMPTY":
            continue
        n = o.name
        if n == pid or n.startswith(pid + "__") or n.startswith(pid + "."):
            out.append(o)
    return out


def _group_bbox(root):
    zs, xs, ys = [], [], []
    if root.type == "MESH":
        objs = [root]
    else:
        objs = [o for o in bpy.context.scene.objects if o.type == "MESH" and
                (o.parent == root or o.name.startswith(root.name + "_"))]
    for o in objs:
        for v in o.data.vertices:
            w = o.matrix_world @ v.co
            xs.append(w.x); ys.append(w.y); zs.append(w.z)
    if not zs:
        return None
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def support_report(plan_path):
    """V10 support / floating gate: an object is only valid if it rests on floor or its support.

    The v9 placement_report checked AABB interpenetration only; a monitor could float at eye
    height and `on` relations were exempt. This is a separate, explicit check.
    """
    plan = json.loads(Path(plan_path).read_text())
    items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
    by_id = {str(it.get("id")): it for it in items if it.get("id")}
    # target tops are measured from built geometry, not from the plan
    tops = {}
    for pid in by_id:
        roots = _roots_for(pid)
        bbs = [_group_bbox(r) for r in roots]
        bbs = [b for b in bbs if b]
        tops[pid] = max(b[5] for b in bbs) if bbs else 0.0
    exempt_kind = {"window", "window_frame", "blind", "blind", "light", "ceiling_light", "door",
                   "door_frame", "air_conditioner", "ac", "lamp", "desk_lamp"}
    exempt_rel = {"on"}  # on is checked by its target top below
    failures, checked = [], 0
    for it in items:
        pid = str(it.get("id") or "")
        if not pid:
            continue
        kind = str(it.get("kind") or "").lower()
        rel = it.get("relation") or {}
        typ = str(rel.get("type") or "").lower()
        target = str(rel.get("target") or "")
        if kind in exempt_kind or typ in ("against",):
            # against a wall still must not be under the floor; enforce z only
            pass
        roots = _roots_for(pid)
        if not roots:
            continue
        checked += 1
        for root in roots:
            bb = _group_bbox(root)
            if not bb:
                continue
            zmin, zmax = bb[4], bb[5]
            if zmin < -0.045:
                failures.append({"id": pid, "root": root.name, "type": "below_floor",
                                 "z_min": round(zmin, 3)})
            elif typ == "on" and target in by_id:
                top = tops.get(target)
                # allow the object's own bottom to sit up to 6 cm above/inside the target top
                if top is not None and abs(zmin - top) > 0.065:
                    failures.append({"id": pid, "root": root.name, "type": "not_on_support",
                                     "z_min": round(zmin, 3), "target": target,
                                     "target_top": round(top, 3)})
            elif zmin > 0.045 and kind not in exempt_kind and typ not in ("on", "against"):
                failures.append({"id": pid, "root": root.name, "type": "floating",
                                 "z_min": round(zmin, 3)})
            if zmax < 0.035 and kind not in exempt_kind:
                failures.append({"id": pid, "root": root.name, "type": "no_height",
                                 "z_max": round(zmax, 3)})
    out = {"ok": not failures, "checked": checked, "failures": failures[:24],
           "summary": f"{len(failures)} support issue(s) across {checked} id(s)"}
    print("SUPPORT_JSON:" + json.dumps(out, ensure_ascii=False))
    return out
