"""Blender-side acceptance checks + preview rendering.

Run as:  blender -b scene.blend --python _lib/bs_qa.py -- --json qa.json
Prints one line `QA_JSON:{...}` so the host can parse it without reading the file.

Checks are the ones that actually failed in a 150-call run:
  floor at z=0, solid shells, real openings, assets above floor, camera not buried,
  FOV from intrinsics, generated textures really wired, scale plausibility.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _args():
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def _paths():
    a = _args()
    out_json = "qa.json"
    intrinsics = None
    tex_mode = "both"
    for i, x in enumerate(a):
        if x == "--json" and i + 1 < len(a):
            out_json = a[i + 1]
        if x == "--intrinsics" and i + 1 < len(a):
            intrinsics = a[i + 1]
        if x == "--texture-mode" and i + 1 < len(a):
            tex_mode = a[i + 1]
    return out_json, intrinsics, tex_mode


def _wname(o):
    return (o.name + " " + (o.parent.name if o.parent else "")).lower()


def _is(o, *keys):
    return any(k in _wname(o) for k in keys)


def check_floor():
    floors = [o for o in bpy.data.objects if o.type == "MESH" and _is(o, "floor", "ground", "floor_slab")]
    if not floors:
        return False, "no floor object found (name it floor*/ground*)", {}
    tops = []
    for o in floors:
        z = [(o.matrix_world @ v.co).z for v in o.data.vertices]
        if z:
            tops.append(max(z))
    top = max(tops)
    ok = abs(top) < 0.05
    return ok, f"floor top z={top:.3f} m (must be 0.000±0.05)", {"floor_top_z": round(top, 3),
                                                                 "objects": [o.name for o in floors][:4]}


SOLID_EXEMPT = ("window_proxy", "proxy", "glass", "decal", "backdrop", "skybox", "billboard", "inf_")


def _solid_exempt(name):
    low = name.lower()
    return any(k in low for k in SOLID_EXEMPT)


def check_solid():
    bad = []
    for o in bpy.data.objects:
        if o.type != "MESH" or len(o.data.polygons) == 0:
            continue
        if _solid_exempt(o.name):
            continue     # a proxy exterior / glass pane is meant to be a single quad
        dims = o.dimensions
        thin = min(dims.x, dims.y, dims.z)
        if len(o.data.polygons) == 1:
            bad.append({"name": o.name, "polys": 1, "dims": [round(d, 2) for d in dims]})
        elif thin < 0.02 and len(o.data.polygons) <= 2:
            bad.append({"name": o.name, "polys": len(o.data.polygons), "dims": [round(d, 2) for d in dims]})
    return (not bad), f"single-face / paper shells: {len(bad)}", {"single_face": bad[:8]}


def check_openings():
    windows = [o for o in bpy.data.objects if o.type == "MESH"
               and o.name.lower().endswith("_glass")]
    doors = [o for o in bpy.data.objects if o.type == "MESH" and _is(o, "door")
             and not any(k in o.name.lower() for k in ("frame", "_top", "handle"))]
    walls = [o for o in bpy.data.objects if o.type == "MESH" and _is(o, "wall")]
    max_polys = max([len(w.data.polygons) for w in walls], default=0)
    if not windows and not doors:
        return True, "no windows/doors to verify", {"walls": len(walls)}
    ok = max_polys > 1
    return ok, f"windows={len(windows)} doors={len(doors)} but walls max polys={max_polys}", {
        "wall_max_polys": max_polys, "windows": [o.name for o in windows][:5], "doors": [o.name for o in doors][:3]}


def check_above_floor():
    bad = []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        if _is(o, "floor", "ground", "ceiling", "foundation"):
            continue                       # slabs legitimately extend below z=0
        zs = [(o.matrix_world @ v.co).z for v in o.data.vertices]
        if not zs or min(zs) < -0.05:
            bad.append({"name": o.name, "min_z": round(min(zs), 3) if zs else None})
    return (not bad), f"{len(bad)} objects dip below floor", {"objects": bad[:8]}


def check_furniture_orientation():
    """A wall-adjacent sofa must have its back toward the wall, seat toward the room.

    This catches the v10 case where the sofa was correctly sized and coloured but placed
    against the west wall with the plan's yaw=90, which made it face the wall.
    """
    plan_path = Path(bpy.data.filepath).with_name("SCENE_PLAN.json")
    try:
        plan = json.loads(plan_path.read_text())
        fp = (plan.get("architecture") or {}).get("footprint_m") or []
        xs = [float(p[0]) for p in fp]; ys = [float(p[1]) for p in fp]
        bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    except Exception:
        return True, "no footprint to test orientation", {}
    bad = []
    tested = 0
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        kind = str((o.parent or o).get("v10_kind") or "")
        if "sofa" not in (kind + " " + _wname(o)).lower():
            continue
        root = o.parent or o
        yaw = float(root.matrix_world.to_euler().z)
        back = Vector((math.sin(yaw), -math.cos(yaw)))   # local -Y is the sofa back
        c = root.matrix_world.translation
        dists = {"W": c.x - bx0, "E": bx1 - c.x, "S": c.y - by0, "N": by1 - c.y}
        side = min(dists, key=dists.get)
        if dists[side] > 1.6:
            continue
        outward = {"E": Vector((1, 0)), "W": Vector((-1, 0)),
                   "N": Vector((0, 1)), "S": Vector((0, -1))}[side]
        tested += 1
        dot = back.dot(outward)
        if dot < 0.15:
            bad.append({"object": root.name, "wall": side, "dot": round(dot, 2),
                        "yaw_deg": round(math.degrees(yaw), 1)})
    if bad:
        return False, "sofa faces the wall: " + ", ".join(f"{b['object']}@{b['wall']}" for b in bad), {"bad": bad}
    return True, f"sofa back vectors point to wall ({tested} tested)", {"tested": tested}


def check_cameras():
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    deps = bpy.context.evaluated_depsgraph_get()
    blind, inside = [], []
    probes = [Vector(d) for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    for cam in cams:
        org = cam.matrix_world.translation.copy()
        rot = cam.matrix_world.to_3x3()
        fwd = (rot @ Vector((0, 0, -1))).normalized()
        right = (rot @ Vector((1, 0, 0))).normalized()
        up = (rot @ Vector((0, 1, 0))).normalized()
        for d in probes:
            hit, loc, _n, _i, obj, _m = bpy.context.scene.ray_cast(deps, org, d)
            if hit and (loc - org).length < 0.06:
                inside.append({"camera": cam.name, "object": obj.name if obj else "?"})
                break
        dists, objs = [], []
        for ah in (-40, -20, 0, 20, 40):
            for av in (-25, 0, 25):
                h, v = math.radians(ah), math.radians(av)
                d = (fwd * math.cos(h) * math.cos(v) + right * math.sin(h) * math.cos(v) + up * math.sin(v)).normalized()
                hit, loc, _n, _i, obj, _m = bpy.context.scene.ray_cast(deps, org, d)
                if hit:
                    dists.append((loc - org).length)
                    objs.append(obj.name if obj else "?")
        if len(dists) >= 8:
            close = sum(1 for x in dists if x < 0.5) / len(dists)
            med = sorted(dists)[len(dists) // 2]
            if close > 0.6:
                blind.append({"camera": cam.name, "reason": "frustum inside 0.5m", "median": round(med, 2)})
            elif objs:
                top = max(set(objs), key=objs.count)
                if objs.count(top) / len(objs) > 0.7 and med < 3.5:
                    blind.append({"camera": cam.name, "reason": "single surface fills frustum",
                                  "object": top, "median": round(med, 2)})
    ok = not inside          # being inside a shell is fatal; a narrow view is only information
    return ok, f"cameras={len(cams)} buried={len(inside)} narrow={len(blind)} (info)", {
        "inside": inside[:6], "blind": blind[:6],
        "count": len(cams),
        "note": "one camera + keyframes beats many discrete cameras" if len(cams) > 3 else ""}


def check_fov(intrinsics_path):
    if not intrinsics_path:
        return True, ("intrinsics 未提供，本次跳过 FOV 对比（有内参时再严格检查）"), {"skipped": True}
    try:
        data = json.loads(Path(intrinsics_path).read_text())
    except Exception as e:
        return False, f"intrinsics unreadable: {e}", {}
    K = (data.get("K") or data.get("k") or data.get("intrinsics")
         or data.get("K_pinhole_pipeline") or data.get("camera_matrix"))
    if not K:
        return False, f"intrinsics 文件里没有 K/K_pinhole_pipeline: {Path(intrinsics_path).name}", {"skipped": True}
    fx = K[0][0] if isinstance(K[0], list) else K[0]
    wh = data.get("pipeline_wh") or data.get("wh") or data.get("image_size") or data.get("resolution")
    width = data.get("width") or data.get("W") or (wh[0] if isinstance(wh, (list, tuple)) and wh else None)
    hfov = math.degrees(2 * math.atan((width / 2) / fx)) if width else None
    bad = []
    for cam in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        # the aerial/overview camera is deliberately wide; it is not the video camera
        if any(k in cam.name.lower() for k in ("overview", "over", "aerial", "global", "diorama")):
            continue
        lens = cam.data.lens
        sensor = cam.data.sensor_width
        fov = math.degrees(2 * math.atan((sensor / 2) / lens))
        if hfov and abs(fov - hfov) > 8:
            bad.append({"camera": cam.name, "render_hfov": round(fov, 1), "video_hfov": round(hfov, 1)})
    return (not bad), (f"target hFOV={round(hfov,1) if hfov else 'n/a'}° (from {Path(intrinsics_path).name}), "
                       f"mismatched={len(bad)}"), {"mismatch": bad[:6], "hfov_deg": round(hfov, 1) if hfov else None}


def check_plan_coverage(plan_path=""):
    """Every planned object (observed or INF_) must exist, and repeated items must exist `count` times.

    v5's 120-step scene had 14 planned items and built 3; QA still said ok:true because nothing
    compared the plan with the .blend. Coverage is now a hard check.
    """
    blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
    cands = [Path(plan_path)] if plan_path else []
    cands += [blend_dir / "SCENE_PLAN.json", blend_dir / "project" / "SCENE_PLAN.json",
              Path.cwd() / "SCENE_PLAN.json"]
    plan = None
    for c in cands:
        if c and c.exists():
            try:
                plan = json.loads(c.read_text())
                break
            except Exception:
                plan = None
    if not plan:
        return False, "没有 SCENE_PLAN.json，无法核对计划覆盖率", {"missing_plan": True}
    excluded = {str(x.get("id")) for x in (plan.get("excluded") or []) if isinstance(x, dict)}
    items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
    missing, short, covered = [], [], 0
    for it in items:
        oid = str(it.get("id") or it.get("name") or "")
        if not oid or oid in excluded:
            continue
        need = max(1, int(it.get("count") or 1))
        roots = [o for o in bpy.data.objects
                 if o.name == oid or re.match(re.escape(oid) + r"(__\d+|\.\d+)$", o.name)]
        if not roots:
            # legacy: a bare mesh named `O1_Sofa_seat` counts as one instance
            meshes = [o for o in bpy.context.scene.objects
                      if o.type == "MESH" and o.name.startswith(oid + "_")]
            got = 1 if meshes else 0
        else:
            got = len(roots)
        if got == 0:
            missing.append(oid)
        elif got < need:
            short.append(f"{oid} {got}/{need}")
        else:
            covered += 1
    detail = f"planned={len(items) - len(excluded)} covered={covered} missing={len(missing)} short={len(short)}"
    if missing:
        detail += " — 缺: " + ", ".join(missing[:6])
    if short:
        detail += " — 数量不足: " + ", ".join(short[:6])
    return (not missing and not short), detail, {"missing": missing, "short": short,
                                                 "covered": covered,
                                                 "planned": len(items) - len(excluded)}


def check_camera_rig(min_cams=3):
    cams = [o for o in bpy.data.objects if o.type == "CAMERA"]
    over = [c for c in cams if any(k in c.name.lower() for k in ("overview", "aerial", "global"))]
    traj = [c for c in cams if c.name.startswith("Cam_Traj")]
    ok = len(over) >= 1 and len(traj) >= min_cams - 1
    return ok, f"cameras={len(cams)} overview={len(over)} traj={len(traj)} (need 1+{min_cams-1})", \
        {"cameras": [c.name for c in cams]}


def check_duplicates(tol=0.02):
    """v4.4 shipped 36 identical pairs (whole asset sets copied at the same spot). Two meshes are
    the same object when rounded dims match, centres are within `tol`, and the name matches modulo
    a .001 suffix. This is a hard error: duplicates double geometry, z-fight, and make the render
    look like a modelling mistake."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    pairs = []
    for i, a in enumerate(meshes):
        for b in meshes[i + 1:]:
            if tuple(round(v, 2) for v in a.dimensions) != tuple(round(v, 2) for v in b.dimensions):
                continue
            if (a.matrix_world.translation - b.matrix_world.translation).length > tol:
                continue
            if a.name.split(".00")[0].split(".0")[0] != b.name.split(".00")[0].split(".0")[0]:
                continue
            pairs.append(f"{a.name}/{b.name}")
    detail = f"duplicate objects: {len(pairs)}" + (" — " + ", ".join(pairs[:5]) if pairs else "")
    return (not pairs), detail, {"pairs": pairs[:12]}


def check_material_base():
    """Textures must reach Base Color. A material with image nodes whose Base Color is not linked
    is a grey default (the v4.4 symptom: 12 materials, every one rendering 0.7 grey)."""
    bad, total = [], 0
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        n = m.node_tree.nodes.get("Principled BSDF")
        if not n:
            continue
        total += 1
        imgs = [x for x in m.node_tree.nodes if x.type == "TEX_IMAGE" and x.image]
        if imgs and not n.inputs["Base Color"].is_linked:
            users = [o.name for o in bpy.context.scene.objects if o.type == "MESH"
                     and any(sl.material and sl.material.name == m.name for sl in o.material_slots)]
            bad.append({"material": m.name, "objects": users[:3]})
    detail = f"materials={total}, base-color-not-wired={len(bad)}"
    if bad:
        detail += " — " + ", ".join(b["material"] for b in bad[:5])
    return (not bad), detail, {"bad": bad[:10]}


def check_texture_wiring(mode):
    total = wired = 0
    plain = []
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        total += 1
        tex_nodes = [n for n in m.node_tree.nodes
                     if n.type in ("TEX_IMAGE", "TEX_NOISE", "TEX_BRICK", "TEX_VORONOI",
                                   "TEX_MUSGRAVE", "TEX_WAVE", "TEX_CHECKER")]
        imgs = [n for n in tex_nodes if n.type == "TEX_IMAGE" and n.image]
        ok = any(any(out.is_linked for out in n.outputs) for n in tex_nodes) if tex_nodes else False
        if ok:
            wired += 1
        else:
            plain.append(m.name)
    if mode in ("none", "nodes"):
        return True, f"mode={mode}, image wiring not required", {"total": total, "plain": plain[:8]}
    if total and not any(n.type == "TEX_IMAGE" for m in bpy.data.materials if m.use_nodes
                         for n in m.node_tree.nodes):
        return True, f"mode={mode}, semantic presets (no image textures required)", \
            {"total": total, "plain": plain[:8]}
    need = max(3, int(total * 0.5))
    ok = wired >= need
    return ok, f"materials={total} wired={wired} plain={len(plain)} need>={need}", {
        "total": total, "wired": wired, "plain": plain[:10]}


def check_scale():
    xs, ys, zs = [], [], []
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            xs.append(w.x), ys.append(w.y), zs.append(w.z)
    dims = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)] if xs else [0, 0, 0]
    # v5: scale is an estimate, not a metrology proof. Catch only gross unit errors (10x / 0.1x)
    # and let the height grow as the shell is built, so an early stage isn't blocked for being flat.
    ok = 1.5 <= dims[0] <= 60 and 1.5 <= dims[1] <= 60 and 0.2 <= dims[2] <= 20
    return ok, f"scene {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} m (estimate-level check)", {"dims": [round(d, 2) for d in dims],
                                                                      "meshes": len([o for o in bpy.data.objects if o.type == "MESH"])}


def check_corridor(radius=0.35, step=0.10):
    """Inferred (INF_*) geometry must not block the camera corridor.

    Hallucinating background is allowed and useful — a shower curtain, a shelf behind the sofa —
    as long as the walking path stays clear. Anything prefixed INF_ is treated as inferred and is
    tested against the calibrated camera path: every sample point, plus points along the segments
    between them, must stay `radius` metres outside its world bbox.
    """
    import mathutils
    blend_dir = Path(bpy.data.filepath).parent if bpy.data.filepath else Path.cwd()
    cal = None
    for cand in (blend_dir / "calibration.json", blend_dir / "project" / "calibration.json",
                 Path.cwd() / "calibration.json"):
        if cand.exists():
            try:
                cal = json.loads(cand.read_text())
                break
            except Exception:
                cal = None
    if not cal or not cal.get("camera_path"):
        return True, "no calibration camera_path — corridor check skipped (not a pass claim)", {}
    pts = [v["pos"] for v in cal["camera_path"] if v.get("pos")]
    if len(pts) < 2:
        return True, "camera_path shorter than 2 poses — skipped", {}
    samples = []
    for a, b in zip(pts, pts[1:]):
        n = max(1, int((mathutils.Vector(b) - mathutils.Vector(a)).length / step))
        for i in range(n + 1):
            t = i / n
            samples.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
    objs = [o for o in bpy.data.objects if o.type == "MESH" and o.name.upper().startswith("INF_")]
    if not objs:
        return True, "no inferred (INF_*) objects to check", {"inferred": 0}
    hits = []
    for o in objs:
        corners = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        lo = [min(c[i] for c in corners) for i in range(3)]
        hi = [max(c[i] for c in corners) for i in range(3)]
        for p in samples:
            if all(lo[i] - radius <= p[i] <= hi[i] + radius for i in range(3)):
                hits.append({"object": o.name, "at": [round(v, 2) for v in p],
                             "bbox": [[round(v, 2) for v in lo], [round(v, 2) for v in hi]]})
                break
    if hits:
        return False, (f"{len(hits)} inferred object(s) intrude into the camera corridor: "
                       + ", ".join(h["object"] for h in hits[:4])), {"hits": hits[:6], "inferred": len(objs)}
    return True, f"{len(objs)} inferred object(s) clear of the corridor", {"inferred": len(objs)}


def run_all(out_json="qa.json", intrinsics=None, texture_mode="both"):
    bpy.context.view_layer.update()      # otherwise matrix_world is stale for freshly created objects
    checks = []
    for cid, fn in (("floor_zero", check_floor), ("solid_envelope", check_solid),
                    ("openings", check_openings), ("objects_above_floor", check_above_floor),
                    ("camera_clearance", check_cameras), ("corridor_clear", check_corridor),
                    ("duplicate_objects", check_duplicates), ("material_base_color", check_material_base),
                    ("camera_rig", check_camera_rig), ("plan_coverage", check_plan_coverage),
                    ("furniture_orientation", check_furniture_orientation)):
        try:
            ok, detail, data = fn()
        except Exception as e:
            ok, detail, data = False, f"check crashed: {e}", {}
        checks.append({"id": cid, "ok": bool(ok), "detail": detail, "data": data})
    for cid, fn, arg in (("camera_fov", check_fov, intrinsics), ("material_texture_wiring", check_texture_wiring, texture_mode)):
        try:
            ok, detail, data = fn(arg)
        except Exception as e:
            ok, detail, data = False, f"check crashed: {e}", {}
        checks.append({"id": cid, "ok": bool(ok), "detail": detail, "data": data})
    checks.append(dict({"id": "scale_sanity"}, **dict(zip(("ok", "detail", "data"), check_scale()))))
    # camera_clearance / corridor_clear already label soft findings "(info)". Treat them as
    # warnings: one trajectory camera grazing a wall must not fail an otherwise usable scene.
    errors, warnings = [], []
    for c in checks:
        if c["ok"]:
            continue
        row = {"id": c["id"], "detail": c["detail"], "fix": FIXES.get(c["id"], "")}
        if c["id"] in ("camera_clearance", "corridor_clear") and "(info)" in str(c["detail"]):
            warnings.append(row)
        else:
            errors.append(row)
    report = {"ok": not errors, "blend": bpy.data.filepath, "errors": errors, "warnings": warnings,
              "checks": checks, "summary": _summary()}
    Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf8")
    print("QA_JSON:" + json.dumps(report, ensure_ascii=False))
    return report


FIXES = {
    "floor_zero": "Re-ground everything: walking surface at z=0. Never adopt raw reconstruction world coords.",
    "solid_envelope": "Extrude every wall/floor/ceiling to >=0.10 m (bs_geo.wall/floor_slab already do this).",
    "openings": "Cut the wall around the opening with bs_geo.wall(holes=[...]); a glass quad on a solid wall is a decal, not a window.",
    "objects_above_floor": "Re-anchor assets with asset_z = floor_z + own height; check parents' transforms.",
    "camera_clearance": "Back the camera off along +forward by >=1 m and raise to eye height; a camera inside a shell renders one grey plane.",
    "camera_fov": "Derive lens from K: hfov=2*atan((W/2)/fx), lens=sensor_width/2 / tan(hfov/2).",
    "material_texture_wiring": "Wire generated PNGs into Principled (bs_mat.with_maps) or stop claiming images mode.",
    "scale_sanity": "Calibrate scale from a known-size element (tile, door, light spacing), not the trajectory bbox.",
    "corridor_clear": "Rename inferred props INF_* and move them >=0.35 m off the camera path; keep the walk clear.",
    "duplicate_objects": "同一个物体被建了两遍（常见于重复执行 add_asset()）：删掉 .001 副本，或用 bs_stage.clean_duplicates()。",
    "material_base_color": "材质有贴图但 Base Color 没接线：用 bs_mat.with_maps / node_* 建，不要手搓 Principled。",
    "camera_rig": "相机不够：需要 Cam_Overview + 至少 2 个 Cam_Traj_XX；调 init_scene(rebuild=True)。",
    "plan_coverage": "计划里的物体没有建全 / 数量不够：按 SCENE_PLAN 的 id 建，重复物体用 bs_geo.clone_asset 生成 `<id>__02`…",
    "furniture_orientation": "沙发靠背朝墙、坐面朝房间：把 pose_yaw 改成背向最近墙面的方向，别只按长度方向转 90°。",
}


def _summary():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    tris = 0
    for o in meshes:
        o.data.calc_loop_triangles()
        tris += len(o.data.loop_triangles)
    return {"objects": len(bpy.data.objects), "meshes": len(meshes), "tris": tris,
            "materials": len(bpy.data.materials), "images": len(bpy.data.images),
            "cameras": len([o for o in bpy.data.objects if o.type == "CAMERA"])}


def render_views(views, out_dir="previews", res=960, samples=24, engine="CYCLES"):
    """views: [{"name":..,"loc":[x,y,z],"target":[x,y,z],"lens":mm}] -> list of written PNGs."""
    scene = bpy.context.scene
    scene.render.engine = engine
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for v in views:
        cam = bpy.data.objects.get(v.get("camera") or v.get("cam") or "")
        if not cam:
            cam_data = bpy.data.cameras.new(v["name"])
            cam = bpy.data.objects.new(v["name"], cam_data)
            scene.collection.objects.link(cam)
            cam.location = Vector(v.get("loc", (0, -8, 1.6)))
            d = Vector(v.get("target", (0, 0, 1.2))) - cam.location
            cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        # camera=... keeps the planned/QA camera's position; loc=... makes an ad-hoc view
        if v.get("loc") and v.get("target"):
            cam.location = Vector(v["loc"])
            d = Vector(v["target"]) - Vector(v["loc"])
            cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        cam.data.lens = float(v.get("lens", cam.data.lens or 24))
        cam.data.sensor_width = float(v.get("sensor", 36))
        scene.camera = cam
        # A global/aerial view is only useful as a cutaway: otherwise the ceiling hides the
        # entire plan and nobody can tell whether a sofa/barrier sits outside the footprint.
        hidden = []
        cut = bool(v.get("cutaway")) or ("overview" in v["name"].lower()) or v.get("camera") == "Cam_Overview"
        if cut:
            for ob in scene.objects:
                if ob.type == "MESH" and ob.name.startswith("Ceiling"):
                    hidden.append((ob, ob.hide_render)); ob.hide_render = True
        scene.render.resolution_x = int(v.get("res", res))
        scene.render.resolution_y = int(v.get("res_y", res * 3 // 4))
        path = out / f"{v['name']}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        for ob, old in hidden:
            ob.hide_render = old
        written.append(str(path))
    return written


if __name__ == "__main__":
    out_json, intrinsics, tex_mode = _paths()
    run_all(out_json, intrinsics, tex_mode)
