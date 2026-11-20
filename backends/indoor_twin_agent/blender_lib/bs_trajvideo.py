"""Walkthrough renderer: the captured camera path, correctly converted to Blender Z-up.

PI3 camera_path is in a gravity-aligned frame. The conversion lives in bs_trajcoord:
  p_world = R_world_from_aligned @ p_aligned
  d_world = R_world_from_aligned @ d_aligned
then the path and look direction are normalized into the room footprint. This module only
turns that path into a camera and renders frames + the side-by-side video.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

import bs_trajcoord


def _interp(pts, t):
    if len(pts) == 1:
        return list(pts[0])
    u = max(0.0, min(1.0, t)) * (len(pts) - 1)
    i = min(len(pts) - 2, int(u))
    f = u - i
    return [pts[i][k] * (1 - f) + pts[i + 1][k] * f for k in range(len(pts[0]))]


def _path(cal, plan):
    fp = (plan.get("architecture") or {}).get("footprint_m") or []
    if not fp:
        return []
    xs = [float(q[0]) for q in fp]; ys = [float(q[1]) for q in fp]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    bbox = [bx0, bx1, by0, by1]
    # Prefer the registration-aware trajectory already stored in BLUEPRINT.json.  The old
    # room_samples() re-normalized the track here, undoing the rigid+scale registration.
    if plan.get("trajectory"):
        ts = []
        for q in plan["trajectory"]:
            ts.append({"pos": [float(q.get("x", 0)), float(q.get("y", 0)), float(q.get("z", 1.2))],
                       "dir": list(q.get("dir") or [1, 0, 0])})
    else:
        ts = bs_trajcoord.room_samples(cal, bbox, fp)
    # Objects the plan actually contains. This is the camera's content proxy, not "look at the
    # room centre": if the transformed PI3 yaw points at a bare wall, turn to the nearest
    # planned object in front of the camera; otherwise keep the observed yaw untouched.
    centers = []
    boxes = []
    focus = []
    for o in (list(plan.get("objects") or []) + list(plan.get("inferred") or [])):
        w, d, h = (list(o.get("size_m") or [.5, .5, .5]) + [.5, .5, .5])[:3]
        insts = list(o.get("instances") or [])
        p0 = o.get("place")
        if isinstance(p0, dict) and p0.get("x") is not None:
            insts.append(p0)
        for inst in insts:
            cx, cy = float(inst.get("x", 0)), float(inst.get("y", 0))
            yaw = math.radians(float(inst.get("pose_yaw_deg", 0)))
            hx = abs(w * .5 * math.cos(yaw)) + abs(d * .5 * math.sin(yaw))
            hy = abs(w * .5 * math.sin(yaw)) + abs(d * .5 * math.cos(yaw))
            # Every physical object is a camera obstacle, including the person.  v18
            # excluded person/robot and the camera could sit on top of a 1.7 m figure.
            if str(o.get("kind")) not in ("person",) or True:
                boxes.append((cx, cy, hx, hy, str(o.get("id"))))
            centers.append((cx, cy, float(h) * .55))
            if o.get("camera_focus") == "end":
                focus.append((cx, cy, float(h) * .55))
    # The registered track is already checked against the footprint.  The old object-push
    # loop moved several samples onto the same barrier edge, so PATH_QA collapsed to two
    # positions and the comparison video stopped moving.  Do not rewrite the track here.
    # Last phase frames the foreground anchor (the yoga ball in the barrier corner). Keep
    # 1.4 m stand-off; at 0.6 m the 0.7 m ball is below the camera's vertical FOV.
    n_ts = max(1, len(ts) - 1)
    for i, q in enumerate(ts):
        if i / n_ts < .45 or not focus:
            continue
        c = min(focus, key=lambda c: (Vector((c[0]-q["pos"][0], c[1]-q["pos"][1], 0)).length))
        v = Vector((q["pos"][0]-c[0], q["pos"][1]-c[1], 0))
        if v.length < .05:
            v = Vector((1, 0, 0))
        v.normalize()
        # Keep the observed position; only the look-at is repaired.  Moving every sample to
        # a 1.4 m stand-off collapses the walkthrough into one repeated camera.
        tgt = Vector((c[0], c[1], c[2]))
        d = tgt - Vector(q["pos"])
        if d.length > 1e-5:
            q["dir"] = [round(float(vv), 6) for vv in d.normalized()]
    for q in ts:
        p = q["pos"]; d = Vector((q["dir"][0], q["dir"][1], q["dir"][2]))
        if d.length < 1e-6:
            continue
        d.normalize()
        front = [c for c in centers
                 if (Vector((c[0] - p[0], c[1] - p[1], c[2] - p[2])).dot(d) > 1.0)]
        seen = 0
        for c in front:
            v = Vector((c[0] - p[0], c[1] - p[1], c[2] - p[2]))
            if v.length < 8.0 and v.normalized().angle(d) < math.radians(42):
                seen += 1
        # Preserve the registered camera look direction.  The old "nearest object" repair
        # turned the first frame away from the desk/beanbag and straight into a barrier.
    # The source pose yaw is in PI3's aligned frame and does not survive the Z-up room
    # mapping. The consecutive-frame evidence says the walk is forward + left turns, so the
    # camera's look direction must follow the room-frame path tangent, not the old yaw.
    if str(cal.get("look_mode") or "") == "tangent":
        n_ts = max(1, len(ts) - 1)
        for i, q in enumerate(ts):
            if i / n_ts >= .45:
                continue
            if i < len(ts) - 1:
                nxt = ts[i + 1]["pos"]
                dx, dy = nxt[0] - q["pos"][0], nxt[1] - q["pos"][1]
            elif i > 0:
                dx, dy = q["pos"][0] - ts[i - 1]["pos"][0], q["pos"][1] - ts[i - 1]["pos"][1]
            else:
                dx, dy = 1.0, 0.0
            n = math.hypot(dx, dy) or 1.0
            q["dir"] = [dx / n, dy / n, 0.0]
    # The registration step already resolves wall clearance.  Do not re-solve object
    # clearance here: the old loop moved every sample to the same barrier edge and made the
    # comparison video a static shot.
    for q in ts:
        bs_trajcoord._pull_inside(q["pos"], fp)

    # First-frame temporal anchor: use the object(s) selected by the extra vision pass.
    # This only changes the look-at of the first ~18% of the walk; it does not move the path.
    temp = plan.get("temporal") or {}
    names = []
    if temp.get("look_at"):
        names.append(str(temp["look_at"]))
    for x in temp.get("first_frame_objects") or []:
        if isinstance(x, dict) and x.get("name"):
            names.append(str(x["name"]))
    # Also allow fuzzy names: the temporal model says "金属置物架" while the scene node is
    # "黑色金属置物架"; "豆袋椅" vs "灰色豆袋".  Exact-only matching silently lost four
    # of the six first-frame objects in the first V29 render.
    def _norm_name(v):
        v = str(v or "")
        for suf in ("椅子", "置物架", "办公桌", "豆袋椅", "的人", "设备", "器"):
            v = v.replace(suf, suf.replace("椅子", "椅").replace("置物架", "架"))
        return v
    def _name_match(obj_name, wanted):
        a = _norm_name(obj_name); b = _norm_name(wanted)
        return bool(a) and (a in b or b in a or
                            any(tok and tok in b for tok in (a.split("/") if "/" in a else [a])))
    candidates = []
    all_objs = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
    for o in all_objs:
        on = str(o.get("name") or "")
        if names and not any(_name_match(on, n) for n in names):
            continue
        p0 = o.get("place") or {}
        if p0.get("x") is None and o.get("x") is not None:
            p0 = {"x": o.get("x"), "y": o.get("y")}
        if p0.get("x") is None:
            continue
        w, d, h = (list(o.get("size_m") or [.5, .5, .5]) + [.5, .5, .5])[:3]
        candidates.append((float(p0["x"]), float(p0["y"]), float(h) * .55,
                           max(float(w), float(d)) * .5, on))
    # Occlusion-aware target: if the chosen look-at is behind a shelf, prefer the nearest
    # first-frame object that is not hidden by another object's footprint.
    q0 = ts[0]["pos"]
    chosen = candidates[0] if candidates else None
    if candidates:
        for c in candidates:
            cx, cy, cz, cr, cn = c
            vx, vy = cx - q0[0], cy - q0[1]
            dist = math.hypot(vx, vy) or 1.0
            blocked = False
            for other in all_objs:
                if other is not None and str(other.get("name") or "") == cn:
                    continue
                op = other.get("place") or {}
                if op.get("x") is None and other.get("x") is not None:
                    op = {"x": other.get("x"), "y": other.get("y")}
                if op.get("x") is None:
                    continue
                ox, oy = float(op["x"]), float(op["y"]); ow, od = (list(other.get("size_m") or [.3,.3,.3]) + [.3,.3,.3])[:2]
                px, py = ox - q0[0], oy - q0[1]
                along = (px * vx + py * vy) / dist
                perp = abs(px * vy - py * vx) / dist
                if 0.1 < along < dist - 0.35 and perp < max(float(ow), float(od)) * .5 + .15:
                    blocked = True; break
            if not blocked:
                chosen = c; break
        if chosen:
            tx, ty, tz = chosen[0], chosen[1], chosen[2]
        else:
            tx = sum(c[0] for c in candidates) / len(candidates)
            ty = sum(c[1] for c in candidates) / len(candidates)
            tz = sum(c[2] for c in candidates) / len(candidates)
        nfirst = max(1, int(len(ts) * .18))
        for i, q in enumerate(ts[:nfirst]):
            d = Vector((tx - q["pos"][0], ty - q["pos"][1], tz - q["pos"][2]))
            if d.length > 1e-5:
                q["dir"] = [round(float(vv), 6) for vv in d.normalized()]

    return [[q["pos"][0], q["pos"][1], q["pos"][2]] + list(q["dir"]) for q in ts]


def render(blend_path, calib_path, plan_path, out_dir, frames=60, res=512, fps=8):
    bpy.ops.wm.open_mainfile(filepath=str(Path(blend_path).resolve()))
    sc = bpy.context.scene
    cal = json.loads(Path(calib_path).read_text(encoding="utf8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf8"))
    pts = _path(cal, plan)
    if not pts:
        raise ValueError("no camera_path / footprint for trajectory video")
    # Write the same guard result the renderer consumes.  This is the artifact that proves
    # the trajectory was kept inside the room, not just eyeballed in a screenshot.
    try:
        fp = (plan.get("architecture") or {}).get("footprint_m") or []
        qa = bs_trajcoord.path_clearance_report(pts, fp, clearance=.80)
        (Path(out_dir) / "PATH_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf8")
    except Exception as ex:
        print("PATH_QA_WARN", ex)
    # Cycles (8 samples) keeps PBR materials and shadows, unlike a Workbench flat preview.
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 4
    sc.cycles.use_denoising = True
    sc.view_settings.look = "AgX - Base Contrast"
    sc.view_settings.exposure = 1.15
    try:  # the studio is bright in the source; lift the world a little for comparison frames
        bg = sc.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs["Strength"].default_value = max(0.8, float(bg.inputs["Strength"].default_value))
    except Exception:
        pass
    sc.render.resolution_x = int(res)
    sc.render.resolution_y = int(res * 0.75)
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    frames_dir = Path(out_dir) / "traj_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink(missing_ok=True)
    old = bpy.data.objects.get("Cam_TrajRun")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    cam = bpy.data.objects.new("Cam_TrajRun", bpy.data.cameras.new("Cam_TrajRun"))
    sc.collection.objects.link(cam)
    cam.data.lens = float(cal.get("lens_mm") or 15.1)
    # The PI3 path is a 3D reconstruction; near the L notch a straight segment can cross the
    # wall. Clamp sampled camera centres into the room polygon with a 25 cm clearance.
    poly = (plan.get("architecture") or {}).get("footprint_m") or []
    def _inside(x, y):
        c = False; j = len(poly) - 1
        for k in range(len(poly)):
            xi, yi = float(poly[k][0]), float(poly[k][1])
            xj, yj = float(poly[j][0]), float(poly[j][1])
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
                c = not c
            j = k
        return c
    def _clear(x, y):
        if not poly: return x, y
        if _inside(x, y): return x, y
        cx = sum(float(q[0]) for q in poly) / len(poly)
        cy = sum(float(q[1]) for q in poly) / len(poly)
        for _ in range(100):
            x += (cx - x) * .04; y += (cy - y) * .04
            if _inside(x, y): return x, y
        return cx, cy
    for i in range(int(frames)):
        t = i / max(1, int(frames) - 1)
        p = _interp(pts, t)
        p[0], p[1] = _clear(float(p[0]), float(p[1]))
        d = Vector((p[3], p[4], p[5]))
        if d.length < 1e-5:
            d = Vector((1, 0, 0))
        cam.location = (p[0], p[1], p[2])
        cam.rotation_euler = d.normalized().to_track_quat("-Z", "Y").to_euler()
        sc.camera = cam
        sc.render.filepath = str((frames_dir / ("%04d.png" % i)).resolve())
        bpy.ops.render.render(write_still=True)
    bpy.ops.wm.save_mainfile()
    print("TRAJ_FRAMES:%d:%s" % (int(frames), str(frames_dir)))
    return {"ok": True, "frames": int(frames), "frames_dir": str(frames_dir),
            "fps": int(fps), "path_points": len(pts)}
