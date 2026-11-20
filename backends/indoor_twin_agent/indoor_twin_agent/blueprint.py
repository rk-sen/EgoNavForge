"""V18 cognition blueprint: gravity-aligned top-down plan.

Inputs:
  calibration.json: path in PI3 gravity-aligned frame + R_world_from_aligned
  ARCHITECTURE.json: footprint / storey height / opening evidence
  SCENE_PLAN.json: object graph

Outputs:
  BLUEPRINT.json: a machine-readable "drawing" (walls, openings, path, object direction)
  blueprint.png: an unoccluded top-down plan used as the cognition artifact

The path is projected with the gravity-aligned rotation, then normalized into the plan
frame.  This is the same Z-up transform used by the walkthrough camera, so the plan and the
trajectory cannot disagree about handedness.  The drawing is intentionally stored before
the 3D shell; it is the "what room did the video see?" artifact, not a post-hoc render.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _inside(x, y, poly):
    c = False; j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            c = not c
        j = i
    return c


def _room_path(cal, poly, margin=.55, registration=None):
    """Project PI3 gravity-aligned camera path into the plan polygon.

    With a registration record the path uses the same similarity transform as the footprint;
    without one (legacy calls) it falls back to the old normalized mapping.
    """
    samples = cal.get("camera_path") or []
    if not samples:
        return []
    R = np.asarray(cal.get("rotation_world_from_aligned") or np.eye(3), dtype=float)
    if R.shape != (3, 3):
        R = np.eye(3)
    raw = [np.asarray((s.get("pos") or [0, 0, 1.2])[:3], dtype=float) for s in samples]
    worlds = [R @ p for p in raw]
    if registration and registration.get("theta_deg") is not None:
        th = math.radians(float(registration.get("theta_deg", 0.0)))
        scale = float(registration.get("scale", 1.0))
        tx, ty = float(registration.get("tx", 0.0)), float(registration.get("ty", 0.0))
        c, sn = math.cos(th), math.sin(th)
        out = []
        for i, (w, s) in enumerate(zip(worlds, samples)):
            x, y = float(w[0]), float(w[1])
            qx = tx + scale * (c * x - sn * y)
            qy = ty + scale * (sn * x + c * y)
            d = R @ _dir(s)
            dx, dy = c * float(d[0]) - sn * float(d[1]), sn * float(d[0]) + c * float(d[1])
            # The camera looks through the room, not at the ceiling: PI3's rotated dir has a
            # large Z component because of its axis convention.  Keep the capture pitch only.
            dz = math.sin(math.radians(float(s.get("pitch", 0.0))))
            # PI3's aligned pos[2] is the eye height (1.0-1.8 m); the rotated world Z is a
            # gravity frame coordinate and can be negative.  Never put the walkthrough camera
            # under the floor.
            ez = float((s.get("pos") or [0, 0, 1.5])[2])
            if not (0.8 <= abs(ez) <= 2.2):
                ez = 1.5
            out.append({"frame": i, "x": round(qx, 3), "y": round(qy, 3),
                        "z": round(max(1.05, min(1.75, abs(ez))), 3),
                        "dir": [round(dx, 4), round(dy, 4), round(dz, 4)]})
        return out
    xs = [float(v[0]) for v in worlds]; ys = [float(v[1]) for v in worlds]
    rx0, rx1, ry0, ry1 = min(xs), max(xs), min(ys), max(ys)
    px0, px1 = min(float(p[0]) for p in poly), max(float(p[0]) for p in poly)
    py0, py1 = min(float(p[1]) for p in poly), max(float(p[1]) for p in poly)
    sx = (px1 - px0 - 2 * margin) / max(1e-6, rx1 - rx0)
    sy = (py1 - py0 - 2 * margin) / max(1e-6, ry1 - ry0)
    flip = bool(cal.get("trajectory_flip_x", False))
    out = []
    for i, (w, s) in enumerate(zip(worlds, samples)):
        ux = (float(w[0]) - rx0) / max(1e-6, rx1 - rx0)
        if flip:
            ux = 1.0 - ux
        q = [px0 + margin + ux * (px1 - px0 - 2 * margin),
             py0 + margin + (float(w[1]) - ry0) * sy]
        # keep the path inside the footprint; a normalized mapping can put an endpoint on a wall
        if not _inside(q[0], q[1], poly):
            cx = sum(float(p[0]) for p in poly) / len(poly)
            cy = sum(float(p[1]) for p in poly) / len(poly)
            for _ in range(20):
                q[0] = q[0] * .78 + cx * .22; q[1] = q[1] * .78 + cy * .22
                if _inside(q[0], q[1], poly):
                    break
        out.append({"frame": i, "x": round(float(q[0]), 3), "y": round(float(q[1]), 3),
                    "z": round(float(w[2]), 3), "dir": [round(float(v), 4) for v in (R @ _dir(s))]})
    return out


def _dir(s):
    y = math.radians(float(s.get("yaw", 0))); p = math.radians(float(s.get("pitch", 0)))
    return np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])


def _gravity(cal):
    R = np.asarray(cal.get("rotation_world_from_aligned") or np.eye(3), dtype=float)
    # PI3's "up" convention is opencv(-Y); convert it to the Blender Z-up world.
    g = R @ np.array([0.0, -1.0, 0.0])
    n = float(np.linalg.norm(g)) or 1.0
    return [round(float(v / n), 5) for v in g]


def build_blueprint(cal, arch, plan, out_json=None, out_png=None):
    poly = [[float(p[0]), float(p[1])] for p in arch["footprint_m"]]
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    w, d = max(xs) - min(xs), max(ys) - min(ys)
    area = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        area += a[0] * b[1] - b[0] * a[1]
    area = abs(area) / 2
    path = _room_path(cal, poly, registration=arch.get("path_registration"))
    # Time-anchor the walkthrough: the temporal pass names the objects visible in the
    # source's first frame.  If those sit near the end of the registered path, the source
    # sequence is running opposite to the stored pose order.  Reverse the camera's time
    # order (positions stay the same); otherwise the comparison starts at the wrong end of
    # the room even when the path itself is geometrically correct.
    temp = plan.get("temporal") or {}
    names = []
    if temp.get("look_at"):
        names.append(str(temp["look_at"]))
    for x in temp.get("first_frame_objects") or []:
        if isinstance(x, dict) and x.get("name"):
            names.append(str(x["name"]))
    targets = []
    for o in (plan.get("objects") or []) + (plan.get("inferred") or []):
        on = str(o.get("name") or "")
        if not names or not any(str(n) in on or on in str(n) for n in names):
            continue
        p0 = o.get("place") or ({"x": o.get("x"), "y": o.get("y")} if o.get("x") is not None else {})
        if p0.get("x") is not None:
            targets.append((float(p0["x"]), float(p0["y"])))
    if path and targets:
        tx = sum(t[0] for t in targets) / len(targets)
        ty = sum(t[1] for t in targets) / len(targets)
        # Choose the earliest trajectory sample that views the first-frame object at a
        # believable distance.  A pure reversal can put the camera 0.7 m from the desk and
        # crop it; a pure start can leave the desk behind a shelf.  This trims the time
        # origin, it does not invent a new path.
        pick = 0
        best = None
        for i, q in enumerate(path):
            dist = math.hypot(q["x"] - tx, q["y"] - ty)
            score = abs(dist - 3.2)
            if 2.0 <= dist <= 4.8:
                score -= 0.5
            if best is None or score < best:
                best = score; pick = i
        path = path[pick:]
        for i, q in enumerate(path):
            q["frame"] = i
    windows = []
    for i, e in enumerate(arch.get("windows") or []):
        windows.append({
            "id": f"Window_{i+1}", "kind": "window", "side": e.get("side"),
            "u_m": e.get("u_m"), "width_m": e.get("width_m"), "height_m": e.get("height_m"),
            "z0_m": e.get("z0_m"), "type": e.get("type") or "plain_glass",
            "mullions": e.get("mullions") or {"cols": 1, "rows": 1},
            "curtain": e.get("curtain") or {"draw": 0.0, "fraction": 0.0, "side": "none"},
            "evidence": "ARCHITECTURE.windows",
        })
    doors = []
    for i, e in enumerate(arch.get("doors") or []):
        doors.append({"id": f"Door_{i+1}", "kind": "door", "side": e.get("side"),
                      "u_m": e.get("u_m"), "center_m": e.get("center_m"),
                      "width_m": e.get("width_m"), "height_m": e.get("height_m"), "z0_m": e.get("z0_m")})
    objects = []
    for o in plan.get("objects") or []:
        p0 = o.get("place") or {}
        inst = o.get("instances") or []
        if p0.get("x") is not None:
            inst = [p0] + inst
        for k, q in enumerate(inst):
            objects.append({"id": o.get("id") if k == 0 else f"{o.get('id')}__{k+1:02d}",
                            "kind": o.get("kind"), "name": o.get("name"),
                            "size_m": o.get("size_m"),
                            "segments": o.get("segments") or [],
                            "x": q.get("x"), "y": q.get("y"),
                            "yaw_deg": q.get("pose_yaw_deg", 0),
                            "direction": [round(math.cos(math.radians(float(q.get("pose_yaw_deg", 0)))), 3),
                                          round(math.sin(math.radians(float(q.get("pose_yaw_deg", 0)))), 3)]})
    bp = {
        "version": "V18",
        "coordinate": "Blender Z-up metres; gravity projected plan",
        "gravity": {"source": cal.get("gravity_up_source"), "vector_world": _gravity(cal),
                    "top_down_plane": "world XY, Z is gravity axis"},
        "room": {"footprint_m": poly, "bbox_m": [min(xs), max(xs), min(ys), max(ys)],
                 "area_m2": round(area, 2), "width_m": round(w, 3), "depth_m": round(d, 3),
                 "ceiling_z_m": arch.get("ceiling_z")},
        "trajectory": path,
        "temporal": plan.get("temporal") or {},
        "windows": windows,
        "doors": doors,
        "objects": objects,
        "source": "gravity_aligned_trajectory + API opening evidence + object graph",
    }
    if out_json:
        Path(out_json).write_text(json.dumps(bp, ensure_ascii=False, indent=2), encoding="utf8")
    if out_png:
        draw_blueprint(bp, out_png)
    return bp


def draw_blueprint(bp, out_png):
    poly = bp["room"]["footprint_m"]
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W = 1500; H = int(W * max(1.0, (y1 - y0 + 1.4) / max(1.0, (x1 - x0 + 1.4))))
    scale = min((W - 100) / max(1.0, x1 - x0), (H - 100) / max(1.0, y1 - y0))
    ox = (W - (x1 - x0) * scale) / 2; oy = (H - (y1 - y0) * scale) / 2
    def P(x, y): return (ox + (x - x0) * scale, oy + (y1 - y) * scale)
    img = Image.new("RGB", (W, H), "#f5f1e8")
    dr = ImageDraw.Draw(img, "RGBA")
    dr.rounded_rectangle((16, 16, W - 16, H - 16), 18, fill="#fffdf8", outline="#1d1d1b", width=3)
    # floor
    dr.polygon([P(*p) for p in poly], fill="#d8d3c8")
    # walls
    dr.line([P(*p) for p in poly + [poly[0]]], fill="#1d1d1b", width=7)
    # window openings: thick blue segments on wall edges, normal offset from side
    for win in bp["windows"]:
        side = win.get("side"); u = float(win.get("u_m") or 0); ww = float(win.get("width_m") or 1.4)
        if side == "W": a = P(x0, y0 + u); b = P(x0, y0 + u + ww)
        elif side == "E": a = P(x1, y0 + u); b = P(x1, y0 + u + ww)
        elif side == "N": a = P(x0 + u, y1); b = P(x0 + u + ww, y1)
        else: a = P(x0 + u, y0); b = P(x0 + u + ww, y0)
        dr.line([a, b], fill="#2478a8", width=11)
        dr.line([a, b], fill="#5ec2e8", width=5)
        # half-curtain marker, if present
        cur = win.get("curtain") or {}
        if float(cur.get("fraction", cur.get("draw", 0)) or 0) > 0:
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            dr.ellipse((mid[0] - 7, mid[1] - 7, mid[0] + 7, mid[1] + 7), fill="#c9a227")
    for door in bp["doors"]:
        side = door.get("side"); u = float(door.get("u_m") or 0); ww = float(door.get("width_m") or .95)
        c = door.get("center_m")
        if c:
            # Use the wall point, then lay the leaf along the nearest wall.
            cx, cy = float(c[0]), float(c[1])
            a = P(cx, cy - ww / 2); b = P(cx, cy + ww / 2)
        elif side == "W": a = P(x0, y0 + u); b = P(x0, y0 + u + ww)
        elif side == "E": a = P(x1, y0 + u); b = P(x1, y0 + u + ww)
        elif side == "N": a = P(x0 + u, y1); b = P(x0 + u + ww, y1)
        else: a = P(x0 + u, y0); b = P(x0 + u + ww, y0)
        dr.line([a, b], fill="#9b5a32", width=10)
    # trajectory + forward arrows
    pts = bp["trajectory"]
    if pts:
        xy = [P(p["x"], p["y"]) for p in pts]
        dr.line(xy, fill="#e04b2f", width=4, joint="curve")
        for i in range(0, len(pts), max(1, len(pts)//9)):
            p = pts[i]; n = pts[min(len(pts)-1, i+1)]
            a = P(p["x"], p["y"]); b = P(n["x"], n["y"])
            if math.hypot(b[0]-a[0], b[1]-a[1]) > 3:
                dr.line([a, b], fill="#e04b2f", width=5)
                dr.ellipse((b[0]-5, b[1]-5, b[0]+5, b[1]+5), fill="#e04b2f")
        dr.text(P(pts[0]["x"], pts[0]["y"] - .25), "START", fill="#e04b2f", font=ImageFont.truetype(DEFAULT_FONT, 16))
    # objects with id + direction tick
    palette = {"barrier": "#1e4fb3", "sofa": "#c86a2a", "desk": "#8b5a2b", "monitor": "#202020",
               "shelf": "#4b5563", "ball": "#2e9e4f", "person": "#7a4a32", "table": "#a16207",
               "beanbag": "#6b7280", "chair": "#404040", "robot": "#111111"}
    for o in bp["objects"]:
        if o.get("x") is None: continue
        p = P(float(o["x"]), float(o["y"]))
        col = palette.get(str(o.get("kind")), "#5b5b5b")
        dr.ellipse((p[0]-6, p[1]-6, p[0]+6, p[1]+6), fill=col)
        dx, dy = o.get("direction") or [0, 0]
        dr.line([p, (p[0] + dx * 22, p[1] - dy * 22)], fill=col, width=3)
        dr.text((p[0]+8, p[1]-8), str(o.get("id")), fill="#222", font=ImageFont.truetype(DEFAULT_FONT, 10))
    # title/scale
    dr.text((34, 30), "V18  gravity-aligned top-down cognition blueprint", fill="#1d1d1b", font=ImageFont.truetype(DEFAULT_FONT, 24))
    dr.text((34, 66), f"room {bp['room']['width_m']:.2f} x {bp['room']['depth_m']:.2f} m   area {bp['room']['area_m2']:.2f} m2   path {len(pts)} samples", fill="#4d4d4d", font=ImageFont.truetype(DEFAULT_FONT, 15))
    dr.text((W - 180, H - 34), "1 m grid", fill="#666", font=ImageFont.truetype(DEFAULT_FONT, 14))
    # 1 m scale bar
    sb = scale * 1.0
    dr.line((W - 160, H - 55, W - 160 + sb, H - 55), fill="#222", width=3)
    dr.line((W - 160, H - 60, W - 160, H - 50), fill="#222", width=3)
    dr.line((W - 160 + sb, H - 60, W - 160 + sb, H - 50), fill="#222", width=3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
