"""PI3 aligned trajectory -> Blender Z-up room coordinates.

The calibration file gives:
  * camera_path[i].pos in the gravity-aligned PI3 frame
  * yaw/pitch in that aligned frame
  * rotation_world_from_aligned R (3x3)

v10 mapped aligned x/y directly into the room, which rotates the walk and the look direction.
The correct order is: p_world = R @ p_aligned, then normalize the world bbox into the room
footprint while preserving its shape; d_world = R @ d_aligned and transform it with the same
x/y scale.  This is a pure-data helper so both QA cameras and the walkthrough video use it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _point_in_poly(x: float, y: float, poly) -> bool:
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
    for _ in range(16):
        p[0] = p[0] * 0.72 + cx * 0.28
        p[1] = p[1] * 0.72 + cy * 0.28
        if _point_in_poly(p[0], p[1], poly):
            break
    return p


def _dir_from_pose(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y = math.radians(float(yaw_deg)); p = math.radians(float(pitch_deg))
    return np.array([math.cos(y) * math.cos(p), math.sin(y) * math.cos(p), math.sin(p)])


def room_samples(cal: dict[str, Any], room_bbox, room_polygon=None, margin: float = 0.75):
    """Return camera samples in the room frame with a normalized forward vector."""
    samples = cal.get("camera_path") or []
    if not samples:
        return []
    R = np.asarray(cal.get("rotation_world_from_aligned") or np.eye(3), dtype=float)
    if R.shape != (3, 3):
        R = np.eye(3)
    raw = [np.asarray((s.get("pos") or [0, 0, 1.2])[:3], dtype=float) for s in samples]
    # PI3 aligned frame -> Blender world frame (Z up)
    worlds = [R @ p for p in raw]
    rx0, rx1 = min(p[0] for p in worlds), max(p[0] for p in worlds)
    ry0, ry1 = min(p[1] for p in worlds), max(p[1] for p in worlds)
    rw, rh = rx1 - rx0, ry1 - ry0
    bx0, bx1, by0, by1 = (float(v) for v in room_bbox[:4])
    sx = (bx1 - bx0 - 2 * margin) / max(1e-6, rw)
    sy = (by1 - by0 - 2 * margin) / max(1e-6, rh)
    # The room's plan axes are not signed relative to PI3's world axes. A mirrored X mapping is
    # still a rigid Z-up transform, just the other handedness. It matters for a walkthrough:
    # the source starts beside the beanbag/barriers and ends at the sofa/ball, so the mirrored
    # path visits those rooms in the observed order instead of walking away from them.
    flip_x = bool(cal.get("trajectory_flip_x", False))
    out = []
    for i, s in enumerate(samples):
        w = worlds[i]
        ux = (w[0] - rx0) / max(1e-6, rw)
        if flip_x:
            ux = 1.0 - ux
        q = [bx0 + margin + ux * (bx1 - bx0 - 2 * margin),
             by0 + margin + (w[1] - ry0) * sy,
             max(1.05, min(1.65, float(w[2]) if abs(w[2]) < 3 else 1.35))]
        _pull_inside(q, room_polygon)
        f_aligned = _dir_from_pose(float(s.get("yaw", 0.0)), float(s.get("pitch", 0.0)))
        f_world = R @ f_aligned
        # Yaw is rotated into the room; pitch is kept from the capture because the PI3
        # vertical axis is not guaranteed to be the Blender camera's local -Z (using R on
        # pitch produced a 30-40° upward stare that no frame in the footage supports).
        yaw_vec = np.array([f_world[0] * sx, f_world[1] * sy, 0.0])
        if np.linalg.norm(yaw_vec) < 1e-6:
            yaw_vec = np.array([1.0, 0, 0])
        yaw_vec = yaw_vec / np.linalg.norm(yaw_vec)
        yaw = math.degrees(math.atan2(float(yaw_vec[1]), float(yaw_vec[0])))
        if flip_x:
            yaw = 180.0 - yaw
            yaw_vec[0] = -yaw_vec[0]
        # PI3's axis sign is not consistent across its own frames. After the Z-up rotation,
        # a yaw can point into the adjacent wall even though the walk continues inside. If the
        # half-metre look-ahead is outside the footprint, flip 180°; this is an axis-sign
        # disambiguation, not a "look at the room centre" fallback.
        def _looks_out(ang):
            fx, fy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            return not _point_in_poly(q[0] + fx * 1.1, q[1] + fy * 1.1, room_polygon)
        if _looks_out(yaw) and not _looks_out(yaw + 180.0):
            yaw += 180.0
        # A per-video global yaw offset is allowed only for the PI3 axis-sign convention;
        # it is a calibration datum (like flip_x), not a hidden "look at the centre" fallback.
        yaw += float(cal.get("trajectory_yaw_offset_deg", 0.0))
        if yaw > 180.0:
            yaw -= 360.0
        if yaw < -180.0:
            yaw += 360.0
        pitch = float(s.get("pitch", 0.0))
        yy = math.radians(yaw); pp = math.radians(pitch)
        f_room = np.array([math.cos(yy) * math.cos(pp), math.sin(yy) * math.cos(pp), math.sin(pp)])
        out.append({"pos": [round(float(v), 4) for v in q],
                    "yaw": round(yaw, 2), "pitch": round(pitch, 2),
                    "dir": [round(float(v), 6) for v in f_room]})
    return out


def sample_forward(sample) -> list[float]:
    d = sample.get("dir")
    if d:
        return [float(v) for v in d]
    return [math.cos(math.radians(float(sample.get("yaw", 0)))),
            math.sin(math.radians(float(sample.get("yaw", 0)))),
            math.sin(math.radians(float(sample.get("pitch", 0))))]


def _wall_distance(x, y, poly):
    """Nearest distance from a point to the footprint boundary."""
    if not poly:
        return 1e9
    best = 1e9
    for i in range(len(poly)):
        ax, ay = float(poly[i][0]), float(poly[i][1])
        bx, by = float(poly[(i + 1) % len(poly)][0]), float(poly[(i + 1) % len(poly)][1])
        vx, vy = bx - ax, by - ay
        ll = vx * vx + vy * vy
        t = 0.0 if ll < 1e-12 else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / ll))
        px, py = ax + t * vx, ay + t * vy
        best = min(best, math.hypot(x - px, y - py))
    return best


def camera_clearance(p, poly, clearance=0.55, iterations=4):
    """Keep a walkthrough camera at least *clearance* metres away from every wall.

    This is deliberately a hard geometric guard, not a look-at fix: a camera that is
    inside/through a wall cannot be rescued by changing its yaw.  The point is first
    pulled into the polygon, then pushed along the inward direction of the nearest
    wall segment until the clearance is satisfied.
    """
    p = [float(p[0]), float(p[1])] if len(p) >= 2 else [0.0, 0.0]
    if not poly:
        return p
    _pull_inside(p, poly)
    cx = sum(float(q[0]) for q in poly) / len(poly)
    cy = sum(float(q[1]) for q in poly) / len(poly)
    for _ in range(max(1, int(iterations))):
        if _wall_distance(p[0], p[1], poly) >= clearance and _point_in_poly(p[0], p[1], poly):
            break
        # move toward the room centre by just enough to clear the nearest edge
        vx, vy = cx - p[0], cy - p[1]
        n = math.hypot(vx, vy) or 1.0
        # step scales with the violation; cap it so a deep violation is resolved gradually
        need = max(0.0, clearance - _wall_distance(p[0], p[1], poly))
        step = min(1.0, need + 0.08)
        p[0] += vx / n * step
        p[1] += vy / n * step
        _pull_inside(p, poly)
    return [round(float(p[0]), 4), round(float(p[1]), 4)]


def path_clearance_report(pts, poly, clearance=0.55):
    """Return a small machine-readable path QA record."""
    items = []
    for i, p in enumerate(pts):
        x, y = float(p[0]), float(p[1])
        items.append({"i": i, "x": round(x, 4), "y": round(y, 4),
                      "wall_m": round(_wall_distance(x, y, poly), 4),
                      "inside": bool(_point_in_poly(x, y, poly))})
    return {"clearance_m": float(clearance),
            "min_wall_m": round(min([it["wall_m"] for it in items] or [0]), 4),
            "outside_count": sum(0 if it["inside"] else 1 for it in items),
            "frames": items}
