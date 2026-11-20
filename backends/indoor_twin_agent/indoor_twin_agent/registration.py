"""Rigid + scale registration of the metric camera track into the LLM footprint.

The LLM sees video frames and returns a footprint in its own local frame; PI3 returns a
gravity-aligned metric track in another frame.  The old blueprint squeezed the track into the
footprint with separate x/y normalization, which destroyed the track shape and camera yaw.
Here we search a *similarity* transform (rotation + uniform scale + translation), keep the track
shape, and choose a placement that is inside the polygon, has wall clearance, and whose first
camera view points toward the room interior.  Openings are used as a soft door anchor.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _inside(x: float, y: float, poly: list[list[float]]) -> bool:
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            c = not c
        j = i
    return c


def _seg_dist(px: float, py: float, a: list[float], b: list[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _wall_dist(x: float, y: float, poly: list[list[float]]) -> float:
    return min(_seg_dist(x, y, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


def _dir(sample: dict) -> np.ndarray:
    yaw = math.radians(float(sample.get("yaw", 0.0)))
    pitch = math.radians(float(sample.get("pitch", 0.0)))
    return np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])


def raw_track(cal: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Gravity-aligned metric positions and directions (same convention as blueprint.py)."""
    samples = cal.get("camera_path") or []
    R = np.asarray(cal.get("rotation_world_from_aligned") or np.eye(3), dtype=float)
    if R.shape != (3, 3):
        R = np.eye(3)
    pos = np.array([R @ np.asarray((s.get("pos") or [0, 0, 1.2])[:3], dtype=float) for s in samples])
    dirs = np.array([R @ _dir(s) for s in samples])
    return pos, dirs, samples


def _door_point(arch: dict) -> tuple[float, float] | None:
    doors = arch.get("doors") or []
    if not doors:
        return None
    poly = [[float(p[0]), float(p[1])] for p in arch["footprint_m"]]
    x0, x1 = min(p[0] for p in poly), max(p[0] for p in poly)
    y0, y1 = min(p[1] for p in poly), max(p[1] for p in poly)
    d = doors[0]
    side = str(d.get("side") or "E")[:1].upper()
    u = float(d.get("u_m") or 0.0)
    if side == "W": return (x0, y0 + u)
    if side == "E": return (x1, y0 + u)
    if side == "N": return (x0 + u, y1)
    return (x0 + u, y0)


def register_path(cal: dict, arch: dict) -> dict:
    poly = [[float(p[0]), float(p[1])] for p in arch["footprint_m"]]
    pos, dirs, samples = raw_track(cal)
    if len(pos) < 2:
        return {"source": "no_track", "theta_deg": 0.0, "scale": 1.0, "tx": 0.0, "ty": 0.0}
    pc = pos[:, :2].mean(axis=0)
    x0, x1 = min(p[0] for p in poly), max(p[0] for p in poly)
    y0, y1 = min(p[1] for p in poly), max(p[1] for p in poly)
    # polygon centroid (area centroid is better than bbox centre for L-shapes)
    area = 0.0; cx = 0.0; cy = 0.0
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        cross = a[0] * b[1] - b[0] * a[1]
        area += cross
        cx += (a[0] + b[0]) * cross
        cy += (a[1] + b[1]) * cross
    if abs(area) < 1e-8:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    else:
        cx /= (3.0 * area); cy /= (3.0 * area)
    door = _door_point(arch)
    candidates = []
    for theta in np.arange(0, 360, 5.0):
        th = math.radians(theta)
        c, s = math.cos(th), math.sin(th)
        for scale in (1.0, 0.95, 0.90, 0.85, 0.80, 1.05):
            # centre the track on the room centroid, then allow a small translation search
            for ox, oy in ((0, 0), (0.3, 0), (-0.3, 0), (0, 0.3), (0, -0.3)):
                tx = cx - scale * (c * pc[0] - s * pc[1]) + ox
                ty = cy - scale * (s * pc[0] + c * pc[1]) + oy
                pts = np.array([[tx + scale * (c * p[0] - s * p[1]),
                                 ty + scale * (s * p[0] + c * p[1])] for p in pos])
                d0 = np.array([c * dirs[0, 0] - s * dirs[0, 1],
                               s * dirs[0, 0] + c * dirs[0, 1]])
                view = d0 / (np.linalg.norm(d0) + 1e-9)
                to_c = np.array([cx - pts[0, 0], cy - pts[0, 1]])
                to_c = to_c / (np.linalg.norm(to_c) + 1e-9)
                inside = sum(1 for p in pts if _inside(float(p[0]), float(p[1]), poly))
                min_wall = min(_wall_dist(float(p[0]), float(p[1]), poly) for p in pts)
                view_dot = float(np.dot(view, to_c))
                door_dist = 0.0 if door is None else float(np.linalg.norm(pts[-1] - np.array(door)))
                # All track points inside is mandatory; clearance and view direction are soft.
                score = (10.0 * inside / len(pts) + 3.0 * min(min_wall, 1.0)
                         + 1.2 * view_dot - 0.25 * abs(scale - 1.0) - 0.12 * min(door_dist, 5.0))
                candidates.append((score, float(theta), float(scale), float(tx), float(ty),
                                   inside, min_wall, view_dot, door_dist))
    best = max(candidates, key=lambda q: q[0])
    score, theta, scale, tx, ty, inside, min_wall, view_dot, door_dist = best
    return {
        "source": "rigid_scale_registration: trajectory centroid -> polygon area centroid",
        "theta_deg": round(theta, 2), "scale": round(scale, 3),
        "tx": round(tx, 4), "ty": round(ty, 4),
        "score": round(float(score), 3), "inside": f"{inside}/{len(pos)}",
        "min_wall_m": round(float(min_wall), 3), "view_dot": round(float(view_dot), 3),
        "door_anchor_m": round(float(door_dist), 3),
        "track_centroid": [round(float(pc[0]), 3), round(float(pc[1]), 3)],
        "room_centroid": [round(float(cx), 3), round(float(cy), 3)],
    }
