"""Room/asset layout checks: keep AABBs inside the L footprint and keep semantic pairs near.

This is intentionally independent of Blender. The agent can run it before building a .blend
and fail the step if an object's AABB leaves the room or a workstation pair is pulled apart by
pose repair.
"""
from __future__ import annotations

import math
from typing import Any


def _inside(x: float, y: float, poly: list[list[float]]) -> bool:
    if not poly or len(poly) < 3:
        return True
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            c = not c
        j = i
    return c


def instance_aabb(inst: list[float], size: list[float]) -> tuple[float, float, float, float]:
    w, d = float(size[0]), float(size[1])
    yaw = math.radians(float(inst[3]) if len(inst) > 3 else 0.0)
    hx = abs(w * .5 * math.cos(yaw)) + abs(d * .5 * math.sin(yaw))
    hy = abs(w * .5 * math.sin(yaw)) + abs(d * .5 * math.cos(yaw))
    x, y = float(inst[0]), float(inst[1])
    return x - hx, x + hx, y - hy, y + hy


def outside_instances(specs: list[dict], polygon: list[list[float]]) -> list[dict]:
    bad = []
    for s in specs:
        size = list(s.get("size_m") or [.5, .5, .5])
        for i, inst in enumerate(s.get("placements") or []):
            x0, x1, y0, y1 = instance_aabb(inst, size)
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            out = [p for p in pts if not _inside(p[0], p[1], polygon)]
            if out:
                bad.append({"id": s["id"], "instance": i, "center": [inst[0], inst[1]],
                            "outside_points": out})
    return bad


def _centers(specs: list[dict]) -> dict[str, list[list[float]]]:
    return {s["id"]: list(s.get("placements") or []) for s in specs}


def group_distances(specs: list[dict], groups: list[dict]) -> list[dict]:
    """For a semantic group, measure pairwise 2D center distances (metres)."""
    cs = _centers(specs)
    out = []
    for g in groups:
        ids = [i for i in g.get("ids", []) if i in cs and cs[i]]
        if len(ids) < 2:
            continue
        ds = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for pa in cs[ids[a]]:
                    for pb in cs[ids[b]]:
                        ds.append({"a": ids[a], "b": ids[b],
                                   "distance_m": round(math.hypot(pa[0]-pb[0], pa[1]-pb[1]), 3)})
        out.append({"group": g.get("name") or g.get("id"), "max_gap_m": g.get("max_gap_m"),
                    "pairs": ds,
                    "ok": all(d["distance_m"] <= float(g.get("max_gap_m") or 999) for d in ds)})
    return out


def clamp_to_polygon(inst: list[float], size: list[float], polygon: list[list[float]],
                     step: float = .05, max_iter: int = 80) -> list[float]:
    """Move a whole instance toward the polygon centroid until its AABB is inside.

    The object is translated as a rigid body; yaw is preserved. Used only when a pose repair
    has pushed an object through a wall.
    """
    if not outside_instances([{"id": "_", "size_m": size, "placements": [inst]}], polygon):
        return inst
    cx = sum(float(p[0]) for p in polygon) / len(polygon)
    cy = sum(float(p[1]) for p in polygon) / len(polygon)
    p = [float(v) for v in inst]
    for _ in range(max_iter):
        dx, dy = cx - p[0], cy - p[1]
        n = math.hypot(dx, dy) or 1.0
        p[0] += dx / n * step
        p[1] += dy / n * step
        if not outside_instances([{"id": "_", "size_m": size, "placements": [p]}], polygon):
            return p
    return p


def report(specs: list[dict], polygon: list[list[float]], groups: list[dict]) -> dict[str, Any]:
    bad = outside_instances(specs, polygon)
    gd = group_distances(specs, groups)
    return {"outside_count": len(bad), "outside": bad, "groups": gd,
            "ok": len(bad) == 0 and all(g["ok"] for g in gd)}
