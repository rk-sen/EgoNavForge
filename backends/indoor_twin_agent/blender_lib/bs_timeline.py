"""Trajectory-ordered layout reasoning + collision repair.

Video is a sequence.  A row of three blue screens seen on the right at the beginning, then more
screens on the right after the turn, implies screens on both sides of the walked lane; it does
not imply four arbitrary outer walls.  This module annotates each planned instance with the
nearest path phase / side / bearing and reports AABB collisions, so the planner can move a rack
out of a barrier instead of discovering the intersection in a hero render.
"""
from __future__ import annotations

import math
from typing import Any


def _instances(o: dict) -> list[dict]:
    out = list(o.get("instances") or [])
    if isinstance(o.get("place"), dict) and o["place"].get("x") is not None:
        out.append(o["place"])
    return out


def annotate(plan: dict, path: list[list[float]]) -> list[dict]:
    rows = []
    if not path:
        return rows
    for oi, o in enumerate(list(plan.get("objects") or []) + list(plan.get("inferred") or [])):
        for ii, inst in enumerate(_instances(o)):
            x, y = float(inst.get("x", 0)), float(inst.get("y", 0))
            best = min(range(len(path)), key=lambda i: (path[i][0]-x)**2 + (path[i][1]-y)**2)
            p = path[best]
            fwd = math.atan2(p[4], p[3])
            bearing = math.atan2(y-p[1], x-p[0]) - fwd
            bearing = (bearing + math.pi) % (2*math.pi) - math.pi
            phase = best / max(1, len(path)-1)
            rows.append({
                "id": o.get("id"), "kind": o.get("kind"), "instance": ii,
                "x": round(x, 2), "y": round(y, 2),
                "phase_t": round(phase, 3),
                "phase_name": "start" if phase < .25 else "early" if phase < .5 else "late" if phase < .8 else "end",
                "side": "left" if bearing >= 0 else "right",
                "bearing_deg": round(math.degrees(bearing), 1),
                "distance_m": round(math.hypot(x-p[0], y-p[1]), 2),
            })
    return rows


def _dims(o: dict) -> tuple[float, float, float]:
    s = list(o.get("size_m") or o.get("dims") or [.5, .5, .5])
    return (float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else .5)


def _aabb(x, y, w, d, yaw_deg=0.0):
    a = math.radians(float(yaw_deg or 0))
    hx = abs(w*.5*math.cos(a)) + abs(d*.5*math.sin(a))
    hy = abs(w*.5*math.sin(a)) + abs(d*.5*math.cos(a))
    return (x-hx, x+hx, y-hy, y+hy)


def collision_pairs(plan: dict, clearance: float = .02) -> list[dict]:
    items = []
    for o in list(plan.get("objects") or []):
        w, d, _ = _dims(o)
        for n, inst in enumerate(_instances(o)):
            items.append((o.get("id"), n, _aabb(float(inst.get("x", 0)), float(inst.get("y", 0)),
                                               w, d, inst.get("pose_yaw_deg", 0))))
    out = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a, b = items[i], items[j]
            if a[0] == b[0]:            # same object's instances are allowed to touch
                continue
            ix = min(a[2][1], b[2][1]) - max(a[2][0], b[2][0]) + clearance
            iy = min(a[2][3], b[2][3]) - max(a[2][2], b[2][2]) + clearance
            if ix > 0 and iy > 0:
                out.append({"a": f"{a[0]}#{a[1]}", "b": f"{b[0]}#{b[1]}",
                            "overlap": [round(ix, 3), round(iy, 3)]})
    return out


def push_out_of_barriers(plan: dict, movable: tuple[str, ...], sides=("north",)) -> dict:
    """Move movable office furniture out of the barrier band, preserving the visual order.

    A rack or desk that is *behind* a screen should be beyond the barrier's thickness; otherwise
    the first hero render shows a shelf inside the fabric.  This is a layout repair, not a
    per-object shape hack.
    """
    moved = []
    for o in plan.get("objects") or []:
        if o.get("id") not in movable:
            continue
        w, d, h = _dims(o)
        for inst in _instances(o):
            for b in plan.get("objects") or []:
                if b.get("kind") != "barrier":
                    continue
                for bi in b.get("instances") or []:
                    bw, bd, _ = _dims(b)
                    ax = _aabb(float(bi.get("x", 0)), float(bi.get("y", 0)), bw, bd,
                               bi.get("pose_yaw_deg", 0))
                    bx = _aabb(float(inst.get("x", 0)), float(inst.get("y", 0)), w, d,
                               inst.get("pose_yaw_deg", 0))
                    ix = min(ax[1], bx[1]) - max(ax[0], bx[0])
                    iy = min(ax[3], bx[3]) - max(ax[2], bx[2])
                    if ix > -.02 and iy > -.02:
                        # furniture rows in this lab sit north of the barrier; push north
                        inst["y"] = round(max(float(inst.get("y", 0)), ax[3] + d*.5 + .08), 3)
                        if isinstance(o.get("place"), dict) and o["place"] is inst:
                            o["place"]["y"] = inst["y"]
                        moved.append({"id": o["id"], "to_y": inst["y"], "from_barrier": b.get("id")})
    return {"moved": moved, "collisions": collision_pairs(plan)}
