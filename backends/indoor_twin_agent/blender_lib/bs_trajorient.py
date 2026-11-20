"""Estimate each object's rough direction from the walkthrough path.

The trajectory is not a metric floor plan.  What it *does* tell us is: near which phase of the
walk an object is first visible, on which side of the path it lies, and roughly how far it is.
That is enough to place a foreground object like the yoga ball in the corner formed by two
barriers, instead of keeping it at an arbitrary coordinate.
"""
from __future__ import annotations

import math
from typing import Any


def annotate(plan: dict[str, Any], path: list[list[float]]) -> list[dict]:
    if not path:
        return []
    # path rows: x,y,z, dirx,diry,dirz
    rows = []
    for o in (list(plan.get("objects") or []) + list(plan.get("inferred") or [])):
        insts = list(o.get("instances") or [])
        if isinstance(o.get("place"), dict) and o["place"].get("x") is not None:
            insts.append(o["place"])
        for n, inst in enumerate(insts):
            x, y = float(inst.get("x", 0)), float(inst.get("y", 0))
            best = min(range(len(path)), key=lambda i: (path[i][0] - x) ** 2 + (path[i][1] - y) ** 2)
            p = path[best]
            dx, dy = x - p[0], y - p[1]
            dist = math.hypot(dx, dy)
            # angle 0 = along the path forward, + = left of the path
            fwd = math.atan2(p[4], p[3])
            bearing = math.atan2(dy, dx) - fwd
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            phase = round(best / max(1, len(path) - 1), 3)
            rows.append({
                "id": o.get("id"), "kind": o.get("kind"), "instance": n,
                "x": round(x, 2), "y": round(y, 2),
                "nearest_path_index": best, "phase_t": phase,
                "distance_m": round(dist, 2),
                "side": "left" if bearing >= 0 else "right",
                "bearing_deg": round(math.degrees(bearing), 1),
                "phase_name": ("start" if phase < .25 else "early" if phase < .5
                               else "late" if phase < .8 else "end"),
            })
    return rows
