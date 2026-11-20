"""V10 floor-plan geometry: arbitrary rectilinear footprint + every opening on it.

v9 took `architecture.room_bbox_m`, built four walls, and read only windows[0]/doors[0].
That is why an L-shaped lab became a 6x9 rectangle with one window and one door. This module
accepts either an explicit `footprint_m` polygon or a list of rectangular `rooms` (union),
decomposes the boundary into directed wall segments, and places each window/door on the
nearest segment. It is deterministic and never uses the trajectory as room size.
"""
from __future__ import annotations

import math
from typing import Any

import bpy

import bs_geo
import bs_mat


def _clean_points(points):
    out = []
    for p in points or []:
        try:
            x, y = float(p[0]), float(p[1])
        except Exception:
            continue
        if not out or abs(out[-1][0] - x) > 1e-6 or abs(out[-1][1] - y) > 1e-6:
            out.append((x, y))
    if len(out) > 3 and out[0] == out[-1]:
        out.pop()
    # remove collinear triples
    changed = True
    while changed and len(out) > 4:
        changed = False
        for i in range(len(out)):
            a, b, c = out[i - 1], out[i], out[(i + 1) % len(out)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cross) < 1e-6:
                out.pop(i)
                changed = True
                break
    return out


def _signed_area(points):
    return 0.5 * sum(points[i][0] * points[(i + 1) % len(points)][1] -
                     points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))


def _ccw(points):
    p = _clean_points(points)
    if len(p) >= 3 and _signed_area(p) < 0:
        p = list(reversed(p))
    return p


def union_outline(rects):
    """Boundary of union of axis-aligned rectangles, returned as a CCW polygon.

    Uses a compressed grid and emits only cells whose neighbour is outside. Shared internal
    edges cancel naturally, so two overlapping rectangles produce one L-shaped outline.
    """
    rr = []
    for r in rects or []:
        try:
            x0, x1, y0, y1 = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        except Exception:
            continue
        if x1 > x0 and y1 > y0:
            rr.append((x0, x1, y0, y1))
    if not rr:
        return []
    xs = sorted({v for r in rr for v in (r[0], r[1])})
    ys = sorted({v for r in rr for v in (r[2], r[3])})
    inside = set()
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(r[0] <= cx <= r[1] and r[2] <= cy <= r[3] for r in rr):
                inside.add((i, j))
    segs = []
    for i, j in sorted(inside):
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
        if (i, j - 1) not in inside:
            segs.append(((x0, y0), (x1, y0)))       # bottom: rightwards
        if (i + 1, j) not in inside:
            segs.append(((x1, y0), (x1, y1)))       # right: upwards
        if (i, j + 1) not in inside:
            segs.append(((x1, y1), (x0, y1)))       # top: leftwards
        if (i - 1, j) not in inside:
            segs.append(((x0, y1), (x0, y0)))       # left: downwards
    # stitch directed segments into loops
    by_start: dict[tuple, list[tuple]] = {}
    for a, b in segs:
        by_start.setdefault(a, []).append((a, b))
    loops = []
    while by_start:
        start = next(iter(by_start))
        cur = start
        loop = [cur]
        while True:
            cand = by_start.get(cur) or []
            if not cand:
                break
            a, b = cand.pop(0)
            if not cand:
                by_start.pop(cur, None)
            cur = b
            loop.append(cur)
            if cur == start or len(loop) > len(segs) + 2:
                break
        if len(loop) >= 4:
            loops.append(_clean_points(loop[:-1]))
    if not loops:
        return []
    return _ccw(max(loops, key=lambda p: abs(_signed_area(p))))


def footprint(plan, cal=None):
    """Return (polygon, source). Explicit polygon > room union > bbox fallback."""
    arch = plan.get("architecture") or {}
    fp = arch.get("footprint_m") or arch.get("floorplan_m") or arch.get("outline_m")
    if isinstance(fp, list) and len(fp) >= 3:
        p = _ccw(fp)
        if len(p) >= 3:
            return p, "footprint_m"
    rooms = arch.get("rooms") or arch.get("zones") or []
    if isinstance(rooms, list) and rooms:
        rects = []
        for r in rooms:
            if isinstance(r, dict):
                v = r.get("rect") or r.get("bbox_m") or r.get("bbox")
                if isinstance(v, (list, tuple)) and len(v) >= 4:
                    rects.append(v)
        if rects:
            p = union_outline(rects)
            if len(p) >= 3:
                return p, "rooms_union"
    raw = arch.get("room_bbox_m") or arch.get("bbox_m") or arch.get("room_size_m")
    if isinstance(raw, dict):
        raw = [raw.get("x0", -4), raw.get("x1", 4), raw.get("y0", -4), raw.get("y1", 4)]
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x0, x1, y0, y1 = (float(v) for v in raw[:4])
    else:
        b = (cal or {}).get("room_bbox_m") or [-4, 4, -4, 4, 2.7]
        x0, x1, y0, y1 = (float(v) for v in b[:4])
    if x1 - x0 < 1.0:
        x1 = x0 + 1.0
    if y1 - y0 < 1.0:
        y1 = y0 + 1.0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "bbox"


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), max(xs), min(ys), max(ys)


def _edges(points):
    out = []
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-5:
            continue
        if abs(dy) < 1e-6:
            # Horizontal edge: classify by the actual y, not by vertex order.  The old
            # order-based comparison labelled the west vertical edge as E for an equal-x
            # edge, which put all west windows on the east wall.
            side = "S" if abs(a[1] - min(p[1] for p in points)) < 1e-5 else "N"
        elif abs(dx) < 1e-6:
            side = "W" if abs(a[0] - min(p[0] for p in points)) < 1e-5 else "E"
        else:
            side = ""
        out.append({"p1": a, "p2": b, "dx": dx, "dy": dy, "len": L,
                    "ux": dx / L, "uy": dy / L, "side": side,
                    "axis": "X" if abs(dy) < 1e-6 else ("Y" if abs(dx) < 1e-6 else "")})
    return out


def _project(edge, p):
    vx, vy = p[0] - edge["p1"][0], p[1] - edge["p1"][1]
    t = max(0.0, min(edge["len"], vx * edge["ux"] + vy * edge["uy"]))
    return t, math.hypot(vx - t * edge["ux"], vy - t * edge["uy"])


def _closest_edge(edges, p):
    return min(edges, key=lambda e: _project(e, p)[1])


def _edge_point(edge, u):
    return (edge["p1"][0] + edge["ux"] * u, edge["p1"][1] + edge["uy"] * u)


def opening_entries(arch):
    """Flatten windows/doors with counts; each entry becomes one physical opening."""
    out = []
    for kind, key in (("window", "windows"), ("door", "doors")):
        raw = arch.get(key) or []
        if not isinstance(raw, list):
            continue
        for e in raw:
            if not isinstance(e, dict):
                continue
            n = max(1, int(e.get("count") or 1))
            width = float(e.get("width_m") or e.get("width") or (1.4 if kind == "window" else 0.95))
            height = float(e.get("height_m") or e.get("height") or (1.5 if kind == "window" else 2.05))
            z0 = float(e.get("z0_m") or e.get("z_bottom_m") or (0.9 if kind == "window" else 0.0))
            base_u = e.get("u_m")
            center = e.get("center_m") or e.get("center")
            side = str(e.get("side") or e.get("wall") or "").upper()[:1]
            for i in range(n):
                c = None
                if isinstance(center, (list, tuple)) and len(center) >= 2:
                    c = (float(center[0]), float(center[1]))
                u = float(base_u) if base_u is not None else None
                if n > 1:
                    # distribute around the supplied anchor; if none, centre on the wall
                    step = width + 0.35
                    off = (i - (n - 1) / 2) * step
                    if c is not None:
                        c = (c[0] + 0 * off, c[1])   # actual offset is along the edge below
                    elif u is not None:
                        u = u + off
                out.append({"kind": kind, "width": width, "height": height, "z0": z0,
                            "u": u, "center_m": c, "side": side, "count": n, "index": i,
                            "blinds": bool(e.get("blinds") or e.get("blind")),
                            "blind": e.get("blind") or ("full" if e.get("blinds") else "none"),
                            "mullions": e.get("mullions") or {},
                            "curtain": e.get("curtain") or {},
                            "type": e.get("type") or "plain_glass",
                            "name": e.get("name") or e.get("id") or ""})
    return out


def _resolve(entry, edges, points):
    """Return (edge, u_center, point). Counts on one wall are distributed, never stacked."""
    if entry.get("center_m") is not None:
        e = _closest_edge(edges, entry["center_m"])
        u, _ = _project(e, entry["center_m"])
        if int(entry.get("count") or 1) > 1:
            step = float(entry["width"]) + 0.35
            u = max(float(entry["width"]) / 2,
                    min(e["len"] - float(entry["width"]) / 2,
                        u + (int(entry.get("index") or 0) - (int(entry["count"]) - 1) / 2) * step))
        return e, u, _edge_point(e, u)
    side = entry.get("side") or ""
    candidates = [x for x in edges if x["side"] == side] if side else list(edges)
    if not candidates:
        candidates = list(edges)
    # choose the longest segment on that side (an L has two W/E edges sometimes)
    e = max(candidates, key=lambda x: x["len"])
    L = e["len"]
    w = float(entry["width"])
    u_in = entry.get("u")
    if u_in is not None:
        # u_m is measured from the min-coordinate corner on the named side (intuitive)
        bx0, bx1, by0, by1 = bbox(points)
        if side in ("N", "S"):
            # choose the edge whose y is on that side, u from left
            p = (bx0 + float(u_in), by1 if side == "N" else by0)
        else:
            p = (bx1 if side == "E" else bx0, by0 + float(u_in))
        e = _closest_edge(edges, p)
        u, _ = _project(e, p)
        return e, u, _edge_point(e, u)
    # centre on the chosen edge; multiple entries get a deterministic row
    n = int(entry.get("count") or 1)
    if n > 1:
        step = w + 0.35
        span = n * step - 0.35
        start = max(w / 2, (L - span) / 2 + w / 2)
        u = start + int(entry.get("index") or 0) * step
        u = min(L - w / 2, u)
    else:
        u = 0.5 * L
    return e, u, _edge_point(e, u)




def _line_intervals(poly, axis, value):
    """Where a horizontal (axis='y') / vertical (axis='x') line lies inside a polygon."""
    hits = []
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if axis == "y":
            if abs(y2 - y1) < 1e-9:
                continue
            if min(y1, y2) <= value <= max(y1, y2):
                t = (value - y1) / (y2 - y1)
                if 0 <= t <= 1:
                    hits.append(x1 + t * (x2 - x1))
        else:
            if abs(x2 - x1) < 1e-9:
                continue
            if min(x1, x2) <= value <= max(x1, x2):
                t = (value - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    hits.append(y1 + t * (y2 - y1))
    hits = sorted(set(round(v, 4) for v in hits))
    out = []
    for a, b in zip(hits[0::2], hits[1::2]):
        if abs(b - a) > 0.25:
            out.append((a, b))
    return out


def tile_seams(points, mat):
    """Physical 1.2 m rubber-tile seams, clipped to the L-shaped floor polygon.

    The shader-only grid was not visible in a Workbench/Cycles material preview; these thin
    strips guarantee the large square floor joints the video shows.
    """
    bx0, bx1, by0, by1 = bbox(points)
    # Keep the seam *inlaid*: a raised strip catches the ceiling light and reads as a white
    # grid, which is not what the source floor looks like. 0.8 mm / 0.003 m and near-black.
    seam_mat = bs_mat.preset("M_FloorSeam", "rubber", (0.045, 0.045, 0.04))
    z, th = 0.0008, 0.003
    # start at the bbox edge so lines land on a regular 1.2m grid
    x = math.floor(bx0 / 1.2) * 1.2
    while x <= bx1 + 1e-6:
        for a, b in _line_intervals(points, "x", x):
            bs_geo.box("FloorSeamX_%05.2f" % x, (th, b - a, th),
                       loc=(x, (a + b) / 2, z), mat=seam_mat, bevel=.0, coll_name="Shell")
        x += 1.2
    y = math.floor(by0 / 1.2) * 1.2
    while y <= by1 + 1e-6:
        for a, b in _line_intervals(points, "y", y):
            bs_geo.box("FloorSeamY_%05.2f" % y, (b - a, th, th),
                       loc=((a + b) / 2, y, z), mat=seam_mat, bevel=.0, coll_name="Shell")
        y += 1.2


def build_shell(plan, cal, mats, height, proxy=True):
    """Build a polygon shell with every window/door, return metadata (no render)."""
    points, source = footprint(plan, cal)
    if len(points) < 3:
        raise ValueError("empty footprint")
    bx0, bx1, by0, by1 = bbox(points)
    edges = _edges(points)
    # floor + ceiling: one polygon prism, not a bbox slab
    bs_geo.poly_slab("Floor", points, -0.20, 0.0, mat=mats["floor"])
    tile_seams(points, mats["floor"])
    bs_geo.poly_slab("Ceiling", points, height - 0.15, height, mat=mats["ceil"])

    entries = opening_entries(plan.get("architecture") or {})
    for e in entries:
        e["_edge"], e["_u"], e["_p"] = _resolve(e, edges, points)
        edge = e["_edge"]
        # Keep every opening fully on its wall segment.  The raw model u_m can land on a
        # corner: a 0.95 m door centered at u=0 becomes a tiny strip and a 1.4 m window
        # extends past the wall.  Clamp both width and center, never silently leave the wall.
        kind = str(e.get("kind") or "window")
        wmin = .65 if kind == "door" else .45
        e["width"] = min(float(e.get("width") or (1.4 if kind == "window" else .95)),
                         max(wmin, float(edge["len"]) - .30))
        half = e["width"] / 2 + .06
        e["_u"] = max(half, min(float(edge["len"]) - half, float(e["_u"])))
    holes = {id(e): [] for e in edges}
    for ent in entries:
        e = ent["_edge"]
        w, h, z0 = float(ent["width"]), float(ent["height"]), float(ent["z0"])
        u0 = max(0.0, ent["_u"] - w / 2)
        u1 = min(e["len"], ent["_u"] + w / 2)
        holes.setdefault(id(e), []).append((u0, u1, z0, z0 + h))
        ent["u0"], ent["u1"] = u0, u1

    walls = []
    for i, e in enumerate(edges):
        walls.append(bs_geo.wall(f"Wall_{i}", e["p1"], e["p2"], height, thickness=0.2,
                                 holes=holes.get(id(e), []), mat=mats["wall"]))
        e["wall"] = walls[-1]

    # frames / glass / door leaves
    win_n = door_n = 0
    for ent in entries:
        e, u = ent["_edge"], ent["_u"]
        cx, cy = _edge_point(e, u)
        normal = "Y" if e["axis"] == "X" else "X"
        zc = ent["z0"] + ent["height"] / 2
        if ent["kind"] == "window":
            nm = f"Window_{win_n}"
            side_sign = {"W": -1.0, "E": 1.0, "S": -1.0, "N": 1.0}.get(str(ent.get("side") or e.get("side")), 1.0)
            bs_geo.window(nm, (cx, cy, zc), ent["width"], ent["height"], normal=normal,
                          frame_mat=mats["frame"], glass_mat=mats["glass"], proxy=True,
                          proxy_sign=side_sign)
            # Window mullions: a plan field, not "one white rectangle". cols/rows are the number
            # of internal bars; 0 means a single unbroken pane.
            mul = ent.get("mullions") or {}
            cols = max(0, int(mul.get("cols", 0)))
            rows = max(0, int(mul.get("rows", 0)))
            for ci in range(1, cols + 1):
                off = (ci / (cols + 1) - .5) * ent["width"] * .94
                dims = (.032, .045, ent["height"] * .90) if normal == "Y" else (.045, .032, ent["height"] * .90)
                loc = (cx + off, cy, zc) if normal == "Y" else (cx, cy + off, zc)
                bs_geo.box(f"Mullion_{win_n}_{ci}", dims, loc=loc, mat=mats["frame"], bevel=.004)
            for ri in range(1, rows + 1):
                off = (ri / (rows + 1) - .5) * ent["height"] * .90
                dims = (ent["width"] * .94, .045, .032) if normal == "Y" else (.045, ent["width"] * .94, .032)
                loc = (cx, cy, zc + off)
                bs_geo.box(f"MullionH_{win_n}_{ri}", dims, loc=loc, mat=mats["frame"], bevel=.004)
            win_n += 1
            blind = ent.get("blind") or ("full" if ent.get("blinds") else "none")
            if blind in ("full", "top_half"):
                blind_m = bs_mat.principled("M_Shell_Blind", (0.72, 0.69, 0.61, 1), rough=0.86)
                z_lo = ent["z0"] + (ent["height"] * .5 if blind == "top_half" else 0)
                z_hi = ent["z0"] + ent["height"]
                nslat = 8
                for bi in range(nslat):
                    zz = z_lo + (bi + .5) * (z_hi - z_lo) / nslat
                    dims = (ent["width"] * .92, .022, .012) if normal == "Y" else (.022, ent["width"] * .92, .012)
                    bs_geo.box(f"Blind_{win_n}_{bi}", dims, loc=(cx, cy, zz), mat=blind_m, bevel=.002)
            # A curtain is a separate evidence-led schema from "blind".  The video shows a
            # few windows with a half-drawn opaque curtain, so this must be baked per window,
            # not attached by default.  Build it from vertical folded panels plus a head rail.
            cur = ent.get("curtain") or {}
            try:
                cur_draw = max(.0, min(1.0, float(cur.get("draw", 0))))
            except Exception:
                cur_draw = 0.0
            if cur_draw > 0.02:
                curtain_m = bs_mat.principled(str(cur.get("material") or "M_Shell_Curtain"),
                                              tuple(cur.get("color") or (0.86, 0.84, 0.78, 1)),
                                              rough=.92)
                fw = ent["width"] * .92
                fh = ent["height"] * .88
                panel_w = fw * cur_draw
                side = str(cur.get("side") or "left").lower()
                nseg = 7
                if normal == "Y":
                    # Wall runs along X: curtain panels vary in X.
                    x0 = cx - fw / 2 if side == "left" else cx + fw / 2 - panel_w
                    for ci in range(nseg):
                        xx = x0 + (ci + .5) * panel_w / nseg
                        fold = .018 * math.sin((ci + .5) / nseg * math.pi * 2.0)
                        dims = (panel_w / nseg * .92, .026 + abs(fold), fh)
                        loc = (xx, cy - .055 - fold, zc)
                        bs_geo.box(f"Curtain_{win_n}_{ci}", dims, loc=loc, mat=curtain_m, bevel=.003)
                    bs_geo.box(f"CurtainRail_{win_n}", (panel_w + .05, .035, .035),
                               loc=(x0 + panel_w / 2, cy - .070, zc + fh / 2), mat=mats["frame"], bevel=.004)
                    bs_geo.cylinder(f"CurtainHook_{win_n}_0", .010, .07,
                                    loc=(x0 + .02, cy - .06, zc + fh / 2 - .03), mat=mats["frame"], seg=8)
                    bs_geo.cylinder(f"CurtainHook_{win_n}_1", .010, .07,
                                    loc=(x0 + panel_w - .02, cy - .06, zc + fh / 2 - .03), mat=mats["frame"], seg=8)
                else:
                    # Wall runs along Y (west/east wall): panels vary in Y, not X.  The old
                    # branch varied X for both normals, so a curtain for a west window was
                    # built near y=0 instead of at the window's y.
                    y0 = cy - fw / 2 if side == "left" else cy + fw / 2 - panel_w
                    for ci in range(nseg):
                        yy = y0 + (ci + .5) * panel_w / nseg
                        fold = .018 * math.sin((ci + .5) / nseg * math.pi * 2.0)
                        dims = (.026 + abs(fold), panel_w / nseg * .92, fh)
                        loc = (cx - .055 - fold, yy, zc)
                        bs_geo.box(f"Curtain_{win_n}_{ci}", dims, loc=loc, mat=curtain_m, bevel=.003)
                    bs_geo.box(f"CurtainRail_{win_n}", (.035, panel_w + .05, .035),
                               loc=(cx - .070, y0 + panel_w / 2, zc + fh / 2), mat=mats["frame"], bevel=.004)
                    bs_geo.cylinder(f"CurtainHook_{win_n}_0", .010, .07,
                                    loc=(cx - .06, y0 + .02, zc + fh / 2 - .03), mat=mats["frame"], seg=8)
                    bs_geo.cylinder(f"CurtainHook_{win_n}_1", .010, .07,
                                    loc=(cx - .06, y0 + panel_w - .02, zc + fh / 2 - .03), mat=mats["frame"], seg=8)
            if False and proxy:
                # Legacy 0.12 m quad removed: it sat immediately behind the glass and hid
                # the new vertical sky/skyline proxy.  Exterior proxy is built by bs_geo.window.
                n = (e["uy"], -e["ux"]) if e["axis"] == "X" else (e["uy"], -e["ux"])
                out = 0.12
                px, py = cx + n[0] * out, cy + n[1] * out
                dims = (ent["width"] * 1.15, ent["height"] * 1.15)
                rot = (90, 0, 0) if normal == "Y" else (0, 90, 0)
                bs_geo.plane(f"WindowProxy_{win_n}", dims, (px, py, zc),
                             rot_xyz=rot, mat=mats["proxy"])
        else:
            nm = f"Door_{door_n}"
            door_n += 1
            leaf = (ent["width"] - .05, .05, ent["height"] - .04) if normal == "Y" else \
                   (.05, ent["width"] - .05, ent["height"] - .04)
            bs_geo.box(f"{nm}_leaf", leaf, loc=(cx, cy, ent["height"] / 2), mat=mats["door"])
            top = (ent["width"], .08, .06) if normal == "Y" else (.08, ent["width"], .06)
            bs_geo.box(f"{nm}_frame_top", top, loc=(cx, cy, ent["height"]), mat=mats["frame"])
            hx = cx + (ent["width"] * .34) * (1 if e["axis"] == "X" else 0)
            hy = cy + (ent["width"] * .34) * (1 if e["axis"] == "Y" else 0)
            axis = "Y" if normal == "Y" else "X"
            bs_geo.cylinder(f"{nm}_handle", .018, .09, loc=(hx, hy, ent["height"] * .45),
                            axis=axis, mat=mats["frame"], seg=10)

    # Linear ceiling lights.  The source has long suspended strip bars, not short slots:
    # the plan carries count/length/rows explicitly; no per-room "one per 2.2 m" guess.
    ls = (plan.get("architecture") or {}).get("lights") or {}
    if isinstance(ls, dict):
        nlights = max(1, min(12, int(ls.get("count", 6))))
        bar_len = float(ls.get("length_m", 3.2))
        rows = max(1, min(nlights, int(ls.get("rows", 2))))
        mode = str(ls.get("mode", "long_strip"))
    else:
        nlights, bar_len, rows, mode = 6, 3.2, 2, "long_strip"
    for i in range(nlights):
        row = i % rows
        col = i // rows
        ncol = max(1, math.ceil(nlights / rows))
        x = bx0 + (col + .5) * (bx1 - bx0) / ncol
        y = by0 + (row + .5) * (by1 - by0) / rows
        if mode == "long_strip":
            dims = (min(bar_len, (bx1 - bx0) / ncol * .96), .12, .06)
        else:
            dims = (.12, min(bar_len, (by1 - by0) / rows * .82), .06)
        bs_geo.box(f"Ceiling_Lamp_{i+1}", dims, loc=(x, y, height - .18), mat=mats["light"])
        # visible suspension stems: otherwise the long bar reads as a floating slot
        for sx in (-1, 1):
            if mode == "long_strip":
                stem = (x + sx * dims[0] * .34, y, height - .10)
            else:
                stem = (x, y + sx * dims[1] * .34, height - .10)
            bs_geo.cylinder(f"LampStem_{i+1}_{sx+1}", .012, .18, loc=stem, mat=mats["frame"], seg=8)
    bs_mat.area_light("Light_Room", ((bx0 + bx1) / 2, (by0 + by1) / 2, height - .30),
                      ((bx0 + bx1) / 2, (by0 + by1) / 2, 0), energy=48, size=max(2.8, (bx1 - bx0) * .45))
    bs_mat.area_light("Light_Window", (bx0 + .6, (by0 + by1) / 2, height * .55),
                      ((bx0 + bx1) / 2, (by0 + by1) / 2, 1.0), energy=24, size=2.2)
    return {"footprint": [[round(x, 2), round(y, 2)] for x, y in points],
            "source": source, "bbox": [bx0, bx1, by0, by1], "walls": len(walls),
            "windows": win_n, "doors": door_n, "height": height,
            "openings": [{"kind": e["kind"], "width": e["width"], "height": e["height"],
                          "z0": e["z0"], "point": [round(e["_p"][0], 2), round(e["_p"][1], 2)],
                          "side": e["_edge"]["side"]} for e in entries]}
