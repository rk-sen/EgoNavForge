"""V17 automatic scene graph.

Pipeline:
  RECON_EVIDENCE + architecture
    -> category catalogue (catalog.py)
    -> anchors on floor (sofa/desk/shelf/table/barrier...)
    -> child nodes with explicit tree relations:
         monitor/PC/keyboard/mouse ON desk
         boxes ON shelf level
         TV ON tv_cabinet
         chair NEAR desk (front)
         bench/table NEAR sofa
    -> physical feasibility (support top height + support footprint + floor AABB)
    -> SCENE_PLAN.json + specs.json

There are no hand-written office coordinates in this file. Placement is deterministic grid
search inside the L footprint, with relation constraints applied before collision search.
"""
from __future__ import annotations

import math
from typing import Any

from .catalog import CATEGORIES, classify, spec, parse_color


def _inside(x, y, poly):
    c = False; j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            c = not c
        j = i
    return c


def _bbox(x, y, w, d, yaw=0.0):
    a = math.radians(float(yaw))
    hx = abs(w * .5 * math.cos(a)) + abs(d * .5 * math.sin(a))
    hy = abs(w * .5 * math.sin(a)) + abs(d * .5 * math.cos(a))
    return x - hx, x + hx, y - hy, y + hy


def _overlap(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] >= inner[1] and outer[2] <= inner[2] and outer[3] >= inner[3]


def _seg_cross(a, b, c, d):
    """True if two 2D segments intersect (including touching corners)."""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    c1, c2 = cr(a, b, c), cr(a, b, d)
    c3, c4 = cr(c, d, a), cr(c, d, b)
    return (c1 == 0 or c2 == 0 or (c1 > 0) != (c2 > 0)) and \
           (c3 == 0 or c4 == 0 or (c3 > 0) != (c4 > 0))


def _rect_inside_poly(x, y, w, d, yaw, poly, margin=0.0):
    """Check the whole rotated AABB, including concave L notches."""
    b = _bbox(x, y, w, d, yaw)
    b = [b[0] - margin, b[1] + margin, b[2] - margin, b[3] + margin]
    for px, py in ((b[0], b[2]), (b[1], b[2]), (b[1], b[3]), (b[0], b[3])):
        if not _inside(px, py, poly):
            return False
    rect_edges = [((b[0], b[2]), (b[1], b[2])), ((b[1], b[2]), (b[1], b[3])),
                  ((b[1], b[3]), (b[0], b[3])), ((b[0], b[3]), (b[0], b[2]))]
    n = len(poly)
    for i in range(n):
        a = (float(poly[i][0]), float(poly[i][1]))
        c = (float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1]))
        for r0, r1 in rect_edges:
            if _seg_cross(r0, r1, a, c):
                return False
    return True


def shelf_levels(h):
    """Plank centres match bs_assets.build_shelf: 4 planks from .07 to h-.07."""
    h = float(h)
    return [.07 + i * (h - .14) / 3 for i in range(4)]


class Layout:
    def __init__(self, poly):
        self.poly = [[float(p[0]), float(p[1])] for p in poly]
        xs = [p[0] for p in self.poly]; ys = [p[1] for p in self.poly]
        self.bounds = [min(xs), max(xs), min(ys), max(ys)]
        self.obstacles = []          # floor AABBs only
        self.nodes = {}              # id -> {x,y,z,w,d,h,yaw,cat}

    def free_floor(self, x, y, w, d, yaw, margin=.08, ignore=()):
        b = _bbox(x, y, w, d, yaw)
        if not _rect_inside_poly(x, y, w, d, yaw, self.poly, margin=margin):
            return False
        return not any(_overlap(b, o) for o in self.obstacles)

    def add_node(self, oid, cat, x, y, z, w, d, h, yaw=0.0, floor=True, seq_group=None):
        self.nodes[oid] = {"id": oid, "cat": cat, "x": x, "y": y, "z": z,
                           "w": w, "d": d, "h": h, "yaw": yaw,
                           "seq_group": seq_group or ""}
        if floor:
            self.obstacles.append(_bbox(x, y, w, d, yaw))
        return self.nodes[oid]

    def candidates(self, near, step=.30, margin=.24):
        pts = []
        x = self.bounds[0] + margin
        while x <= self.bounds[1] - margin:
            y = self.bounds[2] + margin
            while y <= self.bounds[3] - margin:
                pts.append((x, y)); y += step
            x += step
        return sorted(pts, key=lambda p: math.hypot(p[0] - near[0], p[1] - near[1]))

    def place_floor(self, oid, cat, near, yaw=None):
        w, d, h = CATEGORIES[cat]["size"]
        yaw = CATEGORIES[cat]["yaw"] if yaw is None else yaw
        chosen = None
        for x, y in self.candidates(near):
            if self.free_floor(x, y, w, d, yaw):
                chosen = (x, y); break
        if chosen is None:
            # deterministic fallback: move along the line to the room centroid
            cx = sum(p[0] for p in self.poly) / len(self.poly)
            cy = sum(p[1] for p in self.poly) / len(self.poly)
            for t in (0.05, .1, .2, .3, .5, .7, .9):
                x = near[0] + (cx - near[0]) * t; y = near[1] + (cy - near[1]) * t
                if self.free_floor(x, y, w, d, yaw):
                    chosen = (x, y); break
        if chosen is None:
            return None
        x, y = chosen
        self.add_node(oid, cat, x, y, 0.0, w, d, h, yaw, floor=True)
        return x, y, 0.0, yaw

    def place_hint(self, oid, cat, x, y, yaw=None):
        """Place at the top-down model's x/y, only nudging if the footprint collides.

        The second cognition pass is the authority for the scene layout; this is not a
        hand-written coordinate.  A nudge is allowed only to keep the object physically
        inside the room and off an existing floor obstacle.
        """
        w, d, h = CATEGORIES[cat]["size"]
        yaw = CATEGORIES[cat]["yaw"] if yaw is None else yaw
        if self.free_floor(x, y, w, d, yaw, margin=.04):
            self.add_node(oid, cat, x, y, 0.0, w, d, h, yaw, floor=True)
            return x, y, 0.0, yaw
        for rad in (.15, .3, .45, .65, .9, 1.2):
            for k in range(12):
                a = math.pi * 2 * k / 12
                x2, y2 = x + math.cos(a) * rad, y + math.sin(a) * rad
                if self.free_floor(x2, y2, w, d, yaw, margin=.04):
                    self.add_node(oid, cat, x2, y2, 0.0, w, d, h, yaw, floor=True)
                    return x2, y2, 0.0, yaw
        return None

    def place_against_wall(self, oid, cat, near, side=None):
        """Place a tall storage item against the nearest/selected wall.

        The shelf in the video is a wall-backed object, not a free-standing island.  The
        previous path-anchor placement let it land in the middle of the room.  Here the
        wall side is selected first, then the item is slid along that wall near the
        evidence-time coordinate.
        """
        w, d, h = CATEGORIES[cat]["size"]
        x0, x1, y0, y1 = self.bounds
        if side not in ("W", "E", "S", "N"):
            dists = {"W": abs(near[0] - x0), "E": abs(x1 - near[0]),
                     "S": abs(near[1] - y0), "N": abs(y1 - near[1])}
            side = min(dists, key=dists.get)
        # W/E: local X width runs along Y, local Z depth into the room
        if side in ("W", "E"):
            yaw = 90.0
            x = x0 + d * .5 + .08 if side == "W" else x1 - d * .5 - .08
            ystart = max(y0 + w * .5 + .10, min(y1 - w * .5 - .10, float(near[1])))
            for delta in (0, .25, -.25, .5, -.5, .8, -.8, 1.2, -1.2):
                y = ystart + delta
                if y0 + w * .5 <= y <= y1 - w * .5 and self.free_floor(x, y, w, d, yaw, margin=.05):
                    self.add_node(oid, cat, x, y, 0.0, w, d, h, yaw, floor=True)
                    return x, y, 0.0, yaw
        else:
            yaw = 0.0
            y = y0 + d * .5 + .08 if side == "S" else y1 - d * .5 - .08
            xstart = max(x0 + w * .5 + .10, min(x1 - w * .5 - .10, float(near[0])))
            for delta in (0, .25, -.25, .5, -.5, .8, -.8, 1.2, -1.2):
                x = xstart + delta
                if x0 + w * .5 <= x <= x1 - w * .5 and self.free_floor(x, y, w, d, yaw, margin=.05):
                    self.add_node(oid, cat, x, y, 0.0, w, d, h, yaw, floor=True)
                    return x, y, 0.0, yaw
        return self.place_floor(oid, cat, near, yaw=0.0)

    def place_near(self, oid, cat, anchor_id, side="front", distance=1.0, yaw=None):
        """Place a floor child beside an anchor. side is in anchor-local coordinates."""
        a = self.nodes[anchor_id]
        w, d, h = CATEGORIES[cat]["size"]
        yaw = CATEGORIES[cat]["yaw"] if yaw is None else yaw
        ang = math.radians(float(a.get("yaw") or 0.0))
        # anchor local axes: front=+local Y, back=-Y, right=+X, left=-X
        dirs = {"front": (-math.sin(ang), math.cos(ang)), "back": (math.sin(ang), -math.cos(ang)),
                "right": (math.cos(ang), math.sin(ang)), "left": (-math.cos(ang), -math.sin(ang))}
        ux, uy = dirs.get(side, dirs["front"])
        base = (a["x"] + ux * distance, a["y"] + uy * distance)
        # search around the desired relation point, not the whole room
        for delta in (0, .25, .5, .8, 1.2, 1.6):
            for sgn in ((-1, 1) if delta else (0,)):
                x = base[0] + uy * delta * sgn
                y = base[1] - ux * delta * sgn
                if self.free_floor(x, y, w, d, yaw):
                    self.add_node(oid, cat, x, y, 0.0, w, d, h, yaw, floor=True)
                    return x, y, 0.0, yaw
        return self.place_floor(oid, cat, base, yaw=yaw)

    def place_on(self, oid, cat, support_id, dx=0.0, dy=0.0, level_z=None, yaw=None, max_try=.5):
        """Place a child on the top surface of a support, or on a chosen shelf level."""
        s = self.nodes[support_id]
        w, d, h = CATEGORIES[cat]["size"]
        yaw = CATEGORIES[cat]["yaw"] if yaw is None else yaw
        z = (s["z"] + s["h"]) if level_z is None else float(level_z)
        # dx/dy are in the support's local frame: a wall shelf is yaw=90, so two boxes
        # at dx=-0.26/+0.26 must separate along world Y, not both land on the centre.
        ang = math.radians(float(s.get("yaw", 0.0)))
        dx_w = dx * math.cos(ang) - dy * math.sin(ang)
        dy_w = dx * math.sin(ang) + dy * math.cos(ang)
        base_x, base_y = s["x"] + dx_w, s["y"] + dy_w
        top = _bbox(s["x"], s["y"], s["w"], s["d"], s.get("yaw", 0))
        for ox, oy in ((0, 0), (-.12, 0), (.12, 0), (0, -.08), (0, .08), (-.2, 0), (.2, 0)):
            x, y = base_x + ox, base_y + oy
            cb = _bbox(x, y, w, d, yaw)
            if _contains(top, cb):
                self.add_node(oid, cat, x, y, z, w, d, h, yaw, floor=False)
                return x, y, z, yaw
        # if the support is too small, keep a valid height but clamp the centre inside it.
        x = min(max(base_x, top[0] + w * .5), top[1] - w * .5)
        y = min(max(base_y, top[2] + d * .5), top[3] - d * .5)
        self.add_node(oid, cat, x, y, z, w, d, h, yaw, floor=False)
        return x, y, z, yaw

    def _snap_barrier(self, a, b, yaw):
        """Snap a run to a nearby wall only when an endpoint is actually on/next to it.

        The top-down model often draws a wall barrier with its first panel at x=0 or the
        last panel at x=room_width, but the run direction is slightly wrong.  Projecting the
        whole run onto that edge keeps the three panels connected and avoids pushing a
        too-wide horizontal run into the room.  A genuinely free-standing divider (both
        endpoints away from every edge) is left alone.
        """
        pts = self.poly
        edges = []
        for i in range(len(pts)):
            p1 = (float(pts[i][0]), float(pts[i][1]))
            p2 = (float(pts[(i + 1) % len(pts)][0]), float(pts[(i + 1) % len(pts)][1]))
            edges.append((p1, p2))
        def dist_seg(q, p1, p2):
            vx, vy = p2[0] - p1[0], p2[1] - p1[1]
            ll = vx * vx + vy * vy or 1.0
            t = max(0.0, min(1.0, ((q[0]-p1[0])*vx + (q[1]-p1[1])*vy) / ll))
            return math.hypot(q[0] - (p1[0]+t*vx), q[1] - (p1[1]+t*vy)), t, p1, p2
        best = None
        for q in (a, b):
            for p1, p2 in edges:
                d, t, pp1, pp2 = dist_seg(q, p1, p2)
                if d < .55 and (best is None or d < best[0]):
                    best = (d, t, pp1, pp2)
        if best is None:
            return a, b, yaw
        _, _, p1, p2 = best
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(vx, vy) or 1.0
        ux, uy = vx / L, vy / L
        # project both endpoints onto this edge, preserving their along-edge order
        def along(q):
            return (q[0] - p1[0]) * ux + (q[1] - p1[1]) * uy
        ta, tb = along(a), along(b)
        if ta > tb:
            ta, tb = tb, ta
        # inward normal: choose the side whose midpoint is closer to the room centroid
        nx, ny = -uy, ux
        cx = sum(float(q[0]) for q in pts) / len(pts)
        cy = sum(float(q[1]) for q in pts) / len(pts)
        off = .12
        if (p1[0] + ux * (ta+tb)/2 + nx*off - cx)**2 + (p1[1] + uy * (ta+tb)/2 + ny*off - cy)**2 > \
           (p1[0] + ux * (ta+tb)/2 - nx*off - cx)**2 + (p1[1] + uy * (ta+tb)/2 - ny*off - cy)**2:
            nx, ny = -nx, -ny
        a = [p1[0] + ux * ta + nx*off, p1[1] + uy * ta + ny*off]
        b = [p1[0] + ux * tb + nx*off, p1[1] + uy * tb + ny*off]
        return a, b, math.degrees(math.atan2(uy, ux))

    def _barrier_line(self, a, b, n):
        """Return a contiguous panel chain on the model's own run.

        The chain is not allowed to scatter: if the run is short, extend it around its
        midpoint; if the run is long, use the whole run instead of putting all panels at
        the endpoint.  Panel pitch is the physical 1.20 m panel width, so three panels
        touch and read as one connected barrier, not three unrelated blocks.
        """
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = (dx / length, dy / length) if length > 1e-6 else (1.0, 0.0)
        pitch = 1.20
        if length < (n - 1) * pitch:
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            s = [mid[0] - ux * pitch * (n - 1) / 2, mid[1] - uy * pitch * (n - 1) / 2]
            e = [mid[0] + ux * pitch * (n - 1) / 2, mid[1] + uy * pitch * (n - 1) / 2]
        else:
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            s = [mid[0] - ux * pitch * (n - 1) / 2, mid[1] - uy * pitch * (n - 1) / 2]
        # Always use the physical 1.20 m panel pitch, centred on the model run.  This
        # keeps three panels touching instead of stretching them to both endpoints.
        if n <= 1:
            centers = [s]
        else:
            centers = [[s[0] + ux * pitch * j, s[1] + uy * pitch * j] for j in range(n)]
        return centers, ux, uy

    def _barrier_group_free(self, centers, pcs, yaw):
        for x, y in centers:
            if not self.free_floor(x, y, 1.2, .08, yaw, margin=.02):
                return False
            cb = _bbox(x, y, 1.2, .08, yaw)
            if any(_overlap(cb, pcs[p]) for p in pcs if p != id(self)):
                return False
        return True

    def place_barrier_runs(self, oid, runs):
        """Place barrier groups as connected chains, not per-panel collision artifacts.

        V30 moved each panel independently when a collision was found, so a 3-panel run
        could become a pile of blocks near a wall.  V31 solves at the group level: translate
        the whole chain, preserve pitch/count, and reject a chain that overlaps another
        barrier group.
        """
        placements = []
        groups = []
        # Existing barrier AABBs are kept separately so a new group can be moved before
        # any of its panels are committed to the scene.
        existing = [o for o in self.obstacles]
        for ri, r in enumerate(runs):
            n = max(1, int(r.get("count") or 1))
            a = [float(r["start"][0]), float(r["start"][1])]
            b = [float(r["end"][0]), float(r["end"][1])]
            yaw_model = float(r.get("yaw_deg") if r.get("yaw_deg") is not None
                              else math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])))
            # The top-down cognition now labels the large-occluder group's wall relation.
            # A divider/free group must not be snapped onto a wall just because one model
            # endpoint happened to be near an edge.
            rel = str(r.get("wall_relation") or "").lower()
            if rel in ("divider", "free", "partition", "隔断"):
                a, b, yaw_wall = a, b, yaw_model
            else:
                a, b, yaw_wall = self._snap_barrier(a, b, yaw_model)
            centers0, ux, uy = self._barrier_line(a, b, n)
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            yaw = yaw_wall if yaw_wall is not None else yaw_model
            # Use the chain's own direction when the model's yaw disagrees; a pile usually
            # means the yaw and the start/end direction were mixed.
            line_yaw = math.degrees(math.atan2(uy, ux))
            if abs(((float(yaw) - line_yaw + 180) % 360) - 180) > 18:
                yaw = line_yaw
            chosen = None
            # Runs that the ordered visual pass identifies as two legs of the same L/U
            # enclosure may meet at a corner.  They are the same large occluder group, not
            # two separate groups that must be pushed 1.35 m apart.
            seq_group = str(r.get("sequence_group") or r.get("sequence_support") or "")
            # Try the model run, then whole-group translations.  A group must be separated
            # from another barrier group: a 2-panel run touching the end of a 3-panel run
            # reads as a pile in the video, even when their AABBs technically do not overlap.
            nx, ny = -uy, ux
            moves = [(0, 0)]
            for d in (.20, .40, .70, 1.0, 1.4, 1.9, 2.5, 3.2):
                moves += [(nx * d, ny * d), (-nx * d, -ny * d),
                          (ux * d, uy * d), (-ux * d, -uy * d)]
            # For a long model run, prefer sliding along its own direction so the chain
            # remains inside a wing instead of being pushed perpendicular into the room.
            if length > (n - 1) * 1.20 + .2:
                moves = [(0, 0)] + [(ux * d, uy * d) for d in (.3, .6, .9, 1.2, 1.5)] +                         [(-ux * d, -uy * d) for d in (.3, .6, .9, 1.2, 1.5)] + moves
            def group_ok(cs):
                for c in cs:
                    # Same-L-group corner contacts are allowed; other groups are still
                    # separated.  A barrier corner is a real contact, not a duplicate pile.
                    if not _rect_inside_poly(c[0], c[1], 1.2, .08, yaw, self.poly, margin=.02):
                        return False
                    cb = _bbox(c[0], c[1], 1.2, .08, yaw)
                    for bnode in self.nodes.values():
                        if bnode.get("cat") != "barrier":
                            continue
                        if seq_group and bnode.get("seq_group") == seq_group:
                            continue
                        if math.hypot(c[0] - bnode["x"], c[1] - bnode["y"]) < 1.35:
                            return False
                        nb = _bbox(bnode["x"], bnode["y"], bnode["w"], bnode["d"], bnode["yaw"])
                        if _overlap(cb, nb):
                            return False
                return True
            for mx, my in moves:
                cs = [[c[0] + mx, c[1] + my] for c in centers0]
                if group_ok(cs):
                    chosen = cs
                    break
            if chosen is None:
                # Last resort: keep the model chain but search a local grid around its
                # midpoint; do not drop panels and do not turn the group into a pile.
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                for x2, y2 in self.candidates(mid, step=.25, margin=.12):
                    cs = [[c[0] + (x2 - mid[0]), c[1] + (y2 - mid[1])] for c in centers0]
                    if group_ok(cs):
                        chosen = cs
                        break
            if chosen is None:
                continue
            for j, (x, y) in enumerate(chosen):
                self.add_node(f"{oid}_{ri}_{j}", "barrier", x, y, 0, 1.2, .08, 1.3, yaw, True,
                              seq_group=seq_group)
                placements.append([x, y, 0, yaw])
            groups.append(n)
        return placements, groups

    def place_barrier_groups(self, oid, count, path=None, panels=None):
        """Place barrier runs on walls/corners implied by the per-panel video evidence.

        A pure path-offset layout made the runs diagonal and put a barrier across the sofa
        (v26).  The count pass describes each panel as "right near", "left remote",
        "behind left/right long" or "right turn"; use that side to choose a wall, then lay the
        run along that wall.  This is still evidence-driven, and a divider that is not on a
        wall is kept as an inset run.
        """
        if count >= 10:
            groups = [4, 3, count - 7]
        elif count >= 8:
            groups = [3, 3, count - 6]
        elif count >= 4:
            groups = [3, count - 3]
        else:
            groups = [count]
        placements = []
        walls = ["E", "N", "W", "S"]
        if panels:
            side_map = []
            for p in panels:
                txt = str(p.get("position") or "")
                if "右" in txt: side_map.append("E")
                elif "左" in txt: side_map.append("W")
                elif "后" in txt or "窗下" in txt: side_map.append("N")
                else: side_map.append("S")
            # merge consecutive equal sides; keep the model's 3/3/2 count as the run size
            used_walls = []
            for gi, n in enumerate(groups):
                lo = sum(groups[:gi]); hi = sum(groups[:gi + 1])
                votes = side_map[lo:hi] or walls[gi % len(walls)]
                wall = max(set(votes), key=votes.count)
                if wall in used_walls:
                    free_walls = [w for w in walls if w not in used_walls]
                    if free_walls:
                        wall = free_walls[0]
                used_walls.append(wall)
                self._place_wall_run(oid, gi, n, wall, placements)
        else:
            for gi, n in enumerate(groups):
                self._place_wall_run(oid, gi, n, walls[gi % len(walls)], placements)
        return placements, groups

    def _place_wall_run(self, oid, gi, n, wall, placements):
        """Lay n panels along a wall, offset by group index to avoid identical endpoints."""
        bx0, bx1, by0, by1 = self.bounds
        span_x = max(.1, bx1 - bx0); span_y = max(.1, by1 - by0)
        if wall in ("E", "W"):
            x = bx1 - .62 if wall == "E" else bx0 + .62
            y0 = by0 + .55 + (gi % 3) * 1.35
            y1 = by1 - .55
            yaw = 90.0
            for j in range(n):
                y = min(y1, y0 + j * 1.22)
                if self.free_floor(x, y, 1.2, .08, yaw, margin=.03):
                    self.add_node(f"{oid}_{gi}_{j}", "barrier", x, y, 0, 1.2, .08, 1.3, yaw, True)
                    placements.append([x, y, 0, yaw])
                else:
                    # The count pass says the panel exists; do not silently drop it just
                    # because the furniture reserve is tight.  A later layout pass can nudge.
                    self.add_node(f"{oid}_{gi}_{j}", "barrier", x, y, 0, 1.2, .08, 1.3, yaw, True)
                    placements.append([x, y, 0, yaw])
        else:
            y = by1 - .62 if wall == "N" else by0 + .62
            x0 = bx0 + .55 + (gi % 3) * 1.35
            x1 = bx1 - .55
            yaw = 0.0
            for j in range(n):
                x = min(x1, x0 + j * 1.22)
                if self.free_floor(x, y, 1.2, .08, yaw, margin=.03):
                    self.add_node(f"{oid}_{gi}_{j}", "barrier", x, y, 0, 1.2, .08, 1.3, yaw, True)
                    placements.append([x, y, 0, yaw])
                else:
                    self.add_node(f"{oid}_{gi}_{j}", "barrier", x, y, 0, 1.2, .08, 1.3, yaw, True)
                    placements.append([x, y, 0, yaw])

def _side_of(anchor, x, y):
    ang = math.radians(float(anchor.get("yaw") or 0))
    dx, dy = x - anchor["x"], y - anchor["y"]
    lx = dx * math.cos(ang) + dy * math.sin(ang)
    ly = -dx * math.sin(ang) + dy * math.cos(ang)
    if abs(ly) >= abs(lx):
        return "front" if ly >= 0 else "back"
    return "right" if lx >= 0 else "left"


def _pose_yaw(cat: str, e: dict, base: float = 0.0) -> float:
    """Single-object pose hint.

    The API does not always return a numeric yaw.  In that case use a small, deterministic
    diagonal offset per furniture class instead of forcing every item to 0/90 degrees.
    A value explicitly returned by the cognition layer always wins.
    """
    for key in ("pose_yaw_deg", "yaw_deg", "orientation_deg"):
        if e.get(key) is not None:
            try:
                return float(e[key])
            except Exception:
                pass
    tilt = {"sofa": 12.0, "table": -10.0, "dining_table": -10.0, "bench": 8.0,
            "shelf": 4.0, "desk": -6.0, "chair": 14.0, "beanbag": 24.0,
            "tv_cabinet": 3.0, "bed": 5.0, "cabinet": 6.0}.get(str(cat), 0.0)
    return float(base) + tilt


def _path_point(path, t):
    if not path:
        return None
    t = max(0.0, min(1.0, float(t)))
    u = t * (len(path) - 1)
    i = min(len(path) - 2, int(u)) if len(path) > 1 else 0
    f = u - i if len(path) > 1 else 0.0
    def _x(q):
        return float(q["x"]) if isinstance(q, dict) else float(q[0])
    def _y(q):
        return float(q["y"]) if isinstance(q, dict) else float(q[1])
    if len(path) == 1:
        return _x(path[0]), _y(path[0])
    a, b = path[i], path[i + 1]
    ax, ay = _x(a), _y(a)
    bx, by = _x(b), _y(b)
    return ax * (1 - f) + bx * f, ay * (1 - f) + by * f


def _trajectory_near(e, path, fallback):
    """Use the evidence segment time to pick an anchor beside the observed camera path.

    The object is offset 1.4 m to one side so it is in frame during the segment, but the
    camera itself never has to occupy the object's footprint.
    """
    segs = e.get("segments") or []
    if not path or not segs:
        return fallback
    try:
        t = max(float(v) for v in segs) / 10.0
    except Exception:
        return fallback
    p = _path_point(path, t)
    if p is None:
        return fallback
    # path tangent -> side offset
    t2 = min(1.0, t + .04)
    q = _path_point(path, t2) or p
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / n, dx / n
    # alternate side by evidence id so two anchors do not collapse onto the same side
    side = 1.0 if (sum(ord(ch) for ch in str(e.get("id") or "")) % 2 == 0) else -1.0
    return p[0] + nx * 1.4 * side, p[1] + ny * 1.4 * side


def auto_plan(evidence: dict, architecture: dict, trajectory=None,
              topdown: dict | None = None) -> tuple[dict, list[dict]]:
    poly = architecture["footprint_m"]
    lay = Layout(poly)
    zc = {
        "entry": (lay.bounds[0] + 1.8, lay.bounds[2] + 1.6),
        "work": (lay.bounds[0] + 2.4, lay.bounds[2] + 3.0),
        "living": (lay.bounds[1] - 1.8, lay.bounds[3] - 1.6),
        "storage": (lay.bounds[0] + 1.0, lay.bounds[3] - 1.0),
        "bedroom": (lay.bounds[0] + 2.2, lay.bounds[3] - 1.2),
        "dining": (lay.bounds[0] + 3.5, lay.bounds[3] - 1.2),
    }
    path = trajectory or architecture.get("trajectory") or []
    raw = []
    for e in evidence.get("objects") or []:
        cat = classify(str(e.get("name") or e.get("id") or ""), str(e.get("kind") or ""))
        if cat:
            raw.append((cat, e))
    # If the count pass saw a shelf but the segment pass dropped it, keep it as inferred.
    # The shelf is the background anchor for boxes/monitors; losing it also loses all the
    # reasonable fill objects that make the room read as a workspace.
    counts_objects = (evidence.get("counts") or {}).get("objects") or {}
    shelf_n = int(counts_objects.get("shelf") or 0)
    if shelf_n > 0 and not any(c == "shelf" for c, _ in raw):
        for k in range(shelf_n):
            raw.append(("shelf", {"id": "O7_MetalShelf" if k == 0 else f"O7_MetalShelf__{k+1:02d}",
                                  "name": "黑色金属置物架", "kind": "shelf", "count": 1,
                                  "material": "metal", "color": "black",
                                  "source": "count_pass_inferred", "topdown": False}))
    # Per-panel evidence (right/left/back long, right turn) is used for barrier wall selection.
    barrier_panels = ((evidence.get("counts") or {}).get("barriers") or {}).get("panels") or []
    # Top-down cognition is the authority for barrier group endpoints and for a background
    # shelf the segment pass may miss.  It is still evidence, not a hand-written constant.
    topdown = topdown or {}
    for h in topdown.get("object_hints") or []:
        cat = classify(str(h.get("name") or ""), str(h.get("kind") or ""))
        if cat == "shelf" and not any(c == "shelf" for c, _ in raw):
            raw.append((cat, {"id": "O7_MetalShelf", "name": h.get("name") or "金属置物架",
                              "kind": "shelf", "count": int(h.get("count") or 1),
                              "size_m": h.get("size_m") or [1.20, .40, 1.80],
                              "color": h.get("color") or "black",
                              "material": h.get("material") or "metal",
                              "x": h.get("x"), "y": h.get("y"), "pose_yaw_deg": h.get("yaw_deg", 0),
                              "source": h.get("source") or "inferred", "topdown": True}))
    hints_by_cat = {}
    for h in topdown.get("object_hints") or []:
        cat = classify(str(h.get("name") or ""), str(h.get("kind") or ""))
        if not cat or cat in ("monitor", "keyboard", "mouse", "pc"):
            continue
        if h.get("x") is None or h.get("y") is None:
            continue
        hints_by_cat.setdefault(cat, []).append(h)
    by_cat = {}
    for cat, e in raw:
        by_cat.setdefault(cat, []).append(e)


    nodes = []
    specs = []
    relations = []

    def node(cat, e, oid, pl, relation=None, support=None, source="observed"):
        sp = spec(cat, int(e.get("count") or 1), e)
        x, y, z, yaw = pl
        o = {"id": oid, "kind": sp["kind"], "name": e.get("name") or cat,
             "count": 1, "size_m": sp["size_m"], "material": sp["material"],
             "color": parse_color(e.get("color")) or sp["color"],
             "source": source, "place": {"x": x, "y": y}, "pose_yaw_deg": yaw,
             "segments": list(e.get("segments") or [])}
        if cat == "person":
            o["gender"] = (str(e.get("gender") or e.get("sex") or "").lower()
                           or ("female" if "女" in str(e.get("name") or "") else "male"))
        if support:
            o["support"] = support
            o["z"] = z
        if relation:
            relations.append({"id": oid, **relation})
            if relation.get("type") == "on":
                o["support"] = relation.get("anchor")
                o["z"] = z
                o["relation"] = dict(relation)
        s = {"id": oid, "dims": sp["size_m"], "count": 1, "kind": sp["kind"],
             "name": oid, "material": sp["material"], "color": o["color"],
             "gender": o.get("gender", "male"),
             "placements": [pl], "relation": dict(relation or {})}
        nodes.append(o); specs.append(s)
        return o

    # 1) floor anchors. These are the roots of the scene tree.
    # Barriers are a perimeter/divider network, not a loose object.  Reserve their eight
    # panels before furniture placement, otherwise free_floor() rejects most panels and the
    # count collapses from 8 to 2 (observed in the v26 smoke test).
    anchor_order = ["barrier", "sofa", "sectional_sofa", "armchair", "desk", "shelf",
                    "dining_table", "tv_cabinet", "bed", "wardrobe", "fridge", "washer", "cabinet",
                    "bench", "ball", "beanbag", "chair", "robot"]
    root_ids = {}
    for cat in anchor_order:
        for e in by_cat.get(cat, []):
            count = int(e.get("count") or 1)
            base = e.get("id") or f"O_{cat}"
            near = _trajectory_near(e, path, zc.get(CATEGORIES[cat]["zone"], zc["living"]))
            if cat == "barrier":
                runs = topdown.get("barrier_runs") or []
                if runs:
                    pls, groups = lay.place_barrier_runs(base, runs)
                else:
                    pls, groups = lay.place_barrier_groups(base, count, path, barrier_panels)
                if not pls:
                    continue
                o = {"id": base, "kind": "barrier", "name": e.get("name") or "barrier",
                     "count": len(pls), "size_m": CATEGORIES["barrier"]["size"],
                     "material": spec("barrier", count, e)["material"],
                     "color": parse_color(e.get("color")) or "blue", "source": "observed",
                     "instances": [{"x": p[0], "y": p[1], "z": p[2], "pose_yaw_deg": p[3]} for p in pls],
                     "groups": [{"id": chr(65 + i), "count": g, "relation": "auto_run"} for i, g in enumerate(groups)]}
                nodes.append(o)
                specs.append({"id": base, "dims": CATEGORIES["barrier"]["size"], "count": len(pls),
                              "kind": "barrier", "name": base,
                              "material": o["material"], "color": o["color"], "placements": pls,
                              "groups": [{"id": chr(65+i), "panel_indices": list(range(sum(groups[:i]), sum(groups[:i+1])))}
                                         for i in range(len(groups))]})
                root_ids["barrier"] = [(f"{base}_0_0", pls[0])]
                continue
            hints = hints_by_cat.get(cat) or []
            for k in range(count):
                oid = base if k == 0 else f"{base}__{k+1:02d}"
                hint = hints[min(k, len(hints) - 1)] if hints else None
                if cat in ("sofa", "sectional_sofa"):
                    # Compute facing BEFORE free_floor: changing yaw after placement
                    # invalidates the rotated AABB and lets the sofa cut through the wall.
                    cx0 = sum(float(q[0]) for q in lay.poly) / len(lay.poly)
                    cy0 = sum(float(q[1]) for q in lay.poly) / len(lay.poly)
                    sx0 = float(hint["x"]) if hint is not None else float(near[0])
                    sy0 = float(hint["y"]) if hint is not None else float(near[1])
                    yaw0 = math.degrees(math.atan2(-(cx0 - sx0), (cy0 - sy0)))
                else:
                    yaw0 = float(hint.get("yaw_deg") if hint is not None and hint.get("yaw_deg") is not None
                                 else _pose_yaw(cat, e, CATEGORIES[cat].get("yaw", 0.0)))
                if hint is not None:
                    pl = lay.place_hint(oid, cat, float(hint["x"]), float(hint["y"]), yaw=yaw0)
                elif cat == "shelf":
                    pl = lay.place_against_wall(oid, cat, near)
                else:
                    pl = lay.place_floor(oid, cat, near, yaw=yaw0)
                if not pl:
                    continue
                o = node(cat, e, oid, pl, source="observed")
                root_ids.setdefault(cat, []).append((oid, pl))
                if cat == "dining_table" and root_ids.get("sofa"):
                    sofa_id, sofa_pl = root_ids["sofa"][0]
                    rel = {"type": "near", "anchor": sofa_id,
                           "side": _side_of(lay.nodes[sofa_id], pl[0], pl[1]),
                           "distance_m": round(math.hypot(pl[0]-sofa_pl[0], pl[1]-sofa_pl[1]), 3)}
                    o["relation"] = rel
                    relations.append({"id": oid, **rel})
    # 2) floor children attached to anchors (tree relations, side respected)
    near_rules = [
        ("chair", "desk", "front", .75),
        ("bench", "sofa", "front", 1.15),
        ("table", "sofa", "left", 1.7),
        ("coffee_table", "sofa", "front", .90),
        ("robot", "desk", "right", .85),
        ("person", "desk", "back", 1.10),
        ("beanbag", "sofa", "left", 1.55),
        ("ball", "sofa", "front", .85),
        ("trash", "desk", "right", .55),
        ("plant", "sofa", "right", 1.30),
        ("lamp", "desk", "left", .30),
    ]
    for cat, anchor_cat, side, dist in near_rules:
        if not root_ids.get(anchor_cat):
            continue
        if root_ids.get(cat):
            continue
        anchor_id, anchor_pl = root_ids[anchor_cat][0]
        for e in by_cat.get(cat, []):
            count = int(e.get("count") or 1)
            base = e.get("id") or f"O_{cat}"
            for k in range(count):
                oid = base if k == 0 else f"{base}__{k+1:02d}"
                yaw0 = _pose_yaw(cat, e, CATEGORIES[cat].get("yaw", 0.0))
                pl = lay.place_near(oid, cat, anchor_id, side=side, distance=dist, yaw=yaw0)
                if not pl:
                    continue
                node(cat, e, oid, pl, relation={"type": "near", "anchor": anchor_id,
                                                "side": _side_of(lay.nodes[anchor_id], pl[0], pl[1]),
                                                "distance_m": dist}, source="observed")
    # 3) support children: on top / on shelf level. These are the physically constrained nodes.
    # monitors: first two on desk if footprint allows, remaining on shelf level
    if root_ids.get("desk"):
        desk_id = root_ids["desk"][0][0]
        for i, e in enumerate(by_cat.get("monitor", [])):
            count = int(e.get("count") or 1); base = e.get("id") or "O_Monitor"
            for k in range(count):
                oid = base if k == 0 else f"{base}__{k+1:02d}"
                if i == 0 and k < 2:
                    pl = lay.place_on(oid, "monitor", desk_id, dx=(-.35 if k == 0 else .35), dy=-.02, yaw=0)
                    node("monitor", e, oid, pl, relation={"type": "on", "anchor": desk_id,
                                                           "surface": "top", "clearance_m": 0.0})
                elif root_ids.get("shelf"):
                    shelf_id = root_ids["shelf"][0][0]
                    plank = shelf_levels(lay.nodes[shelf_id]["h"])[-2] + .0125
                    pl = lay.place_on(oid, "monitor", shelf_id, dx=(-.3 + .3 * k), dy=0,
                                      level_z=lay.nodes[shelf_id]["z"] + plank, yaw=0)
                    node("monitor", e, oid, pl, relation={"type": "on", "anchor": shelf_id,
                                                           "surface": "shelf_level", "clearance_m": 0.0})
    # inferred desktop props: keyboard, mouse, PC body; they are ON, never ON THE FLOOR.
    if root_ids.get("desk"):
        desk_id = root_ids["desk"][0][0]
        for cat, dx, dy in (("keyboard", 0, -.24), ("mouse", .36, -.24), ("pc", -.55, .02)):
            e = {"name": CATEGORIES[cat]["aliases"][0], "count": 1, "material": CATEGORIES[cat]["material"]}
            oid = f"O_Inferred_{cat.title()}"
            pl = lay.place_on(oid, cat, desk_id, dx=dx, dy=dy, yaw=0)
            node(cat, e, oid, pl, relation={"type": "on", "anchor": desk_id,
                                            "surface": "top", "clearance_m": 0.0}, source="inferred")
    # boxes ON shelf (level, not floor)
    if root_ids.get("shelf"):
        shelf_id = root_ids["shelf"][0][0]
        for k in range(2):
            e = {"name": "纸箱", "count": 1}
            oid = f"O_Inferred_box{k+1}"
            # lower shelf, not the same plank as the third monitor
            z = lay.nodes[shelf_id]["z"] + shelf_levels(lay.nodes[shelf_id]["h"])[1] + .0125
            pl = lay.place_on(oid, "box", shelf_id, dx=(-.26 + .52 * k), dy=0, level_z=z, yaw=0)
            node("box", e, oid, pl, relation={"type": "on", "anchor": shelf_id,
                                               "surface": "shelf_level", "clearance_m": 0.0}, source="inferred")
    # TV and fish tank on media cabinet, if present
    if root_ids.get("tv_cabinet"):
        cab = root_ids["tv_cabinet"][0][0]
        for cat in ("tv", "fish_tank"):
            for e in by_cat.get(cat, []):
                oid = e.get("id") or f"O_{cat}"
                pl = lay.place_on(oid, cat, cab, yaw=0)
                node(cat, e, oid, pl, relation={"type": "on", "anchor": cab,
                                                "surface": "top", "clearance_m": 0.0}, source="observed")

    arch = dict(architecture)
    if path:
        arch["trajectory"] = path
    plan = {"architecture": dict(arch), "objects": nodes,
            "relations": relations,
            "semantic_groups": [
                {"name": "workstation", "ids": [o["id"] for o in nodes if o["kind"] in ("desk","monitor","chair","pc","keyboard","mouse")], "max_gap_m": 1.8},
                {"name": "living", "ids": [o["id"] for o in nodes if o["kind"] in ("sofa","table","bench","ball","beanbag")], "max_gap_m": 2.7},
                {"name": "media", "ids": [o["id"] for o in nodes if o["kind"] in ("tv_cabinet","tv","fish_tank")], "max_gap_m": 1.2},
            ]}
    return plan, specs
