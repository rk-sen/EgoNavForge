"""v9 semantic asset builders. One planned id -> one Empty root + ONE joined mesh.

Templates are deliberately richer than v8's ten boxes: office chair (5-star + castors),
drawer cabinet, fridge, wall AC, desk lamp, keyboard, mouse, monitor, PC tower, cardboard box,
dining table, bench, blinds, pillow/bean-bag (vertex-deformed, not a scaled cube).

Coordinates: metres, Z up, floor z=0. `z0` is the bottom of the item, so an item can sit ON a
desk/shelf with `relation: {"type":"on","target":...}`.
"""
from __future__ import annotations

import json
import math

import bpy
from mathutils import Matrix, Vector

import bs_geo
import bs_mat

_MAT_CACHE = {}


NAMED_COLORS = {
    "blue": (0.035, 0.105, 0.52), "navy": (0.02, 0.04, 0.20),
    "orange": (0.72, 0.22, 0.035), "brown": (0.28, 0.14, 0.055),
    "green": (0.035, 0.48, 0.16), "gray": (0.38, 0.39, 0.40),
    "grey": (0.38, 0.39, 0.40), "black": (0.035, 0.035, 0.035),
    "white": (0.82, 0.82, 0.80), "beige": (0.62, 0.55, 0.43),
    "red": (0.55, 0.05, 0.04), "yellow": (0.75, 0.58, 0.04),
}


def _rgba(color, fallback=(0.55, 0.55, 0.55)):
    if isinstance(color, str):
        c = color.strip()
        if c.startswith("#") and len(c) == 7:
            return tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        if c.lower() in NAMED_COLORS:
            return NAMED_COLORS[c.lower()]
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return tuple(float(x) for x in color[:3])
    return fallback


def _family(kind, name, material=""):
    k = (kind + " " + name + " " + material).lower()
    if any(w in k for w in ("screen", "monitor", "display", "tv")):
        return "screen"
    if any(w in k for w in ("glass", "window", "玻璃")):
        return "glass"
    if any(w in k for w in ("metal", "金属", "steel", "铁", "铝", "chrome")):
        return "metal"
    if any(w in k for w in ("rubber", "橡胶", "floor", "地板")):
        return "rubber"
    if any(w in k for w in ("leather", "皮革", "皮")):
        return "leather"
    if any(w in k for w in ("cardboard", "纸箱", "纸盒", "纸")):
        return "cardboard"
    if any(w in k for w in ("paint", "painted", "lacquer", "烤漆")):
        return "paint"
    if any(w in k for w in ("wood", "木", "desk", "table", "shelf", "柜", "架")):
        return "wood"
    if any(w in k for w in ("fabric", "布", "sofa", "沙发", "cushion", "垫", "ball", "球",
                             "barrier", "屏障", "chair", "椅", "bean")):
        return "fabric"
    if any(w in k for w in ("plastic", "塑料", "abs")):
        return "plastic"
    return "paint"


def _color(kind, name, color=None):
    if color:
        return _rgba(color)
    k = (kind + " " + name).lower()
    if "green" in k or "绿" in k:
        return (0.08, 0.46, 0.20)
    if "orange" in k or "橙" in k:
        return (0.62, 0.25, 0.07)
    if "brown" in k or "棕" in k:
        return (0.30, 0.16, 0.08)
    if "blue" in k or "蓝" in k:
        return (0.06, 0.18, 0.58)
    if "black" in k or "黑" in k:
        return (0.06, 0.06, 0.06)
    if "gray" in k or "grey" in k or "灰" in k:
        return (0.34, 0.34, 0.34)
    if "white" in k or "白" in k:
        return (0.86, 0.86, 0.83)
    if "red" in k or "红" in k:
        return (0.55, 0.10, 0.08)
    return (0.55, 0.52, 0.47)


def material(kind, name, material_name="", color=None):
    fam = _family(kind, name, material_name)
    rgba = _color(kind, name, color)
    # A white office desk top is painted laminate, not a metal/glass slab.  The raw
    # recognition phrase often contains "金属框架/玻璃桌面"; taking that literally made
    # the whole desk metallic-dark in V22.  Color is the stronger evidence for the surface.
    # Desk/table recognition often says "metal frame + laminate top"; using the frame
    # word for the entire object turns the top into a dark metal slab.  The surface
    # colour is stronger evidence: light/grey -> painted laminate; dark -> wood.
    if str(kind).lower() in ("desk", "table"):
        fam = "paint" if sum(rgba[:3]) / 3.0 > 0.30 else "wood"
    key = (fam, tuple(round(v, 3) for v in rgba))
    if key not in _MAT_CACHE:
        _MAT_CACHE[key] = bs_mat.preset("M_%s_%02x%02x%02x" % (
            fam, int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)), fam, rgba)
    return _MAT_CACHE[key]


def _root(name, loc, yaw=0.0, z0=0.0):
    r = bs_geo.asset_root(name, (float(loc[0]), float(loc[1]), float(z0)))
    r.rotation_euler = (0, 0, math.radians(float(yaw or 0.0)))
    return r


def _finish(root, parts):
    bs_geo.parent_all(root, parts)
    bs_geo.join_children(root)
    return root


def build_sofa(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """Lounge sofa with rounded plinth, rolled arms, bulging cushions and a shaped back.

    V12 used four bevel boxes, which read as a box with cushions. The actual video sofa has
    a soft rounded body and rolled arms. Keep the same footprint/dims, but use a pillow body,
    Y-axis arm rolls, back bolsters and shorter feet so the silhouette is curved.
    """
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = []
    # soft plinth: vertex-deformed pillow, not a box
    parts.append(bs_geo.pillow(name + "_plinth", (w * .94, d * .86, h * .34),
                               loc=(0, 0, h * .24), mat=mat, bulge=.24, wrinkles=.006))
    parts.append(bs_geo.box(name + "_back_frame", (w * .94, d * .16, h * .72),
                            loc=(0, -d * .38, h * .56), mat=mat, bevel=.08))
    # rolled arms are horizontal cylinders along the sofa depth
    for sx in (-1, 1):
        parts.append(bs_geo.cylinder(name + "_arm_roll%d" % sx, h * .16, d * .82,
                                     loc=(sx * w * .43, 0, h * .50), axis="Y", mat=mat, seg=18))
        parts.append(bs_geo.box(name + "_arm_base%d" % sx, (w * .10, d * .84, h * .32),
                                loc=(sx * w * .43, 0, h * .24), mat=mat, bevel=.06))
    # two seat + two back cushions; pillow meshes give rounded rims and slight wrinkles
    for i, sx in enumerate((-1, 1)):
        parts.append(bs_geo.pillow(name + "_seat%d" % i, (w * .42, d * .66, h * .18),
                                   loc=(sx * w * .21, -d * .02, h * .43), mat=mat,
                                   wrinkles=.010, bulge=.50))
        parts.append(bs_geo.pillow(name + "_backc%d" % i, (w * .38, d * .16, h * .40),
                                   loc=(sx * w * .22, -d * .34, h * .62), mat=mat,
                                   wrinkles=.014, bulge=.58))
    # rounded bolster at the top of the back, and low feet
    parts.append(bs_geo.cylinder(name + "_bolster", h * .10, w * .88,
                                 loc=(0, -d * .43, h * .88), axis="X", mat=mat, seg=16))
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(bs_geo.cylinder(name + "_foot%d%d" % (sx, sy), .032, .08,
                                         loc=(sx * w * .36, sy * d * .34, .04), mat=mat, seg=10))
    _finish(r, parts)
    return r

def build_barrier(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """Blue crash barrier seen in the video: silver tube frame with blue cloth stretched over it,
    and red/black weighted feet at each end.

    The earlier v10 template used blue posts and blue feet, which read as a blue box, not as
    fabric on a metal frame.
    """
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    frame = material("metal", name + "_frame", "metal", (0.64, 0.66, 0.68))
    red = material("rubber", name + "_red", "rubber", "red")
    black = material("rubber", name + "_black", "rubber", "black")
    # Cloth is a hanging panel that reaches the top rail and the two end posts. The earlier
    # version used 82% width and 70% height, leaving a visible air gap around the cloth, so the
    # blue sheet read as a separate object floating inside the metal frame.
    # A closed thin box is more robust than a high-poly fabric grid after joining; the
    # fabric material still makes it read as cloth, and the frame remains visible.
    panel = bs_geo.box(name + "_cloth", (w * .96, .035, h * .88),
                       loc=(0, 0, h * .08 + h * .44), mat=mat, bevel=.008)
    parts = [panel,
             bs_geo.cylinder(name + "_toprail", .026, w, loc=(0, 0, h - .045),
                             axis="X", mat=frame, seg=12),
             bs_geo.cylinder(name + "_midrail", .022, w, loc=(0, 0, h * .11),
                             axis="X", mat=frame, seg=12)]
    for sx in (-1, 1):
        # Every one of the two feet is a stack: black rubber base + red top plate, NOT one red
        # foot and one black foot. The video's stands are visibly black underneath and red on top.
        parts += [
            bs_geo.cylinder(name + "_post%d" % sx, .027, h, loc=(sx * w * .50, 0, h / 2),
                            mat=frame, seg=10),
            bs_geo.box(name + "_foot_black%d" % sx, (.36, .62, .035),
                       loc=(sx * (w * .50 - .17), 0, .0175), mat=black, bevel=.012),
            bs_geo.box(name + "_foot_red%d" % sx, (.30, .56, .028),
                       loc=(sx * (w * .50 - .17), 0, .049), mat=red, bevel=.012),
            bs_geo.box(name + "_foot_up%d" % sx, (.11, .11, .14),
                       loc=(sx * (w * .50 - .17), 0, .080), mat=red, bevel=.010),
            # metal clamps grip the cloth edge at the top rail and at the mid rail
            bs_geo.box(name + "_clamp_top%d" % sx, (.055, .075, .065),
                       loc=(sx * (w * .47), 0, h * .955), mat=frame, bevel=.006),
            bs_geo.box(name + "_clamp_mid%d" % sx, (.045, .060, .050),
                       loc=(sx * (w * .47), 0, h * .12), mat=frame, bevel=.006),
            # bottom hem bar, so the fabric does not end in a single disconnected polygon
            bs_geo.cylinder(name + "_hem%d" % sx, .012, w * .48,
                            loc=(sx * (w * .24), 0, h * .085), axis="X", mat=frame, seg=8),
        ]
    _finish(r, parts)
    return r



def build_desk(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    top_t = .045
    parts = [bs_geo.box(name + "_top", (w, d, top_t), loc=(0, 0, h - top_t / 2), mat=mat, bevel=.012)]
    # slender steel legs + lower stretcher, so it reads as an office desk rather than a cube
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(bs_geo.box(name + "_leg%d" % i, (.045, .045, h - top_t),
                                loc=(sx * (w / 2 - .055), sy * (d / 2 - .055), (h - top_t) / 2),
                                mat=mat, bevel=.006))
    parts += [bs_geo.box(name + "_modesty", (w * .72, .022, h * .26),
                         loc=(0, -d / 2 + .055, h * .56), mat=mat),
              bs_geo.box(name + "_brace", (w * .78, .025, .03),
                         loc=(0, -d / 2 + .06, h * .22), mat=mat)]
    _finish(r, parts)
    return r


def build_table(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """High bar / work table: thick top, four timber legs and two X braces."""
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_top", (w, d, .075), loc=(0, 0, h - .0375), mat=mat, bevel=.02)]
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(bs_geo.box(name + "_leg%d" % i, (.065, .065, h - .075),
                                loc=(sx * (w / 2 - .10), sy * (d / 2 - .10), (h - .075) / 2),
                                mat=mat, bevel=.008))
    for sy in (-1, 1):
        parts.append(bs_geo.box(name + "_brace%d" % sy, (w - .22, .035, .045),
                                loc=(0, sy * (d / 2 - .10), h * .24), mat=mat))
    _finish(r, parts)
    return r


def build_bench(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.pillow(name + "_seat", (w, d, .10), loc=(0, 0, h - .05), mat=mat,
                           bulge=.30, wrinkles=.008)]
    for i, sx in enumerate((-1, 1)):
        parts.append(bs_geo.box(name + "_leg%d" % i, (.055, d * .72, h - .10),
                                loc=(sx * (w / 2 - .13), 0, (h - .10) / 2), mat=mat))
    # The source bench has a long stretcher between the two end frames, not four isolated legs.
    parts.append(bs_geo.box(name + "_stretcher", (w * .82, .045, .060),
                            loc=(0, 0, h * .24), mat=mat, bevel=.008))
    parts.append(bs_geo.box(name + "_stretcher_lower", (w * .82, .035, .045),
                            loc=(0, 0, h * .12), mat=mat, bevel=.006))
    _finish(r, parts)
    return r


def build_task_chair(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """Office chair: seat, back, gas column, five-star base, castors, armrests."""
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    seat_h = h * .46
    parts = [bs_geo.pillow(name + "_seat", (w, d, .10), loc=(0, 0, seat_h), mat=mat,
                           bulge=.32, wrinkles=.01),
             bs_geo.pillow(name + "_back", (w * .90, .10, h * .46),
                           loc=(0, -d * .46, h * .74), mat=mat, bulge=.25, wrinkles=.012),
             # The gas column must reach the seat underside. v11 ended at 0.62*seat_h and left
             # a visible gap below the cushion; use a full-height column plus a support plate.
             bs_geo.cylinder(name + "_gas", .034, seat_h, loc=(0, 0, seat_h / 2),
                             mat=mat, seg=14),
             bs_geo.box(name + "_seat_plate", (w * .62, d * .62, .025),
                        loc=(0, 0, seat_h - .055), mat=mat, bevel=.006)]
    parts.append(bs_geo.cylinder(name + "_hub", .055, .055, loc=(0, 0, .045), mat=mat, seg=14))
    for i, ang in enumerate((0, 72, 144, 216, 288)):
        dx, dy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        # one continuous spoke from the central hub to the castor, not a floating leg
        parts.append(bs_geo.box(name + "_leg%d" % i, (.44, .042, .038),
                                loc=(dx * w * .235, dy * d * .235, .040), mat=mat, rot_z=float(ang)))
        parts.append(bs_geo.cylinder(name + "_hubspoke%d" % i, .022, .05,
                                     loc=(dx * .055, dy * .055, .040), axis="Z", mat=mat, seg=8))
        parts.append(bs_geo.cylinder(name + "_castor%d" % i, .030, .040,
                                     loc=(dx * w * .47, dy * d * .47, .022), mat=mat, seg=10))
    for sx in (-1, 1):
        parts.append(bs_geo.box(name + "_arm%d" % sx, (.05, d * .50, .035),
                                loc=(sx * (w / 2 - .055), 0, seat_h + .19), mat=mat))
        parts.append(bs_geo.box(name + "_arm_post%d" % sx, (.045, .045, .19),
                                loc=(sx * (w / 2 - .055), -d * .18, seat_h + .095), mat=mat,
                                bevel=.006))
    _finish(r, parts)
    return r


def build_stool(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.pillow(name + "_seat", (w, d, .09), loc=(0, 0, h - .05), mat=mat, bulge=.25)]
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
        parts.append(bs_geo.cylinder(name + "_leg%d" % i, .025, h - .12,
                                    loc=(sx * w * .30, sy * d * .30, (h - .12) / 2), mat=mat, seg=10))
    _finish(r, parts)
    return r


def build_monitor(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """Bezel + emissive screen image + stand + base; screen no longer a black slab."""
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    panel_h = h * .78
    parts = [bs_geo.box(name + "_bezel", (w, max(.035, d * .62), panel_h),
                        loc=(0, 0, h - panel_h / 2), mat=mat, bevel=.012),
             bs_geo.box(name + "_neck", (.055, .055, h * .20),
                        loc=(0, -d * .10, h * .12), mat=mat),
             bs_geo.box(name + "_base", (w * .46, d * .90, .018),
                        loc=(0, 0, .009), mat=mat, bevel=.008)]
    # display on the front (-Y) face; the generated image makes it read as a real monitor
    screen = bs_mat.screen_ui()
    parts.append(bs_geo.plane(name + "_screen", (w * .92, panel_h * .84),
                              loc=(0, -max(.035, d * .62) / 2 - .009, h - panel_h / 2),
                              rot_xyz=(90, 0, 0), mat=screen))
    _finish(r, parts)
    return r


def build_laptop(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    screen = bs_mat.screen_ui()
    parts = [bs_geo.box(name + "_base", (w, d * .62, .022), loc=(0, 0, .011), mat=mat, bevel=.006),
             bs_geo.box(name + "_lid", (w, .018, d * .58), loc=(0, -d * .48, .022 + d * .29),
                        mat=mat, rot_xyz=(-18, 0, 0)),
             bs_geo.plane(name + "_screen", (w * .88, d * .48),
                          loc=(0, -d * .48 - .02, .022 + d * .28),
                          rot_xyz=(72, 0, 0), mat=screen)]
    _finish(r, parts)
    return r


def build_keyboard(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_board", (w, d, .022), loc=(0, 0, .011), mat=mat, bevel=.008)]
    rows, cols = 4, 12
    for iy in range(rows):
        for ix in range(cols):
            parts.append(bs_geo.box("%s_key%d_%d" % (name, ix, iy),
                                    (w / cols * .82, d / rows * .68, .012),
                                    loc=(-w / 2 + (ix + .5) * w / cols, -d / 2 + (iy + .5) * d / rows,
                                         .027), mat=mat))
    _finish(r, parts)
    return r


def build_mouse(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.soft_sphere(name + "_body", (w, d, h * .75), loc=(0, 0, h * .38),
                                mat=mat, squash=.35, seg=16, rings=10)]
    _finish(r, parts)
    return r


def build_pc(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_tower", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.012),
             bs_geo.box(name + "_front", (w * .82, .012, h * .90), loc=(0, -d / 2 - .006, h / 2), mat=mat)]
    _finish(r, parts)
    return r


def build_fridge(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.02)]
    for i, zz in enumerate((h * .72, h * .40)):
        parts.append(bs_geo.box(name + "_door%d" % i, (w * .97, .025, h * .28 if i == 0 else h * .34),
                                loc=(0, -d / 2 - .013, zz), mat=mat, bevel=.015))
        parts.append(bs_geo.box(name + "_handle%d" % i, (.025, .035, h * .18),
                                loc=(-w * .40, -d / 2 - .035, zz), mat=mat))
    _finish(r, parts)
    return r


def build_cabinet(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.015)]
    for i in range(4):
        zz = .10 + i * (h - .20) / 4
        parts.append(bs_geo.box(name + "_drawer%d" % i, (w * .90, .018, (h - .20) / 4 * .82),
                                loc=(0, -d / 2 - .010, zz), mat=mat, bevel=.008))
        parts.append(bs_geo.box(name + "_handle%d" % i, (w * .22, .022, .018),
                                loc=(0, -d / 2 - .022, zz + .02), mat=mat))
    _finish(r, parts)
    return r


def build_shelf(name, dims, loc, mat, yaw=0.0, z0=0.0):
    """Open steel shelving rack: four corner posts, four shelves and diagonal braces."""
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    post = .035
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(bs_geo.box(name + "_post%d%d" % (sx, sy), (post, post, h),
                                    loc=(sx * (w / 2 - post / 2), sy * (d / 2 - post / 2), h / 2),
                                    mat=mat))
    for i in range(4):
        zz = .07 + i * (h - .14) / 3
        parts.append(bs_geo.box(name + "_shelf%d" % i, (w, d * .94, .025),
                                loc=(0, 0, zz), mat=mat, bevel=.006))
    for sx in (-1, 1):
        parts.append(bs_geo.box(name + "_bar%d" % sx, (.022, d * .94, .022),
                                loc=(sx * (w / 2 - post - .01), 0, h * .55), mat=mat))
    _finish(r, parts)
    return r


def build_box(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_box", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.01)]
    for i, sx in enumerate((-1, 1)):
        parts.append(bs_geo.box(name + "_flap%d" % i, (w * .48, d * .96, .012),
                                loc=(sx * w * .24, 0, h + .006), mat=mat, rot_xyz=(0, 0, math.radians(sx * 12))))
    _finish(r, parts)
    return r


def build_ac(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.03),
             bs_geo.box(name + "_louver", (w * .86, .018, h * .22), loc=(0, -d / 2 - .009, h * .55),
                        mat=mat, rot_xyz=(18, 0, 0))]
    _finish(r, parts)
    return r


def build_lamp(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.cylinder(name + "_base", w * .34, .025, loc=(0, 0, .0125), mat=mat, seg=18),
             bs_geo.cylinder(name + "_stem", .014, h * .70, loc=(0, 0, h * .35), mat=mat, seg=10),
             bs_geo.cylinder(name + "_shade", w * .30, h * .20, loc=(0, 0, h * .80),
                             mat=mat, seg=20, r_top=w * .18)]
    _finish(r, parts)
    return r


def build_blind(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_rail", (w, .05, .04), loc=(0, 0, h - .02), mat=mat)]
    n = 9
    for i in range(n):
        parts.append(bs_geo.box(name + "_slat%d" % i, (w, d, .018),
                                loc=(0, 0, (i + .5) * h / n), mat=mat,
                                rot_xyz=(20, 0, 0)))
    _finish(r, parts)
    return r


def build_ball(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.sphere(name + "_ball", h / 2, (0, 0, h / 2), mat=mat,
                           scale=(w / max(h, 1e-6), d / max(h, 1e-6), 1))]
    _finish(r, parts)
    return r


def build_cushion(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.pillow(name + "_pillow", (w, d, h), loc=(0, 0, h / 2), mat=mat,
                           wrinkles=.016, bulge=.60)]
    _finish(r, parts)
    return r


def build_beanbag(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    # V15 used a scaled sphere -> ellipse. The source is a slumped, asymmetric fabric bag.
    parts = [bs_geo.beanbag(name + "_bag", (w, d, h), loc=(0, 0, 0), mat=mat)]
    _finish(r, parts)
    return r


def build_robot(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_chassis", (w * .88, d * .82, h * .34),
                        loc=(0, 0, h * .22), mat=mat, bevel=.03),
             bs_geo.box(name + "_deck", (w * .62, d * .55, h * .18),
                        loc=(0, d * .02, h * .42), mat=mat, bevel=.02),
             bs_geo.cylinder(name + "_lidar", w * .14, h * .16,
                             loc=(0, 0, h * .58), mat=mat, seg=20),
             bs_geo.box(name + "_head", (w * .32, d * .18, h * .12),
                        loc=(0, -d * .26, h * .38), mat=mat, bevel=.015)]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(bs_geo.cylinder(name + "_wheel%d%d" % (sx, sy), h * .12, .05,
                                         loc=(sx * w * .40, sy * d * .38, h * .12),
                                         axis="X", mat=mat, seg=16))
    _finish(r, parts)
    return r


def build_light(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_tube", (w, d, max(h, .06)), loc=(0, 0, max(h, .06) / 2), mat=mat)]
    _finish(r, parts)
    return r


def build_door(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_leaf", (w, .045, h), loc=(0, 0, h / 2), mat=mat, bevel=.012),
             bs_geo.cylinder(name + "_handle", .018, .08, loc=(w * .38, -.045, h * .48),
                             mat=mat, axis="Y", seg=10)]
    _finish(r, parts)
    return r


def build_window(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_frame", (w, .04, h), loc=(0, 0, h / 2), mat=mat),
             bs_geo.box(name + "_glass", (w * .90, .012, h * .86), loc=(0, 0, h / 2), mat=mat)]
    _finish(r, parts)
    return r


def build_default(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.03)]
    _finish(r, parts)
    return r


def build_bed(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_base", (w * .98, d * .98, h * .35), loc=(0, 0, h * .18), mat=mat, bevel=.03),
             bs_geo.pillow(name + "_mattress", (w * .92, d * .90, h * .30), loc=(0, d * .02, h * .48), mat=mat, wrinkles=.008, bulge=.25),
             bs_geo.box(name + "_head", (w * .98, .10, h * .85), loc=(0, -d * .48, h * .42), mat=mat, bevel=.05)]
    for i, sx in enumerate((-.28, .28)):
        parts.append(bs_geo.pillow(name + "_pillow%d" % i, (w * .38, d * .22, h * .16),
                                   loc=(sx * w, -d * .30, h * .62), mat=mat, wrinkles=.012, bulge=.40))
    _finish(r, parts)
    return r


def build_tv(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_screen", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.025),
             bs_geo.box(name + "_bezel", (w * 1.03, .015, h * 1.03), loc=(0, -d / 2 - .008, h / 2), mat=mat),
             bs_geo.box(name + "_stand", (w * .18, d * .55, .06), loc=(0, 0, .03), mat=mat, bevel=.02)]
    _finish(r, parts)
    return r


def build_wardrobe(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.02)]
    for i, sx in enumerate((-1, 1)):
        parts.append(bs_geo.box(name + "_door%d" % i, (w * .47, .018, h * .92),
                                loc=(sx * w * .25, -d / 2 - .010, h * .49), mat=mat, bevel=.008))
        parts.append(bs_geo.cylinder(name + "_handle%d" % i, .012, .10, loc=(sx * w * .44, -d / 2 - .022, h * .50),
                                     axis="Z", mat=mat, seg=10))
    _finish(r, parts)
    return r


def build_fish_tank(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    glass = mat
    parts = []
    for sx in (-1, 1):
        parts.append(bs_geo.box(name + "_glassX%d" % sx, (.015, d, h),
                                loc=(sx * w / 2, 0, h / 2), mat=glass))
    for sy in (-1, 1):
        parts.append(bs_geo.box(name + "_glassY%d" % sy, (w, .015, h),
                                loc=(0, sy * d / 2, h / 2), mat=glass))
    parts.append(bs_geo.box(name + "_base", (w * 1.02, d * 1.02, .04), loc=(0, 0, .02), mat=glass))
    parts.append(bs_geo.box(name + "_water", (w * .94, d * .90, h * .55), loc=(0, 0, h * .56), mat=glass))
    parts.append(bs_geo.box(name + "_gravel", (w * .94, d * .90, .04), loc=(0, 0, .045), mat=mat))
    _finish(r, parts)
    return r


def build_armchair(name, dims, loc, mat, yaw=0.0, z0=0.0):
    # a compact upholstered chair, same soft language as the sofa
    return build_sofa(name, dims, loc, mat, yaw=yaw, z0=z0)


def build_rug(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, _ = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.plane(name + "_rug", (w, d), loc=(0, 0, .012), mat=mat)]
    _finish(r, parts)
    return r


def build_plant(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.cylinder(name + "_pot", w * .32, h * .32, loc=(0, 0, h * .16), mat=mat, seg=18),
             bs_geo.cylinder(name + "_stem", .025, h * .50, loc=(0, 0, h * .52), mat=mat, seg=10)]
    for i, (sx, sy, zz) in enumerate([(0, 0, .88), (.22, 0, .70), (-.20, .05, .62), (0, -.18, .75)]):
        parts.append(bs_geo.sphere(name + "_leaf%d" % i, h * .19, (sx, sy, zz * h),
                                   seg=14, rings=8, mat=mat, scale=(.8, .55, .45)))
    _finish(r, parts)
    return r


def build_washer(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.box(name + "_body", (w, d, h), loc=(0, 0, h / 2), mat=mat, bevel=.03),
             bs_geo.cylinder(name + "_door", w * .28, .035, loc=(0, -d / 2 - .018, h * .46), axis="Y", mat=mat, seg=24),
             bs_geo.box(name + "_panel", (w * .86, .018, h * .12), loc=(0, -d / 2 - .015, h * .86), mat=mat)]
    _finish(r, parts)
    return r


def _public_person(name, loc, yaw, z0, gender):
    """Import the public CC0 Kenney Mini Characters GLB stored in assets/.

    The file and its texture are checked into the project.  This is not a procedural
    capsule fallback: the recognizer chooses male/female and the actual downloaded GLB
    is imported, normalized to 1.72 m, and placed at the detected floor position.
    """
    from pathlib import Path
    g = "female" if str(gender).lower().startswith("f") else "male"
    root = Path(__file__).resolve().parents[1] / "assets"
    supplied = root / "person_supplied.glb"
    blend = root / "person_supplied.blend"
    path = supplied if supplied.exists() else root / ("person_%s.glb" % g)
    if blend.exists():
        # The supplied CAD body is already one joined mesh in the checked-in .blend.
        # Appending it is much faster than glTF-importing dozens of pieces and joining
        # 200k polygons inside every scene build.
        before = set(bpy.data.objects)
        bpy.ops.wm.append(filepath=str(blend), directory=str(blend) + "/Object",
                          filename="PersonSupplied")
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH"]
        if not meshes:
            return None
        obj = meshes[0]
        obj.name = name
        path = blend
    else:
        if not path.exists():
            return None
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=str(path))
        new = [o for o in bpy.data.objects if o not in before]
        meshes = [o for o in new if o.type == "MESH" and o.name != "Icosphere"]
        if not meshes:
            return None
        for o in list(new):
            if o.type == "MESH" and o not in meshes and o.name in bpy.data.objects:
                bpy.data.objects.remove(o, do_unlink=True)
        bpy.ops.object.select_all(action='DESELECT')
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        obj.name = name
    # Detach from the imported armature and remove the armature object.
    if obj.parent:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
    new = [o for o in bpy.data.objects if o not in before]
    for o in [x for x in new if x.type == "ARMATURE"]:
        bpy.data.objects.remove(o, do_unlink=True)
    # Normalize to a 1.72 m standing human; the mini-character source is ~0.79 m.
    bpy.context.view_layer.update()
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    zmin = min(v.z for v in bb); zmax = max(v.z for v in bb)
    height = max(1e-6, zmax - zmin)
    scale = 1.72 / height
    obj.scale = (obj.scale.x * scale, obj.scale.y * scale, obj.scale.z * scale)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True); bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    # Place by world bbox, never by a mesh-data translation.  The imported glTF origin
    # is at the character's centre; using the object location shift is the reliable
    # way to put the feet on the floor.
    bpy.context.view_layer.update()
    bb = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    cx = (min(v.x for v in bb) + max(v.x for v in bb)) * .5
    cy = (min(v.y for v in bb) + max(v.y for v in bb)) * .5
    zmin = min(v.z for v in bb)
    obj.location.x += loc[0] - cx
    obj.location.y += loc[1] - cy
    obj.location.z += z0 - zmin
    bpy.context.view_layer.update()
    obj.rotation_euler = (0, 0, math.radians(yaw))
    obj["source"] = "supplied_cad:x_t converted by CadConvert" if path.name == "person_supplied.glb" else "public_glb:Kenney_Mini_Characters_CC0"
    obj["gender"] = g
    return obj


def build_person(name, dims, loc, mat, yaw=0.0, z0=0.0, gender="male"):
    """Place a public male/female GLB prefab in the scene.

    The public GLB is the primary path.  The procedural body below remains only as a
    fallback for environments where the checked-in GLB/texture is missing.
    """
    pub = _public_person(name, loc, yaw, z0, gender)
    if pub is not None:
        return pub
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    g = "female" if str(gender).lower().startswith("f") else "male"
    skin = bs_mat.principled("M_Person_Skin", (0.72, 0.52, 0.40, 1), rough=.62)
    shirt = bs_mat.principled("M_Person_Shirt_" + g,
                              (0.22, 0.30, 0.42, 1) if g == "male" else (0.55, 0.30, 0.34, 1), rough=.82)
    trousers = bs_mat.principled("M_Person_Trousers_" + g,
                                 (0.12, 0.14, 0.18, 1) if g == "male" else (0.22, 0.20, 0.28, 1), rough=.86)
    hair = bs_mat.principled("M_Person_Hair", (0.06, 0.05, 0.045, 1), rough=.55)
    shoe = bs_mat.principled("M_Person_Shoe", (0.04, 0.04, 0.045, 1), rough=.35)
    sw = w * (0.92 if g == "male" else 0.78)   # shoulder width
    hip = w * (0.72 if g == "male" else 0.82)
    leg_h = h * 0.46
    parts = []

    # shoes / legs (centres, not floor-anchored cylinders)
    for sx in (-1, 1):
        parts.append(bs_geo.box(name + "_shoe%d" % sx, (w * .22, d * .42, h * .035),
                                loc=(sx * hip * .24, d * .03, h * .0175), mat=shoe, bevel=.012))
        parts.append(bs_geo.cylinder(name + "_leg%d" % sx, w * .11, leg_h,
                                     loc=(sx * hip * .24, 0, h * .035 + leg_h / 2),
                                     mat=trousers, seg=12))
    # pelvis and torso
    parts.append(bs_geo.box(name + "_pelvis", (hip, d * .82, h * .10),
                            loc=(0, 0, h * .50), mat=trousers, bevel=.025))
    parts.append(bs_geo.box(name + "_torso", (sw, d * .72, h * .30),
                            loc=(0, 0, h * .705), mat=shirt, bevel=.035))
    if g == "female":
        # skirt as a tapered lower body; this reads differently from the male trouser prefab
        parts.append(bs_geo.cylinder(name + "_skirt", hip * .55, h * .18,
                                     loc=(0, 0, h * .43), mat=trousers, seg=18,
                                     r_top=hip * .38))
    # neck + head + hair
    parts.append(bs_geo.cylinder(name + "_neck", w * .10, h * .055,
                                 loc=(0, 0, h * .875), mat=skin, seg=12))
    parts.append(bs_geo.sphere(name + "_head", h * .105, (0, 0, h * .925),
                              seg=20, rings=12, mat=skin, scale=(.88, .82, 1.0)))
    parts.append(bs_geo.sphere(name + "_hair", h * .112, (0, -d * .03, h * .945),
                              seg=20, rings=12, mat=hair, scale=(.92, .82, .72)))
    # arms: cylinders angled outward, hands at hip height
    for sx in (-1, 1):
        parts.append(bs_geo.cylinder(name + "_arm%d" % sx, w * .075, h * .30,
                                     loc=(sx * sw * .43, 0, h * .69),
                                     mat=shirt, seg=10))
        parts.append(bs_geo.sphere(name + "_hand%d" % sx, w * .065,
                                    loc=(sx * sw * .49, 0, h * .51), seg=12, rings=8, mat=skin))
    if g == "male":
        parts.append(bs_geo.box(name + "_collar", (sw * .55, d * .50, h * .018),
                                loc=(0, 0, h * .845), mat=bs_mat.principled("M_Person_Collar", (.86, .86, .82, 1), rough=.8),
                                bevel=.006))
    _finish(r, parts)
    return r


def build_curtain(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.fabric_panel(name + "_cloth", (w, h), loc=(0, 0, 0), mat=mat,
                                nx=16, nz=12, sag=.10, billow=.03, thickness=.018)]
    _finish(r, parts)
    return r


def build_trash(name, dims, loc, mat, yaw=0.0, z0=0.0):
    w, d, h = dims
    r = _root(name, loc, yaw, z0)
    parts = [bs_geo.cylinder(name + "_bin", w * .45, h, loc=(0, 0, h / 2), mat=mat, seg=18),
             bs_geo.cylinder(name + "_rim", w * .50, .025, loc=(0, 0, h - .012), mat=mat, seg=18)]
    _finish(r, parts)
    return r


BUILDERS = {
    "sofa": build_sofa, "armchair": build_armchair, "barrier": build_barrier, "desk": build_desk,
    "table": build_table, "dining_table": build_table, "bench": build_bench,
    "chair": build_task_chair, "task_chair": build_task_chair, "office_chair": build_task_chair,
    "ergonomic_chair": build_task_chair, "pressure_stool": build_task_chair,
    "stool": build_stool, "monitor": build_monitor, "laptop": build_laptop,
    "keyboard": build_keyboard, "mouse": build_mouse, "pc": build_pc, "computer": build_pc,
    "fridge": build_fridge, "cabinet": build_cabinet, "drawer_cabinet": build_cabinet,
    "bed": build_bed, "tv": build_tv, "tv_cabinet": build_cabinet, "wardrobe": build_wardrobe,
    "fish_tank": build_fish_tank, "rug": build_rug, "plant": build_plant,
    "washer": build_washer, "person": build_person, "curtain": build_curtain,
    "lamp": build_lamp, "ac": build_ac, "curtain": build_curtain, "trash": build_trash,
    "shelf": build_shelf, "box": build_box, "cardboard": build_box,
    "ac": build_ac, "air_conditioner": build_ac, "lamp": build_lamp, "desk_lamp": build_lamp,
    "blind": build_blind, "blinds": build_blind, "ball": build_ball,
    "cushion": build_cushion, "pillow": build_cushion, "beanbag": build_beanbag,
    "bean_bag": build_beanbag, "robot": build_robot, "light": build_light,
    "door": build_door, "safety_door": build_door, "window": build_window, "default": build_default,
}


def build_instance(spec):
    """spec: id, kind, dims, placements [[x,y,z,yaw_deg],...], material, color, relation.

    V10 keeps per-instance yaw: a row of barriers is never all perfectly axis-aligned, and a
    chair/table must face the way the video shows it. `z0` is the bottom of the item.
    """
    oid = spec["id"]
    dims = [float(v) for v in spec["dims"][:3]]
    kind = spec.get("kind") or "default"
    mat = material(kind, spec.get("name") or oid, spec.get("material") or "", spec.get("color"))
    fn = BUILDERS.get(kind, build_default)
    first = spec["placements"][0]
    yaw = float(first[3] if len(first) > 3 else 0)
    if kind == "person":
        root = fn(oid, dims, first, mat, yaw=yaw, z0=float(first[2] if len(first) > 2 else 0),
                  gender=spec.get("gender") or "male")
    else:
        root = fn(oid, dims, first, mat, yaw=yaw, z0=float(first[2] if len(first) > 2 else 0))
    root["v10_kind"] = kind
    root["v10_yaw_deg"] = yaw
    root["v10_relation"] = json.dumps(spec.get("relation") or {}, ensure_ascii=False)
    for i in range(1, len(spec["placements"])):
        q = spec["placements"][i]
        clone = bs_geo.clone_asset(oid, "%s__%02d" % (oid, i + 1),
                                  offset=(q[0] - first[0], q[1] - first[1], q[2] - first[2]))
        # clone_asset copies rotation; override so each instance can have its own pose
        clone.rotation_euler = (0, 0, math.radians(float(q[3] if len(q) > 3 else 0)))
        clone["v10_yaw_deg"] = float(q[3] if len(q) > 3 else 0)
    return root


def build_all(specs):
    out = []
    for s in specs:
        out.append(build_instance(s).name)
    return out
