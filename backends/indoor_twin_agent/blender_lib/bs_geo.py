"""Geometry toolkit used INSIDE Blender (imported by scripts the agent writes).

Everything here is plain `mesh.from_pydata` / bmesh — no bpy.ops modelling, so it is reproducible
headless and never depends on UI context. All builders take `coll` (collection name) and `mat`.

Coordinate convention: Blender Z-up, metres, floor at z=0, +X east, +Y north.
"""
from __future__ import annotations

import math

import bpy
import bs_mat
from mathutils.geometry import tessellate_polygon
from mathutils import Vector, Matrix


# --------------------------------------------------------------------------- scene
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def coll(name: str, parent=None):
    c = bpy.data.collections.get(name)
    if not c:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c


def mat_get(name: str):
    return bpy.data.materials.get(name)


def _finish(obj, coll_name, mat, smooth):
    if coll_name:
        coll(coll_name).objects.link(obj)
    if mat:
        obj.data.materials.append(mat)
    if smooth and obj.type == "MESH":
        for p in obj.data.polygons:
            p.use_smooth = True
    return obj


def new_mesh(name, verts, faces, coll_name="Assets", mat=None, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.validate()
    me.update()
    ob = bpy.data.objects.new(name, me)
    return _finish(ob, coll_name, mat, smooth)


# --------------------------------------------------------------------------- primitives
CUBE_V = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
CUBE_F = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def box(name, size, loc=(0, 0, 0), rot_z=0.0, coll_name="Assets", mat=None, bevel=0.0, rot_xyz=None):
    sx, sy, sz = (size, size, size) if not hasattr(size, "__len__") else size
    verts = [(vx * sx / 2, vy * sy / 2, vz * sz / 2) for vx, vy, vz in CUBE_V]
    ob = new_mesh(name, verts, CUBE_F, coll_name, mat)
    ob.location = Vector(loc)
    if rot_xyz:
        ob.rotation_euler = tuple(math.radians(a) for a in rot_xyz)
    elif rot_z:
        ob.rotation_euler = (0, 0, math.radians(rot_z))
    if bevel > 0:
        m = ob.modifiers.new("bev", "BEVEL")
        m.width = bevel
        m.segments = 2
    return ob


def plane(name, size, loc, rot_xyz=(0, 0, 0), coll_name="Assets", mat=None):
    sx, sy = size
    verts = [(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0)]
    ob = new_mesh(name, verts, [(0, 1, 2, 3)], coll_name, mat)
    ob.location = Vector(loc)
    ob.rotation_euler = tuple(math.radians(a) for a in rot_xyz)
    return ob


def cylinder(name, radius, height, loc, axis="Z", seg=24, coll_name="Assets", mat=None, smooth=True, r_top=None):
    r_top = radius if r_top is None else r_top
    # Center the cylinder on its origin.  v10's version ran from z=0 to z=height, so a post
    # placed at h/2 started at h/2 and ended above the roof, and an X-axis rail started at x=0
    # and stretched only to +w (the barrier measured 3.92 m instead of 2.5 m and its posts
    # floated).  All builders place cylinders at their centre.
    ring = [(math.cos(2 * math.pi * i / seg) * radius, math.sin(2 * math.pi * i / seg) * radius, -height / 2) for i in range(seg)]
    top = [(math.cos(2 * math.pi * i / seg) * r_top, math.sin(2 * math.pi * i / seg) * r_top, height / 2)
           for i in range(seg)]
    verts = ring + top
    faces = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
    faces.append(tuple(range(seg - 1, -1, -1)))
    faces.append(tuple(range(seg, 2 * seg)))
    ob = new_mesh(name, verts, faces, coll_name, mat, smooth)
    if axis == "X":
        ob.rotation_euler = (0, math.radians(90), 0)
    elif axis == "Y":
        ob.rotation_euler = (math.radians(90), 0, 0)
    ob.location = Vector(loc)
    return ob


def fabric_panel(name, size, loc=(0, 0, 0), coll_name="Assets", mat=None, nx=10, nz=8,
                sag=0.035, billow=0.05, thickness=0.025):
    """A closed, subdivided fabric sheet in the XZ plane (y is thickness).

    Used for the blue barrier walls: a flat box reads as a painted board, while a slightly
    sagging grid with a real edge loop reads as cloth and catches workbench shading.
    """
    w, h = float(size[0]), float(size[1])
    verts, faces = [], []
    for side in (-1, 1):
        for iz in range(nz + 1):
            tz = iz / nz
            z = tz * h
            # slight horizontal sag plus a gentle vertical billow; still exactly within [-w/2,w/2]
            for ix in range(nx + 1):
                tx = ix / nx
                x = (tx - .5) * w
                y = side * thickness / 2
                zz = z + (0.02 * h) * math.sin(math.pi * tx) * (1 - tz)
                yy = y + (side) * billow * math.sin(math.pi * tz) * math.sin(math.pi * tx)
                verts.append((loc[0] + x, loc[1] + yy, loc[2] + zz))
    # index helper
    def idx(side, iz, ix):
        return (1 if side > 0 else 0) * ((nz + 1) * (nx + 1)) + iz * (nx + 1) + ix
    for side in (-1, 1):
        for iz in range(nz):
            for ix in range(nx):
                a = idx(side, iz, ix); b = idx(side, iz, ix + 1)
                c = idx(side, iz + 1, ix + 1); d = idx(side, iz + 1, ix)
                faces.append((a, b, c, d) if side > 0 else (a, d, c, b))
    # side walls around the perimeter
    for ix in range(nx):
        faces.append((idx(-1, 0, ix), idx(1, 0, ix), idx(1, 0, ix + 1), idx(-1, 0, ix + 1)))
        faces.append((idx(-1, nz, ix), idx(1, nz, ix), idx(1, nz, ix + 1), idx(-1, nz, ix + 1)))
    for iz in range(nz):
        faces.append((idx(-1, iz, 0), idx(1, iz, 0), idx(1, iz + 1, 0), idx(-1, iz + 1, 0)))
        faces.append((idx(-1, iz, nx), idx(1, iz, nx), idx(1, iz + 1, nx), idx(-1, iz + 1, nx)))
    return new_mesh(name, verts, faces, coll_name=coll_name, mat=mat, smooth=True)


def pillow(name, size, loc=(0, 0, 0), coll_name="Assets", mat=None, wrinkles=0.012, bulge=0.62, nx=8, ny=8):
    """Closed pillow mesh: two subdivided grids, rounded rim, slight wrinkle.

    This is vertex-level modelling, not a scaled cube: the rim falls away and the top/bottom
    bulge, so a deformed cushion can actually be represented without sculpting by hand.
    """
    w, d, h = size
    verts, faces = [], []
    for side, zs in ((1, h / 2), (-1, -h / 2)):
        base = len(verts)
        for iy in range(ny + 1):
            v = iy / ny
            for ix in range(nx + 1):
                u = ix / nx
                x = (u - 0.5) * w
                y = (v - 0.5) * d
                edge = (abs(u - 0.5) * 2) ** 3 + (abs(v - 0.5) * 2) ** 3
                z = zs * (1.0 - bulge * min(1.0, edge))
                if side > 0:
                    z += (wrinkles * math.sin(u * math.pi * 3.0) * math.sin(v * math.pi * 2.0)
                          + wrinkles * 0.5 * math.sin(u * math.pi * 5.0))
                verts.append((loc[0] + x, loc[1] + y, loc[2] + z))
        for iy in range(ny):
            for ix in range(nx):
                a = base + iy * (nx + 1) + ix
                faces.append((a, a + 1, a + nx + 2, a + nx + 1) if side > 0
                             else (a, a + nx + 1, a + nx + 2, a + 1))
    top0, bot0 = 0, (nx + 1) * (ny + 1)
    ring = ([(ix, 0) for ix in range(nx)] + [(nx, iy) for iy in range(ny)]
            + [(ix, ny) for ix in range(nx, 0, -1)] + [(0, iy) for iy in range(ny, 0, -1)])
    for k in range(len(ring)):
        ax, ay = ring[k]
        bx, by = ring[(k + 1) % len(ring)]
        faces.append((top0 + ay * (nx + 1) + ax, top0 + by * (nx + 1) + bx,
                      bot0 + by * (nx + 1) + bx, bot0 + ay * (nx + 1) + ax))
    return new_mesh(name, verts, faces, coll_name=coll_name, mat=mat, smooth=True)


def beanbag(name, size, loc=(0, 0, 0), coll_name="Assets", mat=None, seg=28, rings=16):
    """Slumped fabric bean-bag: lofted rings, flattened bottom, off-centre top slump.

    A scaled sphere reads as an ellipse in the twin. The source bean-bag has a low, wide,
    asymmetric body with a wrinkled top, so this mesh varies radius per ring and drifts the
    upper rings sideways instead of using one analytic ellipsoid.
    """
    w, d, h = size
    verts, faces = [], []
    for iz in range(rings + 1):
        t = iz / rings
        z = h * t
        # wide at 25-45% height, tapers into a soft slumped top; bottom is nearly flat
        r = 0.5 * (1.0 - 0.48 * t ** 1.35) + 0.035 * math.sin(math.pi * min(1.0, t * 1.7))
        if t < .16:
            r *= .82 + .18 * (t / .16)
        cx = w * .055 * t ** 2
        cy = -d * .045 * t ** 2
        for ix in range(seg):
            th = 2 * math.pi * ix / seg
            # irregular fabric folds, strongest around the middle and top
            lump = .022 * math.sin(3 * th + 1.2) * (0.25 + 0.75 * t)
            lump += .012 * math.sin(5 * th - .4) * t
            x = cx + (r * w * .5 + lump) * math.cos(th)
            y = cy + (r * d * .5 + lump) * math.sin(th)
            # make the bottom settle, and put a shallow dent into the top
            zz = z
            if t > .82:
                rr = math.hypot((x - cx) / (w * .5), (y - cy) / (d * .5))
                zz -= h * .055 * (1.0 - min(1.0, rr)) ** 2
            verts.append((loc[0] + x, loc[1] + y, loc[2] + zz))
    for iz in range(rings):
        for ix in range(seg):
            a = iz * seg + ix
            b = iz * seg + (ix + 1) % seg
            c = (iz + 1) * seg + (ix + 1) % seg
            d = (iz + 1) * seg + ix
            faces.append((a, b, c, d))
    faces.append(tuple(range(seg - 1, -1, -1)))
    faces.append(tuple(range(rings * seg, (rings + 1) * seg)))
    return new_mesh(name, verts, faces, coll_name=coll_name, mat=mat, smooth=True)


def soft_sphere(name, size, loc=(0, 0, 0), coll_name="Assets", mat=None, squash=0.28, seg=24, rings=16):
    """Bean-bag / cushion-ish blob: ellipsoid with a flattened bottom and a small dent on top."""
    w, d, h = size
    obj = sphere(name, 0.5, (0, 0, 0), seg=seg, rings=rings, coll_name=coll_name, mat=mat,
                 scale=(w, d, h))
    for v in obj.data.vertices:
        z = v.co.z
        if z < 0:
            v.co.z = z * (1.0 - squash)
        else:
            v.co.z -= 0.12 * h * (1.0 - min(1.0, (v.co.x / max(1e-6, w / 2)) ** 2
                                                  + (v.co.y / max(1e-6, d / 2)) ** 2))
    obj.location = loc
    return obj


def sphere(name, radius, loc, seg=24, rings=16, coll_name="Assets", mat=None, scale=(1, 1, 1)):
    verts, faces = [], []
    for r in range(rings + 1):
        phi = math.pi * r / rings
        for s in range(seg):
            th = 2 * math.pi * s / seg
            verts.append((radius * math.sin(phi) * math.cos(th),
                          radius * math.sin(phi) * math.sin(th),
                          radius * math.cos(phi)))
    for r in range(rings):
        for s in range(seg):
            a = r * seg + s
            b = r * seg + (s + 1) % seg
            c = (r + 1) * seg + (s + 1) % seg
            d = (r + 1) * seg + s
            faces.append((a, b, c, d))
    ob = new_mesh(name, verts, faces, coll_name, mat, True)
    ob.location = Vector(loc)
    ob.scale = Vector(scale)
    return ob


def revolve(name, profile, segments=32, axis="Z", coll_name="Assets", mat=None, smooth=True):
    """profile = [(r, z), ...] revolved around the given axis; builds lathe-style solids."""
    verts, faces = [], []
    for i in range(segments):
        th = 2 * math.pi * i / segments
        for r, z in profile:
            verts.append((r * math.cos(th), r * math.sin(th), z))
    n = len(profile)
    for i in range(segments):
        base, nxt = i * n, ((i + 1) % segments) * n
        for j in range(n - 1):
            faces.append((base + j, base + j + 1, nxt + j + 1, nxt + j))
    if profile[0][0] > 1e-6 and profile[-1][0] > 1e-6:
        faces.append(tuple(range(n - 1, -1, -1)))
        faces.append(tuple(range((segments - 1) * n, segments * n)))
    return new_mesh(name, verts, faces, coll_name, mat, smooth)


def extrude_profile(name, profile, depth, loc=(0, 0, 0), rot_z=0, coll_name="Assets", mat=None, smooth=False):
    """profile in the XZ plane, extruded along Y by `depth` — sofa arms, table legs, mouldings."""
    n = len(profile)
    verts = [(x, -depth / 2, z) for x, z in profile] + [(x, depth / 2, z) for x, z in profile]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    ob = new_mesh(name, verts, faces, coll_name, mat, smooth)
    ob.location = Vector(loc)
    if rot_z:
        ob.rotation_euler = (0, 0, math.radians(rot_z))
    return ob


def tube(name, points, radius, seg=10, coll_name="Assets", mat=None, smooth=True):
    """Sweep a circle along a polyline (pipes, robot arms, cable trays)."""
    verts, faces = [], []
    for k, p in enumerate(points):
        p = Vector(p)
        if k < len(points) - 1:
            d = (Vector(points[k + 1]) - p)
        else:
            d = (p - Vector(points[k - 1]))
        d.normalize()
        up = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
        u = d.cross(up).normalized()
        v = d.cross(u).normalized()
        for i in range(seg):
            th = 2 * math.pi * i / seg
            verts.append(tuple(p + u * (radius * math.cos(th)) + v * (radius * math.sin(th))))
    for k in range(len(points) - 1):
        for i in range(seg):
            a = k * seg + i
            b = k * seg + (i + 1) % seg
            c = (k + 1) * seg + (i + 1) % seg
            d = (k + 1) * seg + i
            faces.append((a, b, c, d))
    faces.append(tuple(range(seg - 1, -1, -1)))
    faces.append(tuple(range((len(points) - 1) * seg, len(points) * seg)))
    return new_mesh(name, verts, faces, coll_name, mat, smooth)


# --------------------------------------------------------------------------- architecture
def _panels_with_holes(u0, u1, z0, z1, holes):
    """Split a rectangle into panels around rectangular holes. Returns [(u0,u1,z0,z1)]."""
    if not holes:
        return [(u0, u1, z0, z1)]
    cols = sorted(holes, key=lambda h: h[0])
    out, u = [], u0
    for hu0, hu1, hz0, hz1 in cols:
        if hu0 > u + 1e-6:
            out.append((u, hu0, z0, z1))
        # above / below the hole
        if hz1 < z1 - 1e-6:
            out.append((max(u, hu0), min(hu1, u1), hz1, z1))
        if hz0 > z0 + 1e-6:
            out.append((max(u, hu0), min(hu1, u1), z0, hz0))
        u = hu1
    if u < u1 - 1e-6:
        out.append((u, u1, z0, z1))
    return [p for p in out if p[1] - p[0] > 1e-4 and p[3] - p[2] > 1e-4]


def wall(name, p1, p2, height, thickness=0.25, base_z=0.0, holes=(), coll_name="Architecture",
         mat=None, extra=0.0):
    """A real solid wall between two floor points, with optional rectangular openings.

    holes: [(u_start, u_end, z_start, z_end)] measured in metres along the wall from p1.
    Each panel is built as a closed box, so the shell always has interior volume —
    a single quad is what makes preview renders look like one grey plane.
    """
    p1, p2 = Vector((p1[0], p1[1], 0)), Vector((p2[0], p2[1], 0))
    length = (p2 - p1).length
    if length < 1e-4:
        raise ValueError("wall endpoints coincide")
    ang = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x))
    objs = []
    for i, (u0, u1, z0, z1) in enumerate(_panels_with_holes(-extra, length + extra, base_z, base_z + height, holes)):
        cx, cy = (u0 + u1) / 2, 0
        w, d = (u1 - u0), (z1 - z0)
        ob = box(f"{name}_p{i}", (w, thickness, d), loc=(cx, cy, (z0 + z1) / 2),
                 coll_name=coll_name, mat=mat)
        objs.append(ob)
    empty = bpy.data.objects.new(name, None)
    coll(coll_name).objects.link(empty)
    empty.location = p1
    empty.rotation_euler = (0, 0, math.radians(ang))
    bpy.context.view_layer.update()
    # Panels are built in WALL-LOCAL coordinates (u along +X from p1), so the parent must apply
    # its own transform: parent_inverse = identity. Using matrix_world.inverted() here cancels
    # the wall transform and leaves every wall lying at the origin along +X — the v4.4/v5 shells
    # all had their walls stacked at (0,0) while QA only counted panel numbers.
    from mathutils import Matrix
    for ob in objs:
        ob.parent = empty
        ob.matrix_parent_inverse = Matrix.Identity(4)
    empty["wall_length"] = length
    empty["wall_height"] = height
    empty["wall_thickness"] = thickness
    return empty


def floor_slab(name, x0, x1, y0, y1, thickness=0.2, top_z=0.0, coll_name="Architecture", mat=None):
    return box(name, (x1 - x0, y1 - y0, thickness), loc=((x0 + x1) / 2, (y0 + y1) / 2, top_z - thickness / 2),
               coll_name=coll_name, mat=mat)


def slab(name, x0, x1, y0, y1, z0, z1, coll_name="Architecture", mat=None):
    return box(name, (x1 - x0, y1 - y0, z1 - z0), loc=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
               coll_name=coll_name, mat=mat)


def poly_slab(name, points, z0, z1, coll_name="Architecture", mat=None, smooth=False):
    """Extrude an arbitrary XY polygon (CCW or CW) between z0 and z1.

    Used by V10 for L-shaped rooms. `tessellate_polygon` handles simple concave polygons
    without requiring a convex bbox, so the notch of an L is not filled with floor.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        raise ValueError("poly_slab needs at least 3 points")
    # normalize CCW; tessellate_polygon expects a closed loop
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts)))
    if area < 0:
        pts = list(reversed(pts))
    n = len(pts)
    verts = [(x, y, z0) for x, y in pts] + [(x, y, z1) for x, y in pts]
    tris = tessellate_polygon([[Vector((p[0], p[1], 0)) for p in pts]])
    faces = []
    for tri in tris:
        i, j, k = [int(v) for v in tri]
        faces.append((k, j, i))                 # bottom, reversed
        faces.append((n + i, n + j, n + k))     # top
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return new_mesh(name, verts, faces, coll_name=coll_name, mat=mat, smooth=smooth)


def window(name, center, width, height, normal="Y", coll_name="Architecture", frame_mat=None, glass_mat=None,
           depth=0.1, proxy=True, proxy_sign=1.0):
    """Frame + glass inserted in an existing hole (the hole itself is cut by wall()).

    `width` always runs ALONG the wall: X for normal="Y" walls, Y for normal="X" walls.
    v5 had this inverted, so every window on an X-normal wall was 1.2 m thick across the wall
    and stuck out of both sides (the floating white bars in the shell previews).
    """
    cx, cy, cz = center
    ex, ey = (width, depth) if normal == "Y" else (depth, width)
    for tag, loc, dims in (
        ("frame_b", (cx, cy, cz - height / 2), (ex, ey, 0.06)),
        ("frame_t", (cx, cy, cz + height / 2), (ex, ey, 0.06)),
    ):
        box(f"{name}_{tag}", dims, loc=loc, coll_name=coll_name, mat=frame_mat)
    if normal == "Y":
        box(f"{name}_frame_l", (0.06, ey, height), loc=(cx - width / 2, cy, cz), coll_name=coll_name, mat=frame_mat)
        box(f"{name}_frame_r", (0.06, ey, height), loc=(cx + width / 2, cy, cz), coll_name=coll_name, mat=frame_mat)
        g = box(f"{name}_glass", (width - 0.05, 0.02, height - 0.05), loc=(cx, cy, cz),
                coll_name=coll_name, mat=glass_mat)
    else:
        box(f"{name}_frame_l", (ex, 0.06, height), loc=(cx, cy - width / 2, cz), coll_name=coll_name, mat=frame_mat)
        box(f"{name}_frame_r", (ex, 0.06, height), loc=(cx, cy + width / 2, cz), coll_name=coll_name, mat=frame_mat)
        g = box(f"{name}_glass", (0.02, width - 0.05, height - 0.05), loc=(cx, cy, cz),
                coll_name=coll_name, mat=glass_mat)
    if proxy:
        # A vertical exterior proxy behind the opening.  v18 built a horizontal slab
        # (z thickness 0.05), so through the window it read as a grey lid rather than
        # a view outside.  Build a vertical sky plane with a simple distant skyline.
        sky = bpy.data.materials.get("M_Proxy_Sky")
        if sky is None:
            sky = bs_mat.principled("M_Proxy_Sky", (0.62, 0.76, 0.90, 1), rough=.95)
        # exterior backdrops sit outside the lit room; a little emission keeps the sky
        # readable through the glass instead of turning into a black hole.
        try:
            bsdf = sky.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                if "Emission Color" in bsdf.inputs:
                    bsdf.inputs["Emission Color"].default_value = (0.50, 0.66, 0.86, 1.0)
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = 0.35
        except Exception:
            pass
        city = bpy.data.materials.get("M_Proxy_City")
        if city is None:
            city = bs_mat.principled("M_Proxy_City", (0.22, 0.26, 0.30, 1), rough=.88)
        sign = 1.0 if float(proxy_sign) >= 0 else -1.0
        sky_off = 2.0 * sign
        city_off = 1.30 * sign
        if normal == "Y":
            px, py = cx, cy + sky_off
            box(f"{name}_proxy_outside", (width * 3.4, 0.04, 5.2),
                loc=(px, py, cz + 0.4), coll_name=coll_name, mat=sky)
            # Put the skyline in front of the sky plane (toward the room), otherwise
            # the opaque sky plane hides it completely.
            for k in range(5):
                bx = cx + (k - 2) * width * .52
                bh = .45 + .16 * (k % 3)
                box(f"{name}_proxy_city{k}", (width * .40, .04, bh),
                    loc=(bx, cy + city_off, cz - height * .45 + bh * .5),
                    coll_name=coll_name, mat=city)
        else:
            px, py = cx + sky_off, cy
            box(f"{name}_proxy_outside", (0.04, width * 3.4, 5.2),
                loc=(px, py, cz + 0.4), coll_name=coll_name, mat=sky)
            for k in range(5):
                by = cy + (k - 2) * width * .52
                bh = .45 + .16 * (k % 3)
                box(f"{name}_proxy_city{k}", (.04, width * .40, bh),
                    loc=(cx + city_off, by, cz - height * .45 + bh * .5),
                    coll_name=coll_name, mat=city)
        # The opening remains the real glass; the proxy is only a distant backdrop.
    return g


def camera(name, loc, target, lens_mm, coll_name="Cameras"):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens_mm
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new(name, cam_data)
    coll(coll_name).objects.link(cam)
    cam.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return cam


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def join_children(root, name=None):
    """Join every direct mesh child of `root` into ONE mesh, keep it parented to the root.

    v7 built each asset as body+top+legs as separate objects. That looks fine in a viewport but
    every instance is a pile of parts: cloning, moving and checking a single item all go wrong.
    One asset = one Empty root + one joined mesh.
    """
    kids = [o for o in root.children if o.type == "MESH"]
    if not kids:
        return None
    bpy.ops.object.select_all(action="DESELECT")
    for o in kids:
        o.hide_set(False)
        o.select_set(True)
    bpy.context.view_layer.objects.active = kids[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name or (root.name + "_mesh")
    joined.parent = root
    joined.matrix_parent_inverse = Matrix.Identity(4)
    return joined


def parent_all(parent_obj, children):
    for c in children:
        c.parent = parent_obj


def stats():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    tris = sum(len(o.data.loop_triangles) if o.data.loop_triangles else sum(len(p.vertices) - 2 for p in o.data.polygons)
               for o in meshes)
    return {"objects": len(bpy.data.objects), "meshes": len(meshes), "materials": len(bpy.data.materials),
            "images": len(bpy.data.images), "tris": tris,
            "camera": [o.name for o in bpy.data.objects if o.type == "CAMERA"]}


def asset_root(name, loc=(0, 0, 0), coll_name="Assets"):
    """One Empty per asset *instance*. Everything that belongs to the sofa (seat/back/arms)
    is parented to `O5_Sofa`; a second sofa is `O5_Sofa__02`. QA's plan_coverage counts these
    roots, so the number of instances the model planned is actually checkable."""
    e = bpy.data.objects.new(name, None)
    coll(coll_name).objects.link(e)
    e.location = Vector(loc)
    return e


def clone_asset(src, new_name, offset=(0, 0, 0), coll_name=None):
    """Deep-copy an Empty and all its children, then move the copy as one rigid body.

    Use this for repeated items (blue barriers, chairs, monitors) instead of re-running the
    builder — v4.4 ended with 36 exactly-overlapping `.001` pairs from doing that.
    """
    from mathutils import Matrix
    if isinstance(src, str):
        src = bpy.data.objects[src]
    root = src.copy()
    root.name = new_name
    target_coll = coll(coll_name) if coll_name else (src.users_collection[0] if src.users_collection else coll("Assets"))
    target_coll.objects.link(root)
    root.parent = None
    root.matrix_parent_inverse = Matrix.Identity(4)
    root.location = src.location + Vector(offset)          # children follow through matrix_world below
    bpy.context.view_layer.update()
    src_inv = src.matrix_world.inverted()
    kids = [o for o in bpy.data.objects if o.parent == src and o is not root]
    for child in kids:
        c = child.copy()
        c.name = f"{new_name}_{child.name.split('_', 1)[-1]}"
        target_coll.objects.link(c)
        c.parent = root
        c.matrix_parent_inverse = Matrix.Identity(4)
        c.matrix_world = root.matrix_world @ (src_inv @ child.matrix_world)
    bpy.context.view_layer.update()
    return root
