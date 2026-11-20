"""Texture generation + PBR material construction (runs inside Blender).

Two families, because a scene needs both:
  * `node_*`  : shader-node graphs (brick, noise rough, brushed metal, fabric weave)
  * `img_*`   : numpy-computed pixels -> real PNG on disk -> image texture wired into Principled

Image textures use `TexCoord.Object` + Mapping instead of UVs: box-projected tiling works on any
mesh produced by bs_geo, so the agent can never "generate a texture and forget to unwrap".
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
import numpy as np


# --------------------------------------------------------------------------- numpy textures
def _grid(w, h, scale):
    x = np.linspace(0, scale, w, endpoint=False)
    y = np.linspace(0, scale, h, endpoint=False)
    return np.meshgrid(x, y)


def _hash(ix, iy, seed=0):
    return np.modf(np.sin(ix * 127.1 + iy * 311.7 + seed * 74.7) * 43758.5453)[0]


def value_noise(u, v, seed=0, period=8):
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = u - iu, v - iv
    fu = fu * fu * (3 - 2 * fu)
    fv = fv * fv * (3 - 2 * fv)
    iu, iv = np.mod(iu, period), np.mod(iv, period)
    a = _hash(iu, iv, seed)
    b = _hash(iu + 1, iv, seed)
    c = _hash(iu, iv + 1, seed)
    d = _hash(iu + 1, iv + 1, seed)
    return (a * (1 - fu) + b * fu) * (1 - fv) + (c * (1 - fu) + d * fu) * fv


def fbm(u, v, seed=0, octaves=5, base=4):
    total, amp, norm = np.zeros_like(u), 1.0, 0.0
    for o in range(octaves):
        total += value_noise(u * base * 2 ** o, v * base * 2 ** o, seed + o, period=base * 2 ** o) * amp
        norm += amp
        amp *= 0.5
    return total / norm


def tex_concrete(w=1024, h=1024, seed=0, tint=(0.62, 0.62, 0.60)):
    u, v = _grid(w, h, 4.0)
    n = fbm(u, v, seed, octaves=6, base=3)
    speck = (fbm(u, v, seed + 9, octaves=2, base=64) > 0.72).astype(np.float32) * 0.18
    rgb = np.stack([np.clip(tint[i] * (0.82 + 0.30 * n) + speck * 0.5, 0, 1) for i in range(3)], axis=-1)
    rough = np.clip(0.80 + 0.15 * (n - 0.5), 0.3, 1.0)
    return rgb, rough, n


def tex_floor_tile(w=1024, h=1024, seed=0, tile=4, grout=0.035, c1=(0.72, 0.70, 0.66), c2=(0.60, 0.58, 0.55)):
    u, v = _grid(w, h, 1.0)
    n = fbm(u, v, seed, octaves=5, base=6)
    cell = np.minimum(np.mod(u * tile, 1), np.mod(v * tile, 1))
    edge = np.minimum(cell, 1 - cell)
    m = (edge < grout).astype(np.float32)
    mosaic = _hash(np.floor(u * tile), np.floor(v * tile), seed) * 0.12
    base = (np.array(c1)[None, None, :] * (1 - mosaic[..., None]) +
            np.array(c2)[None, None, :] * mosaic[..., None])
    rgb = np.clip(base * (0.90 + 0.16 * n[..., None]) * (1 - 0.55 * m[..., None]), 0, 1)
    rough = np.clip(0.45 + 0.25 * n + 0.3 * m, 0.1, 1.0)
    height = 1.0 - m * 0.9
    return rgb, rough, height


def tex_wood(w=1024, h=1024, seed=0, tint=(0.45, 0.30, 0.17), rings=18):
    u, v = _grid(w, h, 1.0)
    warp = fbm(u * 1.5, v * 1.5, seed, octaves=4, base=4)
    grain = np.sin((v * rings + warp * 6.0) * math.pi) * 0.5 + 0.5
    fine = fbm(u * 40, v * 4, seed + 3, octaves=3, base=8)
    tone = 0.70 + 0.30 * grain * (0.6 + 0.4 * fine)
    rgb = np.clip(np.stack([tint[i] * tone for i in range(3)], axis=-1), 0, 1)
    rough = np.clip(0.35 + 0.20 * (1 - grain) + 0.1 * fine, 0.15, 0.9)
    height = grain * 0.6 + fine * 0.4
    return rgb, rough, height


def tex_fabric(w=1024, h=1024, seed=0, tint=(0.20, 0.28, 0.42), pitch=64):
    u, v = _grid(w, h, 1.0)
    weave = 0.5 + 0.5 * np.sin(u * pitch * math.pi) * np.sin(v * pitch * math.pi)
    n = fbm(u, v, seed, octaves=4, base=16)
    rgb = np.clip(np.stack([tint[i] * (0.75 + 0.35 * weave + 0.15 * n) for i in range(3)], axis=-1), 0, 1)
    rough = np.clip(0.85 + 0.12 * (n - 0.5), 0.5, 1.0)
    return rgb, rough, weave * 0.7 + n * 0.3


def tex_metal_brushed(w=1024, h=1024, seed=0, tint=(0.66, 0.67, 0.69), rough=0.30):
    u, v = _grid(w, h, 1.0)
    streak = fbm(u * 2, v * 220, seed, octaves=3, base=6)
    rgb = np.clip(np.stack([tint[i] * (0.88 + 0.20 * streak) for i in range(3)], axis=-1), 0, 1)
    r = np.clip(rough + 0.12 * (streak - 0.5), 0.05, 0.95)
    return rgb, r, streak


def tex_wall_paint(w=1024, h=1024, seed=0, tint=(0.82, 0.79, 0.74)):
    u, v = _grid(w, h, 1.0)
    n = fbm(u, v, seed, octaves=5, base=5)
    blot = fbm(u * 2, v * 2, seed + 5, octaves=3, base=2) * 0.10
    rgb = np.clip(np.stack([tint[i] * (0.94 + 0.09 * n - blot) for i in range(3)], axis=-1), 0, 1)
    rough = np.clip(0.72 + 0.18 * n, 0.2, 1.0)
    return rgb, rough, n * 0.4


def tex_mats(w=1024, h=1024, seed=0, tint=(0.06, 0.24, 0.55), seam=0.02):
    """The blue wall pads / crash mats of a test hall: seat seams + slight sheen variation."""
    u, v = _grid(w, h, 1.0)
    n = fbm(u, v, seed, octaves=4, base=8)
    sy = np.minimum(np.mod(v * 4, 1), 1 - np.mod(v * 4, 1))
    groove = (sy < seam).astype(np.float32)
    rgb = np.clip(np.stack([tint[i] * (0.85 + 0.25 * n) * (1 - 0.5 * groove) for i in range(3)], axis=-1), 0, 1)
    rough = np.clip(0.55 + 0.25 * n + 0.2 * groove, 0.1, 1.0)
    return rgb, rough, 1 - groove


def normal_from_height(height, strength=1.0):
    gy, gx = np.gradient(height.astype(np.float32))
    nx, ny = -gx * strength, -gy * strength
    nz = np.ones_like(nx)
    l = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return np.stack([nx / l, ny / l, nz / l], axis=-1) * 0.5 + 0.5


GENERATORS = {
    "concrete": tex_concrete, "tile": tex_floor_tile, "wood": tex_wood,
    "fabric": tex_fabric, "metal": tex_metal_brushed, "paint": tex_wall_paint, "mats": tex_mats,
}


# --------------------------------------------------------------------------- image IO
def save_png(path, arr):
    """arr float 0..1, shape (h,w,3) or (h,w). Writes a correct 8-bit PNG.

    IMPORTANT: `image.pixels` wants floats 0..1 — do NOT pre-multiply by 255 and cast to uint8,
    which silently produces all-zero maps (black floor, metal-looking paint).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr, 0.0, 1.0)
    name = path.stem
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.new(name, width=w, height=h, alpha=True)
    img.pixels[:] = rgba[::-1].ravel().tolist()   # Blender origin is bottom-left
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    img.source = "FILE"
    img.filepath = str(path)
    return img


def make_maps(name, kind="concrete", out_dir="textures", resolution=1024, seed=0, tint=None):
    """Generate base color + roughness + normal PNGs for one material and return {map: path}."""
    gen = GENERATORS.get(kind, tex_concrete)
    kw = {"w": resolution, "h": resolution, "seed": seed}
    if tint:
        import inspect
        params = inspect.signature(gen).parameters
        if "tint" in params:
            kw["tint"] = tuple(tint)
        elif "c1" in params:                      # tile uses two tile colours
            kw["c1"] = tuple(tint)
            kw["c2"] = tuple(max(0.0, c * 0.82) for c in tint)
    rgb, rough, height = gen(**kw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["base"] = out / f"{name}_base.png"
    paths["rough"] = out / f"{name}_rough.png"
    paths["normal"] = out / f"{name}_normal.png"
    save_png(paths["base"], rgb)
    save_png(paths["rough"], np.repeat(rough[..., None], 3, axis=-1))
    save_png(paths["normal"], normal_from_height(height, 2.0))
    return {k: str(v) for k, v in paths.items()}


def load_image(path, non_color=False):
    img = bpy.data.images.get(Path(path).stem)
    if not img:
        img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


# --------------------------------------------------------------------------- materials
def principled(name, color=(0.7, 0.7, 0.7, 1), rough=0.5, metal=0.0, ior=1.45, emission=None, alpha=1.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()                     # reuse by name must not keep stale brick/stripe nodes
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    bsdf.inputs["Base Color"].default_value = color
    m.diffuse_color = tuple(color[:3]) + (alpha,) if len(color) == 3 else tuple(color)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = ior
    if alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    if emission:
        bsdf.inputs["Emission Color"].default_value = emission
        bsdf.inputs["Emission Strength"].default_value = 2.0
    return m


def _bump_noise(nt, bsdf, scale=8.0, strength=0.08):
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Detail"].default_value = 2.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def preset(name, family="fabric", color=(0.5, 0.5, 0.5), rough=None, metal=None):
    """Material templates by *semantic family*, not by generated stripe textures.

    family: fabric | leather | rubber | wood | metal | screen | plastic | glass | cardboard | paint
    Colors are the only knob the model normally changes; fabric has no corrugation/stripes.
    """
    f = (family or "fabric").lower()
    rgba = tuple(color[:3]) + (1.0,) if len(color) == 3 else tuple(color)
    defaults = {"fabric": 0.86, "leather": 0.42, "rubber": 0.78, "wood": 0.52,
                "metal": 0.34, "screen": 0.18, "plastic": 0.45, "glass": 0.05,
                "cardboard": 0.72, "paint": 0.70}
    m = principled(name, rgba, rough=defaults.get(f, 0.65) if rough is None else rough,
                   metal=(0.0 if metal is None else metal))
    m.diffuse_color = rgba
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if not bsdf:
        return m
    if f == "fabric":
        for key, val in (("Sheen Weight", 0.28), ("Sheen Roughness", 0.65)):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = val
        _bump_noise(nt, bsdf, scale=14.0, strength=0.045)
    elif f == "leather":
        _bump_noise(nt, bsdf, scale=9.0, strength=0.10)
        if "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.12
    elif f == "rubber":
        _bump_noise(nt, bsdf, scale=5.0, strength=0.035)
    elif f == "wood":
        _bump_noise(nt, bsdf, scale=(2.0, 18.0, 2.0) if False else 4.0, strength=0.07)
    elif f == "metal":
        bsdf.inputs["Metallic"].default_value = 0.80
        _bump_noise(nt, bsdf, scale=30.0, strength=0.015)
    elif f == "screen":
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = (0.01, 0.02, 0.03, 1)
            bsdf.inputs["Emission Strength"].default_value = 0.4
    elif f == "glass":
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = 1.0
        m.blend_method = "BLEND"
        bsdf.inputs["Alpha"].default_value = 0.35
    elif f == "cardboard":
        _bump_noise(nt, bsdf, scale=20.0, strength=0.05)
    else:
        _bump_noise(nt, bsdf, scale=10.0, strength=0.02)
    return m


def screen_ui(name="M_Screen_UI", w=512, h=288):
    """A generated desktop/telemetry screen for monitors, TVs and laptops.

    V9 monitors were plain black boxes; the user explicitly asked for the screen to look like a
    computer screen with a page on it. This creates a packed image (no external asset), wires it
    to Base Color + Emission, and keeps the physical bezel separate in the builder.
    """
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    # dark desktop with a lighter app window and a taskbar strip
    ys, xs = np.mgrid[0:h, 0:w]
    u = xs.astype(np.float32) / max(1, w - 1)
    v = ys.astype(np.float32) / max(1, h - 1)
    base = np.zeros((h, w, 4), dtype=np.float32)
    base[..., 0] = 0.035 + 0.05 * (1 - v)
    base[..., 1] = 0.10 + 0.10 * (1 - v)
    base[..., 2] = 0.22 + 0.12 * (1 - v)
    # window frame
    wx0, wx1, wy0, wy1 = int(w * .08), int(w * .92), int(h * .12), int(h * .78)
    base[wy0:wy1, wx0:wx1, :3] = (0.78, 0.82, 0.86)
    base[wy0 + 3:wy0 + 16, wx0:wx1, :3] = (0.10, 0.28, 0.55)
    # code / telemetry-like line blocks
    for row in range(8):
        y = wy0 + 26 + row * 22
        x0 = wx0 + 16
        x1 = wx1 - 18 - (row % 3) * 30
        if x0 < x1:
            base[y:y + 8, x0:x1, :3] = (0.18 + 0.02 * row, 0.22, 0.28)
    # taskbar
    base[int(h * .88):int(h * .96), 8:w - 8, :3] = (0.06, 0.08, 0.12)
    base[int(h * .90):int(h * .94), 12:70, :3] = (0.18, 0.42, 0.72)
    base[..., 3] = 1.0
    img = bpy.data.images.new(name + "_img", width=w, height=h, alpha=True)
    img.pixels = base.reshape(-1).tolist()
    img.pack()
    img.colorspace_settings.name = "sRGB"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if "Emission Color" in bsdf.inputs:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 0.35
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["Metallic"].default_value = 0.0
    m.diffuse_color = (0.18, 0.36, 0.62, 1)
    return m


def floor_rubber(name="M_Floor_Rubber", color=(0.235, 0.235, 0.225)):
    """Dark rubber gym floor with the large square mat seams visible in the video.

    v10 used one near-uniform sheet; this generates a tiny tiling image with 1.2 m seams and
    speckles, wired through Object coordinates so no UV unwrap is needed.
    """
    c = tuple(color[:3]) + (1.0,)
    m = principled(name, color=c, rough=0.88, metal=0.0)
    nt, bsdf = m.node_tree, m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = c
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Metallic"].default_value = 0.0
    try:
        rng = np.random.default_rng(7)
        n = 256
        arr = np.zeros((n, n, 4), dtype=np.float32)
        base = np.array(c[:3], dtype=np.float32)
        # two tiles across the texture -> one tile ~= 1.2 m after Mapping scale 1/2.4
        yy, xx = np.mgrid[0:n, 0:n]
        noise = rng.normal(0, 0.007, (n, n, 1)).astype(np.float32)
        rgb = base[None, None, :] + noise
        seam = ((xx % 128) < 2) | ((yy % 128) < 2)
        rgb[seam] = np.array([0.22, 0.22, 0.205], dtype=np.float32)
        # small pale wear dots / paint flecks, similar to the gym floor in the footage
        for _ in range(18):
            x, y = int(rng.integers(4, n - 4)), int(rng.integers(4, n - 4))
            rgb[y:y+2, x:x+2] = np.array([0.44, 0.44, 0.41], dtype=np.float32)
        arr[..., :3] = np.clip(rgb, 0, 1)
        arr[..., 3] = 1.0
        img = bpy.data.images.get("FloorTiles") or bpy.data.images.new("FloorTiles", n, n, alpha=True)
        img.pixels.foreach_set(arr.ravel())
        img.pack()
        img.colorspace_settings.name = "sRGB"
        vec = _tex_obj_coords(nt)
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1/2.4, 1/2.4, 1/2.4)
        nt.links.new(vec, mp.inputs["Vector"])
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Linear"
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    except Exception:
        pass
    _bump_noise(nt, bsdf, scale=7.0, strength=0.035)
    m.diffuse_color = c
    return m



def floor_pattern(name="M_Floor_Pattern", style="rubber", colors=None, tile_m=0.6):
    """Plan-driven floor material.

    This is the important generalization point: the builder never decides that a room has a
    "dark rubber floor". `architecture.floor` supplies style + colors + tile size; a yellow/blue
    interlocking foam floor is just another data value, not another hard-coded branch.
    """
    style = str(style or "rubber").lower()
    cols = []
    for c in (colors or [(0.235, 0.235, 0.225)]):
        if isinstance(c, str):
            if c.startswith("#") and len(c) == 7:
                cols.append(tuple(int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
            else:
                cols.append({"yellow": (0.86, 0.61, 0.08), "blue": (0.10, 0.26, 0.55),
                             "red": (0.55, 0.08, 0.06), "green": (0.10, 0.40, 0.16),
                             "gray": (0.20, 0.20, 0.19)}.get(c.lower(), (0.235, 0.235, 0.225)))
        else:
            cols.append(tuple(float(v) for v in c[:3]))
    cols = cols or [(0.235, 0.235, 0.225)]
    if style in ("foam", "foam_interlock", "eva", "foam_mat"):
        c1, c2 = cols[0], cols[min(1, len(cols) - 1)]
        n = 512
        yy, xx = np.mgrid[0:n, 0:n]
        # two tiles across the texture; mapping below makes one cell exactly tile_m.
        cell = 256
        checker = ((xx // cell) + (yy // cell)) % 2
        rgb = np.zeros((n, n, 3), np.float32)
        rgb[checker == 0] = c1; rgb[checker == 1] = c2
        # jigsaw-like tabs at the four mid-edges of every cell
        for cx in (cell // 2, cell * 3 // 2):
            for cy in (0, 256, 512):
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                rgb[d < 28] = c2 if ((cx // cell + cy // cell) % 2 == 0) else c1
        for cx in (0, 256, 512):
            for cy in (cell // 2, cell * 3 // 2):
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                rgb[d < 28] = c2 if ((cx // cell + cy // cell) % 2 == 0) else c1
        # bevel / grime at tile borders, not a painted checkerboard
        edge = np.minimum(np.abs((xx % cell) - 0), np.abs((yy % cell) - 0))
        edge = np.minimum(edge, np.minimum(np.abs((xx % cell) - cell), np.abs((yy % cell) - cell)))
        rgb = np.clip(rgb * (0.88 + 0.12 * (edge > 3)[..., None]), 0, 1)
        m = principled(name, color=tuple(c1) + (1.0,), rough=0.72, metal=0.0)
        nt = m.node_tree; bsdf = nt.nodes["Principled BSDF"]
        img = bpy.data.images.get(name + "_img") or bpy.data.images.new(name + "_img", n, n, alpha=True)
        arr = np.ones((n, n, 4), np.float32); arr[..., :3] = rgb
        img.pixels.foreach_set(arr.ravel()); img.pack(); img.colorspace_settings.name = "sRGB"
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1.0 / (2.0 * max(0.05, float(tile_m))),
                                            1.0 / (2.0 * max(0.05, float(tile_m))), 1.0)
        nt.links.new(_tex_obj_coords(nt), mp.inputs["Vector"])
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
        nt.links.new(mp.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        _bump_noise(nt, bsdf, scale=5.0, strength=0.018)
        m.diffuse_color = tuple(c1) + (1.0,)
        return m
    if style in ("tile", "ceramic", "stone"):
        return floor_rubber(name, cols[0])  # tile image is still plan-coloured via base colour
    if style in ("wood", "timber"):
        # a simple plan-coloured wood floor; no hard-coded brown
        m = principled(name, color=tuple(cols[0]) + (1.0,), rough=0.55)
        return m
    return floor_rubber(name, cols[0])



def _tex_obj_coords(nt, loc=(0, 0, 0)):
    """Object-space coords -> no UV unwrap needed for tiled maps."""
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = loc
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    return mp.outputs["Vector"]


def with_maps(name, maps, uv_scale=1.0, rough_scale=None, normal_strength=1.0, tint=None):
    """Wire base/rough/normal PNGs into a Principled BSDF via box projection."""
    m = principled(name)
    if tint:
        m.diffuse_color = tuple(tint[:3]) + (1.0,)
    nt, bsdf = m.node_tree, m.node_tree.nodes["Principled BSDF"]
    vec = _tex_obj_coords(nt)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (uv_scale, uv_scale, uv_scale)

    def img_tex(key, non_color):
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = load_image(maps[key], non_color=non_color)
        node.interpolation = "Smart"
        return node

    if "base" in maps:
        t = img_tex("base", False)
        nt.links.new(vec, t.inputs["Vector"])
        if tint:
            mix = nt.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 1.0
            mix.inputs["Color2"].default_value = (*tint, 1)
            nt.links.new(t.outputs["Color"], mix.inputs["Color1"])
            nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    if "rough" in maps:
        t = img_tex("rough", True)
        nt.links.new(vec, t.inputs["Vector"])
        nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
    if "normal" in maps:
        t = img_tex("normal", True)
        nt.links.new(vec, t.inputs["Vector"])
        nrm = nt.nodes.new("ShaderNodeNormalMap")
        nrm.inputs["Strength"].default_value = normal_strength
        nt.links.new(t.outputs["Color"], nrm.inputs["Color"])
        nt.links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def node_brick(name, c1=(0.55, 0.35, 0.25, 1), c2=(0.44, 0.28, 0.20, 1), mortar=0.02, scale=1.0):
    m = principled(name)
    m.diffuse_color = tuple(c1[:3]) + (1.0,)
    nt, bsdf = m.node_tree, m.node_tree.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale,) * 3
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Mortar Size"].default_value = mortar
    br.inputs["Color1"].default_value = c1
    br.inputs["Color2"].default_value = c2
    nt.links.new(mp.outputs["Vector"], br.inputs["Vector"])
    nt.links.new(br.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], bsdf.inputs["Roughness"])
    return m


def node_noise_rough(name, base=(0.5, 0.5, 0.5, 1), rough=(0.35, 0.85), scale=4.0):
    m = principled(name, color=base, rough=rough[0])
    nt, bsdf = m.node_tree, m.node_tree.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (rough[0],) * 3 + (1,)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (rough[1],) * 3 + (1,)
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Roughness"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.15
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def glass(name, tint=(0.92, 0.96, 1.0, 1), rough=0.02, ior=1.45):
    m = principled(name, color=tint, rough=rough, ior=ior, alpha=0.25)
    bsdf = m.node_tree.nodes["Principled BSDF"]
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
    return m


def emissive(name, color=(1, 0.96, 0.88, 1), strength=4.0):
    m = principled(name, color=color, rough=0.4, emission=color)
    m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = strength
    return m


def wiring_report():
    """Which materials actually have image textures connected to the BSDF."""
    out = {"total": len([m for m in bpy.data.materials if m.use_nodes]), "wired": [], "plain": []}
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        imgs = [n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
        linked = any(n.outputs["Color"].is_linked or n.outputs["Alpha"].is_linked for n in imgs) if imgs else False
        (out["wired"] if linked else out["plain"]).append(m.name)
    return out


def world(color=(0.62, 0.66, 0.72), strength=1.0):
    """Neutral sky/ambient. Keep it weak: a bright flat world hides wrong geometry."""
    w = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*color, 1)
    bg.inputs["Strength"].default_value = strength
    return w


def exposure(value=0.0, view="AgX", look="AgX - Medium High Contrast"):
    sc = bpy.context.scene
    sc.view_settings.view_transform = view
    sc.view_settings.look = look
    sc.view_settings.exposure = value
    return value


def area_light(name, loc, target, energy=200.0, size=2.0, color=(1.0, 0.97, 0.92)):
    """Area light aimed at `target` — the euler bug that made a fill light hit the ceiling cannot happen here."""
    from mathutils import Vector
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = Vector(loc)
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


def sun(name="Sun", energy=4.0, angle_deg=18.0, rot=(55, 0, 35), color=(1.0, 0.96, 0.9)):
    ld = bpy.data.lights.new(name, "SUN")
    ld.energy = energy
    ld.angle = math.radians(angle_deg)
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.rotation_euler = tuple(math.radians(a) for a in rot)
    return ob


def build_library(out_dir="textures", resolution=1024, seed=0, mode="both", kinds=None):
    """Create the standard material set for an equipment/test hall and return {name: material}."""
    kinds = kinds or ["paint", "tile", "concrete", "wood", "fabric", "metal", "mats"]
    lib = {}
    tints = {
        "paint": (0.80, 0.77, 0.72), "tile": (0.70, 0.68, 0.64), "concrete": (0.50, 0.50, 0.49),
        "wood": (0.42, 0.28, 0.16), "fabric": (0.18, 0.26, 0.40), "metal": (0.62, 0.64, 0.66),
        "mats": (0.05, 0.22, 0.52),
    }
    for k in kinds:
        if mode in ("images", "both"):
            maps = make_maps(f"M_{k}", k, out_dir=out_dir, resolution=resolution, seed=seed, tint=tints[k])
            lib[k] = with_maps(f"M_{k}", maps, uv_scale={ "tile": 2.0, "wood": 1.0, "fabric": 3.0 } .get(k, 1.0))
        else:
            lib[k] = node_noise_rough(f"M_{k}", base=(*tints[k], 1.0))
    if mode in ("nodes", "both"):
        lib["brick"] = node_brick("M_brick")
        lib["metal_brushed"] = node_noise_rough("M_metal_brushed", base=(0.62, 0.64, 0.66, 1), rough=(0.15, 0.45), scale=80)
    lib["glass"] = glass("M_glass")
    lib["lamp"] = emissive("M_lamp")
    lib["plastic_dark"] = principled("M_plastic_dark", (0.06, 0.07, 0.08, 1), 0.45)
    lib["plastic_orange"] = principled("M_plastic_orange", (0.85, 0.33, 0.06, 1), 0.4)
    lib["rubber"] = principled("M_rubber", (0.03, 0.03, 0.035, 1), 0.85)
    return lib
