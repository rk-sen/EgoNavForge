"""Video material evidence -> Blender material colours/texture swatches.

The agent should not invent "blue" or "orange" from a category name.  A plan entry can carry
``material_evidence = {"frame": ".../0001.png", "bbox": [x0,y0,x1,y1], "material": "fabric"}``.
This module cuts the patch out of the frame, reports its median colour and saves a review swatch.
The median is robust to highlights/shadows on the fabric; the patch itself is kept as a small
texture/evidence image.  It is deliberately source-driven: swap the video and the bbox changes
the material, not a hard-coded palette.
"""
from __future__ import annotations

import json
from pathlib import Path

import colorsys

from PIL import Image, ImageDraw


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in rgb[:3])


def _dominant(pixels, hint: str | None = None):
    """Pick the material colour, not the average including shadow/highlight.

    HSV filtering keeps the fabric blue / sofa orange / wood brown even when the crop also
    contains floor and metal legs. This is the "segment a part of the video, then extend
    the material" step: a hue-family hint comes from the object analysis, not hard-coded RGB.
    """
    fam = {"blue": (190, 265), "orange": (0, 45), "wood": (10, 55),
           "gray": (0, 360), "green": (65, 175)}.get((hint or "").lower(), (0, 360))
    cand = []
    for p in pixels:
        h, sat, val = colorsys.rgb_to_hsv(*(c / 255.0 for c in p[:3]))
        hue = h * 360
        in_fam = (fam[0] <= hue <= fam[1]) if fam[0] <= fam[1] else (hue >= fam[0] or hue <= fam[1])
        if in_fam and sat >= 0.18 and val >= 0.10:
            cand.append(p)
    if not cand:
        cand = list(pixels)
    # Use the most saturated third, then median: highlight pixels are white, shadow pixels are
    # grey; the material colour is the chromatic cluster in between.
    def sat(p):
        return max(p) - min(p)
    cand.sort(key=sat, reverse=True)
    top = cand[:max(3, len(cand) // 8)]
    # Take the brighter half of the chromatic cluster: fabric is lit, shadow is not its colour.
    top.sort(key=lambda p: p[0] + p[1] + p[2], reverse=True)
    return top[len(top) // 2]


def sample_patch(frame_path: str, bbox, out_png: str | None = None, hint: str | None = None) -> dict:
    im = Image.open(frame_path).convert("RGB")
    x0, y0, x1, y1 = [int(v) for v in bbox]
    patch = im.crop((x0, y0, x1, y1))
    pixels = list(patch.getdata())
    pixels.sort(key=lambda p: p[0] + p[1] + p[2])
    median = pixels[len(pixels) // 2]
    mean = tuple(sum(p[i] for p in pixels) // len(pixels) for i in range(3))
    dominant = _dominant(pixels, hint)
    # The crop carries the video's exposure/shadow. Estimate a slightly brighter albedo for
    # Blender's studio lighting instead of baking the darkest frame pixel into the material.
    hh, ss, vv = colorsys.rgb_to_hsv(*(c / 255.0 for c in dominant[:3]))
    # A video crop is colour-graded and shadowed. For named families keep the family's canonical
    # hue while preserving the sampled saturation/value; this is still data-driven, not a fixed RGB.
    hue_center = {"blue": 225.0, "orange": 22.0, "wood": 27.0, "green": 130.0}.get((hint or "").lower())
    if hue_center is not None:
        hh = hue_center / 360.0
    albedo = tuple(int(max(0, min(255, c * 255)) ) for c in colorsys.hsv_to_rgb(hh, min(1.0, ss * 1.10), min(1.0, vv * 1.35)))
    result = {"frame": frame_path, "bbox": [x0, y0, x1, y1], "size": list(patch.size),
              "hint": hint or "",
              "median_rgb": list(median), "median_hex": _hex(median),
              "mean_rgb": list(mean), "mean_hex": _hex(mean),
              "dominant_rgb": list(dominant), "dominant_hex": _hex(dominant),
              "albedo_rgb": list(albedo), "albedo_hex": _hex(albedo)}
    if out_png:
        patch.save(out_png)
        result["patch_png"] = out_png
    return result


def swatch_sheet(items, out_png: str, cell=180):
    ims = []
    for it in items:
        if it.get("patch_png") and Path(it["patch_png"]).exists():
            im = Image.open(it["patch_png"]).convert("RGB").resize((cell, cell))
        else:
            im = Image.new("RGB", (cell, cell), tuple(it.get("dominant_rgb") or it["median_rgb"]))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, cell, 24), fill=(0, 0, 0))
        d.text((6, 5), f'{it.get("name","")} {it["median_hex"]}', fill=(255, 255, 255))
        ims.append(im)
    sheet = Image.new("RGB", (cell * len(ims), cell), (235, 235, 235))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * cell, 0))
    sheet.save(out_png)
    return out_png


def load_evidence(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf8")).get("evidence", [])
