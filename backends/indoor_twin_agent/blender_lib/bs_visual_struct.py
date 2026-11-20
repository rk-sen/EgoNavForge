"""Image-led structure/material inspection for the lightweight agent.

The point is to stop treating "blind" or "white desk" as a template default.  A window crop is
classified from the video itself: continuous bright glazing -> plain glass; repeated horizontal
dark bands in the upper half -> top-half blind; vertical grid lines -> mullions; a soft opaque
sheet with folds -> curtain.  The result is a proposal with evidence, not a hard-coded schema.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


def _gray_edges(im: Image.Image, axis: str) -> list[float]:
    g = im.convert("L")
    w, h = g.size
    px = g.load()
    vals = []
    if axis == "horizontal":
        for y in range(1, h):
            s = 0.0
            for x in range(1, w, max(1, w // 40)):
                s += abs(px[x, y] - px[x, y - 1])
            vals.append(s / max(1, w // 40))
    else:
        for x in range(1, w):
            s = 0.0
            for y in range(1, h, max(1, h // 40)):
                s += abs(px[x, y] - px[x - 1, y])
            vals.append(s / max(1, h // 40))
    return vals


def inspect_window_crop(path: str | Path, bbox: list[int] | None = None) -> dict[str, Any]:
    """Inspect one window crop and return a structure proposal with measurable evidence."""
    im = Image.open(path).convert("RGB")
    if bbox:
        im = im.crop(tuple(int(v) for v in bbox))
    w, h = im.size
    top = im.crop((0, 0, w, int(h * .52)))
    bottom = im.crop((0, int(h * .52), w, h))
    stat = ImageStat.Stat(im)
    top_edges = _gray_edges(top, "horizontal")
    bot_edges = _gray_edges(bottom, "horizontal")
    # number of strong horizontal bands, localised in the top half
    strong = sum(1 for v in top_edges if v > (sum(top_edges) / max(1, len(top_edges))) * 2.0)
    grid = sum(1 for v in _gray_edges(im, "vertical") if v > 45)
    blind = "top_half" if strong >= 4 and sum(top_edges) > sum(bot_edges) * 1.6 else "none"
    # A curtain is not a blind: it is a soft, low-saturation vertical sheet whose folds give
    # vertical edges while the glazing behind it is still bright.  Measure that separately so
    # the agent can choose "curtain" instead of forcing "blind" on every window.
    pix = list(im.resize((64, 48)).getdata())
    low_sat_bright = 0
    for (r, g, b) in pix:
        mx, mn = max(r, g, b), min(r, g, b)
        if (r + g + b) / 3.0 > 135 and (mx - mn) < 34:
            low_sat_bright += 1
    curtain_score = low_sat_bright / max(1, len(pix))
    curtain = {"draw": 0.0, "side": "left"} if curtain_score >= .55 else None
    if curtain and blind == "none":
        curtain["draw"] = 0.5
    if curtain and blind == "none":
        wtype = "curtain"
    else:
        wtype = "blind_window" if blind == "top_half" else "plain_glass"
    return {
        "type": wtype,
        "blind": blind,
        "curtain": curtain,
        "mullions": {"cols": 1 if grid > 1 else 0, "rows": 1},
        "mean_rgb": [round(float(v), 1) for v in stat.mean[:3]],
        "top_horizontal_bands": strong,
        "vertical_grid_edges": grid,
        "low_saturation_bright_fraction": round(curtain_score, 3),
        "source": str(path),
        "bbox": bbox,
        "confidence": round(min(1.0, max(0.0, .62 + (curtain_score if curtain else 0))), 2),
        "note": "proposal from image; agent must merge with video context, not force a template",
    }


def inspect_flat_color(path: str | Path, bbox: list[int] | None = None) -> dict[str, Any]:
    """Estimate the dominant flat colour of a visible object patch (e.g. the white desk)."""
    im = Image.open(path).convert("RGB")
    if bbox:
        im = im.crop(tuple(int(v) for v in bbox))
    im = im.resize((64, 48))
    px = list(im.getdata())
    # ignore near-black/white highlights and use the median of the middle luminance band
    mid = sorted(px, key=lambda p: sum(p))[len(px) // 2]
    return {"rgb": [int(v) for v in mid], "hex": "#%02x%02x%02x" % tuple(mid),
            "source": str(path), "bbox": bbox}
