"""Component-tree schema for objects that are actually visible in the video.

A single global template is not enough.  This module is the compact, inspectable "tree of
parts" the LLM must fill before building: how many parts, which part is the parent, which
parts physically connect, and which parts the source evidence does not show (those may be
inferred).  Inferred background objects may use a coarse tree; clearly seen objects must have
their connections explicit.  Builders may use bs_assets as reference, but the tree is the
contract they must satisfy.
"""
from __future__ import annotations

TREES = {
    "barrier": [
        {"part": "cloth_panel", "count": 1, "parent": None, "connects": ["top_rail", "posts"]},
        {"part": "top_rail", "count": 1, "parent": "cloth_panel", "connects": ["posts"]},
        {"part": "posts", "count": 2, "parent": None, "connects": ["top_rail", "feet"]},
        {"part": "foot_black_base", "count": 2, "parent": "posts", "connects": ["floor"]},
        {"part": "foot_red_top", "count": 2, "parent": "foot_black_base", "connects": ["posts"]},
        {"part": "top_clamps", "count": 2, "parent": "posts", "connects": ["cloth_panel"]},
    ],
    "chair": [
        {"part": "seat", "count": 1, "parent": None, "connects": ["back", "gas_column"]},
        {"part": "back", "count": 1, "parent": "seat", "connects": []},
        {"part": "gas_column", "count": 1, "parent": "seat", "connects": ["five_star_base"]},
        {"part": "five_star_base", "count": 5, "parent": "gas_column", "connects": ["castors"]},
        {"part": "castors", "count": 5, "parent": "five_star_base", "connects": []},
        {"part": "armrests", "count": 2, "parent": "seat", "connects": []},
    ],
    "bench": [
        {"part": "seat", "count": 1, "parent": None, "connects": ["end_frames"]},
        {"part": "end_frames", "count": 2, "parent": "seat", "connects": ["stretcher"]},
        {"part": "stretcher", "count": 2, "parent": "end_frames", "connects": []},
    ],
    "desk": [
        {"part": "top", "count": 1, "parent": None, "connects": ["legs"]},
        {"part": "legs", "count": 4, "parent": "top", "connects": ["stretcher"]},
        {"part": "stretcher", "count": 1, "parent": "legs", "connects": []},
    ],
    "table": [
        {"part": "top", "count": 1, "parent": None, "connects": ["legs"]},
        {"part": "legs", "count": 4, "parent": "top", "connects": ["stretcher"]},
        {"part": "stretcher", "count": 1, "parent": "legs", "connects": []},
    ],
    "monitor": [
        {"part": "panel", "count": 1, "parent": None, "connects": ["neck"]},
        {"part": "neck", "count": 1, "parent": "panel", "connects": ["stand"]},
        {"part": "stand", "count": 1, "parent": "neck", "connects": ["desk"]},
    ],
    "sofa": [
        {"part": "seat_cushions", "count": 2, "parent": "frame", "connects": []},
        {"part": "back_cushions", "count": 2, "parent": "frame", "connects": []},
        {"part": "frame", "count": 1, "parent": None, "connects": ["seat_cushions", "back_cushions", "feet"]},
        {"part": "feet", "count": 4, "parent": "frame", "connects": ["floor"]},
    ],
    "shelf": [
        {"part": "posts", "count": 4, "parent": None, "connects": ["shelves"]},
        {"part": "shelves", "count": 4, "parent": "posts", "connects": []},
    ],
    "cabinet": [
        {"part": "case", "count": 1, "parent": None, "connects": ["drawers"]},
        {"part": "drawers", "count": 3, "parent": "case", "connects": []},
    ],
    "window": [
        {"part": "glass", "count": 1, "parent": "frame", "connects": []},
        {"part": "frame", "count": 1, "parent": None, "connects": ["glass", "wall"]},
        {"part": "mullions", "count": "cols+rows", "parent": "frame", "connects": ["glass"]},
        {"part": "blind", "count": "0|top_half|full", "parent": "frame", "connects": []},
        {"part": "curtain", "count": "0|1", "parent": "frame", "connects": ["head_rail"]},
        {"part": "curtain_head_rail", "count": 1, "parent": "curtain", "connects": []},
    ],
}


def tree_for(kind: str) -> list[dict]:
    return [dict(p) for p in TREES.get(kind, [{"part": "body", "count": 1, "parent": None, "connects": []}])]


def validate_tree(obj: dict) -> list[str]:
    """Return structural errors the LLM must resolve before calling an asset builder."""
    errs = []
    tree = obj.get("component_tree") or []
    if not tree:
        return ["missing component_tree"]
    names = {p.get("part") for p in tree}
    for p in tree:
        if p.get("parent") and p["parent"] not in names:
            errs.append(f"{p.get('part')}: parent {p['parent']} not in tree")
        for c in (p.get("connects") or []):
            if c not in names and c not in ("floor", "wall", "desk"):
                errs.append(f"{p.get('part')}: connection {c} not represented")
    return errs


def enrich_plan(plan: dict) -> dict:
    for o in (list(plan.get("objects") or []) + list(plan.get("inferred") or [])):
        if not o.get("component_tree"):
            o["component_tree"] = tree_for(str(o.get("kind") or ""))
    for w in (plan.get("architecture") or {}).get("windows") or []:
        if not w.get("components"):
            w["components"] = tree_for("window")
    return plan
