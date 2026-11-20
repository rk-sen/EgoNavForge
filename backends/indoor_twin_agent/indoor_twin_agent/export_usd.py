"""Export a Blender twin as Isaac-friendly USD.

Two products:
  usd/full_scene.usd          whole room (walls, windows, door, ceiling, lights, all objects)
  usd/objects/<item>.usd      one USD per *unique* item type; repeated instances are not re-exported

Collision:
  Blender's USD exporter has no collision switch, so the stage is post-processed with
  UsdPhysics.  Architecture uses triangle-mesh collision ("none"), movable furniture uses
  convex hull/box approximation.  This is the same role collision authoring plays in
  Infinigen's Blender/Isaac exports.

Run under Blender:
  blender -b scene.blend --python export_usd.py -- --project <project>
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import bpy
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

ARCH_TOKENS = ("Wall", "Floor", "Ceiling", "Door", "Mullion", "Window", "Lamp", "Light",
               "FloorSeam", "Cam_", "Sun_")


def _args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", default="")
    return ap.parse_args(argv)


def _is_item(o) -> bool:
    n = o.name
    if o.type not in {"MESH", "EMPTY"}:
        return False
    if o.parent is not None and o.name.startswith("O") and o.parent.name.startswith("O"):
        # child mesh of an item root: root owns it
        return False
    if not n.startswith("O"):
        return False
    return not any(t in n for t in ARCH_TOKENS)


def _descendants(o):
    out = [o]
    for c in o.children:
        out.extend(_descendants(c))
    return out


def _asset_key(o) -> str:
    n = o.name
    n = re.sub(r"__\d+$", "", n)
    n = re.sub(r"_mesh$", "", n)
    # inferred box1 / box2 are the same cardboard box asset
    n = re.sub(r"(?i)(box)\d+$", r"\1", n)
    # monitor/shelf duplicates keep their numbered root in the source name, but the same
    # base item is exported once.
    n = re.sub(r"_\d+$", "", n)
    return n


def _add_collision(prim, approx: str):
    if not prim.IsA(UsdGeom.Mesh):
        return
    UsdPhysics.CollisionAPI.Apply(prim)
    api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    api.CreateApproximationAttr(approx)
    # explicit so Isaac does not silently drop a collider authored by a partial schema
    prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)


def _post_collisions(path: Path, architecture: bool):
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = prim.GetName()
        parent = prim.GetParent().GetName() if prim.GetParent() else ""
        is_arch = architecture and any(t in name or t in parent for t in ARCH_TOKENS)
        approx = "none" if is_arch else "convexHull"
        _add_collision(prim, approx)
    stage.GetRootLayer().Save()


def main():
    a = _args()
    project = Path(a.project)
    out = Path(a.out) if a.out else project / "usd"
    objdir = out / "objects"
    objdir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(project / "scene.blend"))
    items = [o for o in bpy.data.objects if _is_item(o)]
    # unique by asset key, prefer the first instance
    unique = {}
    for o in items:
        unique.setdefault(_asset_key(o), o)
    print("USD_UNIQUE", len(unique), sorted(unique))
    manifest_items = []
    for key, root in sorted(unique.items()):
        # select root + descendants; exporter follows selected hierarchy
        for o in bpy.context.selected_objects:
            o.select_set(False)
        for o in _descendants(root):
            o.select_set(True)
        bpy.context.view_layer.objects.active = root
        fn = objdir / f"{key}.usd"
        bpy.ops.wm.usd_export(
            filepath=str(fn), selected_objects_only=True, visible_objects_only=False,
            export_materials=True, export_lights=False, export_cameras=False,
            export_animation=False, export_normals=True, export_uvmaps=True,
            export_global_up_selection="Z", export_global_forward_selection="Y",
            use_instancing=False, root_prim_path="/World",
        )
        _post_collisions(fn, architecture=False)
        manifest_items.append({"key": key, "file": str(fn.relative_to(out)),
                               "source": root.name,
                               "instances": [o.name for o in items if _asset_key(o) == key]})
    # whole scene: all architecture + all object instances + lights, no cameras
    for o in bpy.context.selected_objects:
        o.select_set(False)
    full = out / "full_scene.usd"
    bpy.ops.wm.usd_export(
        filepath=str(full), selected_objects_only=False, visible_objects_only=True,
        export_materials=True, export_lights=True, export_cameras=False,
        export_animation=False, export_normals=True, export_uvmaps=True,
        export_global_up_selection="Z", export_global_forward_selection="Y",
        use_instancing=True, root_prim_path="/World",
    )
    _post_collisions(full, architecture=True)
    manifest = {
        "full_scene": str(full.relative_to(out)),
        "units": "meters",
        "up_axis": "Z",
        "architecture": "included in full_scene only; walls/windows/door/ceiling/lamps are not separate assets",
        "unique_items": len(manifest_items),
        "items": manifest_items,
        "collision": {"architecture": "triangle mesh (UsdPhysics.MeshCollisionAPI approximation=none)",
                      "items": "convex hull (UsdPhysics.MeshCollisionAPI approximation=convexHull)"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print("USD_DONE", full, len(manifest_items))


if __name__ == "__main__":
    main()
