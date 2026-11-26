"""Build the AssetBank the twin agent retrieves from.

Run inside Blender:

    blender -b --python build_asset_bank.py -- --lib <office_library_clean> --out <AssetBank>

Each asset becomes AT_<id>/ with Prototype.blend (textures packed), Asset.usda,
Metadata.json and Thumbnail.jpg. The sqlite index is what Catalog.entries() reads.
"""
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import bpy


def _args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("pass -- --lib DIR --out DIR")
    rest = argv[argv.index("--") + 1 :]
    lib = out = None
    i = 0
    while i < len(rest):
        if rest[i] == "--lib":
            lib = Path(rest[i + 1])
            i += 2
        elif rest[i] == "--out":
            out = Path(rest[i + 1])
            i += 2
        else:
            raise SystemExit(f"unknown arg {rest[i]}")
    if lib is None or out is None:
        raise SystemExit("need --lib and --out")
    return lib.resolve(), out.resolve()


def texture_kind(cognitive):
    fam = cognitive.get("material_family", "")
    if "wood" in fam:
        return "wood"
    if "cloth" in fam:
        return "cloth"
    if "leather" in fam:
        return "leather"
    if "metal" in fam:
        return "metal"
    if "glass" in fam:
        return "paint"
    if "rubber" in fam:
        return "rubber"
    return "paint"


def aid_for(asset_id):
    return "AT_" + hashlib.sha256(("clean-v3.1|" + asset_id).encode()).hexdigest()[:24]


def pack_images():
    for img in list(bpy.data.images):
        if img.source != "FILE" or img.packed_file is not None:
            continue
        try:
            img.pack()
        except Exception as exc:
            print(f"PACK_SKIP {img.name}: {exc}", flush=True)


def main():
    lib, out = _args()
    cat = json.loads((lib / "catalog.json").read_text(encoding="utf-8"))
    cog = {x["asset_id"]: x for x in json.loads((lib / "cognitive_catalog.json").read_text(encoding="utf-8"))}
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    missing = []
    for asset in cat["assets"]:
        src = lib / asset["usd"]
        cognitive = cog.get(asset["asset_id"])
        if cognitive is None or not src.is_file():
            missing.append(asset["asset_id"])
            print("MISSING", asset["asset_id"], flush=True)
            continue
        at = aid_for(asset["asset_id"])
        folder = out / at
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, folder / "Asset.usda")
        front = lib / asset["thumbnails"]["front"]
        if front.is_file():
            shutil.copy2(front, folder / "Thumbnail.jpg")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.usd_import(filepath=str(src))
        coll = bpy.data.collections.new(asset["asset_id"])
        bpy.context.scene.collection.children.link(coll)
        for obj in list(bpy.data.objects):
            for old in list(obj.users_collection):
                old.objects.unlink(obj)
            coll.objects.link(obj)
        pack_images()
        blend = folder / "Prototype.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend), check_existing=False)
        dims = [float(v) for v in asset["bbox_m"]]
        color = asset.get("display_color") or [0.5, 0.5, 0.5]
        entry = {
            "asset_id": at,
            "query_id": asset["asset_id"],
            "semantic": cognitive["semantic"],
            "dimensions_m": dims,
            "description": asset.get("description", ""),
            "material": {
                "color": color,
                "texture": texture_kind(cognitive),
                "roughness": 0.62,
                "metallic": 0.0,
            },
            "shape_parts": list(cognitive.get("aliases", [])) + [str(t) for t in asset.get("tags", [])],
            "procedural_hash": hashlib.sha256(("native:" + asset["asset_id"]).encode()).hexdigest(),
            "geometry_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "public_prefab": True,
            "provider": (asset.get("source") or {}).get("author", "external"),
            "observation_description": asset.get("description", ""),
            "requested_dimensions_m": dims,
            "prototype_key": asset["asset_id"],
            "blend_path": str(blend),
            "material_definitions": {},
            "preferred_pose": asset["asset_id"] == "v39_person_supplied",
            "aliases": cognitive.get("aliases", []),
            "scene_types": cognitive.get("scene_types", []),
            "spatial_roles": cognitive.get("spatial_roles", []),
            "support_capability": cognitive.get("support_capability", False),
            "material_family": cognitive.get("material_family", ""),
            "negative_roles": cognitive.get("negative_roles", []),
            "instance_unit": cognitive.get("instance_unit", "single"),
            "thumbnail": "Thumbnail.jpg",
            "cognitive": cognitive,
        }
        (folder / "Metadata.json").write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append(entry)
        print(f"BANK {len(manifest)} {asset['asset_id']}", flush=True)

    db = out / "index.sqlite3"
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE assets(id TEXT PRIMARY KEY, semantic TEXT NOT NULL, metadata TEXT NOT NULL)")
    con.execute("CREATE INDEX semantic_lookup ON assets(semantic)")
    for entry in manifest:
        con.execute(
            "INSERT INTO assets VALUES(?,?,?)",
            (entry["asset_id"], entry["semantic"], json.dumps(entry, ensure_ascii=False, sort_keys=True)),
        )
    con.commit()
    con.close()
    (out / "AssetBankManifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BANK DONE {len(manifest)} missing={missing}", flush=True)
    if len(manifest) < 80:
        raise SystemExit(f"asset bank too small: {len(manifest)}")


if __name__ == "__main__":
    main()
