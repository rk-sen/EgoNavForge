"""Run inside Blender: build shell + assets + trajectory frames for one project.

Usage:
  blender -b --factory-startup --python blender_build.py -- \
      --project /abs/run/project [--frames 60] [--res 384] [--fps 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def args_after_dash():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--no-render", action="store_true")
    return ap.parse_args(argv)


def main():
    a = args_after_dash()
    root = Path(a.project).resolve()
    sys.path.insert(0, str(root / "_lib"))
    import bs_layout, bs_assets, bs_trajvideo
    plan_path = root / "SCENE_PLAN.json"
    cal_path = root / "calibration.json"
    specs_path = next((p for p in (root/"specs_v22.json", root/"specs_v16.json", root/"specs.json") if p.exists()), root/"specs_v22.json")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    specs = json.loads(specs_path.read_text(encoding="utf8"))
    # The user supplied a mannequin for the human in this room.  If the count pass did
    # not add a person, keep a person in the scene anyway (the video visibly contains one).
    if not any(str(s.get("kind")) == "person" for s in specs):
        poly = (json.loads((root / "SCENE_PLAN.json").read_text(encoding="utf8"))
                .get("architecture") or {}).get("footprint_m") or [[0, 0], [1, 0], [1, 1], [0, 1]]
        xs = [float(q[0]) for q in poly]; ys = [float(q[1]) for q in poly]
        cx = (min(xs) + max(xs)) * 0.5; cy = (min(ys) + max(ys)) * 0.5
        specs.append({"id": "O18_Person", "dims": [0.45, 0.32, 1.72], "count": 1,
                      "kind": "person", "name": "O18_Person", "material": "布料/皮肤",
                      "color": "blue", "gender": "male",
                      "placements": [[cx, cy, 0.0, 0.0]], "relation": {},
                      "source": "supplied_mannequin_default"})
    # architecture in the plan is the model-generated one; no BLUEPRINT overwrite.
    bs_layout.build(str(cal_path), str(plan_path), str(root/"scene.blend"),
                    str(root/"previews"), res=640, render=False)
    # Barrier groups come from the count pass: e.g. 3+3+2 panels, not the two
    # anchors that happened to be the first visible panels.  Expand them into per-panel
    # placements so the metal frame / cloth / red feet are repeated.
    fp_box = json.loads((root / "SCENE_PLAN.json").read_text(encoding="utf8")).get("architecture", {}).get("footprint_m") or [[0, 0], [1, 0], [1, 1], [0, 1]]
    bx0, bx1 = min(float(q[0]) for q in fp_box), max(float(q[0]) for q in fp_box)
    by0, by1 = min(float(q[1]) for q in fp_box), max(float(q[1]) for q in fp_box)
    for s in specs:
        groups = s.get("groups") or []
        if not groups or s.get("kind") != "barrier":
            continue
        base = s.get("placements") or [[0, 0, 0, 0]]
        total = sum(int(g.get("count") or len(g.get("panel_indices") or []) or 0)
                    for g in groups)
        # auto_scene.py already emitted one placement per panel, with the model's run
        # direction, wall snap and group separation.  The old V10 expansion below used a
        # fixed step and invented rows, overriding those placements and making three
        # groups all land in one cluster.  Only expand when the upstream specs truly have
        # fewer placements than the group count.
        if len(base) >= total and total > 0:
            print("KEEP_BARRIER_PLACEMENTS", s["id"], len(base))
            continue
        b0 = base[0]; yaw0 = float(b0[3] if len(b0) > 3 else 0)
        pls = []
        step = 1.05
        for gi, g in enumerate(groups):
            n = int(g.get("count") or len(g.get("panel_indices") or []) or 0)
            for k in range(n):
                if gi == 0:
                    x, y = b0[0] + k * step, b0[1]
                elif gi == 1:
                    x, y = b0[0], b0[1] + k * step
                else:
                    x, y = b0[0] + (k + 1) * step, b0[1] + step
                x = max(bx0 + 0.80, min(bx1 - 0.80, x))
                y = max(by0 + 0.80, min(by1 - 0.80, y))
                pls.append([x, y, 0.0, (yaw0 + (90 if gi else 0)) % 360])
        s["placements"] = pls
        s["count"] = len(pls)
        print("EXPAND_BARRIER", s["id"], len(pls), [g.get("count") for g in groups])
    ids = [s["id"] for s in specs]
    for o in list(bpy.data.objects):
        if any(o.name == i or o.name.startswith(i+"_") or o.name.startswith(i+"__")
               or o.name.startswith(i+".") for i in ids):
            bpy.data.objects.remove(o, do_unlink=True)
    for s in specs:
        bs_assets.build_instance(s)
    bpy.ops.wm.save_mainfile()
    print("SCENE_BUILT", len(specs), sum(int(s.get("count", 1)) for s in specs), "instances")
    # Full previews: uncut aerial, interior showcase, and a true top-down plan render.
    try:
        import bs_stage
        prev = root / "previews"
        prev.mkdir(parents=True, exist_ok=True)
        bs_stage.render_cam("Cam_Overview", str(prev / "final_overview.png"), res=384,
                            hide=("Ceiling", "Ceiling_Lamp", "INF_Ceiling"))
        bs_stage.render_cam("Cam_Showcase", str(prev / "final_showcase.png"), res=384,
                            hide=("Ceiling", "Ceiling_Lamp", "INF_Ceiling"))
        # camera looking straight down, orthographic, without ceiling
        poly = (json.loads((root/"SCENE_PLAN.json").read_text(encoding="utf8").replace("true","true"))
                .get("architecture") or {}).get("footprint_m") or [[0,0],[1,0],[1,1],[0,1]]
        xs=[float(q[0]) for q in poly]; ys=[float(q[1]) for q in poly]
        cx=(min(xs)+max(xs))*.5; cy=(min(ys)+max(ys))*.5
        size=max(max(xs)-min(xs), max(ys)-min(ys)) + 0.8
        td=bpy.data.cameras.new("Cam_Top"); td.type='ORTHO'; td.ortho_scale=size
        top=bpy.data.objects.new("Cam_Top", td); bpy.context.scene.collection.objects.link(top)
        top.location=(cx,cy,max(3.0, float(((bpy.data.objects.get("Ceiling") and 3.0) or 3.0)))+0.5)
        top.rotation_euler=(0,0,0)
        bpy.context.scene.camera=top
        # Hide the roof and lights only for THIS top-down shot.  v26 saved the .blend while
        # they were still hidden, so opening the scene looked like the ceiling/lights had been
        # deleted.  Keep the objects in the scene, restore their render flags afterwards.
        _hidden = []
        for o in bpy.data.objects:
            if any(o.name.startswith(p) for p in ("Ceiling", "Ceiling_Lamp", "Light_",
                                                   "INF_Ceiling", "INF_Light", "StripLight")):
                _hidden.append((o, bool(o.hide_render))); o.hide_render = True
        bpy.context.scene.render.resolution_x=480; bpy.context.scene.render.resolution_y=480
        bpy.context.scene.render.resolution_percentage=100
        bpy.context.scene.render.filepath=str(prev/"final_top.png")
        bpy.ops.render.render(write_still=True)
        for o, old in _hidden:
            o.hide_render = old
        bpy.ops.wm.save_mainfile()
        print("PREVIEWS", [str(prev/n) for n in ("final_overview.png","final_showcase.png","final_top.png")])
    except Exception as exc:
        print("PREVIEW_WARN", repr(exc))
    if not a.no_render:
        out = bs_trajvideo.render(str(root/"scene.blend"), str(cal_path), str(plan_path),
                                  str(root/"previews"), frames=a.frames, res=a.res, fps=a.fps)
        print("TRAJ", json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
