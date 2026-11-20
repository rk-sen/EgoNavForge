# V37 — render-vs-video QA feedback

V36 is kept unchanged at `v36/` and in `v36_indoor_twin_agent.zip`.

## What was missing in V36

V36 already rendered the same registered trajectory and built `final_compare.mp4`, but the
comparison was only a side-by-side video for a human.  There was no vision pass that read
the twin frame, compared it with the corresponding original frame, and produced a correction.

## Added in V37

`indoor_twin_agent/render_qa_pass.py`

- After the first build + same-time compare, selects four same-time pairs
  (`compare_frames/0000`, `0016`, `0033`, `0049` here).
- Sends them to the vision model as a **same-time original | twin** pair.
- The model returns:
  - per-sample camera match and yaw/pitch correction,
  - barrier issues, object issues, opening issues, material issues,
  - a global yaw correction and score.
- Writes `RENDER_QA.json` and keeps the raw model response.
- A bounded global yaw correction (4° <= |delta| <= 25°) is applied to the registered
  trajectory directions in `SCENE_PLAN.json` and `BLUEPRINT.json`; the scene is re-rendered
  once.  Larger/ambiguous corrections are logged but not auto-applied.
- Object/barrier/material findings are retained in the report and are not silently written
  into the cognition files; those must be handled by the next cognition/layout iteration.

## Real V37 run

- V37 project: `v37/project/`
- First build + QA + one re-render: 17.4 min.
- `RENDER_QA.json`: score 0.25; global yaw -5.0°; applied and re-rendered.
- QA findings are explicit: first frames show heavy camera/framing mismatch; the twin is
  simplified, with barrier position/orientation and several foreground objects not matching.
- This validates the feedback mechanism but also confirms that a single global yaw correction
  is not enough for the first segment.  V38 should apply **per-sample** camera corrections
  and require an image-similarity improvement check before accepting them.

## Entry point

```bash
python3 twin.py --video V.mp4 --trajectory T.json --intrinsics K.json --out OUT
```

No process file was manually edited.  `ARCHITECTURE.json`, `SCENE_PLAN.json`,
`BLUEPRINT.json`, `RENDER_QA.json` are all generated.

## USD / Isaac export added after the V37 QA result

`indoor_twin_agent/export_usd.py` is now part of the normal pipeline (after render QA):

- `usd/full_scene.usd` — whole scene, including walls / windows / door / ceiling /
  lights and every object instance; architecture is not split into item files.
- `usd/objects/*.usd` — one asset per **unique item type**. Repeated barriers,
  monitors, shelves and boxes are exported once each:
  - 8 barrier instances → `O1_BlueBarrier.usd`
  - 2 monitor instances → `O6_Monitor.usd`
  - 2 shelf instances → `O7_MetalShelf.usd`
  - 2 inferred boxes → `O_Inferred_box.usd`
- `usd/manifest.json` — asset list, source roots, instance names, units and collision mode.
- USD is Z-up, metersPerUnit = 1.0.
- Collision is authored with `UsdPhysics.CollisionAPI` + `UsdPhysics.MeshCollisionAPI`:
  - architecture meshes: triangle mesh (`approximation=none`)
  - item meshes: convex hull
- Verified on the current V37 scene: full scene has 152 mesh prims and 152 collision
  APIs; 15 unique item assets; 3 light prims remain in the full scene.

This export is independent of the Kimi cognition API; it runs entirely in Blender +
`pxr`.
