#!/usr/bin/env python3
"""V22 one-command pipeline: video + trajectory + intrinsics + out -> editable Blender twin.

Only the three input files and the output directory are required.  Every intermediate artifact
(calibration, recon evidence, LLM floorplan, scene plan, specs, blueprint, blend, previews) is
generated under <out>/project and consumed by the next stage.

For a reproducible rerun without paying for recon again, `--evidence` can point at an existing
RECON_EVIDENCE.json; this is a debug option, not part of the normal user contract.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE / "indoor_twin_agent"


def _find_blender(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("BLENDER"):
        return os.environ["BLENDER"]
    p = shutil.which("blender")
    if p:
        return p
    for c in ("/home/user/blender/blender-4.2.0-linux-x64/blender", "/usr/local/bin/blender",
              "/Applications/Blender.app/Contents/MacOS/Blender"):
        if Path(c).is_file():
            return c
    raise FileNotFoundError("Blender not found; pass --blender")


def _video_meta(path: Path) -> dict:
    try:
        out = subprocess.check_output(["ffprobe", "-v", "quiet", "-print_format", "json",
                                       "-show_streams", "-show_format", str(path)], text=True)
        d = json.loads(out)
        st = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
        return {"width": st.get("width"), "height": st.get("height"),
                "fps": st.get("r_frame_rate"), "duration": d.get("format", {}).get("duration")}
    except Exception:
        return {}


def _prepare_inputs(video: Path, traj: Path, intr: Path, project: Path) -> None:
    inputs = project / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, inputs / "source_video.mp4")
    shutil.copy2(traj, inputs / "trajectory.json")
    shutil.copy2(intr, inputs / "intrinsics.json")


def _make_compare(project: Path, video: Path, fps: int) -> None:
    """Side-by-side original / same-trajectory twin video, sampled at `fps`."""
    try:
        from PIL import Image, ImageDraw
        prev = project / "previews"
        traj = sorted((prev / "traj_frames").glob("*.png"))
        if not traj:
            return
        orig = prev / "orig_frames"; orig.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", f"fps={fps},scale=384:288:force_original_aspect_ratio=decrease,"
                               "pad=384:288:(ow-iw)/2:(oh-ih)/2",
                        "-q:v", "2", str(orig / "o_%04d.png")], check=False)
        og = sorted(orig.glob("o_*.png"))
        if not og:
            return
        comp = prev / "compare_frames"; comp.mkdir(exist_ok=True)
        for i in range(min(len(traj), len(og))):
            a = Image.open(og[i]).convert("RGB").resize((384, 288))
            b = Image.open(traj[i]).convert("RGB").resize((384, 288))
            c = Image.new("RGB", (768, 288), (0, 0, 0)); c.paste(a, (0, 0)); c.paste(b, (384, 0))
            d = ImageDraw.Draw(c); d.text((8, 8), "original", fill=(255, 255, 255)); d.text((392, 8), "twin", fill=(255, 255, 255))
            c.save(comp / f"{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(comp / "%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(prev / "final_compare.mp4")], check=False)
        print(f"      compare: {prev/'final_compare.mp4'}", flush=True)
    except Exception as exc:
        print(f"COMPARE_WARN {exc!r}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="balanced", choices=["fast", "balanced", "high"])
    ap.add_argument("--model", default=os.environ.get("TWIN_MODEL", "kimi-k2.6"))
    ap.add_argument("--api-base", default=os.environ.get("TWIN_API_BASE", "https://ai.zj-computility.com/maas/v1"))
    ap.add_argument("--api-key-env", default="zhesuan_API_KEY")
    ap.add_argument("--blender", default=None)
    ap.add_argument("--evidence", default="", help="debug/repro: reuse an existing RECON_EVIDENCE.json")
    ap.add_argument("--force-architecture", action="store_true")
    ap.add_argument("--objects", default="", help="optional comma-separated scene objects the user knows are present")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--no-recon", action="store_true", help="stop after floorplan; no Blender")
    ap.add_argument("--no-usd", action="store_true", help="skip the Isaac/USD export stage")
    a = ap.parse_args(argv)
    t0 = time.time()
    video, traj, intr = Path(a.video).expanduser().resolve(), Path(a.trajectory).expanduser().resolve(), Path(a.intrinsics).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    project = out / "project"
    project.mkdir(parents=True, exist_ok=True)
    _prepare_inputs(video, traj, intr, project)

    sys.path.insert(0, str(HERE))
    from indoor_twin_agent.config import RunConfig
    from indoor_twin_agent import inputs_prep
    from indoor_twin_agent.recon import ReconScanner
    from indoor_twin_agent.floorplan_pass import generate_architecture
    from indoor_twin_agent.blueprint import build_blueprint
    from indoor_twin_agent.registration import register_path
    from indoor_twin_agent.topdown_pass import generate_topdown
    from indoor_twin_agent.temporal_pass import generate_temporal
    from indoor_twin_agent.auto_scene import auto_plan
    from indoor_twin_agent.render_qa_pass import run_render_qa, apply_camera_yaw

    inputs_prep.prepare(project, _video_meta(video))
    cal = json.loads((project / "calibration.json").read_text(encoding="utf8"))
    print(f"[1/6] calibration.json  room_bbox={cal.get('room_bbox_m')}", flush=True)

    ev_path = project / "reports" / "recon" / "RECON_EVIDENCE.json"
    if a.evidence:
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(a.evidence).expanduser().resolve(), ev_path)
    cfg = RunConfig(video=str(video), trajectory=str(traj), intrinsics=str(intr), out=str(out),
                    model=a.model, api_base=a.api_base, api_key_env=a.api_key_env,
                    recon_detail_model=a.model)
    if not ev_path.exists():
        print("[2/6] recon: multi-pass API", flush=True)
        ReconScanner(project, cfg, log=print).run()
    else:
        print("[2/6] recon: using supplied RECON_EVIDENCE", flush=True)

    print("[3/6] floorplan: LLM -> ARCHITECTURE.json", flush=True)
    arch = generate_architecture(project, cfg, log=print, force=a.force_architecture)
    print(f"      footprint={arch['footprint_m']} windows={len(arch['windows'])} doors={len(arch['doors'])}", flush=True)
    print("[3b/6] registration: trajectory -> model footprint (rigid + uniform scale)", flush=True)
    reg = register_path(cal, arch)
    arch["path_registration"] = reg
    (project / "PATH_REGISTRATION.json").write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"      theta={reg.get('theta_deg')} scale={reg.get('scale')} inside={reg.get('inside')} "
          f"min_wall={reg.get('min_wall_m')} view_dot={reg.get('view_dot')}", flush=True)

    print("[3c/6] topdown cognition: two API passes -> barrier runs + shelf layout", flush=True)
    topdown = generate_topdown(project, cfg, cal, arch, log=print)
    print(f"      topdown: {len(topdown.get('barrier_runs') or [])} runs, "
          f"{len(topdown.get('object_hints') or [])} object hints", flush=True)

    print("[3d/6] temporal anchor: first frame -> object look-at", flush=True)
    temporal = generate_temporal(project, cfg, cal, arch, topdown,
                                  json.loads(ev_path.read_text(encoding="utf8")), log=print)

    print("[4/6] scene graph: ARCHITECTURE -> SCENE_PLAN/specs/BLUEPRINT", flush=True)
    ev = json.loads(ev_path.read_text(encoding="utf8"))
    if a.objects:
        names = [x.strip() for x in a.objects.split(",") if x.strip()]
        ev["user_objects"] = names
        have = " ".join(str(o.get("name") or o.get("id") or "") for o in ev.get("objects") or []).lower()
        for n in names:
            if n.lower() not in have:
                ev.setdefault("objects", []).append({"id": f"U_{n}", "name": n, "count": 1,
                                                     "source": "user_declared"})
    bp0 = build_blueprint(cal, arch, {"objects": []})
    path = bp0.get("trajectory") or []
    plan, specs = auto_plan(ev, arch, trajectory=path, topdown=topdown)
    # The temporal pass can reverse the path order when the first-frame objects sit near
    # the end of the registered track.  Build the blueprint first, then copy its time-ordered
    # path back into SCENE_PLAN, which is the file bs_trajvideo actually renders.
    plan["topdown"] = topdown
    plan["temporal"] = temporal
    bp = build_blueprint(cal, arch, plan)
    plan["trajectory"] = bp.get("trajectory") or path
    (project / "SCENE_PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf8")
    (project / "specs_v22.json").write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf8")
    (project / "BLUEPRINT.json").write_text(json.dumps(bp, ensure_ascii=False, indent=2), encoding="utf8")
    (project / "ARCHITECTURE.json").write_text(json.dumps(arch, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"      {len(plan['objects'])} nodes, {len(specs)} specs, path={len(path)} samples", flush=True)

    if a.no_recon:
        print("DONE (no Blender)", flush=True)
        return 0

    print("[5/6] Blender: shell + assets + cameras", flush=True)
    if (HERE / "blender_lib").exists():
        shutil.copytree(HERE / "blender_lib", project / "_lib", dirs_exist_ok=True)
    if (HERE / "assets").exists():
        shutil.copytree(HERE / "assets", project / "assets", dirs_exist_ok=True)
    blender = _find_blender(a.blender)
    cmd = [blender, "-b", "--factory-startup", "--python", str(PKG / "blender_build.py"), "--",
           "--project", str(project), "--frames", str(a.frames), "--res", str(a.res), "--fps", str(a.fps)]
    subprocess.run(cmd, check=True)
    _make_compare(project, video, a.fps)

    # One bounded render-vs-video feedback pass.  The first build gives us a same-time
    # pair for every sampled trajectory frame; the vision model is asked only for camera
    # alignment and concrete mismatch notes.  A small global yaw correction is applied
    # once and re-rendered; large or ambiguous corrections are deliberately not applied.
    print("[6b/7] render QA: original frame vs same-time twin frame", flush=True)
    try:
        qa = run_render_qa(project, cfg, cal, plan, log=print)
        raw_delta = qa.get("global_yaw_delta_deg")
        delta = float(raw_delta) if raw_delta is not None else 0.0
        if 4.0 <= abs(delta) <= 25.0:
            apply_camera_yaw(project, delta)
            print(f"[6b/7] apply camera yaw {delta:+.1f} deg; re-rendering once", flush=True)
            subprocess.run(cmd, check=True)
            _make_compare(project, video, a.fps)
        else:
            print(f"[6b/7] no camera yaw applied (delta={delta}); QA notes retained", flush=True)
    except Exception as exc:
        print(f"[6b/7] render QA skipped/failed: {exc!r}", flush=True)

    if not a.no_usd:
        print("[7/7] USD export: full scene + unique item assets + collisions", flush=True)
        subprocess.run([blender, "-b", str(project / "scene.blend"), "--python",
                        str(PKG / "export_usd.py"), "--", "--project", str(project)], check=True)

    print(f"[6/6] done in {(time.time()-t0)/60:.1f} min: {project/'scene.blend'}", flush=True)
    print(f"      previews: {project/'previews'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
