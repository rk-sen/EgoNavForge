#!/usr/bin/env python3
"""Entry point.

  python run.py --video walkthrough.mp4 --trajectory pi3_trajectory.json \
                --intrinsics K_pinhole_pipeline.json --out run01 --preset balanced

  python run.py --selfcheck          # no API key needed: checks tools, blender lib, protocol

Everything is local except the LLM call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from indoor_twin_agent.agent import Agent  # noqa: E402
from indoor_twin_agent.config import Quality, RunConfig  # noqa: E402
from indoor_twin_agent.executor import Executor  # noqa: E402
from indoor_twin_agent.protocol import ProtocolError, parse  # noqa: E402
from indoor_twin_agent.tools import compare_metrics, image_stats  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="video -> Blender indoor digital twin (agent workflow)")
    p.add_argument("--video", help="walkthrough video")
    p.add_argument("--trajectory", help="camera trajectory json (pi3 or any c2w poses)")
    p.add_argument("--intrinsics", help="camera intrinsics json (fx/fy/cx/cy/W/H or K)")
    p.add_argument("--out", default="run", help="output run directory")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "high"])
    p.add_argument("--mesh-detail", choices=["blockout", "balanced", "high", "poly"])
    p.add_argument("--texture-mode", choices=["none", "nodes", "images", "both"])
    p.add_argument("--pbr-level", choices=["flat", "standard", "full"])
    p.add_argument("--no-window-proxy", action="store_true")
    p.add_argument("--target-previews", type=int)
    p.add_argument("--model", default=os.environ.get("TWIN_MODEL", "kimi-k2.6"))
    p.add_argument("--api-base", default=os.environ.get("TWIN_API_BASE", "https://ai.zj-computility.com/maas/v1"))
    p.add_argument("--api-key-env", default="zhesuan_API_KEY")
    p.add_argument("--no-vision", action="store_true", help="numbers only, no images to the model")
    p.add_argument("--max-steps", type=int, default=70)
    p.add_argument("--max-minutes", type=float, default=240)
    p.add_argument("--blender", default=os.environ.get("BLENDER", "blender"))
    p.add_argument("--selfcheck", action="store_true")
    return p


def config_from_args(a) -> RunConfig:
    q = Quality().as_preset(a.preset)
    if a.mesh_detail:
        q.mesh_detail = a.mesh_detail
    if a.texture_mode:
        q.texture_mode = a.texture_mode
    if a.pbr_level:
        q.pbr_level = a.pbr_level
    q.window_proxy = not a.no_window_proxy
    if a.target_previews:
        q.target_previews = a.target_previews
    return RunConfig(video=a.video or "", trajectory=a.trajectory or "", intrinsics=a.intrinsics or "",
                     out=a.out, model=a.model, api_base=a.api_base, api_key_env=a.api_key_env,
                     vision=not a.no_vision, max_steps=a.max_steps, max_minutes=a.max_minutes,
                     blender=a.blender, quality=q)


def selfcheck() -> int:
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="twin_selfcheck_"))
    ex = Executor(tmp)
    try:
        r = ex.run(["python3", "-c", "print('exec ok')"])
        print("[executor]", "ok" if r["ok"] else "FAIL", r["stdout"].strip()[:60])
        ok &= r["ok"]
    except Exception as exc:
        print("[executor] FAIL", exc)
        ok = False
    try:
        ex.run(["python3", "-c", "print(1)"], cwd="../")
    except Exception:
        print("[jail] ok (escape refused)")
    try:
        parse('<tool_call>{"tool":"write_file","args":{"path":"a.py","content":"x"}}</tool_call>', {"write_file"})
        print("[protocol] ok")
    except ProtocolError as exc:
        print("[protocol] FAIL", exc)
        ok = False
    files = sorted(Path(__file__).parent.joinpath("blender_lib").glob("*.py"))
    print("[blender_lib]", ", ".join(f.name for f in files))
    if files:
        import py_compile
        for f in files:
            try:
                py_compile.compile(str(f), doraise=True)
            except Exception as exc:
                print("[blender_lib] FAIL", f.name, exc)
                ok = False
    from PIL import Image
    import numpy as np
    p = tmp / "flat.png"
    Image.fromarray(np.full((64, 64, 3), 200, np.uint8)).save(p)
    s = image_stats(p)
    print("[qa] flat image detected:", s["flat"], "std", s["std"])
    ok &= bool(s["flat"])
    print("\nselfcheck:", "PASS" if ok else "FAIL", "| run dir", tmp)
    return 0 if ok else 1


def main() -> int:
    a = _parser().parse_args()
    if a.selfcheck:
        return selfcheck()
    if not (a.video and a.trajectory and a.intrinsics):
        print("need --video --trajectory --intrinsics (or --selfcheck)")
        return 2
    for f in (a.video, a.trajectory, a.intrinsics):
        if not Path(f).exists():
            print("missing input:", f)
            return 2
    cfg = config_from_args(a)
    root = Path(cfg.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg.save(root / "run_config.json")
    agent = Agent(cfg, root, HERE / "blender_lib", log=lambda s: print(s, flush=True))
    state = agent.run()
    print("\n== run finished ==")
    print(json.dumps({k: state.get(k) for k in ("stage", "steps", "done", "elapsed_min", "llm_calls", "tokens")},
                     ensure_ascii=False, indent=1))
    print("artefacts:", root / "project")
    return 0 if state.get("done") else 1


if __name__ == "__main__":
    sys.exit(main())
