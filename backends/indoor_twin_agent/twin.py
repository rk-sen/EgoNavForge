#!/usr/bin/env python3
"""User-facing entry: video + trajectory + intrinsics + output -> twin.

Examples
--------
export zhesuan_API_KEY=...
python twin.py --video /data/room.mp4 --trajectory /data/traj.json \
               --intrinsics /data/K.json --out /runs/room01

python twin.py ... --check        # validate paths/formats only, no API call
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _json_ok(path: Path) -> tuple[bool, str]:
    try:
        d = json.loads(path.read_text(encoding="utf8"))
    except Exception as exc:
        return False, f"not valid JSON: {exc}"
    if not isinstance(d, (dict, list)):
        return False, "top-level JSON must be object or array"
    return True, "ok"


def _find_blender(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("BLENDER")
    if env:
        return env
    p = shutil.which("blender")
    if p:
        return p
    for c in (
        "/home/user/blender/blender-4.2.0-linux-x64/blender",
        "/usr/local/bin/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ):
        if Path(c).is_file():
            return c
    return "blender"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="One command: walkthrough video + camera trajectory + intrinsics -> editable Blender twin")
    p.add_argument("--video", required=True, help="input walkthrough video (mp4/mov)")
    p.add_argument("--trajectory", required=True, help="camera trajectory JSON (c2w poses; pi3 format accepted)")
    p.add_argument("--intrinsics", required=True, help="camera intrinsics JSON (fx/fy/cx/cy/W/H or K)")
    p.add_argument("--out", required=True, help="output run directory; created if absent")
    p.add_argument("--preset", default="balanced", choices=["fast", "balanced", "high"])
    p.add_argument("--model", default=os.environ.get("TWIN_MODEL", "kimi-k2.6"))
    p.add_argument("--api-base", default=os.environ.get("TWIN_API_BASE", "https://ai.zj-computility.com/maas/v1"))
    p.add_argument("--api-key-env", default="zhesuan_API_KEY", help="environment variable holding the API key")
    p.add_argument("--blender", default=None, help="Blender executable; auto-detected if omitted")
    p.add_argument("--check", action="store_true", help="validate inputs only; do not call API or Blender")
    p.add_argument("--no-window-proxy", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--objects", default="", help="optional comma-separated objects known to be in the scene")
    p.add_argument("--frames", type=int, default=60, help="trajectory comparison frames")
    p.add_argument("--res", type=int, default=384)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--evidence", default="", help="debug: reuse an existing RECON_EVIDENCE.json")
    p.add_argument("--force-architecture", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    video, traj, intr = Path(a.video).expanduser().resolve(), Path(a.trajectory).expanduser().resolve(), Path(a.intrinsics).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()

    problems: list[str] = []
    if not video.is_file():
        problems.append(f"video not found: {video}")
    for label, path in (("trajectory", traj), ("intrinsics", intr)):
        if not path.is_file():
            problems.append(f"{label} not found: {path}")
        else:
            ok, why = _json_ok(path)
            if not ok:
                problems.append(f"{label}: {why}")
    if problems:
        print("INPUT ERROR")
        for s in problems:
            print(" -", s)
        return 2

    print("INPUT OK")
    print(f"  video      {video} ({video.stat().st_size/1e6:.2f} MB)")
    print(f"  trajectory {traj}")
    print(f"  intrinsics {intr}")
    print(f"  out        {out}")
    print(f"  preset     {a.preset}")
    if a.check:
        print("CHECK ONLY: no API / Blender call")
        return 0

    key = os.environ.get(a.api_key_env or "")
    if not key:
        print(f"API KEY ERROR: environment variable {a.api_key_env} is empty.")
        print(f"  export {a.api_key_env}='...'")
        return 2
    blender = _find_blender(a.blender)
    if not Path(blender).is_file() and shutil.which(blender) is None:
        print(f"BLENDER ERROR: not found: {blender}")
        print("  set --blender /absolute/path/to/blender or BLENDER env var")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "pipeline.py"),
           "--video", str(video), "--trajectory", str(traj), "--intrinsics", str(intr),
           "--out", str(out), "--preset", a.preset, "--model", a.model,
           "--api-base", a.api_base, "--api-key-env", a.api_key_env,
           "--blender", blender, "--frames", str(a.frames), "--res", str(a.res), "--fps", str(a.fps)]
    if a.objects:
        cmd += ["--objects", a.objects]
    if a.evidence:
        cmd += ["--evidence", a.evidence]
    if a.force_architecture:
        cmd.append("--force-architecture")
    print("RUN", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(HERE), env=os.environ.copy()).returncode
    if rc == 0:
        print("\nDONE")
        print(f"  run dir : {out}")
        print(f"  project : {out / 'project'}")
        print(f"  scene   : {out / 'project' / 'scene.blend'}")
        print(f"  previews: {out / 'project' / 'previews'}")
    else:
        print(f"\nRUN FAILED rc={rc}; see {out / 'state.json'} and ledger.md")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
