"""Temporal anchor pass.

The same room can look different in the first and last frames because the camera has moved
and turned.  The recon evidence has per-object segment ids, while the top-down pass has
object positions.  This pass asks a vision model to choose which top-down objects are
actually visible in the first frame, and which one should define the first look-at.  It does
not invent coordinates: names are resolved back to the top-down hints and the scene graph.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm import LLM
from .recon import _find_json


PROMPT = """你是相机时序审核员。图1-3是视频最前面的 3 帧，下面是俯视布局（模型已估出的物体名称/x/y/朝向）和相机轨迹。
请只回答“首帧时相机实际看到了哪些物体，以及首帧应该看向哪里”，不要重估户型，不要写物体新坐标。

俯视布局：
{topdown}

轨迹（按时间）：
{trajectory}

已识别物体与出现段：
{evidence}

只输出 JSON：
{"first_frame_objects":[{"name":"俯视布局中的物体名称","why":"画面位置/大小证据"}],
 "look_at":"首帧最应该居中的物体名称",
 "turn_direction":"向左/右/前",
 "path_time":"首帧对应轨迹的起点",
 "notes":"依据图几"}
"""


def _frames(project: Path) -> list[Path]:
    d = project / "previews" / "recon_frames"
    fs = sorted(d.glob("s_*.png")) if d.exists() else []
    if len(fs) < 3:
        d = project / "reports" / "recon" / "keyframes"
        fs = sorted(d.glob("*.png")) if d.exists() else []
    return fs


def _trajectory(cal: dict) -> dict:
    return {"camera_path": cal.get("camera_path") or [],
            "look_mode": cal.get("look_mode"), "gravity": cal.get("gravity_up_source")}


def generate_temporal(project: Path, cfg, cal: dict, arch: dict, topdown: dict,
                      evidence: dict, log=print) -> dict:
    project = Path(project)
    out = project / "TEMPORAL_ANCHOR.json"
    if out.exists():
        try:
            d = json.loads(out.read_text(encoding="utf8"))
            if d.get("look_at") or d.get("first_frame_objects"):
                # cache only when the top-down layout is the same shell
                if d.get("footprint_m") == arch.get("footprint_m"):
                    return d
        except Exception:
            pass
    frames = _frames(project)
    if not frames:
        return {}
    ev_objects = []
    for o in evidence.get("objects") or []:
        ev_objects.append({"id": o.get("id"), "name": o.get("name"),
                           "kind": o.get("kind"), "segments": o.get("segments")})
    prompt = PROMPT.replace("{topdown}", json.dumps(
        {"footprint_m": topdown.get("footprint_m"),
         "object_hints": topdown.get("object_hints") or [],
         "barrier_runs": topdown.get("barrier_runs") or []}, ensure_ascii=False)[:12000])
    prompt = prompt.replace("{trajectory}", json.dumps(_trajectory(cal), ensure_ascii=False)[:6000])
    prompt = prompt.replace("{evidence}", json.dumps(ev_objects, ensure_ascii=False)[:8000])
    llm = LLM(cfg.recon_detail_model or cfg.model, cfg.api_base, cfg.api_key_env,
              temperature=0.15, max_output_tokens=2048, vision=True)
    log(f"[temporal] first-frame anchor pass: {llm.model}, {min(3, len(frames))} frames")
    text = llm.chat([
        {"role": "system", "content": "你是严格的相机时序审核员，只输出 JSON。"},
        {"role": "user", "content": prompt},
    ], [str(p) for p in frames[:3]], retries=1)
    (project / "reports" / "recon" / "temporal_raw.json").write_text(text, encoding="utf8")
    d = _find_json(text) or {}
    d["footprint_m"] = arch.get("footprint_m")
    d["source"] = "llm_temporal_first_frame_pass"
    out.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf8")
    log(f"[temporal] first_frame={[x.get('name') for x in d.get('first_frame_objects') or []]} "
        f"look_at={d.get('look_at')}")
    return d
