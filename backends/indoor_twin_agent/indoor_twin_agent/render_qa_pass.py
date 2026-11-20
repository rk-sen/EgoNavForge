"""Render-vs-video QA feedback.

After the first same-trajectory render, compare the twin frame with the corresponding
original video frame at the same trajectory time.  This is deliberately a *camera/visual*
check, not another free-form scene generator: it returns a global camera yaw correction and
a written list of remaining geometry/material mismatches.  The pipeline applies only the
small, bounded camera correction automatically; object/barrier corrections stay in the QA
report so they do not silently overwrite the cognition files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .llm import LLM
from .recon import _find_json

PROMPT = """这是一次同轨迹的渲染对比。每张图的左半是原视频对应帧，右半是孪生场景在同一轨迹时刻的渲染。
请比较相机取景和场景内容，不要臆造新物体，不要给房间坐标。

请判断：
1) 相机：孪生画面相对原视频是偏左、偏右、偏上、偏下？给出“需要把孪生相机向右转”为正的 yaw_delta_deg；pitch_delta_deg 同理，向上为正。
2) 空间：相机看到的蓝色屏障组、绿色球、橙色沙发、办公桌、窗户的位置和相对前后关系是否一致。
3) 几何：有没有明显的屏障整组错位、物体穿墙/悬空、门窗位置错误。
4) 材质：地板、蓝色布料、橙色沙发、金属架是否明显不像。

只输出 JSON：
{"samples":[{"index":0,"camera_match":0.0,"yaw_delta_deg":0.0,"pitch_delta_deg":0.0,"issues":["..."],"note":""}],
 "global_yaw_delta_deg":0.0,"global_pitch_delta_deg":0.0,
 "barrier_issues":["..."],"object_issues":["..."],"opening_issues":["..."],"material_issues":["..."],
 "score":0.0,"summary":"..."}

注意：yaw 正值表示孪生相机需要向右转；如果左右图像已经对齐，填 0，不要为了凑答案给大角度。
"""


def _pairs(project: Path, n: int = 4) -> list[Path]:
    comp = project / "previews" / "compare_frames"
    files = sorted(comp.glob("*.png"))
    if not files:
        raise FileNotFoundError("no compare_frames; build/render first")
    if len(files) <= n:
        return files
    idx = [int(round(i * (len(files) - 1) / (n - 1))) for i in range(n)]
    return [files[i] for i in idx]


def run_render_qa(project: Path, cfg, cal: dict, plan: dict, log=print) -> dict:
    project = Path(project)
    out = project / "RENDER_QA.json"
    pairs = _pairs(project, 4)
    # one compact image per pair is enough: the left/right labels are already burned in.
    llm = LLM(getattr(cfg, "recon_detail_model", None) or cfg.model, cfg.api_base,
              cfg.api_key_env, temperature=0.2, max_output_tokens=4096, vision=True)
    log(f"[render-qa] comparing {len(pairs)} same-time pairs: {llm.model}")
    text = llm.chat([
        {"role": "system", "content": "你是同轨迹渲染 QA，只输出 JSON，不要推测不可见物体。"},
        {"role": "user", "content": PROMPT},
    ], [str(p) for p in pairs], retries=2)
    (project / "reports").mkdir(exist_ok=True)
    (project / "reports" / "render_qa_raw.json").write_text(text, encoding="utf8")
    data = _find_json(text) or {}
    data["images_used"] = [p.name for p in pairs]
    data["source"] = "same-time original/twin pair review"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    log(f"[render-qa] score={data.get('score')} yaw={data.get('global_yaw_delta_deg')}")
    return data


def apply_camera_yaw(project: Path, delta_deg: float) -> None:
    """Rotate only the registered camera directions; never move object placements."""
    project = Path(project)
    delta = math.radians(float(delta_deg))
    for name in ("SCENE_PLAN.json", "BLUEPRINT.json"):
        p = project / name
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf8"))
        for q in data.get("trajectory") or []:
            d = q.get("dir") or []
            if len(d) < 2:
                continue
            # positive QA yaw = turn right = clockwise in XY = negative angle
            a = math.atan2(float(d[1]), float(d[0])) - delta
            q["dir"] = [round(math.cos(a), 6), round(math.sin(a), 6), 0.0]
        data.setdefault("render_qa", {})["applied_global_yaw_delta_deg"] = float(delta_deg)
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
