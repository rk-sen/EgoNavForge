"""LLM floorplan pass: video + gravity-aligned path + opening evidence -> ARCHITECTURE.json.

This is deliberately separate from ReconScanner: recognition answers "what/windows", while
this pass answers "what polygon / where are the wall openings".  It never accepts a fixed
footprint.  A bbox fallback is only used when the model output cannot be parsed; it is marked
`fallback_bbox` and is never presented as a model estimate.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .llm import LLM
from .recon import _find_json


PROMPT = """你是室内户型测量员。根据这 6 张连续视频关键帧、重力对齐相机轨迹和已认知的门窗/灯/地面，估出房间的 2D 户型。
不要默认矩形；如果画面是 L 形、拐角或有隔断，必须给出多边形。单位米，原点在房间最小 x/y 附近，逆时针。
参考尺度：门洞宽约0.95m，窗宽约1.4m，层高约3.2m，人约1.7m，桌面约0.75m，地砖约1.2m。只做估计，不要伪造毫米精度。
重要：房间可以比相机轨迹覆盖的范围更大。不要因为轨迹短、镜头只看到一角，就把整个房间缩小成轨迹的包络盒；外墙要延伸到画面证据所允许的完整尺度，宁可略大也不要塌缩。L 形要把两条完整边长都估出来。
必须只输出 JSON，不要解释：
{"footprint_m":[[x,y],...],"ceiling_z_m":3.2,"room_bbox_m":[xmin,xmax,ymin,ymax,ceiling],
 "windows":[{"side":"W|E|N|S","u_m":0.0,"width_m":1.4,"height_m":1.0,"z0_m":1.8,"type":"plain_glass|blind|curtain","mullions":{"cols":1,"rows":1},"curtain":{"draw":0.0,"side":"none"}}],
 "doors":[{"side":"W|E|N|S","u_m":0.0,"width_m":0.95,"height_m":2.05}],
 "notes":"依据哪些帧/轨迹判断"}

轨迹摘要：{trajectory}
门窗/地面证据：{evidence}
"""

QA_PROMPT = """你是户型审核员。图1-4是视频关键帧，图5是第一版户型估计画出的俯视示意图（灰色房间、黑色墙线、蓝窗、橙门、红线轨迹）。
请重点审核：
1) 房间轮廓到底是矩形、L 形还是带隔断；拐角/凹口是否与视频一致；
2) 长宽是否明显偏小/偏大；轨迹是否在户型内；门是否在相机最后离开的方向。若第一版把房间缩到轨迹包络、明显偏小，请扩大外墙，不要为了贴合轨迹而缩小房间；
3) 窗的数量/墙侧/宽度是否正确；是否有百叶/窗帘；灯是长条还是短条；
4) 如果第一版错了，给出修正后的 footprint_m / windows / doors / ceiling_z_m；如果第一版基本正确，原样返回并在 issues 写 ok。
只输出 JSON：
{"footprint_m":[[x,y],...],"ceiling_z_m":3.2,"windows":[...],"doors":[...],
 "issues":["..."],"confidence":0.0,"notes":"依据图几"}
"""


def _polygon_area(poly: list[list[float]]) -> float:
    s = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        s += float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
    return abs(s) / 2.0


def _topdown_sketch(arch, cal, out):
    """Draw first-pass footprint + openings + observed path for the QA model."""
    from PIL import Image, ImageDraw
    poly = [[float(p[0]), float(p[1])] for p in arch.get("footprint_m") or []]
    if len(poly) < 3:
        raise ValueError("no polygon for sketch")
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W = 900; H = 700; margin = 64
    scale = min((W - 2 * margin) / max(1e-6, x1 - x0), (H - 2 * margin) / max(1e-6, y1 - y0))
    ox = (W - (x1 - x0) * scale) / 2; oy = (H - (y1 - y0) * scale) / 2
    def P(x, y):
        return (ox + (x - x0) * scale, oy + (y1 - y) * scale)
    im = Image.new("RGB", (W, H), "#f7f4ec"); dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle((10, 10, W - 10, H - 10), outline="#222", width=3)
    dr.polygon([P(*p) for p in poly], fill="#d9d6cd")
    dr.line([P(*p) for p in poly + [poly[0]]], fill="#111", width=6)
    for w in arch.get("windows") or []:
        side = str(w.get("side") or "")[:1].upper(); u = float(w.get("u_m") or 0); ww = float(w.get("width_m") or 1.4)
        if side == "W": a, b = P(x0, y0 + u), P(x0, y0 + u + ww)
        elif side == "E": a, b = P(x1, y0 + u), P(x1, y0 + u + ww)
        elif side == "N": a, b = P(x0 + u, y1), P(x0 + u + ww, y1)
        else: a, b = P(x0 + u, y0), P(x0 + u + ww, y0)
        dr.line([a, b], fill="#1e78b4", width=10)
    for d in arch.get("doors") or []:
        side = str(d.get("side") or "")[:1].upper(); u = float(d.get("u_m") or 0); ww = float(d.get("width_m") or .95)
        if side == "W": a, b = P(x0, y0 + u), P(x0, y0 + u + ww)
        elif side == "E": a, b = P(x1, y0 + u), P(x1, y0 + u + ww)
        elif side == "N": a, b = P(x0 + u, y1), P(x0 + u + ww, y1)
        else: a, b = P(x0 + u, y0), P(x0 + u + ww, y0)
        dr.line([a, b], fill="#d47a22", width=11)
    try:
        from .blueprint import _room_path
        path = _room_path(cal, poly)
        pts = [P(float(q["x"]), float(q["y"])) for q in path]
        if len(pts) > 1:
            dr.line(pts, fill="#c62828", width=4)
            for p in pts:
                dr.ellipse((p[0]-4, p[1]-4, p[0]+4, p[1]+4), fill="#c62828")
    except Exception:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)
    return out


def _normalize_arch(data: dict[str, Any], cal: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    poly = []
    for p in data.get("footprint_m") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            poly.append([round(float(p[0]), 3), round(float(p[1]), 3)])
    if len(poly) < 3:
        raise ValueError("footprint_m needs >=3 points")
    # CCW for stable normals / wall side calculation.
    area2 = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        area2 += a[0] * b[1] - b[0] * a[1]
    if area2 < 0:
        poly = list(reversed(poly))
    if _polygon_area(poly) < 1.0:
        raise ValueError("footprint area too small")

    ceil = float(data.get("ceiling_z_m") or data.get("ceiling_z") or 3.2)
    if not (1.8 <= ceil <= 8.0):
        ceil = 3.2
    windows, doors = [], []
    for e in data.get("windows") or []:
        side = str(e.get("side") or "")[:1].upper()
        if side not in ("W", "E", "N", "S"):
            continue
        windows.append({
            "side": side,
            "u_m": round(float(e.get("u_m") or 0.0), 3),
            "width_m": round(float(e.get("width_m") or 1.4), 3),
            "height_m": round(float(e.get("height_m") or 1.0), 3),
            "z0_m": round(float(e.get("z0_m") or 1.8), 3),
            "type": str(e.get("type") or "plain_glass"),
            "mullions": e.get("mullions") or {"cols": 0, "rows": 0},
            "curtain": e.get("curtain") or {"draw": 0.0, "side": "none"},
        })
    for e in data.get("doors") or []:
        side = str(e.get("side") or "")[:1].upper()
        if side not in ("W", "E", "N", "S"):
            continue
        doors.append({
            "side": side,
            "u_m": round(float(e.get("u_m") or 0.0), 3),
            "width_m": round(float(e.get("width_m") or .95), 3),
            "height_m": round(float(e.get("height_m") or 2.05), 3),
            "z0_m": 0.0,
        })
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    # Evidence supplies floor / lights; model supplies geometry/openings.
    a = dict(ev.get("architecture") or {})
    arch = {
        "footprint_m": poly,
        "ceiling_z": round(ceil, 3),
        "ceiling_z_m": round(ceil, 3),
        "room_bbox_m": [min(xs), max(xs), min(ys), max(ys), round(ceil, 3)],
        "windows": windows,
        "doors": doors,
        "floor": a.get("floor") or {"style": "unknown"},
        "lights": a.get("lights") or {},
        "source": "llm_floorplan_pass: video frames + gravity-aligned path + opening evidence",
        "floorplan_notes": str(data.get("notes") or "")[:1200],
        "footprint_model": "llm",
    }
    return arch


def _fallback_arch(cal: dict[str, Any], ev: dict[str, Any]) -> dict[str, Any]:
    """Only when model output is unusable: use the trajectory bbox, explicitly marked."""
    rb = cal.get("room_bbox_m") or [0, 6, 0, 6, 3.0]
    x0, x1, y0, y1 = float(rb[0]), float(rb[1]), float(rb[2]), float(rb[3])
    ceil = float(rb[4]) if len(rb) > 4 else 3.0
    a = dict(ev.get("architecture") or {})
    return {
        "footprint_m": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "ceiling_z": ceil, "ceiling_z_m": ceil,
        "room_bbox_m": [x0, x1, y0, y1, ceil],
        "windows": [], "doors": [],
        "floor": a.get("floor") or {"style": "unknown"},
        "lights": a.get("lights") or {},
        "source": "fallback_bbox_from_trajectory; model output unusable",
        "footprint_model": "fallback",
    }


def _select_frames(project: Path) -> list[Path]:
    for d in (project / "previews" / "recon_frames", project / "reports" / "recon"):
        if d.exists():
            frames = sorted(d.glob("s_*.png"))
            if len(frames) >= 4:
                idx = [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, len(frames)-1]
                return [frames[min(i, len(frames)-1)] for i in idx]
    # keyframes fallback
    d = project / "reports" / "recon" / "keyframes"
    if d.exists():
        k = sorted(d.glob("*.png"))
        if k:
            return k[:6]
    # Evidence reuse/debug path: deterministic frame extraction, no API.
    video = project / "inputs" / "source_video.mp4"
    if not video.exists():
        vids = sorted((project / "inputs").glob("*.mp4"))
        video = vids[0] if vids else video
    if video.exists():
        import subprocess
        d = project / "previews" / "recon_frames"
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", "fps=1", "-q:v", "2", str(d / "s_%03d.png")], check=True)
        frames = sorted(d.glob("s_*.png"))
        if len(frames) >= 4:
            idx = [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, len(frames)-1]
            return [frames[min(i, len(frames)-1)] for i in idx]
    raise FileNotFoundError("no recon frames; run ReconScanner first")


def generate_architecture(project: Path, cfg, log=print, force: bool = False) -> dict[str, Any]:
    project = Path(project)
    out_json = project / "ARCHITECTURE.json"
    raw_path = project / "reports" / "recon" / "floorplan_raw.json"
    cal = json.loads((project / "calibration.json").read_text(encoding="utf8"))
    ev = json.loads((project / "reports" / "recon" / "RECON_EVIDENCE.json").read_text(encoding="utf8"))
    if out_json.exists() and not force:
        try:
            d = json.loads(out_json.read_text(encoding="utf8"))
            if d.get("footprint_model") == "llm":
                return d
        except Exception:
            pass
    frames = _select_frames(project)
    summary = {
        "trajectory_samples": [{k: p.get(k) for k in ("i", "pos", "yaw")}
                               for p in (cal.get("camera_path") or [])[:8]],
        "room_bbox_from_track_m": cal.get("room_bbox_m"),
        "camera_height_m": (ev.get("architecture") or {}).get("camera_height_m"),
    }
    evidence = {k: (ev.get("architecture") or {}).get(k)
                for k in ("windows", "doors", "floor", "lights", "window_verdict")}
    prompt = PROMPT.replace("{trajectory}", json.dumps(summary, ensure_ascii=False)).replace(
        "{evidence}", json.dumps(evidence, ensure_ascii=False))
    llm = LLM(cfg.recon_detail_model or cfg.model, cfg.api_base, cfg.api_key_env,
              temperature=0.2, max_output_tokens=4096, vision=True)
    log(f"[floorplan] asking {llm.model} with {len(frames)} frames")
    text = llm.chat([
        {"role": "system", "content": "你是严谨的室内户型测量员，只输出 JSON，不要套模板。"},
        {"role": "user", "content": prompt},
    ], [str(p) for p in frames], retries=1)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding="utf8")
    data = _find_json(text) or {}
    try:
        arch = _normalize_arch(data, cal, ev)
        log(f"[floorplan] model footprint {len(arch['footprint_m'])} pts, "
            f"{len(arch['windows'])} windows, {len(arch['doors'])} doors")
    except Exception as exc:
        log(f"[floorplan] model output invalid ({exc}); fallback bbox only")
        arch = _fallback_arch(cal, ev)

    # Second pass: show the first plan as a top-down diagram, plus 4 frames, and ask the model
    # to check the L-shape, dimensions, window/door walls and light type.  This is a real
    # extra API call, not a hand-written correction.
    try:
        sketch = _topdown_sketch(arch, cal, project / "reports" / "recon" / "topdown_sketch_first.png")
        qa_prompt = QA_PROMPT.replace("{trajectory}", json.dumps(summary, ensure_ascii=False)).replace(
            "{evidence}", json.dumps(evidence, ensure_ascii=False))
        qa_llm = LLM(cfg.recon_detail_model or cfg.model, cfg.api_base, cfg.api_key_env,
                     temperature=0.15, max_output_tokens=4096, vision=True)
        log(f"[floorplan] QA pass {qa_llm.model}: 4 frames + first-plan sketch")
        qa_text = qa_llm.chat([
            {"role": "system", "content": "你是严格的户型审核员；只输出 JSON，先指出矛盾再给修正。"},
            {"role": "user", "content": qa_prompt},
        ], [str(p) for p in frames[:4] + [sketch]], retries=1)
        (project / "reports" / "recon" / "floorplan_qa_raw.json").write_text(qa_text, encoding="utf8")
        qa_data = _find_json(qa_text) or {}
        if qa_data.get("footprint_m"):
            try:
                corrected = _normalize_arch(qa_data, cal, ev)
                # The QA image is a diagram; the model sometimes drops all openings because
                # it is looking at the drawing rather than the video.  Do not let the second
                # pass erase a complete opening set from the first pass.
                if not corrected.get("windows") and arch.get("windows"):
                    corrected["windows"] = arch["windows"]
                    corrected["windows_kept_from_first_pass"] = True
                if not corrected.get("doors") and arch.get("doors"):
                    corrected["doors"] = arch["doors"]
                    corrected["doors_kept_from_first_pass"] = True
                corrected["source"] = "two_pass_floorplan: first plan + QA sketch/4 frames"
                corrected["first_pass_footprint_m"] = arch.get("footprint_m")
                corrected["qa_issues"] = qa_data.get("issues") or []
                corrected["qa_confidence"] = qa_data.get("confidence")
                arch = corrected
                log(f"[floorplan] QA corrected footprint={len(arch['footprint_m'])} pts, "
                    f"{len(arch['windows'])} windows, {len(arch['doors'])} doors")
            except Exception as exc:
                log(f"[floorplan] QA output not usable ({exc}); keep first plan")
                arch["qa_issues"] = ["QA output invalid: " + str(exc)]
        else:
            arch["qa_issues"] = qa_data.get("issues") or []
            arch["source"] = "llm_floorplan_pass + QA(no footprint change)"
    except Exception as exc:
        log(f"[floorplan] QA pass failed: {exc!r}; keep first plan")
        arch["qa_issues"] = ["QA pass failed: " + repr(exc)]

    out_json.write_text(json.dumps(arch, ensure_ascii=False, indent=2), encoding="utf8")
    return arch
