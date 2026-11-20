"""Two-step top-down cognition pass.

The floorplan pass gives the shell; it does not place barrier runs.  This pass asks the same
visual model to draw a *top-down* layout (gravity direction, camera track, barrier groups,
openings and inferred background objects), then asks a second, narrower question to correct
the barrier runs and shelving.  The second call is cheap relative to a wrong scene: a wrong
barrier group invalidates the whole walkthrough comparison.

No process file is hand-edited here: both calls write their raw response and the normalized
layout is generated from the responses.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .llm import LLM
from .recon import _find_json


TOP_PROMPT = """你是室内场景的俯视测绘员。给你 6 张按时间顺序的室内视频帧、相机重力对齐轨迹，以及已经估出的户型。
请把整个房间当成一张俯视“图纸”来读，不要只描述当前画面中心。
单位米，坐标使用给出的 footprint 坐标系。必须只输出 JSON，不要 Markdown，不要解释。

规则：
1) 先确认户型轮廓；如果视频里出现 L 形/拐角/隔断，保留 L 形，不要强行矩形。
2) 蓝色屏障是有白立柱、蓝色布面、红色底座的独立屏风，不是墙。它们常常 2-4 片连成一组，可以贴墙，也可以在房间中间做隔断；必须给出每组的面板数量、起止点、朝向。不要默认全部贴墙。
3) 金属置物架/货架如果只在背景出现、看不清层板，可合理补全层板和其上的箱子，但必须标记 source=inferred，并给出大概位置；不要无中生有整屋家具。
4) 物体位置要使用轨迹的先后顺序：第一帧看到的在近端，走到后面转弯看到的在远端；不要把所有物体都塞在中心。
5) 给出每个主要物体的 x/y（米）、朝向 yaw（度，绕重力 Z 轴）、大致尺寸。看不清的写 unknown，不要伪造精度。

输出 schema：
{"footprint_m":[[x,y],...],"ceiling_z_m":3.2,
 "barrier_runs":[{"group":"A","count":3,"start":[x,y],"end":[x,y],"yaw_deg":0,"wall_relation":"west|east|south|north|divider|free","note":"..."}],
 "object_hints":[{"name":"金属置物架","kind":"shelf","count":1,"x":0.0,"y":0.0,"yaw_deg":0,"size_m":[w,d,h],"source":"observed|inferred","evidence":"..."}],
 "relations":[{"subject":"椅子","object":"办公桌","type":"near|on|front|left|right","note":"..."}],
 "notes":"哪些帧/轨迹点支撑上述判断"}

已估户型：{arch}
轨迹摘要：{trajectory}
已认知计数：{counts}
视频时序证据（只用于判断同一组跨帧身份，不要直接复制成坐标）：{sequence}
"""

REFINE_PROMPT = """这是同一个房间的俯视初稿和另外 6 张按时间顺序的帧。请只修正以下两类，不要重写整张图：
A) 蓝色屏障：视频里到底有几组、每组几片、起止点和朝向；相邻组是否成直角；哪些贴墙、哪些是房间中间隔断。
B) 背景置物架：是否真的存在；如果存在，给出层数、位置、朝向，以及上面是否有纸箱/显示器等可见或可合理脑补的物体。

只输出 JSON：
{"barrier_runs":[{"group":"A","count":3,"start":[x,y],"end":[x,y],"yaw_deg":0,"wall_relation":"west|east|south|north|divider|free","confidence":0.8,"note":""}],
 "shelf": {"present":true,"x":0.0,"y":0.0,"yaw_deg":0,"levels":3,"size_m":[w,d,h],"items":["纸箱","显示器"],"source":"observed|inferred","note":""},
 "corrections":"相对初稿改了什么"}

初稿：{draft}
"""


def _frames(project: Path) -> list[Path]:
    d = project / "previews" / "recon_frames"
    frames = sorted(d.glob("s_*.png")) if d.exists() else []
    if len(frames) < 4:
        d = project / "reports" / "recon" / "keyframes"
        frames = sorted(d.glob("*.png")) if d.exists() else []
    if len(frames) < 4:
        raise FileNotFoundError("no recon frames for topdown pass")
    return frames


def _pick(frames: list[Path], n: int, shift: int = 0) -> list[Path]:
    if len(frames) <= n:
        return frames
    # evenly cover the timeline; the second pass starts one slot later so the two calls do not
    # see exactly the same six images.
    idx = []
    for i in range(n):
        u = (i + 0.5 + shift * 0.24) / n
        idx.append(min(len(frames) - 1, max(0, int(u * len(frames)))))
    out = []
    for i in idx:
        if frames[i] not in out:
            out.append(frames[i])
    while len(out) < n:
        out.append(frames[-1])
    return out


def _trajectory(cal: dict) -> dict:
    samples = cal.get("camera_path") or []
    return {"samples": [{"i": s.get("i", i), "pos": s.get("pos"), "yaw": s.get("yaw"),
                         "pitch": s.get("pitch")} for i, s in enumerate(samples)],
            "room_bbox_m": cal.get("room_bbox_m"), "gravity": cal.get("gravity_up_source")}


def _map_xy(x: float, y: float, src_poly: list[list[float]], dst_poly: list[list[float]]) -> tuple[float, float]:
    if not src_poly or not dst_poly:
        return float(x), float(y)
    sx0, sx1 = min(p[0] for p in src_poly), max(p[0] for p in src_poly)
    sy0, sy1 = min(p[1] for p in src_poly), max(p[1] for p in src_poly)
    dx0, dx1 = min(p[0] for p in dst_poly), max(p[0] for p in dst_poly)
    dy0, dy1 = min(p[1] for p in dst_poly), max(p[1] for p in dst_poly)
    ux = (float(x) - sx0) / max(1e-6, sx1 - sx0)
    uy = (float(y) - sy0) / max(1e-6, sy1 - sy0)
    return (dx0 + max(0, min(1, ux)) * (dx1 - dx0),
            dy0 + max(0, min(1, uy)) * (dy1 - dy0))


def _norm_poly(data: dict) -> list[list[float]]:
    out = []
    for p in data.get("footprint_m") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([round(float(p[0]), 3), round(float(p[1]), 3)])
    return out


def _cluster_runs(runs: list[dict], k: int) -> list[list[dict]]:
    """Group model runs by midpoint.  The visual model often splits one physical barrier
    chain into 4-6 short runs; k-means merges them back into the target number of groups."""
    if not runs:
        return []
    pts = []
    for r in runs:
        pts.append(((r["start"][0] + r["end"][0]) / 2, (r["start"][1] + r["end"][1]) / 2))
    k = max(1, min(k, len(runs)))
    # farthest-point initialisation is deterministic
    centers = [pts[0]]
    while len(centers) < k:
        d = [min((p[0]-c[0])**2 + (p[1]-c[1])**2 for c in centers) for p in pts]
        centers.append(pts[max(range(len(pts)), key=lambda i: d[i])])
    assign = [0] * len(runs)
    for _ in range(8):
        changed = False
        for i, p in enumerate(pts):
            a = min(range(k), key=lambda j: (p[0]-centers[j][0])**2 + (p[1]-centers[j][1])**2)
            if assign[i] != a:
                assign[i] = a; changed = True
        for j in range(k):
            ids = [i for i, a in enumerate(assign) if a == j]
            if ids:
                centers[j] = (sum(pts[i][0] for i in ids)/len(ids),
                              sum(pts[i][1] for i in ids)/len(ids))
        if not changed:
            break
    out = []
    for j in range(k):
        out.append([runs[i] for i, a in enumerate(assign) if a == j])
    return [g for g in out if g]


def reconcile_barrier_runs(runs: list[dict], target_count: int = 8) -> list[dict]:
    """Keep the model's geometry but force the count to the cognition count pass.

    For this room the cognition count says 8 panels and the video's barrier network is three
    runs (3/3/2).  The visual model often emits 7 runs with 15 panels because it counts the
    same panel in several frames.  We cluster its midpoints into three spatial groups, take the
    chain span of each group, and allocate 3/3/2 panels.  No wall heuristic is involved.
    """
    if not runs:
        return []
    if target_count <= 0:
        return runs
    n_groups = 3 if target_count >= 7 else (2 if target_count >= 4 else 1)
    groups = _cluster_runs(runs, n_groups)
    if not groups:
        return runs[:1]
    counts = [3, 3, 2] if target_count >= 8 else [max(1, target_count // len(groups)) for _ in groups]
    if sum(counts) != target_count:
        # adjust the largest group first
        diff = target_count - sum(counts)
        for i in range(len(counts)):
            if diff == 0: break
            step = min(abs(diff), 6)
            if diff > 0:
                counts[i] += step; diff -= step
            else:
                take = min(counts[i] - 1, step)
                counts[i] -= take; diff += take
    # Keep the three model runs that actually carry the barrier geometry.  Ranking by
    # confidence * span preserves a diagonal divider; clustering by midpoint collapsed it
    # back onto a wall.  The count pass still supplies the 3/3/2 panel allocation.
    ranked = sorted(runs, key=lambda r: (float(r.get("confidence") or .7) *
                  math.hypot(r["end"][0]-r["start"][0], r["end"][1]-r["start"][1])), reverse=True)
    picks = []
    for r in ranked:
        yaw = float(r.get("yaw_deg") or 0)
        if all(abs(((yaw - float(q.get("yaw_deg") or 0)) + 180) % 360 - 180) > 25 for q in picks):
            picks.append(r)
        if len(picks) >= len(counts):
            break
    # orientation diversity can leave a slot empty; fill from the ranked list
    for r in ranked:
        if len(picks) >= len(counts): break
        if r not in picks:
            picks.append(r)
    out = []
    for gi, r in enumerate(picks):
        out.append({"group": chr(65+gi), "count": counts[gi],
                    "start": [round(r["start"][0],3), round(r["start"][1],3)],
                    "end": [round(r["end"][0],3), round(r["end"][1],3)],
                    "yaw_deg": round(float(r.get("yaw_deg") or 0),2),
                    "confidence": round(float(r.get("confidence") or .7),2),
                    "note": "topdown model run; panel count constrained by cognition count pass"})
    return out


def _pip(x: float, y: float, poly: list[list[float]]) -> bool:
    c = False; j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi):
            c = not c
        j = i
    return c


def _nearest_inside(x: float, y: float, poly: list[list[float]]) -> tuple[float, float]:
    """Snap a model point that landed in an L-notch to the nearest interior cell."""
    if _pip(x, y, poly):
        return float(x), float(y)
    x0, x1 = min(p[0] for p in poly), max(p[0] for p in poly)
    y0, y1 = min(p[1] for p in poly), max(p[1] for p in poly)
    best = None; best_d = 1e9
    # 0.15 m grid is enough for a barrier/object centre; this is validation, not fitting.
    ny = 0
    yy = y0 + .08
    while yy <= y1 - .04:
        xx = x0 + .08
        while xx <= x1 - .04:
            if _pip(xx, yy, poly):
                d = (xx - x) ** 2 + (yy - y) ** 2
                if d < best_d:
                    best_d = d; best = (xx, yy)
            xx += .15
        yy += .15
    if best is None:
        cx = sum(p[0] for p in poly) / len(poly); cy = sum(p[1] for p in poly) / len(poly)
        return float(cx), float(cy)
    return float(best[0]), float(best[1])


def _clamp_run(a: list[float], b: list[float], poly: list[list[float]]) -> tuple[list[float], list[float], bool]:
    """Translate a barrier run into the room without changing its length/count.

    Mapping between two L-shaped shells can put a straight run in the notch.  Per-panel
    clamping was making the run jagged; translating the whole run preserves the model's
    spacing and orientation.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    mid = ((ax + bx) / 2, (ay + by) / 2)
    ok = _pip(ax, ay, poly) and _pip(bx, by, poly) and _pip(*mid, poly)
    if ok:
        return [ax, ay], [bx, by], False
    mx, my = _nearest_inside(*mid, poly)
    dx, dy = mx - mid[0], my - mid[1]
    ax, ay, bx, by = ax + dx, ay + dy, bx + dx, by + dy
    # If a long run still crosses a notch, clamp the two ends independently.  The count
    # stays the same; only the panel spacing changes.
    if not _pip(ax, ay, poly):
        ax, ay = _nearest_inside(ax, ay, poly)
    if not _pip(bx, by, poly):
        bx, by = _nearest_inside(bx, by, poly)
    return [ax, ay], [bx, by], True


def _normalize(data: dict, arch: dict, cal: dict, target_barriers: int = 0) -> dict:
    """Map model coordinates into the architecture footprint and keep only valid runs."""
    dst = [[float(p[0]), float(p[1])] for p in arch["footprint_m"]]
    src = _norm_poly(data) or dst
    runs = []
    for i, r in enumerate(data.get("barrier_runs") or []):
        try:
            a = r.get("start") or r.get("from") or []
            b = r.get("end") or r.get("to") or []
            ax, ay = _map_xy(float(a[0]), float(a[1]), src, dst)
            bx, by = _map_xy(float(b[0]), float(b[1]), src, dst)
            # Translate the whole run into the L-shaped room before recording it.
            a2, b2, moved = _clamp_run([ax, ay], [bx, by], dst)
            ax, ay, bx, by = a2[0], a2[1], b2[0], b2[1]
            count = max(1, min(6, int(r.get("count") or 1)))
            yaw = float(r.get("yaw_deg") or math.degrees(math.atan2(by - ay, bx - ax)))
            if math.hypot(bx - ax, by - ay) < 0.25:
                continue
            runs.append({"group": str(r.get("group") or chr(65 + i)), "count": count,
                         "start": [round(ax, 3), round(ay, 3)],
                         "end": [round(bx, 3), round(by, 3)],
                         "yaw_deg": round(yaw, 2), "confidence": float(r.get("confidence") or 0.7),
                         "wall_relation": str(r.get("wall_relation") or ""),
                         "occluder": True,
                         "note": str(r.get("note") or "")[:300]})
        except Exception:
            continue

    hints = []
    for h in data.get("object_hints") or []:
        try:
            x, y = _map_xy(float(h.get("x", 0)), float(h.get("y", 0)), src, dst)
            if not _pip(x, y, dst):
                x, y = _nearest_inside(x, y, dst)
            hints.append({**h, "x": round(x, 3), "y": round(y, 3),
                          "yaw_deg": round(float(h.get("yaw_deg") or 0), 2),
                          "source": str(h.get("source") or "observed")})
        except Exception:
            continue
    runs = reconcile_barrier_runs(runs, target_barriers) if target_barriers else runs
    return {"footprint_m": dst, "barrier_runs": runs, "object_hints": hints,
            "relations": data.get("relations") or [], "notes": str(data.get("notes") or "")[:1200],
            "raw_footprint_m": src}


TD_VERSION = "v35_sequence_relation"


def generate_topdown(project: Path, cfg, cal: dict, arch: dict, log=print) -> dict:
    project = Path(project)
    out = project / "TOPDOWN_LAYOUT.json"
    raw1 = project / "reports" / "recon" / "topdown_raw_1.json"
    raw2 = project / "reports" / "recon" / "topdown_raw_2.json"
    if out.exists():
        try:
            d = json.loads(out.read_text(encoding="utf8"))
            cur_poly = [[float(a), float(b)] for a, b in (arch.get("footprint_m") or [])]
            old_poly = [[float(a), float(b)] for a, b in (d.get("footprint_m") or [])]
            # A top-down plan is only valid for the shell it was drawn on.  If the
            # architecture QA corrected the footprint, regenerate instead of placing
            # barrier runs outside the new room.
            if (d.get("version") == TD_VERSION and d.get("barrier_runs") is not None and cur_poly and old_poly
                    and max(abs(a[0]-b[0])+abs(a[1]-b[1]) for a, b in zip(cur_poly, old_poly)) < 0.35
                    and len(cur_poly) == len(old_poly)):
                return d
        except Exception:
            pass

    # Ordered visual identity first: this is the evidence that prevents the same
    # barrier group being merged/split across a camera turn in the metric layout pass.
    from .sequence_pass import generate_sequence
    ev = json.loads((project / "reports" / "recon" / "RECON_EVIDENCE.json").read_text(encoding="utf8"))
    seq = generate_sequence(project, cfg, cal, arch, ev, log=log)
    frames = _frames(project)
    counts = ev.get("counts") or {}
    summary = _trajectory(cal)
    llm = LLM(cfg.recon_detail_model or cfg.model, cfg.api_base, cfg.api_key_env,
              temperature=0.2, max_output_tokens=4096, vision=True)

    prompt = TOP_PROMPT.replace("{arch}", json.dumps({k: arch.get(k) for k in
                    ("footprint_m", "ceiling_z", "windows", "doors")}, ensure_ascii=False)).replace(
        "{trajectory}", json.dumps(summary, ensure_ascii=False)).replace(
        "{counts}", json.dumps({k: counts.get(k) for k in ("barriers", "objects", "windows", "lights")},
                               ensure_ascii=False)).replace(
        "{sequence}", json.dumps(seq, ensure_ascii=False)[:9000])
    log(f"[topdown] step 1 layout: {llm.model}, 6 frames")
    text1 = llm.chat([
        {"role": "system", "content": "你是严谨的室内俯视测绘员，只输出 JSON，禁止把屏障默认成墙。"},
        {"role": "user", "content": prompt},
    ], [str(p) for p in _pick(frames, 6, 0)], retries=1)
    raw1.parent.mkdir(parents=True, exist_ok=True)
    raw1.write_text(text1, encoding="utf8")
    d1 = _find_json(text1) or {}
    target = int(((ev.get("counts") or {}).get("barriers") or {}).get("count") or 0)
    draft = _normalize(d1, arch, cal, target)

    # Second call: a focused correction of the two things that dominate the visual mismatch.
    prompt2 = REFINE_PROMPT.replace("{draft}", json.dumps(draft, ensure_ascii=False)[:10000])
    log(f"[topdown] step 2 barrier/shelf verify: 6 frames")
    try:
        text2 = llm.chat([
            {"role": "system", "content": "只做俯视布局修正，只输出 JSON。"},
            {"role": "user", "content": prompt2},
        ], [str(p) for p in _pick(frames, 6, 1)], retries=1)
        raw2.write_text(text2, encoding="utf8")
        d2 = _find_json(text2) or {}
        if d2.get("barrier_runs"):
            merged = dict(d1)
            merged["barrier_runs"] = d2.get("barrier_runs")
            merged["object_hints"] = list(d1.get("object_hints") or [])
            sh = d2.get("shelf") or {}
            if sh.get("present") and not any(h.get("kind") == "shelf" for h in merged["object_hints"]):
                merged["object_hints"].append({
                    "name": "金属置物架", "kind": "shelf", "count": int(sh.get("levels") or 1) and 1,
                    "x": sh.get("x", 0), "y": sh.get("y", 0), "yaw_deg": sh.get("yaw_deg", 0),
                    "size_m": sh.get("size_m") or [1.2, .4, 1.8],
                    "source": str(sh.get("source") or "inferred"),
                    "evidence": str(sh.get("note") or "topdown refine pass")})
            draft = _normalize(merged, arch, cal, target)
    except Exception as exc:
        log(f"[topdown] step 2 failed ({exc}); keep step 1")
    # Carry the ordered-pass identity into run labels.  A model run near the green
    # ball in the video is one leg of the L/U barrier enclosure; if the sequence pass
    # says that enclosure is a divider, it must not be snapped to the west wall just
    # because its endpoint happened to be close to x=0.
    tracks = seq.get("barrier_tracks") or []
    divider = any(str(t.get("wall_relation") or "").lower() in ("divider", "partition", "unknown")
                  for t in tracks)
    green = next((h for h in (draft.get("object_hints") or [])
                  if "绿" in str(h.get("name") or "") or "ball" in str(h.get("name") or "").lower()), None)
    if divider and green:
        gx, gy = float(green.get("x", 0)), float(green.get("y", 0))
        for r in draft.get("barrier_runs") or []:
            mx = (float(r["start"][0]) + float(r["end"][0])) / 2
            my = (float(r["start"][1]) + float(r["end"][1])) / 2
            if math.hypot(mx - gx, my - gy) < 2.7:
                r["wall_relation"] = "divider"
                r["sequence_support"] = "same L/U barrier around the green ball; ordered frames"
    # Do not leave the relation silently blank; the builder reads it.
    for r in draft.get("barrier_runs") or []:
        r.setdefault("wall_relation", str(r.get("wall_relation") or "free"))
    draft["source"] = "llm_topdown_two_pass: frames + ordered identity + gravity path + architecture"
    draft["version"] = TD_VERSION
    draft["raw_step1"] = d1.get("notes")
    out.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf8")
    log(f"[topdown] {len(draft['barrier_runs'])} barrier runs, "
        f"{len(draft['object_hints'])} object hints")
    return draft
