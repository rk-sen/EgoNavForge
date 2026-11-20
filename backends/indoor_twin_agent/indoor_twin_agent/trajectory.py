"""Consecutive-frame trajectory cognition and Blender Z-up axis repair.

The PI3 camera_path is a position track in a gravity-aligned frame. Its yaw does NOT survive
the Z-up room mapping; using it made the first frame show barriers on the left when the video
shows them on the right. This module asks the vision model about *relations between two
consecutive frames* (forward/left/right/turn) and then makes the Blender walkthrough use the
room-frame path tangent as the camera forward.

It is deliberately small: one API call per frame pair, not a 16k-character mega prompt.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

TRAJ_PROMPT = """这2帧是同一段室内走拍视频中相隔约1秒的连续帧，按时间先后。
只判断相机运动和同一组蓝色屏障的连续性，不要重新列物品清单。
1) 相机相对上一帧怎么动：forward/backward/left/right/turn_left/turn_right/up/down；用画面里物体的视差方向说明。
2) 第一帧到第二帧，相机前进方向在画面中对应哪一侧的物体边界？用 left/center/right 描述屏障/桌/窗。
3) 蓝色布面屏障：第二帧里的屏障和第一帧是不是同一组连续屏障？组内能看到几块布面（同一组被遮挡/转角后仍算一组）。
   给 group_id（A/B/C...）、count_panels、connected(true/false)、relative_side、relation(wall/partition/unknown)。
4) 用两帧里同一物体的位移方向，推断第二帧相机朝向相对第一帧转了多少度（左负右正）。
只输出JSON：{"pair":"","motion":"","motion_evidence":"","forward_on_screen":"left|center|right","turn_deg_est":0,
"barrier":{"same_group_as_prev":true,"group_id":"","count_panels":0,"connected":true,"relation":"wall|partition|unknown","relative_side":"","note":""},
"confidence":0.5}"""


def _json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        return {"raw": text}
    try:
        return json.loads(text[a:b + 1])
    except Exception:
        return {"raw": text}


def probe_pairs(llm, frames_dir: Path, pairs: list[tuple[str, str]]) -> list[dict]:
    """Run one bounded vision call per consecutive pair. Cache-friendly for an S1 run."""
    out = []
    for a, b in pairs:
        try:
            raw = llm.chat(
                [{"role": "system", "content": "只输出JSON，不要解释模板；不确定写unknown。"},
                 {"role": "user", "content": TRAJ_PROMPT}],
                [str(frames_dir / a), str(frames_dir / b)], retries=1)
            obj = _json(raw)
            obj["pair"] = [a, b]
            out.append(obj)
        except Exception as exc:
            out.append({"pair": [a, b], "error": str(exc)})
    return out


def cumulative_turn(records: list[dict]) -> float:
    return float(sum(float(r.get("turn_deg_est") or 0) for r in records if not r.get("error")))


def describe(records: list[dict]) -> dict[str, Any]:
    """The compact evidence that S1 can read back without re-parsing the raw API text."""
    groups: dict[str, dict] = {}
    for r in records:
        b = r.get("barrier") or {}
        gid = str(b.get("group_id") or "")
        if not gid:
            continue
        g = groups.setdefault(gid, {"count_panels": 0, "connected": False,
                                    "relation": "unknown", "frames": []})
        g["count_panels"] = max(g["count_panels"], int(b.get("count_panels") or 0))
        g["connected"] = bool(g["connected"] or b.get("connected"))
        if b.get("relation") not in (None, "", "unknown"):
            g["relation"] = b["relation"]
        g["frames"].append(r.get("pair"))
    return {
        "pairs": len(records),
        "cumulative_turn_deg_est": round(cumulative_turn(records), 1),
        "motion_sequence": [r.get("motion") for r in records if not r.get("error")],
        "barrier_groups": groups,
        "forward_on_screen": [r.get("forward_on_screen") for r in records if not r.get("error")],
    }


def tangent_dir(points: list[list[float]], i: int) -> list[float]:
    """Room-frame Z-up look direction from the path itself (no PI3 yaw involved)."""
    if len(points) < 2:
        return [1.0, 0.0, 0.0]
    if i < len(points) - 1:
        a, b = points[i], points[i + 1]
    else:
        a, b = points[-2], points[-1]
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    n = math.hypot(dx, dy) or 1.0
    return [dx / n, dy / n, 0.0]


def right_of_forward(d: list[float]) -> list[float]:
    """Blender XY, Z-up: right = (dy, -dx)."""
    return [float(d[1]), -float(d[0]), 0.0]
