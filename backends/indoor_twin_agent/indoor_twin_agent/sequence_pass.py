"""Sequential visual cognition: video order before metric layout.

The top-down depth pass was good at drawing a floorplate but weak at keeping one barrier
group identical across frames: it saw a blue run in the first frames and another run later,
then merged all panels into one horizontal bar.  This pass asks only for *ordered visual
evidence* (what is left/right/behind the camera in each frame, and which blue panels persist),
without asking for room coordinates.  The top-down pass can then use that temporal identity
as a prior instead of treating every frame as an independent snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

from .llm import LLM
from .recon import _find_json
from .topdown_pass import _frames, _pick, _trajectory

PROMPT = """你是一个视频时序测绘员。给你 4 张按时间顺序排列的室内视频帧，以及相机的重力对齐轨迹。
先不要给房间坐标，也不要急着把蓝色屏障画成一条长墙。请按“帧顺序”观察，再归纳跨帧身份。

重点：
1) 逐帧判断相机运动：前进/后退、左转/右转/原地转，依据背景移动、桌沿和墙线判断。
2) 逐帧记录蓝色屏障：看到几片、是连续的一段还是几段、在画面左/中/右、近/远、是否成直角、是否贴墙、是否有红黑底座。不要跨帧重复计数。
3) 逐帧记录明显的物体：绿色球、橙色沙发、长凳、办公桌/显示器、货架、豆袋椅。给出它们与蓝色屏障的相对前后关系。
4) 最后把同一组屏障跨帧串起来：哪些帧看到的是同一组；这组总共有几片；它相对第一帧相机的位置是左侧/右侧/前方/后方；它更可能是贴墙还是在房间中间隔断。
5) 如果两组屏障在不同帧里先后出现，不要合并；给出它们的出现顺序和相对转向关系。

只输出 JSON：
{"frames":[{"index":"s_001","time_s":0.0,"camera_motion":"forward|back|left|right|rotate_left|rotate_right|stationary|unknown","barriers":{"visible_panels":0,"segments":[{"side":"left|center|right","near_far":"near|mid|far","connected":true,"right_angle":false,"wall_relation":"wall|divider|unknown"}],"note":""},"objects":[{"name":"绿色健身球","side":"left|center|right","relative":"front|behind|left|right","note":""}],"summary":""}],
 "camera_motion_summary":{"forward_back":"...","turns":"...","path_shape":"...","room_axis_hint":"..."},
 "barrier_tracks":[{"id":"A","frames":["s_001","s_002"],"total_panels":3,"relative_to_first":"right|left|front|back","wall_relation":"wall|divider|unknown","right_angle":false,"confidence":0.0,"reason":""}],
 "object_tracks":[{"name":"橙色沙发","frames":["s_006","s_007"],"relative_to_barrier":"in front|behind|left|right","confidence":0.0}],
 "uncertainties":["..."]}

不要把同一片屏障在连续帧里重复当成新组；不要把“相机转向后看到的同一组”拆成两组。
轨迹摘要：{trajectory}
"""


def generate_sequence(project: Path, cfg, cal: dict, arch: dict, evidence: dict, log=print) -> dict:
    project = Path(project)
    out = project / "SEQUENCE_COGNITION.json"
    frames = _frames(project)
    llm = LLM(getattr(cfg, "recon_detail_model", None) or cfg.model, cfg.api_base,
              cfg.api_key_env, temperature=0.2, max_output_tokens=4096, vision=True)
    seq_frames = _pick(frames, 4, 0)
    prompt = PROMPT.replace("{trajectory}", json.dumps(_trajectory(cal), ensure_ascii=False))
    log(f"[sequence] ordered visual cognition: {llm.model}, {len(seq_frames)} frames")
    text = llm.chat([
        {"role": "system", "content": "你只看视频时序和可见关系，不输出房间坐标；只输出 JSON。"},
        {"role": "user", "content": prompt},
    ], [str(p) for p in seq_frames], retries=2)
    (project / "reports" / "recon").mkdir(parents=True, exist_ok=True)
    (project / "reports" / "recon" / "sequence_raw.json").write_text(text, encoding="utf8")
    data = _find_json(text) or {}
    data["source"] = "ordered visual pass: 4 frames + gravity trajectory"
    data["frames_used"] = [p.name for p in seq_frames]
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    return data
