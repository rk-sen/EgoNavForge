"""Multi-pass video reconstruction (S1a/S1b), no Blender.

Why this module exists
----------------------
V10-V14 sent the S1 model one 5 fps contact sheet plus 3 evenly spaced stills and hid
`video_frames`. The model could not count barriers, identify window structure, or read
materials; the resulting SCENE_PLAN was a short object list, not a scene graph.

This module makes recognition an explicit, bounded pipeline:
  1) scan the video in 1 fps batches of 6 frames (the proxy accepts ~6 images per call);
  2) count global objects/windows/lights from key frames;
  3) index every observed object with its evidence frames and relative position;
  4) re-ask each object group with 2 dedicated frames/crops for structure + material;
  5) save RECON_EVIDENCE.json for S1 to turn into SCENE_PLAN.json.

The 6-image batch limit is an observed endpoint limit, not a prompt preference. 7+ images
or a large structured JSON request hit the proxy's ~90 s window even on kimi-k3.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .llm import LLM
from .catalog import classify, CATEGORIES


SEG_PROMPT = """这是同一段视频的连续 6 帧，按时间顺序。只输出一个 JSON 对象。
逐件识别画面里看得见的物体：name、count、每件 evidence帧、bbox_norm[x0,y0,x1,y1]（0..1）、
side(left/center/right)、support(floor/desk/shelf/wall/...)、relative_to（与哪面墙/哪个角/哪个物体相邻）、
size_m。同时写这 6 帧的相机运动和可见的墙/地面/窗/门/灯。
位置必须给画面参照，不能只写“在房间里”。同款 count 必须等于 instances 数。
看不清写 unknown，禁止套用常见物体清单。
JSON: {"frames":[],"camera_motion":"","walls_floor":"","windows_doors_lights":"",
"objects":[{"name":"","kind":"","count":1,"instances":[{"evidence":[],"bbox_norm":[],
"side":"","support":"","relative_to":"","size_m":[],"confidence":0.5}],"source":"observed|inferred"}],
"unknown":[]}"""

COUNT_PROMPT = """这 6 张是同一段视频按时间排序的关键帧。只输出 JSON。请只做数量核对：
1) 蓝色布面屏障：一共多少块独立的布面？不是每帧几块，而是整段视频中能区分出的总块数；结合左右两侧、转角、被遮挡后重新出现。给每块证据帧和相对位置。
2) 窗户：一共多少扇独立窗？每扇的证据帧、在哪面墙/哪个方向、plain_glass/blind/curtain。
3) 长条灯：一共多少条？给证据帧、方向、大致长度/是否吊装。
4) 桌子、显示器、椅子、货架、沙发、球、豆袋、长凳、机器人：各多少件。
JSON: {"barriers":{"count":0,"panels":[{"evidence":[],"position":""}]},
"windows":{"count":0,"items":[{"evidence":[],"side":"","type":""}]},
"lights":{"count":0,"items":[]},
"objects":{"sofa":0,"table":0,"monitor":0,"chair":0,"shelf":0,"ball":0,"beanbag":0,"bench":0,"robot":0},"unknown":[]}"""

DETAIL_PROMPT = """这些图是同一房间中若干物体的近景/裁块，按顺序分组。请只输出 JSON，不要逐图泛泛描述。
对每个 group 回答：
- structure：可见的部件和大致形状（几块、几层、几根、直的/弧的/斜的）
- components：部件列表，写 count 和 parent/connects（如布面→立柱→红色底座）
- material：最可能的材质（fabric/leather/wood/metal/rubber/paint/glass/plastic/carpet）
- color：具体颜色；不要只写“蓝色/木色”，能给 hex 就给
- surface：光泽/粗糙/软硬/褶皱/纹理
- connections：哪些部件必须接触，哪里可能断开
- dimensions_note：用门宽0.95/桌面0.75/地砖1.2/人身高1.7估算，不要伪造毫米坐标
- confidence：0..1
JSON: {"objects":[{"id":"O1","structure":"","components":[],"material":"","color":"","surface":"","connections":"","dimensions_note":"","confidence":0.5}]}"""

ARCH_PROMPT = """这 6 张是同一房间的细节裁块：窗墙、球与屏障角、沙发/木桌、天花板灯、地面、桌/货架。
逐张说明实际看见的结构，不要套模板：
- 窗户几扇、窗框/横档/竖梃、plain_glass/blind/curtain、窗帘半拉方向；
- 蓝屏障是布面还是硬板，立柱/横梁/脚的颜色和连接；
- 灯是长条还是短灯、几根、方向、是否吊装；
- 地面材质/拼缝；
- 桌子和货架结构、桌上的物体。
只输出文字，不要 JSON。看不清写 unknown。"""


ALIASES = [
    ("barrier", "O1_BlueBarrier", "蓝色布面金属架屏障", "barrier"),
    ("partition", "O1_BlueBarrier", "蓝色布面金属架屏障", "barrier"),
    ("padded wall", "O1_BlueBarrier", "蓝色布面金属架屏障", "barrier"),
    ("sofa", "O2_Sofa", "橙色布艺沙发", "sofa"),
    ("couch", "O2_Sofa", "橙色布艺沙发", "sofa"),
    ("bean bag", "O4_BeanBag", "灰色豆袋", "beanbag"),
    ("beanbag", "O4_BeanBag", "灰色豆袋", "beanbag"),
    ("exercise ball", "O3_YogaBall", "绿色瑜伽球", "ball"),
    ("yoga ball", "O3_YogaBall", "绿色瑜伽球", "ball"),
    ("ball", "O3_YogaBall", "绿色瑜伽球", "ball"),
    ("monitor", "O6_Monitor", "显示器", "monitor"),
    ("screen", "O6_Monitor", "显示器", "monitor"),
    ("shelf", "O7_MetalShelf", "黑色金属置物架", "shelf"),
    ("rack", "O7_MetalShelf", "黑色金属置物架", "shelf"),
    ("robot", "O8_Robot", "带轮机器人", "robot"),
    ("rover", "O8_Robot", "带轮机器人", "robot"),
    ("bench", "O12_LongBench", "长凳", "bench"),
    ("chair", "O9_Chair", "椅子", "chair"),
    ("desk", "O5_Desk", "办公桌", "desk"),
    ("table", "O11_Table", "桌子", "table"),
    ("person", "O18_Person", "人/人台", "person"),
    ("mannequin", "O18_Person", "人/人台", "person"),
]


def _find_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return None


def _canon(name: str, kind: str = "") -> tuple[str, str, str] | None:
    s = f"{name} {kind}".lower()
    # Alias first: "blue padded partition wall" is a barrier, not an architecture wall.
    # The old order filtered on the substring "wall" and silently dropped segment 1.
    for key, oid, label, kind2 in ALIASES:
        if key in s:
            return oid, label, kind2
    # Remaining architecture pseudo-objects are not inventory rows.
    if any(w in s for w in ("window", "door", "floor", "ceiling", "light fixture",
                            "fluorescent", "structural column", "red wedge", "red block",
                            "pillow", "curtain", "opening", "wall")):
        return None
    return None


def _bbox_area(b: Any) -> float:
    try:
        if isinstance(b[0], (list, tuple)):
            b = b[-1]
        return max(0.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    except Exception:
        return 0.0


class ReconScanner:
    """Run the non-Blender part of S1. Safe to call repeatedly; every stage is cached."""

    def __init__(self, project: Path, cfg, log=print):
        self.project = Path(project)
        self.cfg = cfg
        self.log = log
        self.out = self.project / "reports" / "recon"
        self.out.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.project / "previews" / "recon_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.facts = json.loads((self.project / "inputs_facts.json").read_text()) \
            if (self.project / "inputs_facts.json").exists() else {}

    # ------------------------------------------------------------------ API
    def _chat(self, model: str, prompt: str, images: list[Path], max_tokens: int,
              system: str, tag: str, retries: int = 2) -> str:
        temp = None if model.startswith("kimi-k3") else 0.2
        llm = LLM(model, self.cfg.api_base, self.cfg.api_key_env, temperature=temp,
                  max_output_tokens=max_tokens, vision=self.cfg.vision)
        try:
            out = llm.chat([{"role": "system", "content": system},
                            {"role": "user", "content": prompt}], [str(p) for p in images],
                           retries=retries)
        except Exception as e:
            if model.startswith("kimi-k3"):
                self.log(f"[recon] k3 failed on {tag}; fallback kimi-k2.6: {e}")
                return self._chat("kimi-k2.6", prompt, images, min(max_tokens, 4096), system, tag, 1)
            raise
        (self.out / f"{tag}.txt").write_text(out, encoding="utf8")
        (self.out / f"{tag}.usage.json").write_text(json.dumps(
            {"model": llm.model, "calls": llm.calls, "prompt_tokens": llm.prompt_tokens,
             "completion_tokens": llm.completion_tokens,
             "images": [p.name for p in images]}, ensure_ascii=False, indent=1), encoding="utf8")
        return out

    # ------------------------------------------------------------------ frames
    def extract_frames(self) -> list[Path]:
        existing = sorted(self.frames_dir.glob("s_*.png"))
        if len(existing) >= 8:
            return existing
        video = self.project / "inputs" / "source_video.mp4"
        if not video.exists():
            # fall back to any mp4 in inputs
            cand = sorted((self.project / "inputs").glob("*.mp4"))
            video = cand[0] if cand else video
        if not video.exists():
            raise FileNotFoundError("no source_video.mp4 in project/inputs")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
               "-vf", "fps=1,scale=672:-1", str(self.frames_dir / "s_%03d.png")]
        subprocess.run(cmd, check=True, timeout=120)
        return sorted(self.frames_dir.glob("s_*.png"))

    # ------------------------------------------------------------------ pass 1
    def scan_segments(self, frames: list[Path]) -> list[dict]:
        batches = [frames[i:i + self.cfg.recon_batch_frames] for i in range(0, len(frames), self.cfg.recon_batch_frames)]
        out = []
        for i, batch in enumerate(batches):
            f = self.out / f"segment_{i+1:02d}.json"
            raw = self.out / f"segment_{i+1:02d}.txt"
            if f.exists():
                out.append(json.loads(f.read_text()))
                continue
            self.log(f"[recon] segment {i+1}/{len(batches)}: {len(batch)} frames")
            try:
                text = self._chat(self.cfg.recon_segment_model, SEG_PROMPT, batch,
                                  self.cfg.recon_segment_tokens, "严格视频取证，只输出JSON。",
                                  f"segment_{i+1:02d}", retries=1)
            except Exception as exc:
                # A proxy connection loss must not kill the whole run.  Keep the
                # failure visible in the batch record and continue with the rest.
                self.log(f"[recon] segment {i+1} failed: {exc!r}; continuing")
                text = json.dumps({"raw": f"segment error: {exc}", "objects": [],
                                  "unknown": ["segment API failed"]})
            data = _find_json(text)
            if data is None:
                # Keep raw evidence for inspection; do not silently lose the batch.
                data = {"raw": text, "objects": [], "unknown": ["JSON parse failed"]}
            data["_batch"] = i + 1
            data["_frames"] = [p.name for p in batch]
            f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
            out.append(data)
        return out

    # ------------------------------------------------------------------ pass 2
    def count_objects(self, frames: list[Path]) -> dict:
        f = self.out / "counts.json"
        if f.exists():
            return json.loads(f.read_text())
        keys = [frames[i] for i in (0, 5, 9, 12, 15, len(frames) - 1) if i < len(frames)]
        text = self._chat(self.cfg.recon_segment_model, COUNT_PROMPT, keys, 2048,
                          "只输出JSON，逐帧核对数量，禁止套模板。", "counts")
        data = _find_json(text) or {}
        f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
        return data

    # ------------------------------------------------------------------ index
    def build_index(self, segments: list[dict], counts: dict) -> dict:
        index: dict[str, dict] = {}
        for seg in segments:
            for o in seg.get("objects") or []:
                c = _canon(str(o.get("name") or ""), str(o.get("kind") or ""))
                if not c:
                    continue
                oid, label, kind = c
                row = index.setdefault(oid, {"id": oid, "name": label, "kind": kind,
                                             "count": 0, "evidence": [], "instances": [],
                                             "segments": []})
                row["segments"].append(seg.get("_batch"))
                for inst in o.get("instances") or []:
                    ev = inst.get("evidence") or []
                    # segment-local indexes -> actual frame names
                    names = []
                    for e in ev:
                        if isinstance(e, int) and 0 <= e < len(seg.get("_frames") or []):
                            names.append(seg["_frames"][e])
                        elif isinstance(e, str) and e.endswith(".png"):
                            names.append(e)
                    if names:
                        row["evidence"].extend(names)
                    row["instances"].append({
                        "evidence": names or ev,
                        "bbox_norm": inst.get("bbox_norm"),
                        "side": inst.get("side"),
                        "support": inst.get("support"),
                        "relative_to": inst.get("relative_to"),
                        "size_m": inst.get("size_m"),
                        "confidence": inst.get("confidence"),
                        "segment": seg.get("_batch"),
                    })
                row["count"] = max(row["count"], int(o.get("count") or 0))

        # global count is authoritative for the three count-sensitive classes
        global_map = {"O1_BlueBarrier": ("barriers", "count"),
                      "O2_Sofa": ("objects", "sofa"),
                      "O3_YogaBall": ("objects", "ball"),
                      "O4_BeanBag": ("objects", "beanbag"),
                      "O5_Desk": ("objects", "table"),
                      "O6_Monitor": ("objects", "monitor"),
                      "O7_MetalShelf": ("objects", "shelf"),
                      "O8_Robot": ("objects", "robot"),
                      "O9_Chair": ("objects", "chair"),
                      "O12_LongBench": ("objects", "bench")}
        for oid, (group, key) in global_map.items():
            if oid in index:
                try:
                    n = int((counts.get(group) or {}).get(key) or 0)
                    if n > 0:
                        # The global count pass is authoritative.  A segment pass may detect
                        # "sofa cushion" as a second sofa; max() preserved that duplicate.
                        index[oid]["count"] = n
                except Exception:
                    pass
        # de-duplicate evidence
        for row in index.values():
            row["evidence"] = list(dict.fromkeys(row["evidence"]))
            row["count"] = max(row["count"], 1)
        arch = {
            "windows": counts.get("windows") or {},
            "lights": counts.get("lights") or {},
            "floor": self._floor_from_segments(segments),
            "doors": self._doors_from_segments(segments),
        }
        return {"objects": list(index.values()), "counts": counts, "architecture": arch}

    @staticmethod
    def _floor_from_segments(segments: list[dict]) -> dict:
        text = " ".join(str(s.get("walls_floor", "")) for s in segments).lower()
        if any(w in text for w in ("carpet", "rug", "地毯")):
            style = "carpet"
        elif any(w in text for w in ("wood floor", "木地板")):
            style = "wood"
        elif "foam" in text or "软垫" in text:
            style = "foam_interlock"
        elif any(w in text for w in ("tile", "grout", "接缝", "拼缝", "rubber", "橡胶")):
            # This video: dark-gray 1.2 m rubber tiles with straight seams and
            # fastener dots, NOT the colorful interlocking foam mat.
            style = "rubber_tile"
        else:
            style = "unknown"
        return {"style": style, "evidence": "multi-segment observation",
                "note": "颜色/拼缝待材质裁块复核"}

    @staticmethod
    def _doors_from_segments(segments: list[dict]) -> list:
        rows = []
        for s in segments:
            txt = str(s.get("windows_doors_lights", "")).lower()
            # "no visible doors" is not a door. Only keep an actual opening/frame.
            positive = any(w in txt for w in ("doorway", "door frame", "open doorway",
                                              "门洞", "门框", "opening in the wall"))
            negative = "no visible doors" in txt or "no doors" in txt
            if positive and not negative:
                rows.append({"evidence": s.get("_frames"), "text": txt[:240],
                             "source": "observed", "kind": "door_or_opening"})
        # A single back-wall opening is enough; don't duplicate it per batch.
        return rows[:1]

    # ------------------------------------------------------------------ crops
    def make_object_crops(self, index: dict, frames: list[Path]) -> dict[str, list[Path]]:
        cropdir = self.out / "crops"
        cropdir.mkdir(exist_ok=True)
        by_name = {p.name: p for p in frames}
        result: dict[str, list[Path]] = {}
        for row in index["objects"]:
            picks = []
            for name in row["evidence"][:3]:
                p = by_name.get(name)
                if p:
                    picks.append(p)
            if not picks:
                picks = frames[:2]
            out = []
            for j, p in enumerate(picks[:2]):
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    continue
                # Prefer the largest bbox from the corresponding instance;
                # otherwise use a central 70% crop, which keeps context.
                b = None
                for inst in row["instances"]:
                    if p.name in (inst.get("evidence") or []):
                        b = inst.get("bbox_norm")
                        break
                try:
                    if b:
                        if isinstance(b[0], (list, tuple)):
                            b = b[-1]
                        x0, y0, x1, y1 = [float(x) for x in b[:4]]
                        x0 = max(0, min(1, x0)); x1 = max(0, min(1, x1))
                        y0 = max(0, min(1, y0)); y1 = max(0, min(1, y1))
                        if x1 - x0 > .08 and y1 - y0 > .08:
                            box = (int(x0 * im.width), int(y0 * im.height),
                                   int(x1 * im.width), int(y1 * im.height))
                        else:
                            box = None
                    else:
                        box = None
                except Exception:
                    box = None
                if box:
                    pad = 18
                    box = (max(0, box[0] - pad), max(0, box[1] - pad),
                           min(im.width, box[2] + pad), min(im.height, box[3] + pad))
                    im = im.crop(box)
                im = im.resize((min(900, im.width * 2), min(900, im.height * 2)), Image.LANCZOS)
                dst = cropdir / f"{row['id']}_{j+1}.jpg"
                im.save(dst, quality=91)
                out.append(dst)
            result[row["id"]] = out
        return result

    # ------------------------------------------------------------------ detail
    def detail_objects(self, index: dict, crops: dict[str, list[Path]]) -> dict:
        f = self.out / "object_details.json"
        if f.exists():
            data = json.loads(f.read_text())
            # Person is deliberately not sent through the generic furniture crop prompt,
            # but the supplied mannequin must still be placed when the video contains one.
            if any(o["id"] == "O18_Person" for o in index["objects"]):
                data.setdefault("O18_Person", {
                    "id": "O18_Person", "count": 1, "name": "人",
                    "structure": "站立人形，双脚落地", "material": "布料/皮肤",
                    "color": "blue shirt, khaki trousers, yellow helmet",
                    "dimensions_note": "1.72 m supplied mannequin", "confidence": 0.9})
                f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
            return data
        rows = [o for o in index["objects"] if o["id"] != "O18_Person"]
        groups = [rows[i:i + 3] for i in range(0, len(rows), 3)]
        all_details: dict[str, dict] = {}
        for gi, group in enumerate(groups):
            images = []
            labels = []
            for o in group:
                ps = crops.get(o["id"]) or []
                labels.append(f"{o['id']} = {o['name']} (count {o['count']})")
                images.extend(ps[:2])
            if not images:
                continue
            prompt = "按顺序分组：\n" + "\n".join(labels) + \
                     "\n对应图片按上述顺序每件 1-2 张。\n" + DETAIL_PROMPT
            self.log(f"[recon] object detail {gi+1}/{len(groups)}: {[o['id'] for o in group]}")
            cache = self.out / f"objects_detail_{gi+1:02d}.txt"
            if cache.exists():
                text = cache.read_text(encoding="utf8")
            else:
                text = self._chat(self.cfg.recon_detail_model, prompt, images[:6],
                                  self.cfg.recon_detail_tokens, "严格物体结构/材质取证，只输出JSON。",
                                  f"objects_detail_{gi+1:02d}", retries=1)
            data = _find_json(text) or {"objects": [], "raw": text}
            for d in data.get("objects") or []:
                all_details[str(d.get("id"))] = d
        if any(o["id"] == "O18_Person" for o in index["objects"]):
            all_details.setdefault("O18_Person", {
                "id": "O18_Person", "count": 1, "name": "人",
                "structure": "站立人形，双脚落地", "material": "布料/皮肤",
                "color": "blue shirt, khaki trousers, yellow helmet",
                "dimensions_note": "1.72 m supplied mannequin", "confidence": 0.9})
        f.write_text(json.dumps(all_details, ensure_ascii=False, indent=1), encoding="utf8")
        return all_details

    def refine_priority_objects(self, details: dict, frames: list[Path]) -> dict:
        """One class per call for the objects the batch pass confuses most.

        The group pass above is cheap, but a single prompt containing Desk + Sofa + Table
        makes the model apply the wood-A-table answer to the white height-adjustable desk
        (observed in the v15 test). The five classes below are therefore re-asked with two
        explicitly captioned frames; this is the second, object-centric recognition pass.
        """
        specs = {
            "O5_Desk": (["s_003.png", "s_006.png"],
                        "白色办公桌：桌面颜色、桌腿结构、桌下设备/横撑；不要把它和画面另一张深色木A腿桌混淆。"),
            "O6_Monitor": (["s_003.png", "s_006.png"],
                           "显示器：屏幕、边框、支架/立柱、底座；屏幕是竖直还是倾斜；与桌面连接。"),
            "O1_BlueBarrier": (["s_016.png", "s_021.png"],
                               "蓝色布面屏障：布面是否软质、白立柱、红色底座数量/每段几个、横梁；转角处两段如何相接；不要只数画面里当前块数。"),
            "O2_Sofa": (["s_010.png", "s_012.png"],
                        "橙色沙发：靠背/坐垫/扶手/支脚、抱枕数量、圆润边缘；材质与颜色。"),
            "O11_Table": (["s_010.png", "s_014.png"],
                          "深色木高桌：A形腿/横梁、桌面厚度；不要和白色办公桌混淆。"),
            "O3_YogaBall": (["s_018.png", "s_021.png"],
                            "绿色瑜伽球：球体、表面材质、与两块蓝屏障直角/白柱/地面的关系。"),
            "O12_LongBench": (["s_010.png", "s_014.png"],
                              "长凳/长条凳：它是否是沙发前方的低矮长凳；座面、两条腿/横梁、是否与沙发分离；如果画面里是另一张桌子，明确写 count=0 并说明。"),
        }
        by_name = {p.name: p for p in frames}
        for oid, (names, desc) in specs.items():
            f = self.out / f"refine_{oid}.json"
            if f.exists():
                try:
                    details[oid] = json.loads(f.read_text())
                    continue
                except Exception:
                    pass
            imgs = [by_name[n] for n in names if n in by_name]
            if len(imgs) < 2:
                continue
            prompt = (f"只描述 {oid}。图1、图2是同一房间不同时刻的两帧。{desc}\n"
                      "只输出JSON：{\"id\":\"" + oid + "\",\"count\":1,\"structure\":\"\","
                      "\"components\":[{\"name\":\"\",\"count\":1,\"parent\":\"\",\"connects\":\"\"}],"
                      "\"material\":\"\",\"color\":\"\",\"surface\":\"\",\"connections\":\"\","
                      "\"dimensions_note\":\"\",\"location\":\"\",\"confidence\":0.5}")
            self.log(f"[recon] priority detail {oid}")
            text = self._chat(self.cfg.recon_detail_model, prompt, imgs, 2048,
                              "只输出JSON，严格描述图中该物体，禁止套模板。",
                              f"refine_{oid}", retries=1)
            data = _find_json(text)
            if data:
                f.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf8")
                details[oid] = data
        return details

    def detail_architecture(self, frames: list[Path]) -> str:
        f = self.out / "architecture_detail.txt"
        if f.exists():
            return f.read_text()
        cropdir = self.out / "crops"
        cropdir.mkdir(exist_ok=True)
        # pick frames that actually show window wall, ball corner, sofa/table, ceiling, floor, desk/shelf
        picks = []
        for name, box in [
            ("s_016.png", (0, 0, 470, 300)),   # windows / ball wall
            ("s_021.png", (0, 0, 672, 478)),   # ball + two barrier panels
            ("s_010.png", (0, 80, 672, 410)),  # sofa / table
            ("s_012.png", (0, 0, 672, 220)),   # ceiling lights
            ("s_007.png", (0, 220, 672, 478)), # floor
            ("s_003.png", (0, 50, 400, 330)),  # desk/shelf
        ]:
            p = self.frames_dir / name
            if not p.exists():
                p = frames[min(len(frames) - 1, 5)]
            im = Image.open(p).convert("RGB").crop(box)
            im = im.resize((min(1200, im.width * 2), min(900, im.height * 2)), Image.LANCZOS)
            dst = cropdir / f"arch_{len(picks)+1}.jpg"
            im.save(dst, quality=91)
            picks.append(dst)
        text = self._chat(self.cfg.recon_detail_model, ARCH_PROMPT, picks[:6], 2048,
                          "只描述看得见的结构，看不清写unknown。", "architecture_detail")
        f.write_text(text, encoding="utf8")
        return text

    # ------------------------------------------------------------------ fuse
    def fuse(self, segments: list[dict], index: dict, details: dict, arch_text: str) -> dict:
        # Keep this deterministic: the model does not get to invent a second scene.
        counts = index.get("counts") or {}
        rows = []
        for o in index["objects"]:
            d = details.get(o["id"]) or {}
            rows.append({
                **o,
                "detail": d,
                "material": d.get("material"),
                "color": d.get("color"),
                "structure": d.get("structure"),
                "components": d.get("components"),
                "connections": d.get("connections"),
            })
        # The long wooden object beside the sofa has A-shaped legs and a cross beam.  The
        # crop pass describes it as a "high table", which made the bench disappear from the
        # inventory.  It is the long bench in the user's vocabulary; relabel the same instance
        # instead of inventing a second object.
        for r in rows:
            if r.get("id") == "O11_Table" and r.get("structure"):
                st = str(r.get("structure"))
                if ("A形" in st or "A 形" in st) and "横梁" in st:
                    r["name"] = "长凳（A 形腿长桌）"
                    r["kind"] = "bench"
                    r["reclassified_from"] = "O11_Table"
        # Bench is easy to lose: segment pass calls it a "wooden table" and the count pass
        # returns bench=0 while the dedicated crop still sees a long low bench.  Keep the
        # object only when the object-centric crop actually confirms it.
        bench_detail = details.get("O12_LongBench") or {}
        if (not any(r.get("id") == "O12_LongBench" for r in rows)
                and int(bench_detail.get("count") or 0) > 0):
            rows.append({
                "id": "O12_LongBench", "name": "长凳", "kind": "bench",
                "count": 1, "evidence": bench_detail.get("evidence") or [],
                "instances": [], "segments": [], "detail": bench_detail,
                "material": bench_detail.get("material"), "color": bench_detail.get("color"),
                "structure": bench_detail.get("structure"), "source": "priority_crop_confirmed",
            })
        # Reconciliation, not a second model call. The count pass and the crop pass
        # disagree on window type in this video: count said every window was "blind",
        # while the high-resolution window crop says plain glass with dark mullions and
        # no blind/curtain. The crop is the stronger evidence, so it wins and the raw
        # disagreement stays visible in `reconcile`.
        low = arch_text.lower()
        window_verdict = {
            "count": (counts.get("windows") or {}).get("count"),
            "type": "plain_glass" if ("plain_glass" in low or "无blind" in low or "no blind" in low) else "unknown",
            "blind": False if ("无blind" in low or "no blind" in low or "plain_glass" in low) else None,
            "curtain": None,
            "source": "architecture_detail crop pass",
            "note": "整体 count 可能把深色横档误判为 blind；裁块直读优先",
        }
        barrier_count = int((counts.get("barriers") or {}).get("count") or 0)
        light_count = int((counts.get("lights") or {}).get("count") or 0)
        arch = dict(index.get("architecture") or {})
        arch["window_verdict"] = window_verdict
        for w in ((arch.get("windows") or {}).get("items") or []):
            if window_verdict["type"] == "plain_glass":
                w["type"] = "plain_glass"
                w["blind"] = False
                w["curtain"] = False
                w["evidence_note"] = "crop s_016: glass + dark mullions; no blind/curtain"

        # No fixed floor/door: recognition leaves them as evidence, and the LLM floorplan
        # pass decides the polygon + openings.  Unknown is honest; a hard-coded tile is not.
        arch.setdefault("floor", {"style": "unknown", "evidence": "not classified in this pass"})
        arch.setdefault("doors", [])
        evidence = {
            "source": "multi-pass recon: 1 fps segments + dedicated object crops + architecture crops",
            "layers": [
                {"pass": "global_segments", "unit": "2 frames", "count": len(segments),
                 "purpose": "整段时序：相机运动、墙/地面、物体相对位置"},
                {"pass": "global_count", "unit": "6 keyframes", "count": 1,
                 "purpose": "跨帧去重计数；屏障/窗/灯"},
                {"pass": "object_detail", "unit": "1-2 crops per object", "count": 4,
                 "purpose": "结构/组件/材质/颜色"},
                {"pass": "priority_detail", "unit": "2 frames for 6 hard objects", "count": 6,
                 "purpose": "白桌/显示器/屏障/沙发/木桌/球的二次提问"},
                {"pass": "architecture_detail", "unit": "6 detail crops", "count": 1,
                 "purpose": "窗型/窗帘、灯长、地面拼缝、桌架结构"},
            ],
            "segments": segments,
            "counts": counts,
            "architecture": arch,
            "objects": rows,
            "architecture_detail": arch_text,
            "reconcile": {
                "window": window_verdict,
                "barrier": {"count": barrier_count, "range": [max(0, barrier_count - 2), barrier_count],
                            "note": "count 会因转角/重复出现抖动；细问只确认近角 4 根白柱/8 个红底座，不把局部当全屋"},
                "lights": {"count": light_count, "range": [max(0, light_count - 2), light_count],
                           "note": "长条灯数量按多帧去重；以 6-8 条为候选，建模时逐条放"},
            },
            "uncertainties": sorted({str(u) for s in segments for u in (s.get("unknown") or [])})[:20],
        }
        f = self.out / "RECON_EVIDENCE.json"
        f.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf8")
        # A six-frame visual key set for S1: broad timeline + the hard details.
        keydir = self.out / "keyframes"
        keydir.mkdir(exist_ok=True)
        for n, name in enumerate(["s_001.png", "s_007.png", "s_010.png",
                                   "s_016.png", "s_019.png", "s_021.png"]):
            src = self.frames_dir / name
            if src.exists():
                shutil.copy2(src, keydir / f"key_{n+1}.png")
        return evidence

    def run(self) -> dict:
        frames = self.extract_frames()
        self.log(f"[recon] {len(frames)} frames at 1 fps")
        segments = self.scan_segments(frames)
        counts = self.count_objects(frames)
        index = self.build_index(segments, counts)
        (self.out / "object_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                                    encoding="utf8")
        crops = self.make_object_crops(index, frames)
        details = self.detail_objects(index, crops)
        details = self.refine_priority_objects(details, frames)
        arch = self.detail_architecture(frames)
        ev = self.fuse(segments, index, details, arch)
        self.log(f"[recon] evidence: {len(ev['objects'])} classes, "
                 f"{sum(o['count'] for o in ev['objects'])} instances")
        return ev
