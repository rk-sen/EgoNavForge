"""v4 loop: belief-backed state machine.

Fixes over v3 (each reported by an actual run):
1. Stages can be **reopened**. Every failing check names its owning stage; a failure regresses the
   machine instead of leaving "S7: keep rendering" as the only instruction.
2. Memory is a **belief document on disk** (objects + camera residual + open issues), rebuilt into
   every request. The last-8-messages window is convenience, never the source of truth.
3. A failing batch no longer drops its tail: skipped calls are queued and shown next turn; safe tails
   (`note`, `list_files`) are replayed automatically.
4. Identical retries are counted. The second identical failure returns `stalled: true` plus explicit
   alternatives, so "retry the same call" cannot pass as progress.
5. Vision failures are attributed: a flat render opens a `flat_render` issue owned by S3 and regresses
   the stage — it is not just a failed tool result.
6. S7 has real content: `finish` is refused unless previews/compare/QA/files all check out, and the
   S7 gate no longer auto-passes.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import inputs_prep, prompts
from .belief import STAGE_OF_CHECK, Belief
from .config import RunConfig
from .executor import Executor
from .llm import LLM, LLMError, estimate_chars
from .recon import ReconScanner
from .protocol import ProtocolError, parse_calls
from .tools import Tools, image_stats

STAGES = ["S1_recon", "S2_shell", "S3_assets", "S4_qa"]
SAFE_TAIL = {"note", "list_files"}
ALTERNATIVES = {
    "patch_file": "锚点对不上时：write_file 整段替换该模块，或换更长的唯一锚点",
    "blender_py": "先确认 .py 已落盘且 blend 存在；用 script= 路径，或 code= 内联",
    "render_views": "平图先查相机是否在墙体内部；用 qa_scene 的 camera_clearance 定位",
    "qa_scene": "用 list_files('*.blend') 确认 blend 名；不要猜路径",
    "read_file": "需要的数字已在 <input_facts>/<belief> 里，换目标再读",
    "write_file": "改动已有模块用 patch_file，别整文件覆盖",
}


class Agent:
    def __init__(self, cfg: RunConfig, root: Path, lib_dir: Path, log=print):
        self.cfg = cfg
        self.root = Path(root)
        self.project = self.root / "project"
        self.project.mkdir(parents=True, exist_ok=True)
        for sub in ("inputs", "_tmp", "_lib", "previews"):
            (self.project / sub).mkdir(exist_ok=True)
        self.log = log
        self.state_path = self.root / "state.json"
        self.ledger_path = self.root / "ledger.md"
        self.t0 = time.time()
        self.state = self._load_state()
        self.belief = Belief(self.project / "belief.json")

        for f in Path(lib_dir).glob("*.py"):
            dst = self.project / "_lib" / f.name
            if dst.resolve() != f.resolve():
                shutil.copy2(f, dst)
        for f in (Path(cfg.video), Path(cfg.trajectory), Path(cfg.intrinsics)):
            if f.exists():
                dst = self.project / "inputs" / f.name
                if f.resolve() != dst.resolve():
                    shutil.copy2(f, dst)

        self.state["video_path"] = str(self.project / "inputs" / Path(cfg.video).name)
        self.state.setdefault("video_meta", self._probe_meta())
        self.facts = inputs_prep.prepare(self.project, self.state.get("video_meta") or {})
        self.log(f"[prelude] ready={self.facts.get('ready')} "
                 f"hfov={self.facts.get('intrinsics', {}).get('hfov_deg')} "
                 f"poses={(self.facts.get('trajectory') or {}).get('poses')}")

        ex = Executor(self.project, blender=cfg.blender, max_seconds=int(min(cfg.max_minutes * 60, 900)))
        self.tools = Tools(self.project, ex, self.project / "_lib", cfg.quality, self.state,
                           self.ledger_path, on_note=lambda line: self.log("  · " + line))
        self.tools.belief = self.belief
        # S1 is plan-only and its sampling tools are hidden. The frames must already exist before
        # the first request, otherwise the model gets text but no images (v10 early runs).
        # Dense temporal window: 5 fps down-sampling from the source (30 fps is legal but
        # 5 frames/second preserves object semantics and structure, unlike random keyframes).
        if not (self.project / "previews" / "timeline_5fps.png").exists():
            try:
                self.tools.call("video_frames", {"fps": 5, "res": 672})
            except Exception as exc:
                self.log(f"[prelude] 5fps frame extraction failed: {exc}")
        # S1 evidence is produced before the S1 dialogue. This is deliberately outside
        # the ReAct loop: object counting and per-object material questions are not
        # "turns", and must not compete with the 70-step budget. Evidence is cached on
        # disk so a restart does not re-bill the API.
        self.recon_evidence: dict | None = None
        recon_path = self.project / "reports" / "recon" / "RECON_EVIDENCE.json"
        if cfg.recon_scan and not (self.project / "SCENE_PLAN.json").exists():
            try:
                if recon_path.exists():
                    self.recon_evidence = json.loads(recon_path.read_text())
                else:
                    self.recon_evidence = ReconScanner(self.project, cfg, self.log).run()
                self.log(f"[recon] loaded {len(self.recon_evidence.get('objects') or [])} object classes")
            except Exception as exc:
                self.log(f"[recon] FAILED (S1 will use the old 4-frame fallback): {exc}")
        elif recon_path.exists():
            try:
                self.recon_evidence = json.loads(recon_path.read_text())
            except Exception:
                pass
        self.llm = LLM(cfg.model, cfg.api_base, cfg.api_key_env, cfg.temperature,
                       cfg.max_output_tokens, cfg.vision)
        self.legal = {t["name"] for t in self.tools.schemas()}
        self.history: list[dict[str, str]] = []

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except Exception:
                pass
        return {"stage": STAGES[0], "steps": 0, "done": False, "stage_evidence": {},
                "summary": "", "attempts": {}, "pending_calls": [], "finished": False}

    def save(self):
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf8")

    def _probe_meta(self) -> dict[str, Any]:
        try:
            r = Executor(self.project / "inputs").run(
                ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json",
                 str(Path(self.state["video_path"]))], timeout=60)
            info = json.loads(r["stdout"])["streams"][0]
        except Exception:
            return {}
        num, _, den = str(info.get("r_frame_rate", "25/1")).partition("/")
        return {"width": info.get("width"), "height": info.get("height"),
                "frames": info.get("nb_frames"), "fps": round(float(num) / float(den or 1), 2),
                "duration_s": round(float(info.get("duration") or 0), 2)}

    # ------------------------------------------------------------------ context
    def _artifact_index(self) -> list[str]:
        rows = []
        for p in sorted(self.project.rglob("*")):
            if not p.is_file() or "_tmp" in p.parts or "_lib" in p.parts:
                continue
            rel = str(p.relative_to(self.project))
            kb = round(p.stat().st_size / 1024, 1)
            note = ""
            if p.suffix == ".json" and p.stat().st_size < 20_000:
                try:
                    d = json.loads(p.read_text())
                    if isinstance(d, dict):
                        if isinstance(d.get("objects"), list):
                            note = f" objects={len(d['objects'])}"
                        elif "ok" in d:
                            note = f" ok={d.get('ok')}"
                        elif isinstance(d.get("camera_path"), list):
                            note = f" poses={len(d['camera_path'])}"
                    elif isinstance(d, list):
                        note = f" list[{len(d)}]"
                except Exception:
                    pass
            rows.append(f"{rel} ({kb}KB){note}")
        return rows[:60]

    def _ledger(self) -> list[str]:
        if not self.ledger_path.exists():
            return []
        return [l.strip() for l in self.ledger_path.read_text().splitlines() if l.strip()]

    def _frame_picks(self, n: int) -> list[str]:
        """Frames are the only window onto the video. Never take the alphabetically first n:
        for a 20 s walkthrough that is the first 6 s only, so everything that appears later
        (the green yoga ball, the orange sofa) is invisible to the model. Sample across the
        whole timeline instead."""
        frames = sorted((self.project / "frames").glob("*.png"))
        if not frames:
            return []
        if len(frames) <= n:
            return [str(p) for p in frames]
        idx = sorted({round(i * (len(frames) - 1) / (n - 1)) for i in range(n)})
        return [str(frames[i]) for i in idx]

    def _images(self) -> list[str]:
        stage = self.state["stage"]
        picks: list[str] = []
        # S1 has already paid for the multi-pass recognizer. Give it exactly those six
        # audit frames, not a contact sheet plus three random stills that re-hide the sofa.
        if stage == "S1_recon" and self.recon_evidence:
            keydir = self.project / "reports" / "recon" / "keyframes"
            keys = sorted(keydir.glob("key_*.png"))
            if keys:
                return [str(p) for p in keys[:6]]
        # Legacy fallback below (kept for projects without recon cache).
        tl = self.project / "previews" / "timeline_5fps.png"
        if tl.exists():
            picks.append(str(tl))
        sheet = self.project / "previews" / "objects_sheet.png"
        if stage == "S3_assets" and sheet.exists():
            picks.append(str(sheet))
        picks += self._frame_picks(4 if stage == "S1_recon" else 2)
        prev = self.project / "previews"
        if prev.is_dir():
            comps = sorted(prev.glob("compare_*.png"), key=lambda p: p.stat().st_mtime)
            if comps:
                picks.append(str(comps[-1]))
            shots = sorted([p for p in prev.glob("*.png") if not p.name.startswith("compare_")],
                           key=lambda p: p.stat().st_mtime)
            limit = 4 if stage == "S1_recon" else 5
            picks += [str(p) for p in shots[-(limit - len(picks)):]] if len(picks) < limit else []
        return picks[:4 if stage == "S1_recon" else 5]

    def _messages(self) -> list[dict[str, str]]:
        stage = self.state["stage"]
        blocks = [
            inputs_prep.facts_block(self.facts),
            prompts.build_layout_block(self.tools.tree_overview(), str(self.project)),
            self.belief.block(),
            prompts.build_state_block(self.state, self._ledger(), self._artifact_index(), stage,
                                      self.cfg.quality.table(), []),
        ]
        if stage == "S1_recon" and self.recon_evidence:
            # Compact, bounded injection. The full RECON_EVIDENCE.json is on disk and can
            # be read by the plan writer; this is the evidence it must not contradict.
            ev = self.recon_evidence
            compact = {
                "architecture": ev.get("architecture"),
                "counts": ev.get("counts"),
                "objects": [{k: o.get(k) for k in ("id", "name", "count", "evidence",
                                                   "material", "color", "structure",
                                                   "components", "connections")}
                            for o in (ev.get("objects") or [])],
                "architecture_detail": (ev.get("architecture_detail") or "")[:3500],
            }
            blocks.append("<recon_evidence>\n" + json.dumps(compact, ensure_ascii=False, indent=1)[:14000]
                          + "\n</recon_evidence>")
        gate_why = (self.state.get("stage_evidence") or {}).get(stage) or ""
        if gate_why and "PASS" not in gate_why:
            blocks.append(prompts.build_gate_block(stage, gate_why,
                                                   [i["check"] for i in self.belief.blocking(stage)]))
        if self.belief.issues():
            blocks.append("<open_issues>\n" + "\n".join(
                f"- [{i['stage']}] {i['check']} (attempts={i['attempts']}): {i['detail'][:160]}"
                for i in self.belief.issues()[:10]) + "\n</open_issues>")
        stall = int(self.state.get("stall_turns") or 0)
        if stall >= 3:
            hard = stall >= 6
            blocks.append("<stall>连续 " + str(stall) + " 轮没有任何新证据"
                          "（阶段/门禁原因 / qa 通过项 / 渲染数 / belief 物体数 / 未关闭问题都没变）。"
                          + ("｜**硬停**：本轮禁止 read_file、prepare_inputs、image_stats；"
                             f"只能做「{stage}」该做的事：{prompts.GATE_ACTIONS.get(stage, '')}"
                             if hard else
                             "禁止再读同一文件；下一步必须产出新东西：改一处几何后 render_views，或修完一项后 qa_scene。")
                          + "</stall>")
        pending = self.state.get("pending_calls") or []
        if pending:
            blocks.append("<pending_calls>上批被跳过的调用，先补做（可改参数）：\n"
                          + json.dumps(pending, ensure_ascii=False) + "\n</pending_calls>")
        schemas = self.tools.schemas()
        if stage == "S1_recon":
            # The request already carries every frame. Exposing the sampling/reading tools here
            # made the model spend whole turns re-sampling and never write the plan.
            allowed = {"write_file", "patch_file", "note", "list_files"}
            schemas = [t for t in schemas if t.get("name") in allowed]
        msgs = [{"role": "system", "content": prompts.SYSTEM + prompts.TOOL_PROTOCOL +
                     "\n可用工具：\n" + json.dumps(schemas, ensure_ascii=False)}]
        if self.state.get("summary"):
            msgs.append({"role": "user", "content": "有损摘要（以 <belief> 与文件为准）：\n" + self.state["summary"]})
        for m in self.history[-8:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": "\n".join(blocks) + "\n\n" + prompts.kickoff(stage)})
        return msgs

    # ------------------------------------------------------------------ gates
    def _qa(self) -> dict | None:
        qa = self.tools.stats.get("qa")
        if qa:
            return qa
        f = self.project / "qa.json"          # survive a restart: the artifact is the truth
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
        return None

    @staticmethod
    def _placement(item: dict) -> bool:
        """A plan item must say where it is: place/cell, per-instance entries, or a relation."""
        rel = item.get("relation") or {}
        if isinstance(rel, dict) and rel.get("type") in ("on", "near", "against") and rel.get("target"):
            return True

        def one(q):
            if isinstance(q, (list, tuple)) and len(q) >= 2:
                return True
            if not isinstance(q, dict):
                return False
            if "x" in q and "y" in q:
                return True
            return bool(q.get("cell") or q.get("place"))
        cnt = int(item.get("count") or 1)
        inst = item.get("instances") or []
        if cnt > 1:                       # several copies: each copy needs its own spot
            return len(inst) >= cnt and all(one(q) for q in inst)
        if one(item.get("place")) or item.get("cell"):
            return True
        return bool(inst) and one(inst[0])

    def _gate(self, stage: str) -> tuple[bool, str]:
        p = self.project
        qa = self._qa() or {}
        blocking = self.belief.blocking(stage)
        if blocking:
            return False, "未关闭问题：" + ", ".join(i["check"] for i in blocking)

        if stage == "S1_recon":
            f = p / "SCENE_PLAN.json"
            if not f.exists():
                return False, ("缺 SCENE_PLAN.json。你已经有跨时间轴的 4 张关键帧（本请求已附带），"
                               "不要再 video_frames/read_file 图片；现在直接 write_file SCENE_PLAN.json。")
            try:
                plan = json.loads(f.read_text())
            except Exception as exc:
                return False, f"SCENE_PLAN.json 非法 JSON: {exc}"
            items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
            if not items:
                return False, "SCENE_PLAN 里没有任何物体（objects/inferred）"
            observed = list(plan.get("objects") or [])
            if not observed:
                return False, ("objects 为空：把所有主要前景都塞进 inferred 等于没有识别。"
                               "沙发/屏障/球/桌/显示器/架/椅子这些 observed 必须先写进 objects。")
            main_kind = {"sofa", "barrier", "ball", "desk", "table", "monitor", "shelf",
                         "chair", "office_chair", "task_chair", "beanbag", "robot", "bench",
                         "stool", "dining_table"}
            main_inf = []
            for o in (plan.get("inferred") or []):
                oid = str(o.get("id") or "")
                kind = str(o.get("kind") or "").lower()
                name = str(o.get("name") or "")
                is_main = kind in main_kind or any(
                    w in name.lower() for w in ("sofa", "沙", "barrier", "屏", "ball", "球",
                                                 "desk", "桌", "monitor", "显示", "shelf", "架"))
                if oid.upper().startswith("INF_") and is_main:
                    main_inf.append(oid)
            if main_inf:
                return False, ("主要前景不能用 INF_ 脑补：" + ", ".join(main_inf[:6]) +
                               "；看见轮廓就写进 objects，并给真实 seen_in/尺寸/pose_yaw_deg。")
            arch = plan.get("architecture") or {}
            fp = arch.get("footprint_m") or arch.get("floorplan_m") or arch.get("outline_m")
            if not fp:
                return False, ("缺 architecture.footprint_m：户型不能退化成 room_bbox_m 方盒；"
                               "用整段视频理解墙线，按 [x,y] 给 4–12 个顶点，凹角/L 形要保留。"
                               "轨迹只给距离参考，不是户型。")
            try:
                pts = [(float(q[0]), float(q[1])) for q in fp if isinstance(q, (list, tuple)) and len(q) >= 2]
            except Exception:
                pts = []
            if len(pts) < 4:
                return False, "architecture.footprint_m 至少 4 个 [x,y] 顶点"
            if len(pts) > 20:
                return False, "architecture.footprint_m 顶点太多（≤ 20），只保留墙线拐点"
            # shoelace area, catches degenerate line/duplicate points
            area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] -
                           pts[(i + 1) % len(pts)][0] * pts[i][1]
                           for i in range(len(pts))) / 2)
            if area < 1.0:
                return False, "architecture.footprint_m 面积过小/退化成线，重新按墙线估"
            if "windows" not in arch or "doors" not in arch:
                return False, ("architecture 必须显式写 windows/doors（可以为 []，但不能不写）；"
                               "视频里不止一个窗/门时按墙段逐条写，不能只写一个。")
            # wall openings need enough fields for the deterministic shell builder
            for kind in ("windows", "doors"):
                for op in (arch.get(kind) or []):
                    if not isinstance(op, dict):
                        return False, f"architecture.{kind} 里每一项必须是对象"
                    if op.get("width_m") is None or op.get("height_m") is None:
                        return False, f"architecture.{kind} 缺少 width_m/height_m：{op}"
            frames = {q.name for q in (p / "frames").glob("*.png")} if (p / "frames").is_dir() else set()
            ids = []
            for it in items:
                oid = str(it.get("id") or "").strip()
                if not oid:
                    return False, "有物体缺 id：" + json.dumps(it, ensure_ascii=False)[:200]
                if not isinstance(it.get("count"), int) or int(it.get("count")) < 1:
                    return False, f"{oid} 没写 count（同款实例数，至少 1）"
                dims = it.get("size_m")
                if not (isinstance(dims, list) and len(dims) >= 3 and all(float(v) > 0 for v in dims[:3])):
                    return False, (f"{oid} 没写 size_m=[宽,深,高]（米）；没有尺寸就无法判断它和桌子/架子/"
                                   "其他物品是否重叠，也无法生成正确的模型")
                if not self._placement(it):
                    return False, (f"{oid} 没写位置：用 place.x/y（米）、cell（A1..D4）或 relation"
                                   "（on/near/against + target）；同款多件写 instances 的逐个位置。"
                                   "只写名字不写位置，场景必然摆错。")
                yaw = it.get("pose_yaw_deg")
                if yaw is None and not (it.get("place") and isinstance(it.get("place"), dict)
                                        and ("yaw_deg" in it.get("place") or "yaw" in it.get("place"))):
                    q = (it.get("instances") or [None])[0]
                    if not (isinstance(q, (list, tuple)) and len(q) >= 4) and not (
                            isinstance(q, dict) and ("pose_yaw_deg" in q or "yaw_deg" in q or "yaw" in q)):
                        return False, (f"{oid} 没写 pose_yaw_deg（或 instances 第 4 项 yaw）："
                                       "每件都要估真实朝向；同款多件每件可不同，不能全 0° 摆得整整齐齐")
                if len(it.get("instances") or []) and len(it.get("instances") or []) < int(it.get("count") or 1):
                    return False, f"{oid} count={it.get('count')} 但 instances 只有 {len(it.get('instances') or [])} 条"
                seen = str(it.get("seen_in") or "").strip()
                if not seen:
                    return False, f"{oid} 没写 seen_in（来自哪一帧）；这是你判断它是真实物品的证据"
                if frames and not any(s in seen for s in frames):
                    return False, f"{oid} 的 seen_in={seen!r} 对不上 frames/ 里任何一张帧"
                ids.append(oid)
            known = {str(x.get("id")) for x in items}
            bad_rel = []
            for it in items:
                rel = it.get("relation") or {}
                if isinstance(rel, dict) and rel.get("type") in ("on", "near") and \
                        str(rel.get("target") or "") not in known:
                    bad_rel.append(f"{it.get('id')}->{rel.get('target')}")
            if bad_rel:
                return False, ("relation.target 不存在：" + ", ".join(bad_rel[:6]) +
                               "；on/near 只能指向同一个 plan 里的 id")
            bad_inf = []
            for o in (plan.get("inferred") or []):
                oid = str(o.get("id") or "")
                if not oid.upper().startswith("INF_"):
                    bad_inf.append(oid or "(缺 id)")
                    continue
                basis = str(o.get("basis") or o.get("reason") or o.get("evidence") or "").strip()
                if len(basis) < 4:
                    bad_inf.append(oid + "(缺 basis)")
            if bad_inf:
                return False, ("inferred 必须以 INF_ 开头且写 basis（依据哪一帧的遮挡/延续）——"
                               "没有识别依据、也不是合理脑补的物品不要进场景：" + ", ".join(bad_inf[:6]))
            total = sum(int(x.get("count") or 1) for x in items)
            self.state["plan_objects"] = len(items)
            self.state["plan_instances"] = total
            return True, (f"plan ok（{len(items)} 个 id / {total} 个实例，footprint {len(pts)} 点，"
                          f"{len(arch.get('windows') or [])} 窗 / {len(arch.get('doors') or [])} 门）")

        if stage == "S2_shell":
            shell = self.state.get("shell") or {}
            if not self.state.get("shell_built"):
                return False, "壳体还没建：先 init_scene()，再 build_shell()"
            if not (p / "scene.blend").exists():
                return False, "缺 scene.blend"
            cams = len(shell.get("cameras") or [])
            if cams < 3:
                return False, f"相机只有 {cams} 个（init_scene 应给 1 全景 + 6 轨迹位）"
            good = [q for q in (shell.get("shots") or []) if not q.get("flat")]
            if not good:
                return False, "壳体预览全是平图/没渲出来，先看 build_shell 的 stderr"
            return True, (f"shell ok（{shell.get('meshes', 0)} 网格 / {cams} 相机 / "
                          f"{shell.get('windows', 0)} 窗 / {shell.get('doors', 0)} 门）")

        if stage == "S3_assets":
            assets = self.state.get("assets") or {}
            if not (p / "build_assets.py").exists():
                return False, "缺 build_assets.py —— 先 scaffold_assets() 生成每个物品的构建代码"
            if not assets:
                return False, "还没跑 build_assets —— 建完所有物品后每件都会被临时相机检查"
            if not assets.get("ok"):
                bad = assets.get("bad_objects") or []
                sep = assets.get("separate_parts") or []
                miss = (assets.get("coverage") or {}).get("missing") or []
                short = (assets.get("coverage") or {}).get("short") or []
                bits = []
                if miss: bits.append("缺 id: " + ", ".join(miss[:6]))
                if short: bits.append("数量不足: " + ", ".join(short[:6]))
                if sep: bits.append("部件未合并: " + ", ".join(sep[:6]))
                if bad: bits.append("单件检查失败: " + ", ".join(bad[:6]))
                ov = (assets.get("placement") or {}).get("overlaps") or []
                if ov: bits.append("摆放重叠: " + ", ".join(f"{o.get('a')}×{o.get('b')}" for o in ov[:6]))
                sup = (assets.get("support") or {}).get("failures") or []
                if sup:
                    bits.append("悬空/支撑: " + ", ".join(
                        f"{x.get('id')}:{x.get('type')}" for x in sup[:6]))
                return False, "；".join(bits) or "build_assets 没成功，看 stderr_tail"
            return True, (f"assets ok（{len(assets.get('objects') or [])} 个 id 全部单件、"
                          f"覆盖 {self.state.get('plan_instances')} 个实例）")

        # S4_qa
        if not qa:
            return False, "还没跑 qa_scene —— 全场景 QA 之后才能 finish"
        return self._finish_readiness()

    def _finish_readiness(self) -> tuple[bool, str]:
        qa = self._qa()
        if not qa:
            return False, "qa_scene 没跑过"
        cams = (qa.get("summary") or {}).get("cameras", 0)
        if cams < 3:
            return False, "相机不足（build_scene 应给 1 全景 + ≥2 轨迹位）"
        ok_shots = [r for r in (self.tools.stats.get("renders") or [])
                    if not image_stats(self.root / r).get("flat")]
        if not ok_shots:
            return False, "没有一张非平预览（init_scene → snapshot → 修几何/灯光）"
        hard = {"floor_zero", "solid_envelope", "openings", "camera_fov", "duplicate_objects",
                "plan_coverage", "camera_rig"}
        bad = [e["id"] for e in qa.get("errors", []) if e["id"] in hard]
        if bad:
            return False, "关键检查未过：" + ", ".join(bad)
        sup = (self.state.get("assets") or {}).get("support") or {}
        if sup and not sup.get("ok"):
            fails = sup.get("failures") or []
            return False, "悬空/支撑未过：" + ", ".join(
                f"{x.get('id')}:{x.get('type')}" for x in fails[:6])
        for need in ("SCENE_PLAN.json", "calibration.json"):
            if not (self.project / need).exists():
                return False, f"缺 {need}"
        blend, last = qa.get("blend"), self.state.get("last_blend")
        if blend and last and Path(blend).name != Path(last).name:
            return False, f"qa 检查的 {Path(blend).name} 不是当前 {Path(last).name}"
        soft = [e["id"] for e in qa.get("errors", []) if e["id"] not in hard]
        return True, ("可用场景：相机 %d、非平预览 %d、%s" % (cams, len(ok_shots),
                      ("剩余警告 " + ", ".join(soft)) if soft else "无警告"))

    def _viable(self) -> bool:
        """A scene is worth continuing from only if it has a camera and one non-flat render."""
        if not (self.project / "calibration.json").exists():
            return False
        cams = (self._qa() or {}).get("summary", {}).get("cameras", 0) or len(self.state.get("cameras") or [])
        run_root = getattr(self, "root", self.project)
        good = [r for r in (self.tools.stats.get("renders") or [])
                if not image_stats(run_root / r).get("flat")]
        return bool(cams) and bool(good)

    def advance(self):
        """Forward on a passing gate; backward when an open issue owns an earlier stage."""
        for _ in range(len(STAGES) * 3):
            cur = self.state["stage"]
            owners = [STAGES.index(i["stage"]) for i in self.belief.issues()
                      if i.get("severity", "block") == "block"
                      if i["stage"] in STAGES and STAGES.index(i["stage"]) < STAGES.index(cur)]
            if owners:
                target = STAGES[min(owners)]
                self.log(f"[gate] reopen {cur} → {target} (未关闭: "
                         + ", ".join(i["check"] for i in self.belief.issues(target)) + ")")
                self.state["stage"] = target
                self.state["stage_evidence"].setdefault("reopened", []).append(
                    {"from": cur, "to": target, "by": [i["check"] for i in self.belief.issues(target)]})
                self.history = []
                self.save()
                continue
            if cur == STAGES[-1]:
                break
            budget = int(getattr(self.cfg.quality, "stage_step_budget", 30) or 30)
            spent = int((self.state.get("stage_steps") or {}).get(cur, 0))
            viable = self._viable()
            if spent >= budget + 12 and viable:
                nxt = STAGES[STAGES.index(cur) + 1]
                self.log(f"[gate] SOFT-ADVANCE {cur} → {nxt} after {spent} steps（已有可用相机与渲染，先往下做）")
                self.state["stage_evidence"][cur] = (f"SOFT-ADVANCED after {spent} steps; "
                                                    + str(self.state["stage_evidence"].get(cur, ""))[:120])
                self.state["stage"] = nxt
                self.state["stage_steps"] = {**(self.state.get("stage_steps") or {}), cur: 0}
                self.history = []
                self.save()
                continue
            ok, why = self._gate(cur)
            self.log(f"[gate] {cur}: {'PASS' if ok else 'WAIT'} — {why}")
            if not ok:
                self.state["stage_evidence"][cur] = why
                break
            idx = STAGES.index(cur)
            self.state["stage_evidence"][cur] = why
            self.state["stage"] = STAGES[idx + 1]
            self.history = []
            self.save()

    # ------------------------------------------------------------------ progress
    def _progress_signature(self) -> str:
        qa = self._qa() or {}
        return "|".join(str(x) for x in (
            self.state.get("stage"),
            (self.state.get("stage_evidence") or {}).get(self.state.get("stage")),
            sorted(qa.get("passed") or []),
            len(self.tools.stats.get("renders") or []),
            len(self.tools.stats.get("comparisons") or []),
            len(self.belief.data.get("facts") or {}),
            sorted(i["check"] for i in self.belief.issues()),
        ))

    # ------------------------------------------------------------------ execution
    def _key(self, tool: str, args: dict) -> str:
        return hashlib.sha1((tool + json.dumps(args, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()[:10]

    def _execute(self, c) -> dict[str, Any]:
        key = self._key(c.tool, c.args)
        attempts = self.state.setdefault("attempts", {})
        res = self.tools.call(c.tool, c.args)
        if not res.get("ok"):
            attempts[key] = attempts.get(key, 0) + 1
            res["attempt"] = attempts[key]
            if attempts[key] > 1:
                res["stalled"] = True
                res["fix_hint"] = (str(res.get("fix_hint", "")) + "｜同一调用第 " + str(attempts[key]) +
                                   " 次失败：禁止再试，换方案。" + ALTERNATIVES.get(c.tool, "换工具或换参数"))
            self.state["last_error"] = {"tool": c.tool, "attempts": attempts[key],
                                        "error": str(res.get("error"))[:120], "fix_hint": res.get("fix_hint")}
        else:
            attempts.pop(key, None)
            self.state.pop("last_error", None)

        if c.tool == "qa_scene":
            if res.get("ok"):
                for chk in ("floor_zero", "camera_fov", "scale_sanity", "solid_envelope", "openings",
                            "objects_above_floor", "material_texture_wiring"):
                    self.belief.close_issue(chk)
                summ = res.get("summary") or {}
                if summ.get("hfov_deg"):
                    self.belief.set_camera(hfov_measured=summ["hfov_deg"],
                                           hfov_target=(self.facts.get("intrinsics") or {}).get("hfov_deg"),
                                           height_m=summ.get("camera_height_m"))
                self.state["last_blend"] = res.get("blend") or self.state.get("last_blend")
            else:
                errs = res.get("errors") or []
                for e in errs:
                    sev = "warn" if e.get("id") in ("camera_clearance", "corridor_clear") else "block"
                    self.belief.open_issue(e.get("id", "qa"), e.get("detail", ""), fix_tool="qa_scene",
                                           severity=sev)
                if not errs and res.get("error"):
                    self.belief.open_issue("qa_error", str(res["error"])[:160])
        if c.tool == "render_views":
            if res.get("ok"):
                self.belief.close_issue("flat_render")     # a good batch clears the whole class
            else:
                for name in res.get("flat") or ["render"]:
                    self.belief.open_issue("flat_render", f"{name} 平图（std 过低）", stage="S2_build",
                                           severity="warn")
        if c.tool == "finish" and res.get("ok"):
            ok, why = self._finish_readiness()
            if not ok:
                return {"ok": False, "tool": "finish", "error": "finish 被拒绝：" + why,
                        "fix_hint": "补齐未就绪项；<open_issues> 列着未关闭问题"}
        return res

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        deadline = self.t0 + cfg.max_minutes * 60
        while not self.state.get("done"):
            if self.state["steps"] >= cfg.max_steps:
                self.log("[stop] step budget exhausted")
                break
            if time.time() > deadline:
                self.log("[stop] time budget exhausted")
                break

            for pc in list(self.state.get("pending_calls") or []):
                if pc["tool"] in SAFE_TAIL:
                    r = self.tools.call(pc["tool"], pc.get("args") or {})
                    self.log(f"   [pending] {pc['tool']}: {'OK' if r.get('ok') else 'FAIL'}")
                    if r.get("ok"):
                        self.state["pending_calls"].remove(pc)
            self.save()

            self.advance()
            self.state["steps"] += 1
            st = self.state["stage"]
            self.state["stage_steps"] = {**(self.state.get("stage_steps") or {}),
                                          st: int((self.state.get("stage_steps") or {}).get(st, 0)) + 1}
            self.state["steps_left"] = cfg.max_steps - self.state["steps"]
            self.state["minutes_left"] = round((deadline - time.time()) / 60, 1)
            self.state["finish_ready"] = self._finish_readiness()[0]
            sig = self._progress_signature()
            hist = self.state.setdefault("progress", [])
            hist.append(sig)
            self.state["progress"] = hist[-6:]
            same = 0
            for prev in reversed(self.state["progress"][:-1]):
                if prev != sig:
                    break
                same += 1
            self.state["stall_turns"] = same + (1 if same else 0)
            if self.state["stall_turns"] >= 3:
                self.log(f"[stall] {self.state['stall_turns']} 轮无新证据")
            self.save()

            msgs = self._messages()
            if estimate_chars(msgs) > cfg.context_chars:
                self.state["summary"] = self.llm.summarize(
                    "\n".join(f"{m['role']}: {str(m['content'])[:3000]}" for m in msgs[-8:]))
                self.history = []
                msgs = self._messages()
                self.log("[memory] compressed (belief 不受影响)")

            try:
                reply = self.llm.chat(msgs, self._images())
            except LLMError as exc:
                self.log(f"[llm] {exc}")
                time.sleep(5)
                continue

            try:
                calls = parse_calls(reply, self.legal)
                if self.state["stage"] == "S1_recon":
                    allowed = {"write_file", "patch_file", "note", "list_files"}
                    calls = [c for c in calls if c.tool in allowed]
                    if not calls:
                        self.history.append({"role": "assistant", "content": reply[:2000]})
                        self.history.append({"role": "user", "content":
                            "S1 只能写 SCENE_PLAN.json。图片已全部随请求发送；不要再抽帧/读文件。" +
                            prompts.GATE_ACTIONS.get("S1_recon", "")})
                        self.log("[s1] refused sampling; must write plan")
                        continue
                if int(self.state.get("stall_turns") or 0) >= 6:
                    calls = [c for c in calls if c.tool not in ("read_file", "prepare_inputs", "image_stats")]
                    if not calls:
                        self.history.append({"role": "assistant", "content": reply[:2000]})
                        self.history.append({"role": "user", "content":
                            "硬停生效：只读调用被拒绝。现在只能做本阶段该做的事：" +
                            prompts.GATE_ACTIONS.get(self.state["stage"], "")})
                        self.log("[stall] refused read-only batch")
                        continue
            except ProtocolError as exc:
                self.log(f"[protocol] {exc}")
                self.history.append({"role": "assistant", "content": reply[:2000]})
                self.history.append({"role": "user", "content": prompts.repair_prompt(str(exc), sorted(self.legal))})
                continue

            self.log(f"\n[step {self.state['steps']}] batch {len(calls)}: " + ", ".join(c.tool for c in calls))
            if calls[0].thought:
                self.log(f"   thought: {calls[0].thought[:180]}")

            results, failure, skipped = [], None, []
            for i, c in enumerate(calls):
                res = self._execute(c)
                self.log(f"   -> {c.tool}: {'OK' if res.get('ok') else 'FAIL'}  {self._short(res)}")
                results.append({"tool": c.tool, "ok": bool(res.get("ok")), "result": self._trim(res)})
                if not res.get("ok"):
                    failure = c.tool
                    skipped = [{"tool": x.tool, "args": x.args} for x in calls[i + 1:]]
                    self.state.setdefault("pending_calls", []).extend(skipped)
                    break
            if failure:
                self.log(f"   (batch stopped at {failure}; queued: {[s['tool'] for s in skipped]})")

            if st == "S1_recon" and any(r.get("tool") == "video_frames" and r.get("ok") for r in results):
                self.state["s1_frames_seen"] = True
            self.history.append({"role": "assistant", "content": reply[:3000]})
            self.history.append({"role": "user", "content": f"执行结果（本批 {len(results)} 项）：\n"
                                                          + json.dumps(results, ensure_ascii=False)})
            done = [r for r in results if r["tool"] == "finish" and r["ok"]]
            if done:
                self.state["done"] = True
                self.state["finish"] = done[0]["result"]
                self.state["finished"] = True
                self.save()
                self.log("\n[done] " + json.dumps(self.state["finish"], ensure_ascii=False)[:400])
                break
            self.save()

        self.state["elapsed_min"] = round((time.time() - self.t0) / 60, 1)
        self.state["llm_calls"] = self.llm.calls
        self.state["tokens"] = {"prompt": self.llm.prompt_tokens, "completion": self.llm.completion_tokens}
        self.save()
        return self.state

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _short(result: dict[str, Any]) -> str:
        if not result.get("ok") and result.get("error"):
            return str(result["error"])[:200]
        for key in ("shots", "files", "frames", "checks", "errors", "summary"):
            if key in result:
                return f"{key}: {json.dumps(result[key], ensure_ascii=False)[:180]}"
        return json.dumps(result, ensure_ascii=False)[:200]

    @staticmethod
    def _trim(result: dict[str, Any]) -> dict[str, Any]:
        """Pass the tool result through unchanged. Earlier versions clipped stdout/content to a few
        hundred chars, which hid the very diagnostic the model needed to fix the call."""
        out = dict(result)
        out.pop("project_tree", None)      # duplicated by <project_layout> every turn
        return out
