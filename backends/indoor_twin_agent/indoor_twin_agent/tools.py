"""Tool layer: the executable surface the model is allowed to touch.

Every tool returns a JSON-able dict. The agent turns it into (a) a short text result and
(b) optional images that get re-attached to the next request.

Nothing here calls an external service; the only network call in the whole project is the LLM.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .executor import ExecError, Executor


# --------------------------------------------------------------------------- image maths
def image_stats(path: Path) -> dict[str, Any]:
    try:
        im = Image.open(path).convert("RGB")
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": str(exc)}
    a = np.asarray(im).astype(np.float32)
    g = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    if g.shape[1] > 1:
        gx[:, :-1] = np.abs(np.diff(g, axis=1))
    if g.shape[0] > 1:
        gy[:-1, :] = np.abs(np.diff(g, axis=0))
    mag = gx + gy                       # per-pixel structure, resolution independent
    return {
        "ok": True, "path": str(path), "size": [im.width, im.height],
        "mean": round(float(g.mean()), 1), "std": round(float(g.std()), 1),
        "p2": round(float(np.percentile(g, 2)), 1), "p98": round(float(np.percentile(g, 98)), 1),
        "high_frac": round(float((g > 245).mean()), 3), "low_frac": round(float((g < 12).mean()), 3),
        "edge_frac": round(float((mag > 14).mean()), 3),
        # v5: an early-stage overview legitimately has low global contrast (floor + guide walls).
        # Structure (edges) is the reliable signal; a real flat/blank render also has almost none.
        "flat": bool(g.std() < 8 or (mag > 14).mean() < 0.02),
    }


def compare_metrics(src: Path, render: Path) -> dict[str, Any]:
    try:
        s = Image.open(src).convert("RGB")
        r = Image.open(render).convert("RGB").resize(s.size)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    sa = np.asarray(s.resize((160, 120))).astype(np.float32)
    ra = np.asarray(r.resize((160, 120))).astype(np.float32)
    l1 = 0.0
    for i in range(3):
        h1 = np.histogram(sa[..., i], bins=16, range=(0, 255))[0].astype(np.float32)
        h2 = np.histogram(ra[..., i], bins=16, range=(0, 255))[0].astype(np.float32)
        h1 /= h1.sum() + 1e-6
        h2 /= h2.sum() + 1e-6
        l1 += float(np.abs(h1 - h2).sum()) / 3
    notes = []
    if l1 > 0.35:
        notes.append("直方图差 %.2f：整体色调/曝光偏离明显" % l1)
    if abs(ra.mean() - sa.mean()) / (sa.mean() + 1e-6) > 0.5:
        notes.append("平均亮度差 %.1f%%" % (100 * abs(ra.mean() - sa.mean()) / (sa.mean() + 1e-6)))
    return {"ok": True, "hist_l1": round(l1, 3),
            "brightness_src": round(float(sa.mean()), 1), "brightness_render": round(float(ra.mean()), 1),
            "edge_src": image_stats(src)["edge_frac"], "edge_render": image_stats(render)["edge_frac"],
            "notes": notes}


def side_by_side(src: Path, render: Path, out: Path, label_l: str = "video", label_r: str = "twin") -> Path:
    s = Image.open(src).convert("RGB")
    r = Image.open(render).convert("RGB")
    h = 480
    def fit(im):
        w = int(im.width * h / im.height)
        return im.resize((w, h))
    s, r = fit(s), fit(r)
    canvas = Image.new("RGB", (s.width + r.width, h + 28), (18, 18, 18))
    canvas.paste(s, (0, 28))
    canvas.paste(r, (s.width, 28))
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), label_l, fill=(235, 235, 235))
    d.text((s.width + 8, 8), label_r, fill=(235, 235, 235))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


# --------------------------------------------------------------------------- tool set
def _json_after(text: str, prefix: str) -> dict:
    if prefix not in text:
        return {}
    try:
        return json.loads(text.split(prefix, 1)[1].split("\n", 1)[0])
    except Exception:
        return {}


def contact_sheet(paths: list[Path], out: Path, cols: int = 4, tile=(320, 240)) -> Path:
    """One image the model can actually look at: every checked item, labelled by id."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        return out
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile[0], rows * tile[1]), (22, 22, 22))
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail(tile)
        except Exception:
            continue
        x, y = (i % cols) * tile[0], (i // cols) * tile[1]
        canvas.paste(im, (x + (tile[0] - im.width) // 2, y + (tile[1] - im.height) // 2))
        d.rectangle((x, y, x + tile[0] - 1, y + tile[1] - 1), outline=(60, 60, 60))
        d.text((x + 6, y + 4), p.stem[:28], fill=(255, 235, 120))
    canvas.save(out)
    return out


class Tools:

    def __init__(self, root: Path, executor: Executor, lib_dir: Path, quality, state: dict,
                 ledger_path: Path, on_note: Callable[[str], None] | None = None):
        self.root = Path(root)
        self.ex = executor
        self.lib_dir = Path(lib_dir)
        self.quality = quality
        self.state = state
        self.ledger_path = Path(ledger_path)
        self.on_note = on_note
        self.stats: dict[str, Any] = {
            "renders": [], "comparisons": [], "qa": None, "qa_blend": None, "qa_stamp": None,
            "writes": 0, "patches": 0, "failures": 0,
        }

    # ------------------------------------------------------------------ schema
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": "list_files", "args": {"glob": "**/*"}, "help": "列出 run 目录下文件（相对路径 + KB）"},
            {"name": "read_file", "args": {"path": "", "max_chars": 12000}, "help": "读文本文件"},
            {"name": "write_file", "args": {"path": "", "content": ""}, "help": "新建/覆盖文件（仅新文件优先使用）"},
            {"name": "patch_file", "args": {"path": "", "old": "", "new": "", "expected_count": 1},
             "help": "定点替换；old 必须唯一出现 expected_count 次，否则拒绝（防误改）"},
            {"name": "run_shell", "args": {"argv": ["python3", "-c", "..."], "timeout": 180},
             "help": "执行 argv（白名单：python3/blender/ffmpeg/ffprobe）"},
            {"name": "blender_py", "args": {"script": "", "args": [], "blend": "", "timeout": 300},
             "help": "在 Blender 里跑脚本；不传 blend 则从空场景开始；工作目录=run/project"},
            {"name": "init_scene", "args": {"rebuild": False},
             "help": "运行时建好舞台：Floor(z=0) + Cam_Overview + Cam_Traj_01..06（沿路径 6 个静止机位），存 scene.blend + 一张全景图。v7 的 build_scene 会自动先调它。"},
            {"name": "scaffold_build", "args": {"overwrite": False},
             "help": "按 SCENE_PLAN 生成完整可运行的 build_all.py 初版（壳体 + 每个 id/count 的初版几何 + 材质 + 灯）"},
            {"name": "build_scene", "args": {"script": "build_all.py", "rebuild": False},
             "help": "执行 build_all.py：先 init 相机/地面，再建壳体与全部物体，保存并渲 1 全景 + 5 轨迹位（v7 快速路径，v8 用 build_shell + build_assets）"},
            {"name": "scaffold_assets", "args": {"overwrite": False},
             "help": "按 plan 的 place/cell/instances 生成 build_assets.py：每个 id 一个类别化单件（沙发/屏障/桌/椅/显示器/架/球/垫/机器人），部件 join 成一个 mesh"},
            {"name": "build_assets", "args": {"script": "build_assets.py"},
             "help": "在已有壳体上执行 build_assets.py；保存后给每个 id 建临时相机逐件渲染、检查是否单件，再删相机；返回 objects_sheet.png"},
            {"name": "inspect_object", "args": {"id": "O1_BlueBarrier"},
             "help": "只重渲一个物品：临时相机取景、渲染、立刻删相机，返回单件/多件与预览图"},
            {"name": "join_asset", "args": {"id": "O1_BlueBarrier"},
             "help": "兜底：把一个物品散落的 mesh 子件 join 成一个 mesh（正常 build_assets 已自动 join）"},
            {"name": "build_shell", "args": {"rebuild": False},
             "help": "按 calibration 的 bbox + SCENE_PLAN 的门窗描述，确定性地建 Floor/Ceiling/四面墙/真洞口/窗框玻璃/门/代理外景/默认灯，并渲 3 张"},
            {"name": "snapshot", "args": {"tag": "s", "res": 640},
             "help": "用现有两个相机渲 1 张全景 + 2 张轨迹位姿（会顺手清重复物体），返回每张 std/mean。每个阶段都该调一次。"},
            {"name": "material_audit", "args": {},
             "help": "列出每个材质的 Base Color 是否接线、被哪些物体使用、有哪些无材质物体"},
            {"name": "render_views", "args": {"blend": "", "views": [{"name": "", "loc": [], "target": [], "lens": 0}],
                                              "res": 960, "samples": 24},
             "help": "渲染多机位，返回每张图 std/edge/mean（flat 即失败）"},
            {"name": "image_stats", "args": {"paths": []}, "help": "图像统计，不看图也能判断空/暗/平"},
            {"name": "video_frames", "args": {"count": 6, "times": [], "fps": 0, "res": 672},
             "help": "从原视频抽帧（count=均匀抽，times=指定秒，fps=逐秒连续下采样，如 5）"},
            {"name": "compare", "args": {"video_frame": "", "render": "", "out": "previews/compare_01.png"},
             "help": "原视频帧 vs 渲染并排 + 直方图/亮度指标"},
            {"name": "prepare_inputs", "args": {"recompute": True},
             "help": "重新解析视频/轨迹/内参并重写 calibration.json（重力对齐/路径/FOV），返回 INPUT_FACTS"},
            {"name": "qa_scene", "args": {"blend": "", "texture_mode": "", "intrinsics": ""},
             "help": "几何/材质断言：floor/厚度/洞口/相机/贴图接线/尺度"},
            {"name": "record_object", "args": {"id": "", "kind": "", "dims_m": [0, 0, 0],
                                              "evidence": "frame t=3.2s, 高约 96px", "confidence": 0.6, "note": ""},
             "help": "把一个物体写进结构化 belief（尺寸/来源帧/置信度）；S4 阶段要求登记足够物体"},
            {"name": "note", "args": {"text": ""}, "help": "写进证据台账（只写被数字或图像证明的事实）"},
            {"name": "finish", "args": {"summary": "", "blend": "", "glb": ""}, "help": "交付；不满足门槛会被拒绝"},
        ]

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return self._fail(name, args, f"tool {name} not implemented",
                              "只允许调用工具表里的工具")
        try:
            res = fn(**args)
        except TypeError as exc:
            return self._fail(name, args, f"bad arguments for {name}: {exc}",
                              f"按 schema 传参：{json.dumps([x for x in self.schemas() if x['name'] == name], ensure_ascii=False)}")
        except ExecError as exc:
            return self._heal_path_error(name, args, str(exc))
        except Exception as exc:  # never kill the loop because a tool raised
            self.stats["failures"] += 1
            return self._fail(name, args, f"{type(exc).__name__}: {exc}", "读报错，改输入或改用别的工具")

        # mechanical self-heal for the most common model mistake: a path that is off by a directory
        if not res.get("ok"):
            hint = res.get("error", "")
            if isinstance(hint, str) and ("not found" in hint or "escapes" in hint):
                return self._heal_path_error(name, args, hint)
        return res

    def _heal_path_error(self, name: str, args: dict[str, Any], err: str) -> dict[str, Any]:
        """If a path argument names an existing file under another directory, rewrite it once and
        retry, telling the model what we changed. Otherwise return a structured fix hint."""
        cand = self._find_by_basename(args)
        if cand:
            fixed = dict(args)
            for k, v in args.items():
                if isinstance(v, str) and v and os.path.basename(v.strip('"\'')) in cand:
                    fixed[k] = cand[os.path.basename(v.strip('"\''))]
                    break
            try:
                res = getattr(self, f"t_{name}")(**fixed)
                if res.get("ok"):
                    res["autocorrected"] = {"from": args, "to": fixed,
                                            "why": "该文件在项目里只有这个位置，已自动改写路径"}
                    self.stats.setdefault("heals", 0)
                    self.stats["heals"] += 1
                    return res
            except Exception:
                pass
        return self._fail(name, args, err, "路径必须相对 project/ 根；先用 list_files 确认存在",
                          extra={"available_scripts": self._scripts(), "available_blends": self._blends()})

    def _find_by_basename(self, args: dict[str, Any]) -> dict[str, str]:
        picked: dict[str, str] = {}
        names = {os.path.basename(str(v).strip('"\'')) for v in args.values()
                 if isinstance(v, str) and v.strip() and "\n" not in v}
        for f in self.root.rglob("*"):
            if f.is_file() and "_tmp" not in f.parts and f.name in names:
                picked[f.name] = str(f.relative_to(self.root))
        return picked

    def _scripts(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*.py")
                      if "_lib" not in p.parts and "_tmp" not in p.parts)[:20]

    def _blends(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*.blend"))[:10]

    def _fail(self, name: str, args: dict[str, Any], err: str, fix_hint: str,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self.stats["failures"] += 1
        out = {"ok": False, "tool": name, "args": args, "error": err, "fix_hint": fix_hint,
               "project_root": str(self.root), "project_tree": self.tree_overview()}
        if extra:
            out.update(extra)
        return out

    def tree_overview(self, limit: int = 60) -> dict[str, Any]:
        """A compact, always-useful map of the run directory: dirs, key files, scripts, blends."""
        dirs, files = {}, []
        for p in sorted(self.root.rglob("*")):
            if "_tmp" in p.parts or "__pycache__" in p.parts:
                continue
            if p.is_dir():
                dirs[str(p.relative_to(self.root))] = len(list(p.glob("*")))
            elif p.is_file():
                files.append(str(p.relative_to(self.root)))
        return {"root": str(self.root), "dirs": dirs, "files": files[:limit]}

    # ------------------------------------------------------------------ files
    def t_list_files(self, glob: str = "**/*"):
        files = []
        for p in sorted(self.root.glob(glob)):
            if p.is_file() and "_tmp" not in p.parts and ".git" not in p.parts:
                files.append({"path": str(p.relative_to(self.root)), "kb": round(p.stat().st_size / 1024, 1)})
        return {"ok": True, "files": files[:200], "count": len(files)}

    def t_read_file(self, path: str, max_chars: int = 12000):
        p = self.ex.resolve(path, must_exist=True)
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".blend", ".zip", ".glb"):
            return {"ok": False, "error": f"{path} 是二进制/图片，不是文本；read_file 不能把它变成文字",
                    "fix_hint": ("图片已经随每轮请求直接发给多模态模型（S1 会跨时间轴附 4–6 帧）。"
                                 "要看视频就调用 video_frames(times=[...])；要写计划就直接 write_file "
                                 "SCENE_PLAN.json。不要再 read_file 图片。")}
        txt = p.read_text(encoding="utf8", errors="replace")
        if "\x00" in txt[:4096] or sum(1 for c in txt[:4096] if ord(c) < 9) > 40:
            return {"ok": False, "error": f"{path} 看起来是二进制，拒绝当文本读",
                    "fix_hint": "图片/二进制不要 read_file；图片由多模态通道发送。"}
        return {"ok": True, "path": path, "chars": len(txt),
                "content": txt[:max_chars], "truncated": len(txt) > max_chars}

    def t_write_file(self, path: str, content: str):
        p = self.ex.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf8")
        self.stats["writes"] += 1
        return {"ok": True, "path": path, "kb": round(len(content) / 1024, 1)}

    def t_patch_file(self, path: str, old: str, new: str, expected_count: int = 1):
        p = self.ex.resolve(path, must_exist=True)
        txt = p.read_text(encoding="utf8")
        n = txt.count(old)
        if n != expected_count:
            return {"ok": False, "error": f"anchor found {n} times, expected {expected_count}; "
                                         f"paste a longer unique snippet or write_file a new module"}
        p.write_text(txt.replace(old, new, expected_count), encoding="utf8")
        self.stats["patches"] += 1
        return {"ok": True, "path": path, "replacements": n}

    # ------------------------------------------------------------------ execution
    def t_run_shell(self, argv: list[str], timeout: int = 180, cwd: str = "."):
        return self.ex.run(argv, timeout=timeout, cwd=cwd)

    def _stage_code(self, name: str, body: str) -> str:
        self._ensure_lib()
        path = self.root / "_tmp" / f"{name}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(body, encoding="utf8")
        return str(path.relative_to(self.root))

    def t_init_scene(self, rebuild: bool = False):
        """Cameras and the walking surface are executor infrastructure. v4.x left them to the model
        and 60+ steps produced zero cameras, so nothing could render or compare."""
        self._ensure_lib()
        lib = str((self.root / "_lib"))
        code = (
            "import sys, json\n"
            f"sys.path.insert(0, {lib!r})\n"
            "import bs_stage\n"
            f"bs_stage.init({str((self.root / 'calibration.json'))!r}, "
            f"{str((self.root / 'scene.blend'))!r}, "
            f"{str((self.root / 'previews' / '00_overview.png'))!r}, rebuild={bool(rebuild)!r})\n"
        )
        res = self.ex.blender_py(self._stage_code("init_scene", code), timeout=300)
        out = {"ok": bool(res.get("ok")), "stdout_tail": (res.get("stdout") or "")[-1500:],
               "stderr_tail": (res.get("stderr") or "")[-900:]}
        for line in (res.get("stdout") or "").splitlines():
            if line.startswith("STAGE_JSON:"):
                try:
                    stage = json.loads(line.split(":", 1)[1])
                    out["cameras"] = stage.get("cameras") or []
                    self.state["cameras"] = out["cameras"]
                except Exception:
                    pass
        if res.get("ok"):
            self.state["last_blend"] = "scene.blend"
            prev = self.root / "previews" / "00_overview.png"
            if prev.exists():
                st = image_stats(prev)
                self.stats["renders"].append(str(prev.relative_to(self.root)))
                out.update({"blend": "scene.blend", "preview": str(prev.relative_to(self.root)), "stats": st})
                if st.get("flat"):
                    out["warning"] = "全景图是平的：检查灯光/世界，或场景里没有几何"
        else:
            out["error"] = "init_scene 失败（Blender 报错见 stderr_tail）"
            out["fix_hint"] = "先 prepare_inputs 生成 calibration.json；再 init_scene(rebuild=True)"
        return out

    def t_scaffold_build(self, overwrite: bool = False):
        """Write a complete first-pass build_all.py from SCENE_PLAN.

        v6 made the model hand-derive a structured plan first and only then build; 17 steps went
        into polishing a door position and no scene existed. v7 starts from running code: every
        planned id/count is already in the script, QA then points at what is wrong.
        """
        self._ensure_lib()
        plan_path = self.root / "SCENE_PLAN.json"
        out_path = self.root / "build_all.py"
        if not plan_path.exists():
            return {"ok": False, "error": "缺 SCENE_PLAN.json",
                    "fix_hint": "先抽帧看懂空间，写出 objects(id,count)"}
        if out_path.exists() and not overwrite:
            return {"ok": False, "error": "build_all.py 已存在",
                    "fix_hint": "用 patch_file 改细节；确实要重生成就 scaffold_build(overwrite=True)"}
        plan = json.loads(plan_path.read_text())
        items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
        cal = json.loads((self.root / "calibration.json").read_text()) if (self.root / "calibration.json").exists() else {}
        bbox = [float(v) for v in (cal.get("room_bbox_m") or [-3, 3, -4, 4, 2.7])]

        # slots along the room perimeter, 1.1 m apart, 0.75 m inset: every instance sits outside
        # the camera corridor, so a first pass cannot block the trajectory.
        slots, step = [], 1.1
        x0, x1, y0, y1 = bbox[0] + .75, bbox[1] - .75, bbox[2] + .75, bbox[3] - .75
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            n = max(1, int(math.dist(a, b) / step))
            for i in range(n):
                t = i / n
                slots.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))

        default_dims = {"sofa": (1.9, .85, .8), "barrier": (1.1, .45, 1.7), "desk": (1.4, .7, .75),
                        "chair": (.6, .6, 1.0), "monitor": (.6, .2, .45), "shelf": (.9, .45, 1.8),
                        "cushion": (.6, .5, .25), "ball": (.65, .65, .65), "cart": (.6, .5, .9),
                        "light": (1.2, .12, .08)}
        specs, cur = [], 0
        for it in items:
            oid = str(it.get("id") or "").strip()
            if not oid:
                continue
            cnt = max(1, int(it.get("count") or 1))
            dims = it.get("size_m") or []
            if not (isinstance(dims, list) and len(dims) >= 3):
                name = (str(it.get("name") or "") + " " + oid).lower()
                dims = next((v for k, v in default_dims.items() if k in name), (0.9, .6, .9))
            placed = []
            for _ in range(cnt):
                if cur >= len(slots):
                    cur = 0
                placed.append([round(slots[cur][0], 2), round(slots[cur][1], 2)])
                cur += 1
            specs.append({"id": oid, "dims": [float(x) for x in dims[:3]], "count": cnt,
                          "positions": placed, "kind": str(it.get("name") or oid)})
        body_json = json.dumps(specs, ensure_ascii=False, indent=1)

        template = '''"""v7 first-pass build. Generated by scaffold_build from SCENE_PLAN.

Every planned id and `count` is already here, so QA can tell you exactly what is missing.
Edit order: dimensions/placement -> per-object parts (replace `build_one`) -> materials.
Never delete ids: `plan_coverage` will name them.
"""
import sys, math
sys.path.insert(0, __LIB__)
import bpy
import bs_geo
import bs_mat
import bs_layout

ROOT = __ROOT__
CAL = ROOT + "/calibration.json"
PLAN = ROOT + "/SCENE_PLAN.json"
BLEND = ROOT + "/scene.blend"

# 0) idempotent: drop last run's objects for every planned id (build_scene may run many times)
IDS = __IDS__
for o in list(bpy.data.objects):
    if any(o.name == i or o.name.startswith(i + "_") or o.name.startswith(i + "__")
           or o.name.startswith(i + ".") for i in IDS):
        bpy.data.objects.remove(o, do_unlink=True)

# 1) deterministic shell: walls / real openings / ceiling / windows / door / default light
bs_layout.build(CAL, PLAN, BLEND, ROOT + "/previews", render=False)

# 2) one semantic material set, each with generated maps (S3 refines tint/rough)
_KIND = [
    ("fabric_blue", ("blue", "barrier"), "fabric", (0.10, 0.24, 0.62)),
    ("fabric_warm", ("sofa", "cushion", "bean"), "fabric", (0.66, 0.34, 0.16)),
    ("wood", ("desk", "table", "wood", "shelf"), "wood", (0.47, 0.31, 0.18)),
    ("metal", ("metal", "cart", "robot", "frame"), "metal", (0.55, 0.56, 0.58)),
    ("screen", ("monitor", "screen", "computer"), "concrete", (0.03, 0.03, 0.04)),
    ("paint", (), "paint", (0.78, 0.76, 0.71)),
]
_MAPS = {}

def mat_for(kind):
    k = kind.lower()
    if "light" in k or "lamp" in k:
        return bs_mat.emissive("M_Lamp", (1.0, 0.96, 0.88, 1), strength=2.4)
    key = next((n for n, words, _, _ in _KIND if any(w in k for w in words)), "paint")
    if key not in _MAPS:
        spec = next(x for x in _KIND if x[0] == key)
        _MAPS[key] = bs_mat.make_maps("mat_" + key, spec[2], ROOT + "/textures", 512, 0, spec[3])
    spec = next(x for x in _KIND if x[0] == key)
    return bs_mat.with_maps("M_" + key, _MAPS[key], uv_scale=1.5, tint=spec[3])

SPECS = __SPECS__

def build_one(name, loc, dims, mat):
    w, d, h = dims
    root = bs_geo.asset_root(name, (loc[0], loc[1], 0))
    legs = []
    if h > 0.45:
        leg_h = max(0.08, h * 0.22)
        for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
            legs.append(bs_geo.box(name + "_leg%d" % i, (0.06, 0.06, leg_h),
                                   loc=(sx * (w / 2 - 0.08), sy * (d / 2 - 0.08), leg_h / 2), mat=mat))
    base_z = max(0.0, h * 0.22)
    parts = legs + [
        bs_geo.box(name + "_body", (w * 0.92, d * 0.88, h * 0.70),
                   loc=(0, 0, base_z + h * 0.35), mat=mat),
        bs_geo.box(name + "_top", (w, d, max(0.03, h * 0.06)),
                   loc=(0, 0, h - max(0.015, h * 0.03)), mat=mat),
    ]
    bs_geo.parent_all(root, parts)
    return root

for spec in SPECS:
    oid, dims, poss = spec["id"], spec["dims"], spec["positions"]
    m = mat_for(spec["kind"])
    build_one(oid, poss[0], dims, m)
    for i in range(1, len(poss)):
        off = (poss[i][0] - poss[0][0], poss[i][1] - poss[0][1], 0)
        bs_geo.clone_asset(oid, "%s__%02d" % (oid, i + 1), offset=off)
print("SCAFFOLD_BUILT", len(SPECS), "ids")
'''
        script = (template.replace("__LIB__", repr(str(self.root / "_lib")))
                  .replace("__ROOT__", repr(str(self.root)))
                  .replace("__IDS__", json.dumps([x["id"] for x in specs], ensure_ascii=False))
                  .replace("__SPECS__", body_json))
        out_path.write_text(script, encoding="utf8")
        self.state["scaffold"] = "build_all.py"
        return {"ok": True, "path": "build_all.py", "ids": len(specs),
                "instances": sum(x["count"] for x in specs),
                "kb": round(out_path.stat().st_size / 1024, 1),
                "next": "直接 build_scene()；初版出来后按 QA 点名改细节"}

    def t_build_scene(self, script: str = "build_all.py", rebuild: bool = False):
        """init cameras/floor -> run build_all.py -> save + 1 overview + 5 trajectory renders."""
        self._ensure_lib()
        sp = self.root / script
        if not sp.exists():
            return {"ok": False, "error": "缺 %s" % script, "fix_hint": "先 scaffold_build() 生成初版脚本"}
        for old in (self.root / "previews").glob("build_*.png"):
            old.unlink(missing_ok=True)
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n" % str(self.root / "_lib")
            + "import bs_stage, bpy\n"
            + "bs_stage.init(%r, %r, %r, rebuild=%r)\n" % (
                str(self.root / "calibration.json"), str(self.root / "scene.blend"),
                str(self.root / "previews" / "00_build.png"), bool(rebuild))
            + "exec(compile(open(%r).read(), %r, 'exec'), {'__name__': '__main__'})\n" % (str(sp), str(sp))
            + "bs_stage.finalize(%r, %r, res=480, traj_limit=0)\n" % (
                str(self.root / "scene.blend"), str(self.root / "previews"))
        )
        res = self.ex.blender_py(self._stage_code("build_scene", code), timeout=900)
        shots, flat, build = [], [], {}
        out = res.get("stdout") or ""
        if "BUILD_JSON:" in out:
            try:
                build = json.loads(out.split("BUILD_JSON:", 1)[1].split("\n", 1)[0])
            except Exception:
                build = {}
        for p in sorted((self.root / "previews").glob("build_*.png")):
            st = image_stats(p)
            rel = str(p.relative_to(self.root))
            self.stats["renders"].append(rel)
            shots.append({"path": rel, "mean": st.get("mean"), "std": st.get("std"),
                          "edge_frac": st.get("edge_frac"), "flat": st.get("flat")})
            if st.get("flat"):
                flat.append(rel)
        summary = {"ok": bool(res.get("ok")) and bool(shots), "script": script,
                   "meshes": build.get("meshes"), "cameras": build.get("cameras"),
                   "shots": shots, "flat": flat,
                   "duplicates_removed": build.get("duplicates_removed")}
        if res.get("ok") and shots:
            self.state["last_blend"] = "scene.blend"
            self.state["cameras"] = build.get("cameras") or []
            self.state["build"] = summary
        else:
            summary["error"] = "build_all.py 执行失败"
            summary["stderr_tail"] = (res.get("stderr") or "")[-3000:]
            summary["stdout_tail"] = (res.get("stdout") or "")[-1500:]
        return summary

    def _grid_xy(self, bbox, cell):
        """`cell` is A-D / 1-4 over the room bbox: A1 = south-west, D4 = north-east."""
        col = "ABCD".index(str(cell)[0].upper()) if str(cell)[0].upper() in "ABCD" else 1
        row = int(str(cell)[1]) if len(str(cell)) > 1 and str(cell)[1].isdigit() else 2
        row = max(1, min(4, row))
        x0, x1, y0, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        x = x0 + (col + 0.5) * (x1 - x0) / 4.0
        y = y0 + (row - 0.5) * (y1 - y0) / 4.0
        return round(x, 2), round(y, 2)

    def _place_of(self, item, bbox):
        """Plan authoring is in metres OR in 4x4 cells; cells are easier to infer from frames."""
        if isinstance(item, dict) and "x" in item and "y" in item:
            return (float(item["x"]), float(item["y"]),
                    float(item.get("pose_yaw_deg") or item.get("yaw_deg") or 0))
        p = item.get("place") or {}
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            return float(p[0]), float(p[1]), float(item.get("pose_yaw_deg") or item.get("yaw_deg") or 0)
        if isinstance(p, dict) and "x" in p and "y" in p:
            return float(p["x"]), float(p["y"]), float(p.get("pose_yaw_deg") or p.get("yaw_deg") or
                                                        item.get("pose_yaw_deg") or item.get("yaw_deg") or 0)
        if isinstance(p, dict) and p.get("cell"):
            x, y = self._grid_xy(bbox, p["cell"])
            return x, y, float(p.get("pose_yaw_deg") or p.get("yaw_deg") or
                               item.get("pose_yaw_deg") or item.get("yaw_deg") or 0)
        if item.get("cell"):
            x, y = self._grid_xy(bbox, item["cell"])
            return x, y, float(item.get("pose_yaw_deg") or item.get("yaw_deg") or 0)
        return None

    def _room_bbox(self, plan):
        """Room size comes from the S1 plan (whole-video estimate), NOT from trajectory length."""
        arch = plan.get("architecture") or {}
        fp = arch.get("footprint_m") or arch.get("floorplan_m") or arch.get("outline_m")
        if isinstance(fp, list) and len(fp) >= 3:
            try:
                xs = [float(q[0]) for q in fp]; ys = [float(q[1]) for q in fp]
                return [min(xs), max(xs), min(ys), max(ys)]
            except Exception:
                pass
        b = arch.get("room_bbox_m") or arch.get("bbox_m") or arch.get("room_size_m")
        if isinstance(b, dict):
            b = [b.get("x0", -3), b.get("x1", 3), b.get("y0", -4), b.get("y1", 4)]
        if not b:
            cal = json.loads((self.root / "calibration.json").read_text()) if (self.root / "calibration.json").exists() else {}
            b = cal.get("room_bbox_m") or [-3, 3, -4, 4, 2.7]
        b = [float(v) for v in b]
        if len(b) >= 4:
            x0, x1, y0, y1 = b[0], b[1], b[2], b[3]
            if x1 - x0 < 1.0 or y1 - y0 < 1.0:
                x0, x1, y0, y1 = -3, 3, -4, 4
        else:
            x0, x1, y0, y1 = -3, 3, -4, 4
        return [x0, x1, y0, y1]

    def t_scaffold_assets(self, overwrite: bool = False):
        """Write a v9 build_assets.py: plan -> bs_assets category templates.

        New: 30+ semantic templates; relation (on/near/against) uses item size and support top;
        collision pass nudges same-level AABB overlaps; materials are semantic presets.
        """
        self._ensure_lib()
        plan_path = self.root / "SCENE_PLAN.json"
        out_path = self.root / "build_assets.py"
        if not plan_path.exists():
            return {"ok": False, "error": "缺 SCENE_PLAN.json", "fix_hint": "先把每一帧里的物品清点成 objects"}
        if out_path.exists() and not overwrite:
            return {"ok": False, "error": "build_assets.py 已存在",
                    "fix_hint": "用 patch_file 改 SPECS/relation；确实要重生成就 overwrite=True"}
        plan = json.loads(plan_path.read_text())
        items = list(plan.get("objects") or []) + list(plan.get("inferred") or [])
        bbox = self._room_bbox(plan)

        defaults = {
            "sofa": (2.0, .9, .8), "barrier": (1.2, .25, 1.6), "desk": (1.4, .7, .75),
            "table": (1.2, .8, .75), "dining_table": (1.6, .9, .75), "bench": (1.4, .45, .45),
            "task_chair": (.65, .65, 1.12), "office_chair": (.65, .65, 1.12),
            "chair": (.6, .6, 1.0), "stool": (.45, .45, .55),
            "monitor": (.55, .22, .45), "laptop": (.38, .26, .03),
            "keyboard": (.45, .16, .03), "mouse": (.10, .16, .06),
            "pc": (.22, .48, .46), "computer": (.22, .48, .46),
            "fridge": (.75, .72, 1.85), "cabinet": (.9, .45, .75),
            "shelf": (.9, .42, 1.9), "box": (.45, .40, .35), "cardboard": (.45, .40, .35),
            "ac": (.95, .30, .32), "lamp": (.28, .28, .48), "desk_lamp": (.28, .28, .48),
            "blind": (1.2, .12, 1.8), "ball": (.65, .65, .65), "cushion": (.65, .55, .20),
            "beanbag": (.85, .85, .55), "robot": (.6, .6, .9), "light": (1.2, .12, .08),
            "door": (1.0, .05, 2.05), "safety_door": (1.0, .06, 2.05),
            "window": (1.2, .05, 1.4),
        }
        aliases = {"bean bag": "beanbag", "office chair": "office_chair", "desk lamp": "lamp",
                   "dining table": "dining_table", "bean_bag": "beanbag", "drawer cabinet": "drawer_cabinet",
                   "air conditioner": "ac", "cardboard box": "box", "pc tower": "pc",
                   "safety door": "safety_door", "ergonomic chair": "ergonomic_chair",
                   "pressure stool": "pressure_stool", "人体工学椅": "ergonomic_chair",
                   "气压凳": "pressure_stool", "安全门": "safety_door"}

        specs, warnings, cursor = [], [], 0
        for it in items:
            oid = str(it.get("id") or "").strip()
            if not oid:
                continue
            cnt = max(1, int(it.get("count") or 1))
            name = str(it.get("name") or oid)
            raw_kind = (str(it.get("kind") or "") + " " + name + " " + oid).lower()
            explicit = str(it.get("kind") or "").lower().replace(" ", "_")
            kind = explicit if explicit in defaults else None
            if not kind:
                kind = next((k for k in defaults if k in raw_kind), None)
            if not kind:
                kind = next((v for k, v in aliases.items() if k in raw_kind), None)
            kind = kind or "default"
            dims = it.get("size_m") or []
            if not (isinstance(dims, list) and len(dims) >= 3):
                dims = defaults.get(kind, (.9, .6, .9))
                warnings.append(f"{oid} 未给 size_m，使用 {kind} 默认尺寸")
            placements = []
            inst = it.get("instances") or []
            base = self._place_of(it, bbox)
            for i in range(cnt):
                q = inst[i] if i < len(inst) else None
                p = self._place_of(q, bbox) if isinstance(q, dict) else None
                if p is None and isinstance(q, (list, tuple)) and len(q) >= 2:
                    p = (float(q[0]), float(q[1]), float(q[3] if len(q) > 3 else 0))
                if p is None and base is not None:
                    p = (base[0] + (float(dims[0]) * 1.1 * i if cnt > 1 else 0), base[1], base[2])
                if p is None:
                    x = bbox[0] + .6 + (cursor % 6) * 1.1
                    y = bbox[2] + .6
                    p = (x, y, 0)
                    warnings.append(f"{oid} 没有 place/cell，按房间边缘兜底")
                x = min(max(float(p[0]), bbox[0] + .35), bbox[1] - .35)
                y = min(max(float(p[1]), bbox[2] + .35), bbox[3] - .35)
                placements.append([round(x, 2), round(y, 2), 0.0, round(float(p[2]), 1)])
                cursor += 1
            specs.append({"id": oid, "dims": [float(v) for v in dims[:3]], "count": cnt,
                          "kind": kind, "name": name, "material": str(it.get("material") or ""),
                          "color": it.get("color"), "placements": placements,
                          "relation": it.get("relation") or {}})

        by_id = {s["id"]: s for s in specs}
        resolved, unresolved = [], []
        for s in specs:
            rel = s.get("relation") or {}
            if not rel:
                continue
            typ, target = str(rel.get("type") or "").lower(), str(rel.get("target") or "")
            off = rel.get("offset_m") or rel.get("offset") or [0, 0]
            if typ in ("on", "near") and target in by_id and by_id[target]["placements"]:
                t = by_id[target]
                tp = t["placements"][0]
                ztop = round(tp[2] + (float(t["dims"][2]) if typ == "on" else 0.0), 2)
                for i, pp in enumerate(s["placements"]):
                    # keep per-instance x/y when the plan gave several copies; relation only
                    # supplies the support height. Otherwise anchor the single item to its target.
                    if i == 0 and len(s["placements"]) <= 1:
                        pp[0] = round(tp[0] + float(off[0]), 2)
                        pp[1] = round(tp[1] + float(off[1] if len(off) > 1 else 0), 2)
                    pp[2] = ztop
                s["relation"]["resolved"] = {"target": target, "type": typ}
                resolved.append(f"{s['id']} {typ} {target}")
            elif typ == "against":
                side = ""
                for tok in str(target).upper().replace("-", "_").split("_"):
                    if tok in ("N", "S", "E", "W"):
                        side = tok
                if not side and str(target).upper()[:1] in ("N", "S", "E", "W"):
                    side = str(target).upper()[:1]
                if side:
                    for pp in s["placements"]:
                        pp[2] = 0.0
                        pp[3] = 90 if side in "EW" else 0
                        if side == "E":
                            pp[0] = round(bbox[1] - float(s["dims"][1]) / 2 - .08, 2)
                        elif side == "W":
                            pp[0] = round(bbox[0] + float(s["dims"][1]) / 2 + .08, 2)
                        elif side == "N":
                            pp[1] = round(bbox[3] - float(s["dims"][1]) / 2 - .08, 2)
                        else:
                            pp[1] = round(bbox[2] + float(s["dims"][1]) / 2 + .08, 2)
                    s["relation"]["resolved"] = {"target": target, "type": "against"}
                    resolved.append(f"{s['id']} against {target}")
            else:
                unresolved.append(s["id"])

        def box_at(s, p):
            w, d, h = s["dims"]
            if abs(float(p[3] if len(p) > 3 else 0)) % 180 == 90:
                w, d = d, w
            return (p[0] - w / 2, p[0] + w / 2, p[1] - d / 2, p[1] + d / 2,
                    p[2], p[2] + h)
        overlaps, nudged = [], 0
        for _ in range(4):
            changed = False
            for i, a in enumerate(specs):
                for j, b in enumerate(specs):
                    if i >= j:
                        continue
                    for pa in a["placements"]:
                        for pb in b["placements"]:
                            ba, bb = box_at(a, pa), box_at(b, pb)
                            if (min(ba[1], bb[1]) - max(ba[0], bb[0]) > .03
                                    and min(ba[3], bb[3]) - max(ba[2], bb[2]) > .03
                                    and min(ba[5], bb[5]) - max(ba[4], bb[4]) > .03):
                                rel_ab = (a.get("relation") or {}).get("target") == b["id"]
                                rel_ba = (b.get("relation") or {}).get("target") == a["id"]
                                if (rel_ab and (a.get("relation") or {}).get("type") == "on") or \
                                   (rel_ba and (b.get("relation") or {}).get("type") == "on"):
                                    continue
                                overlaps.append(f"{a['id']}x{b['id']}")
                                # AABB separation must choose the cheaper axis. The old v10
                                # solver only nudged x, so a barrier row along a wall still
                                # penetrated the desk beside it. Pick x or y by smaller
                                # penetration; rotate dims already handled by box_at().
                                ovx = min(ba[1], bb[1]) - max(ba[0], bb[0]) + .08
                                ovy = min(ba[3], bb[3]) - max(ba[2], bb[2]) + .08
                                axis = 0 if ovx <= ovy else 1
                                delta = (ovx if axis == 0 else ovy)
                                if a["dims"][0] * a["dims"][1] <= b["dims"][0] * b["dims"][1]:
                                    if axis == 0:
                                        pa[0] = round(min(max(pa[0] + (delta if pa[0] >= pb[0] else -delta),
                                                           bbox[0] + .35), bbox[1] - .35), 2)
                                    else:
                                        pa[1] = round(min(max(pa[1] + (delta if pa[1] >= pb[1] else -delta),
                                                           bbox[2] + .35), bbox[3] - .35), 2)
                                else:
                                    if axis == 0:
                                        pb[0] = round(min(max(pb[0] + (delta if pb[0] >= pa[0] else -delta),
                                                           bbox[0] + .35), bbox[1] - .35), 2)
                                    else:
                                        pb[1] = round(min(max(pb[1] + (delta if pb[1] >= pa[1] else -delta),
                                                           bbox[2] + .35), bbox[3] - .35), 2)
                                nudged += 1
                                changed = True
            if not changed:
                break
        overlaps = sorted(set(overlaps))

        import pprint
        body = pprint.pformat(specs, width=118, sort_dicts=False)
        ids = repr([x["id"] for x in specs])
        template = """import sys
sys.path.insert(0, __LIB__)
import bpy
import bs_assets

ROOT = __ROOT__
SPECS = __SPECS__
IDS = __IDS__

for o in list(bpy.data.objects):
    if any(o.name == i or o.name.startswith(i + "_") or o.name.startswith(i + "__")
           or o.name.startswith(i + ".") for i in IDS):
        bpy.data.objects.remove(o, do_unlink=True)

for spec in SPECS:
    bs_assets.build_instance(spec)

print("ASSETS_BUILT", len(SPECS), "ids", sum(x["count"] for x in SPECS), "instances")
"""
        code = (template.replace("__LIB__", repr(str(self.root / "_lib")))
                .replace("__ROOT__", repr(str(self.root)))
                .replace("__SPECS__", body).replace("__IDS__", ids))
        out_path.write_text(code, encoding="utf8")
        return {"ok": True, "path": str(out_path.relative_to(self.root)),
                "ids": len(specs), "instances": sum(s["count"] for s in specs),
                "kinds": sorted({s["kind"] for s in specs}), "warnings": warnings,
                "resolved_relations": resolved, "unresolved_relations": unresolved,
                "overlaps": overlaps, "nudged": nudged,
                "room_bbox_m": bbox, "kb": round(out_path.stat().st_size / 1024, 1),
                "fix_hint": "先看 SPECS 的 placements/relation；材质只改 color；形状改 bs_assets.build_<kind>"}
    def t_build_assets(self, script: str = "build_assets.py"):
        """Run the asset script on the existing shell, then give EVERY id a temporary camera.

        The object camera is created, used, and deleted inside the same Blender process; it is
        never saved into scene.blend. Each asset must be ONE joined mesh, and the returned contact
        sheet is what the model looks at before deciding whether an item is correct.
        """
        self._ensure_lib()
        sp = self.root / script
        if not sp.exists():
            return {"ok": False, "error": "缺 %s" % script, "fix_hint": "先 scaffold_assets() 生成物品脚本"}
        if not (self.root / "scene.blend").exists():
            return {"ok": False, "error": "缺 scene.blend", "fix_hint": "先在 S2 跑 init_scene + build_shell"}
        plan = json.loads((self.root / "SCENE_PLAN.json").read_text())
        ids = [str(x.get("id")) for x in (plan.get("objects") or []) + (plan.get("inferred") or [])
               if x.get("id")]
        for old in (self.root / "previews").glob("build_*.png"):
            old.unlink(missing_ok=True)
        obj_dir = self.root / "previews" / "objects"
        if obj_dir.is_dir():
            for old in obj_dir.glob("*.png"):
                old.unlink(missing_ok=True)
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n" % str(self.root / "_lib")
            + "import bs_stage\n"
            + "bs_stage.init(%r, %r, %r, rebuild=False)\n" % (
                str(self.root / "calibration.json"), str(self.root / "scene.blend"),
                str(self.root / "previews" / "00_assets.png"))
            + "exec(compile(open(%r).read(), %r, 'exec'), {'__name__': '__main__'})\n" % (str(sp), str(sp))
            + "import json as _j, pathlib as _pp\n"
            + "_stage = _pp.Path(%r); _stage.mkdir(parents=True, exist_ok=True)\n" % str(self.root / "previews" / "_stage")
            + "b = bs_stage.finalize(%r, %r, res=480, traj_limit=0)\n" % (
                str(self.root / "scene.blend"), str(self.root / "previews"))
            + "(_stage/'build.json').write_text(_j.dumps(b, ensure_ascii=False))\n"
            + "c = bs_stage.coverage(%r)\n" % str(self.root / "SCENE_PLAN.json")
            + "(_stage/'coverage.json').write_text(_j.dumps(c, ensure_ascii=False))\n"
            + "p = bs_stage.placement_report(%r)\n" % str(self.root / "SCENE_PLAN.json")
            + "(_stage/'placement.json').write_text(_j.dumps(p, ensure_ascii=False))\n"
            + "sp = bs_stage.support_report(%r)\n" % str(self.root / "SCENE_PLAN.json")
            + "(_stage/'support.json').write_text(_j.dumps(sp, ensure_ascii=False))\n"
            + "a = bs_stage.inspect_assets(%r, %r, %r, res=320, samples=4)\n" % (
                str(self.root / "scene.blend"), ids, str(self.root / "previews" / "objects"))
            + "(_stage/'assets.json').write_text(_j.dumps(a, ensure_ascii=False))\n"
        )
        res = self.ex.blender_py(self._stage_code("build_assets", code), timeout=1500)
        out = res.get("stdout") or ""
        stage = self.root / "previews" / "_stage"
        def _load(name):
            p = stage / name
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        build = _load("build.json") or _json_after(out, "BUILD_JSON:")
        coverage = _load("coverage.json") or _json_after(out, "COVERAGE_JSON:")
        asset = _load("assets.json") or _json_after(out, "ASSET_JSON:")
        placement = _load("placement.json") or _json_after(out, "PLACEMENT_JSON:")
        support = _load("support.json") or _json_after(out, "SUPPORT_JSON:")
        shots = []
        for p in sorted((self.root / "previews").glob("build_*.png")):
            st = image_stats(p)
            rel = str(p.relative_to(self.root))
            self.stats["renders"].append(rel)
            shots.append({"path": rel, "mean": st.get("mean"), "std": st.get("std"),
                          "edge_frac": st.get("edge_frac"), "flat": st.get("flat")})
        paths = sorted(obj_dir.glob("*.png"))
        sheet = contact_sheet(paths, self.root / "previews" / "objects_sheet.png")
        rows = {r.get("id"): r for r in (asset.get("objects") or [])}
        separate = [r["id"] for r in rows.values() if r.get("meshes", 1) > 1]
        not_ok = [r["id"] for r in rows.values() if not r.get("ok")]
        summary = {"ok": bool(res.get("ok")) and bool(shots) and coverage.get("ok", False)
                   and placement.get("ok", False) and support.get("ok", True)
                   and not separate and not not_ok,
                   "script": script, "meshes": build.get("meshes"), "cameras": build.get("cameras"),
                   "shots": shots, "coverage": coverage, "placement": placement,
                   "support": support,
                   "objects": asset.get("objects") or [],
                   "separate_parts": separate, "bad_objects": not_ok,
                   "sheet": str(sheet.relative_to(self.root)) if sheet.exists() else "",
                   "duplicates_removed": build.get("duplicates_removed")}
        if res.get("ok") and shots:
            self.state["last_blend"] = "scene.blend"
            self.state["build"] = summary
            self.state["assets"] = summary
            self.state["object_checks"] = {r.get("id"): r for r in (asset.get("objects") or [])}
        else:
            summary["stdout_tail"] = out[-1800:]
            summary["stderr_tail"] = (res.get("stderr") or "")[-3000:]
        return summary

    def t_inspect_object(self, id: str):
        """Re-render ONE item with a temporary camera after a patch; camera is deleted immediately."""
        self._ensure_lib()
        blend = self.state.get("last_blend") or "scene.blend"
        if not (self.root / blend).exists():
            return {"ok": False, "error": "缺 scene.blend", "fix_hint": "先 build_assets"}
        for old in (self.root / "previews" / "objects").glob(f"{id}*.png"):
            old.unlink(missing_ok=True)
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n" % str(self.root / "_lib")
            + "import bs_stage\n"
            + "bs_stage.inspect_assets(%r, %r, %r, res=320, samples=4)\n" % (
                str(self.root / blend), [id], str(self.root / "previews" / "objects"))
        )
        res = self.ex.blender_py(self._stage_code("inspect_object", code), timeout=300)
        asset = _json_after(res.get("stdout") or "", "ASSET_JSON:")
        row = (asset.get("objects") or [{}])[0]
        if row:
            self.state.setdefault("object_checks", {})[id] = row
        sheet = contact_sheet(sorted((self.root / "previews" / "objects").glob("*.png")),
                              self.root / "previews" / "objects_sheet.png")
        return {"ok": bool(row.get("ok")) and row.get("meshes") == 1, "object": row,
                "camera_deleted": True,
                "sheet": str(sheet.relative_to(self.root)) if sheet.exists() else "",
                "stderr_tail": (res.get("stderr") or "")[-1200:]}

    def t_join_asset(self, id: str):
        """Safety net: if a script left loose parts, join them into one mesh under the root."""
        self._ensure_lib()
        blend = self.state.get("last_blend") or "scene.blend"
        code = (
            "import sys\n"
            "sys.path.insert(0, %r)\n" % str(self.root / "_lib")
            + "import bpy, bs_geo\n"
            + "bpy.ops.wm.open_mainfile(filepath=%r)\n" % str((self.root / blend).resolve())
            + "root = bpy.data.objects.get(%r)\n" % id
            + "m = bs_geo.join_children(root) if root else None\n"
            + "bpy.ops.wm.save_mainfile()\n"
            + "print('JOIN_JSON:' + __import__('json').dumps({'ok': m is not None, 'mesh': m.name if m else ''}))\n"
        )
        res = self.ex.blender_py(self._stage_code("join_asset", code), timeout=120)
        info = _json_after(res.get("stdout") or "", "JOIN_JSON:")
        return {"ok": bool(info.get("ok")), "id": id, "mesh": info.get("mesh"),
                "stderr_tail": (res.get("stderr") or "")[-1000:]}

    def t_build_shell(self, rebuild: bool = False):
        """One deterministic call for the whole shell (vln_toolkit-style fixed layout, no randomness).
        Removes 20-40 steps of hand-deriving walls and keeps door/window counts from the plan."""
        self._ensure_lib()
        plan = self.root / "SCENE_PLAN.json"
        if not plan.exists():
            return {"ok": False, "error": "缺 SCENE_PLAN.json", "fix_hint": "先完成 S1 的户型与门窗"}
        if not (self.root / "calibration.json").exists():
            return {"ok": False, "error": "缺 calibration.json", "fix_hint": "先 prepare_inputs / init_scene"}
        if not (self.root / "scene.blend").exists():
            return {"ok": False, "error": "缺 scene.blend", "fix_hint": "先 init_scene() 建相机与地面"}
        lib = str((self.root / "_lib"))
        for old in (self.root / "previews").glob("build_*.png"):
            old.unlink(missing_ok=True)
        code = (
            "import sys\n"
            f"sys.path.insert(0, {lib!r})\n"
            "import bs_layout\n"
            f"bs_layout.build({str((self.root / 'calibration.json'))!r}, "
            f"{str((self.root / 'SCENE_PLAN.json'))!r}, "
            f"{str((self.root / 'scene.blend'))!r}, {str((self.root / 'previews'))!r})\n"
        )
        res = self.ex.blender_py(self._stage_code("build_shell", code), timeout=420)
        shots, flat = [], []
        for p in sorted((self.root / "previews").glob("shell_*.png")):
            st = image_stats(p)
            rel = str(p.relative_to(self.root))
            self.stats["renders"].append(rel)
            shots.append({"path": rel, "std": st.get("std"), "mean": st.get("mean"),
                          "edge_frac": st.get("edge_frac"), "flat": st.get("flat")})
            if st.get("flat"):
                flat.append(rel)
        shell = {}
        try:
            shell = json.loads((self.root / "previews" / "_stage" / "shell.json").read_text())
        except Exception:
            shell = _json_after(res.get("stdout") or "", "SHELL_JSON:")
        if res.get("ok") and shots:
            shell.update({"cameras": sorted({o for o in (shell.get("cameras") or [])}) or
                          ["Cam_Overview", "Cam_Traj_01", "Cam_Traj_02"],
                          "shots": shots})
            self.state["last_blend"] = "scene.blend"
            self.state["shell_built"] = True
            self.state["shell"] = shell
        return {"ok": bool(res.get("ok")) and bool(shots), "blend": "scene.blend", "shots": shots,
                "meshes": shell.get("meshes"), "windows": shell.get("windows"),
                "doors": shell.get("doors"), "cameras": shell.get("cameras") or [],
                "flat": flat, "warning": ("这些外壳预览是平的：" + ", ".join(flat[:3]) if flat else ""),
                "stdout_tail": (res.get("stdout") or "")[-900:],
                "stderr_tail": (res.get("stderr") or "")[-700:]}

    def t_snapshot(self, tag: str = "s", res: int = 640):
        self._ensure_lib()
        blend = self.state.get("last_blend") or "scene.blend"
        if not (self.root / blend).exists():
            cands = sorted(self.root.rglob("*.blend"), key=lambda q: q.stat().st_mtime, reverse=True)
            if not cands:
                return {"ok": False, "error": "还没有 scene.blend", "fix_hint": "先 init_scene()"}
            blend = str(cands[0].relative_to(self.root))
        lib = str((self.root / "_lib"))
        for old in (self.root / "previews").glob("build_*.png"):
            old.unlink(missing_ok=True)
        code = (
            "import sys\n"
            f"sys.path.insert(0, {lib!r})\n"
            "import bs_stage\n"
            f"bs_stage.snapshot({str((self.root / blend))!r}, {str((self.root / 'calibration.json'))!r}, "
            f"{str((self.root / 'previews'))!r}, tag={tag!r}, res={int(res)!r})\n"
        )
        res_ = self.ex.blender_py(self._stage_code("snapshot", code), timeout=300)
        shots, flat = [], []
        for p in sorted((self.root / "previews").glob(f"{tag}_*.png")):
            st = image_stats(p)
            rel = str(p.relative_to(self.root))
            self.stats["renders"].append(rel)
            shots.append({"path": rel, **{k: st.get(k) for k in ("std", "mean", "edge_frac", "flat")}})
            if st.get("flat"):
                flat.append(rel)
        (self.root / "views.json").write_text(json.dumps(
            {"shots": shots, "cameras": sorted({s["path"] for s in shots})}, ensure_ascii=False, indent=1))
        return {"ok": bool(res_.get("ok")) and bool(shots), "blend": blend, "shots": shots, "flat": flat,
                "warning": ("这些机位是平的：" + ", ".join(flat[:4]) + "｜先确认相机的 scene.camera 与几何/灯光"
                            if flat else ""),
                "stderr_tail": (res_.get("stderr") or "")[-700:]}

    def t_material_audit(self):
        self._ensure_lib()
        blend = self.state.get("last_blend") or "scene.blend"
        if not (self.root / blend).exists():
            return {"ok": False, "error": "没有 .blend", "fix_hint": "先 init_scene()"}
        lib = str((self.root / "_lib"))
        code = ("import sys\n" f"sys.path.insert(0, {lib!r})\n" "import bs_stage\n"
                f"bs_stage.audit({str((self.root / blend))!r})\n")
        res_ = self.ex.blender_py(self._stage_code("material_audit", code), timeout=240)
        out = {"ok": bool(res_.get("ok")), "stdout_tail": (res_.get("stdout") or "")[-2000:]}
        # parse the AUDIT_JSON marker instead of trusting the model to read stdout
        for line in (res_.get("stdout") or "").splitlines():
            if line.startswith("AUDIT_JSON:"):
                try:
                    out.update(json.loads(line.split(":", 1)[1]))
                except Exception:
                    pass
        return out

    def t_blender_py(self, script: str = "", args: list[str] | None = None, blend: str = "",
                     timeout: int = 300, code: str = ""):
        """script = a .py path relative to project/ ; code = inline Python (written to _tmp first).
        A "path" that is obviously Python source is re-routed to code instead of failing."""
        self._ensure_lib()
        script = (script or "").strip()
        if code.strip():
            script_path = self._write_inline(code)
        else:
            if not script:
                return {"ok": False, "error": "blender_py 需要 script（项目内 .py 路径）或 code（内联 Python）",
                        "fix_hint": "写文件：write_file(path='build.py', content=...) 再用 script='build.py'"}
            if self._looks_like_code(script):
                script_path = self._write_inline(script)
            else:
                script_path = script.strip().strip('"\'')
        res = self.ex.blender_py(script_path, args=args or [], blend=blend or None, timeout=timeout)
        res.setdefault("script_used", script_path)
        if res.get("ok"):
            blends = [q for q in self.root.rglob("*.blend")]
            if blends:
                newest = max(blends, key=lambda q: q.stat().st_mtime)
                self.state["last_blend"] = str(newest.relative_to(self.root))
                res["blend_written"] = self.state["last_blend"]
                res["blend_mtime"] = newest.stat().st_mtime
            saved = "Saved as \"" if "Saved as" in res.get("stdout", "") else ""
            if saved and not blends:
                res["warning"] = "脚本报告已保存，但 project/ 里没有 .blend —— 你在别的目录存盘了"
                res["fix_hint"] = "在脚本里用绝对路径保存到 project/ 根，例如 BASE=project 的绝对路径"
        return res

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        markers = ("\n", "import ", "exec(", "print(", "bpy.", "def ", "=", ";")
        return any(m in text for m in markers) and not text.strip().endswith(".py")

    def _write_inline(self, code: str) -> str:
        tmp = self.root / "_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        path = tmp / f"inline_{int(time.time() * 1000) % 100000}.py"
        path.write_text(code, encoding="utf8")
        return str(path.relative_to(self.root))

    # ------------------------------------------------------------------ rendering
    def _ensure_lib(self):
        """Copy the Blender helper library into project/_lib. Never copies a file onto itself
        (SameFileError happened when lib_dir *was* project/_lib)."""
        dst = (self.root / "_lib").resolve()
        dst.mkdir(parents=True, exist_ok=True)
        src = self.lib_dir.resolve()
        for f in src.glob("*.py"):
            target = dst / f.name
            if target.resolve() == f.resolve():
                continue
            shutil.copy2(f, target)
        cache = dst / "__pycache__"      # copy2 preserves mtime; a stale .pyc would shadow the fix
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
        return dst

    def t_render_trajectory(self, frames: int = 100, res: int = 384, fps: int = 5):
        """Render the original camera path inside the twin and make a side-by-side video."""
        self._ensure_lib()
        blend = str(self.root / "scene.blend")
        plan = str(self.root / "SCENE_PLAN.json")
        cal = str(self.root / "calibration.json")
        for f in (blend, plan, cal):
            if not Path(f).exists():
                return self._fail("render_trajectory", {}, f"缺 {Path(f).name}",
                                  "先完成 S1/S2 并生成 calibration.json")
        lib = str((self.root / "_lib"))
        code = (
            "import sys\n"
            f"sys.path.insert(0, {lib!r})\n"
            "import bs_trajvideo\n"
            f"bs_trajvideo.render({blend!r}, {cal!r}, {plan!r}, {str(self.root / 'previews')!r}, "
            f"frames={int(frames)}, res={int(res)}, fps={int(fps)})\n"
        )
        res_ = self.ex.blender_py(self._stage_code("render_trajectory", code), timeout=900)
        if not res_.get("ok"):
            return self._fail("render_trajectory", {}, "轨迹渲染失败：Blender 未完成",
                              (res_.get("stderr") or "")[-500:])
        frames_dir = self.root / "previews" / "traj_frames"
        mp4 = self.root / "previews" / "trajectory_render.mp4"
        dur = max(0.5, float(frames) / max(1, int(fps)))
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(int(fps)),
                            "-i", str(frames_dir / "%04d.png"), "-t", f"{dur:.2f}",
                            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(mp4)],
                           check=True, timeout=300)
        except Exception as exc:
            return self._fail("render_trajectory", {}, f"ffmpeg 编码失败: {exc}", "检查 traj_frames")
        compare = None
        video = self.state.get("video_path")
        if video and Path(video).exists():
            out = self.root / "previews" / "trajectory_compare.mp4"
            h = int(int(res) * 0.75)
            filt = (f"[0:v]fps={int(fps)},scale={int(res)}:{h},trim=duration={dur:.2f},setpts=PTS-STARTPTS[a];"
                    f"[1:v]fps={int(fps)},scale={int(res)}:{h},trim=duration={dur:.2f},setpts=PTS-STARTPTS[b];"
                    f"[a][b]hstack")
            try:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
                                "-i", str(mp4), "-filter_complex", filt, "-t", f"{dur:.2f}",
                                "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", str(out)],
                               check=True, timeout=300)
                compare = str(out.relative_to(self.root))
                self.stats.setdefault("comparisons", []).append(compare)
            except Exception:
                compare = None
        self.stats["renders"].append(str(mp4.relative_to(self.root)))
        return {"ok": True, "frames": int(frames), "fps": int(fps), "duration_s": round(dur, 2),
                "video": str(mp4.relative_to(self.root)), "compare": compare,
                "frames_dir": str(frames_dir.relative_to(self.root))}

    def t_render_views(self, blend: str = "", views: list[dict[str, Any]] | None = None, res: int | None = None,
                       samples: int | None = None, script: str = ""):
        self._ensure_lib()
        assumed = False
        if not blend:
            blend = self.state.get("last_blend") or ""
            assumed = bool(blend)
        if not blend or not (self.root / blend).exists():
            cands = sorted(self.root.rglob("*.blend"), key=lambda q: q.stat().st_mtime, reverse=True)
            if cands:
                blend = str(cands[0].relative_to(self.root))
                assumed = True
        if not blend:
            return {"ok": False, "error": "没有 .blend 可渲染：先用 blender_py 生成",
                    "fix_hint": "blender_py(script='build.py') 会用脚本里的保存路径产出 .blend"}
        views = views or [{"name": "v01", "loc": [0, -8, 1.6], "target": [0, 0, 1.2], "lens": 18}]
        res = int(res or self.quality.res)
        samples = int(samples or self.quality.samples)
        payload = {"views": views, "out_dir": str(self.root / "previews"), "res": res, "samples": samples}
        cfg = self.root / "_tmp" / "views.json"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text(json.dumps(payload), encoding="utf8")
        runner = script or "_render_views.py"
        rp = self.root / runner
        if not script:
            rp.write_text(
                "import json,sys\nfrom pathlib import Path\n"
                "sys.path.insert(0, str(Path(__file__).parent / '_lib'))\n"
                "import bs_qa\n"
                f"cfg=json.loads(Path({str(cfg)!r}).read_text())\n"
                "print('RENDERED', json.dumps(bs_qa.render_views(cfg['views'], cfg['out_dir'], cfg['res'], cfg['samples'])))\n",
                encoding="utf8")
        res_ = self.ex.blender_py(runner, blend=blend, timeout=max(240, 90 * len(views)))
        shots = []
        for v in views:
            p = self.root / "previews" / f"{v['name']}.png"
            if p.exists():
                st = image_stats(p)
                st["name"] = v["name"]
                shots.append(st)
                self.stats["renders"].append(str(p.relative_to(self.root)))
        flat = [s["name"] for s in shots if s.get("flat")]
        out = {"ok": res_["ok"] and bool(shots) and not flat,
               "blend": blend, "assumed_blend": assumed,
               "shots": shots, "flat": flat, "seconds": res_["seconds"],
               "stdout": res_["stdout"][-800:], "stderr": res_["stderr"][-800:]}
        if flat:
            out["error"] = (f"flat renders: {flat}. low std means the camera is buried in a shell or facing one "
                            f"untextured surface — fix geometry/camera, do not accept this image")
            out["fix_hint"] = ("这台相机所在位置可能被封在墙里：qa_scene 的 camera_clearance 会给出射线命中；"
                               "把 loc 抬高到 1.4–1.7 m 并退到房间内部")
        return out

    def t_image_stats(self, paths: Iterable[str]):
        rows = []
        for rel in list(paths)[:20]:
            p = self.ex.resolve(rel)
            if p.exists():
                rows.append(image_stats(p))
            else:
                rows.append({"ok": False, "path": rel, "error": "missing"})
        return {"ok": all(r.get("ok") for r in rows), "images": rows}

    # ------------------------------------------------------------------ video
    def t_video_frames(self, count: int = 6, times: list[float] | None = None, fps: float = 0, res: int = 0):
        video = self.state["video_path"]
        out_dir = self.root / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        made = []
        if fps:
            meta = self.state.get("video_meta") or {}
            duration = float(meta.get("duration_s") or 20.0)
            step = 1.0 / max(0.1, float(fps))
            times = [round(i * step, 3) for i in range(max(1, int(duration * float(fps))))]
            times = times[:180]
        if times:
            for i, t in enumerate(times[:20]):
                out = out_dir / f"t_{float(t):06.2f}.png"
                if not out.exists():
                    filt = f"scale={res}:-1" if res else "null"
                    res_ = self.ex.run(["ffmpeg", "-y", "-ss", f"{float(t):.3f}", "-i", video, "-frames:v", "1",
                                        "-vf", filt, str(out)], timeout=90)
                    if not out.exists():
                        return {"ok": False, "error": f"ffmpeg failed: {res_['stderr'][-400:]}"}
                made.append(str(out.relative_to(self.root)))
        else:
            meta = self.state.get("video_meta") or {}
            duration = float(meta.get("duration_s") or 20.0)
            n = max(2, min(int(count), 16))
            for i in range(n):
                t = duration * (i + 0.5) / n
                out = out_dir / f"t_{t:06.2f}.png"
                if not out.exists():
                    filt = f"scale={res}:-1" if res else "null"
                    res_ = self.ex.run(["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video, "-frames:v", "1",
                                        "-vf", filt, str(out)], timeout=90)
                    if not out.exists():
                        return {"ok": False, "error": f"ffmpeg failed: {res_['stderr'][-400:]}"}
                made.append(str(out.relative_to(self.root)))
        if fps:
            try:
                sheet = self.root / "previews" / "timeline_5fps.png"
                sheet.parent.mkdir(parents=True, exist_ok=True)
                import glob
                files = sorted(glob.glob(str(out_dir / "t_*.png")))
                if files:
                    fps_ff = max(1, min(8, len(files)))
                    # a 4-column contact sheet keeps the temporal order readable at 560 px
                    self.ex.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                                 "-pattern_type", "glob", "-i", str(out_dir / "t_*.png"),
                                 "-vf", "scale=160:-1,tile=10x10:padding=2:margin=2",
                                 "-frames:v", "1", str(sheet)], timeout=120)
            except Exception as exc:
                (self.on_note or (lambda _: None))(f"[frames] timeline sheet failed: {exc}")
        return {"ok": True, "frames": made, "count": len(made), "fps": fps,
                "timeline_sheet": "previews/timeline_5fps.png" if fps else "",
                "note": "look at these; write down the evidence frame name for every object you claim"}

    def t_compare(self, video_frame: str, render: str, out: str = "previews/compare_01.png"):
        src = self.ex.resolve(video_frame, must_exist=True)
        ren = self.ex.resolve(render, must_exist=True)
        target = self.ex.resolve(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        p = side_by_side(src, ren, target)
        m = compare_metrics(src, ren)
        self.stats["comparisons"].append(str(p.relative_to(self.root)))
        return {"ok": True, "comparison": str(p.relative_to(self.root)), **m}

    # ------------------------------------------------------------------ QA
    def t_prepare_inputs(self, recompute: bool = True):
        from . import inputs_prep
        facts = inputs_prep.prepare(self.root, self.state.get("video_meta") or {})
        fact_note = "calibration.json 已自动生成（重力对齐/路径/FOV）；尺度与房间尺寸仍需用视频里的已知尺寸元素确认"
        return {"ok": bool(facts.get("ready", {}).get("intrinsics")),
                "ready": facts.get("ready"), "trajectory": facts.get("trajectory"),
                "intrinsics": facts.get("intrinsics"), "note": fact_note,
                "calibration": str(self.root / "calibration.json")}

    def _resolve_intrinsics(self, explicit: str = "") -> str:
        """Return an intrinsics file path relative to project/, or "" — never a guessed file name.

        Order: explicit argument -> calibration.json:intrinsics_path -> content scan of inputs/*.json.
        A file counts as intrinsics if it parses to an `fx` (K / K_pinhole_pipeline / camera_matrix).
        """
        if explicit:
            p = self.root / explicit
            if p.exists():
                self.last_intrinsics_path = explicit
                return explicit
        cal = self.root / "calibration.json"
        if cal.exists():
            try:
                name = (json.loads(cal.read_text()) or {}).get("intrinsics_path")
            except Exception:
                name = None
            if name and (self.root / "inputs" / name).exists():
                self.last_intrinsics_path = f"inputs/{name}"
                return f"inputs/{name}"
        for cand in sorted((self.root / "inputs").glob("*.json")):
            try:
                doc = json.loads(cand.read_text())
            except Exception:
                continue
            K = (doc.get("K") or doc.get("k") or doc.get("intrinsics")
                 or doc.get("K_pinhole_pipeline") or doc.get("camera_matrix"))
            if isinstance(K, list) and K and isinstance(K[0], list):
                self.last_intrinsics_path = str(cand.relative_to(self.root))
                return str(cand.relative_to(self.root))
        self.last_intrinsics_path = ""
        return ""

    def t_qa_scene(self, blend: str = "", texture_mode: str = "", intrinsics: str = ""):
        self._ensure_lib()
        assumed = False
        if not blend:
            blend = self.state.get("last_blend") or ""
            assumed = bool(blend)
        if not blend or not (self.root / str(blend)).exists():
            cands = sorted(self.root.rglob("*.blend"), key=lambda q: q.stat().st_mtime, reverse=True)
            if not cands:
                return {"ok": False, "error": "没有任何 .blend：先 blender_py 跑构建脚本生成场景",
                        "fix_hint": "blender_py(script='build.py', blend='') 会新建场景并保存到 build 脚本指定的路径"}
            blend = str(cands[0].relative_to(self.root))
            assumed = True
        b = self.ex.resolve(blend, must_exist=True)
        mode = texture_mode or self.quality.texture_mode
        out_json = self.root / "qa.json"
        runner = self.root / "_run_qa.py"
        intr_rel = self._resolve_intrinsics(intrinsics)
        intr_abs = str((self.root / intr_rel).resolve()) if intr_rel else ""
        runner.write_text(
            "import sys\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent / '_lib'))\n"
            "import bs_qa\n"
            f"intr = {intr_abs!r} or None\n"
            f"bs_qa.run_all({str(out_json.resolve())!r}, intr, {mode!r})\n",
            encoding="utf8")
        res_ = self.ex.blender_py("_run_qa.py", blend=blend, timeout=240)
        rep = json.loads(out_json.read_text()) if out_json.exists() else None
        if not rep:
            return {"ok": False, "error": "qa produced no json", "stderr": res_["stderr"][-800:]}
        self.stats["qa"] = rep
        self.stats["qa_blend"] = b.stat().st_size
        self.stats["qa_stamp"] = b.stat().st_mtime
        passed = [c["id"] for c in rep["checks"] if c.get("ok")]
        skipped = [c["id"] for c in rep["checks"] if (c.get("data") or {}).get("skipped")]
        belief = getattr(self, "belief", None)
        if belief is not None:
            for cid in passed:
                belief.close_issue(cid)
        return {"ok": rep["ok"], "blend": blend, "assumed_blend": assumed,
                "intrinsics_path": intr_rel or None,
                "errors": rep["errors"], "passed": passed, "skipped": skipped,
                "checks": [{"id": c["id"], "ok": c["ok"], "detail": c["detail"]} for c in rep["checks"]],
                "summary": rep.get("summary", {})}

    # ------------------------------------------------------------------ belief
    def t_record_object(self, id: str, kind: str, dims_m: list[float] | None = None,
                        evidence: str = "", confidence: float = 0.5, note: str = ""):
        if not id or not kind:
            return {"ok": False, "error": "record_object 需要 id 与 kind",
                    "fix_hint": "例：record_object(id='O1_Sofa', kind='sofa', dims_m=[1.9,0.9,0.8], evidence='frame t=3.2s')"}
        belief = getattr(self, "belief", None)
        if belief is None:
            return {"ok": False, "error": "belief 未挂载（运行时错误）",
                    "fix_hint": "用 note 记录，或重启运行时"}
        belief.add_object(id, kind, dims_m, evidence, confidence, note)
        return {"ok": True, "id": id, "facts": len(belief.data["facts"])}

    # ------------------------------------------------------------------ ledger / finish
    def t_note(self, text: str):
        line = time.strftime("[%H:%M:%S] ") + text.strip()
        with self.ledger_path.open("a", encoding="utf8") as fh:
            fh.write(line + "\n")
        if self.on_note:
            self.on_note(line)
        return {"ok": True, "note": line, "ledger_lines": len(self.ledger_path.read_text().splitlines())}

    def t_finish(self, summary: str, blend: str, glb: str = ""):
        """v5: accept a *usable* scene, not a perfect one. Shell integrity, a camera, and a non-flat
        render are hard; material richness / lighting / shot count are warnings."""
        p = self.ex.resolve(blend, must_exist=True)
        problems, warnings = [], []
        if self.stats["writes"] < 1:
            problems.append("没有任何建模脚本被写出")
        if not self.stats["renders"]:
            problems.append("没有成功渲染过任何预览")
        flat = [rel for rel in self.stats["renders"] if image_stats(self.root / rel).get("flat")]
        if len(flat) == len(self.stats["renders"]) and flat:
            problems.append("所有预览都是平的/空的：" + ", ".join(flat[:4]))
        elif flat:
            warnings.append("部分预览偏平：" + ", ".join(flat[:3]))
        if len(self.stats["renders"]) < self.quality.target_previews:
            warnings.append(f"预览 {len(self.stats['renders'])} 张 < 目标 {self.quality.target_previews}（可接受）")
        if not self.stats["comparisons"]:
            warnings.append("没有做原视频帧 vs 渲染的并排对比（建议补，但不阻止交付）")
        qa = self.stats["qa"]
        if not qa:
            problems.append("没有跑过 qa_scene")
        else:
            hard = {"floor_zero", "solid_envelope", "openings", "camera_fov",
                    "duplicate_objects", "plan_coverage", "camera_rig"}
            for e in qa.get("errors", []):
                if e["id"] in hard:
                    problems.append(f"{e['id']}: {str(e.get('detail'))[:80]}")
                else:
                    warnings.append(f"{e['id']}: {str(e.get('detail'))[:60]}")
            if (qa.get("summary") or {}).get("cameras", 0) < 3:
                problems.append("相机不足 3 个（build_scene 应给 1 全景 + ≥2 轨迹位）")
            if self.stats["qa_blend"] != p.stat().st_size or abs((self.stats["qa_stamp"] or 0) - p.stat().st_mtime) > 1:
                problems.append("qa 结论已过期（.blend 在断言之后被改过），重新跑 qa_scene")
        for req in ("SCENE_PLAN.json", "calibration.json"):
            if not (self.root / req).exists():
                problems.append(f"缺少 {req}")
        if not (self.root / "lighting.json").exists():
            warnings.append("缺 lighting.json（只影响灯光可解释性）")
        if problems:
            return {"ok": False, "error": "不能 finish：" + "；".join(problems),
                    "warnings": warnings, "next": "先修 error 里的项；warnings 不阻止交付"}
        return {"ok": True, "finished": True, "summary": summary, "blend": blend,
                "glb": glb if glb and (self.root / glb).exists() else None,
                "render_count": len(self.stats["renders"]), "comparisons": len(self.stats["comparisons"]),
                "warnings": warnings}

