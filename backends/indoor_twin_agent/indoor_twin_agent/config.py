"""Run configuration: the knobs a user is allowed to touch, and how they turn into prompt text."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    # quick look: blockout, no textures, 2 shots
    "fast": {"mesh_detail": "blockout", "texture_mode": "none", "target_previews": 2, "samples": 8, "res": 640},
    # default: real architecture + assets + node materials + image textures
    "balanced": {"mesh_detail": "balanced", "texture_mode": "both", "target_previews": 5, "samples": 24, "res": 960},
    # max: poly-level modelling where it matters, 5 texture maps per material
    "high": {"mesh_detail": "high", "texture_mode": "both", "target_previews": 8, "samples": 48, "res": 1280},
}


@dataclass
class Quality:
    """What the model is allowed/expected to build. Everything here reaches the prompt verbatim."""

    mesh_detail: str = "balanced"      # blockout | balanced | high | poly
    texture_mode: str = "both"         # none | nodes | images | both
    pbr_level: str = "standard"        # flat | standard | full
    window_proxy: bool = True          # build a proxy exterior behind window openings
    target_previews: int = 5
    samples: int = 24
    res: int = 960                     # long edge of preview renders
    min_assets: int = 0                # 0 = derive from plan
    floorspot_m: float = 1.0           # assumed "known size reference" fallback, metres
    stage_step_budget: int = 30        # after this many steps a stage is force-advanced (with a warning)

    def as_preset(self, name: str) -> "Quality":
        base = dataclasses.asdict(self)
        base.update(PRESETS.get(name, {}))
        return Quality(**base)

    def table(self) -> str:
        md = {
            "blockout": "只允许盒体/柱体表达体量；不允许声称完成材质",
            "balanced": "每件资产必须是多部件组合（>=3 个部件，或 >=8 个面片）；允许用基本体拼装，但轮廓要对",
            "high": "关键资产（沙发/桌/机柜/机器人/球）必须用剖面挤出、旋转、扫掠或布尔做出真实轮廓；禁止整件单一 cube",
            "poly": "关键资产按实际轮廓建多边形，允许细分/倒角；次要资产可简化；目标单场景 80k–250k 三角面",
        }[self.mesh_detail if self.mesh_detail in ("blockout", "balanced", "high", "poly") else "balanced"]
        tex = {
            "none": "不生成贴图，材质只用 base color/roughness（会被 finish 拒绝验收“材质完成”）",
            "nodes": "只用 shader 节点程序化（Noise/Voronoi/Brick/Wave + ColorRamp + Bump），每个材质节点链必须真连到 Principled",
            "images": "只用 numpy 算出的 PNG 贴图，必须 UVMap→Mapping→Image→Principled，base color 用 sRGB，rough/normal 用 Non-Color",
            "both": "节点材质做底层变化 + PNG 贴图做法线/粗糙/色斑，两者都真接线",
        }[self.texture_mode if self.texture_mode in ("none", "nodes", "images", "both") else "both"]
        pbr = {
            "flat": "仅 base color，不做 roughness 变化（只用于 blockout）",
            "standard": "每材质必须设 base color / roughness / metallic / IOR，粗糙度要有空间变化（贴图或节点）",
            "full": "standard + 法线/凹凸、边缘磨损、污渍遮罩、透明或自发光按实物语义设置",
        }[self.pbr_level if self.pbr_level in ("flat", "standard", "full") else "standard"]
        proxy = ("窗户洞口后必须建代理外景（室外亮面 + 远处剪影），不允许用纯黑或纯白平面糊住洞口"
                 if self.window_proxy else "窗户洞口后不建外景，但洞口必须真实开洞，玻璃要有透射")
        return (f"- 细节等级 mesh_detail={self.mesh_detail}：{md}\n"
                f"- 贴图模式 texture_mode={self.texture_mode}：{tex}\n"
                f"- PBR 等级 pbr_level={self.pbr_level}：{pbr}\n"
                f"- 窗外代理 window_proxy={'on' if self.window_proxy else 'off'}：{proxy}\n"
                f"- 预览张数 target_previews={self.target_previews}，分辨率长边 {self.res}，采样 {self.samples}")


@dataclass
class RunConfig:
    video: str = ""
    trajectory: str = ""
    intrinsics: str = ""
    out: str = "run"
    model: str = "kimi-k2.6"
    api_base: str = "https://ai.zj-computility.com/maas/v1"
    api_key_env: str = "zhesuan_API_KEY"
    vision: bool = True
    temperature: float = 0.35
    max_output_tokens: int = 16384
    # S1 is now a multi-pass recognizer. The endpoint accepts ~6 images per call; the
    # documented max_output is 16384 for k2.6, but the proxy disconnects near 90 s, so
    # segment JSON uses a deliberately smaller budget. Detail passes may use k3.
    recon_scan: bool = True
    recon_segment_model: str = "kimi-k2.6"
    recon_detail_model: str = "kimi-k2.6"
    recon_segment_tokens: int = 2048
    recon_batch_frames: int = 2
    recon_detail_tokens: int = 4096
    context_chars: int = 260_000       # trigger summarisation above this
    max_steps: int = 70
    max_minutes: float = 240.0
    blender: str = "blender"
    seed: int = 20261112
    quality: Quality = field(default_factory=Quality)

    def to_json(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False), encoding="utf8")
