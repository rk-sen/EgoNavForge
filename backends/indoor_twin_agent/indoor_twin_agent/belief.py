"""Structured belief: what the run actually knows, and what is still broken.

The old design kept "memory" as the last 8 assistant/tool messages plus a free-text ledger. That is
lossy in exactly the wrong way: after a stage flips, the numbers are gone and the model re-reads
source files to recover them. Belief is the opposite: a JSON document on disk, rebuilt into every
request, that survives stage changes, context compression and restarts.

Three things live here:
  facts    — objects with dimensions and the frame/px evidence that justifies them
  camera   — measured FOV / height / residual, not "calibration looks fine"
  issues   — one entry per failing check, carrying the stage that owns it, so a failure can
             reopen a stage instead of being silently forgotten
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAGE_OF_CHECK = {
    "floor_zero": "S2_shell",
    "solid_envelope": "S2_shell",
    "openings": "S2_shell",
    "camera_fov": "S2_shell",
    "camera_rig": "S2_shell",
    "camera_clearance": "S2_shell",
    "scale_sanity": "S2_shell",
    "flat_render": "S2_shell",
    "plan_coverage": "S3_assets",
    "duplicate_objects": "S3_assets",
    "objects_above_floor": "S3_assets",
    "corridor_clear": "S3_assets",
    "material_texture_wiring": "S3_assets",
    "material_base_color": "S3_assets",
    "assets": "S3_assets",
    "lighting": "S3_assets",
}


class Belief:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict[str, Any] = {"facts": {}, "camera": {}, "issues": [], "history": []}
        if self.path.exists():
            try:
                self.data.update(json.loads(self.path.read_text()))
            except Exception:
                pass

    # ------------------------------------------------------------------ persistence
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf8")

    # ------------------------------------------------------------------ facts
    def add_object(self, oid: str, kind: str, dims: list[float] | None = None,
                   evidence: str = "", confidence: float = 0.5, note: str = ""):
        self.data["facts"][oid] = {"kind": kind, "dims_m": dims, "evidence": evidence,
                                   "confidence": round(float(confidence), 2), "note": note}
        self.save()

    def set_camera(self, **kw):
        self.data["camera"].update({k: v for k, v in kw.items() if v is not None})
        self.save()

    # ------------------------------------------------------------------ issues
    def open_issue(self, check: str, detail: str, stage: str | None = None,
                   fix_tool: str = "", fix_args: dict | None = None, severity: str = "block") -> dict:
        stage = stage or STAGE_OF_CHECK.get(check, "S4_qa")
        for it in self.data["issues"]:
            if it["check"] == check:          # one issue per check, not per render batch
                it["detail"] = detail          # keep the newest wording
                it["severity"] = severity
                return it
        item = {"id": f"{check}:{len(self.data['issues']) + 1}", "check": check, "detail": detail,
                "stage": stage, "fix_tool": fix_tool, "fix_args": fix_args or {}, "attempts": 0,
                "severity": severity}
        self.data["issues"].append(item)
        self.data["history"].append({"opened": item["id"], "stage": stage})
        self.save()
        return item

    def close_issue(self, check: str):
        before = len(self.data["issues"])
        self.data["issues"] = [i for i in self.data["issues"] if i["check"] != check]
        if len(self.data["issues"]) != before:
            self.save()

    def issues(self, stage: str | None = None) -> list[dict]:
        its = self.data["issues"]
        return [i for i in its if stage is None or i["stage"] == stage]

    def blocking(self, stage: str) -> list[dict]:
        return [i for i in self.issues(stage) if i.get("severity", "block") == "block"]

    def warnings(self, stage: str | None = None) -> list[dict]:
        return [i for i in self.issues(stage) if i.get("severity") == "warn"]

    def bump(self, check: str):
        for i in self.data["issues"]:
            if i["check"] == check:
                i["attempts"] += 1
        self.save()

    # ------------------------------------------------------------------ prompt
    def block(self) -> str:
        cam = self.data.get("camera") or {}
        facts = self.data.get("facts") or {}
        rows = []
        for oid, f in list(facts.items())[:24]:
            dims = f.get("dims_m")
            d = "x".join(f"{v:g}" for v in dims) if dims else "?"
            rows.append(f"{oid}: {f.get('kind')} {d}m  证据={f.get('evidence') or '-'} 置信={f.get('confidence')}")
        issues = self.data.get("issues") or []
        ill = [f"[{i['stage']}] {i.get('severity','block')} {i['check']}: {i['detail'][:120]}"
               for i in issues[:12]]
        return ("<belief>  这些是**磁盘上的结构化事实**，比对话历史可靠；与历史冲突时以此为准。\n"
                f"相机实测: {json.dumps(cam, ensure_ascii=False)}\n"
                f"已确认物体({len(rows)}): " + ("\n  ".join(rows) if rows else "(还没有)") + "\n"
                f"未关闭问题({len(issues)}): " + ("\n  ".join(ill) if ill else "(无)") + "\n"
                "</belief>")
