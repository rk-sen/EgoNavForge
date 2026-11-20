"""LLM client: OpenAI-compatible chat completions (Kimi / Moonshot by default).

The model is the ONLY external service. Images are sent inline as base64 data URLs when
`vision=True`; if the endpoint rejects images, `text_only` is flipped automatically and the run
continues on numbers alone (every check is numeric on purpose).
"""
from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, model: str, api_base: str, api_key_env: str = "zhesuan_API_KEY",
                 temperature: float = 0.35, max_output_tokens: int = 8192, vision: bool = True):
        if OpenAI is None:
            raise LLMError("openai package missing: pip install openai")
        key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError(f"set {api_key_env} in the environment")
        # The proxy can leave a streaming/blocking request open past its own window; a
        # finite client timeout turns that into a normal retry instead of a dead run.
        self.client = OpenAI(api_key=key, base_url=api_base, timeout=150.0, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.vision = vision
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _image_part(self, path: Path) -> dict[str, Any] | None:
        """Send a compact JPEG, not a 300 KB raw PNG.

        S1 now needs 8 frames from across the timeline (v5/v6 only sent the first few and lost the
        green ball/orange sofa). Eight raw PNGs made the proxy time out, so resize to <= 640 px and
        encode JPEG q=84. The model still sees the whole timeline; the bytes stay manageable.
        """
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            im.thumbnail((560, 560), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=80, optimize=True)
            data = base64.b64encode(buf.getvalue()).decode()
            mime = "image/jpeg"
        except Exception:
            try:
                data = base64.b64encode(path.read_bytes()).decode()
            except OSError:
                return None
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}

    def chat(self, messages: list[dict[str, Any]], image_paths: list[str] | None = None,
             retries: int = 1) -> str:
        payload = [dict(m) for m in messages]
        if image_paths and self.vision:
            parts = [{"type": "text", "text": payload[-1]["content"]}]
            for p in image_paths[:12]:
                part = self._image_part(Path(p))
                if part:
                    parts.append(part)
            payload[-1] = {"role": payload[-1]["role"], "content": parts}

        last_err = None
        for attempt in range(retries + 1):
            try:
                self.calls += 1
                kwargs: dict[str, Any] = {
                    "model": self.model, "messages": payload,
                    "max_tokens": self.max_output_tokens,
                }
                # kimi-k3 rejects temperature and is already a reasoning model: keep its default
                # thinking budget. Older kimi-k2.6 on this proxy needs thinking disabled or a
                # long plan call can exceed the proxy's ~90 s response window.
                if self.temperature is not None and not self.model.startswith("kimi-k3"):
                    kwargs["temperature"] = self.temperature
                if not self.model.startswith("kimi-k3"):
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                resp = self.client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                if usage:
                    self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                if "image" in msg.lower() and self.vision:
                    self.vision = False
                    return self.chat(messages, None, retries=retries)
                if attempt < retries:
                    time.sleep(2 + 3 * attempt)
                    continue
        raise LLMError(f"LLM call failed: {last_err}")

    def summarize(self, text: str) -> str:
        return self.chat([
            {"role": "system", "content": "你是运行记录压缩器，只输出状态摘要，不解释。"},
            {"role": "user", "content": text[:60000]},
        ])


def estimate_chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(json.dumps(m.get("content", ""), ensure_ascii=False)) for m in messages)
