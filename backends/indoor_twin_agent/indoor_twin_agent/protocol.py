"""Wire protocol: a turn may carry a SHORT BATCH of independent tool calls.

Why batches now (v3)
--------------------
v2 allowed exactly one call per turn, which turned every "do X and also write the file X produces"
into two turns with a re-read in between. The model has no memory, so the re-read was inevitable.

Rules that keep batching safe:
  * a batch is executed strictly in order, and execution STOPS at the first failure
  * at most MAX_CALLS calls (default 4); each must be a full JSON object
  * the model is told: batch only actions that do NOT depend on each other's output;
    anything that needs a number from a previous call goes in the NEXT turn
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MAX_CALLS = 4
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
PLAN_RE = re.compile(r"<plan>\s*(.*?)\s*</plan>", re.S)
FENCE_RE = re.compile(r"```(?:json|tool)?\s*(\{.*?\})\s*```", re.S)


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    raw: str = ""


class ProtocolError(Exception):
    pass


def _balanced_objects(text: str) -> list[str]:
    """Every top-level balanced {...} substring, in order."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "{":
            depth, in_str, esc, start = 0, False, False, i
            for j in range(i, len(text)):
                ch = text[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[start:j + 1])
                        i = j + 1
                        break
            else:
                break
        else:
            i += 1
    return out


def _loads(body: str) -> dict[str, Any]:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        fixed = body.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        try:
            return json.loads(fixed)
        except Exception as exc:
            raise ProtocolError(f"tool call is not valid JSON ({exc}). Send strict JSON, no comments, no trailing commas.") from exc


def _one(obj: dict[str, Any], known: set[str]) -> ToolCall:
    tool = obj.get("tool") or obj.get("name") or obj.get("action")
    args = obj.get("args") or obj.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            raise ProtocolError("args must be an object, not a string")
    if not isinstance(tool, str) or not tool:
        raise ProtocolError('missing "tool" field')
    if tool not in known:
        raise ProtocolError(f"unknown tool {tool!r}; available: {', '.join(sorted(known))}")
    if not isinstance(args, dict):
        raise ProtocolError("args must be an object")
    return ToolCall(tool=tool, args=args)


def parse_calls(text: str, known_tools: set[str], max_calls: int = MAX_CALLS) -> list[ToolCall]:
    bodies: list[str]
    m = CALL_RE.findall(text)
    if m:
        bodies = m
    else:
        bodies = [b for b in _balanced_objects(text) if '"tool"' in b or "'tool'" in b]
        if not bodies:
            fenced = FENCE_RE.search(text)
            bodies = [fenced.group(1)] if fenced else []
    if not bodies:
        raise ProtocolError(
            "no tool call found. Reply with "
            '<tool_call>{"tool":"<name>","args":{...}}</tool_call> (1–4 such blocks for independent actions)')

    if len(bodies) > max_calls:
        raise ProtocolError(f"{len(bodies)} calls in one turn exceeds the limit of {max_calls}; "
                            f"split into batches (dependent steps must wait for their result)")
    calls = []
    thought = _thought(text)
    for body in bodies:
        obj = _loads(body)
        call = _one(obj, known_tools)
        call.thought = thought
        call.raw = body
        calls.append(call)
    return calls


def parse(text: str, known_tools: set[str]) -> ToolCall:
    return parse_calls(text, known_tools)[0]


def _thought(text: str) -> str:
    m = re.search(r"<thought>\s*(.*?)\s*</thought>", text, re.S)
    if m:
        return m.group(1).strip()[:400]
    p = PLAN_RE.search(text)
    return p.group(1).strip()[:400] if p else ""
