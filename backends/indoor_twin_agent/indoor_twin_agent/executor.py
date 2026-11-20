"""Local executor: the only bridge between model output and this machine.

Design rules
------------
* nothing is executed through a shell string; argv lists only, so no quoting/injection games
* binaries are allow-listed, cwd is jailed inside the run directory
* every call returns stdout/stderr tails *and* the list of files it created (mtime window),
  so the model is told what actually happened rather than what it hoped would happen
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ALLOWED = {"python", "python3", "blender", "ffmpeg", "ffprobe", "ls", "find", "du", "wc", "tar", "unzip"}


class ExecError(Exception):
    pass


class Executor:
    def __init__(self, root: Path, blender: str = "blender", max_seconds: int = 600, allow: set[str] | None = None):
        self.root = Path(root).resolve()
        self.blender = blender
        self.max_seconds = max_seconds
        self.allow = set(ALLOWED) | (allow or set())

    # ---------------------------------------------------------------- helpers
    def resolve(self, rel: str, must_exist: bool = False) -> Path:
        p = (self.root / rel).resolve() if not Path(rel).is_absolute() else Path(rel).resolve()
        if self.root not in p.parents and p != self.root:
            raise ExecError(f"path escapes run directory: {rel}")
        if must_exist and not p.exists():
            raise ExecError(f"not found: {rel}")
        return p

    def _snapshot(self) -> dict[str, float]:
        snap = {}
        for p in self.root.rglob("*"):
            if p.is_file() and "_tmp" not in p.parts:
                try:
                    snap[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        return snap

    def _new_files(self, before: dict[str, float], limit: int = 12) -> list[dict[str, Any]]:
        out = []
        for p in self.root.rglob("*"):
            if not p.is_file() or "_tmp" in p.parts:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if before.get(str(p)) != st.st_mtime:
                out.append({"path": str(p.relative_to(self.root)), "kb": round(st.st_size / 1024, 1)})
        out.sort(key=lambda d: d["path"])
        return out[:limit]

    def run(self, argv: list[str], timeout: int = 180, cwd: str = ".") -> dict[str, Any]:
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ExecError("argv must be a non-empty list of strings")
        bin_name = Path(argv[0]).name
        if bin_name not in self.allow:
            raise ExecError(f"binary not allowed: {bin_name} (allowed: {', '.join(sorted(self.allow))})")
        timeout = max(1, min(int(timeout), self.max_seconds))
        work = self.resolve(cwd)
        work.mkdir(parents=True, exist_ok=True)
        argv = [self.blender if bin_name == "blender" else argv[0]] + argv[1:]
        before = self._snapshot()
        t0 = time.time()
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        try:
            cp = subprocess.run(argv, cwd=str(work), capture_output=True, text=True,
                                timeout=timeout, env=env, errors="replace")
            rc, out, err = cp.returncode, cp.stdout, cp.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -9
            out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            err = ((exc.stderr or "") if isinstance(exc.stderr, str) else "") + f"\n[timeout after {timeout}s]"
        except FileNotFoundError:
            raise ExecError(f"executable not found: {argv[0]} — install it or pass --blender")
        result = {
            "ok": rc == 0,
            "returncode": rc,
            "seconds": round(time.time() - t0, 2),
            "stdout": out[-4000:],
            "stderr": err[-2000:] if not rc == 0 else err[-600:],
            "new_files": self._new_files(before),
            "argv": argv,
        }
        return result

    # ------------------------------------------------------------- blender
    def blender_py(self, script: str, args: list[str] | None = None, blend: str | None = None,
                   timeout: int = 300, factory_startup: bool = True) -> dict[str, Any]:
        self.resolve(script, must_exist=True)
        argv = ["blender", "-b"]
        if blend:
            argv.append(str(self.resolve(blend, must_exist=True)))   # absolute: Blender must not guess cwd
        elif factory_startup:
            argv += ["--factory-startup"]
        argv += ["--python", str(self.resolve(script, must_exist=True))]
        if args:
            argv += ["--", *args]
        return self.run(argv, timeout=timeout)
