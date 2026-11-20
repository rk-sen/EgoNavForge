"""Resolve project / implementation package paths."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pi3_moge_root() -> Path:
    """Existing pi3_moge_lerobot release (weights + third_party + pipeline code)."""
    env = os.environ.get("EGO_NAV_FORGE_PI3_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # default: sibling release next to this repo
    sibling = project_root().parent / "pi3_moge_lerobot"
    if sibling.is_dir():
        return sibling.resolve()
    # fallback under backends/ if user vendored a copy
    vendored = project_root() / "backends" / "pi3_moge_lerobot"
    return vendored.resolve()


def ensure_backend_path() -> Path:
    """Put Pi3 / MoGe / legacy helpers on sys.path (private to modules)."""
    root = pi3_moge_root()
    for p in (
        root / "pi3_moge_lerobot" / "pipeline",
        root / "third_party" / "Pi3",
        root / "third_party" / "MoGe",
        root / "third_party" / "utils3d",
        root / "third_party" / "pipeline",
    ):
        s = str(p)
        if p.exists() and s not in sys.path:
            sys.path.insert(0, s)
    return root


def default_ckpt() -> Path:
    p = project_root() / "backends" / "checkpoints" / "Pi3X_model.safetensors"
    if p.is_file():
        return p
    return pi3_moge_root() / "checkpoints" / "Pi3X_model.safetensors"


def default_moge() -> Path:
    for p in (
        project_root() / "backends" / "checkpoints" / "moge-2-vitl" / "model.pt",
        pi3_moge_root() / "checkpoints" / "moge-2-vitl" / "model.pt",
    ):
        if p.is_file():
            return p
    return Path("Ruicheng/moge-2-vitl")


def backends_root() -> Path:
    return project_root() / "backends"
