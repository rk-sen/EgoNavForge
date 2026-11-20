"""CLI: ego-nav-forge run --config configs/presets/....yaml"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ego-nav-forge", description="EgoNavForge lifecycle runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a YAML pipeline")
    run.add_argument("--config", "-c", type=Path, required=True)
    run.add_argument("--only", type=str, default="", help="Comma-separated episode ids")
    run.add_argument("--max-eps", type=int, default=0)
    run.add_argument("--work-dir", type=Path, default=None, help="Override work_dir")

    doc = sub.add_parser("doctor", help="Check imports / backends")
    listp = sub.add_parser("list-modules", help="List registered module types")

    args = p.parse_args(argv)

    if args.cmd == "list-modules":
        import ego_nav_forge.modules  # noqa: F401
        from ego_nav_forge.registry import registered

        for name in registered():
            print(name)
        return 0

    if args.cmd == "doctor":
        from ego_nav_forge.paths import (
            default_ckpt,
            ensure_backend_path,
            pi3_moge_root,
            project_root,
        )

        print("project_root", project_root())
        print("pi3_moge_root", pi3_moge_root(), "exists", pi3_moge_root().is_dir())
        ensure_backend_path()
        ok = True
        try:
            import torch

            print("torch", torch.__version__, "cuda", torch.cuda.is_available())
        except Exception as e:
            print("FAIL torch", e)
            ok = False
        ckpt = default_ckpt()
        print("ckpt", ckpt, "exists", ckpt.is_file())
        if not ckpt.is_file():
            ok = False
        if (project_root() / "backends" / "pipeline").exists():
            print("WARN backends/pipeline should not exist (legacy dump)")
            ok = False
        import ego_nav_forge.modules  # noqa: F401
        from ego_nav_forge.registry import registered

        print("modules", registered())
        print("see docs/MODULES.md for I/O contracts")
        print("doctor", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.cmd == "run":
        from ego_nav_forge.runner import Runner

        runner = Runner.from_yaml(args.config)
        if args.work_dir is not None:
            runner.cfg["work_dir"] = str(args.work_dir.resolve())
            runner.work_dir = args.work_dir.resolve()
            runner.work_dir.mkdir(parents=True, exist_ok=True)
        only = [x.strip() for x in args.only.split(",") if x.strip()] or None
        runner.run(only=only, max_eps=args.max_eps)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
