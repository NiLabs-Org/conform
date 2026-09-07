"""Command line entry point.

``conform`` is a thin wrapper over pytest. Anything you can do here you can do
by calling pytest directly, which is deliberate: contributors already know
pytest, and the suite should not invent a second way to run tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from conform import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conform",
        description="Conformance suite for LLM inference engines.",
    )
    parser.add_argument("--version", action="version", version=f"conform {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the suite and write the matrix")
    run.add_argument("-k", dest="keyword", help="Only run tests matching this expression")
    run.add_argument("--engine", help="Only run against this engine")
    run.add_argument("--out", default="results", help="Output directory (default: results)")
    run.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("engines", help="List engines configured in conform.toml")

    args = parser.parse_args(argv)

    if args.command == "engines":
        return _list_engines()
    return _run(args)


def _list_engines() -> int:
    from conform.registry import config_path, load_specs

    try:
        specs = load_specs()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{config_path()}:")
    for spec in specs:
        target = spec.base_url or "in-process"
        print(f"  {spec.name:<14} {spec.kind:<10} {spec.model}  ({target})")
    return 0


def _run(args: argparse.Namespace) -> int:
    import pytest

    tests = Path(__file__).resolve().parents[2] / "tests"
    argv = [
        str(tests) if tests.is_dir() else "tests",
        "-p",
        "conform.report",
        f"--conform-out={args.out}",
    ]
    if args.verbose:
        argv.append("-v")
    if args.keyword:
        argv += ["-k", args.keyword]
    if args.engine:
        # Engine ids are the first bracketed parameter, so a keyword match on
        # the name selects that engine's rows.
        argv += ["-k", args.engine] if not args.keyword else []

    return int(pytest.main(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
