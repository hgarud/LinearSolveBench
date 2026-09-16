"""Participant commands for LinearSolverBench."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from collections.abc import Sequence

from .benchmark import evaluate, repository_root
from .submission import package


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="linear-solver-bench")
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="copy the starter solver")
    init.add_argument("output", type=pathlib.Path)

    run = commands.add_parser("run", help="evaluate a solver")
    run.add_argument("source", type=pathlib.Path)
    run.add_argument("--family", choices=("ns_mesh_pde", "flash"), required=True)
    run.add_argument("--venue", choices=("local", "modal"), default="modal")
    run.add_argument("--case", action="append", dest="case_ids")
    run.add_argument("--output", type=pathlib.Path, default=pathlib.Path("result.json"))
    run.add_argument("--cache", type=pathlib.Path)

    submit = commands.add_parser("submit", help="create a pull-request bundle")
    submit.add_argument("source", type=pathlib.Path)
    submit.add_argument("--name", required=True)
    submit.add_argument(
        "--family", choices=("ns_mesh_pde", "flash"), action="append", required=True
    )
    submit.add_argument("--license", dest="license_name", required=True)
    submit.add_argument("--submitter", required=True)
    submit.add_argument("--output", type=pathlib.Path, required=True)
    return root


def _run(args: argparse.Namespace) -> int:
    if args.command == "init":
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root() / "examples" / "solver.c", args.output)
        print(args.output)
        return 0
    if args.command == "submit":
        package(
            args.source,
            args.output,
            name=args.name,
            families=args.family,
            license_name=args.license_name,
            submitter=args.submitter,
        )
        print(args.output)
        return 0
    report = evaluate(
        args.source,
        family=args.family,
        venue=args.venue,
        case_ids=args.case_ids,
        output=args.output,
        cache=args.cache,
        progress=lambda message: print(message, file=sys.stderr),
    )
    for case in report["cases"]:
        speedup = "-" if case["speedup"] is None else f"{case['speedup']:.3f}x"
        print(f"{case['case_id']}: {'PASS' if case['passed'] else 'FAIL'} {speedup}")
    aggregate = report["geometric_mean_speedup"]
    if aggregate is not None:
        print(f"Geometric mean speedup: {aggregate:.3f}x")
    print(f"Report: {args.output}")
    return 0 if report["all_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(parser().parse_args(argv))
    except KeyboardInterrupt:
        print("linear-solver-bench: cancelled", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"linear-solver-bench: {error}", file=sys.stderr)
        return 2
