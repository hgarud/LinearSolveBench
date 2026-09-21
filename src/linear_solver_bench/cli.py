"""Participant commands for LinearSolverBench."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

from .benchmark import evaluate
from .families import FAMILY_SELECTORS
from .submission import package


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="linear-solver-bench")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="evaluate a solver")
    run.add_argument("source", type=pathlib.Path)
    run.add_argument("--family", choices=FAMILY_SELECTORS, required=True)
    run.add_argument("--venue", choices=("local", "modal"), default="modal")
    run.add_argument("--case", action="append", dest="case_ids")
    run.add_argument("--output", type=pathlib.Path, default=pathlib.Path("result.json"))
    run.add_argument("--cache", type=pathlib.Path)

    submit = commands.add_parser("submit", help="create a pull-request bundle")
    submit.add_argument("source", type=pathlib.Path)
    submit.add_argument("--name", required=True)
    submit.add_argument(
        "--family",
        choices=FAMILY_SELECTORS,
        action="append",
        required=True,
    )
    submit.add_argument("--submitter", required=True)
    submit.add_argument("--output", type=pathlib.Path, required=True)
    return root


def _run(args: argparse.Namespace) -> int:
    if args.command == "submit":
        package(
            args.source,
            args.output,
            name=args.name,
            families=args.family,
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
        suffix = (
            ""
            if "speedup" not in case
            else " " + ("-" if case["speedup"] is None else f"{case['speedup']:.3f}x")
        )
        print(f"{case['case_id']}: {'PASS' if case['passed'] else 'FAIL'}{suffix}")
    score = report["score"]
    if score["metric"] == "dev-cases-solved-v1":
        if report["complete"]:
            print(f"Dev cases solved: {score['solved_cases']}/{score['total_cases']}")
        else:
            print(
                "Evaluated cases solved: "
                f"{score['solved_cases']}/{score['evaluated_cases']} "
                f"(full dev set: {score['total_cases']})"
            )
    elif score["value"] is not None:
        print(f"Geometric mean speedup: {score['value']:.3f}x")
    print(f"Report: {args.output}")
    return 0 if report["family"] == "ns-mesh-pde" or report["all_passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(parser().parse_args(argv))
    except KeyboardInterrupt:
        print("linear-solver-bench: cancelled", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"linear-solver-bench: {error}", file=sys.stderr)
        return 2
