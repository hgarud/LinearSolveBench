"""Small operator/developer command line for the benchmark."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import tempfile
from collections.abc import Sequence

from .compiler import build_candidate, validate_source
from .dataset import (
    dataset_summary,
    load_prepared,
    load_prepared_manifest,
    load_split,
    prepare_split,
)
from .runner import evaluate
from .runtime import HYPRE_REPOSITORY, build_runtime, load_runtime
from .scoring import calibration_from_report, score_report, validate_calibration


def _json_file(path: pathlib.Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _emit(value: object, output: pathlib.Path | None = None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(output)


def _calibration(
    calibration_path: pathlib.Path | None,
) -> dict[str, object] | None:
    if calibration_path is None:
        return None
    calibration = _json_file(calibration_path)
    validate_calibration(calibration)
    return calibration


def _dataset_command(args: argparse.Namespace) -> int:
    if args.dataset_action == "summary":
        _emit(dataset_summary())
    elif args.dataset_action == "list":
        manifest = load_split(args.split)
        _emit(
            {
                "split": args.split,
                "case_count": manifest["case_count"],
                "cases": [
                    {
                        key: case[key]
                        for key in ("index", "case_id", "group", "name", "n", "nnz")
                    }
                    for case in manifest["cases"]
                ],
            }
        )
    elif args.dataset_action == "prepare":
        key = args.rhs_key_file.read_bytes() if args.rhs_key_file else None
        _emit(
            prepare_split(
                args.split,
                args.output,
                rhs_key=key,
                cache=args.cache,
                case_ids=args.case,
            )
        )
    return 0


def _runtime_command(args: argparse.Namespace) -> int:
    runtime = build_runtime(args.output, hypre_source=args.hypre_source, jobs=args.jobs)
    _emit(
        {
            "runtime": str(runtime.root),
            "manifest_sha256": runtime.manifest_sha256,
        }
    )
    return 0


def _candidate_command(args: argparse.Namespace) -> int:
    if args.candidate_action == "source":
        validate_source(args.source)
        _emit({"ok": True, "source": str(args.source.resolve())})
        return 0
    runtime = load_runtime(args.runtime)
    if args.candidate_action == "build":
        build = build_candidate(args.source, args.output, runtime=runtime)
        _emit(
            {
                "ok": True,
                "executable": str(build.executable),
                "source_sha256": build.source_sha256,
                "object_sha256": build.object_sha256,
                "executable_sha256": build.executable_sha256,
                "compile_wall_s": build.compile_wall_s,
                "undefined_symbols": build.undefined_symbols,
                "exported_symbols": build.exported_symbols,
            }
        )
        return 0
    with tempfile.TemporaryDirectory(prefix="lsb-validate-") as temporary:
        build = build_candidate(
            args.source, pathlib.Path(temporary) / "candidate", runtime=runtime
        )
        _emit(
            {
                "ok": True,
                "source_sha256": build.source_sha256,
                "object_sha256": build.object_sha256,
                "executable_sha256": build.executable_sha256,
                "compile_wall_s": build.compile_wall_s,
                "undefined_symbols": build.undefined_symbols,
                "exported_symbols": build.exported_symbols,
            }
        )
    return 0


def _run_command(args: argparse.Namespace) -> int:
    if not math.isfinite(args.deadline) or not 0.0 < args.deadline <= 120.0:
        raise ValueError("case deadline must lie in (0, 120]")
    runtime = load_runtime(args.runtime)
    prepared_manifest = load_prepared_manifest(args.cases)
    systems = load_prepared(args.cases)
    calibration = _calibration(args.calibration)
    deadlines = None
    if calibration is not None:
        observed_cases = [str(case["case_id"]) for case in calibration["cases"]]
        expected_cases = [system.case_id for system in systems]
        if observed_cases != expected_cases:
            raise ValueError("calibration case order differs from prepared cases")
        expected_identity = {
            "venue_id": "local-uncontrolled",
            "runtime_manifest_sha256": runtime.manifest_sha256,
            "dataset_manifest_sha256": prepared_manifest["source_manifest_sha256"],
            "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
        }
        mismatches = sorted(
            key
            for key, expected in expected_identity.items()
            if calibration.get(key) != expected
        )
        if mismatches:
            raise ValueError("calibration identity mismatch: " + ", ".join(mismatches))
        deadlines = {
            str(case["case_id"]): float(case["deadline_s"])
            for case in calibration["cases"]
        }
    with tempfile.TemporaryDirectory(prefix="lsb-run-") as temporary:
        build = build_candidate(
            args.source, pathlib.Path(temporary) / "candidate", runtime=runtime
        )
        report = evaluate(
            build.executable,
            systems,
            deadline_s=args.deadline,
            deadline_by_case=deadlines,
            source_sha256=build.source_sha256,
            executable_sha256=build.executable_sha256,
            runtime_manifest_sha256=runtime.manifest_sha256,
            dataset_manifest_sha256=str(prepared_manifest["source_manifest_sha256"]),
            prepared_manifest_sha256=str(prepared_manifest["manifest_sha256"]),
        )
    _emit(report, args.output)
    return 0 if report["solved_count"] == report["case_count"] else 1


def _calibrate_command(args: argparse.Namespace) -> int:
    if args.maximum_deadline > 120.0:
        raise ValueError("v1 calibration deadlines cannot exceed 120 seconds")
    calibration = calibration_from_report(
        _json_file(args.report),
        deadline_multiplier=args.multiplier,
        minimum_deadline_s=args.minimum_deadline,
        maximum_deadline_s=args.maximum_deadline,
    )
    _emit(calibration, args.output)
    return 0


def _score_command(args: argparse.Namespace) -> int:
    score = score_report(_json_file(args.report), _json_file(args.calibration))
    _emit(score, args.output)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="linear-solver-bench")
    commands = root.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="inspect or prepare corpus data")
    dataset_commands = dataset.add_subparsers(dest="dataset_action", required=True)
    dataset_commands.add_parser("summary")
    listing = dataset_commands.add_parser("list")
    listing.add_argument("--split", choices=("dev", "ranked"), default="dev")
    prepare = dataset_commands.add_parser("prepare")
    prepare.add_argument("--split", choices=("dev", "ranked"), default="dev")
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    prepare.add_argument("--cache", type=pathlib.Path)
    prepare.add_argument("--rhs-key-file", type=pathlib.Path)
    prepare.add_argument("--case", action="append")

    runtime = commands.add_parser("runtime", help="build the pinned HYPRE runtime")
    runtime_commands = runtime.add_subparsers(dest="runtime_action", required=True)
    runtime_build = runtime_commands.add_parser("build")
    runtime_build.add_argument("--output", type=pathlib.Path, required=True)
    runtime_build.add_argument("--hypre-source", default=HYPRE_REPOSITORY)
    runtime_build.add_argument("--jobs", type=int, default=2)

    candidate = commands.add_parser("candidate", help="validate candidate source")
    candidate_commands = candidate.add_subparsers(
        dest="candidate_action", required=True
    )
    source = candidate_commands.add_parser("source", help="run static validation only")
    source.add_argument("source", type=pathlib.Path)
    validate = candidate_commands.add_parser(
        "validate", help="compile, audit, and link"
    )
    validate.add_argument("source", type=pathlib.Path)
    validate.add_argument("--runtime", type=pathlib.Path, required=True)
    build = candidate_commands.add_parser(
        "build", help="build into a persistent workspace"
    )
    build.add_argument("source", type=pathlib.Path)
    build.add_argument("--runtime", type=pathlib.Path, required=True)
    build.add_argument("--output", type=pathlib.Path, required=True)

    run = commands.add_parser("run", help="evaluate a candidate locally")
    run.add_argument("source", type=pathlib.Path)
    run.add_argument("--runtime", type=pathlib.Path, required=True)
    run.add_argument("--cases", type=pathlib.Path, required=True)
    run.add_argument("--output", type=pathlib.Path, required=True)
    run.add_argument("--deadline", type=float, default=120.0)
    run.add_argument("--calibration", type=pathlib.Path)

    calibrate = commands.add_parser("calibrate", help="freeze reference timings")
    calibrate.add_argument("report", type=pathlib.Path)
    calibrate.add_argument("--output", type=pathlib.Path, required=True)
    calibrate.add_argument("--multiplier", type=float, default=4.0)
    calibrate.add_argument("--minimum-deadline", type=float, default=1.0)
    calibrate.add_argument("--maximum-deadline", type=float, default=120.0)

    score = commands.add_parser("score", help="score an evaluation report")
    score.add_argument("report", type=pathlib.Path)
    score.add_argument("calibration", type=pathlib.Path)
    score.add_argument("--output", type=pathlib.Path)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "dataset":
        return _dataset_command(args)
    if args.command == "runtime":
        return _runtime_command(args)
    if args.command == "candidate":
        return _candidate_command(args)
    if args.command == "run":
        return _run_command(args)
    if args.command == "calibrate":
        return _calibrate_command(args)
    if args.command == "score":
        return _score_command(args)
    raise AssertionError("unreachable command")
