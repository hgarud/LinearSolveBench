"""Small operator/developer command line for the benchmark."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
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
from .families import family_listing, resolve_track
from .runner import evaluate
from .runtime import HYPRE_REPOSITORY, build_runtime, load_runtime
from .scoring import calibration_from_report, score_report, validate_calibration


def _release_path(value: str) -> pathlib.Path:
    from .paths import data_dir

    path = pathlib.Path(value).expanduser()
    if path.is_file():
        return path
    if path.name != value or value in {".", ".."}:
        raise ValueError("release must name an installed manifest or an existing file")
    installed = data_dir() / f"{value}.json"
    if not installed.is_file():
        raise ValueError(
            f"release {value!r} is unavailable; supply a release manifest path"
        )
    return installed


def _check_selection(args: argparse.Namespace, release: dict) -> None:
    family = getattr(args, "family", None) or release["family"]
    track = getattr(args, "track", None) or release["track"]
    selected = resolve_track(family, track)
    if (selected.family, selected.track) != (release["family"], release["track"]):
        raise ValueError("requested family/track differs from the prepared release")
    split = getattr(args, "split", None)
    if split is not None and split != release["split"]:
        raise ValueError("requested split differs from the release manifest")


def _pilot_dataset_command(args: argparse.Namespace) -> int:
    from .manifests import load_release
    from .pilot_dataset import prepare_release

    path = _release_path(args.release)
    release = load_release(path)
    _check_selection(args, release)
    if args.dataset_action in {"summary", "list"}:
        value = {
            key: release[key]
            for key in (
                "release_id",
                "family",
                "track",
                "split",
                "contracts",
                "execution",
                "manifest_sha256",
            )
        }
        value["case_count"] = len(release["cases"])
        if args.dataset_action == "list":
            value["cases"] = release["cases"]
        _emit(value)
    else:
        _emit(
            prepare_release(
                path,
                args.output,
                rhs_key=args.rhs_key_file.read_bytes() if args.rhs_key_file else None,
                cache=args.cache,
                offline=args.offline,
                case_ids=args.case,
            )
        )
    return 0


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
    if args.dataset_action == "families":
        _emit({"tracks": family_listing()})
        return 0
    if args.release:
        return _pilot_dataset_command(args)
    if args.family or args.track:
        raise ValueError(
            "pilot families require --release with a published or draft manifest"
        )
    if getattr(args, "offline", False):
        raise ValueError("--offline is supported by pilot release preparation")
    if getattr(args, "split", None) is None:
        args.split = "dev"
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
    if _json_file(args.cases / "manifest.json").get("schema_version") == 2:
        return _run_pilot_command(args)
    if args.family or args.track or args.release:
        raise ValueError("pilot selectors cannot reinterpret v1 prepared cases")
    args.deadline = 120.0 if args.deadline is None else args.deadline
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


def _run_pilot_command(args: argparse.Namespace) -> int:
    from .manifests import load_release
    from .pilot_dataset import iter_pilot_prepared, load_pilot_prepared_manifest
    from .pilot_runner import evaluate_pilot
    from .pilot_scoring import score_pilot_report, validate_replay_reference

    prepared = load_pilot_prepared_manifest(args.cases)
    release = prepared["source_release"]
    _check_selection(args, release)
    if (
        args.release
        and load_release(_release_path(args.release))["manifest_sha256"]
        != release["manifest_sha256"]
    ):
        raise ValueError("selected release differs from prepared cases")
    if (
        args.deadline is not None
        and args.deadline != release["execution"]["case_timeout_seconds"]
    ):
        raise ValueError(
            "pilot deadlines are frozen in the release and cannot be overridden"
        )
    reference = _json_file(args.calibration) if args.calibration else None
    if reference is not None:
        if release["track"] != "replay":
            raise ValueError("NS coverage does not use a timing reference")
        validate_replay_reference(reference)
        anchor = reference["reference_report"]
        if (
            anchor["release_manifest_sha256"] != release["manifest_sha256"]
            or anchor["prepared_manifest_sha256"] != prepared["manifest_sha256"]
        ):
            raise ValueError("reference numerical release differs from prepared cases")
        if anchor["venue"]["id"] != "local-uncontrolled":
            raise ValueError("local execution requires a local reference")
    runtime = load_runtime(args.runtime)
    if (
        reference
        and reference["reference_report"]["runtime_manifest_sha256"]
        != runtime.manifest_sha256
    ):
        raise ValueError("reference runtime differs from selected runtime")
    with tempfile.TemporaryDirectory(prefix="lsb-pilot-") as temporary:
        build = build_candidate(
            args.source, pathlib.Path(temporary) / "candidate", runtime=runtime
        )
        report = evaluate_pilot(
            build.executable,
            iter_pilot_prepared(args.cases),
            prepared,
            source_sha256=build.source_sha256,
            executable_sha256=build.executable_sha256,
            runtime_manifest_sha256=runtime.manifest_sha256,
        )
    if reference is not None:
        score_pilot_report(report, reference)
    _emit(report, args.output)
    # Partial coverage is a valid result. Scoring reports candidate eligibility.
    return 0


def _calibrate_command(args: argparse.Namespace) -> int:
    report = _json_file(args.report)
    if report.get("schema_version") == 2:
        from .pilot_scoring import replay_reference_from_report

        if (args.multiplier, args.minimum_deadline, args.maximum_deadline) != (
            4.0,
            1.0,
            120.0,
        ):
            raise ValueError(
                "pilot reference qualification does not alter frozen deadlines"
            )
        _emit(replay_reference_from_report(report), args.output)
        return 0
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
    report = _json_file(args.report)
    calibration = _json_file(args.calibration) if args.calibration else None
    if report.get("schema_version") == 2:
        from .pilot_scoring import score_pilot_report

        score = score_pilot_report(report, calibration)
    else:
        if calibration is None:
            raise ValueError("v1 scoring requires a calibration artifact")
        score = score_report(report, calibration)
    _emit(score, args.output)
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="linear-solver-bench")
    commands = root.add_subparsers(dest="command", required=True)

    dataset = commands.add_parser("dataset", help="inspect or prepare corpus data")
    dataset_commands = dataset.add_subparsers(dest="dataset_action", required=True)
    summary = dataset_commands.add_parser("summary")
    dataset_commands.add_parser("families")
    listing = dataset_commands.add_parser("list")
    listing.add_argument("--split", choices=("dev", "ranked"))
    prepare = dataset_commands.add_parser("prepare")
    prepare.add_argument("--split", choices=("dev", "ranked"))
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    prepare.add_argument("--cache", type=pathlib.Path)
    prepare.add_argument("--rhs-key-file", type=pathlib.Path)
    prepare.add_argument("--case", action="append")
    prepare.add_argument("--offline", action="store_true")
    for action in (summary, listing, prepare):
        action.add_argument(
            "--release", help="installed release ID or explicit draft manifest path"
        )
        action.add_argument("--family")
        action.add_argument("--track")

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
    run.add_argument("--deadline", type=float)
    run.add_argument("--calibration", type=pathlib.Path)
    run.add_argument("--release")
    run.add_argument("--family")
    run.add_argument("--track")

    calibrate = commands.add_parser("calibrate", help="freeze reference timings")
    calibrate.add_argument("report", type=pathlib.Path)
    calibrate.add_argument("--output", type=pathlib.Path, required=True)
    calibrate.add_argument("--multiplier", type=float, default=4.0)
    calibrate.add_argument("--minimum-deadline", type=float, default=1.0)
    calibrate.add_argument("--maximum-deadline", type=float, default=120.0)

    score = commands.add_parser("score", help="score an evaluation report")
    score.add_argument("report", type=pathlib.Path)
    score.add_argument("calibration", type=pathlib.Path, nargs="?")
    score.add_argument("--output", type=pathlib.Path)
    return root


def _main(argv: Sequence[str] | None = None) -> int:
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


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except (ValueError, OSError) as exc:
        print(f"linear-solver-bench: {exc}", file=sys.stderr)
        return 2
