"""The complete benchmark workflow: load, build, run, verify, report."""

from __future__ import annotations

import json
import math
import pathlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping

from .assets import asset_root
from .data import _iter_validated_cases, load_manifest
from .families import MAGNETIC_DIFFUSION_FLASH, NS_MESH_PDE
from .models import BenchmarkCase, RawExecution
from .verify import RunResult, verify

Progress = Callable[[str], None]


def default_cache() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "linear-solver-bench"


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _reference_result(
    raw: RawExecution | None, case: BenchmarkCase, family: str
) -> RunResult | None:
    if raw is None:
        return None
    result = verify(raw, case, family)
    if not result.passed:
        raise RuntimeError(
            f"fixed reference failed on {case.case_id}: {result.failure}"
        )
    return result


def _case_row(
    case: BenchmarkCase, candidate: RunResult, reference: RunResult | None
) -> dict[str, object]:
    """Construct the same report row for local and batch results."""
    row = {
        "case_id": case.case_id,
        "role": case.role,
        "input_sha256": case.input_sha256,
        "passed": candidate.passed,
        "candidate_seconds": candidate.elapsed_seconds,
        "candidate": candidate.to_dict(),
    }
    if reference is not None:
        row.update(
            {
                "reference_seconds": reference.elapsed_seconds,
                "speedup": (
                    reference.elapsed_seconds / candidate.elapsed_seconds
                    if candidate.passed
                    else None
                ),
                "reference": reference.to_dict(),
            }
        )
    return row


def _report(
    manifest: Mapping[str, object],
    identities: Mapping[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Score rows using a manifest already validated by the loader."""
    complete = [row["case_id"] for row in rows] == [
        case["case_id"] for case in manifest["cases"]
    ]
    all_passed = all(row["passed"] for row in rows)
    scored_rows = [row for row in rows if row["role"] == "scored"]
    scoring_contract = manifest["contracts"]["scoring"]
    if manifest["family"] == NS_MESH_PDE.id:
        solved = sum(bool(row["passed"]) for row in scored_rows)
        total = sum(
            case.get("role", "scored") == "scored" for case in manifest["cases"]
        )
        score = {
            "metric": scoring_contract,
            "value": solved if complete else None,
            "solved_cases": solved,
            "evaluated_cases": len(scored_rows),
            "total_cases": total,
        }
        aggregate = None
    else:
        speedups = [row["speedup"] for row in scored_rows if row["speedup"] is not None]
        aggregate = (
            _geometric_mean(speedups) if complete and all_passed and speedups else None
        )
        score = {"metric": scoring_contract, "value": aggregate}
    return {
        "schema_version": 1,
        "benchmark": "linear-solver-bench",
        "official": complete
        and identities.get("venue", {}).get("id") == "modal-cpu-v1",
        "release_id": manifest["release_id"],
        "family": manifest["family"],
        "split": manifest["split"],
        "manifest_sha256": manifest["manifest_sha256"],
        "complete": complete,
        "all_passed": all_passed,
        "score": score,
        "geometric_mean_speedup": aggregate,
        **dict(identities),
        "cases": rows,
    }


def _local_evaluation(
    candidate: pathlib.Path,
    reference: pathlib.Path | None,
    manifest: Mapping[str, object],
    cases: Iterator[BenchmarkCase],
    cache: pathlib.Path,
    progress: Progress,
) -> dict[str, object]:
    from .build import (
        build_identities,
        build_runtime,
        build_solvers,
        load_runtime,
        run_solver,
    )

    progress("Preparing the pinned HYPRE runtime")
    runtime_path = cache / "runtime"
    runtime = (
        load_runtime(runtime_path)
        if runtime_path.exists()
        else build_runtime(runtime_path)
    )
    with tempfile.TemporaryDirectory(prefix="lsb-build-") as temporary:
        progress(
            "Compiling the fixed reference and candidate"
            if reference is not None
            else "Compiling the candidate"
        )
        builds = build_solvers(candidate, reference, pathlib.Path(temporary), runtime)
        identities = {
            "venue": {"id": "local-uncontrolled"},
            **build_identities(builds, runtime),
        }
        rows = []
        for index, case in enumerate(cases, 1):
            progress(f"Running {case.case_id} ({index})")
            reference_result = _reference_result(
                run_solver(builds["reference"].executable, case.input)
                if "reference" in builds
                else None,
                case,
                manifest["family"],
            )
            candidate_result = verify(
                run_solver(builds["candidate"].executable, case.input),
                case,
                manifest["family"],
            )
            rows.append(_case_row(case, candidate_result, reference_result))
    return _report(manifest, identities, rows)


def _modal_evaluation(
    candidate: pathlib.Path,
    reference: pathlib.Path | None,
    manifest: Mapping[str, object],
    cases: Iterator[BenchmarkCase],
    cache: pathlib.Path,
    progress: Progress,
) -> dict[str, object]:
    from .batch import prepare_inputs
    from .modal import open_session

    progress("Preparing benchmark inputs (downloads and validation)")
    started = time.monotonic()
    prepared = []
    for case in cases:
        prepared.append(case)
        progress(f"Prepared {case.case_id} ({len(prepared)})")
    progress("Preparing the reusable numeric input bundle")
    bundle = prepare_inputs(prepared, cache / "input-bundles")
    statistics = {"input_prepare_seconds": time.monotonic() - started}
    progress("Starting the Modal CPU sandbox")
    started = time.monotonic()
    with open_session(candidate, reference, manifest["execution"]) as session:
        statistics["sandbox_start_seconds"] = time.monotonic() - started
        progress(
            "Compiling the fixed reference and candidate"
            if reference is not None
            else "Compiling the candidate"
        )
        started = time.monotonic()
        identities = session.build()
        statistics["build_seconds"] = time.monotonic() - started
        executions, transport = session.run_cases(bundle, progress)
        statistics.update(transport)
        started = time.monotonic()
        progress("Verifying the returned solutions")
        rows = []
        for case, raw in zip(prepared, executions, strict=True):
            reference_result = _reference_result(
                raw.get("reference"), case, manifest["family"]
            )
            candidate_result = verify(raw["candidate"], case, manifest["family"])
            rows.append(_case_row(case, candidate_result, reference_result))
        statistics["verification_seconds"] = time.monotonic() - started
    report = _report(manifest, identities, rows)
    report["execution_stats"] = statistics
    return report


def evaluate(
    source: str | pathlib.Path,
    *,
    family: str,
    venue: str = "modal",
    case_ids: Iterable[str] | None = None,
    output: str | pathlib.Path | None = None,
    cache: str | pathlib.Path | None = None,
    progress: Progress | None = None,
) -> dict[str, object]:
    """Evaluate one C source file against a public development benchmark."""
    if venue not in {"local", "modal"}:
        raise ValueError("venue must be 'local' or 'modal'")
    emit = progress or (lambda _message: None)
    source = pathlib.Path(source).expanduser()
    if not source.is_file():
        raise ValueError(f"candidate source does not exist: {source}")
    cache_path = pathlib.Path(cache).expanduser() if cache else default_cache()
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(family)
    selected = tuple(case_ids) if case_ids is not None else None
    cases = _iter_validated_cases(manifest, cache_path / "downloads", selected)
    reference = None
    if manifest["family"] == MAGNETIC_DIFFUSION_FLASH.id:
        reference = asset_root("reference") / "reference" / "solver.c"
        if not reference.is_file():
            raise RuntimeError("fixed reference solver is missing")

    # Snapshot source so edits during a long run cannot change the candidate.
    with tempfile.TemporaryDirectory(prefix="lsb-source-") as temporary:
        snapshot = pathlib.Path(temporary) / "solver.c"
        shutil.copyfile(source, snapshot)
        report = (
            _local_evaluation(snapshot, reference, manifest, cases, cache_path, emit)
            if venue == "local"
            else _modal_evaluation(
                snapshot, reference, manifest, cases, cache_path, emit
            )
        )
    if output is not None:
        destination = pathlib.Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report
