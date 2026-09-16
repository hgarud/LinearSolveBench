"""The complete benchmark workflow: load, build, run, verify, report."""

from __future__ import annotations

import json
import math
import pathlib
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping

from .assets import asset_root
from .data import iter_cases, load_manifest
from .models import BenchmarkCase, RawExecution
from .verify import verify

Progress = Callable[[str], None]


def repository_root() -> pathlib.Path:
    return asset_root("reference", "examples")


def default_cache() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "linear-solver-bench"


def _geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _case_rows(
    family: str,
    cases: Iterator[BenchmarkCase],
    run_reference: Callable[[object], RawExecution],
    run_candidate: Callable[[object], RawExecution],
    progress: Progress,
) -> list[dict[str, object]]:
    rows = []
    for index, case in enumerate(cases, 1):
        progress(f"Running {case.case_id} ({index})")
        reference = verify(run_reference(case.input), case, family)
        if not reference.passed:
            raise RuntimeError(
                f"fixed reference failed on {case.case_id}: {reference.failure}"
            )
        candidate = verify(run_candidate(case.input), case, family)
        speedup = None
        if (
            candidate.passed
            and reference.elapsed_seconds is not None
            and candidate.elapsed_seconds is not None
            and candidate.elapsed_seconds > 0
        ):
            speedup = reference.elapsed_seconds / candidate.elapsed_seconds
        rows.append(
            {
                "case_id": case.case_id,
                "role": case.role,
                "input_sha256": case.input_sha256,
                "passed": candidate.passed,
                "reference_seconds": reference.elapsed_seconds,
                "candidate_seconds": candidate.elapsed_seconds,
                "speedup": speedup,
                "reference": reference.to_dict(),
                "candidate": candidate.to_dict(),
            }
        )
    return rows


def _report(
    manifest: Mapping[str, object],
    identities: Mapping[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    complete = [row["case_id"] for row in rows] == [
        case["case_id"] for case in manifest["cases"]
    ]
    all_passed = all(row["passed"] for row in rows)
    scored = [
        row["speedup"]
        for row in rows
        if row["role"] == "scored" and row["speedup"] is not None
    ]
    aggregate = _geometric_mean(scored) if complete and all_passed and scored else None
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
        "geometric_mean_speedup": aggregate,
        **dict(identities),
        "cases": rows,
    }


def _local_evaluation(
    candidate: pathlib.Path,
    reference: pathlib.Path,
    manifest: Mapping[str, object],
    cases: Iterator[BenchmarkCase],
    cache: pathlib.Path,
    progress: Progress,
) -> dict[str, object]:
    from .build import (
        REFERENCE_ID,
        build_reference,
        build_runtime,
        build_solver,
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
        workspace = pathlib.Path(temporary)
        progress("Compiling the fixed reference and candidate")
        reference_build = build_reference(reference, workspace / "reference", runtime)
        candidate_build = build_solver(candidate, workspace / "candidate", runtime)
        rows = _case_rows(
            manifest["family"],
            cases,
            lambda value: run_solver(reference_build.executable, value),
            lambda value: run_solver(candidate_build.executable, value),
            progress,
        )
    identities = {
        "venue": {"id": "local-uncontrolled"},
        "runtime_sha256": runtime.manifest_sha256,
        "reference": {
            "id": REFERENCE_ID,
            "source_sha256": reference_build.source_sha256,
            "executable_sha256": reference_build.executable_sha256,
        },
        "candidate": {
            "source_sha256": candidate_build.source_sha256,
            "executable_sha256": candidate_build.executable_sha256,
        },
    }
    return _report(manifest, identities, rows)


def _modal_evaluation(
    candidate: pathlib.Path,
    reference: pathlib.Path,
    manifest: Mapping[str, object],
    cases: Iterator[BenchmarkCase],
    progress: Progress,
) -> dict[str, object]:
    from .modal import open_session

    progress("Starting the Modal CPU sandbox")
    with open_session(candidate, reference, manifest["execution"]) as session:
        progress("Compiling the fixed reference and candidate")
        identities = session.build()
        rows = _case_rows(
            manifest["family"],
            cases,
            session.run_reference,
            session.run_candidate,
            progress,
        )
    return _report(manifest, identities, rows)


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
    cases = iter_cases(manifest, cache_path / "downloads", selected)
    reference = repository_root() / "reference" / "solver.c"
    if not reference.is_file():
        raise RuntimeError("fixed reference solver is missing")

    # Snapshot source so edits during a long run cannot change the candidate.
    with tempfile.TemporaryDirectory(prefix="lsb-source-") as temporary:
        snapshot = pathlib.Path(temporary) / "solver.c"
        shutil.copyfile(source, snapshot)
        report = (
            _local_evaluation(snapshot, reference, manifest, cases, cache_path, emit)
            if venue == "local"
            else _modal_evaluation(snapshot, reference, manifest, cases, emit)
        )
    if output is not None:
        destination = pathlib.Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report
