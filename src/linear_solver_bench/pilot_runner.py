"""One execution per case, with the same reporting for local and remote runs."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Iterable, Mapping

from .dataset import identity_sha256
from .families import resolve_track
from .models import EvaluationSystem
from .runner import RepeatResult, run_repeat

BENCHMARK_ID = "linear-solver-bench-pilot-cpu-v2"


def local_venue(execution: Mapping) -> dict:
    return {
        "id": "local-uncontrolled",
        "cpus": execution["cpus"],
        "memory_bytes": execution["memory_bytes"],
        "limits_enforced": False,
    }


def evaluate_pilot(
    executable: pathlib.Path,
    systems: Iterable[EvaluationSystem],
    prepared: Mapping,
    *,
    source_sha256: str,
    executable_sha256: str,
    runtime_manifest_sha256: str,
    venue: Mapping | None = None,
    execute: Callable[..., RepeatResult] = run_repeat,
) -> dict:
    """Consume each system once and retain only scalar reports between cases.

    A transport supplies ``execute`` to run the very same case contract remotely.
    Input or infrastructure errors propagate and invalidate the whole evaluation;
    candidate failures are recorded and do not stop subsequent cases.
    """
    from .manifests import validate_release
    from .pilot_dataset import validate_pilot_prepared_manifest
    from .pilot_scoring import validate_pilot_report
    from .workloads import system_digest

    validate_pilot_prepared_manifest(prepared)
    release = prepared["source_release"]
    validate_release(release)
    contract = resolve_track(release["family"], release["track"])
    execution = release["execution"]
    rows = []
    iterator = iter(systems)
    for entry in prepared["cases"]:
        try:
            system = next(iterator)
        except StopIteration as exc:
            raise ValueError("prepared case sequence is incomplete") from exc
        case = release["cases"][entry["source_index"]]
        if (
            system.case_id != entry["case_id"]
            or case["case_id"] != system.case_id
            or system.public.matrix.n != case["n"]
            or system.public.matrix.nnz != case["nnz"]
            or system.public.tolerance != case["tolerance"]
            or system_digest(system) != entry["system_sha256"]
        ):
            raise ValueError("loaded case differs from the selected release")
        run = execute(
            executable,
            system,
            index=0,
            deadline_s=execution["case_timeout_seconds"],
            maximum_iterations=execution["max_iterations"],
            accuracy_contract_id=contract.accuracy_contract_id,
        )
        # Only scalar diagnostics leave this iteration; no retained solution arrays.
        row = {
            "index": len(rows),
            "source_index": entry["source_index"],
            "case_id": system.case_id,
            "n": case["n"],
            "nnz": case["nnz"],
            "tolerance": case["tolerance"],
            "role": case["role"],
            "provenance_group": case["provenance_group"],
            "source_run": case["source_run"],
            "weight": case["weight"],
            "system_sha256": entry["system_sha256"],
            "archive_sha256": entry["sha256"],
            "deadline_s": execution["case_timeout_seconds"],
            "max_iterations": execution["max_iterations"],
            "passed": run.passed
            and run.driver is not None
            and run.driver.elapsed_s <= execution["case_timeout_seconds"],
            "elapsed_s": run.driver.elapsed_s if run.driver else None,
            "execution": run.to_dict(),
        }
        rows.append(row)
        del run, system
    try:
        next(iterator)
    except StopIteration:
        pass
    else:
        raise ValueError("prepared case sequence has unexpected extra cases")
    body = {
        "schema_version": 2,
        "kind": "linear-solver-bench-pilot-report",
        "benchmark_id": BENCHMARK_ID,
        "family": contract.family,
        "track": contract.track,
        "release_id": release["release_id"],
        "split": release["split"],
        "contracts": contract.contracts,
        "repetitions": 1,
        "release": release,
        "release_manifest_sha256": release["manifest_sha256"],
        "prepared_manifest_sha256": prepared["manifest_sha256"],
        "prepared_manifest": dict(prepared),
        "source_sha256": source_sha256,
        "executable_sha256": executable_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "venue": dict(venue) if venue is not None else local_venue(execution),
        "complete_split": [row["case_id"] for row in rows]
        == [case["case_id"] for case in release["cases"]],
        "case_count": len(rows),
        "solved_count": sum(row["passed"] for row in rows),
        "cases": rows,
    }
    result = {**body, "report_sha256": identity_sha256(body)}
    validate_pilot_report(result)
    return result
