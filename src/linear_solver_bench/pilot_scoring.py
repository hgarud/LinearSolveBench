"""Coverage counts and replay speedups bound to a complete frozen case set."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from .dataset import identity_sha256
from .families import resolve_track


def _hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _signed(value: Mapping, field: str) -> None:
    if value.get(field) != identity_sha256(
        {key: item for key, item in value.items() if key != field}
    ):
        raise ValueError(f"{field} mismatch")


def _metric_pass(verification: Mapping, contract, tolerance: float) -> bool:
    metrics = verification.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if contract.track == "coverage":
        gates = {
            "normwise_backward_error": 1e-10,
            "componentwise_backward_error": 1e-8,
            "relative_l2_forward_error": 1e-5,
            "relative_linf_forward_error": 1e-5,
        }
    else:
        gates = {
            "normwise_backward_error": max(tolerance, 20 * math.ulp(1.0)),
            "relative_residual": tolerance * (1 + 1e-6),
        }
    return all(
        type(metrics.get(name)) in (int, float)
        and math.isfinite(metrics[name])
        and 0 <= metrics[name] <= limit
        for name, limit in gates.items()
    )


def validate_pilot_report(report: Mapping) -> None:
    from .manifests import validate_release
    from .pilot_dataset import validate_pilot_prepared_manifest

    _signed(report, "report_sha256")
    release = report.get("release")
    if not isinstance(release, dict):
        raise ValueError("report release is missing")
    validate_release(release)
    prepared = validate_pilot_prepared_manifest(report.get("prepared_manifest"))
    if prepared["source_release"] != release or prepared[
        "manifest_sha256"
    ] != report.get("prepared_manifest_sha256"):
        raise ValueError("report prepared identity differs from its embedded manifest")
    contract = resolve_track(release["family"], release["track"])
    if (
        report.get("schema_version") != 2
        or report.get("kind") != "linear-solver-bench-pilot-report"
        or report.get("benchmark_id") != "linear-solver-bench-pilot-cpu-v2"
        or report.get("contracts") != contract.contracts
        or type(report.get("repetitions")) is not int
        or report["repetitions"] != 1
        or any(
            report.get(k) != release[k]
            for k in ("family", "track", "release_id", "split")
        )
        or report.get("release_manifest_sha256") != release["manifest_sha256"]
    ):
        raise ValueError("report contract differs from release")
    for name in (
        "source_sha256",
        "executable_sha256",
        "runtime_manifest_sha256",
        "prepared_manifest_sha256",
        "release_manifest_sha256",
    ):
        if not _hash(report.get(name)):
            raise ValueError(f"report {name} is invalid")
    execution = release["execution"]
    venue = report.get("venue")
    if (
        not isinstance(venue, dict)
        or set(venue) != {"id", "cpus", "memory_bytes", "limits_enforced"}
        or venue["id"] not in {"local-uncontrolled", "modal-sandbox-pilot-cpu-v2"}
        or venue["cpus"] != execution["cpus"]
        or venue["memory_bytes"] != execution["memory_bytes"]
        or venue["limits_enforced"] is not (venue["id"] != "local-uncontrolled")
    ):
        raise ValueError("report venue or resource contract is invalid")
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or report.get("case_count") != len(cases)
        or len(cases) != len(prepared["cases"])
    ):
        raise ValueError("report case set is empty or incomplete")
    indices = []
    for index, row in enumerate(cases):
        if not isinstance(row, dict) or row.get("index") != index:
            raise ValueError("report case order is invalid")
        source_index = row.get("source_index")
        if type(source_index) is not int or not 0 <= source_index < len(
            release["cases"]
        ):
            raise ValueError("report case is outside release")
        indices.append(source_index)
        source = release["cases"][source_index]
        entry = prepared["cases"][index]
        if (
            any(
                row.get(key) != entry[key]
                for key in ("case_id", "source_index", "system_sha256")
            )
            or row.get("archive_sha256") != entry["sha256"]
        ):
            raise ValueError("report case differs from prepared numerical identity")
        for key in (
            "case_id",
            "n",
            "nnz",
            "tolerance",
            "role",
            "provenance_group",
            "source_run",
            "weight",
        ):
            if row.get(key) != source[key]:
                raise ValueError(f"report case {key} differs from release")
        if (
            row.get("deadline_s") != execution["case_timeout_seconds"]
            or row.get("max_iterations") != execution["max_iterations"]
            or not _hash(row.get("system_sha256"))
            or not _hash(row.get("archive_sha256"))
        ):
            raise ValueError("report numerical or execution identity is invalid")
        run = row.get("execution")
        if (
            not isinstance(run, dict)
            or type(run.get("index")) is not int
            or run["index"] != 0
        ):
            raise ValueError("each pilot case must have one execution at index zero")
        outcome = run.get("outcome")
        if outcome not in {"completed", "timeout", "crash", "invalid_output"}:
            raise ValueError("invalid candidate outcome")
        if not _positive(run.get("process_wall_s")):
            raise ValueError("invalid candidate process time")
        passed = False
        if outcome == "completed":
            driver, verification = run.get("driver"), run.get("verification")
            if not isinstance(driver, dict) or not isinstance(verification, dict):
                raise ValueError("completed case lacks driver or verification evidence")
            elapsed = driver.get("elapsed_s")
            phases = [driver.get(k) for k in ("create_s", "setup_s", "solve_s")]
            if (
                run.get("process_returncode") != 0
                or not _positive(elapsed)
                or row.get("elapsed_s") != elapsed
                or any(
                    type(t) not in (int, float) or not math.isfinite(t) or t < 0
                    for t in phases
                )
                or not math.isclose(sum(phases), elapsed, rel_tol=1e-12, abs_tol=1e-9)
                or type(driver.get("status")) is not int
                or type(driver.get("input_mutated")) is not bool
            ):
                raise ValueError("completed case timing or status is invalid")
            passed = (
                driver["status"] == 0
                and not driver["input_mutated"]
                and elapsed <= row["deadline_s"]
                and verification.get("ok") is True
                and _metric_pass(verification, contract, row["tolerance"])
            )
            if verification.get("ok") is True and not _metric_pass(
                verification, contract, row["tolerance"]
            ):
                raise ValueError("verification verdict contradicts required metrics")
        elif (
            any(run.get(k) is not None for k in ("driver", "verification"))
            or row.get("elapsed_s") is not None
        ):
            raise ValueError(
                "failed process cannot contain successful numerical output"
            )
        if type(row.get("passed")) is not bool or row["passed"] != passed:
            raise ValueError("case pass flag contradicts execution evidence")
    if indices != sorted(set(indices)):
        raise ValueError("report contains reordered or duplicate cases")
    complete = indices == list(range(len(release["cases"])))
    if report.get("complete_split") is not complete:
        raise ValueError("report completeness flag contradicts selected cases")
    if type(report.get("solved_count")) is not int or report["solved_count"] != sum(
        r["passed"] for r in cases
    ):
        raise ValueError("report solved count mismatch")


def replay_reference_from_report(report: Mapping) -> dict:
    """Freeze a single declared reference only after all cases and controls pass."""
    validate_pilot_report(report)
    if report["track"] != "replay":
        raise ValueError("NS coverage does not require a timing reference")
    if not report["complete_split"] or report["solved_count"] != report["case_count"]:
        raise ValueError(
            "replay reference must pass the complete split and its controls"
        )
    body = {
        "schema_version": 2,
        "kind": "linear-solver-bench-replay-reference",
        "reference_report": dict(report),
    }
    return {**body, "calibration_sha256": identity_sha256(body)}


def validate_replay_reference(reference: Mapping) -> None:
    _signed(reference, "calibration_sha256")
    if (
        reference.get("schema_version") != 2
        or reference.get("kind") != "linear-solver-bench-replay-reference"
    ):
        raise ValueError("invalid replay reference schema")
    expected = replay_reference_from_report(reference["reference_report"])
    if dict(reference) != expected:
        raise ValueError("replay reference differs from qualification report")


def score_pilot_report(report: Mapping, reference: Mapping | None = None) -> dict:
    from .manifests import validate_release

    validate_pilot_report(report)
    if not report["complete_split"]:
        raise ValueError("scoring requires the complete frozen split")
    published = True
    try:
        validate_release(report["release"], official=True)
    except ValueError:
        published = False
    official = published and report["venue"]["limits_enforced"]
    base = {
        "schema_version": 2,
        "family": report["family"],
        "track": report["track"],
        "release_id": report["release_id"],
        "split": report["split"],
        "report_sha256": report["report_sha256"],
        "official": official,
        "solved_count": report["solved_count"],
        "case_count": report["case_count"],
        "coverage_fraction": report["solved_count"] / report["case_count"],
    }
    if report["track"] == "coverage":
        if reference is not None:
            raise ValueError("coverage scoring does not use a timing reference")
        return {**base, "eligible": True, "ranking_key": -report["solved_count"]}
    if reference is None:
        return {
            **base,
            "eligible": False,
            "speedup": None,
            "reason": "awaiting_reference",
        }
    validate_replay_reference(reference)
    anchor = reference["reference_report"]
    for name in (
        "family",
        "track",
        "release_manifest_sha256",
        "prepared_manifest_sha256",
        "contracts",
        "repetitions",
        "runtime_manifest_sha256",
        "venue",
        "split",
    ):
        if report[name] != anchor[name]:
            raise ValueError(f"candidate and reference {name} differ")
    per_case, weighted_logs = [], []
    for case, baseline in zip(report["cases"], anchor["cases"], strict=True):
        for key in (
            "case_id",
            "system_sha256",
            "archive_sha256",
            "role",
            "weight",
            "deadline_s",
            "max_iterations",
        ):
            if case[key] != baseline[key]:
                raise ValueError(f"candidate and reference case {key} differ")
        speedup = baseline["elapsed_s"] / case["elapsed_s"] if case["passed"] else None
        if speedup is not None and not _positive(speedup):
            raise ValueError("per-case speedup is not finite and positive")
        per_case.append(
            {"case_id": case["case_id"], "role": case["role"], "speedup": speedup}
        )
        if case["role"] == "scored" and speedup is not None:
            weighted_logs.append(case["weight"] * math.log(speedup))
    eligible = report["solved_count"] == report["case_count"]
    return {
        **base,
        "calibration_sha256": reference["calibration_sha256"],
        "eligible": eligible,
        "speedup": math.exp(math.fsum(weighted_logs)) if eligible else None,
        "reason": None if eligible else "candidate_failed_required_cases",
        "cases": per_case,
    }
