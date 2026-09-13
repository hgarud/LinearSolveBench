"""Calibration artifacts and the lexicographic v1 leaderboard score."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from .accuracy import ACCURACY_CONTRACT_ID


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def validate_report(report: Mapping[str, object]) -> None:
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if digest != _digest(body):
        raise ValueError("evaluation report digest mismatch")
    cases = report.get("cases")
    if (
        report.get("schema_version") != 1
        or report.get("benchmark_id") != "linear-solver-bench-cpu-v1"
        or report.get("accuracy_contract_id") != ACCURACY_CONTRACT_ID
        or report.get("venue_id")
        not in {
            "local-uncontrolled",
            "modal-sandbox-cpu-v1",
        }
        or not isinstance(cases, list)
        or not cases
        or report.get("case_count") != len(cases)
    ):
        raise ValueError("evaluation report case count mismatch")
    case_ids = []
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("case_id"), str):
            raise ValueError("evaluation report contains an invalid case")
        case_ids.append(case["case_id"])
        if case.get("passed") is True:
            elapsed = float(case.get("median_elapsed_s", 0.0))
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError("evaluation report contains an invalid timing")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("evaluation report contains duplicate cases")
    identity_fields = (
        "source_sha256",
        "executable_sha256",
        "runtime_manifest_sha256",
        "dataset_manifest_sha256",
        "prepared_manifest_sha256",
    )
    if not all(
        isinstance(report.get(key), str) and report[key] for key in identity_fields
    ):
        raise ValueError("evaluation report identity is incomplete")
    solved = sum(case.get("passed") is True for case in cases)
    if report.get("solved_count") != solved:
        raise ValueError("evaluation report solved count mismatch")


def calibration_from_report(
    report: Mapping[str, object],
    *,
    deadline_multiplier: float = 4.0,
    minimum_deadline_s: float = 1.0,
    maximum_deadline_s: float = 120.0,
) -> dict[str, object]:
    validate_report(report)
    if not 1.0 < deadline_multiplier <= 10.0:
        raise ValueError("deadline multiplier must lie in (1, 10]")
    if (
        not math.isfinite(minimum_deadline_s)
        or not math.isfinite(maximum_deadline_s)
        or minimum_deadline_s <= 0.0
        or maximum_deadline_s < minimum_deadline_s
        or maximum_deadline_s > 120.0
    ):
        raise ValueError("calibration deadline bounds are invalid")
    cases = []
    for index, case in enumerate(report["cases"]):
        if case.get("passed") is not True:
            raise ValueError("calibration reference must solve every case")
        reference = float(case["median_elapsed_s"])
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("calibration reference time is invalid")
        deadline = min(
            maximum_deadline_s,
            max(minimum_deadline_s, deadline_multiplier * reference),
        )
        cases.append(
            {
                "index": index,
                "case_id": case["case_id"],
                "reference_median_s": reference,
                "deadline_s": deadline,
            }
        )
    body = {
        "schema_version": 1,
        "kind": "linear-solver-bench-calibration-v1",
        "reference_report_sha256": report["report_sha256"],
        "venue_id": report["venue_id"],
        "runtime_manifest_sha256": report["runtime_manifest_sha256"],
        "dataset_manifest_sha256": report["dataset_manifest_sha256"],
        "prepared_manifest_sha256": report["prepared_manifest_sha256"],
        "deadline_multiplier": deadline_multiplier,
        "minimum_deadline_s": minimum_deadline_s,
        "maximum_deadline_s": maximum_deadline_s,
        "case_count": len(cases),
        "cases": cases,
    }
    return {**body, "calibration_sha256": _digest(body)}


def validate_calibration(value: Mapping[str, object]) -> None:
    digest = value.get("calibration_sha256")
    body = {key: item for key, item in value.items() if key != "calibration_sha256"}
    if digest != _digest(body):
        raise ValueError("calibration digest mismatch")
    cases = value.get("cases")
    if (
        value.get("schema_version") != 1
        or value.get("kind") != "linear-solver-bench-calibration-v1"
        or value.get("venue_id") not in {"local-uncontrolled", "modal-sandbox-cpu-v1"}
        or not isinstance(cases, list)
        or not cases
        or value.get("case_count") != len(cases)
    ):
        raise ValueError("calibration case count mismatch")
    minimum = float(value.get("minimum_deadline_s", 0.0))
    maximum = float(value.get("maximum_deadline_s", 0.0))
    multiplier = float(value.get("deadline_multiplier", 0.0))
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or not math.isfinite(multiplier)
        or minimum <= 0.0
        or maximum < minimum
        or maximum > 120.0
        or not 1.0 < multiplier <= 10.0
    ):
        raise ValueError("calibration bounds are invalid")
    case_ids = []
    for index, case in enumerate(cases):
        if (
            not isinstance(case, Mapping)
            or case.get("index") != index
            or not isinstance(case.get("case_id"), str)
        ):
            raise ValueError("calibration contains an invalid case")
        reference = float(case.get("reference_median_s", 0.0))
        deadline = float(case.get("deadline_s", 0.0))
        if (
            not math.isfinite(reference)
            or not math.isfinite(deadline)
            or reference <= 0.0
            or deadline <= 0.0
            or not minimum <= deadline <= maximum
        ):
            raise ValueError("calibration contains an invalid timing")
        case_ids.append(case["case_id"])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("calibration contains duplicate cases")
    for key in (
        "reference_report_sha256",
        "runtime_manifest_sha256",
        "dataset_manifest_sha256",
        "prepared_manifest_sha256",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("calibration identity is incomplete")


def score_report(
    report: Mapping[str, object], calibration: Mapping[str, object]
) -> dict[str, object]:
    validate_report(report)
    validate_calibration(calibration)
    report_cases = report["cases"]
    calibration_cases = calibration["cases"]
    for key in (
        "venue_id",
        "runtime_manifest_sha256",
        "dataset_manifest_sha256",
        "prepared_manifest_sha256",
    ):
        if report.get(key) != calibration.get(key):
            raise ValueError(f"report and calibration {key} differ")
    if len(report_cases) != len(calibration_cases):
        raise ValueError("report and calibration case counts differ")
    ratios = []
    solved = 0
    per_case = []
    for index, (case, reference) in enumerate(
        zip(report_cases, calibration_cases, strict=True)
    ):
        if (
            case.get("case_id") != reference.get("case_id")
            or reference.get("index") != index
        ):
            raise ValueError("report and calibration case order differs")
        reference_s = float(reference["reference_median_s"])
        deadline_s = float(reference["deadline_s"])
        if case.get("passed") is True:
            cost_s = float(case["median_elapsed_s"])
            if not math.isfinite(cost_s) or cost_s <= 0.0:
                raise ValueError("report contains an invalid passing time")
            solved += 1
        else:
            cost_s = 2.0 * deadline_s
        ratio = cost_s / reference_s
        ratios.append(ratio)
        per_case.append(
            {
                "index": index,
                "case_id": case["case_id"],
                "passed": case.get("passed") is True,
                "normalized_cost": ratio,
            }
        )
    penalized_geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return {
        "schema_version": 1,
        "benchmark_id": report["benchmark_id"],
        "report_sha256": report["report_sha256"],
        "calibration_sha256": calibration["calibration_sha256"],
        "solved_count": solved,
        "case_count": len(ratios),
        "penalized_geomean": penalized_geomean,
        "ranking_key": [-solved, penalized_geomean],
        "cases": per_case,
    }
