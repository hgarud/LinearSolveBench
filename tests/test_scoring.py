from __future__ import annotations

import pytest

from linear_solver_bench.runner import CaseResult, RepeatResult, build_report
from linear_solver_bench.scoring import calibration_from_report, score_report
from linear_solver_bench.verify import VerificationResult


def passing_case(small_system, driver_output) -> CaseResult:
    verification = VerificationResult(
        ok=True,
        failure=None,
        metrics=None,
        thresholds=driver_output[1],
    )
    repetitions = tuple(
        RepeatResult(
            index=index,
            outcome="completed",
            process_returncode=0,
            process_wall_s=0.02,
            diagnostics="",
            driver=driver_output[0],
            verification=verification,
        )
        for index in range(3)
    )
    return CaseResult(
        small_system.case_id,
        2,
        4,
        1.0e-4,
        1.0,
        repetitions,
    )


def test_calibration_and_lexicographic_score(small_system, driver_output) -> None:
    case = passing_case(small_system, driver_output)
    report = build_report(
        (case,),
        source_sha256="source",
        executable_sha256="executable",
        runtime_manifest_sha256="runtime",
        dataset_manifest_sha256="dataset",
        prepared_manifest_sha256="prepared",
        venue_id="modal-sandbox-cpu-v1",
    )
    calibration = calibration_from_report(report, minimum_deadline_s=0.001)
    score = score_report(report, calibration)
    assert score["solved_count"] == 1
    assert score["penalized_geomean"] == 1.0
    assert score["ranking_key"] == [-1, 1.0]


def test_unsolved_case_receives_deadline_penalty(small_system, driver_output) -> None:
    reference_case = passing_case(small_system, driver_output)
    reference = build_report(
        (reference_case,),
        source_sha256="source",
        executable_sha256="executable",
        runtime_manifest_sha256="runtime",
        dataset_manifest_sha256="dataset",
        prepared_manifest_sha256="prepared",
        venue_id="modal-sandbox-cpu-v1",
    )
    calibration = calibration_from_report(reference, minimum_deadline_s=0.001)
    failed_case = CaseResult("test-small", 2, 4, 1.0e-4, 1.0, ())
    report = build_report(
        (failed_case,),
        source_sha256="source",
        executable_sha256="executable",
        runtime_manifest_sha256="runtime",
        dataset_manifest_sha256="dataset",
        prepared_manifest_sha256="prepared",
        venue_id="modal-sandbox-cpu-v1",
    )
    score = score_report(report, calibration)
    assert score["solved_count"] == 0
    assert score["penalized_geomean"] == pytest.approx(8.0)
