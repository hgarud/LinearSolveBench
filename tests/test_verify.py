import struct

import numpy as np
import pytest

from linear_solver_bench.models import BenchmarkCase, CaseInput, CsrMatrix, RawExecution
from linear_solver_bench.verify import OUTPUT_HEADER, OUTPUT_MAGIC, verify


def case(target=(1.0, 2.0)):
    matrix = CsrMatrix(
        2,
        np.array([0, 1, 2], dtype=np.uint64),
        np.array([0, 1], dtype=np.uint32),
        np.array([2.0, 3.0]),
    )
    target = np.asarray(target, dtype=np.float64)
    return BenchmarkCase(
        "small",
        CaseInput(matrix, matrix.matvec(target), np.zeros(2), 1.0e-10),
        target=target,
        input_sha256="a" * 64,
    )


def output(solution, *, status=0, mutated=0):
    nanoseconds = (10, 20, 30)
    return OUTPUT_HEADER.pack(
        OUTPUT_MAGIC,
        1,
        status,
        len(solution),
        sum(nanoseconds),
        *nanoseconds,
        mutated,
    ) + struct.pack(f"<{len(solution)}d", *solution)


def test_correct_solution_passes():
    benchmark_case = case()
    raw = RawExecution(0, 0.01, "", output([1.0, 2.0]))

    result = verify(raw, benchmark_case, "ns-mesh-pde")

    assert result.passed
    assert result.elapsed_seconds == pytest.approx(60e-9)
    assert result.metrics.normwise_backward_error == 0


def test_inaccurate_solution_keeps_native_time():
    raw = RawExecution(0, 0.01, "", output([1.0, 1.0]))

    result = verify(raw, case(), "ns-mesh-pde")

    assert not result.passed
    assert result.elapsed_seconds == pytest.approx(60e-9)
    assert "forward" in result.failure or "backward" in result.failure


def test_process_and_protocol_failures_are_results():
    failed = verify(RawExecution(9, 0.01, "boom", None), case(), "ns-mesh-pde")
    malformed = verify(RawExecution(0, 0.01, "", b"short"), case(), "ns-mesh-pde")

    assert not failed.passed and failed.elapsed_seconds is None
    assert not malformed.passed and malformed.elapsed_seconds is None


def test_flash_uses_residual_without_a_target():
    benchmark_case = case()
    benchmark_case = BenchmarkCase(
        benchmark_case.case_id,
        benchmark_case.input,
        target=None,
        input_sha256=benchmark_case.input_sha256,
    )

    result = verify(
        RawExecution(0, 0.01, "", output([1.0, 2.0])),
        benchmark_case,
        "magnetic_diffusion_flash",
    )

    assert result.passed
