from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from linear_solver_bench.build import (
    INPUT_HEADER,
    OUTPUT_HEADER,
    OUTPUT_MAGIC,
    PROTOCOL_VERSION,
    REFERENCE_SOURCE,
    build_reference,
    build_solver,
    encode_input,
    load_runtime,
    output_size,
    run_solver,
)
from linear_solver_bench.models import BenchmarkCase, CaseInput, CsrMatrix
from linear_solver_bench.verify import verify


@pytest.fixture
def small_case() -> BenchmarkCase:
    matrix = CsrMatrix(
        2,
        np.asarray([0, 2, 4], dtype=np.uint64),
        np.asarray([0, 1, 0, 1], dtype=np.uint32),
        np.asarray([4.0, 1.0, 1.0, 3.0], dtype=np.float64),
    )
    target = np.asarray([1.0, 2.0], dtype=np.float64)
    value = CaseInput(
        matrix,
        matrix.matvec(target),
        np.zeros(2, dtype=np.float64),
        1.0e-8,
    )
    return BenchmarkCase("small", value, target)


def _executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(0o755)
    return path


def _output(solution: np.ndarray) -> bytes:
    return (
        OUTPUT_HEADER.pack(
            OUTPUT_MAGIC,
            PROTOCOL_VERSION,
            0,
            len(solution),
            600,
            100,
            200,
            300,
            0,
        )
        + np.asarray(solution, dtype="<f8").tobytes()
    )


def test_binary_input_layout(small_case: BenchmarkCase) -> None:
    payload = encode_input(small_case.input)
    matrix = small_case.input.matrix
    expected = INPUT_HEADER.size + 8 * 3 + 4 * 4 + 8 * 4 + 8 * 2 + 8 * 2
    assert len(payload) == expected
    magic, version, reserved, n, nnz, tolerance = INPUT_HEADER.unpack_from(payload)
    assert (magic, version, reserved, n, nnz) == (b"LSBIN001", 1, 0, 2, 4)
    assert tolerance == small_case.input.tolerance
    assert output_size(matrix.n) == OUTPUT_HEADER.size + 16


def test_each_run_is_a_fresh_process(tmp_path: Path, small_case: BenchmarkCase) -> None:
    payload = _output(small_case.target)
    executable = _executable(
        tmp_path / "solver",
        "import pathlib, sys\n"
        "assert len(sys.argv) == 3\n"
        f"pathlib.Path(sys.argv[2]).write_bytes({payload!r})\n",
    )
    first = run_solver(executable, small_case.input)
    second = run_solver(executable, small_case.input)
    assert first.returncode == second.returncode == 0
    assert first.output == second.output == payload
    assert first.wall_seconds > 0 and second.wall_seconds > 0


def test_runner_bounds_diagnostics_and_output(
    tmp_path: Path, small_case: BenchmarkCase
) -> None:
    executable = _executable(
        tmp_path / "noisy",
        "import pathlib, sys\n"
        "sys.stdout.buffer.write(b'x' * 100000)\n"
        "pathlib.Path(sys.argv[2]).write_bytes(b'y' * 100000)\n",
    )
    result = run_solver(executable, small_case.input)
    assert result.returncode == 0
    assert len(result.diagnostics) == 8000
    assert result.output is not None
    assert len(result.output) == output_size(2) + 1
    assert not verify(result, small_case, "ns-mesh-pde").passed


def test_failed_process_has_no_timing_payload(
    tmp_path: Path, small_case: BenchmarkCase
) -> None:
    result = run_solver(
        _executable(tmp_path / "failed", "raise SystemExit(7)\n"),
        small_case.input,
    )
    assert result.returncode == 7
    assert result.output is None


def test_process_timeout_is_bounded(tmp_path: Path, small_case: BenchmarkCase) -> None:
    executable = _executable(
        tmp_path / "forever",
        "import time\nwhile True: time.sleep(1)\n",
    )
    result = run_solver(executable, small_case.input, timeout_seconds=0.05)
    assert result.returncode == 124
    assert result.wall_seconds < 2
    assert "timed out" in result.diagnostics
    assert result.output is None


def test_real_driver_and_both_solvers(
    tmp_path: Path, small_case: BenchmarkCase
) -> None:
    runtime_path = os.environ.get("LSB_TEST_RUNTIME")
    if not runtime_path:
        pytest.skip("set LSB_TEST_RUNTIME to run native integration tests")
    runtime = load_runtime(Path(runtime_path))
    sources = (Path("examples/solver.c"), REFERENCE_SOURCE)
    builders = (build_solver, build_reference)
    for index, (source, builder) in enumerate(zip(sources, builders, strict=True)):
        compiled = builder(source, tmp_path / f"solver-{index}", runtime)
        raw = run_solver(compiled.executable, small_case.input)
        result = verify(raw, small_case, "ns-mesh-pde")
        assert result.passed, result
        assert result.elapsed_seconds and result.elapsed_seconds > 0
