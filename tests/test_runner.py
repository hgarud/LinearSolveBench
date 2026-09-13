from __future__ import annotations

import pathlib
import sys

import pytest

from linear_solver_bench.runner import build_report, evaluate_case
from linear_solver_bench.scoring import validate_report


def test_fresh_process_runner_and_report(tmp_path: pathlib.Path, small_system) -> None:
    executable = tmp_path / "mock-driver"
    executable.write_text(
        f"""#!{sys.executable}
import pathlib
import struct
import sys

import numpy as np

input_payload = pathlib.Path(sys.argv[1]).read_bytes()
_magic, _version, _reserved, n, nnz, _tolerance = struct.unpack_from(
    "<8sIIQQd", input_payload
)
offset = 40
rows = np.frombuffer(input_payload, dtype="<u8", count=n + 1, offset=offset)
offset += 8 * (n + 1)
columns = np.frombuffer(input_payload, dtype="<u4", count=nnz, offset=offset)
offset += 4 * nnz
values = np.frombuffer(input_payload, dtype="<f8", count=nnz, offset=offset)
offset += 8 * nnz
rhs = np.frombuffer(input_payload, dtype="<f8", count=n, offset=offset)
matrix = np.zeros((n, n), dtype=np.float64)
for row in range(n):
    matrix[row, columns[rows[row]:rows[row + 1]]] = values[rows[row]:rows[row + 1]]
solution = np.linalg.solve(matrix, rhs)
header = struct.pack(
    "<8sIiQQQQQB7x", b"LSBOUT01", 1, 0, n, 600, 100, 200, 300, 0
)
pathlib.Path(sys.argv[2]).write_bytes(header + solution.astype("<f8").tobytes())
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)

    case = evaluate_case(executable, small_system, deadline_s=2.0)
    assert case.passed
    assert len(case.repetitions) == 3
    assert case.median_elapsed_s == pytest.approx(6.0e-7)

    report = build_report(
        (case,),
        source_sha256="source",
        executable_sha256="executable",
        runtime_manifest_sha256="runtime",
        dataset_manifest_sha256="dataset",
        prepared_manifest_sha256="prepared",
        venue_id="local-uncontrolled",
    )
    validate_report(report)
    assert report["solved_count"] == 1
    driver = report["cases"][0]["repetitions"][0]["driver"]
    assert driver["status"] == 0
    assert driver["elapsed_s"] == pytest.approx(6.0e-7)
    assert driver["create_s"] == pytest.approx(1.0e-7)
    assert driver["setup_s"] == pytest.approx(2.0e-7)
    assert driver["solve_s"] == pytest.approx(3.0e-7)
    assert driver["input_mutated"] is False
