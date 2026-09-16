import struct

import numpy as np
import pytest

from linear_solver_bench.benchmark import _case_rows, _report
from linear_solver_bench.models import BenchmarkCase, CaseInput, CsrMatrix, RawExecution
from linear_solver_bench.verify import OUTPUT_HEADER, OUTPUT_MAGIC


def _case(case_id="small", role="scored"):
    matrix = CsrMatrix(
        1,
        np.array([0, 1], dtype=np.uint64),
        np.array([0], dtype=np.uint32),
        np.array([2.0]),
    )
    return BenchmarkCase(
        case_id,
        CaseInput(matrix, np.array([4.0]), np.array([0.0]), 1e-10),
        target=np.array([2.0]),
        input_sha256="a" * 64,
        role=role,
    )


def _execution(nanoseconds, solution=2.0):
    payload = OUTPUT_HEADER.pack(
        OUTPUT_MAGIC,
        1,
        0,
        1,
        nanoseconds,
        0,
        0,
        nanoseconds,
        0,
    ) + struct.pack("<d", solution)
    return RawExecution(0, 0.01, "", payload)


def test_case_rows_run_reference_then_candidate_and_compute_speedup():
    calls = []

    def reference(_input):
        calls.append("reference")
        return _execution(200)

    def candidate(_input):
        calls.append("candidate")
        return _execution(100)

    rows = _case_rows(
        "ns-mesh-pde", iter([_case()]), reference, candidate, lambda _message: None
    )

    assert calls == ["reference", "candidate"]
    assert rows[0]["passed"]
    assert rows[0]["speedup"] == pytest.approx(2.0)


def test_reference_failure_invalidates_evaluation():
    with pytest.raises(RuntimeError, match="fixed reference failed"):
        _case_rows(
            "ns-mesh-pde",
            iter([_case()]),
            lambda _input: _execution(200, solution=0.0),
            lambda _input: _execution(100),
            lambda _message: None,
        )


def test_aggregate_requires_the_complete_passing_split():
    manifest = {
        "release_id": "dev-v1",
        "family": "ns-mesh-pde",
        "split": "dev",
        "manifest_sha256": "b" * 64,
        "cases": [{"case_id": "a"}, {"case_id": "b"}],
    }
    rows = [
        {"case_id": "a", "role": "scored", "passed": True, "speedup": 2.0},
        {"case_id": "b", "role": "scored", "passed": True, "speedup": 8.0},
    ]

    complete = _report(manifest, {"venue": {"id": "test"}}, rows)
    partial = _report(manifest, {"venue": {"id": "test"}}, rows[:1])
    modal = _report(manifest, {"venue": {"id": "modal-cpu-v1"}}, rows)

    assert complete["geometric_mean_speedup"] == pytest.approx(4.0)
    assert not complete["official"]
    assert partial["geometric_mean_speedup"] is None
    assert not partial["official"]
    assert modal["official"]
