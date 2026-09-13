"""Fresh-process local execution and report construction."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import statistics
import tempfile
from dataclasses import dataclass

from .accuracy import ACCURACY_CONTRACT_ID
from .models import EvaluationSystem
from .process import ProcessTimeout, bounded_process
from .protocol import DriverOutput, decode_output, encode_input
from .verify import VerificationResult, verify_solution

REPETITIONS = 3
MAXIMUM_ITERATIONS = 10_000
PROCESS_OVERHEAD_ALLOWANCE_S = 30.0


@dataclass(frozen=True)
class RepeatResult:
    index: int
    outcome: str
    process_returncode: int | None
    process_wall_s: float | None
    diagnostics: str
    driver: DriverOutput | None
    verification: VerificationResult | None

    @property
    def passed(self) -> bool:
        return (
            self.outcome == "completed"
            and self.verification is not None
            and self.verification.ok
        )

    def to_dict(self) -> dict[str, object]:
        driver = None
        if self.driver is not None:
            driver = {
                "status": self.driver.status,
                "elapsed_s": self.driver.elapsed_s,
                "create_s": self.driver.create_s,
                "setup_s": self.driver.setup_s,
                "solve_s": self.driver.solve_s,
                "input_mutated": self.driver.input_mutated,
            }
        return {
            "index": self.index,
            "outcome": self.outcome,
            "process_returncode": self.process_returncode,
            "process_wall_s": self.process_wall_s,
            "diagnostics": self.diagnostics,
            "driver": driver,
            "verification": self.verification.to_dict() if self.verification else None,
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    n: int
    nnz: int
    tolerance: float
    deadline_s: float
    repetitions: tuple[RepeatResult, ...]

    @property
    def passed(self) -> bool:
        return len(self.repetitions) == REPETITIONS and all(
            repeat.passed for repeat in self.repetitions
        )

    @property
    def median_elapsed_s(self) -> float | None:
        if not self.passed:
            return None
        return statistics.median(
            repeat.driver.elapsed_s
            for repeat in self.repetitions
            if repeat.driver is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "n": self.n,
            "nnz": self.nnz,
            "tolerance": self.tolerance,
            "deadline_s": self.deadline_s,
            "passed": self.passed,
            "median_elapsed_s": self.median_elapsed_s,
            "repetitions": [repeat.to_dict() for repeat in self.repetitions],
        }


def _diagnostics(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace")[-8000:]


def run_repeat(
    executable: pathlib.Path,
    system: EvaluationSystem,
    *,
    index: int,
    deadline_s: float,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
) -> RepeatResult:
    with tempfile.TemporaryDirectory(prefix="lsb-case-") as temporary:
        root = pathlib.Path(temporary)
        input_path = root / "input.bin"
        output_path = root / "output.bin"
        input_path.write_bytes(encode_input(system.public))
        env = os.environ.copy()
        env.update(
            {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        command = (
            str(executable),
            str(input_path),
            str(output_path),
            str(maximum_iterations),
            format(deadline_s, ".17g"),
        )
        try:
            process = bounded_process(
                command,
                timeout_s=deadline_s + PROCESS_OVERHEAD_ALLOWANCE_S,
                env=env,
            )
        except ProcessTimeout as exc:
            return RepeatResult(
                index,
                "timeout",
                None,
                deadline_s + PROCESS_OVERHEAD_ALLOWANCE_S,
                _diagnostics(exc.output),
                None,
                None,
            )
        if process.returncode == 124:
            return RepeatResult(
                index,
                "timeout",
                124,
                process.wall_s,
                _diagnostics(process.output),
                None,
                None,
            )
        if process.returncode != 0 or not output_path.is_file():
            return RepeatResult(
                index,
                "crash",
                process.returncode,
                process.wall_s,
                _diagnostics(process.output),
                None,
                None,
            )
        try:
            driver = decode_output(
                output_path.read_bytes(), expected_n=system.public.matrix.n
            )
        except ValueError as exc:
            return RepeatResult(
                index,
                "invalid_output",
                process.returncode,
                process.wall_s,
                f"{_diagnostics(process.output)}\n{exc}".strip(),
                None,
                None,
            )
        verification = verify_solution(
            system,
            driver.solution,
            status=driver.status,
            input_mutated=driver.input_mutated,
        )
        return RepeatResult(
            index,
            "completed",
            process.returncode,
            process.wall_s,
            _diagnostics(process.output),
            driver,
            verification,
        )


def evaluate_case(
    executable: pathlib.Path,
    system: EvaluationSystem,
    *,
    deadline_s: float,
    repetitions: int = REPETITIONS,
) -> CaseResult:
    if repetitions != REPETITIONS:
        raise ValueError("v1 requires exactly three repetitions")
    results = []
    for index in range(repetitions):
        repeat = run_repeat(executable, system, index=index, deadline_s=deadline_s)
        results.append(repeat)
        if repeat.outcome in {"timeout", "crash", "invalid_output"}:
            break
    return CaseResult(
        case_id=system.case_id,
        n=system.public.matrix.n,
        nnz=system.public.matrix.nnz,
        tolerance=system.public.tolerance,
        deadline_s=deadline_s,
        repetitions=tuple(results),
    )


def build_report(
    cases: tuple[CaseResult, ...],
    *,
    source_sha256: str,
    executable_sha256: str,
    runtime_manifest_sha256: str,
    dataset_manifest_sha256: str,
    prepared_manifest_sha256: str,
    venue_id: str,
) -> dict[str, object]:
    case_values = [case.to_dict() for case in cases]
    body = {
        "schema_version": 1,
        "benchmark_id": "linear-solver-bench-cpu-v1",
        "accuracy_contract_id": ACCURACY_CONTRACT_ID,
        "source_sha256": source_sha256,
        "executable_sha256": executable_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "venue_id": venue_id,
        "case_count": len(cases),
        "solved_count": sum(case.passed for case in cases),
        "cases": case_values,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **body,
        "report_sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest(),
    }


def evaluate(
    executable: pathlib.Path,
    systems: tuple[EvaluationSystem, ...],
    *,
    deadline_s: float = 120.0,
    deadline_by_case: dict[str, float] | None = None,
    source_sha256: str,
    executable_sha256: str,
    runtime_manifest_sha256: str,
    dataset_manifest_sha256: str,
    prepared_manifest_sha256: str,
) -> dict[str, object]:
    cases = tuple(
        evaluate_case(
            executable,
            system,
            deadline_s=(
                deadline_by_case[system.case_id]
                if deadline_by_case is not None
                else deadline_s
            ),
        )
        for system in systems
    )
    return build_report(
        cases,
        source_sha256=source_sha256,
        executable_sha256=executable_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        prepared_manifest_sha256=prepared_manifest_sha256,
        venue_id="local-uncontrolled",
    )
