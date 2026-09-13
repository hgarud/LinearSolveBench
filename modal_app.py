"""Official single-Sandbox CPU venue.

Run with:

    modal run modal_app.py --source submissions/starter.c \
        --cases data/prepared/dev-v1 --output results/modal-dev.json

The trusted client retains evaluator archives and exact solutions. The Sandbox
receives only candidate source, the pinned runtime, and one encoded public
matrix input at a time.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from collections.abc import Iterable, Mapping

import modal

from linear_solver_bench.dataset import load_prepared, load_prepared_manifest
from linear_solver_bench.models import EvaluationSystem
from linear_solver_bench.protocol import decode_output, encode_input
from linear_solver_bench.runner import (
    PROCESS_OVERHEAD_ALLOWANCE_S,
    CaseResult,
    RepeatResult,
    build_report,
)
from linear_solver_bench.scoring import validate_calibration
from linear_solver_bench.verify import verify_solution

ROOT = pathlib.Path(__file__).resolve().parent
APP_NAME = "linear-solver-bench-cpu-v1"
BASE_IMAGE = (
    "debian:trixie-slim@sha256:"
    "abc9cb88a5587630d7f915f47b23b0668fe250fbfc6457aa4d52b534c1bbf73f"
)
REMOTE_BENCHMARK = "/opt/LinearSolverBench"
REMOTE_RUNTIME = "/opt/lsb-runtime"
MAXIMUM_SANDBOX_LIFETIME_S = 24 * 60 * 60

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(BASE_IMAGE, add_python="3.12")
    .apt_install("build-essential", "cmake", "git", "ninja-build")
    .pip_install("numpy==2.5.1", "scipy==1.18.0")
    # Keep evaluator archives out of the image. In particular, never add the
    # repository's data/ tree: prepared archives contain x_star.
    .add_local_dir(ROOT / "src", f"{REMOTE_BENCHMARK}/src", copy=True)
    .add_local_dir(ROOT / "native", f"{REMOTE_BENCHMARK}/native", copy=True)
    .add_local_file(
        ROOT / "pyproject.toml", f"{REMOTE_BENCHMARK}/pyproject.toml", copy=True
    )
    .add_local_file(ROOT / "README.md", f"{REMOTE_BENCHMARK}/README.md", copy=True)
    .add_local_file(ROOT / "LICENSE", f"{REMOTE_BENCHMARK}/LICENSE", copy=True)
    .add_local_file(ROOT / "setup.py", f"{REMOTE_BENCHMARK}/setup.py", copy=True)
    .add_local_file(
        ROOT / "benchmark.toml", f"{REMOTE_BENCHMARK}/benchmark.toml", copy=True
    )
    .env({"LINEAR_SOLVER_BENCH_ROOT": REMOTE_BENCHMARK})
    .run_commands(f"python -m pip install --no-deps {REMOTE_BENCHMARK}")
    .run_commands(
        f"linear-solver-bench runtime build --output {REMOTE_RUNTIME} --jobs 2"
    )
)


def _stream_tail(stream, limit: int = 8000) -> str:
    tail = ""
    for chunk in stream:
        tail = (tail + chunk)[-limit:]
    return tail


def _pilot_case_executor(sandbox):
    """Transfer one public system and verify its single execution on the client."""
    case_number = 0

    def execute(
        executable: pathlib.Path,
        system: EvaluationSystem,
        *,
        index: int = 0,
        deadline_s: float,
        maximum_iterations: int,
        accuracy_contract_id: str,
    ) -> RepeatResult:
        nonlocal case_number
        case_root = f"/work/cases/{case_number:04d}"
        case_number += 1
        input_path = f"{case_root}/input.bin"
        output_path = f"{case_root}/output.bin"
        # write_bytes creates parents. Targets and evaluator archives stay local.
        sandbox.filesystem.write_bytes(encode_input(system.public), input_path)
        started = time.monotonic()
        try:
            try:
                process = sandbox.exec(
                    str(executable),
                    input_path,
                    output_path,
                    str(maximum_iterations),
                    format(deadline_s, ".17g"),
                    env={
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                    timeout=math.ceil(deadline_s + PROCESS_OVERHEAD_ALLOWANCE_S),
                )
                diagnostics = (
                    _stream_tail(process.stdout) + _stream_tail(process.stderr)
                )[-8000:]
                returncode = process.wait()
            except modal.exception.ExecTimeoutError:
                return RepeatResult(
                    index,
                    "timeout",
                    None,
                    time.monotonic() - started,
                    "sandbox execution timeout",
                    None,
                    None,
                )
            wall_s = time.monotonic() - started
            # Modal's process.wait() also represents an exec timeout as -1.
            if returncode in {124, -1}:
                return RepeatResult(
                    index, "timeout", returncode, wall_s, diagnostics, None, None
                )
            if returncode != 0:
                return RepeatResult(
                    index, "crash", returncode, wall_s, diagnostics, None, None
                )
            try:
                payload = sandbox.filesystem.read_bytes(output_path)
            except modal.exception.SandboxFilesystemNotFoundError:
                return RepeatResult(
                    index,
                    "invalid_output",
                    returncode,
                    wall_s,
                    f"{diagnostics}\nmissing driver output".strip(),
                    None,
                    None,
                )
            try:
                driver = decode_output(payload, expected_n=system.public.matrix.n)
            except ValueError as exc:
                return RepeatResult(
                    index,
                    "invalid_output",
                    returncode,
                    wall_s,
                    f"{diagnostics}\n{exc}".strip(),
                    None,
                    None,
                )
            verification = verify_solution(
                system,
                driver.solution,
                status=driver.status,
                input_mutated=driver.input_mutated,
                contract_id=accuracy_contract_id,
            )
            return RepeatResult(
                index,
                "completed",
                returncode,
                wall_s,
                diagnostics,
                driver,
                verification,
            )
        finally:
            # Retain only the current public case in the sandbox filesystem.
            # Filesystem or sandbox failures propagate and invalidate the run.
            sandbox.filesystem.remove(case_root, recursive=True)

    return execute


def _evaluate_pilot_sandbox(
    source: pathlib.Path,
    systems: Iterable[EvaluationSystem],
    prepared: Mapping,
    *,
    official: bool = False,
    reference: Mapping | None = None,
) -> dict:
    from linear_solver_bench.manifests import validate_release
    from linear_solver_bench.pilot_dataset import validate_pilot_prepared_manifest
    from linear_solver_bench.pilot_runner import evaluate_pilot
    from linear_solver_bench.pilot_scoring import validate_replay_reference

    validate_pilot_prepared_manifest(prepared, official=official)
    release = validate_release(prepared["source_release"], official=official)
    execution = release["execution"]
    memory_mib, remainder = divmod(execution["memory_bytes"], 1024**2)
    if remainder or memory_mib < 1:
        raise ValueError("Modal memory_bytes must be an exact positive number of MiB")
    timeout = math.ceil(
        300
        + len(prepared["cases"])
        * (execution["case_timeout_seconds"] + PROCESS_OVERHEAD_ALLOWANCE_S + 5)
    )
    if timeout > MAXIMUM_SANDBOX_LIFETIME_S:
        raise ValueError("selected cases exceed Modal's 24-hour Sandbox lifetime")
    venue = {
        "id": "modal-sandbox-pilot-cpu-v2",
        "cpus": execution["cpus"],
        "memory_bytes": execution["memory_bytes"],
        "limits_enforced": True,
    }
    if reference is not None:
        if release["track"] != "replay":
            raise ValueError("coverage scoring does not use a timing reference")
        validate_replay_reference(reference)
        anchor = reference["reference_report"]
        expected = {
            "family": release["family"],
            "track": release["track"],
            "release_manifest_sha256": release["manifest_sha256"],
            "prepared_manifest_sha256": prepared["manifest_sha256"],
            "contracts": release["contracts"],
            "repetitions": 1,
            "venue": venue,
            "split": release["split"],
        }
        for name, value in expected.items():
            if anchor[name] != value:
                raise ValueError(f"candidate and reference {name} differ")
        if len(anchor["cases"]) != len(prepared["cases"]):
            raise ValueError("candidate and reference case sets differ")
        for entry, baseline in zip(prepared["cases"], anchor["cases"], strict=True):
            if (
                entry["case_id"] != baseline["case_id"]
                or entry["system_sha256"] != baseline["system_sha256"]
                or entry["sha256"] != baseline["archive_sha256"]
            ):
                raise ValueError("candidate and reference numerical inputs differ")
    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        cpu=(float(execution["cpus"]), float(execution["cpus"])),
        memory=(memory_mib, memory_mib),
        timeout=timeout,
        block_network=True,
    )
    try:
        if sandbox.exec("mkdir", "-p", "/work").wait() != 0:
            raise RuntimeError("cannot initialize sandbox working directory")
        sandbox.filesystem.copy_from_local(source, "/work/policy.c")
        compile_process = sandbox.exec(
            "linear-solver-bench",
            "candidate",
            "build",
            "/work/policy.c",
            "--runtime",
            REMOTE_RUNTIME,
            "--output",
            "/work/candidate",
            timeout=120,
        )
        compile_stdout = _stream_tail(compile_process.stdout, 256 * 1024)
        compile_stderr = _stream_tail(compile_process.stderr)
        if compile_process.wait() != 0:
            raise RuntimeError(
                "candidate build failed:\n" + compile_stdout + compile_stderr
            )
        build = json.loads(compile_stdout)
        runtime = json.loads(
            sandbox.filesystem.read_text(f"{REMOTE_RUNTIME}/manifest.json")
        )
        if (
            reference is not None
            and reference["reference_report"]["runtime_manifest_sha256"]
            != runtime["manifest_sha256"]
        ):
            raise ValueError("candidate and reference runtime_manifest_sha256 differ")
        return evaluate_pilot(
            pathlib.Path(build["executable"]),
            systems,
            prepared,
            source_sha256=build["source_sha256"],
            executable_sha256=build["executable_sha256"],
            runtime_manifest_sha256=runtime["manifest_sha256"],
            venue=venue,
            execute=_pilot_case_executor(sandbox),
        )
    finally:
        sandbox.terminate()


def _run_pilot(
    source: pathlib.Path,
    cases: pathlib.Path,
    output: pathlib.Path,
    *,
    family: str,
    track: str,
    official: bool,
    calibration: str,
    score_output: str,
) -> None:
    from linear_solver_bench.families import resolve_track
    from linear_solver_bench.pilot_dataset import (
        iter_pilot_prepared,
        load_pilot_prepared_manifest,
    )
    from linear_solver_bench.pilot_scoring import score_pilot_report

    prepared = load_pilot_prepared_manifest(cases, official=official)
    release = prepared["source_release"]
    if bool(family) != bool(track):
        raise ValueError("provide both family and track selectors")
    if family:
        contract = resolve_track(family, track)
        if (contract.family, contract.track) != (release["family"], release["track"]):
            raise ValueError("family/track differs from the prepared release")
    if score_output and [case["case_id"] for case in prepared["cases"]] != [
        case["case_id"] for case in release["cases"]
    ]:
        raise ValueError("scoring requires the complete frozen split")
    if score_output and pathlib.Path(score_output).expanduser().resolve() == output:
        raise ValueError("report and score output paths must differ")
    reference = None
    if calibration:
        reference = json.loads(
            pathlib.Path(calibration).expanduser().read_text(encoding="utf-8")
        )
    report = _evaluate_pilot_sandbox(
        source,
        iter_pilot_prepared(cases),
        prepared,
        official=official,
        reference=reference,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    if score_output:
        score_path = pathlib.Path(score_output).expanduser().resolve()
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(
            json.dumps(
                score_pilot_report(report, reference),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(score_path)


def _calibration(calibration_path: pathlib.Path | None) -> dict[str, object] | None:
    if calibration_path is None:
        return None
    value = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_calibration(value)
    return value


@app.local_entrypoint()
def main(
    source: str,
    cases: str,
    output: str,
    calibration: str = "",
    deadline: float = 120.0,
    family: str = "",
    track: str = "",
    official: bool = False,
    score_output: str = "",
) -> None:
    cases_path = pathlib.Path(cases).expanduser().resolve()
    schema = json.loads((cases_path / "manifest.json").read_text(encoding="utf-8")).get(
        "schema_version"
    )
    if schema == 2:
        _run_pilot(
            pathlib.Path(source).expanduser().resolve(),
            cases_path,
            pathlib.Path(output).expanduser().resolve(),
            family=family,
            track=track,
            official=official,
            calibration=calibration,
            score_output=score_output,
        )
        return
    if family or track or official or score_output:
        raise ValueError(
            "family, track, official, and score-output options require pilot cases"
        )
    if not math.isfinite(deadline) or not 0.0 < deadline <= 120.0:
        raise ValueError("case deadline must lie in (0, 120]")
    source_path = pathlib.Path(source).expanduser().resolve()
    cases_path = pathlib.Path(cases)
    prepared_manifest = load_prepared_manifest(cases_path)
    systems = load_prepared(cases_path)
    calibration_value = _calibration(
        pathlib.Path(calibration).expanduser().resolve() if calibration else None
    )
    deadline_by_case = None
    if calibration_value is not None:
        observed_cases = [str(case["case_id"]) for case in calibration_value["cases"]]
        expected_cases = [system.case_id for system in systems]
        if observed_cases != expected_cases:
            raise ValueError("calibration case order differs from prepared cases")
        expected_identity = {
            "venue_id": "modal-sandbox-cpu-v1",
            "dataset_manifest_sha256": prepared_manifest["source_manifest_sha256"],
            "prepared_manifest_sha256": prepared_manifest["manifest_sha256"],
        }
        mismatches = sorted(
            key
            for key, expected in expected_identity.items()
            if calibration_value.get(key) != expected
        )
        if mismatches:
            raise ValueError("calibration identity mismatch: " + ", ".join(mismatches))
        deadline_by_case = {
            str(case["case_id"]): float(case["deadline_s"])
            for case in calibration_value["cases"]
        }
    case_deadlines = [
        deadline_by_case[system.case_id] if deadline_by_case is not None else deadline
        for system in systems
    ]
    total_timeout = int(
        300
        + 3 * sum(value + PROCESS_OVERHEAD_ALLOWANCE_S + 5 for value in case_deadlines)
    )
    if total_timeout > MAXIMUM_SANDBOX_LIFETIME_S:
        raise ValueError(
            "case deadlines exceed Modal's 24-hour Sandbox lifetime; "
            "provide a valid venue calibration artifact"
        )
    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        cpu=(2.0, 2.0),
        memory=(4096, 4096),
        timeout=total_timeout,
        block_network=True,
    )
    try:
        sandbox.exec("mkdir", "-p", "/work").wait()
        sandbox.filesystem.copy_from_local(source_path, "/work/policy.c")
        compile_process = sandbox.exec(
            "linear-solver-bench",
            "candidate",
            "build",
            "/work/policy.c",
            "--runtime",
            REMOTE_RUNTIME,
            "--output",
            "/work/candidate",
            timeout=120,
        )
        compile_stdout = compile_process.stdout.read()
        compile_stderr = compile_process.stderr.read()
        if compile_process.wait() != 0:
            raise RuntimeError(
                "candidate build failed:\n" + compile_stdout + compile_stderr
            )
        build_metadata = json.loads(compile_stdout)
        remote_executable = str(build_metadata["executable"])
        runtime_manifest = json.loads(
            sandbox.filesystem.read_text(f"{REMOTE_RUNTIME}/manifest.json")
        )
        if (
            calibration_value is not None
            and calibration_value["runtime_manifest_sha256"]
            != runtime_manifest["manifest_sha256"]
        ):
            raise ValueError("calibration runtime identity mismatch")
        results = []
        environment = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        for case_index, system in enumerate(systems):
            case_root = f"/work/cases/{case_index:04d}"
            sandbox.exec("mkdir", "-p", case_root).wait()
            input_path = f"{case_root}/input.bin"
            sandbox.filesystem.write_bytes(encode_input(system.public), input_path)
            case_deadline = (
                deadline_by_case[system.case_id]
                if deadline_by_case is not None
                else deadline
            )
            repeats = []
            for repeat_index in range(3):
                output_path = f"{case_root}/output-{repeat_index}.bin"
                started = time.monotonic()
                try:
                    process = sandbox.exec(
                        remote_executable,
                        input_path,
                        output_path,
                        "10000",
                        format(case_deadline, ".17g"),
                        env=environment,
                        timeout=case_deadline + PROCESS_OVERHEAD_ALLOWANCE_S,
                    )
                    stdout = process.stdout.read()
                    stderr = process.stderr.read()
                    returncode = process.wait()
                    wall_s = time.monotonic() - started
                except (
                    modal.exception.ExecTimeoutError,
                    modal.exception.SandboxTimeoutError,
                ):
                    repeats.append(
                        RepeatResult(
                            repeat_index,
                            "timeout",
                            None,
                            time.monotonic() - started,
                            "sandbox execution timeout",
                            None,
                            None,
                        )
                    )
                    break
                diagnostics = (stdout + stderr)[-8000:]
                if returncode == 124:
                    repeats.append(
                        RepeatResult(
                            repeat_index,
                            "timeout",
                            returncode,
                            wall_s,
                            diagnostics,
                            None,
                            None,
                        )
                    )
                    break
                if returncode != 0:
                    repeats.append(
                        RepeatResult(
                            repeat_index,
                            "crash",
                            returncode,
                            wall_s,
                            diagnostics,
                            None,
                            None,
                        )
                    )
                    break
                try:
                    driver = decode_output(
                        sandbox.filesystem.read_bytes(output_path),
                        expected_n=system.public.matrix.n,
                    )
                except ValueError as exc:
                    repeats.append(
                        RepeatResult(
                            repeat_index,
                            "invalid_output",
                            returncode,
                            wall_s,
                            f"{diagnostics}\n{exc}".strip(),
                            None,
                            None,
                        )
                    )
                    break
                verification = verify_solution(
                    system,
                    driver.solution,
                    status=driver.status,
                    input_mutated=driver.input_mutated,
                )
                repeats.append(
                    RepeatResult(
                        repeat_index,
                        "completed",
                        returncode,
                        wall_s,
                        diagnostics,
                        driver,
                        verification,
                    )
                )
            results.append(
                CaseResult(
                    case_id=system.case_id,
                    n=system.public.matrix.n,
                    nnz=system.public.matrix.nnz,
                    tolerance=system.public.tolerance,
                    deadline_s=case_deadline,
                    repetitions=tuple(repeats),
                )
            )
        report = build_report(
            tuple(results),
            source_sha256=str(build_metadata["source_sha256"]),
            executable_sha256=str(build_metadata["executable_sha256"]),
            runtime_manifest_sha256=str(runtime_manifest["manifest_sha256"]),
            dataset_manifest_sha256=str(prepared_manifest["source_manifest_sha256"]),
            prepared_manifest_sha256=str(prepared_manifest["manifest_sha256"]),
            venue_id="modal-sandbox-cpu-v1",
        )
        output_path = pathlib.Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(output_path)
    finally:
        sandbox.terminate()
