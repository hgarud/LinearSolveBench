from __future__ import annotations

import copy
import hashlib
import pathlib
import weakref

import numpy as np
import pytest

from linear_solver_bench.cli import main
from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.families import resolve_track
from linear_solver_bench.manifests import seal_manifest
from linear_solver_bench.models import EvaluationSystem
from linear_solver_bench.pilot_runner import evaluate_pilot
from linear_solver_bench.pilot_scoring import (
    replay_reference_from_report,
    score_pilot_report,
    validate_pilot_report,
)
from linear_solver_bench.protocol import DriverOutput
from linear_solver_bench.runner import RepeatResult
from linear_solver_bench.verify import verify_solution
from linear_solver_bench.workloads import (
    matrix_digest,
    qualify_ns,
    system_digest,
    vector_digest,
)


def prepared_fixture(small_system, *, replay=False):
    family, track = (
        ("magnetic_diffusion_flash", "replay")
        if replay
        else ("ns-mesh-pde", "coverage")
    )
    contract = resolve_track(family, track)
    key = b"public-test-key-0000"
    systems, cases, prepared_cases = [], [], []
    for index in range(4 if replay else 2):
        system = EvaluationSystem(
            f"case-{index}",
            small_system.public,
            small_system.x_star if not replay else None,
            reference_kind="none" if replay else "manufactured",
        )
        systems.append(system)
        source = {
            "kind": "huggingface" if replay else "suitesparse",
            "url": (
                "https://huggingface.co/datasets/example/replay/resolve/"
                f"{'a' * 40}/case.zip"
                if replay
                else "https://sparse.tamu.edu/MM/Example/mesh.tar.gz"
            ),
            "archive_sha256": "a" * 64,
            "archive_bytes": 512,
            "matrix_sha256": matrix_digest(system.public.matrix),
        }
        if not replay:
            source.update(matrix_id=1, group="Example", name="mesh")
        else:
            source.update(
                b_sha256=vector_digest(system.public.b),
                x0_sha256=vector_digest(system.public.x0),
            )
        admission = {
            "statement": "Synthetic test fixture; not a published scientific case.",
            "citations": ["https://example.org/fixture"],
        }
        if replay:
            admission.update(
                capture_kind="scalar-magnetic-diffusion",
                relative_tolerance=system.public.tolerance,
                absolute_tolerance=0,
                step=index,
                solve_index=0,
            )
        qualification = None if replay else qualify_ns(system)
        cases.append(
            {
                "index": index,
                "case_id": system.case_id,
                "n": system.public.matrix.n,
                "nnz": system.public.matrix.nnz,
                "tolerance": system.public.tolerance,
                "role": "control" if replay and index == 3 else "scored",
                "provenance_group": "group-b" if index == 2 else "group-a",
                "source_run": "run-b" if index == 2 else "run-a",
                "weight": [0.25, 0.25, 0.5, 0][index] if replay else 1,
                "source": source,
                "admission": admission,
                "qualification": qualification,
            }
        )
        prepared_cases.append(
            {
                "index": index,
                "source_index": index,
                "case_id": system.case_id,
                "relpath": f"{index:04d}.npz",
                "sha256": "c" * 64,
                "n": system.public.matrix.n,
                "nnz": system.public.matrix.nnz,
                "tolerance": system.public.tolerance,
                "system_sha256": system_digest(system),
                "qualification": qualification,
            }
        )
    release = seal_manifest(
        {
            "schema_version": 2,
            "kind": "linear-solver-bench-pilot-release",
            "release_id": "test-pilot",
            "family": family,
            "track": track,
            "split": "dev",
            "contracts": contract.contracts,
            "execution": {
                "repetitions": 1,
                "max_iterations": 17,
                "case_timeout_seconds": 30.0,
                "cpus": 2,
                "memory_bytes": 4 * 1024**3,
            },
            "rhs": None
            if replay
            else {
                "scheme": "rademacher-hmac-sha256-v2",
                "draw_index": 0,
                "key_id": hashlib.sha256(key).hexdigest(),
                "public_key_hex": key.hex(),
            },
            "cases": cases,
        }
    )
    prepared = seal_manifest(
        {
            "schema_version": 2,
            "kind": "linear-solver-bench-pilot-prepared",
            "source_release": release,
            "source_manifest_sha256": release["manifest_sha256"],
            **{
                k: release[k]
                for k in (
                    "release_id",
                    "family",
                    "track",
                    "split",
                    "contracts",
                    "execution",
                )
            },
            "cases": prepared_cases,
            "case_count": len(cases),
            "complete_split": True,
        }
    )
    return prepared, systems


def evaluate_fixture(prepared, systems, *, times=None, failed=None):
    times = times or [1.0] * len(systems)
    calls = []

    def execute(executable, system, **settings):
        index = len(calls)
        calls.append((system.case_id, settings))
        if failed == index:
            return RepeatResult(0, "timeout", 124, 30.1, "", None, None)
        solution = np.linalg.solve(
            system.public.matrix.to_scipy().toarray(), system.public.b
        )
        elapsed = times[index]
        driver = DriverOutput(
            0, solution, elapsed, elapsed / 4, elapsed / 4, elapsed / 2, False
        )
        verification = verify_solution(
            system, solution, contract_id=settings["accuracy_contract_id"]
        )
        return RepeatResult(0, "completed", 0, elapsed + 0.1, "", driver, verification)

    report = evaluate_pilot(
        pathlib.Path("unused"),
        iter(systems),
        prepared,
        source_sha256="1" * 64,
        executable_sha256="2" * 64,
        runtime_manifest_sha256="3" * 64,
        execute=execute,
    )
    return report, calls


def resign(report):
    report["report_sha256"] = identity_sha256(
        {k: v for k, v in report.items() if k != "report_sha256"}
    )


def test_coverage_counts_failures_and_propagates_once_contract(small_system):
    prepared, systems = prepared_fixture(small_system)
    report, calls = evaluate_fixture(prepared, systems, failed=0)
    assert len(calls) == 2
    assert all(s["index"] == 0 and s["maximum_iterations"] == 17 for _, s in calls)
    score = score_pilot_report(report)
    assert score["solved_count"] == 1
    assert score["coverage_fraction"] == 0.5
    assert score["ranking_key"] == -1
    assert not score["official"]
    # Timing has no effect on coverage ties.
    slower, _ = evaluate_fixture(prepared, systems, times=[1, 10], failed=0)
    assert score_pilot_report(slower)["ranking_key"] == score["ranking_key"]
    with pytest.raises(ValueError, match="does not require"):
        replay_reference_from_report(report)


def test_replay_weights_controls_and_all_pass_reference(small_system):
    prepared, systems = prepared_fixture(small_system, replay=True)
    reference_report, _ = evaluate_fixture(prepared, systems, times=[8] * 4)
    reference = replay_reference_from_report(reference_report)
    candidate, _ = evaluate_fixture(prepared, systems, times=[4, 2, 8, 1])
    assert score_pilot_report(candidate, reference)["speedup"] == pytest.approx(8**0.25)
    failed, _ = evaluate_fixture(prepared, systems, failed=3)
    assert score_pilot_report(failed, reference)["speedup"] is None
    with pytest.raises(ValueError, match="complete split"):
        replay_reference_from_report(failed)
    assert score_pilot_report(candidate)["reason"] == "awaiting_reference"


@pytest.mark.parametrize(
    "field", ["runtime_manifest_sha256", "prepared_manifest_sha256"]
)
def test_replay_rejects_mismatched_execution_identity(small_system, field):
    prepared, systems = prepared_fixture(small_system, replay=True)
    report, _ = evaluate_fixture(prepared, systems)
    reference = replay_reference_from_report(report)
    candidate = copy.deepcopy(report)
    candidate[field] = "9" * 64
    resign(candidate)
    with pytest.raises(ValueError):
        score_pilot_report(candidate, reference)


def test_replay_rejects_rehashed_substitute_prepared_inputs(small_system):
    prepared, systems = prepared_fixture(small_system, replay=True)
    report, _ = evaluate_fixture(prepared, systems)
    changed = copy.deepcopy(report)
    changed["prepared_manifest"]["cases"][0]["system_sha256"] = "f" * 64
    changed["prepared_manifest"] = seal_manifest(
        {
            key: value
            for key, value in changed["prepared_manifest"].items()
            if key != "manifest_sha256"
        }
    )
    changed["prepared_manifest_sha256"] = changed["prepared_manifest"][
        "manifest_sha256"
    ]
    changed["cases"][0]["system_sha256"] = "f" * 64
    resign(changed)
    with pytest.raises(ValueError, match="captured release"):
        validate_pilot_report(changed)
    with pytest.raises(ValueError, match="captured release"):
        replay_reference_from_report(changed)


def test_rejects_missing_duplicate_and_forged_passing_cases(small_system):
    prepared, systems = prepared_fixture(small_system)
    original, _ = evaluate_fixture(prepared, systems)
    incomplete = copy.deepcopy(original)
    incomplete["cases"].pop()
    incomplete["case_count"] = incomplete["solved_count"] = 1
    resign(incomplete)
    with pytest.raises(ValueError):
        validate_pilot_report(incomplete)
    forged = copy.deepcopy(original)
    forged["cases"][0]["execution"]["verification"]["metrics"][
        "relative_l2_forward_error"
    ] = 0.01
    resign(forged)
    with pytest.raises(ValueError, match="contradicts"):
        validate_pilot_report(forged)
    duplicate = copy.deepcopy(original)
    duplicate["cases"][1] = copy.deepcopy(duplicate["cases"][0])
    duplicate["cases"][1]["index"] = 1
    resign(duplicate)
    with pytest.raises(ValueError):
        validate_pilot_report(duplicate)


def test_partial_preparation_cannot_be_scored_as_complete(small_system):
    prepared, systems = prepared_fixture(small_system)
    body = {k: v for k, v in prepared.items() if k != "manifest_sha256"}
    body.update(cases=body["cases"][:1], case_count=1, complete_split=False)
    report, _ = evaluate_fixture(seal_manifest(body), systems[:1])
    with pytest.raises(ValueError, match="complete frozen split"):
        score_pilot_report(report)


def test_runner_releases_system_before_loading_next(small_system):
    prepared, templates = prepared_fixture(small_system)
    refs = []

    def systems():
        for template in templates:
            assert all(ref() is None for ref in refs)
            value = EvaluationSystem(template.case_id, template.public, template.x_star)
            refs.append(weakref.ref(value))
            yield value
            del value

    def execute(executable, system, **settings):
        return RepeatResult(0, "timeout", 124, 30.1, "", None, None)

    evaluate_pilot(
        pathlib.Path("unused"),
        systems(),
        prepared,
        source_sha256="1" * 64,
        executable_sha256="2" * 64,
        runtime_manifest_sha256="3" * 64,
        execute=execute,
    )
    assert all(ref() is None for ref in refs)


def test_runner_rejects_changed_inputs_before_execution(small_system):
    from dataclasses import replace

    prepared, systems = prepared_fixture(small_system)
    # A solved initial guess preserves every shape and scalar case identifier.
    # It must not be evaluated under the original zero-start numerical identity.
    systems[0] = replace(
        systems[0], public=replace(systems[0].public, x0=systems[0].x_star)
    )

    def execute(*args, **kwargs):
        pytest.fail("substituted inputs reached the candidate executor")

    with pytest.raises(ValueError, match="loaded case differs"):
        evaluate_pilot(
            pathlib.Path("unused"),
            iter(systems),
            prepared,
            source_sha256="1" * 64,
            executable_sha256="2" * 64,
            runtime_manifest_sha256="3" * 64,
            execute=execute,
        )


def test_cli_family_alias_and_unsupported_tracks(capsys):
    assert main(["dataset", "families"]) == 0
    assert '"coverage"' in capsys.readouterr().out
    assert resolve_track("ns_mesh_pde", "coverage").family == "ns-mesh-pde"
    with pytest.raises(ValueError, match="unsupported family/track"):
        resolve_track("ns-mesh-pde", "performance")
    assert main(["dataset", "list", "--family", "ns-mesh-pde"]) == 2
