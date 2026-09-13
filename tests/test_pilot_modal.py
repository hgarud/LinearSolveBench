from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pathlib
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from linear_solver_bench.dataset import identity_sha256
from linear_solver_bench.families import resolve_track
from linear_solver_bench.manifests import RELEASE_KIND, RHS_SCHEME, seal_manifest
from linear_solver_bench.pilot_dataset import IDENTITY_FIELDS, PREPARED_KIND
from linear_solver_bench.protocol import OUTPUT_HEADER, OUTPUT_MAGIC, encode_input
from linear_solver_bench.workloads import (
    matrix_digest,
    qualify_ns,
    system_digest,
    vector_digest,
)


class FakeImage:
    @classmethod
    def from_registry(cls, *args, **kwargs):
        return cls()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self


class FakeApp:
    def __init__(self, name):
        self.name = name

    def local_entrypoint(self):
        return lambda function: function


@pytest.fixture
def venue_module(monkeypatch):
    module = types.ModuleType("modal")
    module.Image = FakeImage
    module.App = FakeApp
    module.Sandbox = types.SimpleNamespace(create=None)
    module.exception = types.SimpleNamespace(
        ExecTimeoutError=type("ExecTimeoutError", (Exception,), {}),
        SandboxTimeoutError=type("SandboxTimeoutError", (Exception,), {}),
        SandboxFilesystemNotFoundError=type(
            "SandboxFilesystemNotFoundError", (Exception,), {}
        ),
    )
    monkeypatch.setitem(sys.modules, "modal", module)
    path = pathlib.Path(__file__).resolve().parents[1] / "modal_app.py"
    spec = importlib.util.spec_from_file_location("lsb_test_modal_venue", path)
    venue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(venue)
    return venue


def output_bytes(system):
    solution = system.x_star
    if solution is None:
        solution = np.linalg.solve(
            system.public.matrix.to_scipy().toarray(), system.public.b
        )
    return (
        OUTPUT_HEADER.pack(
            OUTPUT_MAGIC,
            1,
            0,
            system.public.matrix.n,
            100_000_000,
            10_000_000,
            30_000_000,
            60_000_000,
            0,
        )
        + solution.astype("<f8").tobytes()
    )


class FakeProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)

    def wait(self):
        return self.returncode


class FakeFilesystem:
    def __init__(self, venue, systems):
        self.venue = venue
        self.systems = systems
        self.files = {}
        self.transfers = []
        self.copies = []
        self.removed = []

    def copy_from_local(self, source, destination):
        self.copies.append((source, destination))

    def write_bytes(self, payload, destination):
        assert not any(name.startswith("/work/cases/") for name in self.files)
        case_index = len(self.transfers)
        assert payload == encode_input(self.systems[case_index].public)
        if self.systems[case_index].x_star is not None:
            assert self.systems[case_index].x_star.tobytes() not in payload
        self.transfers.append((payload, destination))
        self.files[destination] = payload

    def read_bytes(self, path):
        if path not in self.files:
            raise self.venue.modal.exception.SandboxFilesystemNotFoundError()
        return self.files[path]

    def read_text(self, path):
        assert path.endswith("/manifest.json")
        return json.dumps({"manifest_sha256": "c" * 64})

    def remove(self, path, recursive=False):
        assert recursive
        self.removed.append(path)
        self.files = {
            name: data
            for name, data in self.files.items()
            if not name.startswith(path + "/")
        }


class FakeSandbox:
    def __init__(self, venue, systems, outcomes=None):
        self.venue = venue
        self.systems = systems
        self.filesystem = FakeFilesystem(venue, systems)
        self.outcomes = outcomes or [0] * len(systems)
        self.executions = []
        self.compiles = 0
        self.terminated = False
        self.creation = None

    def create(self, **kwargs):
        self.creation = kwargs
        return self

    def exec(self, *args, **kwargs):
        if args[0] == "mkdir":
            return FakeProcess()
        if args[0] == "linear-solver-bench":
            assert args[1:3] == ("candidate", "build")
            self.compiles += 1
            return FakeProcess(
                stdout=json.dumps(
                    {
                        "executable": "/work/candidate/solver",
                        "source_sha256": "a" * 64,
                        "executable_sha256": "b" * 64,
                    }
                )
            )
        assert args[0] == "/work/candidate/solver"
        index = len(self.executions)
        self.executions.append((args, kwargs))
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "missing":
            return FakeProcess()
        if outcome == "malformed":
            self.filesystem.files[args[2]] = b"invalid-output"
            return FakeProcess()
        if outcome == 0:
            self.filesystem.files[args[2]] = output_bytes(self.systems[index])
        return FakeProcess(returncode=outcome)

    def terminate(self):
        self.terminated = True


@pytest.fixture
def prepared_cases(small_system):
    systems = tuple(
        replace(small_system, case_id=f"fixture-{index}") for index in range(2)
    )
    contract = resolve_track("ns-mesh-pde", "coverage")
    key = b"public-test-rhs-key"
    body = {
        "schema_version": 2,
        "kind": RELEASE_KIND,
        "release_id": "transport-fixture",
        "family": contract.family,
        "track": contract.track,
        "split": "dev",
        "contracts": contract.contracts,
        "execution": {
            "repetitions": 1,
            "max_iterations": 5000,
            "case_timeout_seconds": 0.25,
            "memory_bytes": 2 * 1024**3,
            "cpus": 2,
        },
        "rhs": {
            "scheme": RHS_SCHEME,
            "key_id": hashlib.sha256(key).hexdigest(),
            "public_key_hex": key.hex(),
            "draw_index": 0,
        },
        "cases": [
            {
                "index": index,
                "case_id": system.case_id,
                "n": system.public.matrix.n,
                "nnz": system.public.matrix.nnz,
                "tolerance": system.public.tolerance,
                "role": "scored",
                "provenance_group": "test-group",
                "source_run": f"test-run-{index}",
                "weight": 1,
                "source": {
                    "kind": "suitesparse",
                    "url": f"https://sparse.tamu.edu/MM/Example/fixture-{index}.tar.gz",
                    "archive_sha256": "d" * 64,
                    "archive_bytes": 1024,
                    "matrix_sha256": matrix_digest(system.public.matrix),
                    "matrix_id": index + 1,
                    "group": "Example",
                    "name": f"fixture-{index}",
                },
                "admission": {
                    "statement": "Synthetic transport test fixture.",
                    "citations": ["https://example.org/fixture"],
                },
                "qualification": None,
            }
            for index, system in enumerate(systems)
        ],
    }
    release = seal_manifest(body)
    prepared = seal_manifest(
        {
            "schema_version": 2,
            "kind": PREPARED_KIND,
            "source_release": release,
            "source_manifest_sha256": release["manifest_sha256"],
            **{name: release[name] for name in IDENTITY_FIELDS},
            "case_count": len(systems),
            "complete_split": True,
            "cases": [
                {
                    "index": index,
                    "source_index": index,
                    "case_id": system.case_id,
                    "relpath": f"{system.case_id}.npz",
                    "n": system.public.matrix.n,
                    "nnz": system.public.matrix.nnz,
                    "tolerance": system.public.tolerance,
                    "system_sha256": system_digest(system),
                    "sha256": "2" * 64,
                    "qualification": qualify_ns(system),
                }
                for index, system in enumerate(systems)
            ],
        }
    )
    return prepared, systems


def attach_sandbox(venue, systems, outcomes=None):
    sandbox = FakeSandbox(venue, systems, outcomes)
    venue.modal.Sandbox.create = sandbox.create
    return sandbox


def test_pilot_modal_compiles_once_and_keeps_targets_local(
    venue_module, prepared_cases, tmp_path
):
    prepared, systems = prepared_cases
    sandbox = attach_sandbox(venue_module, systems)

    def stream():
        for index, system in enumerate(systems):
            assert len(sandbox.executions) == index
            assert len(sandbox.filesystem.removed) == index
            yield system

    report = venue_module._evaluate_pilot_sandbox(
        tmp_path / "policy.c", stream(), prepared
    )
    assert sandbox.compiles == 1
    assert len(sandbox.executions) == 2
    assert len(sandbox.filesystem.copies) == 1
    assert sandbox.filesystem.files == {}
    assert sandbox.creation["cpu"] == (2.0, 2.0)
    assert sandbox.creation["memory"] == (2048, 2048)
    assert sandbox.creation["block_network"] is True
    assert sandbox.creation["timeout"] == 371
    for args, kwargs in sandbox.executions:
        assert args[3:] == ("5000", "0.25")
        assert kwargs["timeout"] == 31
    assert sandbox.terminated
    assert report["repetitions"] == 1
    assert report["solved_count"] == 2
    assert report["venue"]["limits_enforced"] is True
    assert all(case["execution"]["index"] == 0 for case in report["cases"])
    assert "x_star" not in json.dumps(report)
    assert report["report_sha256"] == identity_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (9, "crash"),
        (124, "timeout"),
        (-1, "timeout"),
        ("missing", "invalid_output"),
        ("malformed", "invalid_output"),
    ],
)
def test_candidate_failure_continues_without_retry(
    venue_module, prepared_cases, tmp_path, outcome, expected
):
    prepared, systems = prepared_cases
    sandbox = attach_sandbox(venue_module, systems, [outcome, 0])
    report = venue_module._evaluate_pilot_sandbox(
        tmp_path / "policy.c", iter(systems), prepared
    )
    assert sandbox.compiles == 1
    assert len(sandbox.executions) == 2
    assert report["cases"][0]["execution"]["outcome"] == expected
    assert report["solved_count"] == 1
    assert sandbox.terminated


def test_exec_timeout_is_a_candidate_failure(venue_module, prepared_cases, tmp_path):
    prepared, systems = prepared_cases
    sandbox = attach_sandbox(
        venue_module, systems, [venue_module.modal.exception.ExecTimeoutError(), 0]
    )
    report = venue_module._evaluate_pilot_sandbox(
        tmp_path / "policy.c", iter(systems), prepared
    )
    assert len(sandbox.executions) == 2
    assert report["cases"][0]["execution"]["outcome"] == "timeout"
    assert report["solved_count"] == 1


@pytest.mark.parametrize("failure", ["sandbox", "transport"])
def test_infrastructure_failure_invalidates_run(
    venue_module, prepared_cases, tmp_path, failure
):
    prepared, systems = prepared_cases
    error = (
        venue_module.modal.exception.SandboxTimeoutError()
        if failure == "sandbox"
        else OSError("transport disconnected")
    )
    sandbox = attach_sandbox(venue_module, systems, [error, 0])
    with pytest.raises(type(error)):
        venue_module._evaluate_pilot_sandbox(
            tmp_path / "policy.c", iter(systems), prepared
        )
    assert len(sandbox.executions) == 1
    assert sandbox.terminated


def test_official_rejects_draft_before_sandbox_creation(
    venue_module, prepared_cases, tmp_path
):
    prepared, systems = prepared_cases
    sandbox = attach_sandbox(venue_module, systems)
    with pytest.raises(ValueError):
        venue_module._evaluate_pilot_sandbox(
            tmp_path / "policy.c", iter(systems), prepared, official=True
        )
    assert sandbox.creation is None


def test_modal_requires_exact_mib_allocation(venue_module, prepared_cases, tmp_path):
    prepared, systems = prepared_cases
    release = prepared["source_release"]
    release["execution"]["memory_bytes"] += 1
    prepared["source_release"] = seal_manifest(
        {key: value for key, value in release.items() if key != "manifest_sha256"}
    )
    prepared["source_manifest_sha256"] = prepared["source_release"]["manifest_sha256"]
    prepared = seal_manifest(
        {key: value for key, value in prepared.items() if key != "manifest_sha256"}
    )
    sandbox = attach_sandbox(venue_module, systems)
    with pytest.raises(ValueError, match="exact positive number of MiB"):
        venue_module._evaluate_pilot_sandbox(
            tmp_path / "policy.c", iter(systems), prepared
        )
    assert sandbox.creation is None


def test_main_dispatches_pilot_before_legacy_loading(
    venue_module, tmp_path, monkeypatch
):
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "manifest.json").write_text('{"schema_version": 2}', encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        venue_module,
        "_run_pilot",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        venue_module,
        "load_prepared_manifest",
        lambda *args: pytest.fail("legacy loader called"),
    )
    venue_module.main(
        source="policy.c",
        cases=str(cases),
        output=str(tmp_path / "report.json"),
        family="ns_mesh_pde",
        track="coverage",
    )
    assert len(observed) == 1
    assert observed[0][1]["family"] == "ns_mesh_pde"


def test_flash_modal_preserves_captured_input_and_emits_awaiting_reference(
    venue_module, prepared_cases, tmp_path
):
    from linear_solver_bench.archive import encode_system

    prepared, originals = prepared_cases
    systems = tuple(
        replace(
            system,
            public=replace(system.public, x0=system.x_star * 0.25),
            reference_kind="none",
            x_star=None,
        )
        for system in originals
    )
    release = prepared["source_release"]
    contract = resolve_track("magnetic_diffusion_flash", "replay")
    release.update(
        family=contract.family,
        track=contract.track,
        contracts=contract.contracts,
        rhs=None,
    )
    for case, system in zip(release["cases"], systems, strict=True):
        case["weight"] = 0.5
        case["source"] = {
            "kind": "huggingface",
            "url": (
                "https://huggingface.co/datasets/example/test-capture/resolve/"
                f"{'a' * 40}/case-{case['index']}.zip"
            ),
            "archive_sha256": "d" * 64,
            "archive_bytes": 1024,
            "matrix_sha256": matrix_digest(system.public.matrix),
            "b_sha256": vector_digest(system.public.b),
            "x0_sha256": vector_digest(system.public.x0),
        }
        case["admission"].update(
            capture_kind="scalar-magnetic-diffusion",
            relative_tolerance=case["tolerance"],
            absolute_tolerance=0,
            step=case["index"],
            solve_index=0,
        )
    release = seal_manifest(
        {key: value for key, value in release.items() if key != "manifest_sha256"}
    )
    prepared.update(
        source_release=release,
        source_manifest_sha256=release["manifest_sha256"],
        **{name: release[name] for name in IDENTITY_FIELDS},
    )
    cases = tmp_path / "cases"
    cases.mkdir()
    for entry, system in zip(prepared["cases"], systems, strict=True):
        payload = encode_system(system, schema_version=2)
        entry.update(
            system_sha256=system_digest(system),
            qualification=None,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        (cases / entry["relpath"]).write_bytes(payload)
    prepared = seal_manifest(
        {key: value for key, value in prepared.items() if key != "manifest_sha256"}
    )
    (cases / "manifest.json").write_text(json.dumps(prepared), encoding="utf-8")
    sandbox = attach_sandbox(venue_module, systems)
    report_path, score_path = tmp_path / "report.json", tmp_path / "score.json"
    venue_module._run_pilot(
        tmp_path / "policy.c",
        cases,
        report_path,
        family=contract.family,
        track=contract.track,
        official=False,
        calibration="",
        score_output=str(score_path),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert sandbox.compiles == 1
    assert len(sandbox.executions) == 2
    assert report["solved_count"] == 2
    assert report["track"] == "replay"
    assert score["speedup"] is None
    assert score["reason"] == "awaiting_reference"
    assert score["official"] is False
