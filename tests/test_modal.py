"""Modal transport tests use fakes: no credentials or network are needed."""

from __future__ import annotations

import importlib
import json
import pathlib
import subprocess
import sys
import types
import zlib

import numpy as np
import pytest

from linear_solver_bench import modal as venue
from linear_solver_bench.models import CaseInput, CsrMatrix, RawExecution


class FakeProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = [stdout] if stdout else []
        self.stderr = [stderr] if stderr else []

    def wait(self):
        return self.returncode


class FakeFilesystem:
    def __init__(self):
        self.files = {}
        self.copies = []
        self.removed = []

    def copy_from_local(self, local, remote):
        self.copies.append((pathlib.Path(local), remote))
        self.files[remote] = pathlib.Path(local).read_bytes()

    def write_bytes(self, value, path):
        self.files[path] = value

    def read_text(self, path):
        return self.files[path].decode()

    def remove(self, path, *, recursive=False):
        assert recursive
        self.removed.append(path)
        for name in list(self.files):
            if name == path or name.startswith(path + "/"):
                del self.files[name]


class FakeSandbox:
    def __init__(self):
        self.filesystem = FakeFilesystem()
        self.creation = None
        self.terminated = False
        self.builds = 0
        self.runs = []
        self.outcomes = []

    def create(self, **kwargs):
        self.creation = kwargs
        return self

    def exec(self, *args, **kwargs):
        if args[:3] == ("mkdir", "-p", "/work"):
            return FakeProcess()
        if args[:3] == ("python", "-c", venue._BUILD_EXEC):
            self.builds += 1
            self.filesystem.files["/work/build.json"] = json.dumps(
                {
                    "candidate_source_sha256": "a" * 64,
                    "candidate_executable_sha256": "b" * 64,
                    "reference_source_sha256": "c" * 64,
                    "reference_executable_sha256": "d" * 64,
                    "runtime_manifest_sha256": "e" * 64,
                    "image_id": "im-test",
                }
            ).encode()
            return FakeProcess()
        if args[:3] == ("python", "-c", venue._INPUT_EXEC):
            compressed = self.filesystem.files[args[4]]
            assert int(args[3]) == len(zlib.decompress(compressed))
            self.runs.append((args, kwargs))
            outcome = self.outcomes.pop(0) if self.outcomes else 0
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome == 0:
                self.filesystem.files[args[7]] = b"driver output"
            if outcome == "input-error":
                return FakeProcess(125, stderr=venue._INPUT_ERROR.encode())
            return FakeProcess(outcome)
        if args[:2] == ("head", "-c"):
            value = self.filesystem.files.get(args[3])
            if value is None:
                return FakeProcess(1)
            return FakeProcess(stdout=value[: int(args[2])])
        raise AssertionError(f"unexpected sandbox command: {args!r}")

    def terminate(self):
        self.terminated = True


class FakeImage:
    calls = []

    @classmethod
    def from_registry(cls, *args, **kwargs):
        cls.calls = [("from_registry", args, kwargs)]
        return cls()

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self

        return call


def fake_modal(sandbox):
    return types.SimpleNamespace(
        Image=FakeImage,
        App=types.SimpleNamespace(lookup=lambda *args, **kwargs: ("app", args, kwargs)),
        Sandbox=types.SimpleNamespace(create=sandbox.create),
    )


@pytest.fixture
def case():
    matrix = CsrMatrix(
        2,
        np.array([0, 2, 4], dtype=np.uint64),
        np.array([0, 1, 0, 1], dtype=np.uint32),
        np.array([3.0, 1.0, 1.0, 2.0], dtype=np.float64),
    )
    return CaseInput(
        matrix,
        np.array([1.0, 2.0], dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        1e-6,
    )


@pytest.fixture
def sources(tmp_path):
    candidate = tmp_path / "candidate.c"
    reference = tmp_path / "reference.c"
    candidate.write_text("candidate")
    reference.write_text("reference")
    return candidate, reference


def open_fake(monkeypatch, sources, *, sandbox=None):
    sandbox = sandbox or FakeSandbox()
    monkeypatch.setattr(venue, "require_modal", lambda: fake_modal(sandbox))
    context = venue.open_session(*sources, {"cpus": 2, "memory_bytes": 4 * 1024**3})
    return sandbox, context


def test_import_does_not_load_optional_modal():
    code = """
import importlib.abc
import sys

class RejectModal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "modal" or fullname.startswith("modal."):
            raise AssertionError("Modal was imported eagerly")

sys.meta_path.insert(0, RejectModal())
import linear_solver_bench.modal
assert "modal" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_missing_modal_has_an_actionable_error(monkeypatch):
    real_import = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "modal":
            raise ModuleNotFoundError("missing", name="modal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(venue.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"linear-solver-bench\[modal\]"):
        venue.require_modal()


def test_image_is_pinned_and_contains_only_code_and_native(monkeypatch):
    monkeypatch.setattr(
        venue, "require_modal", lambda: types.SimpleNamespace(Image=FakeImage)
    )
    venue.create_image()
    assert FakeImage.calls[0] == (
        "from_registry",
        (venue.BASE_IMAGE,),
        {"add_python": "3.12"},
    )
    copies = [call for call in FakeImage.calls if call[0] == "add_local_dir"]
    assert [call[1][0].name for call in copies] == ["linear_solver_bench", "native"]
    assert all(call[1][0].name not in {"families", "reference"} for call in copies)
    command = next(call for call in FakeImage.calls if call[0] == "run_commands")
    assert "build_runtime" in command[1][0]


def test_session_uses_one_network_disabled_fixed_resource_sandbox(monkeypatch, sources):
    sandbox, context = open_fake(monkeypatch, sources)
    with context as session:
        assert isinstance(session, venue.ModalSession)
    assert sandbox.creation["cpu"] == (2.0, 2.0)
    assert sandbox.creation["memory"] == (4096, 4096)
    assert sandbox.creation["block_network"] is True
    assert sandbox.creation["timeout"] == venue.SANDBOX_LIFETIME_SECONDS
    assert sandbox.terminated


def test_sandbox_start_failure_is_actionable(monkeypatch, sources):
    sandbox = FakeSandbox()

    def fail(**_kwargs):
        raise Exception("image build failed")

    sandbox.create = fail
    monkeypatch.setattr(venue, "require_modal", lambda: fake_modal(sandbox))

    with pytest.raises(RuntimeError, match="cannot start Modal sandbox"):
        with venue.open_session(*sources, {"cpus": 2, "memory_bytes": 4 * 1024**3}):
            pass


def test_build_compiles_both_solvers_once_and_reports_identities(monkeypatch, sources):
    sandbox, context = open_fake(monkeypatch, sources)
    with context as session:
        first = session.build()
        assert session.build() == first
    assert sandbox.builds == 1
    assert sandbox.filesystem.copies == [
        (sources[0], "/work/candidate.c"),
        (sources[1], "/work/reference.c"),
    ]
    assert first["candidate"] == {
        "source_sha256": "a" * 64,
        "executable_sha256": "b" * 64,
    }
    assert first["reference"] == {
        "id": venue.REFERENCE_ID,
        "source_sha256": "c" * 64,
        "executable_sha256": "d" * 64,
    }
    assert first["runtime_sha256"] == "e" * 64
    assert first["venue"] == {
        "id": "modal-cpu-v1",
        "image_id": "im-test",
        "base_image": venue.BASE_IMAGE,
        "cpus": 2,
        "memory_bytes": 4 * 1024**3,
        "limits_enforced": True,
    }


def test_reference_and_candidate_are_fresh_processes_in_the_same_sandbox(
    monkeypatch, sources, case
):
    sandbox, context = open_fake(monkeypatch, sources)
    with context as session:
        session.build()
        reference = session.run_reference(case)
        candidate = session.run_candidate(case)
    assert isinstance(reference, RawExecution)
    assert isinstance(candidate, RawExecution)
    assert [run[0][5] for run in sandbox.runs] == [
        "/work/reference",
        "/work/candidate",
    ]
    assert sandbox.runs[0][0][4] != sandbox.runs[1][0][4]
    assert all(run[1]["timeout"] == venue.RUN_TIMEOUT_SECONDS for run in sandbox.runs)
    assert len(sandbox.filesystem.copies) == 2  # only the two sources
    assert len(sandbox.filesystem.removed) == 2
    assert not any(name.startswith("/work/cases/") for name in sandbox.filesystem.files)


def test_transport_accepts_case_input_only(monkeypatch, sources, case):
    sandbox, context = open_fake(monkeypatch, sources)
    with context as session:
        session.build()
        with pytest.raises(TypeError, match="CaseInput"):
            session.run_candidate(object())
    assert not sandbox.runs


def test_output_transport_is_bounded(monkeypatch, sources, case):
    sandbox = FakeSandbox()
    original_exec = sandbox.exec

    def oversized(*args, **kwargs):
        process = original_exec(*args, **kwargs)
        if args[:3] == ("python", "-c", venue._INPUT_EXEC):
            sandbox.filesystem.files[args[7]] = b"x" * 10_000
        return process

    sandbox.exec = oversized
    sandbox, context = open_fake(monkeypatch, sources, sandbox=sandbox)
    with context as session:
        session.build()
        result = session.run_candidate(case)
    assert len(result.output) == venue.OUTPUT_HEADER.size + case.matrix.n * 8 + 1


@pytest.mark.parametrize("returncode", [9, 42])
def test_solver_failure_is_raw_evidence(monkeypatch, sources, case, returncode):
    sandbox = FakeSandbox()
    sandbox.outcomes = [returncode]
    sandbox, context = open_fake(monkeypatch, sources, sandbox=sandbox)
    with context as session:
        session.build()
        result = session.run_candidate(case)
    assert result.returncode == returncode
    assert result.output is None


@pytest.mark.parametrize("returncode", [124, -1])
def test_platform_interruption_invalidates_evaluation(
    monkeypatch, sources, case, returncode
):
    sandbox = FakeSandbox()
    sandbox.outcomes = [returncode]
    sandbox, context = open_fake(monkeypatch, sources, sandbox=sandbox)
    with pytest.raises(RuntimeError, match="incomplete"):
        with context as session:
            session.build()
            session.run_reference(case)
    assert sandbox.terminated


def test_input_preparation_failure_invalidates_evaluation(monkeypatch, sources, case):
    sandbox = FakeSandbox()
    sandbox.outcomes = ["input-error"]
    sandbox, context = open_fake(monkeypatch, sources, sandbox=sandbox)
    with pytest.raises(RuntimeError, match="input preparation failed"):
        with context as session:
            session.build()
            session.run_candidate(case)
    assert sandbox.terminated


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        ({"cpus": 0, "memory_bytes": 1024**2}, "cpus"),
        ({"cpus": True, "memory_bytes": 1024**2}, "cpus"),
        ({"cpus": 1, "memory_bytes": 0}, "whole number of MiB"),
        ({"cpus": 1, "memory_bytes": 1024**2 + 1}, "whole number of MiB"),
    ],
)
def test_resources_are_validated_before_modal_is_loaded(
    monkeypatch, sources, execution, message
):
    monkeypatch.setattr(
        venue,
        "require_modal",
        lambda: pytest.fail("Modal loaded before resource validation"),
    )
    with pytest.raises(ValueError, match=message):
        with venue.open_session(*sources, execution):
            pass


@pytest.mark.parametrize("corruption", [None, "too-large", "truncated", "trailing"])
def test_input_decompressor_is_bounded(tmp_path, corruption):
    original = bytes(range(256)) * 9_000
    compressed = zlib.compress(original, level=1)
    expected_size = len(original)
    if corruption == "too-large":
        expected_size -= 1
    elif corruption == "truncated":
        compressed = compressed[:-1]
    elif corruption == "trailing":
        compressed += b"trailing data"
    archive = tmp_path / "input.zlib"
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    executable = tmp_path / "copy.py"
    archive.write_bytes(compressed)
    executable.write_text(
        f"#!{sys.executable}\n"
        "import pathlib,sys\n"
        "pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes())\n"
    )
    executable.chmod(0o755)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            venue._INPUT_EXEC,
            str(expected_size),
            str(archive),
            str(executable),
            str(input_path),
            str(output_path),
        ],
        capture_output=True,
    )
    if corruption is None:
        assert result.returncode == 0
        assert output_path.read_bytes() == original
        assert not archive.exists()
    else:
        assert result.returncode == 125
        assert venue._INPUT_ERROR.encode() in result.stderr
        assert not output_path.exists()
        assert input_path.stat().st_size <= expected_size
