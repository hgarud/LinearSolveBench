import json

import pytest

from linear_solver_bench import build
from linear_solver_bench.submission import check, package


def test_package_creates_the_exact_public_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "validate_source", lambda _source: None)
    source = tmp_path / "candidate.c"
    source.write_text("int solver_create(void) { return 0; }\n", encoding="utf-8")
    output = tmp_path / "entry"

    metadata = package(
        source,
        output,
        name="small-solver",
        families=["ns_mesh_pde"],
        license_name="Apache-2.0",
        submitter="Ada",
    )

    assert {path.name for path in output.iterdir()} == {
        "README.md",
        "solver.c",
        "submission.json",
    }
    assert check(output) == metadata


def test_check_rejects_a_changed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "validate_source", lambda _source: None)
    source = tmp_path / "candidate.c"
    source.write_text("initial", encoding="utf-8")
    output = tmp_path / "entry"
    package(
        source,
        output,
        name="small-solver",
        families=["flash"],
        license_name="MIT",
        submitter="Ada",
    )
    (output / "solver.c").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="recorded hash"):
        check(output)


def test_check_rejects_non_string_names(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "validate_source", lambda _source: None)
    source = tmp_path / "candidate.c"
    source.write_text("initial", encoding="utf-8")
    output = tmp_path / "entry"
    package(
        source,
        output,
        name="small-solver",
        families=["flash"],
        license_name="MIT",
        submitter="Ada",
    )
    metadata_path = output / "submission.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["name"] = 7
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="name must use"):
        check(output)
