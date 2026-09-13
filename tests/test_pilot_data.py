from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.io import mmwrite

from linear_solver_bench.archive import encode_system
from linear_solver_bench.families import resolve_track
from linear_solver_bench.manifests import (
    RHS_SCHEME,
    seal_manifest,
    validate_release,
    validate_release_splits,
)
from linear_solver_bench.models import CsrMatrix
from linear_solver_bench.pilot_dataset import (
    iter_pilot_prepared,
    load_pilot_prepared_manifest,
    prepare_release,
)
from linear_solver_bench.sources.download import fetch_archive
from linear_solver_bench.sources.flash import load_flash
from linear_solver_bench.workloads import (
    matrix_digest,
    qualify_ns,
    rademacher_target,
    system_digest,
    vector_digest,
)


def reseal(value):
    return seal_manifest(
        {key: item for key, item in value.items() if key != "manifest_sha256"}
    )


def make_release(case, *, family="ns-mesh-pde"):
    track = resolve_track(family, "coverage" if family == "ns-mesh-pde" else "replay")
    key = b"public-test-recipe-key-32-bytes!!"
    return seal_manifest(
        {
            "schema_version": 2,
            "kind": "linear-solver-bench-pilot-release",
            "release_id": "synthetic-test-v2",
            "family": family,
            "track": track.track,
            "split": "dev",
            "contracts": track.contracts,
            "execution": {
                "repetitions": 1,
                "max_iterations": 5000,
                "case_timeout_seconds": 10,
                "memory_bytes": 1024**3,
                "cpus": 1,
            },
            "rhs": {
                "scheme": RHS_SCHEME,
                "key_id": hashlib.sha256(key).hexdigest(),
                "public_key_hex": key.hex(),
                "draw_index": 0,
            }
            if family == "ns-mesh-pde"
            else None,
            "cases": [case],
        }
    )


def make_ns(tmp_path):
    operator = sparse.csr_matrix([[4.0, 1.0], [2.0, 3.0]])
    matrix = CsrMatrix.from_scipy(operator)
    buffer = io.BytesIO()
    mmwrite(buffer, operator)
    path = tmp_path / "synthetic.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("synthetic/synthetic.mtx")
        info.size = len(buffer.getvalue())
        archive.addfile(info, io.BytesIO(buffer.getvalue()))
    source = {
        "kind": "suitesparse",
        "url": "https://sparse.tamu.edu/MM/Example/synthetic.tar.gz",
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_bytes": path.stat().st_size,
        "matrix_sha256": matrix_digest(matrix),
        "matrix_id": 1,
        "group": "Example",
        "name": "synthetic",
    }
    case = {
        "index": 0,
        "case_id": "synthetic-ns",
        "n": 2,
        "nnz": 4,
        "tolerance": 1e-12,
        "role": "scored",
        "provenance_group": "synthetic",
        "source_run": "synthetic-operator",
        "weight": 1,
        "source": source,
        "admission": {
            "statement": "Synthetic test fixture only; not a scientific release.",
            "citations": ["https://example.org/synthetic-fixture"],
        },
        "qualification": None,
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / source["archive_sha256"]).write_bytes(path.read_bytes())
    return make_release(case), cache


def make_flash(tmp_path):
    matrix = CsrMatrix.from_scipy(sparse.csr_matrix([[3.0, -1.0], [-1.0, 2.0]]))
    b, x0 = np.array([1.25, -3.5]), np.array([0.25, -0.75])
    members = {}
    for name, value in (("b.npy", b), ("x0.npy", x0)):
        buffer = io.BytesIO()
        np.save(buffer, value)
        members[name] = buffer.getvalue()
    buffer = io.BytesIO()
    sparse.save_npz(buffer, matrix.to_scipy())
    members["matrix.npz"] = buffer.getvalue()
    metadata = {
        "schema_version": 1,
        "case_id": "synthetic-flash",
        "family": "magnetic_diffusion_flash",
        "capture_kind": "scalar-magnetic-diffusion",
        "n": 2,
        "nnz": 4,
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 0,
        "provenance_group": "synthetic",
        "source_run": "synthetic-run",
        "step": 1,
        "solve_index": 0,
        "matrix_sha256": matrix_digest(matrix),
        "b_sha256": vector_digest(b),
        "x0_sha256": vector_digest(x0),
        "files": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in members.items()
        },
    }
    members["case.json"] = json.dumps(metadata).encode()
    path = tmp_path / "synthetic-flash.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    source = {
        "kind": "huggingface",
        "url": "https://huggingface.co/datasets/example/synthetic/resolve/"
        + "a" * 40
        + "/synthetic.zip",
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_bytes": path.stat().st_size,
        "matrix_sha256": matrix_digest(matrix),
        "b_sha256": vector_digest(b),
        "x0_sha256": vector_digest(x0),
    }
    case = {
        "index": 0,
        "case_id": "synthetic-flash",
        "n": 2,
        "nnz": 4,
        "tolerance": 1e-8,
        "role": "scored",
        "provenance_group": "synthetic",
        "source_run": "synthetic-run",
        "weight": 1,
        "source": source,
        "admission": {
            "statement": "Synthetic fixture, not a captured scientific case.",
            "citations": ["https://example.org/synthetic-fixture"],
            "capture_kind": "scalar-magnetic-diffusion",
            "relative_tolerance": 1e-8,
            "absolute_tolerance": 0,
            "step": 1,
            "solve_index": 0,
        },
        "qualification": None,
    }
    return path, case, members


def test_ns_offline_prepare_qualifies_and_streams(tmp_path):
    release, cache = make_ns(tmp_path)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(release))
    prepared = prepare_release(
        manifest, tmp_path / "prepared", cache=cache, offline=True
    )
    assert prepared["complete_split"] is True
    assert prepared["cases"][0]["qualification"]["qualified"] is True
    systems = iter_pilot_prepared(tmp_path / "prepared")
    system = next(systems)
    assert system.reference_kind == "manufactured"
    assert np.all(np.abs(system.x_star) == 1)
    assert np.array_equal(system.public.b, system.public.matrix.matvec(system.x_star))
    assert np.array_equal(system.public.x0, np.zeros(2))
    with pytest.raises(StopIteration):
        next(systems)
    with pytest.raises(ValueError, match="trusted published|qualification"):
        load_pilot_prepared_manifest(tmp_path / "prepared", official=True)


def test_flash_preserves_warm_start_and_has_no_manufactured_truth(tmp_path):
    path, case, _ = make_flash(tmp_path)
    validate_release(make_release(case, family="magnetic_diffusion_flash"))
    system = load_flash(path, case)
    assert system.reference_kind == "none" and system.x_star is None
    assert np.array_equal(system.public.b, [1.25, -3.5])
    assert np.array_equal(system.public.x0, [0.25, -0.75])
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / case["source"]["archive_sha256"]).write_bytes(path.read_bytes())
    manifest = tmp_path / "flash.json"
    manifest.write_text(
        json.dumps(make_release(case, family="magnetic_diffusion_flash"))
    )
    prepare_release(manifest, tmp_path / "prepared", cache=cache, offline=True)
    restored = next(iter_pilot_prepared(tmp_path / "prepared"))
    assert np.array_equal(restored.public.x0, system.public.x0)


def test_release_rejects_unsupported_or_untrusted_contracts(tmp_path):
    release, _ = make_ns(tmp_path)
    validate_release(release)
    for field, value in (
        ("schema_version", 1),
        ("track", "performance"),
        ("family", "ns_mesh_pde"),
    ):
        invalid = copy.deepcopy(release)
        invalid[field] = value
        with pytest.raises(ValueError):
            validate_release(reseal(invalid))
    invalid = copy.deepcopy(release)
    invalid["execution"]["repetitions"] = 3
    with pytest.raises(ValueError, match="exactly one"):
        validate_release(reseal(invalid))
    invalid = copy.deepcopy(release)
    invalid["cases"][0]["admission"]["extra"] = "not public schema"
    with pytest.raises(ValueError, match="schema"):
        validate_release(reseal(invalid))


def test_prepared_tampering_and_wrong_key_fail(tmp_path):
    release, cache = make_ns(tmp_path)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(release))
    with pytest.raises(ValueError, match="key"):
        prepare_release(
            manifest, tmp_path / "wrong", rhs_key=b"x" * 32, cache=cache, offline=True
        )
    prepare_release(manifest, tmp_path / "prepared", cache=cache, offline=True)
    path = tmp_path / "prepared" / "0000.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="digest"):
        next(iter_pilot_prepared(tmp_path / "prepared"))


@pytest.mark.parametrize("change", ["b", "x0", "matrix", "reference"])
def test_flash_rejects_rehashed_prepared_inputs(tmp_path, monkeypatch, change):
    from linear_solver_bench.manifests import PUBLISHED_RELEASES

    archive, case, _ = make_flash(tmp_path)
    release = make_release(case, family="magnetic_diffusion_flash")
    identity = tuple(
        release[name] for name in ("release_id", "family", "track", "split")
    )
    monkeypatch.setitem(PUBLISHED_RELEASES, identity, release["manifest_sha256"])
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(release), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / case["source"]["archive_sha256"]).write_bytes(archive.read_bytes())
    root = tmp_path / "prepared"
    prepared = prepare_release(manifest, root, cache=cache, offline=True)
    original = next(iter_pilot_prepared(root, official=True))
    if change in {"b", "x0"}:
        changed = replace(
            original, public=replace(original.public, **{change: np.zeros(2)})
        )
    elif change == "matrix":
        operator = CsrMatrix.from_scipy(original.public.matrix.to_scipy() * 2)
        changed = replace(original, public=replace(original.public, matrix=operator))
    else:
        changed = replace(original, x_star=np.ones(2), reference_kind="numerical")
    entry = prepared["cases"][0]
    payload = encode_system(changed, schema_version=2)
    (root / entry["relpath"]).write_bytes(payload)
    entry.update(
        sha256=hashlib.sha256(payload).hexdigest(), system_sha256=system_digest(changed)
    )
    (root / "manifest.json").write_text(json.dumps(reseal(prepared)), encoding="utf-8")
    # Even a fully rehashed prepared archive cannot redefine a trusted capture.
    with pytest.raises(ValueError, match="captured release"):
        load_pilot_prepared_manifest(root, official=True)
    with pytest.raises(ValueError, match="captured release"):
        next(iter_pilot_prepared(root))


def test_ns_loader_rejects_requalified_substitute_operator(tmp_path):
    release, cache = make_ns(tmp_path)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(release), encoding="utf-8")
    root = tmp_path / "prepared"
    prepared = prepare_release(manifest, root, cache=cache, offline=True)
    original = next(iter_pilot_prepared(root))
    operator = CsrMatrix.from_scipy(original.public.matrix.to_scipy() * 2)
    changed = replace(
        original,
        public=replace(
            original.public, matrix=operator, b=operator.matvec(original.x_star)
        ),
    )
    entry = prepared["cases"][0]
    payload = encode_system(changed, schema_version=2)
    (root / entry["relpath"]).write_bytes(payload)
    entry.update(
        sha256=hashlib.sha256(payload).hexdigest(),
        system_sha256=system_digest(changed),
        qualification=qualify_ns(changed),
    )
    (root / "manifest.json").write_text(json.dumps(reseal(prepared)), encoding="utf-8")
    with pytest.raises(ValueError, match="numerical inputs"):
        next(iter_pilot_prepared(root))


@pytest.mark.parametrize("field", ["b_sha256", "x0_sha256"])
def test_flash_requires_frozen_vector_identities(tmp_path, field):
    archive, case, _ = make_flash(tmp_path)
    case["source"][field] = "invalid"
    with pytest.raises(ValueError, match="SHA-256"):
        validate_release(make_release(case, family="magnetic_diffusion_flash"))
    case["source"][field] = "f" * 64
    with pytest.raises(ValueError, match="metadata disagrees"):
        load_flash(archive, case)


def test_hmac_generator_is_domain_separated_and_stable():
    arguments = {
        "release_id": "release-v2",
        "family": "ns-mesh-pde",
        "matrix_sha256": "0" * 64,
        "n": 300,
        "draw_index": 0,
        "key": b"x" * 32,
    }
    first = rademacher_target(**arguments)
    assert np.array_equal(first, rademacher_target(**arguments))
    assert (
        vector_digest(first)
        == "5d14f06b0fe1111b251ba7a18306925508894700323d94ac32b26362528812ad"
    )
    assert np.mean(first * first) == 1
    for name, value in (
        ("release_id", "release-v3"),
        ("family", "other"),
        ("matrix_sha256", "1" * 64),
        ("draw_index", 1),
        ("key", b"y" * 32),
    ):
        assert not np.array_equal(
            first, rademacher_target(**{**arguments, name: value})
        )


def test_offline_cache_rejects_corruption(tmp_path):
    release, cache = make_ns(tmp_path)
    source = release["cases"][0]["source"]
    cached = fetch_archive(source, cache, offline=True)
    cached.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="invalid size or digest"):
        fetch_archive(source, cache, offline=True)


def test_download_resumes_verified_byte_range(tmp_path, monkeypatch):
    payload = b"downloaded-public-archive"
    source = {
        "url": "https://example.org/archive.zip",
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_bytes": len(payload),
    }
    (tmp_path / (source["archive_sha256"] + ".part")).write_bytes(payload[:5])

    class Response(io.BytesIO):
        status = 206
        headers = {
            "Content-Length": str(len(payload) - 5),
            "Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}",
        }

    class Opener:
        def open(self, request, timeout):
            assert request.headers["Range"] == "bytes=5-"
            return Response(payload[5:])

    monkeypatch.setattr("urllib.request.build_opener", lambda handler: Opener())
    assert fetch_archive(source, tmp_path).read_bytes() == payload


def test_only_official_suitesparse_archive_redirects_may_use_http():
    import urllib.request

    from linear_solver_bench.sources.download import _SourceRedirect

    request = urllib.request.Request("https://sparse.tamu.edu/MM/Example/test.tar.gz")
    handler = _SourceRedirect(suitesparse=True)
    upgraded = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "http://sparse-files.engr.tamu.edu/MM/Example/test.tar.gz",
    )
    assert (
        upgraded.full_url == "http://sparse-files.engr.tamu.edu/MM/Example/test.tar.gz"
    )
    with pytest.raises(ValueError, match="outside HTTPS"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "http://example.org/test.tar.gz"
        )
    with pytest.raises(ValueError, match="outside HTTPS"):
        _SourceRedirect().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://sparse-files.engr.tamu.edu/MM/Example/test.tar.gz",
        )


def test_flash_rejects_extra_and_deceptive_array_members(tmp_path):
    path, case, members = make_flash(tmp_path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("../outside", b"no")
    with pytest.raises(ValueError, match="exactly four"):
        load_flash(path, case)
    deceptive = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        deceptive, {"descr": "<f8", "fortran_order": False, "shape": (2**40,)}
    )
    members["b.npy"] = deceptive.getvalue()
    metadata = json.loads(members["case.json"])
    metadata["files"]["b.npy"] = hashlib.sha256(members["b.npy"]).hexdigest()
    members["case.json"] = json.dumps(metadata).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    with pytest.raises(ValueError, match="shape or dtype"):
        load_flash(path, case)


def test_replay_weights_and_split_exposure_are_frozen(tmp_path):
    _, case, _ = make_flash(tmp_path)
    release = make_release(case, family="magnetic_diffusion_flash")
    second = copy.deepcopy(case)
    second.update(index=1, case_id="synthetic-other", source_run="synthetic-run-2")
    second["admission"]["step"] = 2
    release["cases"].append(second)
    for entry in release["cases"]:
        entry["weight"] = 0.5
    release = reseal(release)
    validate_release(release)
    wrong = copy.deepcopy(release)
    wrong["cases"][0]["weight"] = 0.25
    wrong["cases"][1]["weight"] = 0.75
    with pytest.raises(ValueError, match="equal mass"):
        validate_release(reseal(wrong))
    ranked = copy.deepcopy(release)
    ranked["split"] = "ranked"
    with pytest.raises(ValueError, match="exposure crosses"):
        validate_release_splits([release, reseal(ranked)])


def test_prepared_iteration_does_not_decode_remaining_cases_early(tmp_path):
    release, cache = make_ns(tmp_path)
    second = copy.deepcopy(release["cases"][0])
    second.update(index=1, case_id="synthetic-other")
    release["cases"].append(second)
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(reseal(release)))
    output = tmp_path / "prepared"
    prepared = prepare_release(manifest, output, cache=cache, offline=True)
    (output / prepared["cases"][1]["relpath"]).write_bytes(b"corrupt")
    systems = iter_pilot_prepared(output)
    assert next(systems).case_id == "synthetic-ns"
    with pytest.raises(ValueError, match="digest"):
        next(systems)
    partial = prepare_release(
        manifest,
        tmp_path / "partial",
        cache=cache,
        offline=True,
        case_ids=["synthetic-other"],
    )
    assert partial["complete_split"] is False
    assert partial["cases"][0]["source_index"] == 1
