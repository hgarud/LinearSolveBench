"""Frozen corpus loading and trusted SuiteSparse system preparation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping

import numpy as np
from scipy import sparse
from scipy.io import mmread

from .archive import decode_system, encode_system
from .models import CsrMatrix, EvaluationSystem, MatrixInput
from .paths import data_dir

CATALOGUE_SHA256 = "8226ebe538dcd0590d1415a2057da8df5e332d5b51886654a3d9a0c0181b92cd"
QUALIFICATION_SHA256 = (
    "30c7b765ed3f8c9712897c0be1d84890921cec90f5c6da122e9108100c8e3044"
)
RANKED_TOLERANCE_TIERS = (1.0e-4, 1.0e-3, 1.0e-2, 4.0e-2)
RANKED_MANIFEST_SHA256 = (
    "85ad37de2d7fc4b787e7eb0d5248eaebccf7ccaa04fd6f24e58637d37d6ef69f"
)
DEV_MANIFEST_SHA256 = "b743e8cf87f7c286b84d3ddac567d90460228b194df43e49c9e7fa8398cea454"
RHS_SCHEME = "rademacher-hmac-pcg64-v1"
MAX_DOWNLOAD_BYTES = 8 * 1024**3
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_ARCHIVE_MEMBERS = 128


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def identity_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_sha256(matrix: CsrMatrix) -> str:
    digest = hashlib.sha256(b"nsl-suitesparse-csr-float64-v1\0")
    digest.update(matrix.n.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(matrix.row_offsets, dtype="<u8").tobytes())
    digest.update(np.asarray(matrix.column_indices, dtype="<u4").tobytes())
    digest.update(np.asarray(matrix.values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _load_json(path: pathlib.Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def load_catalogue(root: pathlib.Path | None = None) -> dict[str, object]:
    value = _load_json((root or data_dir()) / "catalogue.json")
    claimed = value.get("catalogue_sha256")
    body = {key: item for key, item in value.items() if key != "catalogue_sha256"}
    if claimed != CATALOGUE_SHA256 or identity_sha256(body) != claimed:
        raise ValueError("catalogue identity mismatch")
    records = value.get("records")
    if value.get("eligible_count") != 350 or not isinstance(records, list):
        raise ValueError("catalogue count mismatch")
    return value


def load_qualification(root: pathlib.Path | None = None) -> dict[str, object]:
    value = _load_json((root or data_dir()) / "qualification.json")
    claimed = value.get("condition_results_sha256")
    body = {
        key: item for key, item in value.items() if key != "condition_results_sha256"
    }
    if claimed != QUALIFICATION_SHA256 or identity_sha256(body) != claimed:
        raise ValueError("qualification identity mismatch")
    if (
        value.get("schema_version") != 2
        or value.get("kind") != "linear-solver-bench-condition-evidence-v2"
        or value.get("method_counts") != {"spqr-guarded-onenormest-v1": 350}
        or value.get("policy", {}).get("condition_norm") != "1"
    ):
        raise ValueError("qualification policy mismatch")
    if value.get("complete") is not True or value.get("received_case_count") != 350:
        raise ValueError("qualification artifact is incomplete")
    results = value.get("results")
    if not isinstance(results, list) or len(results) != 350:
        raise ValueError("qualification result count mismatch")
    observed = Counter(
        result.get("condition", {}).get("status")
        for result in results
        if isinstance(result, Mapping)
    )
    if observed != Counter({"estimated": 213, "rank_deficient": 137}):
        raise ValueError("qualification status counts mismatch")
    return value


def _validate_split(value: dict[str, object], expected_split: str) -> dict[str, object]:
    if value.get("schema_version") != 1 or value.get("split") != expected_split:
        raise ValueError("dataset split identity mismatch")
    digest = value.get("manifest_sha256")
    if not isinstance(digest, str):
        raise ValueError("dataset split digest is missing")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if identity_sha256(body) != digest:
        raise ValueError("dataset split digest mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list) or value.get("case_count") != len(cases):
        raise ValueError("dataset split count mismatch")
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("index") != index:
            raise ValueError("dataset split case order is invalid")
    return value


def load_split(split: str, root: pathlib.Path | None = None) -> dict[str, object]:
    if split not in {"dev", "ranked"}:
        raise ValueError("split must be 'dev' or 'ranked'")
    filename = "dev-v1.json" if split == "dev" else "ranked-v1.json"
    value = _load_json((root or data_dir()) / filename)
    expected_count = 8 if split == "dev" else 203
    validated = _validate_split(value, split)
    if validated.get("case_count") != expected_count:
        raise ValueError(f"{split} split must contain {expected_count} cases")
    expected_digest = DEV_MANIFEST_SHA256 if split == "dev" else RANKED_MANIFEST_SHA256
    if validated.get("manifest_sha256") != expected_digest:
        raise ValueError(f"{split} split is not the frozen v1 manifest")
    return validated


def dataset_summary(root: pathlib.Path | None = None) -> dict[str, object]:
    catalogue = load_catalogue(root)
    qualification = load_qualification(root)
    ranked = load_split("ranked", root)
    dev = load_split("dev", root)
    return {
        "screened": catalogue["eligible_count"],
        "full_rank": qualification["status_counts"]["estimated"],
        "rank_deficient": qualification["status_counts"]["rank_deficient"],
        "condition_unsupported": sum(
            result["condition"]["status"] == "estimated"
            and result["condition"]["tightest_supported_tolerance"] is None
            for result in qualification["results"]
        ),
        "ranked": ranked["case_count"],
        "development": dev["case_count"],
        "tolerance_tiers": list(RANKED_TOLERANCE_TIERS),
    }


def rademacher_x_star(case_id: str, n: int, key: bytes) -> np.ndarray:
    if not case_id or type(n) is not int or n <= 0:
        raise ValueError("invalid case identity")
    if not isinstance(key, bytes) or len(key) < 16:
        raise ValueError("RHS key must contain at least 16 bytes")
    message = f"{RHS_SCHEME}\0{case_id}".encode("ascii")
    seed = int.from_bytes(hmac.new(key, message, hashlib.sha256).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    bits = generator.integers(0, 2, size=n, dtype=np.int8)
    return np.asarray(2 * bits - 1, dtype=np.float64)


def materialize_system(
    case: Mapping[str, object], matrix: sparse.spmatrix, key: bytes
) -> EvaluationSystem:
    canonical = CsrMatrix.from_scipy(matrix)
    if canonical.n != case.get("n") or canonical.nnz != case.get("nnz"):
        raise ValueError("canonical matrix disagrees with frozen metadata")
    expected_matrix_sha256 = case.get("matrix_sha256")
    if matrix_sha256(canonical) != expected_matrix_sha256:
        raise ValueError("canonical matrix digest mismatch")
    case_id = str(case["case_id"])
    x_star = rademacher_x_star(case_id, canonical.n, key)
    b = canonical.matvec(x_star)
    if not np.all(np.isfinite(b)):
        raise ValueError("manufactured right-hand side is nonfinite")
    return EvaluationSystem(
        case_id=case_id,
        public=MatrixInput(
            matrix=canonical,
            b=b,
            x0=np.zeros(canonical.n, dtype=np.float64),
            tolerance=float(case["tolerance"]),
        ),
        x_star=x_star,
    )


def _download(url: str, target: pathlib.Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and sha256_file(target) == expected_sha256:
        return
    request = urllib.request.Request(
        url, headers={"User-Agent": "linear-solver-bench/0.1"}
    )
    temporary_path: pathlib.Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) > MAX_DOWNLOAD_BYTES:
                raise ValueError("SuiteSparse archive exceeds the download limit")
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=f".{target.name}.", delete=False
            ) as temporary:
                temporary_path = pathlib.Path(temporary.name)
                digest = hashlib.sha256()
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            "SuiteSparse archive exceeds the download limit"
                        )
                    digest.update(block)
                    temporary.write(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded SuiteSparse archive digest mismatch")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_matrix_market_archive(
    path: pathlib.Path, case: Mapping[str, object]
) -> sparse.csr_matrix:
    expected_member = f"{case['name']}/{case['name']}.mtx"
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = tuple(archive.getmembers())
            if not members or len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("unsafe SuiteSparse archive member count")
            for member in members:
                pure = pathlib.PurePosixPath(member.name)
                if (
                    pure.is_absolute()
                    or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe SuiteSparse archive member")
            matches = [
                member
                for member in members
                if member.isfile() and member.name == expected_member
            ]
            if len(matches) != 1 or matches[0].size > MAX_MEMBER_BYTES:
                raise ValueError("expected Matrix Market member is absent or too large")
            source = archive.extractfile(matches[0])
            if source is None:
                raise ValueError("Matrix Market member is unreadable")
            with source:
                loaded = mmread(source, spmatrix=True)
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("SuiteSparse archive is unreadable") from exc
    if not sparse.issparse(loaded) or np.iscomplexobj(loaded.data):
        raise ValueError("SuiteSparse matrix must be real and sparse")
    matrix = sparse.csr_matrix(loaded, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if matrix.shape != (case["n"], case["n"]) or matrix.nnz != case["nnz"]:
        raise ValueError("Matrix Market payload disagrees with frozen metadata")
    return matrix


def prepare_split(
    split: str,
    output: pathlib.Path,
    *,
    rhs_key: bytes | None = None,
    cache: pathlib.Path | None = None,
    case_ids: Iterable[str] | None = None,
) -> dict[str, object]:
    manifest = load_split(split)
    if rhs_key is None:
        public_key = manifest.get("public_rhs_key_hex")
        if not isinstance(public_key, str):
            raise ValueError("ranked preparation requires an operator-held RHS key")
        rhs_key = bytes.fromhex(public_key)
    selected_ids = set(case_ids or ())
    cases = [
        case
        for case in manifest["cases"]
        if not selected_ids or case["case_id"] in selected_ids
    ]
    if selected_ids != {case["case_id"] for case in cases} and selected_ids:
        raise ValueError("one or more requested case IDs are not in the split")
    output = pathlib.Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("prepared output directory must be absent or empty")
    cache_root = pathlib.Path(cache or data_dir() / "downloads").resolve()
    if cache_root == output or output in cache_root.parents:
        raise ValueError("download cache must be outside the prepared output")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        prepared_cases = []
        for output_index, case in enumerate(cases):
            archive = (
                cache_root / f"{int(case['matrix_id']):04d}-{case['case_id']}.tar.gz"
            )
            _download(str(case["source_url"]), archive, str(case["archive_sha256"]))
            matrix = load_matrix_market_archive(archive, case)
            system = materialize_system(case, matrix, rhs_key)
            payload = encode_system(system)
            filename = f"{output_index:04d}-{case['case_id']}.npz"
            (staging / filename).write_bytes(payload)
            prepared_cases.append(
                {
                    "index": output_index,
                    "case_id": case["case_id"],
                    "source_index": case["index"],
                    "relpath": filename,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "n": case["n"],
                    "nnz": case["nnz"],
                    "tolerance": case["tolerance"],
                }
            )
        body = {
            "schema_version": 1,
            "kind": "linear-solver-bench-prepared-systems",
            "source_manifest_sha256": manifest["manifest_sha256"],
            "rhs_scheme": RHS_SCHEME,
            "case_count": len(prepared_cases),
            "cases": prepared_cases,
        }
        prepared_manifest = {**body, "manifest_sha256": identity_sha256(body)}
        (staging / "manifest.json").write_text(
            json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return prepared_manifest


def load_prepared_manifest(root: pathlib.Path) -> dict[str, object]:
    root = pathlib.Path(root).resolve()
    manifest = _load_json(root / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "linear-solver-bench-prepared-systems"
        or manifest.get("rhs_scheme") != RHS_SCHEME
    ):
        raise ValueError("prepared manifest identity mismatch")
    digest = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if digest != identity_sha256(body):
        raise ValueError("prepared manifest digest mismatch")
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or manifest.get("case_count") != len(cases)
    ):
        raise ValueError("prepared case list is invalid")
    return manifest


def load_prepared(root: pathlib.Path) -> tuple[EvaluationSystem, ...]:
    root = pathlib.Path(root).resolve()
    manifest = load_prepared_manifest(root)
    cases = manifest["cases"]
    systems = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or case.get("index") != index:
            raise ValueError("prepared case order mismatch")
        relpath = pathlib.PurePosixPath(str(case.get("relpath", "")))
        if relpath.is_absolute() or len(relpath.parts) != 1 or relpath.suffix != ".npz":
            raise ValueError("prepared case path is invalid")
        path = root / relpath.name
        if not path.is_file() or sha256_file(path) != case["sha256"]:
            raise ValueError("prepared case digest mismatch")
        system = decode_system(path.read_bytes())
        if (
            system.case_id != case["case_id"]
            or system.public.matrix.n != case["n"]
            or system.public.matrix.nnz != case["nnz"]
            or system.public.tolerance != case["tolerance"]
        ):
            raise ValueError("prepared case identity mismatch")
        systems.append(system)
    return tuple(systems)
