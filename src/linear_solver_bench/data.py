"""Load the two public datasets directly from their source archives."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import tarfile
import tempfile
import urllib.error
import urllib.request
import warnings
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import numpy as np
from scipy import sparse
from scipy.io import mmread

from .assets import asset_root
from .families import FLASH, NS_MESH_PDE, Family, resolve_family
from .models import BenchmarkCase, CaseInput, CsrMatrix

MAX_ARCHIVE_BYTES = 8 * 1024**3
_MANIFEST_DIGESTS = {
    NS_MESH_PDE.id: "4f40ea3871e0e7c8cc9aa221793a21c87ecdd2909c2e48155ede06d3648844c0",
    FLASH.id: "0746f3d4c419ce91469252407588eb585de08766b5425c5796cd1f76f1f54168",
}
_FLASH_FILES = {"matrix.npz", "b.npy", "x0.npy", "case.json"}


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_keys(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has unexpected fields")
    return value


def _validate_manifest(manifest: object, family: Family | None = None) -> dict:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("benchmark manifest must contain a case list")
    ids = {NS_MESH_PDE.id, FLASH.id}
    if manifest.get("family") not in ids:
        raise ValueError("manifest has an unsupported family")
    if family is not None and manifest["family"] != family.id:
        raise ValueError("manifest family does not match the requested family")
    if not manifest["cases"]:
        raise ValueError("benchmark manifest contains no cases")
    seen = set()
    for index, case in enumerate(manifest["cases"]):
        if not isinstance(case, dict) or case.get("index") != index:
            raise ValueError("manifest case order is invalid")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("manifest case IDs must be unique, nonempty strings")
        seen.add(case_id)
        _positive_int(case.get("n"), "n")
        _positive_int(case.get("nnz"), "nnz")
        tolerance = case.get("tolerance")
        if not isinstance(tolerance, (int, float)) or not 0 < tolerance < 1:
            raise ValueError("case tolerance must be between zero and one")
        source = case.get("source")
        if not isinstance(source, dict):
            raise ValueError("case source must be an object")
        _digest(source.get("archive_sha256"), "archive_sha256")
        _digest(source.get("matrix_sha256"), "matrix_sha256")
        archive_bytes = _positive_int(source.get("archive_bytes"), "archive_bytes")
        if archive_bytes > MAX_ARCHIVE_BYTES:
            raise ValueError("source archive exceeds the download limit")
    return manifest


def load_manifest(family: str | Family) -> dict:
    """Load and authenticate a bundled public development manifest."""
    family = resolve_family(family) if isinstance(family, str) else family
    if family not in {NS_MESH_PDE, FLASH}:
        raise ValueError("unsupported benchmark family")
    try:
        path = asset_root("families") / "families" / family.manifest
        manifest = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read the bundled benchmark manifest") from exc
    manifest = _validate_manifest(manifest, family)
    claimed = _digest(manifest.get("manifest_sha256"), "manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(_json_bytes(body)).hexdigest()
    if claimed != actual or claimed != _MANIFEST_DIGESTS[family.id]:
        raise ValueError("bundled benchmark manifest has the wrong digest")
    return manifest


def _fetch(source: Mapping[str, object], cache: Path) -> Path:
    """Download one archive and return it only after size and hash verification."""
    expected_hash = _digest(source.get("archive_sha256"), "archive_sha256")
    expected_size = _positive_int(source.get("archive_bytes"), "archive_bytes")
    if expected_size > MAX_ARCHIVE_BYTES:
        raise ValueError("source archive exceeds the download limit")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / expected_hash
    if target.is_file() and not target.is_symlink():
        if (
            target.stat().st_size == expected_size
            and _sha256_file(target) == expected_hash
        ):
            return target
        target.unlink()

    url = source.get("url")
    if not isinstance(url, str):
        raise ValueError("source URL must be text")
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "file"}:
        raise ValueError("source URL must use HTTPS")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            total = 0
            if parsed.scheme == "file":
                response = Path(unquote(parsed.path)).open("rb")
            else:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "linear-solver-bench/1",
                        "Accept-Encoding": "identity",
                    },
                )
                response = urllib.request.urlopen(request, timeout=120)
            with response:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > expected_size:
                        raise ValueError("download exceeds its declared size")
                    digest.update(block)
                    output.write(block)
        if total != expected_size:
            raise ValueError("download size disagrees with the manifest")
        if digest.hexdigest() != expected_hash:
            raise ValueError("download digest disagrees with the manifest")
        os.replace(temporary, target)
        return target
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError("archive download failed") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _matrix_sha256(matrix: CsrMatrix) -> str:
    digest = hashlib.sha256(b"linear-solver-bench-csr-float64-v2\0")
    digest.update(matrix.n.to_bytes(8, "little"))
    digest.update(np.asarray(matrix.row_offsets, dtype="<u8").tobytes())
    digest.update(np.asarray(matrix.column_indices, dtype="<u4").tobytes())
    digest.update(np.asarray(matrix.values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _vector_sha256(vector: np.ndarray) -> str:
    digest = hashlib.sha256(b"linear-solver-bench-vector-float64-v2\0")
    digest.update(len(vector).to_bytes(8, "little"))
    digest.update(np.asarray(vector, dtype="<f8").tobytes())
    return digest.hexdigest()


def _input_sha256(case_id: str, value: CaseInput, target: np.ndarray | None) -> str:
    body = {
        "case_id": case_id,
        "matrix_sha256": _matrix_sha256(value.matrix),
        "b_sha256": _vector_sha256(value.b),
        "x0_sha256": _vector_sha256(value.x0),
        "tolerance": value.tolerance,
        "reference_kind": "none" if target is None else "manufactured",
        "reference_sha256": None if target is None else _vector_sha256(target),
    }
    return hashlib.sha256(_json_bytes(body)).hexdigest()


def _load_suitesparse(path: Path, case: Mapping[str, object]) -> CsrMatrix:
    source = case["source"]
    expected = f"{source['name']}/{source['name']}.mtx"
    matrix = None
    expanded = 0
    seen = set()
    try:
        with tarfile.open(path, "r:gz") as archive:
            for count, member in enumerate(archive, 1):
                name = PurePosixPath(member.name)
                expanded += member.size
                if (
                    count > 128
                    or member.size < 0
                    or expanded > MAX_ARCHIVE_BYTES
                    or name.is_absolute()
                    or ".." in name.parts
                    or member.name in seen
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe SuiteSparse archive member")
                seen.add(member.name)
                if member.name != expected:
                    continue
                handle = archive.extractfile(member)
                if not member.isfile() or handle is None:
                    raise ValueError("SuiteSparse matrix member is unreadable")
                with handle:
                    header = handle.readline(1024).decode("ascii").lower().split()
                    if (
                        len(header) != 5
                        or header[:3]
                        != [
                            "%%matrixmarket",
                            "matrix",
                            "coordinate",
                        ]
                        or header[3] not in {"real", "integer", "pattern"}
                    ):
                        raise ValueError(
                            "SuiteSparse matrix must be real coordinate data"
                        )
                    dimensions = handle.readline(4096)
                    while dimensions.startswith(b"%"):
                        dimensions = handle.readline(4096)
                    size = [int(item) for item in dimensions.split()]
                    if len(size) != 3 or size[:2] != [case["n"], case["n"]]:
                        raise ValueError(
                            "Matrix Market dimensions disagree with the case"
                        )
                    handle.seek(0)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="The default value for `spmatrix` is changing",
                            category=DeprecationWarning,
                        )
                        loaded = mmread(handle)
                if not sparse.issparse(loaded) or np.iscomplexobj(loaded.data):
                    raise ValueError("SuiteSparse matrix must be real and sparse")
                matrix = CsrMatrix.from_scipy(loaded)
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        raise ValueError("SuiteSparse archive is unreadable") from exc
    if matrix is None:
        raise ValueError("SuiteSparse matrix is missing from its archive")
    if (
        matrix.n != case["n"]
        or matrix.nnz != case["nnz"]
        or _matrix_sha256(matrix) != source["matrix_sha256"]
    ):
        raise ValueError("SuiteSparse matrix disagrees with the manifest")
    scale = float(np.max(np.abs(matrix.values), initial=0))
    difference = matrix.to_scipy() - matrix.to_scipy().T
    asymmetry = float(np.max(np.abs(difference.data), initial=0))
    if scale == 0 or asymmetry <= 1e-12 * scale:
        raise ValueError("NS mesh matrices must be materially nonsymmetric")
    return matrix


def _rademacher_target(manifest: Mapping[str, object], matrix: CsrMatrix) -> np.ndarray:
    rhs = manifest.get("rhs")
    if not isinstance(rhs, dict) or rhs.get("scheme") != "rademacher-hmac-sha256-v2":
        raise ValueError("NS manifest has an unsupported RHS recipe")
    try:
        key = bytes.fromhex(rhs["public_key_hex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("NS development manifest has no valid public RHS key") from exc
    if len(key) < 16 or hashlib.sha256(key).hexdigest() != rhs.get("key_id"):
        raise ValueError("NS public RHS key does not match its commitment")
    release_id = manifest.get("release_id")
    domain = _json_bytes(
        {
            "scheme": rhs["scheme"],
            "release_id": (
                "ns-mesh-cohort-v2" if release_id == "ns-mesh" else release_id
            ),
            "family": manifest["family"],
            "matrix_sha256": _matrix_sha256(matrix),
            "rhs_kind": "rademacher",
            "draw_index": rhs["draw_index"],
        }
    )
    raw = bytearray()
    for block in range((matrix.n + 255) // 256):
        message = domain + b"\0" + block.to_bytes(8, "little")
        raw.extend(hmac.digest(key, message, "sha256"))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[
        : matrix.n
    ]
    return bits.astype(np.float64) * 2 - 1


def _read_npy(payload: bytes, shape: tuple[int, ...], kinds: set[str]) -> np.ndarray:
    handle = io.BytesIO(payload)
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        actual, _, dtype = np.lib.format.read_array_header_1_0(
            handle, max_header_size=4096
        )
    elif version == (2, 0):
        actual, _, dtype = np.lib.format.read_array_header_2_0(
            handle, max_header_size=4096
        )
    else:
        raise ValueError("unsupported NPY format")
    if actual != shape or dtype.hasobject or dtype.kind not in kinds:
        raise ValueError("array shape or dtype disagrees with the case")
    count = int(np.prod(actual, dtype=np.int64)) if actual else 1
    if len(payload) - handle.tell() != count * dtype.itemsize:
        raise ValueError("array byte count disagrees with its header")
    return np.load(io.BytesIO(payload), allow_pickle=False, max_header_size=4096)


def _flash_files(path: Path, case: Mapping[str, object]) -> dict[str, bytes]:
    limits = {
        "matrix.npz": 32 * case["nnz"] + 16 * (case["n"] + 1) + 65536,
        "b.npy": 8 * case["n"] + 4096,
        "x0.npy": 8 * case["n"] + 4096,
        "case.json": 65536,
    }
    result = {}
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if (
                        member.filename not in _FLASH_FILES
                        or member.filename in result
                        or member.is_dir()
                        or member.file_size > limits[member.filename]
                    ):
                        raise ValueError("unsafe FLASH archive member")
                    result[member.filename] = archive.read(member)
        else:
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    if (
                        member.name not in _FLASH_FILES
                        or member.name in result
                        or not member.isfile()
                        or member.size < 0
                        or member.size > limits[member.name]
                    ):
                        raise ValueError("unsafe FLASH archive member")
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("FLASH archive member is unreadable")
                    result[member.name] = handle.read(limits[member.name] + 1)
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError("FLASH archive is unreadable") from exc
    if set(result) != _FLASH_FILES:
        raise ValueError("FLASH archive must contain exactly four files")
    return result


def _flash_matrix(payload: bytes, n: int, nnz: int) -> CsrMatrix:
    limits = {
        "data.npy": nnz * 8 + 4096,
        "indices.npy": nnz * 8 + 4096,
        "indptr.npy": (n + 1) * 8 + 4096,
        "format.npy": 4096,
        "shape.npy": 4096,
        "_is_array.npy": 4096,
    }
    arrays = {}
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                if (
                    member.filename not in limits
                    or member.filename in arrays
                    or member.file_size > limits[member.filename]
                ):
                    raise ValueError("unsafe compressed matrix member")
                arrays[member.filename] = archive.read(member)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("FLASH matrix is unreadable") from exc
    required = {"data.npy", "indices.npy", "indptr.npy", "format.npy", "shape.npy"}
    if not required.issubset(arrays):
        raise ValueError("FLASH matrix archive is incomplete")
    values = _read_npy(arrays["data.npy"], (nnz,), {"f"})
    columns = _read_npy(arrays["indices.npy"], (nnz,), {"i", "u"})
    rows = _read_npy(arrays["indptr.npy"], (n + 1,), {"i", "u"})
    shape = _read_npy(arrays["shape.npy"], (2,), {"i", "u"})
    form = _read_npy(arrays["format.npy"], (), {"S"})
    if "_is_array.npy" in arrays:
        _read_npy(arrays["_is_array.npy"], (), {"b"})
    if tuple(shape) != (n, n) or form.item() != b"csr" or values.dtype != np.float64:
        raise ValueError("FLASH matrix must be float64 CSR")
    if (
        np.any(columns < 0)
        or np.any(columns >= n)
        or np.any(rows < 0)
        or np.any(rows > nnz)
    ):
        raise ValueError("FLASH CSR indices are outside the matrix")
    if np.any(values == 0):
        raise ValueError("FLASH matrix contains explicit zeros")
    return CsrMatrix(
        n,
        np.asarray(rows, dtype=np.uint64),
        np.asarray(columns, dtype=np.uint32),
        values,
    )


def _load_flash(path: Path, case: Mapping[str, object]) -> CaseInput:
    files = _flash_files(path, case)
    try:
        metadata = json.loads(files["case.json"].decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("FLASH metadata is unreadable") from exc
    expected_keys = {
        "schema_version",
        "case_id",
        "family",
        "capture_kind",
        "n",
        "nnz",
        "relative_tolerance",
        "absolute_tolerance",
        "provenance_group",
        "source_run",
        "step",
        "solve_index",
        "matrix_sha256",
        "b_sha256",
        "x0_sha256",
        "files",
    }
    _strict_keys(metadata, expected_keys, "FLASH metadata")
    source = case["source"]
    admission = case["admission"]
    expected = {
        "schema_version": 1,
        "case_id": case["case_id"],
        "family": FLASH.id,
        "capture_kind": "scalar-magnetic-diffusion",
        "n": case["n"],
        "nnz": case["nnz"],
        "relative_tolerance": case["tolerance"],
        "absolute_tolerance": 0,
        "provenance_group": case["provenance_group"],
        "source_run": case["source_run"],
        "step": admission["step"],
        "solve_index": admission["solve_index"],
        "matrix_sha256": source["matrix_sha256"],
        "b_sha256": source["b_sha256"],
        "x0_sha256": source["x0_sha256"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("FLASH metadata disagrees with the manifest")
    digests = _strict_keys(
        metadata["files"], {"matrix.npz", "b.npy", "x0.npy"}, "FLASH files"
    )
    if any(
        hashlib.sha256(files[name]).hexdigest() != digest
        for name, digest in digests.items()
    ):
        raise ValueError("FLASH file digest mismatch")
    matrix = _flash_matrix(files["matrix.npz"], case["n"], case["nnz"])
    b = _read_npy(files["b.npy"], (case["n"],), {"f"})
    x0 = _read_npy(files["x0.npy"], (case["n"],), {"f"})
    value = CaseInput(matrix, b, x0, case["tolerance"])
    if (
        _matrix_sha256(matrix) != source["matrix_sha256"]
        or _vector_sha256(value.b) != source["b_sha256"]
        or _vector_sha256(value.x0) != source["x0_sha256"]
    ):
        raise ValueError("FLASH numerical inputs disagree with the manifest")
    return value


def iter_cases(
    manifest: dict,
    cache: Path,
    case_ids: Iterable[str] | None = None,
) -> Iterator[BenchmarkCase]:
    """Download, authenticate, decode, and yield one case at a time."""
    manifest = _validate_manifest(manifest)
    requested = None if case_ids is None else list(case_ids)
    known = {case["case_id"] for case in manifest["cases"]}
    if requested is not None:
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("case selection must contain distinct case IDs")
        missing = set(requested) - known
        if missing:
            raise ValueError(f"unknown case ID: {sorted(missing)[0]}")
        selected = set(requested)
    else:
        selected = known

    for case in manifest["cases"]:
        if case["case_id"] not in selected:
            continue
        archive = _fetch(case["source"], Path(cache))
        if manifest["family"] == NS_MESH_PDE.id:
            matrix = _load_suitesparse(archive, case)
            target = _rademacher_target(manifest, matrix)
            value = CaseInput(
                matrix,
                matrix.matvec(target),
                np.zeros(matrix.n, dtype=np.float64),
                case["tolerance"],
            )
        else:
            target = None
            value = _load_flash(archive, case)
        identity = _input_sha256(case["case_id"], value, target)
        qualification = case.get("qualification")
        if qualification is not None and qualification.get("system_sha256") != identity:
            raise ValueError("decoded inputs differ from the qualified system")
        yield BenchmarkCase(
            case_id=case["case_id"],
            input=value,
            target=target,
            input_sha256=identity,
            role=case.get("role", "scored"),
        )
