from __future__ import annotations

import io

import numpy as np
import pytest

from linear_solver_bench.archive import decode_system, encode_system
from linear_solver_bench.models import CsrMatrix


def test_archive_round_trip(small_system) -> None:
    decoded = decode_system(encode_system(small_system))
    assert decoded.case_id == small_system.case_id
    assert decoded.public.tolerance == small_system.public.tolerance
    np.testing.assert_array_equal(
        decoded.public.matrix.row_offsets, small_system.public.matrix.row_offsets
    )
    np.testing.assert_array_equal(decoded.public.b, small_system.public.b)
    np.testing.assert_array_equal(decoded.x_star, small_system.x_star)
    assert not decoded.x_star.flags.writeable


def test_archive_rejects_extra_fields(small_system) -> None:
    payload = encode_system(small_system)
    with np.load(io.BytesIO(payload), allow_pickle=False) as original:
        fields = {name: original[name] for name in original.files}
    fields["unexpected"] = np.asarray(1)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    with pytest.raises(ValueError, match="schema"):
        decode_system(buffer.getvalue())


def test_csr_rejects_duplicate_or_unsorted_columns() -> None:
    with pytest.raises(ValueError, match="strictly sorted"):
        CsrMatrix(
            n=2,
            row_offsets=np.asarray([0, 2, 2], dtype=np.uint64),
            column_indices=np.asarray([1, 1], dtype=np.uint32),
            values=np.asarray([1.0, 2.0], dtype=np.float64),
        )


def test_legacy_archive_bytes_are_unchanged(small_system) -> None:
    public = small_system.public
    expected = io.BytesIO()
    np.savez_compressed(
        expected,
        case_id=np.asarray(small_system.case_id),
        row_offsets=public.matrix.row_offsets,
        column_indices=public.matrix.column_indices,
        values=public.matrix.values,
        b=public.b,
        x0=public.x0,
        tolerance=np.asarray(public.tolerance, dtype=np.float64),
        x_star=small_system.x_star,
    )
    assert encode_system(small_system) == expected.getvalue()


@pytest.mark.parametrize("reference_kind", ["manufactured", "numerical", "none"])
def test_v2_archive_reference_semantics(small_system, reference_kind) -> None:
    from dataclasses import replace

    system = replace(
        small_system,
        reference_kind=reference_kind,
        x_star=None if reference_kind == "none" else small_system.x_star,
    )
    payload = encode_system(system, schema_version=2)
    decoded = decode_system(payload)
    assert decoded.reference_kind == reference_kind
    if reference_kind == "none":
        assert decoded.x_star is None
    else:
        np.testing.assert_array_equal(decoded.x_star, small_system.x_star)
    with np.load(io.BytesIO(payload), allow_pickle=False) as fields:
        assert fields["schema_version"].dtype == np.uint32
        assert fields["schema_version"] == 2
        assert ("x_star" in fields) == (reference_kind != "none")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", np.asarray(3, dtype=np.uint32)),
        ("schema_version", np.asarray(2, dtype=np.int64)),
        ("reference_kind", np.asarray("unknown")),
        ("reference_kind", np.asarray("none")),
        ("b", np.asarray([1, 2], dtype=np.float32)),
        ("unexpected", np.asarray(0)),
    ],
)
def test_v2_archive_rejects_invalid_fields(small_system, field, value) -> None:
    with np.load(io.BytesIO(encode_system(small_system, schema_version=2))) as archive:
        fields = {name: archive[name] for name in archive.files}
    fields[field] = value
    payload = io.BytesIO()
    np.savez_compressed(payload, **fields)
    with pytest.raises(ValueError):
        decode_system(payload.getvalue())


def test_reference_kind_requires_consistent_target(small_system) -> None:
    from dataclasses import replace

    for kind in ("manufactured", "numerical"):
        with pytest.raises(ValueError, match="requires x_star"):
            replace(small_system, reference_kind=kind, x_star=None)
    with pytest.raises(ValueError, match="must not contain"):
        replace(small_system, reference_kind="none")
    with pytest.raises(ValueError, match="legacy"):
        encode_system(
            replace(small_system, reference_kind="numerical"), schema_version=1
        )


def test_empty_csr_rows_are_valid() -> None:
    from scipy import sparse

    dense = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    matrix = CsrMatrix.from_scipy(sparse.csr_matrix(dense))
    np.testing.assert_array_equal(matrix.to_scipy().toarray(), dense)


@pytest.mark.parametrize("value", [1.0 + 1.0j, np.inf, np.nan, 2**53 + 1])
def test_sparse_conversion_rejects_lossy_or_nonfinite_values(value) -> None:
    from scipy import sparse

    with pytest.raises(ValueError):
        CsrMatrix.from_scipy(sparse.csr_matrix(np.asarray([[value]])))


def test_sparse_conversion_canonicalizes_without_integer_wrap() -> None:
    from scipy import sparse

    value = 2**62
    source = sparse.coo_matrix(
        (np.asarray([value, value], dtype=np.int64), ([0, 0], [0, 0])), shape=(1, 1)
    )
    matrix = CsrMatrix.from_scipy(source)
    assert matrix.values[0] == float(2**63)
    assert source.nnz == 2


def test_sparse_conversion_rejects_native_dimension_overflow() -> None:
    from scipy import sparse

    source = sparse.coo_matrix((2**31, 2**31), dtype=np.float64)
    with pytest.raises(ValueError, match="int32"):
        CsrMatrix.from_scipy(source)
