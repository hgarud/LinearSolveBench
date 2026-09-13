"""Binary protocol between the trusted orchestrator and native driver."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .models import MatrixInput

INPUT_MAGIC = b"LSBIN001"
OUTPUT_MAGIC = b"LSBOUT01"
PROTOCOL_VERSION = 1
INPUT_HEADER = struct.Struct("<8sIIQQd")
OUTPUT_HEADER = struct.Struct("<8sIiQQQQQB7x")


@dataclass(frozen=True)
class DriverOutput:
    status: int
    solution: np.ndarray
    elapsed_s: float
    create_s: float
    setup_s: float
    solve_s: float
    input_mutated: bool


def encode_input(value: MatrixInput) -> bytes:
    matrix = value.matrix
    return b"".join(
        (
            INPUT_HEADER.pack(
                INPUT_MAGIC,
                PROTOCOL_VERSION,
                0,
                matrix.n,
                matrix.nnz,
                value.tolerance,
            ),
            np.asarray(matrix.row_offsets, dtype="<u8").tobytes(),
            np.asarray(matrix.column_indices, dtype="<u4").tobytes(),
            np.asarray(matrix.values, dtype="<f8").tobytes(),
            np.asarray(value.b, dtype="<f8").tobytes(),
            np.asarray(value.x0, dtype="<f8").tobytes(),
        )
    )


def decode_output(payload: bytes, *, expected_n: int) -> DriverOutput:
    if len(payload) < OUTPUT_HEADER.size:
        raise ValueError("driver output is truncated")
    magic, version, status, n, elapsed, create, setup, solve, mutated = (
        OUTPUT_HEADER.unpack_from(payload)
    )
    if magic != OUTPUT_MAGIC or version != PROTOCOL_VERSION or n != expected_n:
        raise ValueError("driver output header is invalid")
    if mutated not in (0, 1):
        raise ValueError("driver mutation flag is invalid")
    expected_size = OUTPUT_HEADER.size + expected_n * 8
    if len(payload) != expected_size:
        raise ValueError("driver output size is invalid")
    if elapsed <= 0 or create + setup + solve != elapsed:
        raise ValueError("driver timing fields are invalid")
    solution = np.frombuffer(payload, dtype="<f8", offset=OUTPUT_HEADER.size).copy()
    return DriverOutput(
        status=status,
        solution=solution,
        elapsed_s=elapsed * 1e-9,
        create_s=create * 1e-9,
        setup_s=setup * 1e-9,
        solve_s=solve * 1e-9,
        input_mutated=bool(mutated),
    )
