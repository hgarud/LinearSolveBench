"""Versioned acceptance limits, independent of a solver's stopping request."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ACCURACY_CONTRACT_ID = "three-gate-v1"
NS_MESH_ACCURACY_CONTRACT_ID = "ns-mesh-accuracy-v1"
FLASH_REPLAY_ACCURACY_CONTRACT_ID = "flash-replay-accuracy-v1"
FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
FORWARD_FACTOR = 5.0
FORWARD_FLOOR = 2.0e-12
FORWARD_CEILING = 2.0e-1


@dataclass(frozen=True)
class AccuracyThresholds:
    normwise_backward_error: float
    componentwise_backward_error: float | None
    forward_error: float | None
    relative_residual: float | None = None

    @classmethod
    def for_contract(cls, contract_id: str, tolerance: object) -> AccuracyThresholds:
        legacy = cls.from_tolerance(tolerance)
        if contract_id == ACCURACY_CONTRACT_ID:
            return legacy
        if contract_id == NS_MESH_ACCURACY_CONTRACT_ID:
            return cls(1.0e-10, 1.0e-8, 1.0e-5)
        if contract_id == FLASH_REPLAY_ACCURACY_CONTRACT_ID:
            return cls(
                normwise_backward_error=legacy.normwise_backward_error,
                componentwise_backward_error=None,
                forward_error=None,
                relative_residual=float(tolerance) * (1.0 + 1.0e-6),
            )
        raise ValueError(f"unknown accuracy contract: {contract_id}")

    @classmethod
    def from_tolerance(cls, value: object) -> AccuracyThresholds:
        if isinstance(value, bool):
            raise TypeError("tolerance must be a number")
        tolerance = float(value)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        return cls(
            normwise_backward_error=max(tolerance, 20.0 * FLOAT64_EPSILON),
            componentwise_backward_error=max(5.0 * tolerance, 100.0 * FLOAT64_EPSILON),
            forward_error=min(
                FORWARD_CEILING,
                max(FORWARD_FACTOR * tolerance, FORWARD_FLOOR),
            ),
        )
