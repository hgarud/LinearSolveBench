"""The immutable v1 mapping from public tolerance to trusted limits."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ACCURACY_CONTRACT_ID = "three-gate-v1"
FLOAT64_EPSILON = float(np.finfo(np.float64).eps)
FORWARD_FACTOR = 5.0
FORWARD_FLOOR = 2.0e-12
FORWARD_CEILING = 2.0e-1


@dataclass(frozen=True)
class AccuracyThresholds:
    normwise_backward_error: float
    componentwise_backward_error: float
    forward_error: float

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
