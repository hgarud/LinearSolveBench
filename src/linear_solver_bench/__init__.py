"""Trusted infrastructure for LinearSolverBench."""

from .accuracy import ACCURACY_CONTRACT_ID, AccuracyThresholds
from .models import CsrMatrix, EvaluationSystem, MatrixInput

__all__ = [
    "ACCURACY_CONTRACT_ID",
    "AccuracyThresholds",
    "CsrMatrix",
    "EvaluationSystem",
    "MatrixInput",
]

__version__ = "0.1.0"
