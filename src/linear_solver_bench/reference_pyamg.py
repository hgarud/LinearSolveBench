"""Optional fixed PyAMG reference for offline NS qualification.

Every solve starts from zero and uses the original operator. Two fixed
extended-residual corrections strengthen the empirical feasibility witness;
no inverse-norm certificate or candidate timing baseline is claimed.
Operators must enforce memory and wall-time limits in an isolated process.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import scipy
from scipy.sparse.linalg import gmres

from .manifests import (
    PYAMG_INEXACT_QUALIFICATION_CONFIG,
    PYAMG_INEXACT_QUALIFICATION_METHOD,
    PYAMG_QUALIFICATION_CONFIG,
    PYAMG_QUALIFICATION_METHOD,
    QUALIFICATION_LIMITS,
    validate_qualification,
)
from .verify import accuracy_metrics
from .workloads import _extended_residual, system_digest


def qualify_ns_pyamg(system):
    """Retain the original v1 method and its exact evidence format."""
    return _qualify(system, PYAMG_QUALIFICATION_METHOD, PYAMG_QUALIFICATION_CONFIG)


def qualify_ns_pyamg_inexact(system):
    """Use fixed inexact inner corrections; only the final gates admit a case."""
    return _qualify(
        system, PYAMG_INEXACT_QUALIFICATION_METHOD, PYAMG_INEXACT_QUALIFICATION_CONFIG
    )


def _qualify(system, method, config):
    try:
        import pyamg
    except ImportError as exc:
        raise RuntimeError(
            "install linear-solver-bench[qualification] for this offline method"
        ) from exc

    if pyamg.__version__ != config["pyamg_version"]:
        raise ValueError("the reference requires exactly PyAMG 5.3.0")
    if system.reference_kind != "manufactured" or system.x_star is None:
        raise ValueError("NS qualification requires its manufactured target")
    matrix = system.public.matrix
    operator = matrix.to_scipy().copy()
    # These constant coarse-space candidates depend on dimension only, never on
    # the manufactured target. Local Jacobi weighting avoids randomized spectral
    # estimates. 'symmetric' strength is a graph rule, not a symmetry assumption:
    # symmetry='nonsymmetric' builds distinct restriction and prolongation.
    hierarchy = pyamg.smoothed_aggregation_solver(
        operator,
        B=np.ones((matrix.n, 1)),
        BH=np.ones((matrix.n, 1)),
        symmetry=config["symmetry"],
        strength=tuple(config["strength"]),
        aggregate=config["aggregate"],
        smooth=tuple(config["smooth"]),
        presmoother=tuple(config["presmoother"]),
        postsmoother=tuple(config["postsmoother"]),
        improve_candidates=config["improve_candidates"],
        diagonal_dominance=config["diagonal_dominance"],
        max_levels=config["max_levels"],
        max_coarse=config["max_coarse"],
        coarse_solver=config["coarse_solver"],
        keep=config["keep"],
    )
    # PyAMG 5.3.0 aspreconditioner applies precisely one cycle with a zero guess.
    # The hierarchy and linear cycle are reused unchanged for every GMRES call.
    preconditioner = hierarchy.aspreconditioner(cycle=config["preconditioner_cycle"])
    solves = []

    def solve(rhs, rtol):
        if not np.all(np.isfinite(rhs)):
            raise ValueError("reference correction is not finite")
        iterations = 0

        def count(_value):
            nonlocal iterations
            iterations += 1

        answer, info = gmres(
            operator,
            rhs,
            x0=np.zeros(matrix.n),
            M=preconditioner,
            restart=config["restart"],
            maxiter=config["max_restart_cycles"],
            rtol=rtol,
            atol=config["atol"],
            callback=count,
            callback_type="pr_norm",
        )
        if info != 0 or not np.all(np.isfinite(answer)):
            raise ValueError("independent PyAMG-GMRES reference did not converge")
        record = {"info": int(info), "inner_iterations": iterations}
        if method == PYAMG_INEXACT_QUALIFICATION_METHOD:
            # Diagnostic recomputation on the original operator, not the
            # preconditioned callback norm or a convergence guarantee.
            residual = _extended_residual(matrix, rhs, answer)
            numerator = np.sqrt(np.sum(residual * residual, dtype=np.longdouble))
            extended_rhs = rhs.astype(np.longdouble)
            denominator = np.sqrt(
                np.sum(extended_rhs * extended_rhs, dtype=np.longdouble)
            )
            relative = (
                float(numerator / denominator)
                if denominator
                else (0.0 if numerator == 0 else math.inf)
            )
            if not math.isfinite(relative):
                raise ValueError("reference solve residual diagnostic is not finite")
            record.update(
                rtol=rtol,
                residual_method="extended-precision-residual-v1",
                relative_l2_residual=relative,
            )
        solves.append(record)
        return answer

    reference = solve(system.public.b, config["rtol"])
    for _ in range(config["refinement_steps"]):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = np.asarray(
                _extended_residual(matrix, system.public.b, reference), dtype=np.float64
            )
            reference = reference + solve(
                residual, config.get("refinement_rtol", config["rtol"])
            )
        if not np.all(np.isfinite(reference)):
            raise ValueError("refined independent reference is not finite")
    observed = accuracy_metrics(system, reference)
    metrics = {}
    for name, limit in QUALIFICATION_LIMITS.items():
        value = getattr(observed, name)
        if value is None or not math.isfinite(value) or value > limit:
            raise ValueError(
                f"independent reference lacks the tenfold margin for {name}"
            )
        metrics[name] = float(value)
    formation = _extended_residual(matrix, system.public.b, system.x_star)
    maximum = np.max(np.abs(formation), initial=np.longdouble(0))
    norm = np.max(
        np.abs(system.public.b.astype(np.longdouble)), initial=np.longdouble(0)
    )
    relative = float(maximum / norm) if norm else (0.0 if maximum == 0 else math.inf)
    if not math.isfinite(relative):
        raise ValueError("RHS formation diagnostic is not finite")
    evidence = {
        "method": method,
        "qualified": True,
        "system_sha256": system_digest(system),
        "metrics": metrics,
        "formation": {
            "method": "extended-precision-residual-v1",
            "relative_linf_residual": relative,
            "verified_forward_bound": None,
        },
        "uncertainty": "empirical-feasibility-only",
        "reference_solver": {
            "configuration": copy.deepcopy(config),
            "implementation": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pyamg": pyamg.__version__,
            },
            "solves": solves,
            "hierarchy": [
                {"n": int(level.A.shape[0]), "nnz": int(level.A.nnz)}
                for level in hierarchy.levels
            ],
        },
    }

    validate_qualification(evidence, system_digest(system))
    return evidence
