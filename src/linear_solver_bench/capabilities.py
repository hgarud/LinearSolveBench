"""Canonical audit identity of the candidate-visible native HYPRE surface."""

from __future__ import annotations

import hashlib

KRYLOV_PRIMITIVE_ABI = "hypre-native-krylov-solver-v4"

_STANDARD_SETUP_SOLVE = (
    "HYPRE_Int(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector)"
)
_SOLVER_INT_SETTER = "HYPRE_Int(HYPRE_Solver,HYPRE_Int)"
_SOLVER_REAL_SETTER = "HYPRE_Int(HYPRE_Solver,HYPRE_Real)"

# This is trusted audit and prompt metadata, not an operation table passed to a
# candidate. Candidate source calls every name below directly.
NATIVE_HYPRE_CAPABILITY_GROUPS = (
    (
        "krylov",
        (
            ("hypre_ParKrylovCAlloc", "void *(size_t,size_t,HYPRE_MemoryLocation)"),
            ("hypre_ParKrylovFree", "HYPRE_Int(void *)"),
            ("hypre_ParKrylovCommInfo", "HYPRE_Int(void *,HYPRE_Int *,HYPRE_Int *)"),
            ("hypre_ParKrylovCreateVector", "void *(void *)"),
            ("hypre_ParKrylovCreateVectorArray", "void *(HYPRE_Int,void *)"),
            ("hypre_ParKrylovDestroyVector", "HYPRE_Int(void *)"),
            ("hypre_ParKrylovCopyVector", "HYPRE_Int(void *,void *)"),
            ("hypre_ParKrylovClearVector", "HYPRE_Int(void *)"),
            ("hypre_ParKrylovScaleVector", "HYPRE_Int(HYPRE_Complex,void *)"),
            ("hypre_ParKrylovAxpy", "HYPRE_Int(HYPRE_Complex,void *,void *)"),
            ("hypre_ParKrylovInnerProd", "HYPRE_Real(void *,void *)"),
            (
                "hypre_ParKrylovInnerProdTagged",
                "HYPRE_Int(void *,void *,HYPRE_Int *,HYPRE_Complex **)",
            ),
            (
                "hypre_ParKrylovMassInnerProd",
                "HYPRE_Int(void *,void **,HYPRE_Int,HYPRE_Int,void *)",
            ),
            (
                "hypre_ParKrylovMassDotpTwo",
                "HYPRE_Int(void *,void *,void **,HYPRE_Int,HYPRE_Int,void *,void *)",
            ),
            (
                "hypre_ParKrylovMassAxpy",
                "HYPRE_Int(HYPRE_Complex *,void **,void *,HYPRE_Int,HYPRE_Int)",
            ),
            ("hypre_ParKrylovMatvecCreate", "void *(void *,void *)"),
            (
                "hypre_ParKrylovMatvec",
                "HYPRE_Int(void *,HYPRE_Complex,void *,void *,HYPRE_Complex,void *)",
            ),
            (
                "hypre_ParKrylovMatvecT",
                "HYPRE_Int(void *,HYPRE_Complex,void *,void *,HYPRE_Complex,void *)",
            ),
            ("hypre_ParKrylovMatvecDestroy", "HYPRE_Int(void *)"),
            ("hypre_ParKrylovIdentitySetup", "HYPRE_Int(void *,void *,void *,void *)"),
            ("hypre_ParKrylovIdentity", "HYPRE_Int(void *,void *,void *,void *)"),
        ),
    ),
    (
        "matrix",
        (
            (
                "HYPRE_ParCSRMatrixGetComm",
                "HYPRE_Int(HYPRE_ParCSRMatrix,MPI_Comm *)",
            ),
            (
                "HYPRE_ParCSRMatrixGetDims",
                "HYPRE_Int(HYPRE_ParCSRMatrix,HYPRE_BigInt *,HYPRE_BigInt *)",
            ),
            (
                "HYPRE_ParCSRMatrixGetLocalRange",
                (
                    "HYPRE_Int(HYPRE_ParCSRMatrix,HYPRE_BigInt *,HYPRE_BigInt *,"
                    "HYPRE_BigInt *,HYPRE_BigInt *)"
                ),
            ),
        ),
    ),
    (
        "diagonal_scaling",
        (
            ("HYPRE_ParCSRDiagScaleSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParCSRDiagScale", _STANDARD_SETUP_SOLVE),
        ),
    ),
    (
        "boomeramg",
        (
            ("HYPRE_BoomerAMGCreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_BoomerAMGDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_BoomerAMGSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_BoomerAMGSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_BoomerAMGSolveT", _STANDARD_SETUP_SOLVE),
            ("HYPRE_BoomerAMGSetTol", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetMaxIter", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetMinIter", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetMaxCoarseSize", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetMinCoarseSize", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetMaxLevels", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetStrongThreshold", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetMaxRowSum", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetNonGalerkinTol", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetCoarsenType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetMeasureType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetAggNumLevels", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetNumPaths", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetInterpType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetInterpRefine", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetTruncFactor", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetPMaxElmts", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetAggInterpType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetAggTruncFactor", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetAggPMaxElmts", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetCycleType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetNumSweeps", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetRelaxType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetRelaxOrder", _SOLVER_INT_SETTER),
            (
                "HYPRE_BoomerAMGSetCycleNumSweeps",
                "HYPRE_Int(HYPRE_Solver,HYPRE_Int,HYPRE_Int)",
            ),
            (
                "HYPRE_BoomerAMGSetCycleRelaxType",
                "HYPRE_Int(HYPRE_Solver,HYPRE_Int,HYPRE_Int)",
            ),
            ("HYPRE_BoomerAMGSetRelaxWt", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetOuterWt", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetChebyOrder", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetChebyScale", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetChebyVariant", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetChebyEigEst", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetChebyFraction", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGSetSmoothType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetSmoothNumLevels", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGSetSmoothNumSweeps", _SOLVER_INT_SETTER),
        ),
    ),
    (
        "boomeramg_dd",
        (
            ("HYPRE_BoomerAMGDDCreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_BoomerAMGDDDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_BoomerAMGDDSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_BoomerAMGDDSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_BoomerAMGDDSetFACNumRelax", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetFACNumCycles", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetFACCycleType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetFACRelaxType", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetFACRelaxWeight", _SOLVER_REAL_SETTER),
            ("HYPRE_BoomerAMGDDSetStartLevel", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetPadding", _SOLVER_INT_SETTER),
            ("HYPRE_BoomerAMGDDSetNumGhostLayers", _SOLVER_INT_SETTER),
        ),
    ),
    (
        "fsai",
        (
            ("HYPRE_FSAICreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_FSAIDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_FSAISetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_FSAISolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_FSAISetAlgoType", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetMaxSteps", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetMaxStepSize", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetMaxNnzRow", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetNumLevels", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetMaxIterations", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetEigMaxIters", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetZeroGuess", _SOLVER_INT_SETTER),
            ("HYPRE_FSAISetThreshold", _SOLVER_REAL_SETTER),
            ("HYPRE_FSAISetKapTolerance", _SOLVER_REAL_SETTER),
            ("HYPRE_FSAISetOmega", _SOLVER_REAL_SETTER),
            ("HYPRE_FSAISetTolerance", _SOLVER_REAL_SETTER),
        ),
    ),
    (
        "parcheby",
        (
            ("HYPRE_ParChebyCreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_ParChebyDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_ParChebySetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParChebySolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParChebySetMaxIterations", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetOrder", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetVariant", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetScale", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetEigEst", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetZeroGuess", _SOLVER_INT_SETTER),
            ("HYPRE_ParChebySetTolerance", _SOLVER_REAL_SETTER),
            ("HYPRE_ParChebySetEigRatio", _SOLVER_REAL_SETTER),
            (
                "HYPRE_ParChebySetMinMaxEigEst",
                "HYPRE_Int(HYPRE_Solver,HYPRE_Real,HYPRE_Real)",
            ),
        ),
    ),
    (
        "parasails",
        (
            ("HYPRE_ParaSailsCreate", "HYPRE_Int(MPI_Comm,HYPRE_Solver *)"),
            ("HYPRE_ParaSailsDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_ParaSailsSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParaSailsSolve", _STANDARD_SETUP_SOLVE),
            (
                "HYPRE_ParaSailsSetParams",
                "HYPRE_Int(HYPRE_Solver,HYPRE_Real,HYPRE_Int)",
            ),
            ("HYPRE_ParaSailsSetFilter", _SOLVER_REAL_SETTER),
            ("HYPRE_ParaSailsSetSym", _SOLVER_INT_SETTER),
            ("HYPRE_ParaSailsSetLoadbal", _SOLVER_REAL_SETTER),
            ("HYPRE_ParaSailsSetReuse", _SOLVER_INT_SETTER),
        ),
    ),
    (
        "euclid",
        (
            ("HYPRE_EuclidCreate", "HYPRE_Int(MPI_Comm,HYPRE_Solver *)"),
            ("HYPRE_EuclidDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_EuclidSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_EuclidSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_EuclidSetLevel", _SOLVER_INT_SETTER),
            ("HYPRE_EuclidSetBJ", _SOLVER_INT_SETTER),
            ("HYPRE_EuclidSetRowScale", _SOLVER_INT_SETTER),
            ("HYPRE_EuclidSetSparseA", _SOLVER_REAL_SETTER),
            ("HYPRE_EuclidSetILUT", _SOLVER_REAL_SETTER),
        ),
    ),
    (
        "pilut",
        (
            ("HYPRE_ParCSRPilutCreate", "HYPRE_Int(MPI_Comm,HYPRE_Solver *)"),
            ("HYPRE_ParCSRPilutDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_ParCSRPilutSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParCSRPilutSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ParCSRPilutSetMaxIter", _SOLVER_INT_SETTER),
            ("HYPRE_ParCSRPilutSetFactorRowSize", _SOLVER_INT_SETTER),
            ("HYPRE_ParCSRPilutSetDropTolerance", _SOLVER_REAL_SETTER),
        ),
    ),
    (
        "ilu",
        (
            ("HYPRE_ILUCreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_ILUDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_ILUSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ILUSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_ILUSetMaxIter", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetTriSolve", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetLowerJacobiIters", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetUpperJacobiIters", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetLevelOfFill", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetMaxNnzPerRow", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetSchurMaxIter", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetType", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetLocalReordering", _SOLVER_INT_SETTER),
            ("HYPRE_ILUSetTol", _SOLVER_REAL_SETTER),
            ("HYPRE_ILUSetDropThreshold", _SOLVER_REAL_SETTER),
            ("HYPRE_ILUSetNSHDropThreshold", _SOLVER_REAL_SETTER),
        ),
    ),
    (
        "schwarz",
        (
            ("HYPRE_SchwarzCreate", "HYPRE_Int(HYPRE_Solver *)"),
            ("HYPRE_SchwarzDestroy", "HYPRE_Int(HYPRE_Solver)"),
            ("HYPRE_SchwarzSetup", _STANDARD_SETUP_SOLVE),
            ("HYPRE_SchwarzSolve", _STANDARD_SETUP_SOLVE),
            ("HYPRE_SchwarzSetVariant", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetOverlap", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetDomainType", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetNonSymm", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetLocalSolverType", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetILUKLevelOfFill", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetILUTMaxNnzPerRow", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetMaxIter", _SOLVER_INT_SETTER),
            ("HYPRE_SchwarzSetRelaxWeight", _SOLVER_REAL_SETTER),
            ("HYPRE_SchwarzSetILUTDroptol", _SOLVER_REAL_SETTER),
            ("HYPRE_SchwarzSetTol", _SOLVER_REAL_SETTER),
        ),
    ),
)

KRYLOV_PRIMITIVE_SIGNATURES = tuple(
    item for _group, signatures in NATIVE_HYPRE_CAPABILITY_GROUPS for item in signatures
)
KRYLOV_PRIMITIVE_SYMBOLS = tuple(
    symbol for symbol, _signature in KRYLOV_PRIMITIVE_SIGNATURES
)
STOCK_PRECONDITIONER_FAMILIES = tuple(
    group
    for group, _signatures in NATIVE_HYPRE_CAPABILITY_GROUPS
    if group not in {"krylov", "matrix"}
)

if len(KRYLOV_PRIMITIVE_SYMBOLS) != len(set(KRYLOV_PRIMITIVE_SYMBOLS)):
    raise RuntimeError("native HYPRE capability catalog contains duplicate symbols")

KRYLOV_PRIMITIVE_SURFACE_CANONICAL = (
    KRYLOV_PRIMITIVE_ABI
    + "\n"
    + "\n".join(
        f"{group}/{symbol}:{signature}"
        for group, signatures in NATIVE_HYPRE_CAPABILITY_GROUPS
        for symbol, signature in signatures
    )
    + "\n"
).encode("ascii")
KRYLOV_PRIMITIVE_SURFACE_SHA256 = hashlib.sha256(
    KRYLOV_PRIMITIVE_SURFACE_CANONICAL
).hexdigest()

__all__ = [
    "KRYLOV_PRIMITIVE_ABI",
    "KRYLOV_PRIMITIVE_SIGNATURES",
    "KRYLOV_PRIMITIVE_SURFACE_CANONICAL",
    "KRYLOV_PRIMITIVE_SURFACE_SHA256",
    "KRYLOV_PRIMITIVE_SYMBOLS",
    "NATIVE_HYPRE_CAPABILITY_GROUPS",
    "STOCK_PRECONDITIONER_FAMILIES",
]
