"""Strict, content-addressed public release manifests for the two pilot tracks.

A digest makes a manifest immutable, not authoritative. Official evaluation also
requires its digest in the packaged release registry. Explicit file paths permit
unpublished, clearly identified draft releases for preparation and local work.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

from .dataset import identity_sha256
from .families import resolve_track

RELEASE_KIND = "linear-solver-bench-pilot-release"
RHS_SCHEME = "rademacher-hmac-sha256-v2"
# A release is added here only after its data and qualification are published.
PUBLISHED_RELEASES: dict[tuple[str, str, str, str], str] = {
    (
        "ns-mesh-cohort-v2",
        "ns-mesh-pde",
        "coverage",
        "dev",
    ): "9927ea72bac431d871b0547b966f9ce78b550ca0d1285c7060cdcb2e3e688ecc",
    (
        "ns-mesh-cohort-v2",
        "ns-mesh-pde",
        "coverage",
        "ranked",
    ): "a86a2d42765364bcadd48fdef22dc2f0900d04e98fe4fe49fe551b84d591ad52",
    (
        "ns-mesh-dev-pilot",
        "ns-mesh-pde",
        "coverage",
        "dev",
    ): "93d48d6ef9298752f634e1411d8b21f7e8912e6193b15ece838fe80918c71b5e",
    (
        "ns-mesh-pilot",
        "ns-mesh-pde",
        "coverage",
        "dev",
    ): "25e79afb724d2f33c2438513f3ea13aeaebfced24fb8c4bba6d5c78e5edfa296",
    (
        "ns-mesh-pilot",
        "ns-mesh-pde",
        "coverage",
        "ranked",
    ): "61fb1ec691400d0e1bc83d8524f5e90f1e6479941e31e4f8e0ccd840163da241",
    (
        "flash-replay-pilot",
        "magnetic_diffusion_flash",
        "replay",
        "dev",
    ): "02a8e24e667b96029f0c8db627f87c237549778a63f3342a73010e7278876669",
    (
        "flash-replay-pilot",
        "magnetic_diffusion_flash",
        "replay",
        "ranked",
    ): "1ce0ec20c66123773215543e2d43375104f43113a9416c59fb0e9625a9975b79",
}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\Z")
QUALIFICATION_LIMITS = {
    "normwise_backward_error": 1e-11,
    "componentwise_backward_error": 1e-9,
    "relative_l2_forward_error": 1e-6,
    "relative_linf_forward_error": 1e-6,
}
ITERATIVE_QUALIFICATION_METHOD = "independent-equilibrated-gmres-ilu-refined-v1"
# Offline reference configuration, not a candidate stopping rule or timing baseline.
ITERATIVE_QUALIFICATION_CONFIG = {
    "equilibration": "max-absolute-row-then-column-v1",
    "initial_guess": "zero",
    "preconditioner": "scipy-superlu-ilu",
    "drop_tol": 1e-4,
    "fill_factor": 10.0,
    "drop_rule": "basic,area",
    "permc_spec": "COLAMD",
    "diag_pivot_thresh": 1.0,
    "superlu_equilibration": False,
    "krylov": "scipy-gmres-left-preconditioned",
    "restart": 50,
    "max_restart_cycles": 250,
    "rtol": 1e-13,
    "atol": 0.0,
    "refinement_steps": 2,
    "refinement_residual": "extended-precision-residual-v1",
}


PYAMG_QUALIFICATION_METHOD = "independent-pyamg-sa-gmres-refined-v1"
PYAMG_QUALIFICATION_CONFIG = {
    "pyamg_version": "5.3.0",
    "operator": "original-float64-csr",
    "symmetry": "nonsymmetric",
    "right_near_nullspace": "ones",
    "left_near_nullspace": "ones",
    "strength": ["symmetric", {"theta": 0.0}],
    "aggregate": "standard",
    "smooth": ["jacobi", {"omega": 4.0 / 3.0, "degree": 1, "weighting": "local"}],
    "presmoother": ["gauss_seidel", {"sweep": "symmetric", "iterations": 2}],
    "postsmoother": ["gauss_seidel", {"sweep": "symmetric", "iterations": 2}],
    "improve_candidates": None,
    "diagonal_dominance": False,
    "max_levels": 25,
    "max_coarse": 500,
    "coarse_solver": "splu",
    "keep": False,
    "preconditioner_cycle": "V",
    "initial_guess": "zero",
    "restart": 50,
    "max_restart_cycles": 250,
    "rtol": 1e-13,
    "atol": 0.0,
    "refinement_steps": 2,
    "refinement_residual": "extended-precision-residual-v1",
}


# Keep v1 evidence reproducible; v2 changes only the two inner correction stops.
PYAMG_INEXACT_QUALIFICATION_METHOD = "independent-pyamg-sa-gmres-refined-v2"
PYAMG_INEXACT_QUALIFICATION_CONFIG = {
    **copy.deepcopy(PYAMG_QUALIFICATION_CONFIG),
    "refinement_rtol": 1e-2,
}


def strict_object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} fields do not match the public schema")
    return value


def digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def finite_number(value: object, label: str, *, positive: bool = True) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} has an invalid sign")
    return float(value)


def public_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not PUBLIC_ID.fullmatch(value):
        raise ValueError(f"{label} must be a short public identifier")
    return value


def https_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("public URL must be a string")
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.port not in {None, 443}
        or parts.query
        or parts.fragment
    ):
        raise ValueError("public source URLs must be plain HTTPS URLs")
    return value


def _source(value: object, family: str) -> None:
    common = {"kind", "url", "archive_sha256", "archive_bytes", "matrix_sha256"}
    expected = common | (
        {"matrix_id", "group", "name"}
        if family == "ns-mesh-pde"
        else {"b_sha256", "x0_sha256"}
    )
    source = strict_object(value, expected, "source")
    url = urlsplit(https_url(source["url"]))
    digest(source["archive_sha256"], "archive_sha256")
    digest(source["matrix_sha256"], "matrix_sha256")
    if family == "magnetic_diffusion_flash":
        digest(source["b_sha256"], "b_sha256")
        digest(source["x0_sha256"], "x0_sha256")
    size = positive_int(source["archive_bytes"], "archive_bytes")
    if size > 8 * 1024**3:
        raise ValueError("source archive exceeds the download limit")
    if family == "ns-mesh-pde":
        if source["kind"] != "suitesparse":
            raise ValueError("NS mesh requires a SuiteSparse source")
        positive_int(source["matrix_id"], "matrix_id")
        for key in ("group", "name"):
            public_id(source[key], key)
        if url.hostname not in {
            "sparse.tamu.edu",
            "suitesparse-collection-website.herokuapp.com",
        }:
            raise ValueError("SuiteSparse source host is not supported")
        expected_path = f"/MM/{source['group']}/{source['name']}.tar.gz"
        if url.path != expected_path:
            raise ValueError(
                "SuiteSparse URL must identify its original matrix archive"
            )
    elif (
        source["kind"] != "huggingface"
        or url.hostname != "huggingface.co"
        or not re.fullmatch(
            r"/datasets/[^/]+/[^/]+/resolve/[0-9a-f]{40}/[^?#]+\.(?:zip|tar\.gz)",
            url.path,
        )
    ):
        raise ValueError("FLASH requires a commit-pinned Hugging Face case archive")


def validate_qualification(value: object, system_sha256: str | None = None) -> None:
    dominance = (
        isinstance(value, dict)
        and value.get("method") == "strict-row-diagonal-dominance-v1"
    )
    iterative = isinstance(value, dict) and value.get("method") in {
        ITERATIVE_QUALIFICATION_METHOD,
        PYAMG_QUALIFICATION_METHOD,
        PYAMG_INEXACT_QUALIFICATION_METHOD,
    }
    keys = {
        "method",
        "qualified",
        "system_sha256",
        "metrics",
        "formation",
        "uncertainty",
    }
    if dominance:
        keys.add("certificate")
    if iterative:
        keys.add("reference_solver")
    evidence = strict_object(
        value,
        keys,
        "NS qualification",
    )
    if (
        evidence["method"]
        not in {
            "independent-sparse-lu-v1",
            "independent-sparse-lu-refined-v1",
            "strict-row-diagonal-dominance-v1",
            ITERATIVE_QUALIFICATION_METHOD,
            PYAMG_QUALIFICATION_METHOD,
            PYAMG_INEXACT_QUALIFICATION_METHOD,
        }
        or evidence["qualified"] is not True
    ):
        raise ValueError(
            "NS qualification requires passing independent reference evidence"
        )
    digest(evidence["system_sha256"], "qualified system digest")
    if system_sha256 is not None and evidence["system_sha256"] != system_sha256:
        raise ValueError("qualification is bound to different numerical inputs")
    metrics = strict_object(
        evidence["metrics"], set(QUALIFICATION_LIMITS), "qualification metrics"
    )
    for name, limit in QUALIFICATION_LIMITS.items():
        if finite_number(metrics[name], name, positive=False) > limit:
            raise ValueError(f"NS qualification lacks the tenfold margin for {name}")
    formation = strict_object(
        evidence["formation"],
        {"method", "relative_linf_residual", "verified_forward_bound"},
        "RHS formation",
    )
    expected_method = (
        "exact-binary64-residual-v1" if dominance else "extended-precision-residual-v1"
    )
    if formation["method"] != expected_method:
        raise ValueError("unknown RHS formation evidence")
    finite_number(
        formation["relative_linf_residual"], "formation residual", positive=False
    )
    if dominance:
        _validate_dominance_certificate(evidence)
    elif formation["verified_forward_bound"] is not None:
        raise ValueError(
            "empirical reference evidence cannot claim a verified forward bound"
        )
    if not dominance and evidence["uncertainty"] != "empirical-feasibility-only":
        raise ValueError("qualification must state its empirical uncertainty")
    if iterative:
        _validate_iterative_reference(evidence["reference_solver"], evidence["method"])


def _same_typed_value(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _same_typed_value(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _same_typed_value(actual, wanted)
            for actual, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def _validate_iterative_reference(value: object, method: str) -> None:
    inexact = method == PYAMG_INEXACT_QUALIFICATION_METHOD
    multigrid = method in {
        PYAMG_QUALIFICATION_METHOD,
        PYAMG_INEXACT_QUALIFICATION_METHOD,
    }
    expected = {
        PYAMG_QUALIFICATION_METHOD: PYAMG_QUALIFICATION_CONFIG,
        PYAMG_INEXACT_QUALIFICATION_METHOD: PYAMG_INEXACT_QUALIFICATION_CONFIG,
        ITERATIVE_QUALIFICATION_METHOD: ITERATIVE_QUALIFICATION_CONFIG,
    }[method]
    keys = {"configuration", "implementation", "solves"}
    if multigrid:
        keys.add("hierarchy")
    reference = strict_object(value, keys, "iterative reference")
    if not _same_typed_value(reference["configuration"], expected):
        raise ValueError("iterative reference configuration does not match its version")
    version_keys = {"numpy", "scipy"} | ({"pyamg"} if multigrid else set())
    versions = strict_object(
        reference["implementation"], version_keys, "reference implementation"
    )
    for version in versions.values():
        if not isinstance(version, str) or not re.fullmatch(
            r"[0-9][A-Za-z0-9.+_-]{0,79}", version
        ):
            raise ValueError("reference library version must be a short version string")
    if multigrid and versions["pyamg"] != expected["pyamg_version"]:
        raise ValueError("PyAMG implementation does not match its frozen version")
    solves = reference["solves"]
    if not isinstance(solves, list) or len(solves) != 3:
        raise ValueError("iterative reference requires one solve and two corrections")
    limit = expected["restart"] * expected["max_restart_cycles"]
    for index, solve in enumerate(solves):
        solve_keys = {"info", "inner_iterations"}
        if inexact:
            solve_keys |= {"rtol", "residual_method", "relative_l2_residual"}
        strict_object(solve, solve_keys, "reference solve")
        if inexact:
            rtol = expected["rtol"] if index == 0 else expected["refinement_rtol"]
            if not _same_typed_value(solve["rtol"], rtol):
                raise ValueError("reference solve tolerance differs from its version")
            if solve["residual_method"] != "extended-precision-residual-v1":
                raise ValueError("unknown reference solve residual diagnostic")
            finite_number(
                solve["relative_l2_residual"], "solve residual", positive=False
            )
        if type(solve["info"]) is not int or solve["info"] != 0:
            raise ValueError("each iterative reference solve must converge")
        iterations = solve["inner_iterations"]
        if type(iterations) is not int or not 0 <= iterations <= limit:
            raise ValueError("reference iteration count exceeds the fixed budget")
    if multigrid:
        levels = reference["hierarchy"]
        if (
            not isinstance(levels, list)
            or not 1 <= len(levels) <= expected["max_levels"]
        ):
            raise ValueError("invalid multigrid hierarchy length")
        previous = None
        for level in levels:
            strict_object(level, {"n", "nnz"}, "multigrid level")
            size = positive_int(level["n"], "level n")
            nnz = positive_int(level["nnz"], "level nnz")
            if nnz > size * size or (previous is not None and size >= previous):
                raise ValueError(
                    "multigrid hierarchy must have valid, decreasing dimensions"
                )
            previous = size
        if len(levels) < expected["max_levels"] and previous > expected["max_coarse"]:
            raise ValueError("multigrid hierarchy stopped above its coarse-size limit")


def _validate_dominance_certificate(evidence: dict) -> None:
    from fractions import Fraction

    certificate = strict_object(
        evidence["certificate"],
        {
            "arithmetic",
            "witness",
            "metric_evaluation",
            "minimum_row_margin",
            "maximum_formation_error",
        },
        "row dominance certificate",
    )
    if (
        certificate["arithmetic"] != "exact-binary64-integer-v1"
        or certificate["witness"] != "manufactured-target"
        or certificate["metric_evaluation"] != "binary64"
        or evidence["uncertainty"] != "verified-stored-system-forward-bound"
    ):
        raise ValueError("unknown dominance certificate or witness semantics")
    margin = finite_number(certificate["minimum_row_margin"], "row dominance margin")
    error = finite_number(
        certificate["maximum_formation_error"], "formation error bound", positive=False
    )
    forward = finite_number(
        evidence["formation"]["verified_forward_bound"],
        "verified forward bound",
        positive=False,
    )
    if Fraction(forward) * Fraction(margin) < Fraction(error):
        raise ValueError("verified forward bound understates the dominance certificate")
    if forward > min(
        QUALIFICATION_LIMITS["relative_l2_forward_error"],
        QUALIFICATION_LIMITS["relative_linf_forward_error"],
    ):
        raise ValueError("verified forward bound lacks the tenfold margin")
    for name in ("relative_l2_forward_error", "relative_linf_forward_error"):
        if evidence["metrics"][name] != 0.0:
            raise ValueError(
                "manufactured-target witness must have zero measured forward error"
            )


def validate_release(value: object, *, official: bool = False) -> dict:
    release = strict_object(
        value,
        {
            "schema_version",
            "kind",
            "release_id",
            "family",
            "track",
            "split",
            "contracts",
            "execution",
            "rhs",
            "cases",
            "manifest_sha256",
        },
        "pilot release",
    )
    if release["schema_version"] != 2 or release["kind"] != RELEASE_KIND:
        raise ValueError("unsupported pilot release schema")
    public_id(release["release_id"], "release_id")
    track = resolve_track(release["family"], release["track"])
    if release["family"] != track.family:
        raise ValueError("release manifests must use the canonical family ID")
    if release["split"] not in ("dev", "ranked"):
        raise ValueError("split must be dev or ranked")
    expected_contracts = {
        "accuracy": track.accuracy_contract_id,
        "execution": track.execution_contract_id,
        "scoring": track.scoring_contract_id,
    }
    if release["contracts"] != expected_contracts:
        raise ValueError("release contracts do not match its family and track")
    execution = strict_object(
        release["execution"],
        {
            "repetitions",
            "max_iterations",
            "case_timeout_seconds",
            "memory_bytes",
            "cpus",
        },
        "execution",
    )
    if type(execution["repetitions"]) is not int or execution["repetitions"] != 1:
        raise ValueError("pilot execution requires exactly one process per case")
    for name in ("max_iterations", "memory_bytes", "cpus"):
        positive_int(execution[name], name)
    if execution["max_iterations"] > 2**31 - 1:
        raise ValueError("iteration cap exceeds the native integer range")
    finite_number(execution["case_timeout_seconds"], "case_timeout_seconds")
    if track.family == "ns-mesh-pde":
        rhs = strict_object(
            release["rhs"],
            {"scheme", "key_id", "public_key_hex", "draw_index"},
            "RHS recipe",
        )
        if (
            rhs["scheme"] != RHS_SCHEME
            or type(rhs["draw_index"]) is not int
            or rhs["draw_index"] < 0
        ):
            raise ValueError("unsupported manufactured RHS recipe")
        digest(rhs["key_id"], "RHS key_id")
        if rhs["public_key_hex"] is not None:
            try:
                key = bytes.fromhex(rhs["public_key_hex"])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid public RHS key") from exc
            if len(key) < 16 or hashlib.sha256(key).hexdigest() != rhs["key_id"]:
                raise ValueError("public RHS key does not match its commitment")
        if release["split"] == "dev" and rhs["public_key_hex"] is None:
            raise ValueError("development RHS keys must be public")
        if release["split"] == "ranked" and rhs["public_key_hex"] is not None:
            raise ValueError("ranked RHS keys must remain operator-held")
    elif release["rhs"] is not None:
        raise ValueError(
            "FLASH replay preserves captured RHS and has no manufactured recipe"
        )
    cases = release["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("a pilot release must contain cases")
    observed = set()
    run_groups = {}
    for index, case in enumerate(cases):
        case = strict_object(
            case,
            {
                "index",
                "case_id",
                "n",
                "nnz",
                "tolerance",
                "role",
                "provenance_group",
                "source_run",
                "weight",
                "source",
                "admission",
                "qualification",
            },
            "release case",
        )
        if type(case["index"]) is not int or case["index"] != index:
            raise ValueError("release case order mismatch")
        for key in ("case_id", "provenance_group", "source_run"):
            public_id(case[key], key)
        if case["case_id"] in observed:
            raise ValueError("duplicate release case ID")
        observed.add(case["case_id"])
        group = run_groups.setdefault(case["source_run"], case["provenance_group"])
        if group != case["provenance_group"]:
            raise ValueError("a source run cannot belong to multiple provenance groups")
        positive_int(case["n"], "n")
        positive_int(case["nnz"], "nnz")
        if case["n"] > 2**31 - 1 or case["nnz"] > 2**31 - 1:
            raise ValueError("case dimensions exceed the sequential native runtime")
        if finite_number(case["tolerance"], "tolerance") >= 1:
            raise ValueError("tolerance must be less than one")
        if case["role"] not in ("scored", "control"):
            raise ValueError("case role must be scored or control")
        weight = finite_number(case["weight"], "weight", positive=False)
        if (case["role"] == "control") != (weight == 0):
            raise ValueError("only correctness controls have zero weight")
        if track.track == "coverage" and (case["role"] != "scored" or weight != 1):
            raise ValueError("coverage counts every case once with unit weight")
        _source(case["source"], track.family)
        admission_keys = {"statement", "citations"}
        if track.family == "magnetic_diffusion_flash":
            admission_keys |= {
                "capture_kind",
                "relative_tolerance",
                "absolute_tolerance",
                "step",
                "solve_index",
            }
        admission = strict_object(
            case["admission"], admission_keys, "scientific admission"
        )
        if (
            not isinstance(admission["statement"], str)
            or not admission["statement"].strip()
        ):
            raise ValueError("scientific admission must explain source provenance")
        citations = admission["citations"]
        if not isinstance(citations, list) or not citations:
            raise ValueError("scientific admission needs public source citations")
        for citation in citations:
            https_url(citation)
        if track.family == "magnetic_diffusion_flash":
            finite_number(
                admission["absolute_tolerance"], "absolute tolerance", positive=False
            )
            finite_number(admission["relative_tolerance"], "relative tolerance")
            if (
                admission["capture_kind"] != "scalar-magnetic-diffusion"
                or admission["absolute_tolerance"] != 0
                or admission["relative_tolerance"] != case["tolerance"]
            ):
                raise ValueError(
                    "capture tolerance or scientific family is incompatible with replay"
                )
            for name in ("step", "solve_index"):
                if type(admission[name]) is not int or admission[name] < 0:
                    raise ValueError("capture ordering must use nonnegative integers")
            if case["qualification"] is not None:
                raise ValueError(
                    "FLASH reference qualification is a separate replay artifact"
                )
        elif case["qualification"] is not None:
            validate_qualification(case["qualification"])
        elif official:
            raise ValueError("published NS cases need numerical qualification evidence")
    if track.track == "replay":
        scored = [case for case in cases if case["role"] == "scored"]
        groups = {case["provenance_group"] for case in scored}
        if not groups:
            raise ValueError("replay requires at least one scored case")
        for case in scored:
            group = case["provenance_group"]
            runs = {
                item["source_run"]
                for item in scored
                if item["provenance_group"] == group
            }
            count = sum(
                item["provenance_group"] == group
                and item["source_run"] == case["source_run"]
                for item in scored
            )
            expected = 1 / (len(groups) * len(runs) * count)
            if not math.isclose(case["weight"], expected, rel_tol=1e-14, abs_tol=0):
                raise ValueError(
                    "replay weights must give equal mass to groups, runs, and cases"
                )
    claimed = digest(release["manifest_sha256"], "manifest_sha256")
    body = {key: item for key, item in release.items() if key != "manifest_sha256"}
    if claimed != identity_sha256(body):
        raise ValueError("release manifest digest mismatch")
    identity = tuple(release[key] for key in ("release_id", "family", "track", "split"))
    if official and PUBLISHED_RELEASES.get(identity) != claimed:
        raise ValueError(
            "release is not a trusted published pilot release; "
            "use draft/local execution"
        )
    return release


def load_release(path: pathlib.Path, *, official: bool = False) -> dict:
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read pilot release manifest") from exc
    return validate_release(value, official=official)


def seal_manifest(body: Mapping[str, object]) -> dict:
    """Add a deterministic content identity; this does not publish a release."""
    if "manifest_sha256" in body:
        raise ValueError("manifest body already contains its digest")
    return {**body, "manifest_sha256": identity_sha256(body)}


def validate_release_splits(releases: list[dict]) -> None:
    """Check publication exposure boundaries across a release's supplied splits.

    Scientific near-duplicate relationships must use a shared provenance group;
    exact operators and source runs are checked independently of that declaration.
    """
    splits = set()
    identities = set()
    exposures = {}
    for value in releases:
        release = validate_release(value)
        identities.add(tuple(release[key] for key in ("release_id", "family", "track")))
        split = release["split"]
        if split in splits:
            raise ValueError("release split was supplied more than once")
        splits.add(split)
        for case in release["cases"]:
            for kind, identity in (
                ("case", case["case_id"]),
                ("provenance", case["provenance_group"]),
                ("run", case["source_run"]),
                ("operator", case["source"]["matrix_sha256"]),
            ):
                previous = exposures.setdefault((kind, identity), split)
                if previous != split:
                    raise ValueError(
                        f"{kind} exposure crosses development and ranked splits"
                    )
    if len(identities) != 1:
        raise ValueError(
            "split checks require exactly one release/family/track identity"
        )
