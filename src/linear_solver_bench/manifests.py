"""Strict, content-addressed public release manifests for the two pilot tracks.

A digest makes a manifest immutable, not authoritative. Official evaluation also
requires its digest in the packaged release registry. Explicit file paths permit
unpublished, clearly identified draft releases for preparation and local work.
"""

from __future__ import annotations

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
PUBLISHED_RELEASES: dict[tuple[str, str, str, str], str] = {}
HEX256 = re.compile(r"[0-9a-f]{64}\Z")
PUBLIC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,159}\Z")
QUALIFICATION_LIMITS = {
    "normwise_backward_error": 1e-11,
    "componentwise_backward_error": 1e-9,
    "relative_l2_forward_error": 1e-6,
    "relative_linf_forward_error": 1e-6,
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
    evidence = strict_object(
        value,
        {"method", "qualified", "system_sha256", "metrics", "formation", "uncertainty"},
        "NS qualification",
    )
    if (
        evidence["method"] != "independent-sparse-lu-v1"
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
    if formation["method"] != "extended-precision-residual-v1":
        raise ValueError("unknown RHS formation evidence")
    finite_number(
        formation["relative_linf_residual"], "formation residual", positive=False
    )
    if formation["verified_forward_bound"] is not None:
        raise ValueError(
            "empirical reference evidence cannot claim a verified forward bound"
        )
    if evidence["uncertainty"] != "empirical-feasibility-only":
        raise ValueError("qualification must state its empirical uncertainty")


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
