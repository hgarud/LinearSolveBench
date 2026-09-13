"""Bounded, resumable downloads authenticated by frozen content hashes."""

from __future__ import annotations

import os
import pathlib
import urllib.error
import urllib.request
from collections.abc import Mapping
from urllib.parse import urlsplit

from ..dataset import sha256_file
from ..manifests import digest, https_url, positive_int

MAX_DOWNLOAD_BYTES = 8 * 1024**3


class _SourceRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, suitesparse: bool = False):
        super().__init__()
        self.suitesparse = suitesparse

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # SuiteSparse's public archive service uses HTTP redirects. Only that
        # known service may downgrade; the frozen SHA-256 authenticates its bytes.
        # Hugging Face and its signed storage redirects must remain HTTPS.
        redirect = urlsplit(newurl)
        allowed_http = (
            self.suitesparse
            and redirect.scheme == "http"
            and redirect.hostname
            in {
                "sparse.tamu.edu",
                "suitesparse-collection-website.herokuapp.com",
                "sparse-files.engr.tamu.edu",
            }
        )
        if (
            redirect.username
            or redirect.password
            or redirect.port not in {None, 80 if allowed_http else 443}
        ):
            raise ValueError("invalid data redirect authority")
        if redirect.scheme != "https" and not allowed_http:
            raise ValueError("data downloads must not redirect outside HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_archive(
    source: Mapping[str, object], cache: pathlib.Path, *, offline: bool = False
) -> pathlib.Path:
    """Return verified bytes; interrupted downloads retain a resumable .part file."""
    url = https_url(source["url"])
    expected_hash = digest(source["archive_sha256"], "archive_sha256")
    expected_size = positive_int(source["archive_bytes"], "archive_bytes")
    if expected_size > MAX_DOWNLOAD_BYTES:
        raise ValueError("archive exceeds the download limit")
    cache = pathlib.Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / expected_hash
    if target.is_file():
        if (
            target.stat().st_size == expected_size
            and sha256_file(target) == expected_hash
        ):
            return target
        if offline:
            raise ValueError("cached archive has invalid size or digest")
        target.unlink()
    if offline:
        raise ValueError(f"archive {expected_hash} is unavailable in the offline cache")
    partial = cache / f"{expected_hash}.part"
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    if offset == expected_size:
        if sha256_file(partial) == expected_hash:
            os.replace(partial, target)
            return target
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "linear-solver-bench/0.2", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    opener = urllib.request.build_opener(
        _SourceRedirect(suitesparse=source.get("kind") == "suitesparse")
    )
    try:
        with opener.open(request, timeout=120) as response:
            status = response.status
            if status == 206:
                expected_range = f"bytes {offset}-{expected_size - 1}/{expected_size}"
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("download response has an inconsistent byte range")
            elif status == 200:
                offset = 0  # Servers may ignore Range; restart the local file.
            else:
                raise ValueError("unexpected archive download status")
            declared = response.headers.get("Content-Length")
            if declared is not None and int(declared) != expected_size - offset:
                raise ValueError("download size disagrees with the release manifest")
            with partial.open("ab" if offset else "wb") as handle:
                total = offset
                while block := response.read(
                    min(1024 * 1024, expected_size - total + 1)
                ):
                    total += len(block)
                    if total > expected_size:
                        raise ValueError("download exceeds its declared size")
                    handle.write(block)
        if total != expected_size:
            raise ValueError("archive download is incomplete; partial data retained")
        if sha256_file(partial) != expected_hash:
            partial.unlink(missing_ok=True)
            raise ValueError("downloaded archive digest mismatch")
        os.replace(partial, target)
        return target
    except (OSError, urllib.error.URLError) as exc:
        raise ValueError(
            "archive download failed; partial data retained for resumption"
        ) from exc
