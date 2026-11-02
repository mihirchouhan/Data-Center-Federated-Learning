"""Verified resumable downloads and safe, selective M100 TAR extraction.

Only the Python standard library is required. Large archives remain outside the
project ZIP. A completed archive is published only after its MD5 has passed.
"""
from __future__ import annotations

import hashlib
import http.client
import logging
import ntpath
import os
import re
import shutil
import ssl
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from .io_utils import atomic_write, sync_directory

_CHUNK_SIZE = 1024 * 1024
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_TRANSIENT_HTTP = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """The remote response or downloaded archive could not be verified."""


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # Published archive integrity checksum, not security.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _result(path: Path, md5: str, cached: bool, attempts: int) -> dict:
    return {"status": "cached" if cached else "downloaded", "cached": cached,
            "downloaded": not cached, "path": str(path), "bytes": path.stat().st_size,
            "md5": md5, "attempts": attempts}


def download_file(url: str, destination: Path, expected_md5: str,
                  expected_size: int | None = None, logger=None,
                  retries: int = 4, timeout: float = 60,
                  ca_bundle: str | Path | None = None) -> dict:
    """Download an HTTP(S) archive, retaining a resumable ``.part`` on failure.

    ``retries`` is the number of additional attempts after the initial request.
    Existing completed files are reused only when their size (if supplied) and
    checksum pass. A bad existing destination survives until a verified download
    replaces it, and is never reported as usable. A server that ignores Range
    receives a clean restart rather than an append to the partial file.
    ``ca_bundle`` adds trusted PEM CA certificates to Python's default trust
    store; HTTPS certificate and hostname verification remain enabled.
    """
    if not isinstance(url, str) or not url.lower().startswith(("http://", "https://")):
        raise ValueError("Download URL must use HTTP or HTTPS")
    expected_md5 = expected_md5.lower()
    if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
        raise ValueError("expected_md5 must contain exactly 32 hexadecimal characters")
    if expected_size is not None and (isinstance(expected_size, bool)
                                     or not isinstance(expected_size, int)
                                     or expected_size < 0):
        raise ValueError("expected_size must be a nonnegative integer or None")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    log = logger or logging.getLogger(__name__)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.exists():
        if not destination.is_file():
            raise DownloadError(f"Destination is not a regular file: {destination}")
        size_ok = expected_size is None or destination.stat().st_size == expected_size
        if size_ok and _md5(destination) == expected_md5:
            log.info("Using checksum-verified cached archive: %s", destination)
            return _result(destination, expected_md5, True, 0)
        log.warning("Existing archive failed verification; downloading a replacement: %s",
                    destination)
    if partial.exists() and not partial.is_file():
        raise DownloadError(f"Partial download is not a regular file: {partial}")
    if partial.exists() and expected_size is not None:
        partial_size = partial.stat().st_size
        if partial_size > expected_size:
            partial.unlink()
        elif partial_size == expected_size:
            if _md5(partial) == expected_md5:
                os.replace(partial, destination)
                sync_directory(destination.parent)
                return _result(destination, expected_md5, False, 0)
            partial.unlink()

    context = None
    if url.lower().startswith("https://"):
        context = ssl.create_default_context()
        if ca_bundle is not None:
            ca_path = Path(os.path.expandvars(str(ca_bundle))).expanduser()
            try:
                context.load_verify_locations(cafile=str(ca_path))
            except (OSError, ssl.SSLError) as error:
                raise DownloadError(
                    f"Cannot load trusted CA bundle {ca_path}: {error}. "
                    "Supply a readable PEM CA file with --ca-bundle."
                ) from error
            log.info("Using additional trusted CA certificates: %s", ca_path)

    last_error = None
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity", "User-Agent": "M100-training-template/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            log.info("Archive request %d/%d, resume offset %d bytes: %s",
                     attempt + 1, retries + 1, offset, destination.name)
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                status = response.status
                content_length_text = response.headers.get("Content-Length")
                try:
                    content_length = (int(content_length_text)
                                      if content_length_text is not None else None)
                except ValueError as error:
                    raise DownloadError("Invalid HTTP Content-Length") from error
                if content_length is not None and content_length < 0:
                    raise DownloadError("Negative HTTP Content-Length")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise DownloadError("Encoded HTTP bodies cannot be resumed safely")
                if status == 206:
                    match = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if match is None:
                        raise DownloadError("206 response requires a valid Content-Range")
                    start, end, total = map(int, match.groups())
                    if start != offset or end < start or end != total - 1:
                        raise DownloadError("HTTP Content-Range does not match the requested remainder")
                    response_size = end - start + 1
                    if content_length is not None and content_length != response_size:
                        raise DownloadError("Content-Length disagrees with Content-Range")
                    if expected_size is not None and total != expected_size:
                        raise DownloadError("Content-Range total disagrees with expected archive size")
                    mode = "ab" if offset else "wb"
                elif status == 200:
                    if offset:
                        log.warning("Server ignored Range; restarting %s from byte zero", destination.name)
                    offset = 0
                    response_size = content_length
                    if (expected_size is not None and content_length is not None
                            and content_length != expected_size):
                        raise DownloadError("Content-Length disagrees with expected archive size")
                    mode = "wb"
                else:
                    raise DownloadError(f"Unexpected HTTP response status: {status}")
                received = 0
                last_progress_at = time.monotonic()
                with partial.open(mode) as handle:
                    try:
                        while True:
                            block = response.read(_CHUNK_SIZE)
                            if not block:
                                break
                            handle.write(block)
                            received += len(block)
                            if time.monotonic() - last_progress_at >= 30:
                                log.info("Downloading %s: %d bytes saved so far", destination.name, offset + received)
                                handle.flush()
                                os.fsync(handle.fileno())
                                last_progress_at = time.monotonic()
                    finally:
                        handle.flush()
                        os.fsync(handle.fileno())
                if response_size is not None and received != response_size:
                    raise DownloadError(f"Incomplete HTTP body: expected {response_size}, received {received}")
            actual_size = partial.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise DownloadError(f"Archive size mismatch: expected {expected_size}, received {actual_size}")
            actual_md5 = _md5(partial)
            if actual_md5 != expected_md5:
                # A complete bad body cannot be repaired by resuming from its end.
                partial.unlink()
                raise DownloadError(f"Archive MD5 mismatch: expected {expected_md5}, received {actual_md5}")
            os.replace(partial, destination)
            sync_directory(destination.parent)
            log.info("Archive verified: %s (%d bytes)", destination.name, actual_size)
            return _result(destination, actual_md5, False, attempt + 1)
        except urllib.error.HTTPError as error:
            last_error = error
            error.close()
            if error.code == 416 and partial.exists():
                # The partial file may already be complete when its size was not known.
                size_ok = expected_size is None or partial.stat().st_size == expected_size
                if size_ok and _md5(partial) == expected_md5:
                    os.replace(partial, destination)
                    sync_directory(destination.parent)
                    return _result(destination, expected_md5, False, attempt + 1)
                partial.unlink()
            elif error.code not in _TRANSIENT_HTTP:
                raise DownloadError(f"HTTP {error.code} while downloading {url}") from error
        except (urllib.error.URLError, ssl.SSLError, http.client.HTTPException, TimeoutError,
                ConnectionError, DownloadError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            if isinstance(reason, ssl.SSLCertVerificationError):
                saved = partial.stat().st_size if partial.exists() else 0
                raise DownloadError(
                    f"TLS certificate verification failed: {reason}. "
                    f"Saved partial download: {partial} ({saved} bytes). "
                    "Use --ca-bundle /path/to/trusted-ca.pem or set SSL_CERT_FILE "
                    "to a trusted CA bundle supplied by your administrator, "
                    "then rerun the same command to resume."
                ) from error
            last_error = error
        if attempt < retries:
            delay = min(15, 2 ** attempt)
            log.warning("Download attempt failed (%s); retrying in %d seconds", last_error, delay)
            time.sleep(delay)
    raise DownloadError(f"Download failed after {retries + 1} attempts: {last_error}") from last_error


def _safe_member_path(name: str) -> PurePosixPath:
    # Interpret backslashes as separators on every OS, including Linux servers.
    normalized = name.replace("\\", "/")
    if (not normalized or "\x00" in normalized or normalized.startswith("/")
            or ntpath.splitdrive(normalized)[0]):
        raise ValueError(f"Unsafe absolute archive member: {name!r}")
    path = PurePosixPath(normalized)
    if any(part == ".." or ":" in part for part in path.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    return path


def _plugin_alias(plugin: str) -> str:
    value = plugin.lower()
    return value[:-4] if value.endswith("_pub") else value


def extract_selected(archive: Path, destination: Path,
                     metrics: set[tuple[str, str]], logger=None) -> dict:
    """Extract only requested plugin/metric Parquet files from an M100 TAR.

    Hive paths are retained from ``year_month=...`` onward; wrapper directories
    are removed. All links, special files and traversal names are rejected, even
    when they would not be selected. Files are always rewritten atomically;
    matching size alone is not evidence of a completed earlier extraction.
    The caller can persist this result in its own completed-stage manifest.
    """
    log = logger or logging.getLogger(__name__)
    archive, destination = Path(archive), Path(destination)
    wanted = {(_plugin_alias(plugin), metric) for plugin, metric in metrics}
    if not wanted:
        raise ValueError("At least one plugin/metric pair must be selected")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    files = total_bytes = 0
    paths = []
    found = set()
    with tarfile.open(archive, mode="r|*") as bundle:
        for member in bundle:
            member_path = _safe_member_path(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Archive links are not allowed: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Archive special files are not allowed: {member.name!r}")
            if not member.isfile() or member_path.suffix.lower() != ".parquet":
                continue
            parts = member_path.parts
            starts = [i for i, part in enumerate(parts) if part.startswith("year_month=")]
            if len(starts) != 1:
                continue
            relative = PurePosixPath(*parts[starts[0]:])
            plugin_parts = [part[7:] for part in relative.parts if part.startswith("plugin=")]
            metric_parts = [part[7:] for part in relative.parts if part.startswith("metric=")]
            if len(plugin_parts) != 1 or len(metric_parts) != 1:
                continue
            pair = (_plugin_alias(plugin_parts[0]), metric_parts[0])
            if pair not in wanted:
                continue
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Extraction would escape destination: {member.name!r}")
            source = bundle.extractfile(member)
            if source is None:
                raise tarfile.ExtractError(f"Cannot read selected member: {member.name}")
            with source:
                def copy(handle):
                    shutil.copyfileobj(source, handle, _CHUNK_SIZE)
                    if handle.tell() != member.size:
                        raise tarfile.ReadError(f"Truncated member: {member.name}")
                atomic_write(target, copy)
            files += 1
            total_bytes += member.size
            paths.append(relative.as_posix())
            found.add(pair)
    log.info("Selected extraction complete: %d files, %d bytes from %s",
             files, total_bytes, archive.name)
    return {"archive": str(archive), "destination": str(destination), "files": files,
            "bytes": total_bytes, "paths": paths,
            "metrics": [list(pair) for pair in sorted(found)],
            "missing_metrics": [list(pair) for pair in sorted(wanted - found)]}
