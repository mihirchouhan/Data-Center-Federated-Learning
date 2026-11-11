#!/usr/bin/env python3
"""Version 2 March acquisition: strong ETag preferred, guarded Last-Modified fallback.

No old downloader, pipeline, archive reader or model module is imported.
Production transfer requires --authorize-transfer AND --quota-confirmed, live
official exact metadata, and isolated fresh or receipt-owned destinations.
"""
from __future__ import annotations
import argparse
import contextlib
from email.utils import parsedate_to_datetime, format_datetime
from dataclasses import dataclass
import fcntl
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import re
import ssl
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path("/media/csedept/cse2014/Project/m100_training_template")
CONFIG = ROOT / "configs/acquisition/march_2022_v1.json"
RECORD = "7590061"
NAME = "22-03.tar"
MD5 = "4c74ad95b6d837d409e24216389bb25e"
METADATA_URL = "https://zenodo.org/api/records/" + RECORD
DATA_REL = "data/march_2022_acquisition/v1"
OUTPUT_REL = "output/march_2022_acquisition/v1"
CHUNK = 8 * 1024 * 1024

class SafetyError(RuntimeError):
    """Stop; retained evidence must not be deleted or overwritten."""
class IncompleteBody(SafetyError):
    """Only a verified prefix may be resumed."""
class TransientStatus(SafetyError):
    pass

def digest(data):
    return hashlib.sha256(data).hexdigest()

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def safe_path(path):
    path = Path(path).absolute()
    for p in reversed((path, *path.parents)):
        if p.is_symlink():
            raise SafetyError("Symlink path: " + str(p))
    return path

def regular(path, links=1):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != links:
        raise SafetyError("Unowned/nonregular/hardlinked file: " + str(path))
    return s

def read_file(path):
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        regular(path)
        with os.fdopen(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)

def hashes(path):
    sha, md = hashlib.sha256(), hashlib.md5()
    safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, "rb", closefd=False) as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                sha.update(b); md.update(b)
        s = os.fstat(fd)
        return {"sha256": sha.hexdigest(), "md5": md.hexdigest(), "bytes": s.st_size}
    finally:
        os.close(fd)

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: os.fsync(fd)
    finally: os.close(fd)

def create_bytes(path, data):
    safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    sync_dir(Path(path).parent)

def create_json(path, obj):
    """Exclusive, durable record. An interrupted malformed record blocks resume."""
    safe_path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(canonical(obj) + b"\n"); f.flush(); os.fsync(f.fileno())
    sync_dir(Path(path).parent)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def tls_context(ca_bundle=None):
    c = ssl.create_default_context()
    if ca_bundle:
        safe_path(ca_bundle)
        c.load_verify_locations(cafile=str(ca_bundle))
    if c.verify_mode != ssl.CERT_REQUIRED or not c.check_hostname:
        raise SafetyError("Certificate and hostname validation required")
    return c

def opener(ca_bundle=None):
    # Empty ProxyHandler avoids environment proxy redirects/identity changes.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
        urllib.request.HTTPSHandler(context=tls_context(ca_bundle)))

def validate_url(url, synthetic=False):
    if not isinstance(url, str):
        raise SafetyError("Resolved URL required")
    u = urllib.parse.urlsplit(url)
    if u.username or u.password or u.fragment or not u.path:
        raise SafetyError("Unexpected URL identity")
    if synthetic:
        try: loop = ipaddress.ip_address(u.hostname).is_loopback
        except ValueError: loop = False
        if not loop or u.scheme not in ("http", "https"):
            raise SafetyError("Synthetic transport is literal loopback only")
    elif u.scheme != "https" or u.hostname != "zenodo.org" or u.port not in (None, 443):
        raise SafetyError("Only resolved HTTPS zenodo.org file URLs accepted")
    return url

def parse_metadata(raw):
    """Pure official-record parser; never guesses a file URL or exact size."""
    try: d = json.loads(raw)
    except (ValueError, TypeError) as e: raise SafetyError("Invalid official JSON") from e
    if str(d.get("id")) != RECORD or str(d.get("metadata", {}).get("version")) != "1.0.0":
        raise SafetyError("Record/version mismatch")
    files = d.get("files")
    # Current supported schema is the documented Zenodo legacy files-list form.
    if not isinstance(files, list):
        raise SafetyError("Unsupported official files schema; review required")
    matches = [f for f in files if f.get("key") == NAME]
    if len(files) != 1 or len(matches) != 1:
        raise SafetyError("Unexpected or duplicate archive entries")
    f = matches[0]
    if f.get("checksum") != "md5:" + MD5:
        raise SafetyError("Publisher checksum mismatch")
    n = f.get("size")
    if type(n) is not int or n <= 0:
        raise SafetyError("Exact positive integer size required")
    links = f.get("links", {})
    url = links.get("content") or links.get("self")
    validate_url(url)
    tail = urllib.parse.unquote(urllib.parse.urlsplit(url).path)
    if not (tail.endswith("/" + NAME) or tail.endswith("/" + NAME + "/content")):
        raise SafetyError("Resolved file name mismatch")
    return Spec(url, n, MD5, digest(raw), RECORD, "1.0.0")

def fetch_official_metadata(ca_bundle=None):
    """Called only after separate CLI authorization; requests JSON, not an archive."""
    request = urllib.request.Request(METADATA_URL, headers={"Accept": "application/json",
        "Accept-Encoding": "identity", "User-Agent": "M100-guarded-acquisition/1"})
    with opener(ca_bundle).open(request, timeout=30) as r:
        if r.status != 200 or r.geturl() != METADATA_URL:
            raise SafetyError("Official metadata response identity changed")
        if "json" not in r.headers.get("Content-Type", "").lower():
            raise SafetyError("Official response is not JSON")
        raw = r.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise SafetyError("Official metadata exceeds cap")
    return parse_metadata(raw), raw

# Conservative local clock-skew policy, not a fixed minimum mandated by RFC 9110.
LAST_MODIFIED_DATE_MARGIN_SECONDS = 60


def http_date(value):
    """Accept canonical IMF-fixdate only; unsupported forms fail closed.

    The exact received value is retained for If-Range, never guessed from
    metadata, local mtime, a prior failed response, or the client clock.
    """
    if not isinstance(value, str) or not re.fullmatch(
            r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), [0-9]{2} "
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
            r"[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2} GMT", value):
        raise SafetyError("Invalid/unsupported canonical HTTP date")
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None or format_datetime(parsed, usegmt=True) != value:
            raise ValueError("Date or weekday is not canonical")
        return parsed
    except (ValueError, TypeError, OverflowError) as error:
        raise SafetyError("Invalid HTTP date or weekday") from error


def strong_etag(value):
    return isinstance(value, str) and bool(re.fullmatch(r'"[\x21\x23-\x7e\x80-\xff]*"', value))


def validate_validator(validator):
    """Validate versioned, typed checkpoint evidence without changing it."""
    if validator is None:
        return None
    if not isinstance(validator, dict):
        raise SafetyError("Invalid recorded validator schema; no legacy conversion")
    kind = validator.get("kind")
    if kind == "etag":
        if set(validator) != {"kind", "value"} or not strong_etag(validator["value"]):
            raise SafetyError("Strong ETag required; weak/malformed ETag rejected")
    elif kind == "last_modified":
        if set(validator) != {"kind", "value", "response_date"}:
            raise SafetyError("Invalid recorded Last-Modified evidence")
        modified = http_date(validator["value"])
        date = http_date(validator["response_date"])
        if (date - modified).total_seconds() < LAST_MODIFIED_DATE_MARGIN_SECONDS:
            raise SafetyError("Last-Modified lacks conservative strong-date evidence")
    else:
        raise SafetyError("Unknown recorded validator kind")
    return validator


def response_validator(headers):
    """Prefer a strong ETag; date fallback only when no ETag was received."""
    for field in ("ETag", "Last-Modified", "Date"):
        if len(headers.get_all(field, [])) > 1:
            raise SafetyError("Duplicate framing or identity headers")
    tags = headers.get_all("ETag", [])
    if tags:
        # A weak/malformed ETag is not permission to substitute an HTTP date.
        candidate = {"kind": "etag", "value": tags[0]}
    else:
        modified, date = headers.get("Last-Modified"), headers.get("Date")
        if modified is None or date is None:
            raise SafetyError("Strong ETag or valid Last-Modified plus Date required")
        candidate = {"kind": "last_modified", "value": modified, "response_date": date}
    return validate_validator(candidate)


def match_validator(recorded, candidate):
    validate_validator(candidate)
    if recorded is None:
        return candidate
    validate_validator(recorded)
    if recorded["kind"] != candidate["kind"]:
        raise SafetyError("Remote validator kind changed; no upgrade or downgrade")
    if recorded["value"] != candidate["value"]:
        name = "ETag" if recorded["kind"] == "etag" else "Last-Modified"
        raise SafetyError("Remote " + name + " changed")
    # Date normally changes on subsequent responses. Preserve the original
    # proof in the immutable checkpoint chain; each new response is validated.
    return recorded


@dataclass(frozen=True)
class Spec:
    url: str
    size: int
    md5: str
    metadata_sha256: str
    record: str = RECORD
    version: str = "1.0.0"
    def identity(self):
        return self.__dict__.copy()

@dataclass(frozen=True)
class Limits:
    max_requests: int = 5
    max_storage_bytes: int = 80_000_000_000
    reserve_free_bytes: int = 10_000_000_000
    evidence_reserve_bytes: int = 1024 * 1024
    timeout: float = 60
    chunk_bytes: int = CHUNK

@contextlib.contextmanager
def acquire_lock(path, create=False):
    safe_path(path)
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create: flags |= os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        regular(path)
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e: raise SafetyError("Concurrent acquisition writer") from e
        yield
    finally:
        os.close(fd)  # persistent lock inode; no truncate/unlink on resume

def retained_bytes(*roots):
    total = 0
    for root in roots:
        safe_path(root)
        if not root.exists(): continue
        for p in root.rglob("*"):
            if p.is_symlink(): raise SafetyError("Symlink in acquisition tree")
            if p.is_file(): total += regular(p, links=p.stat().st_nlink).st_size
            elif not p.is_dir(): raise SafetyError("Unexpected acquisition entry")
    return total

class Engine:
    """Single writer, immutable evidence, strict HTTP framing and no-clobber final."""
    def __init__(self, spec, data, output, binding, limits=Limits(), *,
                 synthetic=False, sleep=time.sleep, ca_bundle=None, metadata_bytes=None):
        validate_url(spec.url, synthetic)
        if type(spec.size) is not int or spec.size <= 0 or not re.fullmatch("[a-f0-9]{32}", spec.md5):
            raise SafetyError("Exact size/checksum required")
        if not re.fullmatch("[a-f0-9]{64}", spec.metadata_sha256):
            raise SafetyError("Metadata hash required")
        if (type(limits.max_requests) is not int or not 1 <= limits.max_requests <= 5
            or type(limits.chunk_bytes) is not int or not 1 <= limits.chunk_bytes <= CHUNK
            or type(limits.max_storage_bytes) is not int or type(limits.reserve_free_bytes) is not int
            or limits.max_storage_bytes <= 0 or limits.reserve_free_bytes < 0
            or limits.evidence_reserve_bytes < 4096 or limits.timeout <= 0):
            raise SafetyError("Invalid retry/storage/IO bounds")
        if not synthetic and (spec.record, spec.version, spec.md5) != (RECORD, "1.0.0", MD5):
            raise SafetyError("Production identity must be pinned")
        if metadata_bytes is not None and digest(metadata_bytes) != spec.metadata_sha256:
            raise SafetyError("Official metadata raw-byte hash mismatch")
        self.metadata_bytes = metadata_bytes
        self.spec, self.data, self.output = spec, safe_path(data), safe_path(output)
        if self.data == self.output or self.data in self.output.parents or self.output in self.data.parents:
            raise SafetyError("Data/evidence namespaces must be disjoint")
        self.binding, self.limits, self.synthetic, self.sleep = binding, limits, synthetic, sleep
        self.transport = opener(ca_bundle)
        self.owner = None
    def space(self, extra=0):
        used = retained_bytes(self.data, self.output)
        if used + extra + self.limits.evidence_reserve_bytes > self.limits.max_storage_bytes:
            raise SafetyError("Acquisition retained-byte limit")
        p = self.data if self.data.exists() else self.data.parent
        while not p.exists(): p = p.parent
        v = os.statvfs(p)
        if v.f_flag & os.ST_RDONLY or v.f_bavail * v.f_frsize < extra + self.limits.reserve_free_bytes:
            raise SafetyError("Read-only filesystem or insufficient free space")
    def event(self, kind, **details):
        self.space(4096)
        create_json(self.output / ("event-" + uuid.uuid4().hex + ".json"),
            {"kind": kind, "owner": self.owner, **details})
    def prepare(self, resume_owner=None):
        for p in (self.data, self.output): safe_path(p)
        final = self.data / NAME
        if final.exists() or final.is_symlink(): raise SafetyError("Final destination exists")
        expected = {"source": self.spec.identity(), "binding": self.binding,
                    "uid": os.getuid(), "data": str(self.data), "output": str(self.output),
                    "limits": self.limits.__dict__}
        if resume_owner:
            if not re.fullmatch("[a-f0-9]{32}", resume_owner): raise SafetyError("Invalid owner token")
            records = [json.loads(read_file(p / "session.json")) for p in (self.data, self.output)]
            if records[0] != records[1] or records[0] != {**expected, "owner": resume_owner}:
                raise SafetyError("Resume ownership/source/configuration mismatch")
            self.owner = resume_owner
            for root in (self.data, self.output):
                for p in root.iterdir():
                    if p.is_symlink(): raise SafetyError("Symlink in session")
                    n = p.name
                    if root == self.data:
                        if n not in {"session.json", ".acquisition.lock", "payload.part"}:
                            raise SafetyError("Unexpected data artifact")
                    elif n not in {"session.json", "official_metadata.json"} and not re.fullmatch(
                        r"(request|event|state)-[a-f0-9]{32}\.json", n):
                        raise SafetyError("Unexpected evidence artifact")
                    regular(p)
        else:
            if any(p.exists() or p.is_symlink() for p in (self.data, self.output)):
                raise SafetyError("Unexpected existing acquisition directory")
            self.space(self.spec.size)
            # Exclusive session-local directories; parent paths are reviewed separately.
            for p in (self.data, self.output): p.mkdir(parents=True, mode=0o700, exist_ok=False)
            self.owner = uuid.uuid4().hex
            for p in (self.data, self.output):
                create_json(p / "session.json", {**expected, "owner": self.owner})
            if self.metadata_bytes is not None:
                create_bytes(self.output / "official_metadata.json", self.metadata_bytes)
        if self.metadata_bytes is not None:
            if digest(read_file(self.output / "official_metadata.json")) != self.spec.metadata_sha256:
                raise SafetyError("Retained official metadata changed")
        return not bool(resume_owner)
    def checkpoint(self, validator, previous=None, prefix=None):
        validate_validator(validator)
        h = hashes(self.data / "payload.part") if prefix is None else dict(prefix)
        if (self.data / "payload.part").stat().st_size != h["bytes"]:
            raise SafetyError("Checkpoint length changed")
        if h["bytes"] > self.spec.size: raise SafetyError("Oversized partial")
        if h["bytes"] and validator is None: raise SafetyError("No strong validator for resume")
        state = {"owner": self.owner, "prefix": h, "validator": validator,
                 "previous": previous, "source_sha256": digest(canonical(self.spec.identity()))}
        path = self.output / ("state-" + uuid.uuid4().hex + ".json")
        create_json(path, state)
        return digest(canonical(state)), state
    def latest(self):
        paths = list(self.output.glob("state-*.json")); states = {}
        for p in paths:
            s = json.loads(read_file(p))
            validate_validator(s["validator"])
            if s["prefix"]["bytes"] and s["validator"] is None:
                raise SafetyError("No strong validator for resume")
            if s["owner"] != self.owner or s["source_sha256"] != digest(canonical(self.spec.identity())):
                raise SafetyError("Checkpoint source/owner mismatch")
            states[digest(canonical(s))] = s
        if not states: raise SafetyError("Missing durable prefix checkpoint")
        if len(states) != len(paths): raise SafetyError("Duplicate state records")
        parents = {s["previous"] for s in states.values() if s["previous"] is not None}
        if not parents.issubset(states): raise SafetyError("Broken checkpoint chain")
        tips = set(states) - parents
        if len(tips) != 1: raise SafetyError("Checkpoint branches")
        tip = tips.pop(); seen = set(); p = tip
        while p is not None:
            if p in seen: raise SafetyError("Checkpoint cycle")
            seen.add(p); p = states[p]["previous"]
        if len(seen) != len(states): raise SafetyError("Disconnected checkpoints")
        current = hashes(self.data / "payload.part")
        if current != states[tip]["prefix"]: raise SafetyError("Uncommitted/tampered partial prefix")
        return tip, states[tip]
    def publish(self):
        h = hashes(self.data / "payload.part")
        if h["bytes"] != self.spec.size or h["md5"] != self.spec.md5:
            self.event("checksum_or_size_failure", actual=h)
            raise SafetyError("Final checksum/size mismatch; partial retained")
        # Atomic name creation on the SAME filesystem; never os.replace().
        dfd = os.open(self.data, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try: os.link("payload.part", NAME, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
        finally: os.close(dfd)
        sync_dir(self.data)
        final = self.data / NAME
        if hashes(final) != h: raise SafetyError("Final reload hash mismatch")
        self.event("published", final=NAME, hashes=h)
        create_json(self.output / "completed.json", {"owner": self.owner,
            "source": self.spec.identity(), "binding": self.binding, "hashes": h,
            "requests": len(list(self.output.glob("request-*.json")))})
        return {"status": "verified_download_complete", "owner": self.owner, **h}
    def run(self, resume_owner=None):
        fresh = self.prepare(resume_owner)
        with acquire_lock(self.data / ".acquisition.lock", create=fresh):
            if fresh:
                fd = os.open(self.data / "payload.part",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                os.close(fd); sync_dir(self.data)
                self.checkpoint(None)
            previous, state = self.latest()
            validator = state["validator"]
            offset = state["prefix"]["bytes"]
            sha_acc, md_acc = hashlib.sha256(), hashlib.md5()
            fd = os.open(self.data / "payload.part", os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as f:
                for block in iter(lambda: f.read(CHUNK), b""):
                    sha_acc.update(block); md_acc.update(block)
            if (sha_acc.hexdigest(), md_acc.hexdigest()) != (state["prefix"]["sha256"], state["prefix"]["md5"]):
                raise SafetyError("Prefix changed while preparing resume")
            if offset and not validator: raise SafetyError("No strong validator for resume")
            if offset == self.spec.size: return self.publish()
            while len(list(self.output.glob("request-*.json"))) < self.limits.max_requests:
                self.space(self.spec.size - offset)
                headers = {"Accept-Encoding": "identity", "Connection": "close"}
                if offset:
                    headers.update(Range=f"bytes={offset}-", **{"If-Range": validate_validator(validator)["value"]})
                create_json(self.output / ("request-" + uuid.uuid4().hex + ".json"),
                    {"owner": self.owner, "url": self.spec.url, "offset": offset, "headers": headers})
                transient = False
                try:
                    request = urllib.request.Request(self.spec.url, headers=headers)
                    try: response = self.transport.open(request, timeout=self.limits.timeout)
                    except urllib.error.HTTPError as error: response = error
                    with response as r:
                        values = list(r.headers.items())
                        if len(canonical(values)) > 32768:
                            raise SafetyError("Response headers exceed evidence limit")
                        self.event("response", status=r.code, headers=values, url=r.geturl())
                        if r.geturl() != self.spec.url: raise SafetyError("Response URL identity changed")
                        if r.code in {408, 425, 429, 500, 502, 503, 504}:
                            raise TransientStatus("Transient HTTP " + str(r.code))
                        if r.code not in (200, 206): raise SafetyError("Rejected HTTP " + str(r.code))
                        if r.headers.get("Content-Encoding", "identity").lower() != "identity":
                            raise SafetyError("Encoded body prohibited")
                        for critical in ("Content-Length", "Content-Range", "ETag", "Last-Modified", "Date", "Content-Encoding", "Transfer-Encoding"):
                            if len(r.headers.get_all(critical, [])) > 1:
                                raise SafetyError("Duplicate framing or identity headers")
                        n = r.headers.get("Content-Length")
                        if n is None or not n.isdigit() or int(n) != self.spec.size - offset:
                            raise SafetyError("Exact remaining Content-Length required")
                        if r.headers.get("Transfer-Encoding"):
                            raise SafetyError("Ambiguous/chunked HTTP framing prohibited")
                        if offset:
                            expected_range = f"bytes {offset}-{self.spec.size-1}/{self.spec.size}"
                            if r.code != 206 or r.headers.get("Content-Range") != expected_range:
                                raise SafetyError("Range ignored or inconsistent")
                        elif r.code != 200 or r.headers.get("Content-Range"):
                            raise SafetyError("Unexpected initial range")
                        candidate = response_validator(r.headers)
                        validator = match_validator(validator, candidate)
                        fd = os.open(self.data / "payload.part", os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                        try:
                            if os.fstat(fd).st_size != offset: raise SafetyError("Partial length changed")
                            while offset < self.spec.size:
                                read_error = None
                                try: b = r.read(min(self.limits.chunk_bytes, self.spec.size-offset))
                                except http.client.IncompleteRead as error:
                                    b, read_error = error.partial, error
                                if not b: raise IncompleteBody("Short/interrupted response")
                                if len(b) > min(self.limits.chunk_bytes, self.spec.size-offset):
                                    raise SafetyError("Transport violated exact read bound")
                                self.space(len(b))
                                # A low-level short write must not lose or mis-account bytes.
                                view = memoryview(b)
                                while view:
                                    written = os.write(fd, view)
                                    if written <= 0: raise SafetyError("Short filesystem write")
                                    view = view[written:]
                                os.fsync(fd); offset += len(b)
                                sha_acc.update(b); md_acc.update(b)
                                previous, state = self.checkpoint(validator, previous,
                                    {"bytes": offset, "sha256": sha_acc.hexdigest(), "md5": md_acc.hexdigest()})
                                if read_error: raise IncompleteBody("Interrupted framed response")
                        finally: os.close(fd)
                        # HTTPResponse honors the validated Content-Length framing.
                    return self.publish()
                except (IncompleteBody, TransientStatus, TimeoutError, ConnectionError,
                        http.client.HTTPException) as e:
                    self.event("interruption_or_transient", error=type(e).__name__, detail=str(e), offset=offset)
                    transient = True
                except BaseException as e:
                    self.event("stopped", error=type(e).__name__, detail=str(e), offset=offset)
                    raise
                if transient:
                    count = len(list(self.output.glob("request-*.json")))
                    if count >= self.limits.max_requests: break
                    self.sleep(min(15, 2 ** (count-1)))
            raise SafetyError("Bounded request budget exhausted; evidence retained")

def load_config(path):
    raw = read_file(path); d = json.loads(raw)
    expected = {"version", "record", "archive", "published_md5", "displayed_size",
        "exact_bytes", "resolved_url", "default_mode", "authorization_required",
        "data_relative", "output_relative", "max_requests", "max_storage_bytes",
        "reserve_free_bytes", "metadata_url", "ca_bundle"}
    if set(d) != expected or (d["version"], d["record"], d["archive"], d["published_md5"]) != (1, RECORD, NAME, MD5):
        raise SafetyError("Unexpected acquisition configuration")
    if (d["default_mode"] != "dry_run" or d["authorization_required"] is not True
        or d["data_relative"] != DATA_REL or d["output_relative"] != OUTPUT_REL
        or d["metadata_url"] != METADATA_URL):
        raise SafetyError("Unresolved, isolated declaration must remain unchanged")
    if d["exact_bytes"] is None:
        if d["resolved_url"] is not None: raise SafetyError("Incomplete metadata snapshot")
    else:
        if type(d["exact_bytes"]) is not int or d["exact_bytes"] <= 0:
            raise SafetyError("Exact snapshot size required")
        validate_url(d["resolved_url"])
    return d, digest(raw)

def validate_snapshot(spec, declaration):
    if declaration["exact_bytes"] is not None and (
        spec.size != declaration["exact_bytes"] or spec.url != declaration["resolved_url"]):
        raise SafetyError("Live metadata differs from recorded acquisition snapshot")

def local_transfer_precheck(declaration, binding, resume_owner=None):
    """Read-only early guard; no migration or request-budget reset is implied."""
    data, output = (safe_path(ROOT / declaration[k]) for k in ("data_relative", "output_relative"))
    if not resume_owner:
        if any(p.exists() or p.is_symlink() for p in (data, output)):
            raise SafetyError("Unexpected existing acquisition directory; preserve historical session")
        return
    if not re.fullmatch("[a-f0-9]{32}", resume_owner):
        raise SafetyError("Invalid owner token")
    a, b = (json.loads(read_file(p / "session.json")) for p in (data, output))
    if (a != b or a["owner"] != resume_owner or a["uid"] != os.getuid()
            or a["binding"] != binding or a["data"] != str(data) or a["output"] != str(output)):
        raise SafetyError("Resume ownership/source/configuration mismatch; migration not authorized")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--authorize-transfer", action="store_true")
    p.add_argument("--quota-confirmed", action="store_true")
    p.add_argument("--resume-owner")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--test-work-dir", type=Path)
    a = p.parse_args(argv)
    if a.self_test:
        if a.authorize_transfer or a.quota_confirmed or a.resume_owner:
            raise SafetyError("Synthetic and production modes cannot be combined")
        return synthetic_tests(a.test_work_dir)
    if a.test_work_dir: raise SafetyError("--test-work-dir is self-test only")
    d, config_sha = load_config(a.config)
    for rel in (d["data_relative"], d["output_relative"]): safe_path(ROOT / rel)
    if not a.authorize_transfer:
        print(json.dumps({"status": "dry_run_snapshot_available" if d["exact_bytes"] is not None else "dry_run_metadata_unresolved", "transfer_enabled": False,
            "exact_bytes": d["exact_bytes"], "resolved_url": d["resolved_url"], "live_metadata_verified": False, "directories_created": False,
            "metadata_or_archive_requests": 0, "config_sha256": config_sha}))
        return 0
    if not a.quota_confirmed: raise SafetyError("Separate user quota confirmation required")
    # This production branch is never invoked during implementation/tests.
    binding = {"config_sha256": config_sha, "helper_sha256": hashes(Path(__file__))["sha256"]}
    local_transfer_precheck(d, binding, a.resume_owner)
    spec, raw = fetch_official_metadata(d["ca_bundle"])
    validate_snapshot(spec, d)
    limits = Limits(max_requests=d["max_requests"], max_storage_bytes=d["max_storage_bytes"],
                    reserve_free_bytes=d["reserve_free_bytes"])
    engine = Engine(spec, ROOT / DATA_REL, ROOT / OUTPUT_REL,
        binding,
        limits, ca_bundle=d["ca_bundle"], metadata_bytes=raw)
    result = engine.run(a.resume_owner)
    print(json.dumps(result, sort_keys=True)); return 0

# Synthetic suite is defined below. It never invokes the production branch.

def synthetic_tests(workdir=None):
    """Run only synthetic transfers against literal loopback; retain test evidence."""
    import unittest
    from unittest import mock
    import tempfile
    import threading
    import socket
    import subprocess
    import shutil
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="m100-p18-tests-"))
    if workdir:
        safe_path(work)
        work.mkdir(mode=0o700, parents=True, exist_ok=False)
    body = bytes(range(256)) * 64
    requests_seen = []
    production_calls = []
    original_connect = socket.socket.connect
    def loopback_connect(sock, address):
        if isinstance(address, tuple):
            try: local = ipaddress.ip_address(address[0]).is_loopback
            except ValueError: local = False
            if not local: raise AssertionError("Non-loopback socket blocked by test harness")
        return original_connect(sock, address)

    class Server:
        def __init__(self, modes=None, tls=False):
            self.modes = list(modes or ["ok"]); self.seen = []
            outer = self
            class Handler(BaseHTTPRequestHandler):
                protocol_version = "HTTP/1.1"
                def log_message(self, *args): pass
                def do_GET(self):
                    mode = outer.modes.pop(0) if outer.modes else "ok"
                    row = {"path": self.path, "range": self.headers.get("Range"),
                           "if_range": self.headers.get("If-Range"), "mode": mode}
                    outer.seen.append(row); requests_seen.append(row)
                    if mode == "redirect":
                        self.send_response(302); self.send_header("Location", "/must-not-follow")
                        self.send_header("Content-Length", "0"); self.end_headers(); return
                    if mode in ("416", "503"):
                        self.send_response(int(mode)); self.send_header("Content-Length", "0")
                        self.end_headers(); return
                    off = int(self.headers.get("Range", "bytes=0-")[6:-1])
                    if mode == "ignore": off = 0
                    content = body[off:]
                    status = 206 if off else 200
                    self.send_response(status)
                    length = len(content) + (1 if mode == "oversize" else 0)
                    self.send_header("Content-Length", str(length))
                    if mode == "duplicate_length": self.send_header("Content-Length", str(length))
                    if mode != "missing_etag":
                        self.send_header("ETag", '"changed"' if mode == "etag_change" else '"fixture-v1"')
                    if off:
                        a = off + (1 if mode == "wrong_range" else 0)
                        self.send_header("Content-Range", f"bytes {a}-{len(body)-1}/{len(body)}")
                    if mode == "encoded": self.send_header("Content-Encoding", "gzip")
                    self.end_headers()
                    try:
                        if mode in ("short", "drop"):
                            self.wfile.write(content[:len(content)//2]); self.wfile.flush()
                            self.close_connection = True; self.connection.shutdown(socket.SHUT_RDWR)
                        elif mode == "corrupt":
                            self.wfile.write(bytes([content[0] ^ 1]) + content[1:])
                        else:
                            self.wfile.write(content)
                    except (BrokenPipeError, ConnectionResetError, OSError): pass
            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.server.daemon_threads = True
            if tls:
                certdir = work / ("cert-" + uuid.uuid4().hex); certdir.mkdir()
                binary = shutil.which("openssl")
                if not binary: raise AssertionError("OpenSSL required for actual untrusted TLS fixture")
                subprocess.run([binary, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(certdir/"key.pem"), "-out", str(certdir/"cert.pem"),
                    "-days", "1", "-subj", "/CN=wrong-host.invalid"], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(certdir/"cert.pem", certdir/"key.pem")
                self.cert_path = certdir/"cert.pem"
                self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
            self.url = ("https" if tls else "http") + "://127.0.0.1:" + str(self.server.server_port) + "/fixture.bin"
            self.thread = threading.Thread(target=self.server.serve_forever,
                kwargs={"poll_interval": .01}, daemon=True); self.thread.start()
        def close(self):
            self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    class Cases(unittest.TestCase):
        def setUp(self):
            self.base = work / (self._testMethodName + "-" + uuid.uuid4().hex)
            self.base.mkdir()
            self.servers = []
        def tearDown(self):
            for s in self.servers: s.close()
        def engine(self, modes=None, *, md5=None, count=3, cap=2_000_000, tls=False):
            server = Server(modes, tls); self.servers.append(server)
            spec = Spec(server.url, len(body), md5 or hashlib.md5(body).hexdigest(),
                digest(b"synthetic official metadata"), "synthetic", "1")
            limits = Limits(count, cap, 0, 65536, 2, 2048)
            e = Engine(spec, self.base/"data", self.base/"evidence", {"test": "fixture-v1"},
                limits, synthetic=True, sleep=lambda _: None)
            return e, server
        def prefix(self, e, modes=None):
            # Controlled stop after a durable short-body prefix, before retry.
            e.sleep = lambda _: (_ for _ in ()).throw(KeyboardInterrupt("synthetic interruption"))
            with self.assertRaises(KeyboardInterrupt): e.run()
            self.assertGreater((e.data/"payload.part").stat().st_size, 0)
            return e.owner
        def assert_retained(self, e):
            self.assertTrue((e.data/"payload.part").exists())
            self.assertFalse((e.data/NAME).exists())
            self.assertTrue(list(e.output.glob("event-*.json")))
        def metadata(self):
            return {"id": int(RECORD), "metadata": {"version": "1.0.0"},
                "files": [{"key": NAME, "size": 123, "checksum": "md5:"+MD5,
                    "links": {"self": "https://zenodo.org/api/files/example/" + NAME}}]}
        def test_exact_size_success_and_sha256(self):
            e,s = self.engine(); out = e.run()
            self.assertEqual(out["status"], "verified_download_complete")
            self.assertEqual((e.data/NAME).read_bytes(), body)
            self.assertEqual(out["sha256"], digest(body))
            self.assertEqual(out["bytes"], len(body))
            self.assertEqual((e.data/"payload.part").stat().st_ino,(e.data/NAME).stat().st_ino)
            self.assertEqual(len(s.seen), 1)
        def test_checksum_failure_retained(self):
            e,_ = self.engine(["corrupt"])
            with self.assertRaisesRegex(SafetyError,"checksum"): e.run()
            self.assert_retained(e); self.assertEqual((e.data/"payload.part").stat().st_size,len(body))
        def test_short_body_retained_request_budget(self):
            e,s = self.engine(["short"],count=1)
            with self.assertRaisesRegex(SafetyError,"budget"): e.run()
            self.assert_retained(e); self.assertEqual(len(s.seen),1)
        def test_oversized_length_rejected_before_body(self):
            e,_ = self.engine(["oversize"])
            with self.assertRaisesRegex(SafetyError,"Content-Length"): e.run()
            self.assert_retained(e); self.assertEqual((e.data/"payload.part").stat().st_size,0)
        def test_interruption_and_valid_resume(self):
            e,s = self.engine(["short","ok"]); token=self.prefix(e)
            e.sleep=lambda _:None; out=e.run(token)
            self.assertEqual(out["md5"],hashlib.md5(body).hexdigest())
            self.assertEqual(s.seen[1]["range"],"bytes=8192-")
            self.assertEqual(s.seen[1]["if_range"],'"fixture-v1"')
        def test_transient_short_body_auto_resume(self):
            e,s=self.engine(["short","ok"]); e.run()
            self.assertEqual(len(s.seen),2); self.assertTrue(s.seen[1]["range"])
        def test_wrong_owner_rejected_without_request(self):
            e,s=self.engine(["short"]); self.prefix(e)
            with self.assertRaisesRegex(SafetyError,"ownership"):e.run("0"*32)
            self.assertEqual(len(s.seen),1)
        def test_changed_source_checksum_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            other=Engine(Spec(e.spec.url,e.spec.size,"0"*32,e.spec.metadata_sha256,"synthetic","1"),
                e.data,e.output,e.binding,e.limits,synthetic=True)
            with self.assertRaisesRegex(SafetyError,"ownership"):other.run(token)
            self.assertEqual(len(s.seen),1)
        def test_changed_exact_size_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            other=Engine(Spec(e.spec.url,e.spec.size+1,e.spec.md5,e.spec.metadata_sha256,"synthetic","1"),
                e.data,e.output,e.binding,e.limits,synthetic=True)
            with self.assertRaisesRegex(SafetyError,"ownership"):other.run(token)
            self.assertEqual(len(s.seen),1)
        def test_changed_configuration_binding_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e);e.binding={"test":"changed"}
            with self.assertRaisesRegex(SafetyError,"ownership"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_tampered_prefix_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            p=e.data/"payload.part"
            with p.open("r+b") as f:f.write(b"x")
            with self.assertRaisesRegex(SafetyError,"prefix"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_uncommitted_trailing_bytes_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            with (e.data/"payload.part").open("ab") as f:f.write(b"uncommitted")
            with self.assertRaisesRegex(SafetyError,"prefix"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_server_ignores_range(self):
            e,s=self.engine(["short","ignore"]);token=self.prefix(e)
            h=hashes(e.data/"payload.part")
            with self.assertRaisesRegex(SafetyError,"Content-Length|Range"):e.run(token)
            self.assertEqual(h,hashes(e.data/"payload.part"));self.assertEqual(len(s.seen),2)
        def test_inconsistent_range_preserves_prefix(self):
            e,_=self.engine(["short","wrong_range"]);token=self.prefix(e);h=hashes(e.data/"payload.part")
            with self.assertRaisesRegex(SafetyError,"Range"):e.run(token)
            self.assertEqual(h,hashes(e.data/"payload.part"))
        def test_http416_never_deletes_partial(self):
            e,_=self.engine(["short","416"]);token=self.prefix(e);h=hashes(e.data/"payload.part")
            with self.assertRaisesRegex(SafetyError,"416"):e.run(token)
            self.assertEqual(h,hashes(e.data/"payload.part"))
        def test_remote_validator_changed(self):
            e,_=self.engine(["short","etag_change"]);token=self.prefix(e)
            with self.assertRaisesRegex(SafetyError,"ETag changed"):e.run(token)
            self.assert_retained(e)
        def test_missing_strong_validator_rejected(self):
            e,_=self.engine(["missing_etag"])
            with self.assertRaisesRegex(SafetyError,"Strong ETag"):e.run()
            self.assert_retained(e)
        def test_redirect_not_followed(self):
            e,s=self.engine(["redirect"])
            with self.assertRaisesRegex(SafetyError,"302"):e.run()
            self.assertEqual(len(s.seen),1);self.assertEqual(s.seen[0]["path"],"/fixture.bin")
        def test_actual_untrusted_tls_rejected(self):
            e,s=self.engine(tls=True)
            with self.assertRaises(urllib.error.URLError):e.run()
            self.assertEqual(len(s.seen),0);self.assert_retained(e)
        def test_tls_context_enforces_cert_and_hostname(self):
            c=tls_context();self.assertTrue(c.check_hostname);self.assertEqual(c.verify_mode,ssl.CERT_REQUIRED)
        def test_response_identity_mismatch(self):
            e,s=self.engine()
            class WrongResponse:
                code=200
                headers={}
                def __enter__(self):return self
                def __exit__(self,*a):return False
                def geturl(self):return s.url+"/changed"
            e.transport=mock.Mock();e.transport.open.return_value=WrongResponse()
            with self.assertRaisesRegex(SafetyError,"identity"):e.run()
            self.assert_retained(e);self.assertEqual(len(s.seen),0)
        def test_existing_final_not_overwritten(self):
            e,s=self.engine();e.data.mkdir();(e.data/NAME).write_bytes(b"old")
            with self.assertRaisesRegex(SafetyError,"Final destination"):e.run()
            self.assertEqual((e.data/NAME).read_bytes(),b"old");self.assertEqual(len(s.seen),0)
        def test_racing_final_publication_no_clobber(self):
            e,_=self.engine();original=e.publish
            def race():
                (e.data/NAME).write_bytes(b"racing-writer")
                return original()
            e.publish=race
            with self.assertRaises(FileExistsError):e.run()
            self.assertEqual((e.data/NAME).read_bytes(),b"racing-writer")
            self.assertEqual((e.data/"payload.part").read_bytes(),body)
        def test_symlink_data_destination_rejected(self):
            e,s=self.engine();target=self.base/"outside";target.mkdir();e.data.symlink_to(target)
            with self.assertRaisesRegex(SafetyError,"Symlink"):e.run()
            self.assertEqual(list(target.iterdir()),[]);self.assertEqual(len(s.seen),0)
        def test_symlink_final_destination_rejected(self):
            e,s=self.engine();e.data.mkdir();(e.data/NAME).symlink_to(self.base/"absent")
            with self.assertRaisesRegex(SafetyError,"Final destination"):e.run()
            self.assertEqual(len(s.seen),0)
        def test_concurrent_writer_lock(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            with acquire_lock(e.data/".acquisition.lock"):
                with self.assertRaisesRegex(SafetyError,"Concurrent"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_request_budget_across_resume(self):
            e,s=self.engine(["short","503","503"],count=2);token=self.prefix(e);e.sleep=lambda _:None
            with self.assertRaisesRegex(SafetyError,"budget"):e.run(token)
            with self.assertRaisesRegex(SafetyError,"budget"):e.run(token)
            self.assertEqual(len(s.seen),2)
        def test_transient_retry_limit(self):
            e,s=self.engine(["503"]*5,count=3)
            with self.assertRaisesRegex(SafetyError,"budget"):e.run()
            self.assertEqual(len(s.seen),3)
        def test_storage_limit_before_creation(self):
            e,s=self.engine(cap=100)
            with self.assertRaisesRegex(SafetyError,"retained-byte"):e.run()
            self.assertFalse(e.data.exists());self.assertEqual(len(s.seen),0)
        def test_free_space_limit(self):
            e,s=self.engine()
            e.limits=Limits(3,2_000_000,10**30,65536,2,2048)
            with self.assertRaisesRegex(SafetyError,"free space"):e.run()
            self.assertFalse(e.data.exists());self.assertEqual(len(s.seen),0)
        def test_duplicate_framing_rejected(self):
            e,_=self.engine(["duplicate_length"])
            with self.assertRaisesRegex(SafetyError,"Duplicate"):e.run()
            self.assert_retained(e)
        def test_encoded_body_rejected(self):
            e,_=self.engine(["encoded"])
            with self.assertRaisesRegex(SafetyError,"Encoded"):e.run()
            self.assert_retained(e)
        def test_unexpected_directory_refused(self):
            e,s=self.engine();e.output.mkdir();p=e.output/"old";p.write_bytes(b"preserve")
            with self.assertRaisesRegex(SafetyError,"Unexpected existing"):e.run()
            self.assertEqual(p.read_bytes(),b"preserve");self.assertEqual(len(s.seen),0)
        def test_unexpected_resume_file_refused(self):
            e,s=self.engine(["short"]);token=self.prefix(e);(e.output/"unowned").write_bytes(b"old")
            with self.assertRaisesRegex(SafetyError,"Unexpected evidence"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_dry_run_no_network_no_directories(self):
            p=self.base/"config.json";p.write_text(json.dumps(TEST_CONFIG))
            with mock.patch.object(sys.modules[__name__],"fetch_official_metadata",
                    side_effect=AssertionError("production metadata call")) as fetched:
                with contextlib.redirect_stdout(__import__("io").StringIO()) as out:
                    self.assertEqual(main(["--config",str(p)]),0)
                self.assertFalse(json.loads(out.getvalue())["transfer_enabled"]);fetched.assert_not_called()
        def test_authorization_without_quota_stops_before_network(self):
            p=self.base/"config.json";p.write_text(json.dumps(TEST_CONFIG))
            with mock.patch.object(sys.modules[__name__],"fetch_official_metadata") as fetched:
                with self.assertRaisesRegex(SafetyError,"quota"):
                    main(["--config",str(p),"--authorize-transfer"])
                fetched.assert_not_called()
        def test_selftest_cannot_enable_production(self):
            with self.assertRaisesRegex(SafetyError,"cannot be combined"):
                main(["--self-test","--authorize-transfer"])
        def test_metadata_valid_exact_fields(self):
            m=self.metadata();s=parse_metadata(canonical(m))
            self.assertEqual(s.size,123);self.assertEqual(s.url,m["files"][0]["links"]["self"])
        def test_metadata_rounded_size_refused(self):
            m=self.metadata();m["files"][0]["size"]="31.5 GB"
            with self.assertRaisesRegex(SafetyError,"Exact"):parse_metadata(canonical(m))
        def test_metadata_missing_url_not_guessed(self):
            m=self.metadata();m["files"][0]["links"]={}
            with self.assertRaises(SafetyError):parse_metadata(canonical(m))
        def test_metadata_wrong_checksum(self):
            m=self.metadata();m["files"][0]["checksum"]="md5:"+"0"*32
            with self.assertRaisesRegex(SafetyError,"checksum"):parse_metadata(canonical(m))
        def test_metadata_record_and_version(self):
            for key in ("id","version"):
                m=self.metadata()
                if key=="id":m["id"]=1
                else:m["metadata"]["version"]="2"
                with self.assertRaisesRegex(SafetyError,"Record/version"):parse_metadata(canonical(m))
        def test_metadata_duplicate_and_schema(self):
            m=self.metadata();m["files"]*=2
            with self.assertRaises(SafetyError):parse_metadata(canonical(m))
            m["files"]={}
            with self.assertRaises(SafetyError):parse_metadata(canonical(m))
        def test_metadata_insecure_and_wrong_host(self):
            for url in ("http://zenodo.org/"+NAME,"https://evil.invalid/"+NAME,
                        "https://user:password@zenodo.org/"+NAME):
                m=self.metadata();m["files"][0]["links"]["self"]=url
                with self.assertRaises(SafetyError):parse_metadata(canonical(m))
        def test_limits_invalid_request_counts(self):
            e,_=self.engine()
            for n in (0,6,True):
                with self.assertRaises(SafetyError):
                    Engine(e.spec,e.data,e.output,e.binding,Limits(n,2_000_000,0,65536,2,2048),synthetic=True)
        def test_synthetic_nonloopback_prohibited(self):
            e,_=self.engine()
            with self.assertRaisesRegex(SafetyError,"loopback"):
                Engine(Spec("http://example.com/a",1,e.spec.md5,e.spec.metadata_sha256),
                    e.data,e.output,e.binding,e.limits,synthetic=True)

        def test_actual_trusted_certificate_wrong_hostname(self):
            e,s=self.engine(tls=True)
            e.transport=opener(s.cert_path)
            with self.assertRaises(urllib.error.URLError) as error:e.run()
            self.assertIsInstance(error.exception.reason,ssl.SSLCertVerificationError)
            self.assertIn("IP address mismatch",str(error.exception.reason))
            self.assertEqual(len(s.seen),0);self.assert_retained(e)
        def test_metadata_boolean_size_rejected(self):
            m=self.metadata();m["files"][0]["size"]=True
            with self.assertRaises(SafetyError):parse_metadata(canonical(m))
        def test_resume_lock_inode_and_bytes_unchanged(self):
            e,_=self.engine(["short","ok"]);token=self.prefix(e)
            p=e.data/".acquisition.lock";before=(p.stat().st_ino,p.read_bytes(),p.stat().st_mtime_ns)
            e.sleep=lambda _:None;e.run(token)
            self.assertEqual(before,(p.stat().st_ino,p.read_bytes(),p.stat().st_mtime_ns))
        def test_resume_partial_symlink_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            p=e.data/"payload.part";target=self.base/"saved-prefix";p.rename(target);p.symlink_to(target)
            with self.assertRaises(SafetyError):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_official_raw_metadata_hash_binding(self):
            e,_=self.engine()
            with self.assertRaisesRegex(SafetyError,"raw-byte hash"):
                Engine(e.spec,e.data,e.output,e.binding,e.limits,synthetic=True,metadata_bytes=b"changed")
        def test_broken_checkpoint_chain_rejected(self):
            e,s=self.engine(["short"]);token=self.prefix(e)
            p=next(e.output.glob("state-*.json"));obj=json.loads(p.read_text());obj["previous"]="0"*64
            p.write_bytes(canonical(obj))
            with self.assertRaisesRegex(SafetyError,"chain|Checkpoint"):e.run(token)
            self.assertEqual(len(s.seen),1)
        def test_production_identity_not_substitutable(self):
            e,_=self.engine()
            with self.assertRaises(SafetyError):
                Engine(e.spec,e.data,e.output,e.binding,e.limits)
        def test_injected_response_read_oversize_never_appended(self):
            # HTTP implementation normally respects read(n); defend custom transports too.
            e,s=self.engine()
            class Response:
                code=200
                headers=__import__("email.message",fromlist=["Message"]).Message()
                headers["Content-Length"]=str(len(body))
                headers["ETag"]='"fixture-v1"'
                def __enter__(self):return self
                def __exit__(self,*a):return False
                def geturl(self):return s.url
                def read(self,n):return b"x"*(n+1)
            e.transport=mock.Mock();e.transport.open.return_value=Response()
            with self.assertRaisesRegex(SafetyError,"read bound"):e.run()
            self.assertEqual((e.data/"payload.part").stat().st_size,0)


        def test_official_content_url_schema(self):
            m=self.metadata()
            m["files"][0]["links"]["self"]="https://zenodo.org/api/records/7590061/files/22-03.tar/content"
            s=parse_metadata(canonical(m));self.assertEqual(s.url,m["files"][0]["links"]["self"])
        def test_metadata_wrong_content_file_rejected(self):
            m=self.metadata();m["files"][0]["links"]["self"]="https://zenodo.org/api/records/7590061/files/other.tar/content"
            with self.assertRaisesRegex(SafetyError,"file name"):parse_metadata(canonical(m))
        def test_live_snapshot_mismatch_rejected(self):
            s=parse_metadata(canonical(self.metadata()))
            for d in ({"exact_bytes":124,"resolved_url":s.url},
                      {"exact_bytes":123,"resolved_url":s.url+"?changed=1"}):
                with self.assertRaisesRegex(SafetyError,"snapshot"):validate_snapshot(s,d)
        def test_resolved_snapshot_dry_run_no_network(self):
            d=dict(TEST_CONFIG);d["exact_bytes"]=123
            d["resolved_url"]="https://zenodo.org/api/files/example/"+NAME
            p=self.base/"resolved.json";p.write_text(json.dumps(d))
            with mock.patch.object(sys.modules[__name__],"fetch_official_metadata") as fetch:
                with contextlib.redirect_stdout(__import__("io").StringIO()) as out:
                    self.assertEqual(main(["--config",str(p)]),0)
                self.assertEqual(json.loads(out.getvalue())["exact_bytes"],123);fetch.assert_not_called()

        def test_old_synthetic_files_preserved(self):
            protected=self.base/"historical";protected.mkdir();old=protected/"receipt.json"
            old.write_bytes(b'{"immutable":true}\n')
            before=(hashes(old),old.stat().st_mtime_ns,stat.S_IMODE(old.stat().st_mode))
            e,_=self.engine();e.run()
            self.assertEqual(before,(hashes(old),old.stat().st_mtime_ns,stat.S_IMODE(old.stat().st_mode)))

    # A fixture declaration does not resolve real March metadata.
    TEST_CONFIG = {
        "version":1,"record":RECORD,"archive":NAME,"published_md5":MD5,"displayed_size":"31.5 GB",
        "exact_bytes":None,"resolved_url":None,"default_mode":"dry_run","authorization_required":True,
        "data_relative":DATA_REL,"output_relative":OUTPUT_REL,"max_requests":5,
        "max_storage_bytes":80_000_000_000,"reserve_free_bytes":10_000_000_000,
        "metadata_url":METADATA_URL,"ca_bundle":None}
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            super().startTest(test); production_calls.append(test.id().rsplit(".",1)[-1])
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    began=time.monotonic()
    with mock.patch.object(socket.socket,"connect",loopback_connect):
        result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    print(json.dumps({"status":"synthetic_tests_passed" if result.wasSuccessful() else "synthetic_tests_failed",
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "skipped":len(result.skipped),"tests":production_calls,"workdir":str(work),
        "loopback_HTTP_requests":len(requests_seen),"production_transfer_path_invoked":False,
        "non_loopback_socket_connections_allowed":False,"March_bytes_transferred":0,
        "seconds":time.monotonic()-began},sort_keys=True))
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    try: raise SystemExit(main())
    except (SafetyError, OSError, ValueError) as error:
        print("ACQUISITION SAFETY STOP: " + str(error), file=sys.stderr)
        raise SystemExit(2)
