"""Durable same-directory replacement and a single-writer process lock."""
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path


def sync_directory(directory: Path):
    if os.name == "posix":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_write(path: Path, writer):
    """Write, flush and replace; the old target survives an incomplete write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path):
    def copy(handle):
        with source.open("rb") as old:
            shutil.copyfileobj(old, handle)
    atomic_write(destination, copy)


def write_json(path: Path, payload):
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda handle: handle.write(data))


class RunLock(AbstractContextManager):
    """OS-held lock; process death releases it, even when its file remains."""
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0, os.SEEK_END)
                if self.handle.tell() == 0:
                    self.handle.write(b"\0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Another training process holds the output lock: {self.path}") from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode())
        self.handle.flush()
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
        # Keep the lock inode. Removing it could let concurrent processes lock different files.
        return False
