"""Exclusive new synthetic artifacts, restricted to approved Phase 1.3 output."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]

def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()

def checked_path(path):
    p=Path(path).absolute()
    allowed=ROOT/"output/phase1_3"
    if not p.resolve().is_relative_to(allowed.resolve()) or p==allowed:
        raise ValueError("Output must be a new child of output/phase1_3")
    for parent in (p,*p.parents):
        if parent.is_symlink():
            raise ValueError("Symlink output rejected")
        if parent==allowed:
            break
    return p

def write_json(path,document):
    p=checked_path(path)
    payload=json.dumps(document,indent=2,sort_keys=True,allow_nan=False)+"\n"
    # Parents must already be explicitly created in the new namespace.
    with p.open("x",encoding="utf-8") as handle:
        handle.write(payload)
    return digest_file(p)
