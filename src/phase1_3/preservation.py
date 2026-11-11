"""Read-only byte/stat comparison. Never repairs historical evidence."""
from pathlib import Path
import os
import stat
import json
import hashlib

EXCLUDED={".git",".venv","__pycache__"}

def inventory(root):
    root=Path(root);result={}
    for base,dirs,files in os.walk(root,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED)
        for n in sorted(files):
            p=Path(base,n);s=p.lstat()
            if not stat.S_ISREG(s.st_mode):
                raise ValueError("Nonregular path: "+str(p))
            with p.open("rb") as handle:
                h=hashlib.file_digest(handle,"sha256").hexdigest()
            a=p.stat()
            if (s.st_size,s.st_mtime_ns,s.st_mode,s.st_ino)!=(a.st_size,a.st_mtime_ns,a.st_mode,a.st_ino):
                raise ValueError("File changed during hash: "+str(p))
            result[str(p.relative_to(root))]={"sha256":h,"bytes":s.st_size,
                "mtime_ns":s.st_mtime_ns,"mode":stat.S_IMODE(s.st_mode)}
    return result

def compare_records(before,after,allowlist):
    mismatch=[]
    for p,e in before.items():
        if p not in after:
            mismatch.append({"path":p,"field":"missing"})
        else:
            for k in ("sha256","bytes","mtime_ns","mode"):
                if e[k]!=after[p][k]:
                    mismatch.append({"path":p,"field":k,"expected":e[k],"actual":after[p][k]})
    added=sorted(set(after)-set(before))
    for p in added:
        if p not in allowlist:
            mismatch.append({"path":p,"field":"unexpected_new_file"})
    return {"preexisting_checked":len(before),"new_files":added,"mismatches":mismatch,
            "unchanged":not mismatch}

def verify(root):
    root=Path(root)
    before=json.loads((root/"output/phase1_3/preservation_before.json").read_text())
    allowed=json.loads((root/"output/phase1_3/output_allowlist.json").read_text())["exact_new_files"]
    after=inventory(root)
    result=compare_records(before["files"],after,allowed)
    for rel in ("src/phase1_3","tests/phase1_3","configs/phase1_3","output/phase1_3"):
        for p in (root/rel).rglob("*"):
            if p.is_file() and str(p.relative_to(root)) not in allowed:
                result["mismatches"].append({"path":str(p.relative_to(root)),"field":"unallowlisted_new_path"})
    result["unchanged"]=not result["mismatches"]
    result.update(old_inventory_digest=before["inventory_digest"],atime_compared=False,
        excluded_directories=sorted(EXCLUDED),real_data_parsed=False,
        historical_preflight_result="FAILED_UNCHANGED")
    return result
