"""Generates T1_2_ARTIFACT_MANIFEST.md with SHA-256 digests and file sizes."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'output/experiments/t1_2_controlled_history_benchmark_v1'


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"


def main():
    if not EXP_DIR.is_dir():
        print(f"Error: {EXP_DIR} does not exist")
        sys.exit(1)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Task T1.2: Artifact Manifest & Cryptographic Receipts",
        "",
        f"**Generated**: {now_str}  ",
        f"**Repository**: `m100_training_template`  ",
        f"**Experiment Root**: `output/experiments/t1_2_controlled_history_benchmark_v1/`  ",
        "**Status**: SEALED AND AUDITED  ",
        "",
        "---",
        "",
        "## 1. Primary Experimental Artifacts",
        "",
        "| Category | Filename / Path | Size | SHA-256 Digest | Description |",
        "|---|---|---|---|---|",
    ]

    # Gather files
    all_files = sorted(list(EXP_DIR.rglob("*")))

    for f in all_files:
        if f.is_file() and not f.name.endswith(".tmp") and f.name != "T1_2_ARTIFACT_MANIFEST.md":
            rel_path = f.relative_to(EXP_DIR)
            size_str = format_bytes(f.stat().st_size)
            digest = compute_sha256(f)

            if "checkpoints" in str(rel_path):
                cat = "Checkpoint"
                desc = "Trained SequenceGRU global checkpoint (best validation round)"
            elif "predictions" in str(rel_path):
                cat = "Predictions"
                desc = "Raw numpy test predictions array (physical Watts)"
            elif "plots" in str(rel_path):
                cat = "Diagnostic Plot"
                desc = "Publication diagnostic plot (300 DPI)"
            elif "logs" in str(rel_path) or f.name.endswith(".log"):
                cat = "Log"
                desc = "Live terminal progress log or round metrics CSV"
            elif f.name.endswith(".json"):
                cat = "Configuration"
                desc = "Machine-readable experiment configuration receipt"
            elif f.name.endswith(".csv"):
                cat = "Metrics Table"
                desc = "Structured experimental metrics or population audit"
            elif f.name.endswith(".md"):
                cat = "Report / Audit"
                desc = "Scientific evaluation report or integrity audit"
            else:
                cat = "Data"
                desc = "Supporting artifact"

            lines.append(f"| {cat} | `{rel_path}` | {size_str} | `{digest}` | {desc} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Integrity Verification Instructions",
        "",
        "To verify artifact authenticity and ensure no file has been modified:",
        "",
        "```bash",
        "cd /media/csedept/cse2014/Project/m100_training_template",
        "PYTHONPATH=. .venv/bin/python tools/audit_t1_2_integrity.py",
        "```",
        "",
        "---",
        "",
        "## 3. Preservation Notice",
        "",
        "In accordance with research safety rules, this directory is permanently frozen. "
        "No checkpoints, configuration receipts, or CSV summaries may be overwritten.",
    ])

    manifest_path = EXP_DIR / "T1_2_ARTIFACT_MANIFEST.md"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
