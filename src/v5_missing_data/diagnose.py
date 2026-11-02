"""Read-only v5 missing-data coverage diagnostic; no training or imputation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.history_coverage_v4 import audit_node, preflight
from src.history_data_v4 import (
    ReferenceData, build_binned_grid, node_features
)
from src.v5_missing_data.eligibility import FEATURES

ROOT = Path(__file__).resolve().parents[2]
HISTORIES = (6, 36, 144)
MASKED = {7, 8}
HORIZON = 6


def masked_coverage(features, boundaries, history):
    """Vectorized eligibility; missing masked weather is ignored."""
    power = features[:, 0]
    n = len(power)

    bad = ~np.isfinite(features)
    bad[:, list(MASKED)] = False
    bad_prefix = np.vstack([
        np.zeros((1, features.shape[1]), dtype=np.int64),
        np.cumsum(bad, axis=0, dtype=np.int64),
    ])
    bad_power = ~np.isfinite(power)
    target_prefix = np.r_[0, np.cumsum(bad_power, dtype=np.int64)]

    results = {}
    for split, (lower, upper) in boundaries.items():
        origins = np.arange(lower, upper, dtype=np.int64)
        enough_history = origins >= history
        target_inside = origins + HORIZON <= upper
        valid = enough_history & target_inside

        candidates = origins[valid]
        if len(candidates):
            valid_hist = (
                bad_prefix[candidates] == bad_prefix[candidates - history]
            ).all(axis=1)
            valid_target = (
                target_prefix[candidates + HORIZON]
                == target_prefix[candidates]
            )
            accepted = candidates[valid_hist & valid_target]
        else:
            accepted = candidates

        results[split] = {
            "candidate_origins": int(len(origins)),
            "accepted": int(len(accepted)),
            "history_too_short": int((~enough_history).sum()),
            "target_crosses_split_end": int((enough_history & ~target_inside).sum()),
            "invalid_history_or_target": int(
                len(candidates) - len(accepted)
            ),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    reference_dir = Path(args.reference_data_dir).resolve()
    raw_root = Path(args.raw_data_dir).resolve()
    output = Path(args.output_dir).absolute()
    allowed = ROOT / "output/experiments/v5_missing_data/diagnostic"
    if (not output.resolve().is_relative_to(allowed.resolve())
            or output.resolve() == allowed.resolve()
            or output.is_symlink()):
        raise ValueError("Output must be a new subdirectory under the v5 diagnostic directory")
    if output.exists():
        raise ValueError("Output already exists; refusing to overwrite it")

    config = json.loads(config_path.read_text())
    checked = preflight(ROOT, config, reference_dir, raw_root)
    contract, meta, edges, boundaries, preflight_info = checked

    output.mkdir(parents=True, exist_ok=False)
    reference = ReferenceData(reference_dir, contract, meta)

    import tempfile
    with tempfile.TemporaryDirectory(prefix=".v5-scratch-", dir=output) as tmp:
        grid, weather, quality = build_binned_grid(
            raw_root, config["raw_files"], reference.names, edges, meta,
            Path(tmp), config["reading_limits"]
        )

        original_totals = dict.fromkeys(boundaries, 0)
        masked_totals = {
            str(h): dict.fromkeys(boundaries, 0) for h in HISTORIES
        }
        feature_missing = {
            split: dict.fromkeys(FEATURES, 0) for split in boundaries
        }
        rows = []

        for index, name in enumerate(reference.names):
            features = node_features(grid[index], weather, edges)

            # Keep the original v4 selection and byte-exact baseline check.
            old_selected, old_targets, _ = audit_node(
                features, edges, boundaries, histories=(6,)
            )
            replay_counts = reference.verify_node(
                index, features, edges, old_selected[6], old_targets
            )
            for split, count in replay_counts.items():
                original_totals[split] += count

            # Count missing values by feature and chronological split.
            for split, (lo, hi) in boundaries.items():
                section = features[lo:hi]
                for j, feature in enumerate(FEATURES):
                    feature_missing[split][feature] += int(
                        (~np.isfinite(section[:, j])).sum()
                    )

            for history in HISTORIES:
                coverage = masked_coverage(features, boundaries, history)
                for split, info in coverage.items():
                    masked_totals[str(history)][split] += info["accepted"]
                    rows.append({
                        "node": str(name),
                        "node_index": index,
                        "history_steps": history,
                        "history_hours": history / 6,
                        "split": split,
                        **info,
                    })

    with (output / "masked_coverage_by_node.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "node", "node_index", "history_steps", "history_hours",
            "split", "candidate_origins", "accepted", "history_too_short",
            "target_crosses_split_end", "invalid_history_or_target"
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "complete",
        "purpose": "coverage diagnostic only",
        "training_or_imputation_or_metric_evaluation": False,
        "weather_policy": "channels 7 and 8 ignored for history eligibility",
        "original_reference_replay": {
            "status": "exact_match",
            "counts": original_totals,
        },
        "masked_weather_eligible_windows": masked_totals,
        "missing_bins_by_feature_and_split": feature_missing,
        "histories_steps": list(HISTORIES),
        "history_hours": [h / 6 for h in HISTORIES],
        "target": "observed future node power over six 10-minute bins",
        "preflight": preflight_info,
        "source_quality": quality,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "original_reference_replay": original_totals,
        "masked_weather_eligible_windows": masked_totals,
        "missing_bins_by_feature_and_split": feature_missing,
        "training_or_imputation_or_metric_evaluation": False,
    }, indent=2))


if __name__ == "__main__":
    main()
