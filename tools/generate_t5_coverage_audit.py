"""
Generate T5_WEATHER_COVERAGE.csv for Task T5 Environmental Ablation.

Audits exact availability, missingness, finite percentages, and physical distributions
for outdoor temperature and humidity across all operational splits.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from src.temporal_generalization_v1 import ExtendedTelemetryDataset


def generate_weather_coverage_csv(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "T5_WEATHER_COVERAGE.csv"

    print("Loading extended telemetry dataset for weather coverage audit...", flush=True)
    data = ExtendedTelemetryDataset()
    feat = data.features  # (979, 12960, 13)
    n_nodes = feat.shape[0]

    splits = [
        {
            "split_name": "train",
            "description": "Candidate 1 Training Split",
            "start_utc": "2022-01-01T00:00:00Z",
            "end_utc": "2022-01-22T00:00:00Z",
            "bin_start": 0,
            "bin_end": 3024,
            "usable_windows": 394100,
            "usable_nodes": 979,
        },
        {
            "split_name": "val",
            "description": "Candidate 1 Validation Split",
            "start_utc": "2022-01-22T00:00:00Z",
            "end_utc": "2022-01-28T00:00:00Z",
            "bin_start": 3024,
            "bin_end": 3888,
            "usable_windows": 157484,
            "usable_nodes": 979,
        },
        {
            "split_name": "test_t1",
            "description": "Canonical T1 Held-Out Test",
            "start_utc": "2022-01-28T00:00:00Z",
            "end_utc": "2022-02-04T00:00:00Z",
            "bin_start": 3888,
            "bin_end": 4896,
            "usable_windows": 243138,
            "usable_nodes": 978,
        },
        {
            "split_name": "test_q1",
            "description": "Extended Q1 Temporal Evaluation (Overlapping)",
            "start_utc": "2022-01-28T00:00:00Z",
            "end_utc": "2022-04-01T00:00:00Z",
            "bin_start": 3888,
            "bin_end": 12960,
            "usable_windows": 3609285,
            "usable_nodes": 979,
        },
        {
            "split_name": "test_march",
            "description": "March Temporal Evaluation Window (Overlapping)",
            "start_utc": "2022-03-01T00:00:00Z",
            "end_utc": "2022-04-01T00:00:00Z",
            "bin_start": 8496,
            "bin_end": 12960,
            "usable_windows": 2229042,
            "usable_nodes": 976,
        },
    ]

    rows = []
    for sp in splits:
        s = sp["bin_start"]
        e = sp["bin_end"]
        n_bins = e - s
        total_eval_cells = n_nodes * n_bins

        # Node 0 represents facility-wide weather station telemetry
        t_sub = feat[0, s:e, 7]
        h_sub = feat[0, s:e, 8]

        t_finite = np.isfinite(t_sub)
        t_finite_bins = int(np.sum(t_finite))
        t_missing_bins = int(n_bins - t_finite_bins)
        t_pct = (t_finite_bins / n_bins) * 100.0
        t_mean = float(np.nanmean(t_sub))
        t_std = float(np.nanstd(t_sub))
        t_min = float(np.nanmin(t_sub))
        t_max = float(np.nanmax(t_sub))

        h_finite = np.isfinite(h_sub)
        h_finite_bins = int(np.sum(h_finite))
        h_missing_bins = int(n_bins - h_finite_bins)
        h_pct = (h_finite_bins / n_bins) * 100.0
        h_mean = float(np.nanmean(h_sub))
        h_std = float(np.nanstd(h_sub))
        h_min = float(np.nanmin(h_sub))
        h_max = float(np.nanmax(h_sub))

        rows.append({
            "split_name": sp["split_name"],
            "description": sp["description"],
            "start_utc": sp["start_utc"],
            "end_utc": sp["end_utc"],
            "total_bins": n_bins,
            "usable_nodes": sp["usable_nodes"],
            "usable_windows": sp["usable_windows"],
            "window_reduction_due_to_weather_pct": 0.0,
            "outdoor_temp_coverage_pct": round(t_pct, 4),
            "outdoor_temp_missing_bins": t_missing_bins,
            "outdoor_temp_mean_c": round(t_mean, 2),
            "outdoor_temp_std_c": round(t_std, 2),
            "outdoor_temp_min_c": round(t_min, 2),
            "outdoor_temp_max_c": round(t_max, 2),
            "outdoor_humidity_coverage_pct": round(h_pct, 4),
            "outdoor_humidity_missing_bins": h_missing_bins,
            "outdoor_humidity_mean_pct": round(h_mean, 2),
            "outdoor_humidity_std_pct": round(h_std, 2),
            "outdoor_humidity_min_pct": round(h_min, 2),
            "outdoor_humidity_max_pct": round(h_max, 2),
            "causal_ff30_applied": "YES (max 3 bins = 30m, split resets)",
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {out_csv} ({len(rows)} split rows).", flush=True)


if __name__ == "__main__":
    t5_dir = ROOT / "output/experiments/t5_weather_ablation_v1"
    generate_weather_coverage_csv(t5_dir)
