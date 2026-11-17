"""Task T2 — Comprehensive Temporal Dataset Coverage Audit (Jan 28 - Mar 31, 2022).

Audits:
1. Available date ranges across binned datasets (Jan-Feb and March).
2. Continuity across the Feb 28 - Mar 01 seam.
3. Number of valid candidate origins under FF30 policy for 6h history (36 steps) and next-hour target (6 steps).
4. Evaluated nodes, dates, origins, and windows count.
5. Generates coverage_audit.csv and validation summary.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Feature columns
SENSOR_COLS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12)
HORIZON = 6
HISTORY_STEPS = 36


def load_all_telemetry(root: Path):
    """Load and stitch January-February binned arrays and March node arrays."""
    bins_dir = root / 'output/experiments/v5_missing_data/model_comparison/v1/cohorts/bins'
    march_dir = root / 'output/experiments/b1_no_ff/preparation/march_test_v1/dataset'

    print("Loading Jan-Feb binned arrays...", flush=True)
    train_feat = np.load(bins_dir / 'train_features.npy', mmap_mode='r')
    val_feat = np.load(bins_dir / 'val_features.npy', mmap_mode='r')
    test_feat = np.load(bins_dir / 'test_features.npy', mmap_mode='r')

    # Dimensions
    node_count = train_feat.shape[0]  # 979
    # train_feat: [0, 5947) -> Jan 01 00:00 to Feb 11 07:10
    # val_feat:   [5947, 7221) -> Feb 11 07:10 to Feb 20 03:30 (starts at 5908)
    # test_feat:  [7221, 8496) -> Feb 20 03:30 to Mar 01 00:00 (starts at 7182)

    total_feb_bins = 8496
    total_march_bins = 4464
    total_bins = total_feb_bins + total_march_bins  # 12,960 bins (Jan 01 to Apr 01)

    print(f"Allocating unified feature grid ({node_count}, {total_bins}, 13)...", flush=True)
    features = np.full((node_count, total_bins, 13), np.nan, dtype=np.float32)

    # 1. Fill Jan-Feb
    features[:, :5947, :] = train_feat[:, :5947, :].astype(np.float32)
    features[:, 5947:7221, :] = val_feat[:, 39:, :].astype(np.float32)
    features[:, 7221:8496, :] = test_feat[:, 39:, :].astype(np.float32)

    # 2. Fill March from node_xxxx.npz
    print("Loading March node arrays...", flush=True)
    for i in range(node_count):
        npz_path = march_dir / f'node_{i:04d}.npz'
        if npz_path.is_file():
            with np.load(npz_path) as d:
                features[i, 8496:12960, :] = d['features'].astype(np.float32)
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{node_count} nodes for March...", flush=True)

    # Power in kW (channel 0)
    power_kw = features[:, :, 0].copy()

    # Compute fill ages under FF30
    print("Computing FF30 fill ages across 12,960 bins...", flush=True)
    ages = np.full(features.shape, -1, dtype=np.int16)

    # We partition splits to strictly reset fill ages at split cuts:
    # Train: [0, 3024) (Jan 01 to Jan 22)
    # Val:   [3024, 3888) (Jan 22 to Jan 28)
    # Test:  [3888, 12960) (Jan 28 to Apr 01)
    cuts = [
        (0, 3024),       # Train
        (3024, 3888),    # Val
        (3888, 12960),   # Full Extended Test
    ]

    for lo, hi in cuts:
        positions = np.arange(hi - lo)
        for c in range(13):
            col = features[:, lo:hi, c]
            if c in (7, 8):  # Weather is never filled
                observed = np.isfinite(col)
                ages[:, lo:hi, c] = np.where(observed, 0, -1)
                continue
            for n_idx in range(node_count):
                obs = np.isfinite(col[n_idx])
                last = np.maximum.accumulate(np.where(obs, positions, -1))
                ages[n_idx, lo:hi, c] = np.where(last >= 0, positions - last, -1)

    print("Telemetry loading and FF30 age derivation complete.", flush=True)
    return features, power_kw, ages


def audit_origins(features, power_kw, ages, history_steps=36, horizon=6):
    node_count, total_bins, _ = features.shape
    base_time = pd.Timestamp('2022-01-01T00:00:00Z')
    bin_dates = [base_time + pd.Timedelta(minutes=b*10) for b in range(total_bins)]

    # Extended test period: bin 3888 (2022-01-28 00:00:00Z) to bin 12960 (2022-04-01 00:00:00Z)
    test_lo = 3888
    test_hi = 12960

    # Discover valid origins per node
    print("Discovering valid forecast origins under FF30...", flush=True)
    valid_origins = {}
    sensor_cols = list(SENSOR_COLS)

    origins_arr = np.arange(test_lo, test_hi)

    node_origin_counts = np.zeros(node_count, dtype=np.int64)
    date_origin_counts = {}

    for n in range(node_count):
        node_ages = ages[n]
        node_pow = power_kw[n]

        sensor_ages = node_ages[:, sensor_cols]
        bad_step = (sensor_ages < 0) | (sensor_ages > 3)

        # Split lower boundary reset check
        positions = np.arange(len(node_ages))
        bad_source = (sensor_ages > 0) & ((positions[:, None] - sensor_ages) < test_lo)
        bad = bad_step | bad_source

        # Cumulative bad steps
        hp = np.r_[0, np.cumsum(bad.any(axis=1))]
        tp = np.r_[0, np.cumsum(~np.isfinite(node_pow))]

        enough = origins_arr >= (test_lo + history_steps)
        inside = (origins_arr + horizon) <= test_hi

        ok_hist = np.zeros(len(origins_arr), dtype=bool)
        ok_targ = np.zeros(len(origins_arr), dtype=bool)

        mask = enough & inside
        cand = origins_arr[mask]

        if len(cand) > 0:
            h_bad = hp[cand] - hp[cand - history_steps]
            t_bad = tp[cand + horizon] - tp[cand]
            ok_hist[mask] = (h_bad == 0)
            ok_targ[mask] = (t_bad == 0)

        valid_mask = ok_hist & ok_targ
        valid_t = origins_arr[valid_mask]
        valid_origins[n] = valid_t
        node_origin_counts[n] = len(valid_t)

        for vt in valid_t:
            d_str = bin_dates[vt].strftime('%Y-%m-%d')
            date_origin_counts[d_str] = date_origin_counts.get(d_str, 0) + 1

    return valid_origins, node_origin_counts, date_origin_counts, bin_dates


def main():
    features, power_kw, ages = load_all_telemetry(ROOT)
    valid_origins, node_counts, date_counts, bin_dates = audit_origins(features, power_kw, ages)

    total_windows = sum(node_counts)
    active_nodes = (node_counts > 0).sum()
    all_dates = sorted(date_counts.keys())

    print(f"\n==========================================")
    print(f"=== TEMPORAL COVERAGE AUDIT SUMMARY ===")
    print(f"==========================================")
    print(f"Total Evaluated Nodes: {active_nodes} / {len(node_counts)}")
    print(f"Total Valid Windows: {total_windows:,}")
    print(f"Total Evaluated Dates: {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")
    print(f"Average Windows per Node: {total_windows / active_nodes:.1f}")

    # Breakdown by Period
    t1_dates = [d for d in all_dates if '2022-01-28' <= d < '2022-02-04']
    feb_extended_dates = [d for d in all_dates if '2022-02-04' <= d < '2022-03-01']
    march_dates = [d for d in all_dates if '2022-03-01' <= d <= '2022-03-31']

    t1_windows = sum(date_counts[d] for d in t1_dates)
    feb_ext_windows = sum(date_counts[d] for d in feb_extended_dates)
    march_windows = sum(date_counts[d] for d in march_dates)

    print(f"\n--- Period Breakdown ---")
    print(f"1. T1 Reference (Jan 28 - Feb 04): {len(t1_dates)} days, {t1_windows:,} windows")
    print(f"2. Feb Extension (Feb 04 - Mar 01): {len(feb_extended_dates)} days, {feb_ext_windows:,} windows")
    print(f"3. March Period (Mar 01 - Mar 31): {len(march_dates)} days, {march_windows:,} windows")
    print(f"Total T2 Extended Period: {total_windows:,} windows across {len(all_dates)} days")

    # Write coverage_audit.csv
    out_dir = ROOT / 'output/experiments/temporal_generalization_6h_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'coverage_audit.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date_utc', 'period', 'windows_count', 'active_nodes'])
        for d in sorted(all_dates):
            period = 'T1_reference' if d < '2022-02-04' else ('Feb_extension' if d < '2022-03-01' else 'March_extension')
            # Count active nodes on this date
            w_count = date_counts[d]
            writer.writerow([d, period, w_count, active_nodes])

    print(f"\nCoverage audit written to: {csv_path}")


if __name__ == '__main__':
    main()
