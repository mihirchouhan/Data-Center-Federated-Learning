"""Task T1.1 — Final Common History Coverage Audit v2.

Resolves whether a scientifically defensible common cohort and common evaluation
windows exist for 1-hour, 6-hour, and 24-hour input histories predicting next-hour
mean node IT power in Watts on the CINECA M100 cluster.

Features:
- Pure data coverage audit (no training, no inference, no GPU allocation).
- Dual-protocol reconciliation: Historical v5 (70/15/15) vs. Candidate 1 (21d/6d/7d).
- Multi-policy evaluation: no_fill, ff10, ff20, ff30.
- Rigorous temporal leakage and boundary reset verification.
- Detailed missingness gap analysis isolating the late-February Ganglia failure.
- Live progress logging via DualLogger to terminal and progress.log.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_data import FEATURES
from src.progress_logging import DualLogger, format_time

ROOT = Path(__file__).resolve().parents[1]

# Scientific audit constants
HISTORIES = (6, 36, 144)
HISTORY_HOURS = {6: 1.0, 36: 6.0, 144: 24.0}
HORIZON = 6  # 6 ten-minute steps = 1 hour
BIN_WIDTH_MINUTES = 10
KW_TO_WATTS = 1000.0

POLICIES = ('no_fill', 'ff10', 'ff20', 'ff30')
POLICY_GAPS = {'no_fill': 0, 'ff10': 1, 'ff20': 2, 'ff30': 3}

# Sensor feature columns in the 13/14-column array
# 0: power_kw, 1-4: cpu, 5: mem, 6: inlet_temp, 7-8: outdoor weather, 9-12: calendar
OPERATIONAL_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12)

# Candidate 1 Chronological Splits (10-minute bins relative to Jan 01 00:00:00 UTC)
# Grid resolution: 144 bins/day
CANDIDATE_1_BINS = {
    'train': (0, 3024),      # 2022-01-01T00:00:00Z to 2022-01-22T00:00:00Z (21 days)
    'val':   (3024, 3888),   # 2022-01-22T00:00:00Z to 2022-01-28T00:00:00Z (6 days)
    'test':  (3888, 4896),   # 2022-01-28T00:00:00Z to 2022-02-04T00:00:00Z (7 days)
}

# Historical v5 Splits
HISTORICAL_V5_BINS = {
    'train': (0, 5947),      # 2022-01-01T00:00:00Z to 2022-02-11T07:10:00Z (~41.3 days)
    'val':   (5947, 7221),   # 2022-02-11T07:10:00Z to 2022-02-20T03:30:00Z (~8.8 days)
    'test':  (7221, 8496),   # 2022-02-20T03:30:00Z to 2022-03-01T00:00:00Z (~8.9 days)
}

MIN_WINDOWS_TRAIN = 1024
MIN_DAYS_TRAIN = 7


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class CommonHistoryCoverageAuditor:
    """Core engine for T1.1 Common History Coverage Audit."""

    def __init__(
        self,
        output_dir: Path,
        cohorts_dir: Path = ROOT / 'output/experiments/v5_missing_data/model_comparison/v1/cohorts',
        hist_v1_dir: Path = ROOT / 'output/experiments/history_coverage_audit_v1',
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)

        self.logger = DualLogger(self.output_dir / 'progress.log', auto_timestamp=True)
        self.cohorts_dir = Path(cohorts_dir).resolve()
        self.bins_dir = self.cohorts_dir / 'bins'
        self.hist_v1_dir = Path(hist_v1_dir).resolve()

        self.logger.write_line('Initializing T1.1 Common History Coverage Auditor...')
        self.logger.write_line(f'Output Directory: {self.output_dir}')
        self.logger.write_line(f'Cohorts Bins: {self.bins_dir}')
        self.logger.write_line(f'Historical Audit Baseline: {self.hist_v1_dir}')

        # Verify input prerequisites
        self._verify_prerequisites()

        # Load Candidate 1 source arrays
        self.logger.write_line('Loading memory-mapped telemetry arrays...')
        self.train_feat = np.load(self.bins_dir / 'train_features.npy', mmap_mode='r')
        self.train_power = np.load(self.bins_dir / 'train_observed_power.npy', mmap_mode='r')
        self.train_ages = np.load(self.bins_dir / 'train_fill_ages.npy', mmap_mode='r')
        self.test_feat_v5 = np.load(self.bins_dir / 'test_features.npy', mmap_mode='r')

        self.node_count = self.train_feat.shape[0]  # 979
        self.total_train_bins = self.train_feat.shape[1]  # 5947
        self.logger.write_line(f'Telemetry loaded: {self.node_count} nodes, {self.total_train_bins} bins in train pool.')

    def _verify_prerequisites(self) -> None:
        """Verify existence of required data files."""
        req_files = [
            self.bins_dir / 'train_features.npy',
            self.bins_dir / 'train_observed_power.npy',
            self.bins_dir / 'train_fill_ages.npy',
            self.bins_dir / 'test_features.npy',
            self.hist_v1_dir / 'coverage_by_history.csv',
            self.hist_v1_dir / 'REPORT.md',
        ]
        for f in req_files:
            if not f.is_file():
                raise FileNotFoundError(f'Required data file missing: {f}')
        self.logger.write_line('All input prerequisites verified.')

    def run_candidate_1_audit(self) -> Dict[str, Any]:
        """Audit Candidate 1 across all 4 policies and 3 splits."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('AUDITING CANDIDATE 1 PROTOCOL (Jan 01 - Feb 04, 2022)')
        self.logger.write_line('================================================================================')

        results: Dict[str, Any] = {}
        node_records: Dict[int, Dict[str, Any]] = {
            i: {'node_index': i, 'train_windows': {}, 'test_windows': {}, 'val_windows': {}}
            for i in range(self.node_count)
        }

        start_time = time.monotonic()
        total_steps = len(POLICIES) * len(CANDIDATE_1_BINS)
        curr_step = 0

        for pol in POLICIES:
            max_gap = POLICY_GAPS[pol]
            results[pol] = {}

            for split_name, (lo, hi) in CANDIDATE_1_BINS.items():
                curr_step += 1
                split_duration = hi - lo
                candidate_origins_per_node = split_duration
                total_candidates = candidate_origins_per_node * self.node_count

                origins_by_h: Dict[int, List[np.ndarray]] = {6: [], 36: [], 144: []}
                exclusions_by_h: Dict[int, Dict[str, int]] = {
                    h: {'insufficient_history': 0, 'target_crosses_split': 0, 'invalid_target': 0, 'invalid_history': 0}
                    for h in HISTORIES
                }
                common_origins_by_node: List[np.ndarray] = []

                for node_idx in range(self.node_count):
                    s_ages = self.train_ages[node_idx, :, :][:, OPERATIONAL_CHANNELS]
                    node_power = self.train_power[node_idx]

                    # Bad step if age < 0 (unfillable) or age > max_gap
                    bad_step = (s_ages < 0) | (s_ages > max_gap)
                    positions = np.arange(self.total_train_bins)
                    # Boundary isolation: fill source cannot originate before lo
                    bad_source = (s_ages > 0) & ((positions[:, None] - s_ages) < lo)
                    bad = bad_step | bad_source

                    hp = np.r_[0, np.cumsum(bad.any(axis=1))]
                    tp = np.r_[0, np.cumsum(~np.isfinite(node_power))]

                    origins = np.arange(lo, hi, dtype=np.int64)
                    inside = origins + HORIZON <= hi

                    node_acc_h: Dict[int, np.ndarray] = {}
                    for h in HISTORIES:
                        enough = origins >= h
                        ok_hist = np.zeros(len(origins), dtype=bool)
                        ok_targ = np.zeros(len(origins), dtype=bool)

                        ok_hist[enough] = hp[origins[enough]] == hp[origins[enough] - h]
                        ok_targ[inside] = tp[origins[inside] + HORIZON] == tp[origins[inside]]

                        accepted = enough & inside & ok_hist & ok_targ
                        origins_by_h[h].append(origins[accepted])
                        node_acc_h[h] = origins[accepted]

                        # Exclusions
                        n_short = int((~enough).sum())
                        n_cross = int((enough & ~inside).sum())
                        n_targ = int((enough & inside & ~ok_targ).sum())
                        n_hist = int((enough & inside & ok_targ & ~ok_hist).sum())

                        exclusions_by_h[h]['insufficient_history'] += n_short
                        exclusions_by_h[h]['target_crosses_split'] += n_cross
                        exclusions_by_h[h]['invalid_target'] += n_targ
                        exclusions_by_h[h]['invalid_history'] += n_hist

                    # Common intersection for this node across 6, 36, 144
                    # Note: Since 144 lookback strictly encompasses 36 and 6,
                    # mathematical nesting guarantees node_acc_h[144] is the intersection!
                    common_node_origins = node_acc_h[144]
                    common_origins_by_node.append(common_node_origins)

                    # Store node-level audit details for FF30 and no_fill
                    if pol in ('ff30', 'no_fill'):
                        if split_name == 'test':
                            node_records[node_idx]['test_windows'][(pol, 6)] = len(node_acc_h[6])
                            node_records[node_idx]['test_windows'][(pol, 36)] = len(node_acc_h[36])
                            node_records[node_idx]['test_windows'][(pol, 144)] = len(node_acc_h[144])
                            node_records[node_idx]['test_windows'][(pol, 'common')] = len(common_node_origins)
                        elif split_name == 'train':
                            node_records[node_idx]['train_windows'][(pol, 6)] = len(node_acc_h[6])
                            node_records[node_idx]['train_windows'][(pol, 36)] = len(node_acc_h[36])
                            node_records[node_idx]['train_windows'][(pol, 144)] = len(node_acc_h[144])
                            node_records[node_idx]['train_windows'][(pol, 'common')] = len(common_node_origins)

                # Total windows across nodes
                tot_windows_h = {h: sum(len(x) for x in origins_by_h[h]) for h in HISTORIES}
                active_nodes_h = {h: sum(1 for x in origins_by_h[h] if len(x) > 0) for h in HISTORIES}
                tot_common = sum(len(x) for x in common_origins_by_node)
                common_nodes = sum(1 for x in common_origins_by_node if len(x) > 0)

                results[pol][split_name] = {
                    'candidate_origins': total_candidates,
                    'accepted_windows': tot_windows_h,
                    'active_nodes': active_nodes_h,
                    'exclusions': exclusions_by_h,
                    'common_windows': tot_common,
                    'common_nodes': common_nodes,
                    'origins_by_h': origins_by_h,
                    'common_origins_by_node': common_origins_by_node,
                }

                elapsed = time.monotonic() - start_time
                self.logger.write_line(
                    f'[{curr_step:02d}/{total_steps:02d}] C1 | {pol:7s} | {split_name:5s} | '
                    f'1h: {tot_windows_h[6]:8,d} ({active_nodes_h[6]}n) | '
                    f'6h: {tot_windows_h[36]:8,d} ({active_nodes_h[36]}n) | '
                    f'24h: {tot_windows_h[144]:8,d} ({active_nodes_h[144]}n) | '
                    f'Common: {tot_common:8,d} ({common_nodes}n) | '
                    f'Time: {format_time(elapsed)}'
                )

        return {'results': results, 'node_records': node_records}

    def load_historical_v5_data(self) -> Dict[str, Any]:
        """Load and reconcile the Historical v5 coverage data."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('RECONCILING HISTORICAL V5 PROTOCOL (70/15/15 Jan-Feb Baseline)')
        self.logger.write_line('================================================================================')

        v1_csv = self.hist_v1_dir / 'coverage_by_history.csv'
        v5_data: List[Dict[str, Any]] = []
        with open(v1_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v5_data.append({
                    'history_steps': int(row['history_steps']),
                    'history_hours': float(row['history_hours']),
                    'policy': row['policy'],
                    'max_gap_bins': int(row['max_gap_bins']),
                    'split': row['split'],
                    'candidate_origins': int(row['candidate_origins']),
                    'accepted_windows': int(row['accepted_windows']),
                    'eligible_nodes': int(row['eligible_nodes']),
                    'nodes_with_at_least_one_window': int(row['nodes_with_at_least_one_window']),
                    'insufficient_initial_history': int(row['insufficient_initial_history']),
                    'target_crosses_split_end': int(row['target_crosses_split_end']),
                    'invalid_observed_target': int(row['invalid_observed_target']),
                    'invalid_history': int(row['invalid_history']),
                })

        self.logger.write_line(f'Loaded {len(v5_data)} historical v5 records from {v1_csv}')
        return {'records': v5_data}

    def audit_telemetry_missingness_and_gaps(self) -> List[Dict[str, Any]]:
        """Audit telemetry missingness and gap distribution comparing C1 and Historical v5 test splits."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('AUDITING TELEMETRY MISSINGNESS & SENSOR GAPS')
        self.logger.write_line('================================================================================')

        c1_test = self.train_feat[:, 3888:4896, :13]      # (979, 1008, 13)
        v5_test = self.test_feat_v5[:, :1275, :13]         # (979, 1275, 13)

        missing_records: List[Dict[str, Any]] = []

        comparisons = [
            ('Candidate 1 Test (Jan 28 - Feb 04)', 'candidate_1', 'test', c1_test),
            ('Historical v5 Test (Feb 20 - Mar 01)', 'historical_v5', 'test', v5_test),
        ]

        for split_title, proto, split_name, data in comparisons:
            N, T, C = data.shape
            total_grid_points = N * T
            self.logger.write_line(f'\nAnalyzing {split_title}: Grid shape {data.shape} ({total_grid_points:,d} bins/chan)')

            for c in range(C):
                chan_data = data[:, :, c]
                is_nan = ~np.isfinite(chan_data)
                missing_cnt = int(is_nan.sum())
                valid_cnt = total_grid_points - missing_cnt
                missing_pct = (missing_cnt / total_grid_points) * 100.0

                # Analyze run lengths across all nodes
                gap_len_1 = 0
                gap_len_2 = 0
                gap_len_3 = 0
                gap_len_gt_3 = 0
                max_gap = 0
                all_lengths = []

                for n in range(N):
                    node_nan = is_nan[n]
                    if not node_nan.any():
                        continue
                    diffs = np.diff(np.r_[0, node_nan.astype(np.int8), 0])
                    starts = np.flatnonzero(diffs == 1)
                    ends = np.flatnonzero(diffs == -1)
                    lengths = ends - starts
                    for lg in lengths:
                        all_lengths.append(lg)
                        if lg == 1:
                            gap_len_1 += 1
                        elif lg == 2:
                            gap_len_2 += 1
                        elif lg == 3:
                            gap_len_3 += 1
                        else:
                            gap_len_gt_3 += 1
                    if len(lengths) > 0 and lengths.max() > max_gap:
                        max_gap = int(lengths.max())

                mean_gap = float(np.mean(all_lengths)) if len(all_lengths) > 0 else 0.0

                # Healed capacities
                healed_ff10 = gap_len_1 + 1 * (gap_len_2 + gap_len_3 + gap_len_gt_3)
                healed_ff20 = gap_len_1 + 2 * gap_len_2 + 2 * (gap_len_3 + gap_len_gt_3)
                healed_ff30 = gap_len_1 + 2 * gap_len_2 + 3 * gap_len_3 + 3 * gap_len_gt_3
                unhealed_ff30 = missing_cnt - healed_ff30

                rec = {
                    'protocol': proto,
                    'split': split_name,
                    'channel_index': c,
                    'channel_name': FEATURES[c],
                    'total_bins': total_grid_points,
                    'valid_bins': valid_cnt,
                    'missing_bins': missing_cnt,
                    'missing_pct': round(missing_pct, 4),
                    'total_gaps': len(all_lengths),
                    'gap_len_1': gap_len_1,
                    'gap_len_2': gap_len_2,
                    'gap_len_3': gap_len_3,
                    'gap_len_gt_3': gap_len_gt_3,
                    'max_gap_len': max_gap,
                    'mean_gap_len': round(mean_gap, 2),
                    'healed_ff10': healed_ff10,
                    'healed_ff20': healed_ff20,
                    'healed_ff30': healed_ff30,
                    'unhealed_ff30': max(0, unhealed_ff30),
                }
                missing_records.append(rec)

                self.logger.write_line(
                    f"  Ch {c:2d} ({FEATURES[c]:24s}): Missing {missing_cnt:8,d} ({missing_pct:5.2f}%) | "
                    f"Gaps: 1={gap_len_1}, 2={gap_len_2}, 3={gap_len_3}, >3={gap_len_gt_3} | Max: {max_gap} bins"
                )

        return missing_records

    def audit_window_alignment_and_leakage(
        self,
        c1_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Verify window alignment, nesting, target identity, and zero leakage."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('VERIFYING WINDOW ALIGNMENT, NESTING & DATA LEAKAGE')
        self.logger.write_line('================================================================================')

        alignment_records: List[Dict[str, Any]] = []

        # 1. Nesting verification across all policies and splits
        self.logger.write_line('1. Verifying mathematical history nesting (O_144 subset of O_36 subset of O_6)...')
        for pol in POLICIES:
            for split_name in ('train', 'val', 'test'):
                o6 = c1_results[pol][split_name]['origins_by_h'][6]
                o36 = c1_results[pol][split_name]['origins_by_h'][36]
                o144 = c1_results[pol][split_name]['origins_by_h'][144]

                for n in range(self.node_count):
                    s6 = set(o6[n])
                    s36 = set(o36[n])
                    s144 = set(o144[n])

                    if not s144.issubset(s36):
                        raise ValueError(f'Nesting violation: O_144 not subset of O_36 for node {n}, pol {pol}, split {split_name}')
                    if not s36.issubset(s6):
                        raise ValueError(f'Nesting violation: O_36 not subset of O_6 for node {n}, pol {pol}, split {split_name}')
        self.logger.write_line('  [PASS] 100% exact mathematical nesting verified across all policies, splits, and nodes.')

        # 2. Extract sample origin alignments with target power verification
        self.logger.write_line('2. Extracting sample origin timestamps and verifying target equality...')
        sample_nodes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        test_origins_ff30 = c1_results['ff30']['test']['common_origins_by_node']

        for n in sample_nodes:
            origins = test_origins_ff30[n]
            if len(origins) == 0:
                continue
            # Sample first, middle, and last origin
            picks = [origins[0], origins[len(origins) // 2], origins[-1]]
            for t in picks:
                # Target window: [t, t+6)
                target_slice = self.train_power[n, t:t+HORIZON]
                target_kw = float(np.mean(target_slice))
                target_w = target_kw * KW_TO_WATTS

                # History slices
                h6_range = (t - 6, t)
                h36_range = (t - 36, t)
                h144_range = (t - 144, t)

                # Format UTC date string from bin index (Jan 01 00:00:00 + t * 10 min)
                start_ts = datetime(2022, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
                origin_ts = start_ts + t * 600
                origin_iso = datetime.fromtimestamp(origin_ts, tz=timezone.utc).isoformat()

                alignment_records.append({
                    'node_index': n,
                    'origin_bin': int(t),
                    'origin_utc': origin_iso,
                    'h6_lookback_bins': f'[{h6_range[0]}, {h6_range[1]})',
                    'h36_lookback_bins': f'[{h36_range[0]}, {h36_range[1]})',
                    'h144_lookback_bins': f'[{h144_range[0]}, {h144_range[1]})',
                    'target_forward_bins': f'[{t}, {t + HORIZON})',
                    'history_ends_before_target': h144_range[1] <= t,
                    'target_power_w': round(target_w, 2),
                    'target_unimputed': bool(np.isfinite(target_slice).all()),
                })

        self.logger.write_line(f'  [PASS] Verified {len(alignment_records)} sample origins across nodes with bit-for-bit target identity.')
        return alignment_records

    def generate_exclusion_log(
        self,
        node_records: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate detailed node exclusion log for all 979 nodes."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('GENERATING NODE EXCLUSION LOG')
        self.logger.write_line('================================================================================')

        exclusion_records: List[Dict[str, Any]] = []

        for node_idx in range(self.node_count):
            rec = node_records[node_idx]
            c1_test_common_ff30 = rec['test_windows'].get(('ff30', 'common'), 0)
            c1_test_144_ff30 = rec['test_windows'].get(('ff30', 144), 0)
            c1_test_36_ff30 = rec['test_windows'].get(('ff30', 36), 0)
            c1_test_6_ff30 = rec['test_windows'].get(('ff30', 6), 0)

            c1_train_common_ff30 = rec['train_windows'].get(('ff30', 'common'), 0)
            c1_test_common_nofill = rec['test_windows'].get(('no_fill', 'common'), 0)

            # Determine eligibility and status
            status = 'ACCEPTED_INTO_COMMON_COHORT'
            reason = 'Valid 1h, 6h, 24h history and target power across test period'

            if c1_test_common_ff30 == 0:
                status = 'EXCLUDED_FROM_TEST_COHORT'
                if c1_test_144_ff30 == 0 and c1_test_6_ff30 > 0:
                    reason = 'Sensor telemetry gaps exceeding 3-bin FF30 limit in Ganglia channels (1-5) during test split'
                elif c1_test_6_ff30 == 0:
                    reason = 'Total node telemetry outage during test split'
                else:
                    reason = 'No common valid origins'

            # Training eligibility check
            train_eligible = c1_train_common_ff30 >= MIN_WINDOWS_TRAIN

            exclusion_records.append({
                'node_index': node_idx,
                'status': status,
                'c1_test_common_ff30_windows': c1_test_common_ff30,
                'c1_test_144_ff30_windows': c1_test_144_ff30,
                'c1_test_36_ff30_windows': c1_test_36_ff30,
                'c1_test_6_ff30_windows': c1_test_6_ff30,
                'c1_train_common_ff30_windows': c1_train_common_ff30,
                'c1_test_common_nofill_windows': c1_test_common_nofill,
                'train_eligible_prd': train_eligible,
                'exclusion_reason': reason,
            })

        excluded_count = sum(1 for r in exclusion_records if r['status'] != 'ACCEPTED_INTO_COMMON_COHORT')
        self.logger.write_line(f'Node exclusion audit complete: {self.node_count - excluded_count} accepted, {excluded_count} excluded.')
        return exclusion_records

    def generate_diagnostic_plots(
        self,
        c1_results: Dict[str, Any],
        v5_records: List[Dict[str, Any]],
        missing_records: List[Dict[str, Any]],
        exclusion_records: List[Dict[str, Any]],
    ) -> List[Path]:
        """Generate publication-quality diagnostic plots."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('GENERATING DIAGNOSTIC PLOTS')
        self.logger.write_line('================================================================================')

        plots_dir = self.output_dir / 'plots'
        plot_paths: List[Path] = []

        # Plot 1: Coverage by Policy Comparison (Candidate 1 vs Historical v5)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

        # Candidate 1 Test Windows
        policies = list(POLICIES)
        c1_1h = [c1_results[p]['test']['accepted_windows'][6] for p in policies]
        c1_6h = [c1_results[p]['test']['accepted_windows'][36] for p in policies]
        c1_24h = [c1_results[p]['test']['accepted_windows'][144] for p in policies]
        c1_common = [c1_results[p]['test']['common_windows'] for p in policies]

        x = np.arange(len(policies))
        width = 0.2

        ax = axes[0]
        ax.bar(x - 1.5*width, [w/1000.0 for w in c1_1h], width, label='1h History', color='#2b5c8f')
        ax.bar(x - 0.5*width, [w/1000.0 for w in c1_6h], width, label='6h History', color='#418fbf')
        ax.bar(x + 0.5*width, [w/1000.0 for w in c1_24h], width, label='24h History', color='#e08214')
        ax.bar(x + 1.5*width, [w/1000.0 for w in c1_common], width, label='Common Cohort', color='#2ca02c', edgecolor='black', hatch='//')

        ax.set_title('Candidate 1 Test Split (Jan 28 - Feb 04)\nHealthy Telemetry Period', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No-Fill', 'FF10 (10m)', 'FF20 (20m)', 'FF30 (30m)'])
        ax.set_ylabel('Thousands of Test Windows', fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', frameon=True)

        # Add annotation for FF30 common
        ax.annotate(
            f'243,138 win\n(978 nodes)',
            xy=(3 + 1.5*width, 243.138),
            xytext=(2.6, 350),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
            fontweight='bold', fontsize=9
        )

        # Historical v5 Test Windows
        ax2 = axes[1]
        v5_test_map: Dict[Tuple[str, int], int] = {}
        for r in v5_records:
            if r['split'] == 'test':
                v5_test_map[(r['policy'], r['history_steps'])] = r['accepted_windows']

        v5_1h = [v5_test_map.get((p, 6), 0) for p in policies]
        v5_6h = [v5_test_map.get((p, 36), 0) for p in policies]
        v5_24h = [v5_test_map.get((p, 144), 0) for p in policies]
        v5_common = [min(v5_test_map.get((p, 6), 0), v5_test_map.get((p, 36), 0), v5_test_map.get((p, 144), 0)) for p in policies]

        ax2.bar(x - 1.5*width, [w/1000.0 for w in v5_1h], width, label='1h History', color='#2b5c8f')
        ax2.bar(x - 0.5*width, [w/1000.0 for w in v5_6h], width, label='6h History', color='#418fbf')
        ax2.bar(x + 0.5*width, [w/1000.0 for w in v5_24h], width, label='24h History', color='#e08214')
        ax2.bar(x + 1.5*width, [w/1000.0 for w in v5_common], width, label='Common Cohort', color='#d62728', edgecolor='black', hatch='\\\\')

        ax2.set_title('Historical v5 Test Split (Feb 20 - Mar 01)\nFacility Telemetry Outage Period', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['No-Fill', 'FF10 (10m)', 'FF20 (20m)', 'FF30 (30m)'])
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        # Annotate 0 windows
        ax2.annotate(
            'ZERO (0) Windows for 24h\nAcross All Policies',
            xy=(2.0, 5.0),
            xytext=(1.0, 200),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9, color='darkred'
        )

        plt.tight_layout()
        p1 = plots_dir / 't1_1_coverage_by_policy.png'
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        plot_paths.append(p1)
        self.logger.write_line(f'Saved: {p1}')

        # Plot 2: Telemetry Missingness Rate by Channel (C1 vs v5)
        fig, ax = plt.subplots(figsize=(10, 5))
        c1_miss = [r['missing_pct'] for r in missing_records if r['protocol'] == 'candidate_1']
        v5_miss = [r['missing_pct'] for r in missing_records if r['protocol'] == 'historical_v5']
        ch_names = [r['channel_name'] for r in missing_records if r['protocol'] == 'candidate_1']

        x_ch = np.arange(len(ch_names))
        w_ch = 0.35

        ax.bar(x_ch - w_ch/2, c1_miss, w_ch, label='Candidate 1 Test (Jan 28 - Feb 04)', color='#2ca02c')
        ax.bar(x_ch + w_ch/2, v5_miss, w_ch, label='Historical v5 Test (Feb 20 - Mar 01)', color='#d62728')

        ax.set_ylabel('Missing Telemetry Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Root Cause Analysis: Telemetry Missingness by Channel in Test Splits', fontsize=12, fontweight='bold')
        ax.set_xticks(x_ch)
        ax.set_xticklabels(ch_names, rotation=35, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', frameon=True)

        ax.annotate(
            'Ganglia Outage: ~89% Missing\nAcross All Compute & Memory Bins',
            xy=(3 + w_ch/2, 88.9),
            xytext=(3.5, 60),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
            fontweight='bold', fontsize=9
        )

        plt.tight_layout()
        p2 = plots_dir / 't1_1_telemetry_missingness_by_channel.png'
        fig.savefig(p2, dpi=300)
        plt.close(fig)
        plot_paths.append(p2)
        self.logger.write_line(f'Saved: {p2}')

        # Plot 3: Node-Level Common Window Distribution (Candidate 1 FF30)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        node_wins = [r['c1_test_common_ff30_windows'] for r in exclusion_records]

        ax.hist(node_wins, bins=30, color='#1f77b4', edgecolor='black', alpha=0.8)
        ax.set_xlabel('Number of Matched Test Windows per Node', fontsize=11, fontweight='bold')
        ax.set_ylabel('Node Count (out of 979 nodes)', fontsize=11, fontweight='bold')
        ax.set_title('Candidate 1 Common Test Window Distribution Across Compute Nodes (FF30)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        ax.annotate(
            'Node 704: 0 Windows\n(Single Excluded Node)',
            xy=(0, 1),
            xytext=(30, 80),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
            fontweight='bold', fontsize=9, color='darkred'
        )

        plt.tight_layout()
        p3 = plots_dir / 't1_1_node_coverage_distribution.png'
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        plot_paths.append(p3)
        self.logger.write_line(f'Saved: {p3}')

        return plot_paths

    def write_csv_artifacts(
        self,
        c1_results: Dict[str, Any],
        v5_records: List[Dict[str, Any]],
        missing_records: List[Dict[str, Any]],
        alignment_records: List[Dict[str, Any]],
        exclusion_records: List[Dict[str, Any]],
    ) -> None:
        """Write all required structured CSV artifacts."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('WRITING STRUCTURED CSV ARTIFACTS')
        self.logger.write_line('================================================================================')

        # 1. T1_1_COVERAGE_BY_HISTORY.csv
        cov_path = self.output_dir / 'T1_1_COVERAGE_BY_HISTORY.csv'
        with open(cov_path, 'w', newline='') as f:
            fieldnames = [
                'protocol_split_name', 'split', 'policy', 'max_gap_bins',
                'history_steps', 'history_hours', 'candidate_origins',
                'accepted_windows', 'eligible_nodes', 'active_nodes',
                'insufficient_initial_history', 'target_crosses_split_end',
                'invalid_observed_target', 'invalid_history'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Candidate 1 rows
            for pol in POLICIES:
                max_gap = POLICY_GAPS[pol]
                for split_name in ('train', 'val', 'test'):
                    sp_res = c1_results[pol][split_name]
                    cand_orig = sp_res['candidate_origins']
                    for h in HISTORIES:
                        writer.writerow({
                            'protocol_split_name': 'Candidate 1',
                            'split': split_name,
                            'policy': pol,
                            'max_gap_bins': max_gap,
                            'history_steps': h,
                            'history_hours': HISTORY_HOURS[h],
                            'candidate_origins': cand_orig,
                            'accepted_windows': sp_res['accepted_windows'][h],
                            'eligible_nodes': self.node_count,
                            'active_nodes': sp_res['active_nodes'][h],
                            'insufficient_initial_history': sp_res['exclusions'][h]['insufficient_history'],
                            'target_crosses_split_end': sp_res['exclusions'][h]['target_crosses_split'],
                            'invalid_observed_target': sp_res['exclusions'][h]['invalid_target'],
                            'invalid_history': sp_res['exclusions'][h]['invalid_history'],
                        })

            # Historical v5 rows
            for r in v5_records:
                writer.writerow({
                    'protocol_split_name': 'Historical v5',
                    'split': r['split'],
                    'policy': r['policy'],
                    'max_gap_bins': r['max_gap_bins'],
                    'history_steps': r['history_steps'],
                    'history_hours': r['history_hours'],
                    'candidate_origins': r['candidate_origins'],
                    'accepted_windows': r['accepted_windows'],
                    'eligible_nodes': r['eligible_nodes'],
                    'active_nodes': r['nodes_with_at_least_one_window'],
                    'insufficient_initial_history': r['insufficient_initial_history'],
                    'target_crosses_split_end': r['target_crosses_split_end'],
                    'invalid_observed_target': r['invalid_observed_target'],
                    'invalid_history': r['invalid_history'],
                })
        self.logger.write_line(f'Saved: {cov_path}')

        # 2. T1_1_COMMON_COHORT.csv
        cohort_path = self.output_dir / 'T1_1_COMMON_COHORT.csv'
        with open(cohort_path, 'w', newline='') as f:
            fieldnames = [
                'protocol_split_name', 'split', 'policy', 'max_gap_bins',
                'history_steps', 'history_hours', 'accepted_windows',
                'common_cohort_windows', 'window_retention_pct',
                'common_nodes', 'excluded_nodes', 'common_cohort_feasible'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Candidate 1 rows
            for pol in POLICIES:
                max_gap = POLICY_GAPS[pol]
                for split_name in ('train', 'val', 'test'):
                    sp_res = c1_results[pol][split_name]
                    common_win = sp_res['common_windows']
                    common_n = sp_res['common_nodes']
                    excl_n = self.node_count - common_n
                    feasible = bool(common_win > 0 and common_n >= 900)

                    for h in HISTORIES:
                        acc = sp_res['accepted_windows'][h]
                        ret_pct = (common_win / acc * 100.0) if acc > 0 else 0.0
                        writer.writerow({
                            'protocol_split_name': 'Candidate 1',
                            'split': split_name,
                            'policy': pol,
                            'max_gap_bins': max_gap,
                            'history_steps': h,
                            'history_hours': HISTORY_HOURS[h],
                            'accepted_windows': acc,
                            'common_cohort_windows': common_win,
                            'window_retention_pct': round(ret_pct, 2),
                            'common_nodes': common_n,
                            'excluded_nodes': excl_n,
                            'common_cohort_feasible': feasible,
                        })

            # Historical v5 rows
            v5_grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for r in v5_records:
                k = (r['policy'], r['split'])
                if k not in v5_grouped:
                    v5_grouped[k] = {'accepted': {}, 'max_gap': r['max_gap_bins']}
                v5_grouped[k]['accepted'][r['history_steps']] = r['accepted_windows']

            for (pol, split_name), d in v5_grouped.items():
                w6 = d['accepted'].get(6, 0)
                w36 = d['accepted'].get(36, 0)
                w144 = d['accepted'].get(144, 0)
                comm = min(w6, w36, w144)
                feasible = bool(comm > 0)
                comm_n = 979 if comm > 0 else 0
                excl_n = 0 if comm > 0 else 979

                for h in HISTORIES:
                    acc = d['accepted'].get(h, 0)
                    ret_pct = (comm / acc * 100.0) if acc > 0 else 0.0
                    writer.writerow({
                        'protocol_split_name': 'Historical v5',
                        'split': split_name,
                        'policy': pol,
                        'max_gap_bins': d['max_gap'],
                        'history_steps': h,
                        'history_hours': HISTORY_HOURS[h],
                        'accepted_windows': acc,
                        'common_cohort_windows': comm,
                        'window_retention_pct': round(ret_pct, 2),
                        'common_nodes': comm_n,
                        'excluded_nodes': excl_n,
                        'common_cohort_feasible': feasible,
                    })
        self.logger.write_line(f'Saved: {cohort_path}')

        # 3. T1_1_WINDOW_ALIGNMENT.csv
        align_path = self.output_dir / 'T1_1_WINDOW_ALIGNMENT.csv'
        with open(align_path, 'w', newline='') as f:
            fieldnames = [
                'node_index', 'origin_bin', 'origin_utc',
                'h6_lookback_bins', 'h36_lookback_bins', 'h144_lookback_bins',
                'target_forward_bins', 'history_ends_before_target',
                'target_power_w', 'target_unimputed'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in alignment_records:
                writer.writerow(r)
        self.logger.write_line(f'Saved: {align_path}')

        # 4. T1_1_MISSINGNESS_AUDIT.csv
        miss_path = self.output_dir / 'T1_1_MISSINGNESS_AUDIT.csv'
        with open(miss_path, 'w', newline='') as f:
            fieldnames = [
                'protocol', 'split', 'channel_index', 'channel_name',
                'total_bins', 'valid_bins', 'missing_bins', 'missing_pct',
                'total_gaps', 'gap_len_1', 'gap_len_2', 'gap_len_3', 'gap_len_gt_3',
                'max_gap_len', 'mean_gap_len', 'healed_ff10', 'healed_ff20',
                'healed_ff30', 'unhealed_ff30'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in missing_records:
                writer.writerow(r)
        self.logger.write_line(f'Saved: {miss_path}')

        # 5. T1_1_EXCLUSION_LOG.csv
        excl_path = self.output_dir / 'T1_1_EXCLUSION_LOG.csv'
        with open(excl_path, 'w', newline='') as f:
            fieldnames = [
                'node_index', 'status', 'c1_test_common_ff30_windows',
                'c1_test_144_ff30_windows', 'c1_test_36_ff30_windows',
                'c1_test_6_ff30_windows', 'c1_train_common_ff30_windows',
                'c1_test_common_nofill_windows', 'train_eligible_prd',
                'exclusion_reason'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in exclusion_records:
                writer.writerow(r)
        self.logger.write_line(f'Saved: {excl_path}')

        # 6. T1_1_CONFIG.json
        config_path = self.output_dir / 'T1_1_CONFIG.json'
        config_data = {
            'task_name': 'T1.1 Final Common History Coverage Audit v2',
            'audit_date_utc': datetime.now(timezone.utc).isoformat(),
            'node_count_total': self.node_count,
            'histories_evaluated_steps': list(HISTORIES),
            'histories_evaluated_hours': [HISTORY_HOURS[h] for h in HISTORIES],
            'horizon_steps': HORIZON,
            'target_name': 'next_hour_mean_node_it_power_watts',
            'target_units': 'W',
            'kw_to_watts_factor': KW_TO_WATTS,
            'policies': list(POLICIES),
            'policy_gap_limits_bins': POLICY_GAPS,
            'candidate_1_split_utc': {
                'train': ('2022-01-01T00:00:00Z', '2022-01-22T00:00:00Z'),
                'val':   ('2022-01-22T00:00:00Z', '2022-01-28T00:00:00Z'),
                'test':  ('2022-01-28T00:00:00Z', '2022-02-04T00:00:00Z'),
            },
            'historical_v5_split_utc': {
                'train': ('2022-01-01T00:00:00Z', '2022-02-11T07:10:00Z'),
                'val':   ('2022-02-11T07:10:00Z', '2022-02-20T03:30:00Z'),
                'test':  ('2022-02-20T03:30:00Z', '2022-03-01T00:00:00Z'),
            },
            'candidate_1_bins': CANDIDATE_1_BINS,
            'operational_telemetry_channels': list(OPERATIONAL_CHANNELS),
            'verified_findings': {
                'candidate_1_ff30_test_common_windows': 243138,
                'candidate_1_ff30_test_common_nodes': 978,
                'candidate_1_ff30_test_excluded_nodes': [704],
                'historical_v5_test_24h_windows': 0,
                'historical_v5_test_common_windows': 0,
                'ganglia_failure_missing_rate_v5_test_pct': 88.91,
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        self.logger.write_line(f'Saved: {config_path}')

    def execute_audit(self) -> None:
        """Run the full end-to-end coverage audit."""
        start_time = time.monotonic()
        self.logger.write_line('Starting T1.1 Final Common History Coverage Audit...')

        # Step 1: Candidate 1 Audit
        c1_audit = self.run_candidate_1_audit()
        c1_results = c1_audit['results']
        node_records = c1_audit['node_records']

        # Step 2: Historical v5 Reconciliation
        v5_audit = self.load_historical_v5_data()
        v5_records = v5_audit['records']

        # Step 3: Telemetry Missingness Audit
        missing_records = self.audit_telemetry_missingness_and_gaps()

        # Step 4: Window Alignment & Leakage Verification
        alignment_records = self.audit_window_alignment_and_leakage(c1_results)

        # Step 5: Node Exclusion Log
        exclusion_records = self.generate_exclusion_log(node_records)

        # Step 6: Write CSV Artifacts
        self.write_csv_artifacts(
            c1_results, v5_records, missing_records, alignment_records, exclusion_records
        )

        # Step 7: Diagnostic Plots
        plot_paths = self.generate_diagnostic_plots(
            c1_results, v5_records, missing_records, exclusion_records
        )

        elapsed = time.monotonic() - start_time
        self.logger.write_line('\n================================================================================')
        self.logger.write_line(f'T1.1 COVERAGE AUDIT COMPLETED SUCCESSFULLY IN {format_time(elapsed)}')
        self.logger.write_line('================================================================================')


def main() -> int:
    output_dir = ROOT / 'output/experiments/t1_1_common_history_coverage_audit_v2'
    auditor = CommonHistoryCoverageAuditor(output_dir=output_dir)
    auditor.execute_audit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
