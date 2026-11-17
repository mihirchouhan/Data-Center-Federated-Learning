"""
Task T7: Missing Telemetry Robustness Experiment Runner.

Evaluates inference-time robustness of the selected FedAvg-GRU forecasting model
(frozen Task T4/T5 winner checkpoints) under varying operational telemetry missingness:

Missingness Levels:
- 0% (Reference baseline)
- 5%, 10%, 20%, 30%

Missingness Mechanisms:
- Random Missingness (independent Bernoulli dropout across operational channels)
- Contiguous Missing Blocks (outages of block lengths L=3 [30m], L=6 [1h], L=12 [2h])

Feature Group Ablation / Channel Sensitivity:
- CPU features (Channels 1, 2, 3, 4)
- Memory feature (Channel 5)
- Inlet temperature (Channel 6)
- Power telemetry (Channel 0)

Causal Preprocessing:
- Causal forward-fill FF30 (maximum 3 bins = 30 minutes) applied after introducing missingness.
- Fallback for unfillable gaps (> 30 min): Standardized zero-mean imputation (x_scaled = 0.0).
- Fallback for persistence: Last available power in 36-step lookback; y_mean if completely lost.
- Target power P_target is NEVER masked (strict ground truth).
- Weather channels [7, 8] remain masked to 0.0 consistent with T5.

Evaluated on:
- Canonical T1 Held-Out Test (243,138 windows, 978 nodes) across all conditions
- Q1 Extended (3,609,285 windows) and March Window (2,229,042 windows) for key conditions

Mandatory real-time terminal progress logging via src.progress_logging.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.history_length_experiment_v1 import SequenceGRU
from src.history_length_real_pilot_v1 import Candidate1RealData
from src.progress_logging import (
    DualLogger,
    EvaluationProgressLogger,
    ExperimentStageTracker,
    calculate_eta,
    format_time,
)
from src.temporal_generalization_v1 import ExtendedTelemetryDataset

KW_TO_WATTS = 1000.0
HORIZON_STEPS = 6
HISTORY_STEPS = 36
OPERATIONAL_CHANNELS = [0, 1, 2, 3, 4, 5, 6]
WEATHER_CHANNELS = [7, 8]
CALENDAR_CHANNELS = [9, 10, 11, 12]

FEATURE_GROUPS = {
    'power': [0],
    'cpu': [1, 2, 3, 4],
    'memory': [5],
    'inlet_temp': [6],
}

T4_EXPECTED_HASHES = {
    42: ('fedavg_winner_seed42_best.pt', '11572caf6ebb73cf96bdebb743216c0142a7324ac58bfcd48e8f8c1ecd1d8c14'),
    43: ('fedavg_winner_seed43_best.pt', '10d5f965dff828a944c93a8d33796920667fbba15f48411b29ffdc6c0dc654c5'),
    44: ('fedavg_winner_seed44_best.pt', '7d2071cef79c027df06db9a689904c577222fb9a498d627371ac719a65f0c6ad'),
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class T7MissingTelemetryManager:
    """
    Manages controlled missingness generation, causal FF30 tracking, and model evaluation.
    """

    def __init__(
        self,
        output_dir: Path = ROOT / 'output/experiments/t7_missing_telemetry_robustness_v1',
        device_str: str = 'cuda:1',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available() and 'cuda' in device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device('cpu')

        print(f"[T7Manager] Initialized on device: {self.device}", flush=True)

        # Verify reference T4 checkpoints cryptographically
        self._verify_reference_checkpoints()

        # Load Candidate 1 reference dataset
        print("[T7Manager] Loading Candidate 1 dataset...", flush=True)
        self.t1_data = Candidate1RealData()
        self.node_count = self.t1_data.node_count
        self.train_origins = self.t1_data.valid_origins['train']
        self.val_origins = self.t1_data.valid_origins['val']
        self.test_t1_origins = self.t1_data.valid_origins['test']

        # Training moments
        self.y_mean_kw = float(self.t1_data.scalers['y_mean'][0])
        self.y_std_kw = float(self.t1_data.scalers['y_std'][0])

        # Precompute residual moments on training set
        train_res_list = []
        for n_idx in range(self.node_count):
            orig = self.train_origins[n_idx]
            if len(orig) == 0:
                continue
            tb = orig[:, None] + np.arange(HORIZON_STEPS)
            t_kw = np.mean(self.t1_data.power[n_idx, tb], axis=1)
            t_steps = orig[:, None] + np.arange(-HISTORY_STEPS, 0)
            p_kw = self.t1_data.power[n_idx, t_steps[:, -1]]
            train_res_list.append(t_kw - p_kw)

        all_res_kw = np.concatenate(train_res_list, axis=0)
        self.res_mean_kw = float(np.mean(all_res_kw))
        self.res_std_kw = float(np.std(all_res_kw))
        print(f"  Residual moments: mean={self.res_mean_kw*KW_TO_WATTS:.4f} W, std={self.res_std_kw*KW_TO_WATTS:.4f} W", flush=True)

        # Lazy loader for extended telemetry
        self._ext_data: Optional[ExtendedTelemetryDataset] = None

    def _verify_reference_checkpoints(self):
        """Verify T4/T5 reference checkpoints against expected SHA-256 digests."""
        print("[T7Manager] Verifying frozen reference checkpoints...", flush=True)
        for seed, (fname, exp_h) in T4_EXPECTED_HASHES.items():
            p = ROOT / f"output/experiments/t4_model_improvement_v1/checkpoints/{fname}"
            if not p.exists():
                raise FileNotFoundError(f"Missing T4 checkpoint: {p}")
            act_h = sha256_file(p)
            assert act_h == exp_h, f"Checkpoint mismatch for seed {seed}: {act_h} != {exp_h}"
            print(f"  Seed {seed} Checkpoint verified: {fname} [{act_h[:16]}...]", flush=True)

    def get_extended_data(self) -> ExtendedTelemetryDataset:
        if self._ext_data is None:
            print("[T7Manager] Loading ExtendedTelemetryDataset for Q1/March evaluation...", flush=True)
            self._ext_data = ExtendedTelemetryDataset()
        return self._ext_data

    def apply_missingness_and_causal_ff30(
        self,
        features: np.ndarray,
        ages: np.ndarray,
        target_channels: Sequence[int],
        missing_type: str,
        missing_pct: float,
        block_len: int = 1,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Applies artificial missingness to specified operational channels and recomputes causal FF30.

        Returns:
        - filled_features: Array with causal FF30 applied; unfillable values set to NaN.
        - final_ages: Array of new causal ages (-1: unobserved, 0: observed, 1..3: filled, >3: unfillable).
        - audit_metrics: Complete accounting dictionary before and after FF30.
        """
        N, T, C = features.shape
        rng = np.random.default_rng(np.random.SeedSequence([seed, 777, int(missing_pct * 100), block_len]))

        # Target channels must be in operational channels
        target_ch_list = list(target_channels)
        C_target = len(target_ch_list)

        # 1. Natural observation mask (observed directly, age == 0, finite)
        is_nat_obs = np.isfinite(features[:, :, target_ch_list]) & (ages[:, :, target_ch_list] == 0)
        is_nat_missing = ~is_nat_obs

        # 2. Generate artificial missing mask on observed entries
        art_mask = np.zeros((N, T, C_target), dtype=bool)

        if missing_type == 'none' or missing_pct == 0.0:
            pass  # art_mask is all False
        elif missing_type == 'random':
            art_mask = (rng.random((N, T, C_target)) < missing_pct) & is_nat_obs
        elif missing_type == 'contiguous':
            L = block_len
            n_blocks = int(round((T * missing_pct) / L))
            for n in range(N):
                for idx, c in enumerate(target_ch_list):
                    if n_blocks == 0:
                        continue
                    max_starts = max(1, (T - L) // L)
                    actual_blocks = min(n_blocks, max_starts)
                    starts = rng.choice(T - L, size=actual_blocks, replace=False)
                    for s in starts:
                        art_mask[n, s:s + L, idx] = True
            art_mask = art_mask & is_nat_obs
        elif missing_type == 'blackout':
            art_mask = is_nat_obs.copy()  # 100% of target channels dropped
        else:
            raise ValueError(f"Unknown missing_type: {missing_type}")

        # 3. Combined effective observation mask
        is_effective_obs = is_nat_obs & ~art_mask

        # 4. Trace causal forward fill across time
        filled_features = features.copy()
        final_ages = ages.copy()

        pos = np.arange(T)
        total_filled_values = 0
        total_unfilled_values = 0

        for idx, c in enumerate(target_ch_list):
            for n in range(N):
                obs = is_effective_obs[n, :, idx]
                last_obs_pos = np.maximum.accumulate(np.where(obs, pos, -1))
                new_age = np.where(obs, 0, np.where(last_obs_pos >= 0, pos - last_obs_pos, -1))

                # Identify forward-fillable bins (age 1, 2, 3)
                fill_mask = (new_age > 0) & (new_age <= 3)
                unfill_mask = (new_age > 3) | (new_age < 0)

                # Apply causal fill from last observed position
                if fill_mask.any():
                    valid_fill_src = last_obs_pos[fill_mask]
                    filled_features[n, fill_mask, c] = features[n, valid_fill_src, c]
                    total_filled_values += int(np.sum(fill_mask))

                # Set unfillable bins to NaN
                if unfill_mask.any():
                    filled_features[n, unfill_mask, c] = np.nan
                    total_unfilled_values += int(np.sum(unfill_mask))

                final_ages[n, :, c] = new_age

        # 5. Compile missingness accounting
        total_target_values = N * T * C_target
        nat_missing_count = int(np.sum(is_nat_missing))
        art_missing_count = int(np.sum(art_mask))
        total_missing_before_ff = nat_missing_count + art_missing_count
        pct_missing_before_ff = (total_missing_before_ff / max(1, total_target_values)) * 100.0

        pct_filled_by_ff = (total_filled_values / max(1, total_missing_before_ff)) * 100.0
        pct_still_unavailable = (total_unfilled_values / max(1, total_target_values)) * 100.0

        # Count affected nodes (nodes with at least one artificial drop)
        affected_nodes_count = int(np.sum(art_mask.any(axis=(1, 2))))

        audit_metrics = {
            'total_operational_values': total_target_values,
            'naturally_missing_values': nat_missing_count,
            'artificial_missing_values': art_missing_count,
            'total_missing_before_ff': total_missing_before_ff,
            'pct_missing_before_ff': pct_missing_before_ff,
            'values_filled_by_ff': total_filled_values,
            'pct_filled_by_ff': pct_filled_by_ff,
            'values_still_unavailable': total_unfilled_values,
            'pct_still_unavailable_after_ff': pct_still_unavailable,
            'affected_nodes': affected_nodes_count,
        }

        return filled_features, final_ages, audit_metrics

    def evaluate_t1_under_missingness(
        self,
        model: nn.Module,
        filled_features: np.ndarray,
        final_ages: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates model on Canonical T1 Held-Out Test under modified features.
        Target power is strictly unmasked.
        """
        model.eval()
        lo, hi = 3888, 4896
        H = HISTORY_STEPS

        preds_all = []
        trues_all = []
        pers_all = []
        per_node_mae = {}
        per_node_rmse = {}
        per_node_pers_mae = {}

        t0 = time.time()
        with torch.no_grad():
            for node_idx in range(self.node_count):
                orig = self.test_t1_origins[node_idx]
                if len(orig) == 0:
                    continue

                B = len(orig)

                # Target is strictly UNMASKED true power
                target_bins = orig[:, None] + np.arange(HORIZON_STEPS)
                targets_kw = np.mean(self.t1_data.power[node_idx, target_bins], axis=1, keepdims=True)
                true_w = targets_kw * KW_TO_WATTS

                # Extract history from modified features
                # Offset relative to slice start (lo - H)
                slice_start = lo - H
                rel_orig = orig - slice_start
                time_indices = rel_orig[:, None] + np.arange(-H, 0)

                hist = filled_features[node_idx, time_indices].copy()

                # Robust persistence derivation
                # Step t-1 is at time_indices[:, -1]
                last_power_kw = hist[:, -1, 0].copy()
                pers_nan = np.isnan(last_power_kw)
                if pers_nan.any():
                    # Search backwards in history for last available valid power
                    for b_idx in np.where(pers_nan)[0]:
                        valid_steps = np.where(np.isfinite(hist[b_idx, :, 0]))[0]
                        if len(valid_steps) > 0:
                            last_power_kw[b_idx] = hist[b_idx, valid_steps[-1], 0]
                        else:
                            last_power_kw[b_idx] = self.y_mean_kw  # Training mean fallback

                pers_w = (last_power_kw[:, None] * KW_TO_WATTS).astype(np.float32)

                # Standardize features
                scaled_hist = (hist - self.t1_data.scalers['x_mean'][:13]) / self.t1_data.scalers['x_std'][:13]
                # Fallback: unfillable gaps set to 0.0 (standardized mean imputation)
                scaled_hist[np.isnan(scaled_hist)] = 0.0
                # Weather channels 7 and 8 masked to 0.0
                scaled_hist[:, :, WEATHER_CHANNELS] = 0.0

                node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
                x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

                bx = torch.from_numpy(x_scaled).to(self.device)
                pred_res = model(bx).cpu().numpy()

                delta_kw = pred_res * self.res_std_kw + self.res_mean_kw
                pred_w = pers_w + delta_kw * KW_TO_WATTS

                preds_all.append(pred_w)
                trues_all.append(true_w)
                pers_all.append(pers_w)

                n_err = pred_w - true_w
                n_pers_err = pers_w - true_w
                per_node_mae[node_idx] = float(np.mean(np.abs(n_err)))
                per_node_rmse[node_idx] = float(np.sqrt(np.mean(n_err ** 2)))
                per_node_pers_mae[node_idx] = float(np.mean(np.abs(n_pers_err)))

        eval_time = time.time() - t0
        preds_all = np.vstack(preds_all)
        trues_all = np.vstack(trues_all)
        pers_all = np.vstack(pers_all)

        errors = preds_all - trues_all
        pers_errors = pers_all - trues_all

        pooled_mae = float(np.mean(np.abs(errors)))
        pooled_rmse = float(np.sqrt(np.mean(errors ** 2)))
        pers_pooled_mae = float(np.mean(np.abs(pers_errors)))
        pers_pooled_rmse = float(np.sqrt(np.mean(pers_errors ** 2)))

        macro_mae = float(np.mean(list(per_node_mae.values())))
        macro_rmse = float(np.mean(list(per_node_rmse.values())))
        pers_macro_mae = float(np.mean(list(per_node_pers_mae.values())))

        nodes_beating = sum(1 for n, m in per_node_mae.items() if m < per_node_pers_mae[n])
        total_eval_nodes = len(per_node_mae)

        return {
            'period': 'test_t1',
            'windows': len(errors),
            'nodes': total_eval_nodes,
            'pooled_mae_w': pooled_mae,
            'pooled_rmse_w': pooled_rmse,
            'macro_mae_w': macro_mae,
            'macro_rmse_w': macro_rmse,
            'pers_pooled_mae_w': pers_pooled_mae,
            'pers_pooled_rmse_w': pers_pooled_rmse,
            'pers_macro_mae_w': pers_macro_mae,
            'nodes_beating_pers': nodes_beating,
            'pct_nodes_beating_pers': (nodes_beating / total_eval_nodes) * 100.0,
            'per_node_mae': per_node_mae,
            'per_node_rmse': per_node_rmse,
            'per_node_pers_mae': per_node_pers_mae,
            'eval_time_seconds': eval_time,
            'throughput_windows_sec': len(errors) / max(0.001, eval_time),
        }

    def evaluate_extended_under_missingness(
        self,
        model: nn.Module,
        period: str = 'test_q1',
        missing_type: str = 'none',
        missing_pct: float = 0.0,
        block_len: int = 1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Evaluates model on Q1 Extended or March Window under missingness.
        """
        ext = self.get_extended_data()
        model.eval()

        if period == 'test_q1':
            lo, hi = 3888, 12960
        elif period == 'test_march':
            lo, hi = 8496, 12960
        else:
            raise ValueError(f"Unknown period: {period}")

        H = HISTORY_STEPS

        # Apply missingness to slice [lo - H, hi)
        slice_start = lo - H
        sub_features = ext.features[:, slice_start:hi, :13].copy()
        sub_ages = ext.ages[:, slice_start:hi, :13].copy()

        filled_sub, _, _ = self.apply_missingness_and_causal_ff30(
            features=sub_features,
            ages=sub_ages,
            target_channels=OPERATIONAL_CHANNELS,
            missing_type=missing_type,
            missing_pct=missing_pct,
            block_len=block_len,
            seed=seed,
        )

        preds_all = []
        trues_all = []
        pers_all = []
        per_node_mae = {}
        per_node_rmse = {}
        per_node_pers_mae = {}

        t0 = time.time()
        with torch.no_grad():
            for node_idx in range(self.node_count):
                all_orig = ext.origins['t2'][node_idx]
                orig = all_orig[(all_orig >= lo) & (all_orig < hi)]
                if len(orig) == 0:
                    continue

                B = len(orig)
                rel_orig = orig - slice_start

                tb = orig[:, None] + np.arange(HORIZON_STEPS)
                targets_kw = np.mean(ext.power[node_idx, tb], axis=1, keepdims=True)
                true_w = targets_kw * KW_TO_WATTS

                time_indices = rel_orig[:, None] + np.arange(-H, 0)
                hist = filled_sub[node_idx, time_indices].copy()

                # Persistence
                last_power_kw = hist[:, -1, 0].copy()
                pers_nan = np.isnan(last_power_kw)
                if pers_nan.any():
                    for b_idx in np.where(pers_nan)[0]:
                        valid_steps = np.where(np.isfinite(hist[b_idx, :, 0]))[0]
                        if len(valid_steps) > 0:
                            last_power_kw[b_idx] = hist[b_idx, valid_steps[-1], 0]
                        else:
                            last_power_kw[b_idx] = ext.y_mean_kw

                pers_w = (last_power_kw[:, None] * KW_TO_WATTS).astype(np.float32)

                scaled_hist = (hist - ext.scalers['x_mean'][:13]) / ext.scalers['x_std'][:13]
                scaled_hist[np.isnan(scaled_hist)] = 0.0
                scaled_hist[:, :, WEATHER_CHANNELS] = 0.0

                node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
                x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

                bx = torch.from_numpy(x_scaled).to(self.device)
                pred_res = model(bx).cpu().numpy()

                delta_kw = pred_res * self.res_std_kw + self.res_mean_kw
                pred_w = pers_w + delta_kw * KW_TO_WATTS

                preds_all.append(pred_w)
                trues_all.append(true_w)
                pers_all.append(pers_w)

                n_err = pred_w - true_w
                n_pers_err = pers_w - true_w
                per_node_mae[node_idx] = float(np.mean(np.abs(n_err)))
                per_node_rmse[node_idx] = float(np.sqrt(np.mean(n_err ** 2)))
                per_node_pers_mae[node_idx] = float(np.mean(np.abs(n_pers_err)))

        eval_time = time.time() - t0
        preds_all = np.vstack(preds_all)
        trues_all = np.vstack(trues_all)
        pers_all = np.vstack(pers_all)

        errors = preds_all - trues_all
        pers_errors = pers_all - trues_all

        pooled_mae = float(np.mean(np.abs(errors)))
        pooled_rmse = float(np.sqrt(np.mean(errors ** 2)))
        pers_pooled_mae = float(np.mean(np.abs(pers_errors)))
        pers_pooled_rmse = float(np.sqrt(np.mean(pers_errors ** 2)))

        macro_mae = float(np.mean(list(per_node_mae.values())))
        macro_rmse = float(np.mean(list(per_node_rmse.values())))
        pers_macro_mae = float(np.mean(list(per_node_pers_mae.values())))

        nodes_beating = sum(1 for n, m in per_node_mae.items() if m < per_node_pers_mae[n])
        total_eval_nodes = len(per_node_mae)

        return {
            'period': period,
            'windows': len(errors),
            'nodes': total_eval_nodes,
            'pooled_mae_w': pooled_mae,
            'pooled_rmse_w': pooled_rmse,
            'macro_mae_w': macro_mae,
            'macro_rmse_w': macro_rmse,
            'pers_pooled_mae_w': pers_pooled_mae,
            'pers_pooled_rmse_w': pers_pooled_rmse,
            'pers_macro_mae_w': pers_macro_mae,
            'nodes_beating_pers': nodes_beating,
            'pct_nodes_beating_pers': (nodes_beating / total_eval_nodes) * 100.0,
            'per_node_mae': per_node_mae,
            'per_node_rmse': per_node_rmse,
            'per_node_pers_mae': per_node_pers_mae,
            'eval_time_seconds': eval_time,
            'throughput_windows_sec': len(errors) / max(0.001, eval_time),
        }

    def run_all_robustness_experiments(self, seeds: Sequence[int] = (42, 43, 44)) -> Dict[str, Any]:
        """
        Execute full T7 Missing Telemetry Robustness Experiment.
        """
        # Load models for all seeds
        models = {}
        for seed in seeds:
            m = SequenceGRU(
                input_dim=14,
                hidden_dim=64,
                num_layers=2,
                node_count=self.node_count,
                node_embedding_dim=8,
            ).to(self.device)
            ckpt_p = ROOT / f"output/experiments/t4_model_improvement_v1/checkpoints/{T4_EXPECTED_HASHES[seed][0]}"
            ckpt = torch.load(ckpt_p, map_location=self.device)
            m.load_state_dict(ckpt['model_state_dict'])
            m.eval()
            models[seed] = m

        # Pre-extract slice for T1
        lo, hi = 3888, 4896
        H = HISTORY_STEPS
        t1_slice_features = self.t1_data.features[:, lo - H:hi, :13].copy()
        t1_slice_ages = self.t1_data.ages[:, lo - H:hi, :13].copy()

        # Build condition matrix
        conditions: List[Dict[str, Any]] = [
            # 1. Baseline Reference (0% missingness)
            {
                'condition_id': 'cond_00_ref_0pct',
                'label': 'Reference Baseline (0% Missing)',
                'missing_type': 'none',
                'missing_pct': 0.0,
                'block_len': 1,
                'target_channels': OPERATIONAL_CHANNELS,
                'feature_group': 'all_operational',
            },
        ]

        # 2. Random Missingness: 5%, 10%, 20%, 30%
        for pct in [0.05, 0.10, 0.20, 0.30]:
            conditions.append({
                'condition_id': f"cond_rand_{int(pct*100):02d}pct",
                'label': f"Random {int(pct*100)}% Missing",
                'missing_type': 'random',
                'missing_pct': pct,
                'block_len': 1,
                'target_channels': OPERATIONAL_CHANNELS,
                'feature_group': 'all_operational',
            })

        # 3. Contiguous Missingness: L=3 (30m), L=6 (1h), L=12 (2h) across 5%, 10%, 20%, 30%
        for L in [3, 6, 12]:
            l_str = f"l{L}"
            time_str = f"{L*10}m" if L < 6 else f"{L//6}h"
            for pct in [0.05, 0.10, 0.20, 0.30]:
                conditions.append({
                    'condition_id': f"cond_cont_{l_str}_{int(pct*100):02d}pct",
                    'label': f"Contiguous L={L} ({time_str}) {int(pct*100)}% Missing",
                    'missing_type': 'contiguous',
                    'missing_pct': pct,
                    'block_len': L,
                    'target_channels': OPERATIONAL_CHANNELS,
                    'feature_group': 'all_operational',
                })

        # 4. Feature Group Sensitivity / Channel Ablation (at 20% random and 100% blackout)
        for grp_name, chs in FEATURE_GROUPS.items():
            # 20% random dropout on this group
            conditions.append({
                'condition_id': f"group_{grp_name}_rand20",
                'label': f"Feature Group: {grp_name.title()} (20% Random Missing)",
                'missing_type': 'random',
                'missing_pct': 0.20,
                'block_len': 1,
                'target_channels': chs,
                'feature_group': grp_name,
            })
            # 100% complete blackout on this group
            conditions.append({
                'condition_id': f"group_{grp_name}_blackout",
                'label': f"Feature Group: {grp_name.title()} (100% Sensor Outage)",
                'missing_type': 'blackout',
                'missing_pct': 1.0,
                'block_len': 1,
                'target_channels': chs,
                'feature_group': grp_name,
            })

        total_experiments = len(conditions) * len(seeds)
        suite_tracker = ExperimentStageTracker(
            suite_name="T7 Missing Telemetry Robustness Experiment",
            total_experiments=total_experiments,
            output_dir=self.output_dir,
            log_filename="progress.log",
        )

        records_per_condition: List[Dict[str, Any]] = []
        per_node_records: List[Dict[str, Any]] = []
        missingness_audit_records: List[Dict[str, Any]] = []

        clean_ref_mae: Dict[int, float] = {}
        exp_counter = 0

        # Execute T1 evaluation over all conditions
        for cond in conditions:
            cond_id = cond['condition_id']
            cond_label = cond['label']
            m_type = cond['missing_type']
            m_pct = cond['missing_pct']
            b_len = cond['block_len']
            t_chs = cond['target_channels']
            f_grp = cond['feature_group']

            # Generate masked data once per condition using fixed reference seed (42)
            filled_feat, final_ages, audit = self.apply_missingness_and_causal_ff30(
                features=t1_slice_features,
                ages=t1_slice_ages,
                target_channels=t_chs,
                missing_type=m_type,
                missing_pct=m_pct,
                block_len=b_len,
                seed=42,  # Deterministic identical mask across model seeds
            )

            missingness_audit_records.append({
                'condition_id': cond_id,
                'condition_label': cond_label,
                'missing_type': m_type,
                'missing_pct': m_pct,
                'block_len': b_len,
                'feature_group': f_grp,
                'target_channels': str(t_chs),
                **audit,
            })

            # Evaluate across all seeds on the identical masked dataset
            for seed in seeds:
                exp_counter += 1
                suite_tracker.set_experiment(exp_counter, cond_label, seed)

                t_eval_start = time.time()
                res = self.evaluate_t1_under_missingness(
                    model=models[seed],
                    filled_features=filled_feat,
                    final_ages=final_ages,
                )
                eval_elapsed = time.time() - t_eval_start

                if cond_id == 'cond_00_ref_0pct':
                    clean_ref_mae[seed] = res['pooled_mae_w']

                ref_mae = clean_ref_mae[seed]
                deg_abs = res['pooled_mae_w'] - ref_mae
                deg_pct = (deg_abs / ref_mae) * 100.0

                print(
                    f"  [T1] Pooled MAE: {res['pooled_mae_w']:.2f} W (Degradation: {deg_abs:+.2f} W, {deg_pct:+.2f}%) | "
                    f"Macro MAE: {res['macro_mae_w']:.2f} W | Pers MAE: {res['pers_pooled_mae_w']:.2f} W | Beating: {res['pct_nodes_beating_pers']:.1f}%",
                    flush=True,
                )

                records_per_condition.append({
                    'condition_id': cond_id,
                    'condition_label': cond_label,
                    'missing_type': m_type,
                    'missing_pct': m_pct,
                    'block_len': b_len,
                    'feature_group': f_grp,
                    'seed': seed,
                    'period': 'test_t1',
                    'windows': res['windows'],
                    'nodes': res['nodes'],
                    'pooled_mae_w': res['pooled_mae_w'],
                    'pooled_rmse_w': res['pooled_rmse_w'],
                    'macro_mae_w': res['macro_mae_w'],
                    'macro_rmse_w': res['macro_rmse_w'],
                    'pers_pooled_mae_w': res['pers_pooled_mae_w'],
                    'pers_pooled_rmse_w': res['pers_pooled_rmse_w'],
                    'pers_macro_mae_w': res['pers_macro_mae_w'],
                    'nodes_beating_pers': res['nodes_beating_pers'],
                    'pct_nodes_beating_pers': res['pct_nodes_beating_pers'],
                    'degradation_mae_vs_clean_w': deg_abs,
                    'pct_degradation_mae_vs_clean': deg_pct,
                    'eval_time_seconds': eval_elapsed,
                })

                # Store per-node metrics for key conditions
                if cond_id in ['cond_00_ref_0pct', 'cond_rand_10pct', 'cond_cont_l6_10pct', 'cond_cont_l6_30pct']:
                    for n_idx, m_mae in res['per_node_mae'].items():
                        per_node_records.append({
                            'condition_id': cond_id,
                            'seed': seed,
                            'period': 'test_t1',
                            'node_idx': n_idx,
                            'model_mae_w': m_mae,
                            'model_rmse_w': res['per_node_rmse'][n_idx],
                            'pers_mae_w': res['per_node_pers_mae'][n_idx],
                            'diff_vs_pers_w': m_mae - res['per_node_pers_mae'][n_idx],
                        })

        # =====================================================================
        # Chronological Temporal Generalization under Missingness (Q1 & March)
        # =====================================================================
        print("\n[T7Manager] Evaluating Chronological Generalization under Missingness (Q1 & March)...", flush=True)
        key_temporal_conditions = [
            ('cond_00_ref_0pct', 'Reference (0% Missing)', 'none', 0.0, 1),
            ('cond_rand_10pct', 'Random 10% Missing', 'random', 0.10, 1),
            ('cond_cont_l6_10pct', 'Contiguous L=6 10% Missing', 'contiguous', 0.10, 6),
            ('cond_cont_l6_20pct', 'Contiguous L=6 20% Missing', 'contiguous', 0.20, 6),
        ]

        clean_ext_mae: Dict[Tuple[str, int], float] = {}

        for period in ['test_q1', 'test_march']:
            for cond_id, cond_label, m_type, m_pct, b_len in key_temporal_conditions:
                for seed in seeds:
                    t_eval_start = time.time()
                    res = self.evaluate_extended_under_missingness(
                        model=models[seed],
                        period=period,
                        missing_type=m_type,
                        missing_pct=m_pct,
                        block_len=b_len,
                        seed=42,  # Deterministic mask
                    )
                    eval_elapsed = time.time() - t_eval_start

                    if cond_id == 'cond_00_ref_0pct':
                        clean_ext_mae[(period, seed)] = res['pooled_mae_w']

                    ref_mae = clean_ext_mae[(period, seed)]
                    deg_abs = res['pooled_mae_w'] - ref_mae
                    deg_pct = (deg_abs / ref_mae) * 100.0

                    print(
                        f"  [{period.upper()} | {cond_label} | Seed {seed}] Pooled MAE: {res['pooled_mae_w']:.2f} W (Degradation: {deg_abs:+.2f} W, {deg_pct:+.2f}%) | "
                        f"Macro MAE: {res['macro_mae_w']:.2f} W | Pers MAE: {res['pers_pooled_mae_w']:.2f} W",
                        flush=True,
                    )

                    records_per_condition.append({
                        'condition_id': cond_id,
                        'condition_label': cond_label,
                        'missing_type': m_type,
                        'missing_pct': m_pct,
                        'block_len': b_len,
                        'feature_group': 'all_operational',
                        'seed': seed,
                        'period': period,
                        'windows': res['windows'],
                        'nodes': res['nodes'],
                        'pooled_mae_w': res['pooled_mae_w'],
                        'pooled_rmse_w': res['pooled_rmse_w'],
                        'macro_mae_w': res['macro_mae_w'],
                        'macro_rmse_w': res['macro_rmse_w'],
                        'pers_pooled_mae_w': res['pers_pooled_mae_w'],
                        'pers_pooled_rmse_w': res['pers_pooled_rmse_w'],
                        'pers_macro_mae_w': res['pers_macro_mae_w'],
                        'nodes_beating_pers': res['nodes_beating_pers'],
                        'pct_nodes_beating_pers': res['pct_nodes_beating_pers'],
                        'degradation_mae_vs_clean_w': deg_abs,
                        'pct_degradation_mae_vs_clean': deg_pct,
                        'eval_time_seconds': eval_elapsed,
                    })

        suite_tracker.finish_suite()

        # =====================================================================
        # AGGREGATION AND ARTIFACT EXPORTS
        # =====================================================================
        print("\n[T7Manager] Aggregating results and exporting artifacts...", flush=True)

        # 1. T7_MISSINGNESS_AUDIT.csv
        df_audit = pd.DataFrame(missingness_audit_records)
        csv_audit = self.output_dir / 'T7_MISSINGNESS_AUDIT.csv'
        df_audit.to_csv(csv_audit, index=False)
        print(f"  Exported: {csv_audit} ({len(df_audit)} rows)", flush=True)

        # 2. T7_RESULTS_PER_CONDITION.csv
        df_per_cond = pd.DataFrame(records_per_condition)
        csv_per_cond = self.output_dir / 'T7_RESULTS_PER_CONDITION.csv'
        df_per_cond.to_csv(csv_per_cond, index=False)
        print(f"  Exported: {csv_per_cond} ({len(df_per_cond)} rows)", flush=True)

        # 3. T7_PER_NODE_RESULTS.csv
        df_per_node = pd.DataFrame(per_node_records)
        csv_per_node = self.output_dir / 'T7_PER_NODE_RESULTS.csv'
        df_per_node.to_csv(csv_per_node, index=False)
        print(f"  Exported: {csv_per_node} ({len(df_per_node)} rows)", flush=True)

        # 4. T7_RESULTS_SUMMARY.csv
        summary_rows = []
        for period in ['test_t1', 'test_q1', 'test_march']:
            sub_p = df_per_cond[df_per_cond['period'] == period]
            if len(sub_p) == 0:
                continue

            for cond_id in sub_p['condition_id'].unique():
                sub_c = sub_p[sub_p['condition_id'] == cond_id]
                first_row = sub_c.iloc[0]

                p_mae_m = sub_c['pooled_mae_w'].mean()
                p_mae_s = sub_c['pooled_mae_w'].std()
                p_rmse_m = sub_c['pooled_rmse_w'].mean()
                p_rmse_s = sub_c['pooled_rmse_w'].std()

                m_mae_m = sub_c['macro_mae_w'].mean()
                m_mae_s = sub_c['macro_mae_w'].std()
                m_rmse_m = sub_c['macro_rmse_w'].mean()
                m_rmse_s = sub_c['macro_rmse_w'].std()

                pers_mae = sub_c['pers_pooled_mae_w'].iloc[0]
                pers_rmse = sub_c['pers_pooled_rmse_w'].iloc[0]

                deg_abs_m = sub_c['degradation_mae_vs_clean_w'].mean()
                deg_abs_s = sub_c['degradation_mae_vs_clean_w'].std()
                deg_pct_m = sub_c['pct_degradation_mae_vs_clean'].mean()

                summary_rows.append({
                    'period': period,
                    'condition_id': cond_id,
                    'condition_label': first_row['condition_label'],
                    'missing_type': first_row['missing_type'],
                    'missing_pct': first_row['missing_pct'],
                    'block_len': first_row['block_len'],
                    'feature_group': first_row['feature_group'],
                    'windows': first_row['windows'],
                    'nodes': first_row['nodes'],
                    'pooled_mae_mean_w': round(p_mae_m, 4),
                    'pooled_mae_std_w': round(p_mae_s, 4),
                    'pooled_rmse_mean_w': round(p_rmse_m, 4),
                    'pooled_rmse_std_w': round(p_rmse_s, 4),
                    'macro_mae_mean_w': round(m_mae_m, 4),
                    'macro_mae_std_w': round(m_mae_s, 4),
                    'macro_rmse_mean_w': round(m_rmse_m, 4),
                    'macro_rmse_std_w': round(m_rmse_s, 4),
                    'pers_pooled_mae_w': round(pers_mae, 4),
                    'pers_pooled_rmse_w': round(pers_rmse, 4),
                    'nodes_beating_pers_mean': round(sub_c['nodes_beating_pers'].mean(), 1),
                    'pct_nodes_beating_pers_mean': round(sub_c['pct_nodes_beating_pers'].mean(), 2),
                    'degradation_mae_mean_w': round(deg_abs_m, 4),
                    'degradation_mae_std_w': round(deg_abs_s, 4),
                    'pct_degradation_mean': round(deg_pct_m, 2),
                })

        df_summary = pd.DataFrame(summary_rows)
        csv_summary = self.output_dir / 'T7_RESULTS_SUMMARY.csv'
        df_summary.to_csv(csv_summary, index=False)
        print(f"  Exported: {csv_summary} ({len(df_summary)} rows)", flush=True)

        # 5. T7_CHANNEL_IMPACT.csv (Feature Group Degradation Ranking)
        impact_rows = []
        t1_summary = df_summary[df_summary['period'] == 'test_t1'].set_index('condition_id')
        ref_mae_val = t1_summary.loc['cond_00_ref_0pct', 'pooled_mae_mean_w']

        for grp_name in FEATURE_GROUPS.keys():
            # 20% random
            r_cid = f"group_{grp_name}_rand20"
            if r_cid in t1_summary.index:
                r_mae = t1_summary.loc[r_cid, 'pooled_mae_mean_w']
                r_deg = r_mae - ref_mae_val
                r_pct = (r_deg / ref_mae_val) * 100.0
            else:
                r_mae, r_deg, r_pct = None, None, None

            # 100% blackout
            b_cid = f"group_{grp_name}_blackout"
            if b_cid in t1_summary.index:
                b_mae = t1_summary.loc[b_cid, 'pooled_mae_mean_w']
                b_deg = b_mae - ref_mae_val
                b_pct = (b_deg / ref_mae_val) * 100.0
            else:
                b_mae, b_deg, b_pct = None, None, None

            impact_rows.append({
                'feature_group': grp_name,
                'channels': str(FEATURE_GROUPS[grp_name]),
                'ref_clean_mae_w': ref_mae_val,
                'rand20_mae_w': round(r_mae, 4) if r_mae else None,
                'rand20_degradation_w': round(r_deg, 4) if r_deg else None,
                'rand20_pct_degradation': round(r_pct, 2) if r_pct else None,
                'blackout_mae_w': round(b_mae, 4) if b_mae else None,
                'blackout_degradation_w': round(b_deg, 4) if b_deg else None,
                'blackout_pct_degradation': round(b_pct, 2) if b_pct else None,
            })

        df_impact = pd.DataFrame(impact_rows).sort_values(by='blackout_degradation_w', ascending=False)
        csv_impact = self.output_dir / 'T7_CHANNEL_IMPACT.csv'
        df_impact.to_csv(csv_impact, index=False)
        print(f"  Exported: {csv_impact} ({len(df_impact)} rows)", flush=True)

        # 6. Generate Publication Diagnostic Plots
        self._generate_plots(df_summary, df_audit, df_impact)

        return {
            'df_audit': df_audit,
            'df_per_cond': df_per_cond,
            'df_summary': df_summary,
            'df_impact': df_impact,
        }

    def _generate_plots(
        self,
        df_summary: pd.DataFrame,
        df_audit: pd.DataFrame,
        df_impact: pd.DataFrame,
    ):
        """Generate high-resolution publication diagnostic plots for T7."""
        plots_dir = self.output_dir / 'plots'
        t1_sum = df_summary[df_summary['period'] == 'test_t1']

        # 1. Degradation Curves: Random vs Contiguous across Missingness Levels
        fig, ax = plt.subplots(figsize=(10, 5.5))
        levels = [0.0, 0.05, 0.10, 0.20, 0.30]
        x_pcts = [0, 5, 10, 20, 30]

        # Random
        rand_maes = [t1_sum[t1_sum['condition_id'] == f"cond_rand_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] if p > 0 else t1_sum[t1_sum['condition_id'] == 'cond_00_ref_0pct']['pooled_mae_mean_w'].values[0] for p in levels]
        ax.plot(x_pcts, rand_maes, marker='o', linewidth=2.2, color='#1f77b4', label='Random Missingness (Scattered)')

        # Contiguous L=3 (30 min)
        l3_maes = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l3_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] if p > 0 else rand_maes[0] for p in levels]
        ax.plot(x_pcts, l3_maes, marker='s', linewidth=2.0, color='#2ca02c', linestyle='--', label='Contiguous L=3 (30 min, Fully FF30-Fillable)')

        # Contiguous L=6 (1 hour)
        l6_maes = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l6_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] if p > 0 else rand_maes[0] for p in levels]
        ax.plot(x_pcts, l6_maes, marker='^', linewidth=2.2, color='#ff7f0e', label='Contiguous L=6 (1 hour, Partial FF30-Fillable)')

        # Contiguous L=12 (2 hours)
        l12_maes = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l12_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] if p > 0 else rand_maes[0] for p in levels]
        ax.plot(x_pcts, l12_maes, marker='d', linewidth=2.2, color='#d62728', label='Contiguous L=12 (2 hours, Severe Outage)')

        ax.axhline(33.07, color='black', linestyle=':', linewidth=1.5, label='Clean Persistence (33.07 W)')
        ax.set_xlabel('Artificial Missingness Injected (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Canonical T1 Test MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Forecasting Degradation under Missing Operational Telemetry', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        p1 = plots_dir / 't7_missingness_degradation_curves.png'
        plt.savefig(p1, dpi=300)
        plt.close()
        print(f"  Saved plot: {p1}", flush=True)

        # 2. Block Length Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5.5))
        pct_labels = ['5% Missing', '10% Missing', '20% Missing', '30% Missing']
        x = np.arange(len(pct_labels))
        width = 0.20

        vals_l3 = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l3_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] for p in [0.05, 0.10, 0.20, 0.30]]
        vals_l6 = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l6_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] for p in [0.05, 0.10, 0.20, 0.30]]
        vals_l12 = [t1_sum[t1_sum['condition_id'] == f"cond_cont_l12_{int(p*100):02d}pct"]['pooled_mae_mean_w'].values[0] for p in [0.05, 0.10, 0.20, 0.30]]

        ax.bar(x - width, vals_l3, width, label='L=3 steps (30 min)', color='#2ca02c', alpha=0.85, edgecolor='black')
        ax.bar(x, vals_l6, width, label='L=6 steps (1 hour)', color='#ff7f0e', alpha=0.85, edgecolor='black')
        ax.bar(x + width, vals_l12, width, label='L=12 steps (2 hours)', color='#d62728', alpha=0.85, edgecolor='black')

        ax.axhline(rand_maes[0], color='blue', linestyle='--', linewidth=1.2, label=f'Clean Reference ({rand_maes[0]:.2f} W)')
        ax.set_ylabel('Pooled Test MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Impact of Sensor Outage Duration (Block Length) on Forecasting Accuracy', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pct_labels, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        p2 = plots_dir / 't7_block_length_comparison.png'
        plt.savefig(p2, dpi=300)
        plt.close()
        print(f"  Saved plot: {p2}", flush=True)

        # 3. Channel Group Sensitivity / Ablation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        grp_labels = [g.replace('_', ' ').title() for g in df_impact['feature_group']]
        r20_deg = df_impact['rand20_degradation_w'].values
        blk_deg = df_impact['blackout_degradation_w'].values

        colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

        # 20% Random
        bars1 = ax1.bar(grp_labels, r20_deg, color=colors, alpha=0.85, edgecolor='black', width=0.5)
        ax1.set_ylabel('MAE Degradation vs. Clean (Watts)', fontsize=11, fontweight='bold')
        ax1.set_title('Feature Group Sensitivity (20% Random Missing)', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        for b, v in zip(bars1, r20_deg):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"+{v:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 100% Blackout
        bars2 = ax2.bar(grp_labels, blk_deg, color=colors, alpha=0.85, edgecolor='black', width=0.5)
        ax2.set_ylabel('MAE Degradation vs. Clean (Watts)', fontsize=11, fontweight='bold')
        ax2.set_title('Feature Group Sensitivity (100% Sensor Blackout)', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        for b, v in zip(bars2, blk_deg):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.05, f"+{v:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        p3 = plots_dir / 't7_channel_group_sensitivity.png'
        plt.savefig(p3, dpi=300)
        plt.close()
        print(f"  Saved plot: {p3}", flush=True)

        # 4. Temporal Robustness across Deployment Windows
        fig, ax = plt.subplots(figsize=(10, 5.5))
        periods = ['test_t1', 'test_q1', 'test_march']
        period_labels = ['Canonical T1\n(Jan 28 - Feb 04)', 'Q1 Extended\n(Jan 28 - Apr 01)', 'March Window\n(Mar 01 - Apr 01)']
        x = np.arange(len(periods))
        width = 0.20

        for idx, (cid, lbl, col) in enumerate([
            ('cond_00_ref_0pct', 'Clean Reference (0%)', '#1f77b4'),
            ('cond_rand_10pct', 'Random 10% Missing', '#2ca02c'),
            ('cond_cont_l6_10pct', 'Contiguous L=6 10% Missing', '#ff7f0e'),
            ('cond_cont_l6_20pct', 'Contiguous L=6 20% Missing', '#d62728'),
        ]):
            vals = [df_summary[(df_summary['period'] == p) & (df_summary['condition_id'] == cid)]['pooled_mae_mean_w'].values[0] for p in periods]
            ax.bar(x + (idx - 1.5) * width, vals, width, label=lbl, color=col, alpha=0.85, edgecolor='black')

        ax.set_ylabel('Pooled MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Generalization under Telemetry Missingness Across Windows', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(period_labels, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        p4 = plots_dir / 't7_temporal_robustness.png'
        plt.savefig(p4, dpi=300)
        plt.close()
        print(f"  Saved plot: {p4}", flush=True)


if __name__ == '__main__':
    manager = T7MissingTelemetryManager(device_str='cuda:1')
    results = manager.run_all_robustness_experiments(seeds=[42, 43, 44])
    print("\n[T7Manager] All robustness experiments completed successfully!", flush=True)
