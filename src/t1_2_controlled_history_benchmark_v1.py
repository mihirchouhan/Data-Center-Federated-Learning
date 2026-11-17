"""Task T1.2 — Controlled History-Length Benchmark Execution Runner v1.

Executes the primary controlled benchmark comparing 1-hour (H=6), 6-hour (H=36),
and 24-hour (H=144) input histories predicting next-hour mean node IT power in Watts
on the CINECA M100 supercomputer cluster.

Features:
- Pure offline trace simulation on frozen Candidate 1 dataset.
- Exact Candidate 1 FF30 protocol:
    Train: Jan 01–Jan 22, 2022 UTC (394,100 windows, 979 nodes)
    Val:   Jan 22–Jan 28, 2022 UTC (157,484 windows, 929 nodes)
    Test:  Jan 28–Feb 04, 2022 UTC (243,138 windows, 978 nodes, Node 704 excluded)
- Secondary Option A No-Fill sensitivity evaluation on 50,392 un-imputed windows across 635 nodes.
- Pre-flight residual-anchor identity verification across H in {6, 36, 144}.
- Sequential execution across 9 FedAvg fits:
    1-hour (H=6):   Seeds 42, 43, 44 (200 rounds each)
    6-hour (H=36):  Seeds 42, 43, 44 (200 rounds each)
    24-hour (H=144): Seeds 42, 43, 44 (200 rounds each)
- SequenceGRU architecture: 49,561 parameters (hidden_dim=64, 2 layers, node_embed=8, dropout=0.1).
- Weather channels [7, 8] masked to 0.0 (consistent with T5 weather ablation and T6).
- Mandatory real-time terminal progress logging via DualLogger and FedAvgProgressLogger.
- Comprehensive evaluation: Pooled and Macro MAE/RMSE, Persistence comparison, Node win rate.
- Automated generation of structured CSVs and diagnostic plots.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import csv
from datetime import datetime, timezone
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
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.history_length_experiment_v1 import SequenceGRU
from src.history_length_real_pilot_v1 import Candidate1RealData
from src.progress_logging import (
    DualLogger,
    FedAvgProgressLogger,
    calculate_eta,
    format_time,
)

KW_TO_WATTS = 1000.0
HISTORIES = (6, 36, 144)
HISTORY_HOURS = {6: 1.0, 36: 6.0, 144: 24.0}
SEEDS = (42, 43, 44)
HORIZON = 6
OPERATIONAL_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12)
CANDIDATE_1_BINS = {
    'train': (0, 3024),
    'val': (3024, 3888),
    'test': (3888, 4896),
}


class T12BenchmarkManager:
    """Orchestrates the controlled 1h/6h/24h history length benchmark."""

    def __init__(
        self,
        output_dir: Path,
        device: Optional[torch.device] = None,
        rounds: int = 200,
        clients_per_round: int = 98,
        batch_size: int = 64,
        lr: float = 0.001,
        eval_batch_size: int = 4096,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir / 'predictions'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = DualLogger(
            log_path=self.output_dir / 'progress.log',
            auto_timestamp=True,
        )

        # Select GPU device (prefer cuda:1, fallback cuda:0, CPU)
        if device is None:
            if torch.cuda.is_available():
                chosen = None
                for dev_id in [1, 0]:
                    try:
                        _ = torch.zeros(1, device=f'cuda:{dev_id}')
                        chosen = torch.device(f'cuda:{dev_id}')
                        break
                    except Exception:
                        continue
                self.device = chosen if chosen is not None else torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.rounds = rounds
        self.clients_per_round = clients_per_round
        self.batch_size = batch_size
        self.lr = lr
        self.eval_batch_size = eval_batch_size

        self.logger.write_line('================================================================================')
        self.logger.write_line('TASK T1.2: CONTROLLED HISTORY-LENGTH BENCHMARK INITIALIZATION')
        self.logger.write_line(f'Output Directory: {self.output_dir}')
        self.logger.write_line(f'Execution Device: {self.device}')
        self.logger.write_line(f'Rounds: {self.rounds} | Clients/Round: {self.clients_per_round} | LR: {self.lr}')
        self.logger.write_line('================================================================================')

        # Load underlying telemetry data
        self.logger.write_line('Loading memory-mapped Candidate 1 telemetry arrays...')
        self.data = Candidate1RealData()
        self.node_count = self.data.node_count  # 979
        self.train_origins = self.data.valid_origins['train']
        self.val_origins = self.data.valid_origins['val']
        self.test_origins = self.data.valid_origins['test']

        # Discover Option A No-Fill test origins
        self.logger.write_line('Discovering Secondary Option A No-Fill test origins...')
        self.opt_a_origins = self._discover_option_a_origins()

        # Compute strictly training-set residual moments
        self._compute_training_residual_moments()

    def _discover_option_a_origins(self) -> Dict[int, np.ndarray]:
        """Discover un-imputed test origins (strictly age == 0 on operational channels)."""
        lo, hi = CANDIDATE_1_BINS['test']
        opt_a_origins: Dict[int, np.ndarray] = {}
        tot_windows = 0
        active_nodes = 0

        for n in range(self.node_count):
            s_ages = self.data.ages[n, :, :][:, OPERATIONAL_CHANNELS]
            bad_step = s_ages != 0
            hp = np.r_[0, np.cumsum(bad_step.any(axis=1))]
            tp = np.r_[0, np.cumsum(~np.isfinite(self.data.power[n]))]

            origins = np.arange(lo, hi, dtype=np.int64)
            inside = origins + HORIZON <= hi
            enough = origins >= 144
            ok_hist = hp[origins] == hp[origins - 144]
            ok_targ = tp[origins + HORIZON] == tp[origins]

            acc = enough & inside & ok_hist & ok_targ
            accepted_origins = origins[acc]
            opt_a_origins[n] = accepted_origins
            tot_windows += len(accepted_origins)
            if len(accepted_origins) > 0:
                active_nodes += 1

        self.logger.write_line(f'Option A Test Discovery: {tot_windows:,} windows across {active_nodes} active nodes')
        assert tot_windows == 50392, f'Expected 50,392 Option A windows, got {tot_windows}'
        assert active_nodes == 635, f'Expected 635 Option A nodes, got {active_nodes}'
        return opt_a_origins

    def verify_residual_anchor_identity(self) -> None:
        """Verify that y_last(t) is 100% numerically identical across H in {6, 36, 144}."""
        self.logger.write_line('\n================================================================================')
        self.logger.write_line('PRE-FLIGHT RESIDUAL-ANCHOR IDENTITY VERIFICATION')
        self.logger.write_line('================================================================================')

        splits_to_check = [
            ('train', self.train_origins, 394100, 979),
            ('val', self.val_origins, 157484, 929),
            ('test (primary)', self.test_origins, 243138, 978),
            ('test (option_a)', self.opt_a_origins, 50392, 635),
        ]

        for split_label, origins_dict, expected_win, expected_nodes in splits_to_check:
            tot_checked = 0
            active_checked = 0
            self.logger.write_line(f'Auditing residual-anchor across H=6, 36, 144 on {split_label}...')

            for n in range(self.node_count):
                orig = origins_dict[n]
                if len(orig) == 0:
                    continue
                active_checked += 1
                tot_checked += len(orig)

                _, _, p6 = self.data.get_client_batch(n, orig, 6)
                _, _, p36 = self.data.get_client_batch(n, orig, 36)
                _, _, p144 = self.data.get_client_batch(n, orig, 144)

                if not np.array_equal(p6, p36):
                    raise ValueError(f'CRITICAL: Residual-anchor mismatch between H=6 and H=36 on {split_label}, node {n}')
                if not np.array_equal(p36, p144):
                    raise ValueError(f'CRITICAL: Residual-anchor mismatch between H=36 and H=144 on {split_label}, node {n}')

            assert tot_checked == expected_win, f'Expected {expected_win} windows, audited {tot_checked}'
            assert active_checked == expected_nodes, f'Expected {expected_nodes} nodes, audited {active_checked}'
            self.logger.write_line(f'  [PASS] {split_label}: 100% bitwise identity verified across all {tot_checked:,} windows ({active_checked} nodes)')

        self.logger.write_line('Pre-flight Residual Anchor Identity Verification: CERTIFIED IDENTICAL.')

    def _compute_training_residual_moments(self) -> None:
        """Compute residual moments mu_r, sigma_r strictly on Candidate 1 train origins [0, 3024)."""
        self.logger.write_line('Computing training residual moments on [0, 3024)...')
        all_res_kw = []

        for n in range(self.node_count):
            orig = self.train_origins[n]
            if len(orig) == 0:
                continue
            _, by, pers_w = self.data.get_client_batch(n, orig, 36)
            y_kw = by.numpy() * self.data.scalers['y_std'][0] + self.data.scalers['y_mean'][0]
            pers_kw = pers_w[:, None] / 1000.0
            all_res_kw.append(y_kw - pers_kw)

        all_res_kw = np.vstack(all_res_kw)
        self.mu_r_kw = float(np.mean(all_res_kw))
        self.sigma_r_kw = float(np.std(all_res_kw))
        self.y_mean_kw = float(self.data.scalers['y_mean'][0])
        self.y_std_kw = float(self.data.scalers['y_std'][0])

        self.logger.write_line(
            f'Training Residual Moments: mu_r = {self.mu_r_kw:.8f} kW ({self.mu_r_kw*KW_TO_WATTS:.4f} W), '
            f'sigma_r = {self.sigma_r_kw:.8f} kW ({self.sigma_r_kw*KW_TO_WATTS:.4f} W)'
        )
        self.logger.write_line(
            f'Training Power Moments: y_mean = {self.y_mean_kw:.6f} kW ({self.y_mean_kw*KW_TO_WATTS:.2f} W), '
            f'y_std = {self.y_std_kw:.6f} kW ({self.y_std_kw*KW_TO_WATTS:.2f} W)'
        )

    def prepare_history_datasets(
        self,
        H: int,
    ) -> Tuple[
        Dict[int, Dict[str, Any]],  # train_clients
        Dict[str, Any],             # val_data
        Dict[str, Any],             # test_primary_data
        Dict[str, Any],             # test_option_a_data
    ]:
        """Pre-extract all tensors into host RAM for a specific history length H."""
        h_hours = HISTORY_HOURS[H]
        self.logger.write_line(f'\n--- Pre-extracting In-Memory Datasets for H={H} ({h_hours:.1f}h history) ---')
        t0 = time.time()

        # 1. Training clients
        train_clients: Dict[int, Dict[str, Any]] = {}
        total_train_win = 0
        for n in range(self.node_count):
            orig = self.train_origins[n]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self.data.get_client_batch(n, orig, H)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            pers_kw = pers_w[:, None] / 1000.0
            res_kw = y_kw - pers_kw
            res_scaled = (res_kw - self.mu_r_kw) / self.sigma_r_kw

            train_clients[n] = {
                'x': bx,  # (N, H, 14) with weather channels [7, 8] masked to 0.0
                'y_res': torch.from_numpy(res_scaled.astype(np.float32)),
                'pers_w': pers_w,
                'n_samples': len(orig),
            }
            total_train_win += len(orig)

        # 2. Validation dataset
        val_x_list, val_pers_list, val_true_list, val_node_list = [], [], [], []
        for n in range(self.node_count):
            orig = self.val_origins[n]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self.data.get_client_batch(n, orig, H)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            true_w = (y_kw * KW_TO_WATTS).astype(np.float32)

            val_x_list.append(bx)
            val_pers_list.append(pers_w)
            val_true_list.append(true_w)
            val_node_list.append(np.full(len(orig), n, dtype=np.int32))

        val_data = {
            'x': torch.cat(val_x_list, dim=0),
            'pers_w': np.concatenate(val_pers_list, axis=0),
            'true_w': np.vstack(val_true_list).squeeze(-1),
            'node_ids': np.concatenate(val_node_list, axis=0),
        }

        # 3. Primary Test dataset
        test_x_list, test_pers_list, test_true_list, test_node_list = [], [], [], []
        for n in range(self.node_count):
            orig = self.test_origins[n]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self.data.get_client_batch(n, orig, H)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            true_w = (y_kw * KW_TO_WATTS).astype(np.float32)

            test_x_list.append(bx)
            test_pers_list.append(pers_w)
            test_true_list.append(true_w)
            test_node_list.append(np.full(len(orig), n, dtype=np.int32))

        test_primary_data = {
            'x': torch.cat(test_x_list, dim=0),
            'pers_w': np.concatenate(test_pers_list, axis=0),
            'true_w': np.vstack(test_true_list).squeeze(-1),
            'node_ids': np.concatenate(test_node_list, axis=0),
        }

        # 4. Secondary Option A No-Fill Test dataset
        opt_x_list, opt_pers_list, opt_true_list, opt_node_list = [], [], [], []
        for n in range(self.node_count):
            orig = self.opt_a_origins[n]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self.data.get_client_batch(n, orig, H)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            true_w = (y_kw * KW_TO_WATTS).astype(np.float32)

            opt_x_list.append(bx)
            opt_pers_list.append(pers_w)
            opt_true_list.append(true_w)
            opt_node_list.append(np.full(len(orig), n, dtype=np.int32))

        test_option_a_data = {
            'x': torch.cat(opt_x_list, dim=0),
            'pers_w': np.concatenate(opt_pers_list, axis=0),
            'true_w': np.vstack(opt_true_list).squeeze(-1),
            'node_ids': np.concatenate(opt_node_list, axis=0),
        }

        elapsed = time.time() - t0
        self.logger.write_line(
            f'Pre-extraction complete in {elapsed:.1f}s: '
            f'Train={total_train_win:,} win ({len(train_clients)} nodes), '
            f'Val={len(val_data["x"]):,} win, '
            f'Primary Test={len(test_primary_data["x"]):,} win, '
            f'Option A Test={len(test_option_a_data["x"]):,} win'
        )

        return train_clients, val_data, test_primary_data, test_option_a_data

    def evaluate_validation(
        self,
        model: nn.Module,
        val_data: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Evaluate model on validation windows in fast mini-batches on device."""
        model.eval()
        val_x = val_data['x']
        val_pers_w = val_data['pers_w']
        val_true_w = val_data['true_w']
        N = len(val_x)

        abs_err_total = 0.0
        sq_err_total = 0.0

        with torch.no_grad():
            for i in range(0, N, self.eval_batch_size):
                end = min(i + self.eval_batch_size, N)
                bx = val_x[i:end].to(self.device)
                out = model(bx).squeeze(-1)  # (B,)

                # Vectorized reconstruction on GPU
                delta_kw = out * self.sigma_r_kw + self.mu_r_kw
                pers_b = torch.from_numpy(val_pers_w[i:end]).to(self.device)
                true_b = torch.from_numpy(val_true_w[i:end]).to(self.device)

                pw = pers_b + delta_kw * KW_TO_WATTS
                err = pw - true_b

                abs_err_total += torch.sum(torch.abs(err)).item()
                sq_err_total += torch.sum(err ** 2).item()

        mae_w = abs_err_total / N
        rmse_w = np.sqrt(sq_err_total / N)
        return float(mae_w), float(rmse_w)

    def evaluate_test_dataset(
        self,
        model: nn.Module,
        data: Dict[str, Any],
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Evaluate model on a test dataset, returning pooled, macro, and per-node metrics."""
        model.eval()
        x = data['x']
        pers_w = data['pers_w']
        true_w = data['true_w']
        node_ids = data['node_ids']
        N = len(x)

        preds_w = []
        with torch.no_grad():
            for i in range(0, N, self.eval_batch_size):
                end = min(i + self.eval_batch_size, N)
                bx = x[i:end].to(self.device)
                out = model(bx).cpu().numpy().squeeze(-1)
                delta_kw = out * self.sigma_r_kw + self.mu_r_kw
                pw = pers_w[i:end] + delta_kw * KW_TO_WATTS
                preds_w.append(pw)

        preds_w = np.concatenate(preds_w, axis=0)

        # Pooled metrics
        model_errors = preds_w - true_w
        pers_errors = pers_w - true_w

        pooled_mae = float(np.mean(np.abs(model_errors)))
        pooled_rmse = float(np.sqrt(np.mean(model_errors ** 2)))
        pers_pooled_mae = float(np.mean(np.abs(pers_errors)))
        pers_pooled_rmse = float(np.sqrt(np.mean(pers_errors ** 2)))

        # Per-node metrics
        unique_nodes = np.unique(node_ids)
        per_node_records = []
        node_maes = []
        node_rmses = []
        pers_node_maes = []
        pers_node_rmses = []
        nodes_beating_pers = 0

        for nid in unique_nodes:
            mask = (node_ids == nid)
            n_win = int(mask.sum())
            n_err = model_errors[mask]
            n_pers_err = pers_errors[mask]

            n_mae = float(np.mean(np.abs(n_err)))
            n_rmse = float(np.sqrt(np.mean(n_err ** 2)))
            p_mae = float(np.mean(np.abs(n_pers_err)))
            p_rmse = float(np.sqrt(np.mean(n_pers_err ** 2)))

            delta_mae = n_mae - p_mae
            beats = (n_mae < p_mae)
            if beats:
                nodes_beating_pers += 1

            node_maes.append(n_mae)
            node_rmses.append(n_rmse)
            pers_node_maes.append(p_mae)
            pers_node_rmses.append(p_rmse)

            per_node_records.append({
                'node_id': int(nid),
                'test_windows': n_win,
                'model_mae_w': round(n_mae, 4),
                'model_rmse_w': round(n_rmse, 4),
                'pers_mae_w': round(p_mae, 4),
                'pers_rmse_w': round(p_rmse, 4),
                'delta_mae_w': round(delta_mae, 4),
                'beats_persistence': bool(beats),
            })

        macro_mae = float(np.mean(node_maes))
        macro_rmse = float(np.mean(node_rmses))
        pers_macro_mae = float(np.mean(pers_node_maes))
        pers_macro_rmse = float(np.mean(pers_node_rmses))
        win_rate_pct = (nodes_beating_pers / len(unique_nodes)) * 100.0

        improvement_w = pers_pooled_mae - pooled_mae
        improvement_pct = (improvement_w / pers_pooled_mae) * 100.0

        return {
            'dataset_name': dataset_name,
            'windows': N,
            'nodes': len(unique_nodes),
            'pooled_mae_w': round(pooled_mae, 4),
            'pooled_rmse_w': round(pooled_rmse, 4),
            'macro_mae_w': round(macro_mae, 4),
            'macro_rmse_w': round(macro_rmse, 4),
            'pers_pooled_mae_w': round(pers_pooled_mae, 4),
            'pers_pooled_rmse_w': round(pers_pooled_rmse, 4),
            'pers_macro_mae_w': round(pers_macro_mae, 4),
            'pers_macro_rmse_w': round(pers_macro_rmse, 4),
            'mae_improvement_w': round(improvement_w, 4),
            'mae_improvement_pct': round(improvement_pct, 4),
            'node_win_rate_pct': round(win_rate_pct, 4),
            'per_node_records': per_node_records,
            'raw_predictions': preds_w.astype(np.float32),
        }

    def train_fedavg_fit(
        self,
        H: int,
        seed: int,
        train_clients: Dict[int, Dict[str, Any]],
        val_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Train a single FedAvg SequenceGRU model for 200 communication rounds."""
        h_hours = HISTORY_HOURS[H]
        config_name = f'FedAvg-{h_hours:.0f}h'

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize global model
        model = SequenceGRU(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            output_dim=1,
            node_count=self.node_count,
            node_embedding_dim=8,
            dropout=0.1,
        ).to(self.device)

        # Pre-instantiate single local model worker to avoid reallocation
        local_model = SequenceGRU(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            output_dim=1,
            node_count=self.node_count,
            node_embedding_dim=8,
            dropout=0.1,
        ).to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == 49561, f'Expected 49,561 parameters, got {param_count}'

        loss_fn = nn.MSELoss()
        eligible_clients = list(train_clients.keys())

        # Progress logger
        fed_logger = FedAvgProgressLogger(
            experiment_name='T1.2-HistoryBenchmark',
            config_name=config_name,
            seed=seed,
            total_rounds=self.rounds,
            total_clients=self.clients_per_round,
            device=str(self.device),
            output_dir=self.output_dir,
            log_filename='progress.log',
            csv_filename=f'round_progress_h{H}_seed{seed}.csv',
        )

        fed_logger.start_training()
        round_history = []
        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        best_round = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None

        fit_start_time = time.monotonic()
        recent_round_durations: List[float] = []

        for r in range(1, self.rounds + 1):
            r_start = time.monotonic()
            fed_logger.start_round(r)

            # Sample 98 clients uniformly without replacement
            rng = np.random.RandomState(seed * 1000 + r)
            sampled_clients = rng.choice(eligible_clients, size=self.clients_per_round, replace=False)

            client_weights = []
            client_states = []
            round_train_losses = []

            global_state = model.state_dict()

            for c_idx, c_id in enumerate(sampled_clients, start=1):
                c_data = train_clients[c_id]
                n_samples = c_data['n_samples']
                client_weights.append(n_samples)

                # Reset local model to current global parameters
                local_model.load_state_dict(global_state)
                local_model.train()
                optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)

                cx = c_data['x'].to(self.device)
                cy = c_data['y_res'].to(self.device)

                # Local training (1 epoch)
                indices = torch.randperm(n_samples)
                c_loss_accum = 0.0
                n_batches = 0

                for b_start in range(0, n_samples, self.batch_size):
                    b_idx = indices[b_start:b_start + self.batch_size]
                    bx = cx[b_idx]
                    by = cy[b_idx]

                    optimizer.zero_grad()
                    pred = local_model(bx)
                    loss = loss_fn(pred, by)
                    loss.backward()
                    optimizer.step()

                    c_loss_accum += loss.item()
                    n_batches += 1

                c_avg_loss = c_loss_accum / max(1, n_batches)
                round_train_losses.append(c_avg_loss)

                # Store client parameters on CPU for aggregation
                c_state = {k: v.cpu().clone() for k, v in local_model.state_dict().items()}
                client_states.append(c_state)

                # Update live terminal progress
                if c_idx % 20 == 0 or c_idx == self.clients_per_round:
                    fed_logger.update_client_progress(
                        client_idx=c_idx,
                        epoch=1,
                        total_epochs=1,
                        loss=c_avg_loss,
                    )

            # Sample-weighted aggregation
            total_weight = sum(client_weights)
            new_state = OrderedDict()
            for key in global_state.keys():
                weighted_sum = sum(
                    client_states[i][key].float() * client_weights[i]
                    for i in range(len(sampled_clients))
                )
                new_state[key] = (weighted_sum / total_weight).to(self.device)

            model.load_state_dict(new_state)

            # Evaluate on validation set
            val_mae, val_rmse = self.evaluate_validation(model, val_data)
            avg_train_loss = float(np.mean(round_train_losses))

            # Track best checkpoint based strictly on validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_round = r
                best_state = {k: v.cpu().clone() for k, v in new_state.items()}

            r_duration = time.monotonic() - r_start
            recent_round_durations.append(r_duration)
            if len(recent_round_durations) > 10:
                recent_round_durations.pop(0)

            # Complete round logging
            fed_logger.complete_round(
                train_loss=avg_train_loss,
                val_mae=val_mae,
                val_rmse=val_rmse,
            )

            round_history.append({
                'round': r,
                'train_loss': avg_train_loss,
                'val_mae_w': val_mae,
                'val_rmse_w': val_rmse,
                'best_round': best_round,
                'best_val_mae_w': best_val_mae,
                'round_time_sec': round(r_duration, 2),
            })

        fit_wall_time = time.monotonic() - fit_start_time
        fed_logger.finish_training()

        # Save best checkpoint
        ckpt_path = self.checkpoints_dir / f'fedavg_h{H}_seed{seed}_best.pt'
        torch.save({
            'model_state_dict': best_state,
            'history_steps': H,
            'history_hours': h_hours,
            'seed': seed,
            'best_round': best_round,
            'best_val_mae_w': best_val_mae,
            'best_val_rmse_w': best_val_rmse,
            'param_count': param_count,
            'wall_seconds': fit_wall_time,
        }, ckpt_path)

        fit_summary = {
            'history_steps': H,
            'history_hours': h_hours,
            'seed': seed,
            'best_round': best_round,
            'best_val_mae_w': round(best_val_mae, 4),
            'best_val_rmse_w': round(best_val_rmse, 4),
            'training_wall_seconds': round(fit_wall_time, 2),
            'checkpoint_path': str(ckpt_path),
        }

        self.logger.write_line(
            f'>>> Completed Fit: H={H} ({h_hours:.0f}h), Seed {seed} | '
            f'Best Round: {best_round} | Best Val MAE: {best_val_mae:.2f} W | Time: {fit_wall_time:.1f}s <<<'
        )

        return fit_summary, best_state, round_history

    def run_full_benchmark(self) -> None:
        """Run all 9 FedAvg fits sequentially across H in {6, 36, 144} and Seeds {42, 43, 44}."""
        # 1. Pre-flight verification
        self.verify_residual_anchor_identity()

        all_seed_results: List[Dict[str, Any]] = []
        all_per_node_records: List[Dict[str, Any]] = []
        all_option_a_records: List[Dict[str, Any]] = []
        all_round_histories: Dict[str, List[Dict[str, Any]]] = {}

        benchmark_start_time = time.monotonic()

        # Loop through the 3 history configurations
        for H in HISTORIES:
            h_hours = HISTORY_HOURS[H]
            self.logger.write_line('\n================================================================================')
            self.logger.write_line(f'BENCHMARK TRACK: {h_hours:.0f}-HOUR HISTORY (H={H} BINS)')
            self.logger.write_line('================================================================================')

            # Pre-extract in-memory datasets for this history length
            train_clients, val_data, test_primary, test_option_a = self.prepare_history_datasets(H)

            # Loop through the 3 stochastic seeds
            for seed in SEEDS:
                fit_key = f'h{H}_seed{seed}'
                self.logger.write_line(f'\n>>> STARTING FIT: H={H} ({h_hours:.0f}h), Seed {seed} <<<')

                fit_summary, best_state, round_history = self.train_fedavg_fit(
                    H=H,
                    seed=seed,
                    train_clients=train_clients,
                    val_data=val_data,
                )
                all_round_histories[fit_key] = round_history

                # Load best checkpoint into model for test evaluations
                eval_model = SequenceGRU(
                    input_dim=14,
                    hidden_dim=64,
                    num_layers=2,
                    output_dim=1,
                    node_count=self.node_count,
                    node_embedding_dim=8,
                    dropout=0.1,
                ).to(self.device)
                eval_model.load_state_dict(best_state)

                # 1. Primary Held-Out Test Evaluation (243,138 windows, 978 nodes)
                self.logger.write_line(f'Evaluating best checkpoint on Primary Held-Out Test (243,138 win)...')
                prim_eval = self.evaluate_test_dataset(eval_model, test_primary, 'primary_test')

                # Save raw primary predictions
                prim_pred_path = self.predictions_dir / f'primary_preds_h{H}_seed{seed}.npy'
                np.save(prim_pred_path, prim_eval['raw_predictions'])

                # Record per-node primary results
                for pnr in prim_eval['per_node_records']:
                    all_per_node_records.append({
                        'history_hours': h_hours,
                        'history_steps': H,
                        'seed': seed,
                        'node_id': pnr['node_id'],
                        'test_windows': pnr['test_windows'],
                        'model_mae_w': pnr['model_mae_w'],
                        'model_rmse_w': pnr['model_rmse_w'],
                        'pers_mae_w': pnr['pers_mae_w'],
                        'pers_rmse_w': pnr['pers_rmse_w'],
                        'delta_mae_w': pnr['delta_mae_w'],
                        'beats_persistence': pnr['beats_persistence'],
                    })

                # 2. Secondary Option A No-Fill Test Evaluation (50,392 windows, 635 nodes)
                self.logger.write_line(f'Evaluating best checkpoint on Secondary Option A No-Fill Test (50,392 win)...')
                opt_eval = self.evaluate_test_dataset(eval_model, test_option_a, 'option_a_nofill_test')

                # Save raw Option A predictions
                opt_pred_path = self.predictions_dir / f'option_a_preds_h{H}_seed{seed}.npy'
                np.save(opt_pred_path, opt_eval['raw_predictions'])

                # Record per-node Option A results
                for onr in opt_eval['per_node_records']:
                    all_option_a_records.append({
                        'history_hours': h_hours,
                        'history_steps': H,
                        'seed': seed,
                        'node_id': onr['node_id'],
                        'test_windows': onr['test_windows'],
                        'model_mae_w': onr['model_mae_w'],
                        'model_rmse_w': onr['model_rmse_w'],
                        'pers_mae_w': onr['pers_mae_w'],
                        'pers_rmse_w': onr['pers_rmse_w'],
                        'delta_mae_w': onr['delta_mae_w'],
                        'beats_persistence': onr['beats_persistence'],
                    })

                # Combine seed record
                combined_seed_row = {
                    'history_hours': h_hours,
                    'history_steps': H,
                    'seed': seed,
                    'best_round': fit_summary['best_round'],
                    'val_mae_w': fit_summary['best_val_mae_w'],
                    'val_rmse_w': fit_summary['best_val_rmse_w'],
                    'primary_test_windows': prim_eval['windows'],
                    'primary_test_nodes': prim_eval['nodes'],
                    'primary_pooled_mae_w': prim_eval['pooled_mae_w'],
                    'primary_pooled_rmse_w': prim_eval['pooled_rmse_w'],
                    'primary_macro_mae_w': prim_eval['macro_mae_w'],
                    'primary_macro_rmse_w': prim_eval['macro_rmse_w'],
                    'primary_pers_pooled_mae_w': prim_eval['pers_pooled_mae_w'],
                    'primary_pers_macro_mae_w': prim_eval['pers_macro_mae_w'],
                    'primary_mae_improvement_w': prim_eval['mae_improvement_w'],
                    'primary_mae_improvement_pct': prim_eval['mae_improvement_pct'],
                    'primary_node_win_rate_pct': prim_eval['node_win_rate_pct'],
                    'option_a_windows': opt_eval['windows'],
                    'option_a_nodes': opt_eval['nodes'],
                    'option_a_pooled_mae_w': opt_eval['pooled_mae_w'],
                    'option_a_pooled_rmse_w': opt_eval['pooled_rmse_w'],
                    'option_a_macro_mae_w': opt_eval['macro_mae_w'],
                    'option_a_macro_rmse_w': opt_eval['macro_rmse_w'],
                    'option_a_pers_pooled_mae_w': opt_eval['pers_pooled_mae_w'],
                    'option_a_pers_macro_mae_w': opt_eval['pers_macro_mae_w'],
                    'option_a_mae_improvement_w': opt_eval['mae_improvement_w'],
                    'option_a_mae_improvement_pct': opt_eval['mae_improvement_pct'],
                    'option_a_node_win_rate_pct': opt_eval['node_win_rate_pct'],
                    'training_wall_seconds': fit_summary['training_wall_seconds'],
                }
                all_seed_results.append(combined_seed_row)

                self.logger.write_line(
                    f'Results for H={H} Seed {seed}: '
                    f'Primary Pooled MAE = {prim_eval["pooled_mae_w"]:.2f} W (Pers {prim_eval["pers_pooled_mae_w"]:.2f} W, '
                    f'Improvement {prim_eval["mae_improvement_pct"]:.2f}%, WinRate {prim_eval["node_win_rate_pct"]:.1f}%) | '
                    f'Option A Pooled MAE = {opt_eval["pooled_mae_w"]:.2f} W (Pers {opt_eval["pers_pooled_mae_w"]:.2f} W)'
                )

            # Free up CPU memory before next history
            del train_clients, val_data, test_primary, test_option_a

        total_benchmark_time = time.monotonic() - benchmark_start_time
        self.logger.write_line('\n================================================================================')
        self.logger.write_line(f'ALL 9 FITS COMPLETED IN {total_benchmark_time:.1f}s ({format_time(total_benchmark_time)})')
        self.logger.write_line('================================================================================')

        # ----------------------------------------------------------------------
        # Export Structured Artifacts
        # ----------------------------------------------------------------------
        self._export_structured_csvs(all_seed_results, all_per_node_records, all_option_a_records)
        self._export_config_json(all_seed_results)
        self._generate_diagnostic_plots(all_seed_results, all_per_node_records, all_option_a_records, all_round_histories)

    def _export_structured_csvs(
        self,
        seed_results: List[Dict[str, Any]],
        per_node_records: List[Dict[str, Any]],
        option_a_records: List[Dict[str, Any]],
    ) -> None:
        """Export T1_2_RESULTS_BY_SEED.csv, T1_2_RESULTS_SUMMARY.csv, and per-node CSVs."""
        self.logger.write_line('Exporting structured CSV artifacts...')

        # 1. T1_2_RESULTS_BY_SEED.csv
        df_seeds = pd.DataFrame(seed_results)
        seeds_csv = self.output_dir / 'T1_2_RESULTS_BY_SEED.csv'
        df_seeds.to_csv(seeds_csv, index=False)
        self.logger.write_line(f'Saved: {seeds_csv}')

        # 2. T1_2_RESULTS_SUMMARY.csv (Grouped by history length, mean +/- std across 3 seeds)
        summary_rows = []
        for H in HISTORIES:
            h_hours = HISTORY_HOURS[H]
            sub = df_seeds[df_seeds['history_steps'] == H]

            summary_rows.append({
                'history_hours': h_hours,
                'history_steps': H,
                'seeds_count': len(sub),
                # Primary Test
                'primary_pooled_mae_mean': round(sub['primary_pooled_mae_w'].mean(), 4),
                'primary_pooled_mae_std': round(sub['primary_pooled_mae_w'].std(), 4),
                'primary_pooled_rmse_mean': round(sub['primary_pooled_rmse_w'].mean(), 4),
                'primary_pooled_rmse_std': round(sub['primary_pooled_rmse_w'].std(), 4),
                'primary_macro_mae_mean': round(sub['primary_macro_mae_w'].mean(), 4),
                'primary_macro_mae_std': round(sub['primary_macro_mae_w'].std(), 4),
                'primary_macro_rmse_mean': round(sub['primary_macro_rmse_w'].mean(), 4),
                'primary_macro_rmse_std': round(sub['primary_macro_rmse_w'].std(), 4),
                'primary_pers_pooled_mae': round(sub['primary_pers_pooled_mae_w'].mean(), 4),
                'primary_pers_macro_mae': round(sub['primary_pers_macro_mae_w'].mean(), 4),
                'primary_mae_improvement_mean_w': round(sub['primary_mae_improvement_w'].mean(), 4),
                'primary_mae_improvement_mean_pct': round(sub['primary_mae_improvement_pct'].mean(), 4),
                'primary_node_win_rate_mean_pct': round(sub['primary_node_win_rate_pct'].mean(), 4),
                # Secondary Option A Test
                'option_a_pooled_mae_mean': round(sub['option_a_pooled_mae_w'].mean(), 4),
                'option_a_pooled_mae_std': round(sub['option_a_pooled_mae_w'].std(), 4),
                'option_a_macro_mae_mean': round(sub['option_a_macro_mae_w'].mean(), 4),
                'option_a_macro_mae_std': round(sub['option_a_macro_mae_w'].std(), 4),
                'option_a_pers_pooled_mae': round(sub['option_a_pers_pooled_mae_w'].mean(), 4),
                'option_a_pers_macro_mae': round(sub['option_a_pers_macro_mae_w'].mean(), 4),
                'option_a_mae_improvement_mean_w': round(sub['option_a_mae_improvement_w'].mean(), 4),
                'option_a_mae_improvement_mean_pct': round(sub['option_a_mae_improvement_pct'].mean(), 4),
                'option_a_node_win_rate_mean_pct': round(sub['option_a_node_win_rate_pct'].mean(), 4),
                # Runtime
                'training_wall_seconds_mean': round(sub['training_wall_seconds'].mean(), 2),
            })

        df_summary = pd.DataFrame(summary_rows)
        summary_csv = self.output_dir / 'T1_2_RESULTS_SUMMARY.csv'
        df_summary.to_csv(summary_csv, index=False)
        self.logger.write_line(f'Saved: {summary_csv}')

        # 3. T1_2_PER_NODE_RESULTS.csv
        df_per_node = pd.DataFrame(per_node_records)
        per_node_csv = self.output_dir / 'T1_2_PER_NODE_RESULTS.csv'
        df_per_node.to_csv(per_node_csv, index=False)
        self.logger.write_line(f'Saved: {per_node_csv} ({len(df_per_node):,} rows)')

        # 4. T1_2_OPTION_A_NOFILL_RESULTS.csv
        df_opt_a = pd.DataFrame(option_a_records)
        opt_a_csv = self.output_dir / 'T1_2_OPTION_A_NOFILL_RESULTS.csv'
        df_opt_a.to_csv(opt_a_csv, index=False)
        self.logger.write_line(f'Saved: {opt_a_csv} ({len(df_opt_a):,} rows)')

    def _export_config_json(self, seed_results: List[Dict[str, Any]]) -> None:
        """Export comprehensive execution configuration receipt."""
        config = {
            'task_id': 'T1.2',
            'task_title': 'Controlled History-Length Benchmark: 1h vs 6h vs 24h',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'protocol': 'Candidate 1 FF30 Protocol (Certified T1.1)',
            'missing_data_policy': 'FF30 (Causal forward-fill <= 30 min, boundary-reset)',
            'horizon_steps': HORIZON,
            'horizon_minutes': 60,
            'target_description': 'Next-hour mean node IT power in Watts',
            'histories_evaluated': [
                {'hours': 1.0, 'steps': 6},
                {'hours': 6.0, 'steps': 36},
                {'hours': 24.0, 'steps': 144},
            ],
            'stochastic_seeds': list(SEEDS),
            'model_architecture': {
                'class': 'SequenceGRU',
                'input_dim': 14,
                'hidden_dim': 64,
                'num_layers': 2,
                'node_count': self.node_count,
                'node_embedding_dim': 8,
                'dropout': 0.1,
                'output_dim': 1,
                'total_parameters': 49561,
            },
            'federated_parameters': {
                'algorithm': 'FedAvg (Sample-Weighted Aggregation)',
                'rounds': self.rounds,
                'clients_per_round': self.clients_per_round,
                'client_sampling_fraction': 0.10,
                'local_epochs': 1,
                'local_batch_size': self.batch_size,
                'learning_rate': self.lr,
                'optimizer': 'Adam',
                'loss_function': 'MSELoss',
            },
            'splits': {
                'train': {'start': '2022-01-01T00:00:00Z', 'end': '2022-01-22T00:00:00Z', 'windows': 394100, 'nodes': 979},
                'val':   {'start': '2022-01-22T00:00:00Z', 'end': '2022-01-28T00:00:00Z', 'windows': 157484, 'nodes': 929},
                'test_primary': {'start': '2022-01-28T00:00:00Z', 'end': '2022-02-04T00:00:00Z', 'windows': 243138, 'nodes': 978},
                'test_option_a_nofill': {'start': '2022-01-28T00:00:00Z', 'end': '2022-02-04T00:00:00Z', 'windows': 50392, 'nodes': 635},
            },
            'normalization': {
                'mu_r_kw': self.mu_r_kw,
                'sigma_r_kw': self.sigma_r_kw,
                'y_mean_kw': self.y_mean_kw,
                'y_std_kw': self.y_std_kw,
                'formula': 'pred_w = pers_w + (pred_norm_res * sigma_r + mu_r) * 1000.0',
            },
            'hardware': {
                'device': str(self.device),
                'device_name': torch.cuda.get_device_name(self.device) if 'cuda' in str(self.device) else 'CPU',
            },
        }

        config_path = self.output_dir / 'T1_2_CONFIG.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        self.logger.write_line(f'Saved: {config_path}')

    def _generate_diagnostic_plots(
        self,
        seed_results: List[Dict[str, Any]],
        per_node_records: List[Dict[str, Any]],
        option_a_records: List[Dict[str, Any]],
        round_histories: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Generate all required publication-quality diagnostic plots."""
        self.logger.write_line('Generating publication-quality diagnostic plots...')
        df_seeds = pd.DataFrame(seed_results)
        df_per_node = pd.DataFrame(per_node_records)
        df_opt_a = pd.DataFrame(option_a_records)

        # Plot 1: History Comparison (Pooled & Macro MAE/RMSE with error bars)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        h_labels = ['1-Hour (H=6)', '6-Hour (H=36)', '24-Hour (H=144)']
        h_vals = [6, 36, 144]

        # MAE panel
        p_mae_means = [df_seeds[df_seeds['history_steps'] == h]['primary_pooled_mae_w'].mean() for h in h_vals]
        p_mae_stds = [df_seeds[df_seeds['history_steps'] == h]['primary_pooled_mae_w'].std() for h in h_vals]
        m_mae_means = [df_seeds[df_seeds['history_steps'] == h]['primary_macro_mae_w'].mean() for h in h_vals]
        m_mae_stds = [df_seeds[df_seeds['history_steps'] == h]['primary_macro_mae_w'].std() for h in h_vals]
        pers_pooled_mae = df_seeds['primary_pers_pooled_mae_w'].iloc[0]
        pers_macro_mae = df_seeds['primary_pers_macro_mae_w'].iloc[0]

        x_pos = np.arange(len(h_vals))
        width = 0.35

        axes[0].bar(x_pos - width/2, p_mae_means, width, yerr=p_mae_stds, label='Pooled MAE', color='#1a73e8', capsize=5, alpha=0.85)
        axes[0].bar(x_pos + width/2, m_mae_means, width, yerr=m_mae_stds, label='Macro MAE', color='#34a853', capsize=5, alpha=0.85)
        axes[0].axhline(pers_pooled_mae, color='#ea4335', linestyle='--', linewidth=2, label=f'Persistence Pooled ({pers_pooled_mae:.2f} W)')
        axes[0].axhline(pers_macro_mae, color='#fbbc04', linestyle=':', linewidth=2, label=f'Persistence Macro ({pers_macro_mae:.2f} W)')
        axes[0].set_ylabel('Mean Absolute Error (Watts)', fontsize=11)
        axes[0].set_title('T1.2 Forecast Accuracy: MAE across History Lengths', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(h_labels, fontsize=10)
        axes[0].legend(fontsize=9)
        axes[0].grid(axis='y', linestyle='--', alpha=0.3)

        # RMSE panel
        p_rmse_means = [df_seeds[df_seeds['history_steps'] == h]['primary_pooled_rmse_w'].mean() for h in h_vals]
        p_rmse_stds = [df_seeds[df_seeds['history_steps'] == h]['primary_pooled_rmse_w'].std() for h in h_vals]
        m_rmse_means = [df_seeds[df_seeds['history_steps'] == h]['primary_macro_rmse_w'].mean() for h in h_vals]
        m_rmse_stds = [df_seeds[df_seeds['history_steps'] == h]['primary_macro_rmse_w'].std() for h in h_vals]
        pers_pooled_rmse = df_seeds['primary_pers_pooled_rmse_w'].iloc[0]
        pers_macro_rmse = df_seeds['primary_pers_macro_rmse_w'].iloc[0]

        axes[1].bar(x_pos - width/2, p_rmse_means, width, yerr=p_rmse_stds, label='Pooled RMSE', color='#1a73e8', capsize=5, alpha=0.85)
        axes[1].bar(x_pos + width/2, m_rmse_means, width, yerr=m_rmse_stds, label='Macro RMSE', color='#34a853', capsize=5, alpha=0.85)
        axes[1].axhline(pers_pooled_rmse, color='#ea4335', linestyle='--', linewidth=2, label=f'Persistence Pooled ({pers_pooled_rmse:.2f} W)')
        axes[1].axhline(pers_macro_rmse, color='#fbbc04', linestyle=':', linewidth=2, label=f'Persistence Macro ({pers_macro_rmse:.2f} W)')
        axes[1].set_ylabel('Root Mean Square Error (Watts)', fontsize=11)
        axes[1].set_title('T1.2 Forecast Accuracy: RMSE across History Lengths', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(h_labels, fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plot1_path = self.plots_dir / 't1_2_history_comparison_mae_rmse.png'
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        self.logger.write_line(f'Saved: {plot1_path}')

        # Plot 2: Round Convergence Curves
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {6: '#1a73e8', 36: '#34a853', 144: '#f2994a'}

        for H in HISTORIES:
            h_hours = HISTORY_HOURS[H]
            for s_idx, s in enumerate(SEEDS):
                key = f'h{H}_seed{s}'
                if key in round_histories:
                    rounds_data = round_histories[key]
                    r_nums = [rd['round'] for rd in rounds_data]
                    val_maes = [rd['val_mae_w'] for rd in rounds_data]
                    lbl = f'{h_hours:.0f}h History (Seed {s})' if s_idx == 0 else None
                    alpha = 0.9 if s == 42 else 0.5
                    ls = '-' if s == 42 else ('--' if s == 43 else ':')
                    ax.plot(r_nums, val_maes, color=colors[H], linestyle=ls, alpha=alpha, label=lbl, linewidth=1.8)

        ax.set_xlabel('Federated Communication Round', fontsize=11)
        ax.set_ylabel('Validation MAE (Watts)', fontsize=11)
        ax.set_title('T1.2 Federated Convergence: Validation MAE across 200 Rounds', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plot2_path = self.plots_dir / 't1_2_round_convergence.png'
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        self.logger.write_line(f'Saved: {plot2_path}')

        # Plot 3: Per-Node Improvement Distribution (Delta vs Persistence)
        fig, ax = plt.subplots(figsize=(10, 6))
        for H in HISTORIES:
            h_hours = HISTORY_HOURS[H]
            sub = df_per_node[(df_per_node['history_steps'] == H) & (df_per_node['seed'] == 42)]
            deltas = sub['delta_mae_w']
            ax.hist(deltas, bins=50, alpha=0.4, label=f'{h_hours:.0f}h History (Mean {deltas.mean():.2f} W)', color=colors[H], density=True)
            ax.axvline(deltas.mean(), color=colors[H], linestyle='--', linewidth=2)

        ax.axvline(0.0, color='#ea4335', linestyle='-', linewidth=2, label='Persistence Baseline (0.0 W)')
        ax.set_xlabel('Per-Node MAE Delta vs. Persistence Baseline (Watts, <0 is improvement)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('T1.2 Distribution of Per-Node Improvement over Persistence', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plot3_path = self.plots_dir / 't1_2_node_improvement_distribution.png'
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        self.logger.write_line(f'Saved: {plot3_path}')

        # Plot 4: Primary FF30 vs Secondary Option A No-Fill Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        p_opt_mae_means = [df_seeds[df_seeds['history_steps'] == h]['option_a_pooled_mae_w'].mean() for h in h_vals]
        p_opt_mae_stds = [df_seeds[df_seeds['history_steps'] == h]['option_a_pooled_mae_w'].std() for h in h_vals]

        ax.bar(x_pos - width/2, p_mae_means, width, yerr=p_mae_stds, label='Primary FF30 Test (243,138 win, 978 nodes)', color='#1a73e8', capsize=5, alpha=0.85)
        ax.bar(x_pos + width/2, p_opt_mae_means, width, yerr=p_opt_mae_stds, label='Secondary Option A No-Fill (50,392 win, 635 nodes)', color='#f2994a', capsize=5, alpha=0.85)

        ax.set_ylabel('Pooled MAE (Watts)', fontsize=11)
        ax.set_title('T1.2 Evaluation Sensitivity: Primary FF30 vs. Option A No-Fill', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(h_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plot4_path = self.plots_dir / 't1_2_option_a_nofill_comparison.png'
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        self.logger.write_line(f'Saved: {plot4_path}')


def main():
    parser = argparse.ArgumentParser(description='Task T1.2 Controlled History-Length Benchmark Runner')
    parser.add_argument('--output-dir', type=str, default='output/experiments/t1_2_controlled_history_benchmark_v1')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    dev = torch.device(args.device) if torch.cuda.is_available() and 'cuda' in args.device else torch.device('cpu')
    manager = T12BenchmarkManager(
        output_dir=Path(args.output_dir),
        device=dev,
        rounds=args.rounds,
    )
    manager.run_full_benchmark()


if __name__ == '__main__':
    main()
