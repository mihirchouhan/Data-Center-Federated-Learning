"""
Task T5: Environmental / Weather Ablation — Controlled Experiment Runner.

Evaluates whether outdoor environmental telemetry (outdoor temperature and humidity)
contributes to federated next-hour node IT power forecasting on the CINECA M100 cluster.

Four controlled configurations:
- Condition A: No outdoor weather (T4 frozen masked-weather reference).
- Condition B: Outdoor temperature only (channel 7 active, channel 8 masked).
- Condition C: Outdoor humidity only (channel 7 masked, channel 8 active).
- Condition D: Outdoor temperature + humidity (both channels 7 and 8 active).

All configurations use:
- 6-hour history (36 ten-minute bins)
- 2-layer SequenceGRU, hidden_dim=64, node_embedding_dim=8 (49,561 params)
- Persistence Residual learning target with L1 (MAE) loss
- Sample-weighted FedAvg, 200 rounds, 98 clients/round, local epochs=1, lr=0.001
- Seeds: 42, 43, 44
- Real-time terminal progress logging via src.progress_logging
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    FedAvgProgressLogger,
    calculate_eta,
    format_time,
)
from src.temporal_generalization_v1 import ExtendedTelemetryDataset

KW_TO_WATTS = 1000.0
HORIZON_STEPS = 6
HISTORY_STEPS = 36
SEEDS = (42, 43, 44)

WEATHER_CHANNELS = {
    'outdoor_temperature_c': 7,
    'outdoor_humidity_percent': 8,
}

WEATHER_CONDITIONS = {
    'condition_a_no_weather': {
        'label': 'No Weather (T4 Reference)',
        'zero_channels': [7, 8],
        'temp_active': False,
        'humidity_active': False,
    },
    'condition_b_temp_only': {
        'label': 'Temperature Only',
        'zero_channels': [8],
        'temp_active': True,
        'humidity_active': False,
    },
    'condition_c_humidity_only': {
        'label': 'Humidity Only',
        'zero_channels': [7],
        'temp_active': False,
        'humidity_active': True,
    },
    'condition_d_temp_and_humidity': {
        'label': 'Temperature + Humidity',
        'zero_channels': [],
        'temp_active': True,
        'humidity_active': True,
    },
}


def apply_weather_mask(x: torch.Tensor, zero_channels: Sequence[int]) -> torch.Tensor:
    """Zero out specified weather channels in input tensor."""
    if not zero_channels:
        return x
    x_masked = x.clone()
    x_masked[:, :, zero_channels] = 0.0
    return x_masked


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class T5ExperimentManager:
    """
    Orchestrates Task T5 Weather Ablation training, evaluation, and reporting.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = ROOT / 'output/experiments/t5_weather_ablation_v1',
        device_str: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)

        if device_str is None:
            if torch.cuda.is_available():
                # Prefer cuda:1 if available (more free memory on shared node)
                chosen = 1 if torch.cuda.device_count() > 1 else 0
                self.device = torch.device(f'cuda:{chosen}')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_str)

        print(f"[T5Manager] Initialized on device: {self.device}", flush=True)

        # 1. Load Candidate 1 Dataset for Training & Validation
        print("[T5Manager] Loading Candidate 1 reference dataset...", flush=True)
        self.t1_data = Candidate1RealData()
        self.node_count = self.t1_data.node_count
        self.train_origins = self.t1_data.valid_origins['train']
        self.val_origins = self.t1_data.valid_origins['val']
        self.test_t1_origins = self.t1_data.valid_origins['test']

        # Moments
        self.y_mean_kw = float(self.t1_data.scalers['y_mean'][0])
        self.y_std_kw = float(self.t1_data.scalers['y_std'][0])

        # Precompute residual moments on training set
        print("[T5Manager] Computing training moments for target and persistence residual...", flush=True)
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
        print(f"  Target Power (kW): mean={self.y_mean_kw*KW_TO_WATTS:.2f} W, std={self.y_std_kw*KW_TO_WATTS:.2f} W", flush=True)
        print(f"  Residual (kW):     mean={self.res_mean_kw*KW_TO_WATTS:.4f} W, std={self.res_std_kw*KW_TO_WATTS:.4f} W", flush=True)

        # Pre-extract validation set (with unmasked weather)
        self._pre_extract_validation()

        # Pre-extract training batches resident on device for fast FedAvg rounds
        self._pre_extract_training_clients()

        # 2. Lazy load Extended Telemetry for Q1 and March evaluations
        self._ext_data: Optional[ExtendedTelemetryDataset] = None

    def get_extended_data(self) -> ExtendedTelemetryDataset:
        """Lazy loader for Q1 and March extended telemetry."""
        if self._ext_data is None:
            print("[T5Manager] Loading ExtendedTelemetryDataset for Q1/March evaluation...", flush=True)
            self._ext_data = ExtendedTelemetryDataset()
        return self._ext_data

    def _extract_unmasked_batch(
        self,
        node_idx: int,
        origins: np.ndarray,
        history_steps: int = HISTORY_STEPS,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Extract batch with TRUE weather telemetry (channels 7 and 8 unmasked).
        Causal forward-fill FF30 applied. Standardized using training moments.
        """
        H = history_steps
        B = len(origins)

        # Target in standardized residual units
        target_bins = origins[:, None] + np.arange(HORIZON_STEPS)
        targets_kw = np.mean(self.t1_data.power[node_idx, target_bins], axis=1, keepdims=True)
        y_scaled = (targets_kw - self.y_mean_kw) / self.y_std_kw

        # History features
        time_indices = origins[:, None] + np.arange(-H, 0)
        raw_hist = self.t1_data.features[node_idx, time_indices].copy()
        hist_ages = self.t1_data.ages[node_idx, time_indices]

        # Causal forward fill (FF30 up to 3 bins = 30 min)
        fill = (hist_ages > 0) & (hist_ages <= 3)
        if fill.any():
            rows, steps, cols = np.where(fill)
            src_times = time_indices[rows, steps] - hist_ages[rows, steps, cols]
            raw_hist[rows, steps, cols] = self.t1_data.features[node_idx, src_times, cols]

        # Persistence baseline in physical Watts: last observed step P[t-1]
        pers_w = (raw_hist[:, -1, 0] * KW_TO_WATTS).astype(np.float32)

        # Standardize features using training moments across all 13 telemetry channels
        scaled_hist = (raw_hist - self.t1_data.scalers['x_mean'][:13]) / self.t1_data.scalers['x_std'][:13]

        # Append categorical node index
        node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
        x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(x_scaled),
            torch.from_numpy(y_scaled.astype(np.float32)),
            pers_w,
        )

    def _pre_extract_validation(self):
        """Pre-extract validation set with full unmasked weather."""
        print("[T5Manager] Pre-extracting validation set (with unmasked weather channels)...", flush=True)
        val_x_list = []
        val_y_res_list = []
        val_pers_list = []
        val_node_list = []

        total_val_windows = 0
        for node_idx in range(self.node_count):
            orig = self.val_origins[node_idx]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self._extract_unmasked_batch(node_idx, orig, HISTORY_STEPS)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            pers_kw = pers_w[:, None] / KW_TO_WATTS
            res_kw = y_kw - pers_kw
            res_scaled = (res_kw - self.res_mean_kw) / self.res_std_kw

            val_x_list.append(bx)
            val_y_res_list.append(torch.from_numpy(res_scaled.astype(np.float32)))
            val_pers_list.append(pers_w)
            val_node_list.append(np.full(len(orig), node_idx, dtype=np.int32))
            total_val_windows += len(orig)

        self.val_x = torch.cat(val_x_list, dim=0).to(self.device)
        self.val_y_res = torch.cat(val_y_res_list, dim=0).to(self.device)
        self.val_pers_w = np.concatenate(val_pers_list, axis=0)
        self.val_node_ids = np.concatenate(val_node_list, axis=0)

        # Ground truth target in physical Watts
        val_targets_kw = self.val_y_res.cpu().numpy() * self.res_std_kw + self.res_mean_kw + (self.val_pers_w[:, None] / KW_TO_WATTS)
        self.val_true_w = (val_targets_kw * KW_TO_WATTS).astype(np.float32)

        pers_mae = float(np.mean(np.abs(self.val_pers_w[:, None] - self.val_true_w)))
        pers_rmse = float(np.sqrt(np.mean((self.val_pers_w[:, None] - self.val_true_w) ** 2)))
        print(f"  Validation Windows: {total_val_windows:,} across {self.node_count} nodes", flush=True)
        print(f"  Persistence Baseline on Val: MAE={pers_mae:.2f} W, RMSE={pers_rmse:.2f} W", flush=True)
        self.val_pers_mae = pers_mae
        self.val_pers_rmse = pers_rmse

    def _pre_extract_training_clients(self):
        """Pre-extract training batches per client and place directly on GPU."""
        print("[T5Manager] Pre-extracting training batches per client onto GPU...", flush=True)
        self.train_clients: Dict[int, Dict[str, Any]] = {}
        total_train_windows = 0

        for node_idx in range(self.node_count):
            orig = self.train_origins[node_idx]
            if len(orig) == 0:
                continue
            bx, by, pers_w = self._extract_unmasked_batch(node_idx, orig, HISTORY_STEPS)
            y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
            pers_kw = pers_w[:, None] / KW_TO_WATTS
            res_kw = y_kw - pers_kw
            res_scaled = (res_kw - self.res_mean_kw) / self.res_std_kw

            # Place directly on device for fast FedAvg rounds
            self.train_clients[node_idx] = {
                'x': bx.to(self.device),
                'y_res': torch.from_numpy(res_scaled.astype(np.float32)).to(self.device),
                'pers_w': pers_w,
                'n_samples': len(orig),
            }
            total_train_windows += len(orig)

        self.eligible_clients = list(self.train_clients.keys())
        print(f"  Training Windows: {total_train_windows:,} across {len(self.eligible_clients)} eligible clients on {self.device}", flush=True)

    def evaluate_model_on_val(
        self,
        model: nn.Module,
        zero_channels: Sequence[int],
        batch_size: int = 4096,
    ) -> Tuple[float, float]:
        """Fast validation evaluation of a model under specified weather mask."""
        model.eval()
        preds_w = []
        n_val = self.val_x.shape[0]

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                end = min(i + batch_size, n_val)
                bx = apply_weather_mask(self.val_x[i:end], zero_channels)
                pers_b = self.val_pers_w[i:end, None]

                out = model(bx).cpu().numpy()
                delta_kw = out * self.res_std_kw + self.res_mean_kw
                p_w = pers_b + delta_kw * KW_TO_WATTS
                preds_w.append(p_w)

        preds_w = np.vstack(preds_w)
        errors = preds_w - self.val_true_w
        mae_w = float(np.mean(np.abs(errors)))
        rmse_w = float(np.sqrt(np.mean(errors ** 2)))
        return mae_w, rmse_w

    def run_fedavg_training(
        self,
        condition_id: str,
        seed: int,
        rounds: int = 200,
        clients_per_round: int = 98,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Execute 200 communication rounds of FedAvg for a weather condition and seed.
        Integrates mandatory real-time terminal progress logging.
        """
        cond_meta = WEATHER_CONDITIONS[condition_id]
        cond_label = cond_meta['label']
        zero_channels = cond_meta['zero_channels']

        # Deterministic seed initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rng = np.random.default_rng(np.random.SeedSequence([seed, 50, 0, 1]))

        # Global model architecture matching T4 winner
        global_model = SequenceGRU(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=self.node_count,
            node_embedding_dim=8,
        ).to(self.device)

        param_count = sum(p.numel() for p in global_model.parameters())
        loss_fn = nn.L1Loss()

        # Mandatory Terminal & File Progress Logger
        progress_logger = FedAvgProgressLogger(
            experiment_name="T5",
            config_name=cond_label,
            seed=seed,
            total_rounds=rounds,
            total_clients=clients_per_round,
            device=str(self.device),
            output_dir=self.output_dir,
            log_filename="progress.log",
            csv_filename="round_progress.csv",
            interactive=True,
        )
        progress_logger.start_training()

        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        best_round = -1
        best_state = None

        t0 = time.time()

        for r in range(1, rounds + 1):
            progress_logger.start_round(r)

            # Sample 98 clients (10%) uniformly
            sampled_clients = rng.choice(self.eligible_clients, size=clients_per_round, replace=False)

            client_weights = []
            client_states = []
            round_loss_sum = 0.0
            round_sample_count = 0

            global_state = copy.deepcopy(global_model.state_dict())

            for c_idx, c_id in enumerate(sampled_clients, start=1):
                c_data = self.train_clients[c_id]
                n_samples = c_data['n_samples']
                client_weights.append(n_samples)

                local_model = SequenceGRU(
                    input_dim=14,
                    hidden_dim=64,
                    num_layers=2,
                    node_count=self.node_count,
                    node_embedding_dim=8,
                ).to(self.device)
                local_model.load_state_dict(global_state)
                local_model.train()

                optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)

                # Apply condition's weather mask
                cx = apply_weather_mask(c_data['x'], zero_channels)
                cy = c_data['y_res']

                client_loss_val = 0.0
                n_batches = 0

                for _ in range(local_epochs):
                    indices = torch.randperm(n_samples, device=self.device)
                    for b_start in range(0, n_samples, batch_size):
                        b_idx = indices[b_start:b_start + batch_size]
                        bx = cx[b_idx]
                        by = cy[b_idx]

                        optimizer.zero_grad()
                        pred = local_model(bx)
                        loss = loss_fn(pred, by)
                        loss.backward()
                        optimizer.step()

                        client_loss_val += loss.item()
                        n_batches += 1

                avg_c_loss = client_loss_val / max(1, n_batches)
                round_loss_sum += avg_c_loss * n_samples
                round_sample_count += n_samples

                client_states.append(local_model.state_dict())

                # Live client progress update on terminal
                progress_logger.update_client_progress(
                    client_idx=c_idx,
                    epoch=1,
                    total_epochs=local_epochs,
                    loss=float(avg_c_loss),
                )

            # Sample-weighted aggregation
            total_weight = sum(client_weights)
            new_state = OrderedDict()
            for key in global_state.keys():
                weighted_sum = sum(client_states[i][key].float() * client_weights[i] for i in range(len(sampled_clients)))
                new_state[key] = weighted_sum / total_weight

            global_model.load_state_dict(new_state)

            # Evaluate on validation set
            val_mae, val_rmse = self.evaluate_model_on_val(global_model, zero_channels)
            train_round_loss = round_loss_sum / max(1, round_sample_count)

            # Checkpoint selection strictly on validation MAE
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_round = r
                best_state = copy.deepcopy(new_state)

            # Print & persist mandatory 5-line round completion block
            progress_logger.complete_round(
                train_loss=float(train_round_loss),
                val_mae=float(val_mae),
                val_rmse=float(val_rmse),
                total_epochs=local_epochs,
            )

        progress_logger.finish_training()
        elapsed_total = time.time() - t0

        return {
            'condition_id': condition_id,
            'condition_label': cond_label,
            'seed': seed,
            'zero_channels': list(zero_channels),
            'param_count': param_count,
            'rounds': rounds,
            'best_round': best_round,
            'val_mae_w': best_val_mae,
            'val_rmse_w': best_val_rmse,
            'best_state': best_state,
            'elapsed_seconds': elapsed_total,
        }

    def evaluate_on_canonical_t1_test(
        self,
        model: nn.Module,
        zero_channels: Sequence[int],
    ) -> Dict[str, Any]:
        """Evaluate model on Canonical T1 Held-Out Test period (243,138 windows, 978 nodes)."""
        model.eval()
        preds_all = []
        trues_all = []
        pers_all = []
        node_ids_all = []

        per_node_mae = {}
        per_node_rmse = {}
        per_node_pers_mae = {}

        with torch.no_grad():
            for node_idx in range(self.node_count):
                orig = self.test_t1_origins[node_idx]
                if len(orig) == 0:
                    continue

                bx, by, pers_w = self._extract_unmasked_batch(node_idx, orig, HISTORY_STEPS)
                bx = apply_weather_mask(bx, zero_channels).to(self.device)

                out = model(bx).cpu().numpy()
                delta_kw = out * self.res_std_kw + self.res_mean_kw
                pred_w = pers_w[:, None] + delta_kw * KW_TO_WATTS

                y_kw = by.numpy() * self.y_std_kw + self.y_mean_kw
                true_w = y_kw * KW_TO_WATTS

                preds_all.append(pred_w)
                trues_all.append(true_w)
                pers_all.append(pers_w[:, None])
                node_ids_all.append(np.full(len(orig), node_idx, dtype=np.int32))

                n_err = pred_w - true_w
                n_pers_err = pers_w[:, None] - true_w
                per_node_mae[node_idx] = float(np.mean(np.abs(n_err)))
                per_node_rmse[node_idx] = float(np.sqrt(np.mean(n_err ** 2)))
                per_node_pers_mae[node_idx] = float(np.mean(np.abs(n_pers_err)))

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
        }

    def evaluate_on_extended_test(
        self,
        model: nn.Module,
        zero_channels: Sequence[int],
        period: str = 'test_q1',
    ) -> Dict[str, Any]:
        """
        Evaluate model on Q1 Extended ([3888, 12960)) or March Window ([8496, 12960)).
        """
        ext = self.get_extended_data()
        model.eval()

        # Origin range
        if period == 'test_q1':
            lo, hi = 3888, 12960
        elif period == 'test_march':
            lo, hi = 8496, 12960
        else:
            raise ValueError(f"Unknown period: {period}")

        preds_all = []
        trues_all = []
        pers_all = []
        per_node_mae = {}
        per_node_rmse = {}
        per_node_pers_mae = {}

        # Pre-filter origins
        with torch.no_grad():
            for node_idx in range(self.node_count):
                all_orig = ext.origins['t2'][node_idx]
                orig = all_orig[(all_orig >= lo) & (all_orig < hi)]
                if len(orig) == 0:
                    continue

                # Batch extraction from extended data
                # Causal forward fill applied, weather standardized
                H = HISTORY_STEPS
                B = len(orig)

                # Target in physical kW
                tb = orig[:, None] + np.arange(HORIZON_STEPS)
                targets_kw = np.mean(ext.power[node_idx, tb], axis=1, keepdims=True)

                # History
                time_indices = orig[:, None] + np.arange(-H, 0)
                raw_hist = ext.features[node_idx, time_indices].copy()
                hist_ages = ext.ages[node_idx, time_indices]

                # Causal forward fill FF30
                fill = (hist_ages > 0) & (hist_ages <= 3)
                if fill.any():
                    rows, steps, cols = np.where(fill)
                    src_times = time_indices[rows, steps] - hist_ages[rows, steps, cols]
                    raw_hist[rows, steps, cols] = ext.features[node_idx, src_times, cols]

                pers_w = (raw_hist[:, -1, 0] * KW_TO_WATTS).astype(np.float32)

                # Standardize
                scaled_hist = (raw_hist - ext.scalers['x_mean'][:13]) / ext.scalers['x_std'][:13]
                # If any missing weather remains (> 3 bins gap), zero it out to mean
                scaled_hist[np.isnan(scaled_hist)] = 0.0

                node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
                x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

                bx = torch.from_numpy(x_scaled)
                bx = apply_weather_mask(bx, zero_channels).to(self.device)

                out = model(bx).cpu().numpy()
                delta_kw = out * self.res_std_kw + self.res_mean_kw
                pred_w = pers_w[:, None] + delta_kw * KW_TO_WATTS
                true_w = targets_kw * KW_TO_WATTS

                preds_all.append(pred_w)
                trues_all.append(true_w)
                pers_all.append(pers_w[:, None])

                n_err = pred_w - true_w
                n_pers_err = pers_w[:, None] - true_w
                per_node_mae[node_idx] = float(np.mean(np.abs(n_err)))
                per_node_rmse[node_idx] = float(np.sqrt(np.mean(n_err ** 2)))
                per_node_pers_mae[node_idx] = float(np.mean(np.abs(n_pers_err)))

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
        }


    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Execute full T5 Weather Ablation suite across all 4 conditions and 3 seeds.
        """
        suite_tracker = ExperimentStageTracker(
            suite_name="T5 Environmental / Weather Ablation",
            total_experiments=12,  # 4 conditions * 3 seeds
            output_dir=self.output_dir,
            log_filename="progress.log",
        )

        all_seed_results = []
        per_node_records = []
        checkpoints_registry = {}

        # -------------------------------------------------------------------
        # CONDITION A: No Outdoor Weather (T4 Frozen Reference)
        # -------------------------------------------------------------------
        suite_tracker.set_stage("CONDITION_A_EVALUATION", "Evaluating Frozen T4 Reference on T1, Q1, March")
        t4_ckpt_dir = ROOT / 'output/experiments/t4_model_improvement_v1/checkpoints'
        cond_a_meta = WEATHER_CONDITIONS['condition_a_no_weather']

        for seed in SEEDS:
            exp_num = len(all_seed_results) + 1
            suite_tracker.set_experiment(exp_num, cond_a_meta['label'], seed)

            ckpt_path = t4_ckpt_dir / f'fedavg_winner_seed{seed}_best.pt'
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Required T4 reference checkpoint missing: {ckpt_path}")

            sha = sha256_file(ckpt_path)
            checkpoints_registry[f"condition_a_seed{seed}"] = {
                'path': str(ckpt_path),
                'sha256': sha,
                'condition': 'condition_a_no_weather',
                'seed': seed,
                'source': 't4_frozen',
            }

            model = SequenceGRU(
                input_dim=14,
                hidden_dim=64,
                num_layers=2,
                node_count=self.node_count,
                node_embedding_dim=8,
            ).to(self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt['model_state_dict'])

            # Evaluate on all three test periods
            for period_name in ['test_t1', 'test_q1', 'test_march']:
                if period_name == 'test_t1':
                    res = self.evaluate_on_canonical_t1_test(model, cond_a_meta['zero_channels'])
                else:
                    res = self.evaluate_on_extended_test(model, cond_a_meta['zero_channels'], period=period_name)

                all_seed_results.append({
                    'condition_id': 'condition_a_no_weather',
                    'condition_label': cond_a_meta['label'],
                    'temp_active': False,
                    'humidity_active': False,
                    'seed': seed,
                    'period': period_name,
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
                    'checkpoint_sha256': sha,
                })

                if period_name == 'test_t1':
                    for n_idx, n_mae in res['per_node_mae'].items():
                        per_node_records.append({
                            'condition_id': 'condition_a_no_weather',
                            'seed': seed,
                            'period': 'test_t1',
                            'node_idx': n_idx,
                            'model_mae_w': n_mae,
                            'model_rmse_w': res['per_node_rmse'][n_idx],
                            'pers_mae_w': res['per_node_pers_mae'][n_idx],
                            'beats_pers': int(n_mae < res['per_node_pers_mae'][n_idx]),
                        })

        suite_tracker.finish_stage()

        # -------------------------------------------------------------------
        # CONDITIONS B, C, D: Weather Ablations
        # -------------------------------------------------------------------
        active_conditions = [
            'condition_b_temp_only',
            'condition_c_humidity_only',
            'condition_d_temp_and_humidity',
        ]

        for cond_id in active_conditions:
            cond_meta = WEATHER_CONDITIONS[cond_id]
            suite_tracker.set_stage(f"TRAIN_{cond_id.upper()}", f"Training {cond_meta['label']} across Seeds 42, 43, 44")

            for seed in SEEDS:
                exp_num = len(all_seed_results) // 3 + 1
                suite_tracker.set_experiment(exp_num, cond_meta['label'], seed)

                train_res = self.run_fedavg_training(
                    condition_id=cond_id,
                    seed=seed,
                    rounds=200,
                    clients_per_round=98,
                    learning_rate=0.001,
                    local_epochs=1,
                    batch_size=64,
                )

                # Save locked checkpoint
                ckpt_path = self.output_dir / 'checkpoints' / f't5_{cond_id}_seed{seed}_best.pt'
                torch.save({
                    'model_state_dict': train_res['best_state'],
                    'condition_id': cond_id,
                    'condition_label': cond_meta['label'],
                    'zero_channels': cond_meta['zero_channels'],
                    'seed': seed,
                    'best_round': train_res['best_round'],
                    'val_mae_w': train_res['val_mae_w'],
                    'val_rmse_w': train_res['val_rmse_w'],
                }, ckpt_path)

                sha = sha256_file(ckpt_path)
                checkpoints_registry[f"{cond_id}_seed{seed}"] = {
                    'path': str(ckpt_path),
                    'sha256': sha,
                    'condition': cond_id,
                    'seed': seed,
                    'source': 't5_trained',
                    'best_round': train_res['best_round'],
                    'val_mae_w': train_res['val_mae_w'],
                }

                # Evaluate best model across all 3 test periods
                eval_model = SequenceGRU(
                    input_dim=14,
                    hidden_dim=64,
                    num_layers=2,
                    node_count=self.node_count,
                    node_embedding_dim=8,
                ).to(self.device)
                eval_model.load_state_dict(train_res['best_state'])

                for period_name in ['test_t1', 'test_q1', 'test_march']:
                    if period_name == 'test_t1':
                        res = self.evaluate_on_canonical_t1_test(eval_model, cond_meta['zero_channels'])
                    else:
                        res = self.evaluate_on_extended_test(eval_model, cond_meta['zero_channels'], period=period_name)

                    all_seed_results.append({
                        'condition_id': cond_id,
                        'condition_label': cond_meta['label'],
                        'temp_active': cond_meta['temp_active'],
                        'humidity_active': cond_meta['humidity_active'],
                        'seed': seed,
                        'period': period_name,
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
                        'checkpoint_sha256': sha,
                    })

                    if period_name == 'test_t1':
                        for n_idx, n_mae in res['per_node_mae'].items():
                            per_node_records.append({
                                'condition_id': cond_id,
                                'seed': seed,
                                'period': 'test_t1',
                                'node_idx': n_idx,
                                'model_mae_w': n_mae,
                                'model_rmse_w': res['per_node_rmse'][n_idx],
                                'pers_mae_w': res['per_node_pers_mae'][n_idx],
                                'beats_pers': int(n_mae < res['per_node_pers_mae'][n_idx]),
                            })

            suite_tracker.finish_stage()

        # -------------------------------------------------------------------
        # Stage 5: Compile Reports and Export CSV Deliverables
        # -------------------------------------------------------------------
        suite_tracker.set_stage("REPORTING", "Generating CSV deliverables and diagnostic plots")

        df_per_seed = pd.DataFrame(all_seed_results)
        df_per_seed.to_csv(self.output_dir / 'T5_RESULTS_PER_SEED.csv', index=False)

        df_per_node = pd.DataFrame(per_node_records)
        df_per_node.to_csv(self.output_dir / 'T5_PER_NODE_RESULTS.csv', index=False)

        # Generate T5_RESULTS_SUMMARY.csv
        summary_rows = []
        for period in ['test_t1', 'test_q1', 'test_march']:
            sub_p = df_per_seed[df_per_seed['period'] == period]
            # Baseline (Condition A) metrics for comparison
            cond_a_sub = sub_p[sub_p['condition_id'] == 'condition_a_no_weather']
            ref_pooled_mae = cond_a_sub['pooled_mae_w'].mean()
            ref_pooled_rmse = cond_a_sub['pooled_rmse_w'].mean()
            ref_macro_mae = cond_a_sub['macro_mae_w'].mean()

            # Persistence reference
            pers_pooled_mae = sub_p['pers_pooled_mae_w'].iloc[0]
            pers_pooled_rmse = sub_p['pers_pooled_rmse_w'].iloc[0]
            pers_macro_mae = sub_p['pers_macro_mae_w'].iloc[0]

            for cond_id in WEATHER_CONDITIONS.keys():
                sub_c = sub_p[sub_p['condition_id'] == cond_id]
                pooled_mae_mean = sub_c['pooled_mae_w'].mean()
                pooled_mae_std = sub_c['pooled_mae_w'].std()
                pooled_rmse_mean = sub_c['pooled_rmse_w'].mean()
                pooled_rmse_std = sub_c['pooled_rmse_w'].std()
                macro_mae_mean = sub_c['macro_mae_w'].mean()
                macro_mae_std = sub_c['macro_mae_w'].std()
                macro_rmse_mean = sub_c['macro_rmse_w'].mean()
                macro_rmse_std = sub_c['macro_rmse_w'].std()
                nodes_beating_mean = sub_c['nodes_beating_pers'].mean()
                pct_beating_mean = sub_c['pct_nodes_beating_pers'].mean()

                delta_mae_vs_a = pooled_mae_mean - ref_pooled_mae
                pct_delta_mae_vs_a = (delta_mae_vs_a / ref_pooled_mae) * 100.0

                delta_mae_vs_pers = pooled_mae_mean - pers_pooled_mae
                pct_delta_mae_vs_pers = (delta_mae_vs_pers / pers_pooled_mae) * 100.0

                summary_rows.append({
                    'period': period,
                    'condition_id': cond_id,
                    'condition_label': WEATHER_CONDITIONS[cond_id]['label'],
                    'temp_active': WEATHER_CONDITIONS[cond_id]['temp_active'],
                    'humidity_active': WEATHER_CONDITIONS[cond_id]['humidity_active'],
                    'windows': sub_c['windows'].iloc[0],
                    'nodes': sub_c['nodes'].iloc[0],
                    'pooled_mae_mean_w': round(pooled_mae_mean, 4),
                    'pooled_mae_std_w': round(pooled_mae_std, 4),
                    'pooled_rmse_mean_w': round(pooled_rmse_mean, 4),
                    'pooled_rmse_std_w': round(pooled_rmse_std, 4),
                    'macro_mae_mean_w': round(macro_mae_mean, 4),
                    'macro_mae_std_w': round(macro_mae_std, 4),
                    'macro_rmse_mean_w': round(macro_rmse_mean, 4),
                    'macro_rmse_std_w': round(macro_rmse_std, 4),
                    'pers_pooled_mae_w': round(pers_pooled_mae, 4),
                    'pers_pooled_rmse_w': round(pers_pooled_rmse, 4),
                    'pers_macro_mae_w': round(pers_macro_mae, 4),
                    'nodes_beating_pers_mean': round(nodes_beating_mean, 1),
                    'pct_nodes_beating_pers_mean': round(pct_beating_mean, 2),
                    'delta_mae_vs_condition_a_w': round(delta_mae_vs_a, 4),
                    'pct_delta_mae_vs_condition_a': round(pct_delta_mae_vs_a, 2),
                    'delta_mae_vs_pers_w': round(delta_mae_vs_pers, 4),
                    'pct_delta_mae_vs_pers': round(pct_delta_mae_vs_pers, 2),
                })

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_dir / 'T5_RESULTS_SUMMARY.csv', index=False)

        # Generate T5_ABLATION_COMPARISON.csv (direct head-to-head on Canonical T1 Test)
        t1_summary = df_summary[df_summary['period'] == 'test_t1']
        df_ablation = t1_summary[[
            'condition_id',
            'condition_label',
            'temp_active',
            'humidity_active',
            'pooled_mae_mean_w',
            'pooled_mae_std_w',
            'pooled_rmse_mean_w',
            'pooled_rmse_std_w',
            'macro_mae_mean_w',
            'macro_rmse_mean_w',
            'nodes_beating_pers_mean',
            'pct_nodes_beating_pers_mean',
            'delta_mae_vs_condition_a_w',
            'pct_delta_mae_vs_condition_a',
            'delta_mae_vs_pers_w',
            'pct_delta_mae_vs_pers',
        ]]
        df_ablation.to_csv(self.output_dir / 'T5_ABLATION_COMPARISON.csv', index=False)

        # Generate Publication Plots
        self._generate_plots(df_summary, df_per_node)

        # Save Checkpoint Registry
        with open(self.output_dir / 'checkpoints_registry.json', 'w', encoding='utf-8') as f:
            json.dump(checkpoints_registry, f, indent=2)

        suite_tracker.finish_stage()
        suite_tracker.finish_suite()

        print("\n[T5Manager] All T5 experiments, evaluations, and artifacts generated successfully!", flush=True)
        return {
            'df_summary': df_summary,
            'checkpoints_registry': checkpoints_registry,
        }

    def _generate_plots(self, df_summary: pd.DataFrame, df_per_node: pd.DataFrame):
        """Generate high-resolution publication diagnostic plots."""
        plots_dir = self.output_dir / 'plots'

        # 1. Bar Chart: MAE & RMSE across conditions on Canonical T1 Test
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        t1_sub = df_summary[df_summary['period'] == 'test_t1']
        labels = [WEATHER_CONDITIONS[c]['label'] for c in t1_sub['condition_id']]
        maes = t1_sub['pooled_mae_mean_w'].values
        mae_errs = t1_sub['pooled_mae_std_w'].values
        rmses = t1_sub['pooled_rmse_mean_w'].values
        rmse_errs = t1_sub['pooled_rmse_std_w'].values

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # MAE
        bars1 = ax1.bar(labels, maes, yerr=mae_errs, capsize=5, color=colors, alpha=0.85, edgecolor='black', width=0.55)
        ax1.axhline(33.07, color='black', linestyle='--', linewidth=1.5, label='Persistence (33.07 W)')
        ax1.set_ylabel('Pooled Test MAE (Watts)', fontsize=11, fontweight='bold')
        ax1.set_title('Canonical T1 Held-Out Test: MAE by Weather Condition', fontsize=12, fontweight='bold')
        ax1.set_ylim(20, 36)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True)
        for bar, val in zip(bars1, maes):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        # RMSE
        bars2 = ax2.bar(labels, rmses, yerr=rmse_errs, capsize=5, color=colors, alpha=0.85, edgecolor='black', width=0.55)
        ax2.axhline(64.05, color='black', linestyle='--', linewidth=1.5, label='Persistence (64.05 W)')
        ax2.set_ylabel('Pooled Test RMSE (Watts)', fontsize=11, fontweight='bold')
        ax2.set_title('Canonical T1 Held-Out Test: RMSE by Weather Condition', fontsize=12, fontweight='bold')
        ax2.set_ylim(50, 68)
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', frameon=True)
        for bar, val in zip(bars2, rmses):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.45, f"{val:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plot1_path = plots_dir / 't5_weather_ablation_mae_rmse.png'
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot1_path}", flush=True)

        # 2. Node Improvement Distribution: Delta vs Condition A
        fig, ax = plt.subplots(figsize=(10, 5.5))
        # Pivot per node MAE on seed 42
        p_sub = df_per_node[df_per_node['seed'] == 42]
        p_pivot = p_sub.pivot(index='node_idx', columns='condition_id', values='model_mae_w')

        if 'condition_a_no_weather' in p_pivot.columns:
            ref = p_pivot['condition_a_no_weather']
            for cond_id, col, lbl in [
                ('condition_b_temp_only', '#ff7f0e', 'Temp Only vs No Weather'),
                ('condition_c_humidity_only', '#2ca02c', 'Humidity Only vs No Weather'),
                ('condition_d_temp_and_humidity', '#d62728', 'Temp+Humidity vs No Weather'),
            ]:
                if cond_id in p_pivot.columns:
                    diff = p_pivot[cond_id] - ref  # negative means improved
                    ax.hist(diff, bins=40, alpha=0.55, color=col, label=f"{lbl} (Mean: {diff.mean():+.3f} W)")

        ax.axvline(0.0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Per-Node MAE Difference vs No-Weather (Watts) [Left = Weather Improves]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Node Count (out of 978 nodes)', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Weather Impact on Per-Node Forecasting Accuracy', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plot2_path = plots_dir / 't5_node_improvement_distribution.png'
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot2_path}", flush=True)

        # 3. Temporal Generalization Comparison across evaluation windows
        fig, ax = plt.subplots(figsize=(10, 5.5))
        periods = ['test_t1', 'test_q1', 'test_march']
        period_labels = ['Canonical T1\n(Jan 28 - Feb 04)', 'Q1 Extended\n(Jan 28 - Apr 01)', 'March Window\n(Mar 01 - Apr 01)']
        x = np.arange(len(periods))
        width = 0.18

        for idx, (cond_id, col) in enumerate([
            ('condition_a_no_weather', '#1f77b4'),
            ('condition_b_temp_only', '#ff7f0e'),
            ('condition_c_humidity_only', '#2ca02c'),
            ('condition_d_temp_and_humidity', '#d62728'),
        ]):
            vals = [df_summary[(df_summary['period'] == p) & (df_summary['condition_id'] == cond_id)]['pooled_mae_mean_w'].values[0] for p in periods]
            ax.bar(x + (idx - 1.5) * width, vals, width, label=WEATHER_CONDITIONS[cond_id]['label'], color=col, alpha=0.85, edgecolor='black')

        ax.set_ylabel('Pooled MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Generalization: Weather Impact Across Deployment Windows', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(period_labels, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plot3_path = plots_dir / 't5_temporal_generalization_daily.png'
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot3_path}", flush=True)


if __name__ == '__main__':
    mgr = T5ExperimentManager()
    mgr.run_all_experiments()
