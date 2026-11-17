"""
Task T6: Architectural Family Generalization Benchmark Runner.

Evaluates next-hour node IT power forecasting on the CINECA M100 cluster across
three distinct architectural families under controlled conditions:

1. Condition A (Reference Baseline):
   - 2-layer SequenceGRU under sample-weighted FedAvg (T4/T5 winner baseline).
   - Parameters: 49,561 (hidden_dim=64, node_embedding_dim=8).
   - Evaluated using frozen T4 checkpoints with weather channels [7, 8] masked to 0.0.

2. Condition B (Matched-Capacity Recurrent Family):
   - 2-layer SequenceLSTM under sample-weighted FedAvg.
   - Parameters: 63,449 (hidden_dim=64, node_embedding_dim=8).
   - Sample-weighted FedAvg, 200 communication rounds, 98 clients/round, local epochs=1, lr=0.001.
   - Weather channels [7, 8] masked to 0.0.
   - Checkpointed based on validation MAE.

3. Condition C (Centralized Tree-Based Family):
   - Centralized Gradient Boosted Decision Trees (XGBoost).
   - NOT a federated algorithm. All 394,100 training windows pooled into a centralized tabular matrix.
   - Input: Flattened 36 history steps of 14 features (504 features), weather channels [7, 8] masked.
   - Target: Standardized persistence residual.
   - GPU-accelerated histogram trees (tree_method='hist', objective='reg:absoluteerror', device='cuda:1').
   - Hyperparameters: max_depth=6, lr=0.08, subsample=0.8, colsample_bytree=0.8.

All conditions evaluated across Seeds 42, 43, 44 on:
- Canonical T1 Held-Out Test: 243,138 windows, 978 nodes
- Q1 Extended Temporal Evaluation: 3,609,285 windows, 979 nodes
- March Temporal Evaluation Window: 2,229,042 windows, 976 nodes

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

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
WEATHER_CHANNELS = [7, 8]  # outdoor_temperature_c, outdoor_humidity_percent

T6_CONDITIONS = {
    'condition_a_fedavg_gru': {
        'label': 'FedAvg-GRU (T4 Reference)',
        'model_family': 'Recurrent Neural Network (GRU)',
        'training_paradigm': 'Federated Learning (Sample-Weighted FedAvg)',
        'param_count': 49561,
        'architecture': '2-layer SequenceGRU',
        'source': 't4_frozen',
    },
    'condition_b_fedavg_lstm': {
        'label': 'FedAvg-LSTM',
        'model_family': 'Recurrent Neural Network (LSTM)',
        'training_paradigm': 'Federated Learning (Sample-Weighted FedAvg)',
        'param_count': 63449,
        'architecture': '2-layer SequenceLSTM',
        'source': 't6_trained',
    },
    'condition_c_centralized_xgboost': {
        'label': 'Centralized XGBoost',
        'model_family': 'Gradient Boosted Decision Trees',
        'training_paradigm': 'Centralized Non-Federated Baseline',
        'param_count': 0,  # Tree ensemble
        'architecture': 'Histogram GBDT (504 features)',
        'source': 't6_trained',
    },
}


def apply_weather_mask(x: torch.Tensor, zero_channels: Sequence[int] = (7, 8)) -> torch.Tensor:
    """Mask specified weather channels to 0.0 in-place on a clone."""
    if not zero_channels:
        return x
    xc = x.clone()
    xc[:, :, zero_channels] = 0.0
    return xc


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class SequenceLSTM(nn.Module):
    """
    Sequence-to-one 2-layer LSTM with categorical node embeddings.
    Predicts next-hour persistence residual in standardized units.
    Total parameters with hidden_dim=64 and node_embedding_dim=8: 63,449.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        node_count: int = 979,
        node_embedding_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_embedding = nn.Embedding(node_count, node_embedding_dim) if node_count > 0 else None
        recurrent_in_dim = (input_dim - 1 + node_embedding_dim) if node_count > 0 else input_dim
        self.lstm = nn.LSTM(
            recurrent_in_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        if self.node_embedding is not None:
            features = x[:, :, :-1]
            node_ids = x[:, :, -1].long()
            embedded = self.node_embedding(node_ids)
            x = torch.cat([features, embedded], dim=-1)

        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step)


class T6ArchitectureBenchmarkManager:
    """
    Orchestrates the T6 Architectural Family Generalization Benchmark.
    """

    def __init__(
        self,
        output_dir: Path = ROOT / 'output/experiments/t6_architecture_benchmark_v1',
        device_str: str = 'cuda:1',
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available() and 'cuda' in device_str:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device('cpu')

        print(f"[T6Manager] Initialized on device: {self.device}", flush=True)

        # 1. Load Candidate 1 Dataset for Training & Validation
        print("[T6Manager] Loading Candidate 1 reference dataset...", flush=True)
        self.t1_data = Candidate1RealData()
        self.node_count = self.t1_data.node_count
        self.train_origins = self.t1_data.valid_origins['train']
        self.val_origins = self.t1_data.valid_origins['val']
        self.test_t1_origins = self.t1_data.valid_origins['test']

        # Moments
        self.y_mean_kw = float(self.t1_data.scalers['y_mean'][0])
        self.y_std_kw = float(self.t1_data.scalers['y_std'][0])

        # Precompute residual moments on training set
        print("[T6Manager] Computing training moments for target and persistence residual...", flush=True)
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

        # Pre-extract validation set (with weather masked to 0.0)
        self._pre_extract_validation()

        # Pre-extract training batches resident on device for fast FedAvg rounds
        self._pre_extract_training_clients()

        # 2. Lazy load Extended Telemetry for Q1 and March evaluations
        self._ext_data: Optional[ExtendedTelemetryDataset] = None

    def get_extended_data(self) -> ExtendedTelemetryDataset:
        """Lazy loader for Q1 and March extended telemetry."""
        if self._ext_data is None:
            print("[T6Manager] Loading ExtendedTelemetryDataset for Q1/March evaluation...", flush=True)
            self._ext_data = ExtendedTelemetryDataset()
        return self._ext_data

    def _extract_unmasked_batch(
        self,
        node_idx: int,
        origins: np.ndarray,
        history_steps: int = HISTORY_STEPS,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Extract batch with causal forward-fill FF30 and masked weather (channels 7 and 8).
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

        # Mask weather channels 7 and 8 to 0.0
        scaled_hist[:, :, WEATHER_CHANNELS] = 0.0

        # Append categorical node index
        node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
        x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(x_scaled),
            torch.from_numpy(y_scaled.astype(np.float32)),
            pers_w,
        )

    def _pre_extract_validation(self):
        """Pre-extract validation set with masked weather channels."""
        print("[T6Manager] Pre-extracting validation set...", flush=True)
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
        """Pre-extract training batches per client and place on GPU."""
        print("[T6Manager] Pre-extracting training batches per client onto GPU...", flush=True)
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

            # Place on device for fast FedAvg rounds
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
        batch_size: int = 4096,
    ) -> Tuple[float, float]:
        """Fast validation evaluation of a PyTorch recurrent model."""
        model.eval()
        preds_w = []
        n_val = self.val_x.shape[0]

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                end = min(i + batch_size, n_val)
                bx = self.val_x[i:end]
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

    def run_fedavg_lstm_training(
        self,
        seed: int,
        rounds: int = 200,
        clients_per_round: int = 98,
        learning_rate: float = 0.001,
        local_epochs: int = 1,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Execute 200 communication rounds of FedAvg for SequenceLSTM under seed.
        Integrates mandatory real-time terminal progress logging.
        """
        # Deterministic seed initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rng = np.random.default_rng(np.random.SeedSequence([seed, 50, 0, 1]))

        # Global model architecture: 2-layer SequenceLSTM
        global_model = SequenceLSTM(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=self.node_count,
            node_embedding_dim=8,
        ).to(self.device)

        # Single reusable local model to minimize memory reallocations
        local_model = SequenceLSTM(
            input_dim=14,
            hidden_dim=64,
            num_layers=2,
            node_count=self.node_count,
            node_embedding_dim=8,
        ).to(self.device)

        param_count = sum(p.numel() for p in global_model.parameters())
        assert param_count == 63449, f"Expected 63,449 parameters for LSTM, got {param_count}"
        loss_fn = nn.L1Loss()

        progress_logger = FedAvgProgressLogger(
            experiment_name="T6",
            config_name="FedAvg-LSTM",
            seed=seed,
            total_rounds=rounds,
            total_clients=clients_per_round,
            device=str(self.device),
            output_dir=self.output_dir,
            log_filename="progress.log",
            csv_filename=f"fedavg_lstm_seed{seed}_progress.csv",
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

                local_model.load_state_dict(global_state)
                local_model.train()

                optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)

                cx = c_data['x']
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

                client_states.append(copy.deepcopy(local_model.state_dict()))

                if c_idx % 20 == 0 or c_idx == clients_per_round:
                    progress_logger.update_client_progress(
                        client_idx=c_idx,
                        epoch=1,
                        total_epochs=local_epochs,
                        loss=float(avg_c_loss),
                    )

            # Sample-weighted FedAvg aggregation
            total_round_samples = sum(client_weights)
            new_global_state = OrderedDict()
            for key in global_state.keys():
                weighted_sum = sum(
                    client_states[i][key].float() * (client_weights[i] / total_round_samples)
                    for i in range(len(client_states))
                )
                new_global_state[key] = weighted_sum

            global_model.load_state_dict(new_global_state)
            round_loss = round_loss_sum / max(1, round_sample_count)

            # Evaluate on validation set
            val_mae, val_rmse = self.evaluate_model_on_val(global_model)
            is_best = val_mae < best_val_mae
            if is_best:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
                best_round = r
                best_state = copy.deepcopy(global_model.state_dict())

            progress_logger.complete_round(
                train_loss=float(round_loss),
                val_mae=float(val_mae),
                val_rmse=float(val_rmse),
                total_epochs=local_epochs,
            )

        elapsed_total = time.time() - t0
        progress_logger.finish_training()

        return {
            'seed': seed,
            'best_round': best_round,
            'val_mae_w': best_val_mae,
            'val_rmse_w': best_val_rmse,
            'best_state': best_state,
            'elapsed_seconds': elapsed_total,
        }

    def train_centralized_xgboost(
        self,
        seed: int,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        num_boost_round: int = 150,
        early_stopping_rounds: int = 25,
    ) -> Dict[str, Any]:
        """
        Train Centralized Gradient Boosted Decision Tree (XGBoost) baseline on GPU.
        Input: Flattened 36-step history (504 features).
        Target: Standardized persistence residual.
        """
        print(f"\n[T6Manager] Preparing Centralized XGBoost training data for Seed {seed}...", flush=True)
        t0 = time.time()

        # Build centralized training matrix from client batches
        train_x_list = []
        train_y_list = []
        for c_id in self.eligible_clients:
            c_data = self.train_clients[c_id]
            # Flatten 36 steps of 14 features -> 504 features
            flat_x = c_data['x'].reshape(c_data['n_samples'], -1).cpu().numpy()
            flat_y = c_data['y_res'].cpu().numpy().squeeze()
            train_x_list.append(flat_x)
            train_y_list.append(flat_y)

        X_train = np.concatenate(train_x_list, axis=0)
        y_train = np.concatenate(train_y_list, axis=0)
        print(f"  Centralized X_train: {X_train.shape}, y_train: {y_train.shape} in {time.time()-t0:.2f}s", flush=True)

        # Flatten validation matrix
        X_val = self.val_x.reshape(self.val_x.shape[0], -1).cpu().numpy()
        y_val = self.val_y_res.cpu().numpy().squeeze()

        # Create DMatrix objects on GPU
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'tree_method': 'hist',
            'device': str(self.device),
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': seed,
            'random_state': seed,
        }

        print(f"[T6Manager] Training XGBoost on {self.device} (max_depth={max_depth}, lr={learning_rate}, subsample={subsample})...", flush=True)
        evals_result = {}
        t_train_start = time.time()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=25,
        )
        elapsed_train = time.time() - t_train_start

        best_iteration = booster.best_iteration
        best_val_mae_scaled = booster.best_score
        # Convert val MAE to physical Watts
        best_val_mae_w = best_val_mae_scaled * self.res_std_kw * KW_TO_WATTS

        print(f"[T6Manager] XGBoost trained: best iteration {best_iteration}/{num_boost_round}, Val MAE: {best_val_mae_w:.2f} W in {elapsed_train:.2f}s", flush=True)

        return {
            'seed': seed,
            'booster': booster,
            'best_iteration': best_iteration,
            'val_mae_w': best_val_mae_w,
            'elapsed_seconds': elapsed_train,
            'num_features': X_train.shape[1],
        }

    def evaluate_model_on_canonical_t1_test(
        self,
        model_or_booster: Any,
        is_xgboost: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on Canonical T1 Held-Out Test period (243,138 windows, 978 nodes)."""
        if not is_xgboost:
            model_or_booster.eval()

        preds_all = []
        trues_all = []
        pers_all = []
        node_ids_all = []

        per_node_mae = {}
        per_node_rmse = {}
        per_node_pers_mae = {}

        t0 = time.time()
        with torch.no_grad():
            for node_idx in range(self.node_count):
                orig = self.test_t1_origins[node_idx]
                if len(orig) == 0:
                    continue

                bx, by, pers_w = self._extract_unmasked_batch(node_idx, orig, HISTORY_STEPS)

                if is_xgboost:
                    bx_flat = bx.reshape(len(orig), -1).numpy()
                    pred_res = model_or_booster.predict(xgb.DMatrix(bx_flat))[:, None]
                else:
                    bx = bx.to(self.device)
                    pred_res = model_or_booster(bx).cpu().numpy()

                delta_kw = pred_res * self.res_std_kw + self.res_mean_kw
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

    def evaluate_model_on_extended_test(
        self,
        model_or_booster: Any,
        period: str = 'test_q1',
        is_xgboost: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on Q1 Extended ([3888, 12960)) or March Window ([8496, 12960)).
        """
        ext = self.get_extended_data()
        if not is_xgboost:
            model_or_booster.eval()

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

        t0 = time.time()
        with torch.no_grad():
            for node_idx in range(self.node_count):
                all_orig = ext.origins['t2'][node_idx]
                orig = all_orig[(all_orig >= lo) & (all_orig < hi)]
                if len(orig) == 0:
                    continue

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
                scaled_hist[np.isnan(scaled_hist)] = 0.0

                # Weather channels 7 and 8 masked to 0.0
                scaled_hist[:, :, WEATHER_CHANNELS] = 0.0

                node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
                x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

                bx = torch.from_numpy(x_scaled)

                if is_xgboost:
                    bx_flat = bx.reshape(B, -1).numpy()
                    pred_res = model_or_booster.predict(xgb.DMatrix(bx_flat))[:, None]
                else:
                    bx = bx.to(self.device)
                    pred_res = model_or_booster(bx).cpu().numpy()

                delta_kw = pred_res * self.res_std_kw + self.res_mean_kw
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

    def run_all_benchmark(self, seeds: Sequence[int] = (42, 43, 44)) -> Dict[str, Any]:
        """
        Execute full T6 Architecture Benchmark:
        - Condition A: SequenceGRU (frozen T4 checkpoints, weather masked)
        - Condition B: SequenceLSTM (trained under FedAvg for each seed)
        - Condition C: Centralized XGBoost (trained centrally on GPU for each seed)
        All evaluated on Canonical T1, Q1 Extended, and March Window.
        """
        suite_tracker = ExperimentStageTracker(
            suite_name="T6 Architectural Family Generalization Benchmark",
            total_experiments=9,
            output_dir=self.output_dir,
            log_filename="progress.log",
        )

        records_per_seed: List[Dict[str, Any]] = []
        per_node_records: List[Dict[str, Any]] = []
        resource_records: List[Dict[str, Any]] = []
        checkpoints_registry: Dict[str, Any] = {}

        # =====================================================================
        # STAGE 1: Condition A (Reference FedAvg-GRU)
        # =====================================================================
        cond_a_meta = T6_CONDITIONS['condition_a_fedavg_gru']

        for s_idx, seed in enumerate(seeds, start=1):
            suite_tracker.set_experiment(s_idx, cond_a_meta['label'], seed)
            print(f"\n{'='*70}\n[Stage 1/3] Condition A: FedAvg-GRU | Seed {seed}\n{'='*70}", flush=True)
            ckpt_path = ROOT / f"output/experiments/t4_model_improvement_v1/checkpoints/fedavg_winner_seed{seed}_best.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"T4 checkpoint not found: {ckpt_path}")

            sha = sha256_file(ckpt_path)
            checkpoints_registry[f"condition_a_seed{seed}"] = {
                'path': str(ckpt_path),
                'sha256': sha,
                'condition': 'condition_a_fedavg_gru',
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

            # Record resources
            # FL comm volume: 49,561 params * 4 bytes * 99 clients * 200 rounds = 3,743.4 MB
            comm_mb_round = (49561 * 4 * 99) / (1024 * 1024)
            total_comm_mb = comm_mb_round * 200

            eval_time_t1 = 0.0
            throughput_t1 = 0.0

            for period_name in ['test_t1', 'test_q1', 'test_march']:
                if period_name == 'test_t1':
                    res = self.evaluate_model_on_canonical_t1_test(model, is_xgboost=False)
                    eval_time_t1 = res['eval_time_seconds']
                    throughput_t1 = res['throughput_windows_sec']
                else:
                    res = self.evaluate_model_on_extended_test(model, period=period_name, is_xgboost=False)

                print(f"  [{period_name.upper()}] Pooled MAE: {res['pooled_mae_w']:.2f} W | Macro MAE: {res['macro_mae_w']:.2f} W | Pers MAE: {res['pers_pooled_mae_w']:.2f} W | Beating: {res['pct_nodes_beating_pers']:.1f}%", flush=True)

                records_per_seed.append({
                    'condition_id': 'condition_a_fedavg_gru',
                    'condition_label': cond_a_meta['label'],
                    'model_family': cond_a_meta['model_family'],
                    'training_paradigm': cond_a_meta['training_paradigm'],
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
                    'checkpoint_path': str(ckpt_path),
                    'checkpoint_sha256': sha,
                })

                for n_idx, m_mae in res['per_node_mae'].items():
                    per_node_records.append({
                        'condition_id': 'condition_a_fedavg_gru',
                        'seed': seed,
                        'period': period_name,
                        'node_idx': n_idx,
                        'model_mae_w': m_mae,
                        'model_rmse_w': res['per_node_rmse'][n_idx],
                        'pers_mae_w': res['per_node_pers_mae'][n_idx],
                        'diff_vs_pers_w': m_mae - res['per_node_pers_mae'][n_idx],
                    })

            resource_records.append({
                'condition_id': 'condition_a_fedavg_gru',
                'condition_label': cond_a_meta['label'],
                'model_family': cond_a_meta['model_family'],
                'seed': seed,
                'parameter_count': 49561,
                'fl_comm_mb_per_round': comm_mb_round,
                'fl_total_comm_mb': total_comm_mb,
                'centralized_data_transfer_mb': 0.0,
                'training_time_seconds': 0.0,  # Frozen baseline
                't1_eval_time_seconds': eval_time_t1,
                't1_throughput_windows_sec': throughput_t1,
                'peak_gpu_vram_mb': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0,
                'peak_host_ram_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            })

        # =====================================================================
        # STAGE 2: Condition B (FedAvg-LSTM Training & Evaluation)
        # =====================================================================
        cond_b_meta = T6_CONDITIONS['condition_b_fedavg_lstm']

        for s_idx, seed in enumerate(seeds, start=4):
            suite_tracker.set_experiment(s_idx, cond_b_meta['label'], seed)
            print(f"\n{'='*70}\n[Stage 2/3] Condition B: FedAvg-LSTM | Seed {seed}\n{'='*70}", flush=True)

            train_res = self.run_fedavg_lstm_training(seed=seed, rounds=200, clients_per_round=98)

            ckpt_filename = f"t6_fedavg_lstm_seed{seed}_best.pt"
            ckpt_path = self.output_dir / 'checkpoints' / ckpt_filename
            torch.save({
                'model_state_dict': train_res['best_state'],
                'seed': seed,
                'best_round': train_res['best_round'],
                'val_mae_w': train_res['val_mae_w'],
                'val_rmse_w': train_res['val_rmse_w'],
                'param_count': 63449,
                'architecture': '2-layer SequenceLSTM',
            }, ckpt_path)

            sha = sha256_file(ckpt_path)
            checkpoints_registry[f"condition_b_seed{seed}"] = {
                'path': str(ckpt_path),
                'sha256': sha,
                'condition': 'condition_b_fedavg_lstm',
                'seed': seed,
                'source': 't6_trained',
                'best_round': train_res['best_round'],
                'val_mae_w': train_res['val_mae_w'],
            }

            eval_model = SequenceLSTM(
                input_dim=14,
                hidden_dim=64,
                num_layers=2,
                node_count=self.node_count,
                node_embedding_dim=8,
            ).to(self.device)
            eval_model.load_state_dict(train_res['best_state'])

            comm_mb_round = (63449 * 4 * 99) / (1024 * 1024)
            total_comm_mb = comm_mb_round * 200
            eval_time_t1 = 0.0
            throughput_t1 = 0.0

            for period_name in ['test_t1', 'test_q1', 'test_march']:
                if period_name == 'test_t1':
                    res = self.evaluate_model_on_canonical_t1_test(eval_model, is_xgboost=False)
                    eval_time_t1 = res['eval_time_seconds']
                    throughput_t1 = res['throughput_windows_sec']
                else:
                    res = self.evaluate_model_on_extended_test(eval_model, period=period_name, is_xgboost=False)

                print(f"  [{period_name.upper()}] Pooled MAE: {res['pooled_mae_w']:.2f} W | Macro MAE: {res['macro_mae_w']:.2f} W | Pers MAE: {res['pers_pooled_mae_w']:.2f} W | Beating: {res['pct_nodes_beating_pers']:.1f}%", flush=True)

                records_per_seed.append({
                    'condition_id': 'condition_b_fedavg_lstm',
                    'condition_label': cond_b_meta['label'],
                    'model_family': cond_b_meta['model_family'],
                    'training_paradigm': cond_b_meta['training_paradigm'],
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
                    'checkpoint_path': str(ckpt_path),
                    'checkpoint_sha256': sha,
                })

                for n_idx, m_mae in res['per_node_mae'].items():
                    per_node_records.append({
                        'condition_id': 'condition_b_fedavg_lstm',
                        'seed': seed,
                        'period': period_name,
                        'node_idx': n_idx,
                        'model_mae_w': m_mae,
                        'model_rmse_w': res['per_node_rmse'][n_idx],
                        'pers_mae_w': res['per_node_pers_mae'][n_idx],
                        'diff_vs_pers_w': m_mae - res['per_node_pers_mae'][n_idx],
                    })

            resource_records.append({
                'condition_id': 'condition_b_fedavg_lstm',
                'condition_label': cond_b_meta['label'],
                'model_family': cond_b_meta['model_family'],
                'seed': seed,
                'parameter_count': 63449,
                'fl_comm_mb_per_round': comm_mb_round,
                'fl_total_comm_mb': total_comm_mb,
                'centralized_data_transfer_mb': 0.0,
                'training_time_seconds': train_res['elapsed_seconds'],
                't1_eval_time_seconds': eval_time_t1,
                't1_throughput_windows_sec': throughput_t1,
                'peak_gpu_vram_mb': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0,
                'peak_host_ram_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            })

        # =====================================================================
        # STAGE 3: Condition C (Centralized XGBoost Training & Evaluation)
        # =====================================================================
        cond_c_meta = T6_CONDITIONS['condition_c_centralized_xgboost']

        for s_idx, seed in enumerate(seeds, start=7):
            suite_tracker.set_experiment(s_idx, cond_c_meta['label'], seed)
            print(f"\n{'='*70}\n[Stage 3/3] Condition C: Centralized XGBoost | Seed {seed}\n{'='*70}", flush=True)

            xgb_res = self.train_centralized_xgboost(seed=seed)
            booster = xgb_res['booster']

            ckpt_filename = f"t6_centralized_xgboost_seed{seed}.json"
            ckpt_path = self.output_dir / 'checkpoints' / ckpt_filename
            booster.save_model(str(ckpt_path))

            sha = sha256_file(ckpt_path)
            checkpoints_registry[f"condition_c_seed{seed}"] = {
                'path': str(ckpt_path),
                'sha256': sha,
                'condition': 'condition_c_centralized_xgboost',
                'seed': seed,
                'source': 't6_trained',
                'best_iteration': xgb_res['best_iteration'],
                'val_mae_w': xgb_res['val_mae_w'],
            }

            # Centralized data transfer size: 394,100 windows * 504 float32 * 4 bytes = 757.7 MB
            centralized_transfer_mb = (394100 * 504 * 4) / (1024 * 1024)
            eval_time_t1 = 0.0
            throughput_t1 = 0.0

            for period_name in ['test_t1', 'test_q1', 'test_march']:
                if period_name == 'test_t1':
                    res = self.evaluate_model_on_canonical_t1_test(booster, is_xgboost=True)
                    eval_time_t1 = res['eval_time_seconds']
                    throughput_t1 = res['throughput_windows_sec']
                else:
                    res = self.evaluate_model_on_extended_test(booster, period=period_name, is_xgboost=True)

                print(f"  [{period_name.upper()}] Pooled MAE: {res['pooled_mae_w']:.2f} W | Macro MAE: {res['macro_mae_w']:.2f} W | Pers MAE: {res['pers_pooled_mae_w']:.2f} W | Beating: {res['pct_nodes_beating_pers']:.1f}%", flush=True)

                records_per_seed.append({
                    'condition_id': 'condition_c_centralized_xgboost',
                    'condition_label': cond_c_meta['label'],
                    'model_family': cond_c_meta['model_family'],
                    'training_paradigm': cond_c_meta['training_paradigm'],
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
                    'checkpoint_path': str(ckpt_path),
                    'checkpoint_sha256': sha,
                })

                for n_idx, m_mae in res['per_node_mae'].items():
                    per_node_records.append({
                        'condition_id': 'condition_c_centralized_xgboost',
                        'seed': seed,
                        'period': period_name,
                        'node_idx': n_idx,
                        'model_mae_w': m_mae,
                        'model_rmse_w': res['per_node_rmse'][n_idx],
                        'pers_mae_w': res['per_node_pers_mae'][n_idx],
                        'diff_vs_pers_w': m_mae - res['per_node_pers_mae'][n_idx],
                    })

            resource_records.append({
                'condition_id': 'condition_c_centralized_xgboost',
                'condition_label': cond_c_meta['label'],
                'model_family': cond_c_meta['model_family'],
                'seed': seed,
                'parameter_count': booster.num_boosted_rounds(),
                'fl_comm_mb_per_round': 0.0,
                'fl_total_comm_mb': 0.0,
                'centralized_data_transfer_mb': centralized_transfer_mb,
                'training_time_seconds': xgb_res['elapsed_seconds'],
                't1_eval_time_seconds': eval_time_t1,
                't1_throughput_windows_sec': throughput_t1,
                'peak_gpu_vram_mb': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0,
                'peak_host_ram_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            })

        suite_tracker.finish_suite()

        # =====================================================================
        # AGGREGATION AND EXPORT
        # =====================================================================
        print("\n[T6Manager] Aggregating results across seeds and conditions...", flush=True)

        # 1. T6_RESULTS_PER_SEED.csv
        df_per_seed = pd.DataFrame(records_per_seed)
        csv_per_seed = self.output_dir / 'T6_RESULTS_PER_SEED.csv'
        df_per_seed.to_csv(csv_per_seed, index=False)
        print(f"  Exported: {csv_per_seed} ({len(df_per_seed)} rows)", flush=True)

        # 2. T6_PER_NODE_RESULTS.csv
        df_per_node = pd.DataFrame(per_node_records)
        csv_per_node = self.output_dir / 'T6_PER_NODE_RESULTS.csv'
        df_per_node.to_csv(csv_per_node, index=False)
        print(f"  Exported: {csv_per_node} ({len(df_per_node)} rows)", flush=True)

        # 3. T6_RESOURCE_USAGE.csv
        df_resources = pd.DataFrame(resource_records)
        csv_resources = self.output_dir / 'T6_RESOURCE_USAGE.csv'
        df_resources.to_csv(csv_resources, index=False)
        print(f"  Exported: {csv_resources} ({len(df_resources)} rows)", flush=True)

        # 4. T6_RESULTS_SUMMARY.csv
        summary_rows = []
        for period in ['test_t1', 'test_q1', 'test_march']:
            sub_p = df_per_seed[df_per_seed['period'] == period]
            # Reference Condition A for delta
            cond_a_sub = sub_p[sub_p['condition_id'] == 'condition_a_fedavg_gru']
            ref_mae = cond_a_sub['pooled_mae_w'].mean()

            for cond_id in ['condition_a_fedavg_gru', 'condition_b_fedavg_lstm', 'condition_c_centralized_xgboost']:
                sub_c = sub_p[sub_p['condition_id'] == cond_id]
                meta = T6_CONDITIONS[cond_id]

                p_mae_mean = sub_c['pooled_mae_w'].mean()
                p_mae_std = sub_c['pooled_mae_w'].std()
                p_rmse_mean = sub_c['pooled_rmse_w'].mean()
                p_rmse_std = sub_c['pooled_rmse_w'].std()

                m_mae_mean = sub_c['macro_mae_w'].mean()
                m_mae_std = sub_c['macro_mae_w'].std()
                m_rmse_mean = sub_c['macro_rmse_w'].mean()
                m_rmse_std = sub_c['macro_rmse_w'].std()

                pers_p_mae = sub_c['pers_pooled_mae_w'].iloc[0]
                pers_p_rmse = sub_c['pers_pooled_rmse_w'].iloc[0]
                pers_m_mae = sub_c['pers_macro_mae_w'].iloc[0]

                nodes_beat_mean = sub_c['nodes_beating_pers'].mean()
                pct_beat_mean = sub_c['pct_nodes_beating_pers'].mean()

                delta_vs_a = p_mae_mean - ref_mae
                pct_vs_a = (delta_vs_a / ref_mae) * 100.0

                delta_vs_pers = p_mae_mean - pers_p_mae
                pct_vs_pers = (delta_vs_pers / pers_p_mae) * 100.0

                summary_rows.append({
                    'period': period,
                    'condition_id': cond_id,
                    'condition_label': meta['label'],
                    'model_family': meta['model_family'],
                    'training_paradigm': meta['training_paradigm'],
                    'windows': sub_c['windows'].iloc[0],
                    'nodes': sub_c['nodes'].iloc[0],
                    'pooled_mae_mean_w': round(p_mae_mean, 4),
                    'pooled_mae_std_w': round(p_mae_std, 4),
                    'pooled_rmse_mean_w': round(p_rmse_mean, 4),
                    'pooled_rmse_std_w': round(p_rmse_std, 4),
                    'macro_mae_mean_w': round(m_mae_mean, 4),
                    'macro_mae_std_w': round(m_mae_std, 4),
                    'macro_rmse_mean_w': round(m_rmse_mean, 4),
                    'macro_rmse_std_w': round(m_rmse_std, 4),
                    'pers_pooled_mae_w': round(pers_p_mae, 4),
                    'pers_pooled_rmse_w': round(pers_p_rmse, 4),
                    'pers_macro_mae_w': round(pers_m_mae, 4),
                    'nodes_beating_pers_mean': round(nodes_beat_mean, 1),
                    'pct_nodes_beating_pers_mean': round(pct_beat_mean, 2),
                    'delta_mae_vs_condition_a_w': round(delta_vs_a, 4),
                    'pct_delta_mae_vs_condition_a': round(pct_vs_a, 2),
                    'delta_mae_vs_pers_w': round(delta_vs_pers, 4),
                    'pct_delta_mae_vs_pers': round(pct_vs_pers, 2),
                })

        df_summary = pd.DataFrame(summary_rows)
        csv_summary = self.output_dir / 'T6_RESULTS_SUMMARY.csv'
        df_summary.to_csv(csv_summary, index=False)
        print(f"  Exported: {csv_summary} ({len(df_summary)} rows)", flush=True)

        # 5. T6_MODEL_COMPARISON.csv (Pairwise differences)
        comparison_rows = []
        for period in ['test_t1', 'test_q1', 'test_march']:
            sub_p = df_summary[df_summary['period'] == period].set_index('condition_id')
            gru_mae = sub_p.loc['condition_a_fedavg_gru', 'pooled_mae_mean_w']
            lstm_mae = sub_p.loc['condition_b_fedavg_lstm', 'pooled_mae_mean_w']
            xgb_mae = sub_p.loc['condition_c_centralized_xgboost', 'pooled_mae_mean_w']

            # LSTM vs GRU
            diff_lg = lstm_mae - gru_mae
            comparison_rows.append({
                'period': period,
                'comparison': 'FedAvg-LSTM vs FedAvg-GRU',
                'model_1': 'FedAvg-LSTM',
                'model_2': 'FedAvg-GRU',
                'model_1_mae_w': lstm_mae,
                'model_2_mae_w': gru_mae,
                'abs_diff_w': round(diff_lg, 4),
                'pct_diff': round((diff_lg / gru_mae) * 100.0, 2),
                'superior_model': 'FedAvg-GRU' if diff_lg > 0 else 'FedAvg-LSTM',
            })

            # XGBoost vs GRU
            diff_xg = xgb_mae - gru_mae
            comparison_rows.append({
                'period': period,
                'comparison': 'Centralized XGBoost vs FedAvg-GRU',
                'model_1': 'Centralized XGBoost',
                'model_2': 'FedAvg-GRU',
                'model_1_mae_w': xgb_mae,
                'model_2_mae_w': gru_mae,
                'abs_diff_w': round(diff_xg, 4),
                'pct_diff': round((diff_xg / gru_mae) * 100.0, 2),
                'superior_model': 'FedAvg-GRU' if diff_xg > 0 else 'Centralized XGBoost',
            })

            # XGBoost vs LSTM
            diff_xl = xgb_mae - lstm_mae
            comparison_rows.append({
                'period': period,
                'comparison': 'Centralized XGBoost vs FedAvg-LSTM',
                'model_1': 'Centralized XGBoost',
                'model_2': 'FedAvg-LSTM',
                'model_1_mae_w': xgb_mae,
                'model_2_mae_w': lstm_mae,
                'abs_diff_w': round(diff_xl, 4),
                'pct_diff': round((diff_xl / lstm_mae) * 100.0, 2),
                'superior_model': 'FedAvg-LSTM' if diff_xl > 0 else 'Centralized XGBoost',
            })

        df_comparison = pd.DataFrame(comparison_rows)
        csv_comparison = self.output_dir / 'T6_MODEL_COMPARISON.csv'
        df_comparison.to_csv(csv_comparison, index=False)
        print(f"  Exported: {csv_comparison} ({len(df_comparison)} rows)", flush=True)

        # 6. Generate diagnostic comparison plots
        self._generate_plots(df_summary, df_per_node, df_resources)

        return {
            'df_per_seed': df_per_seed,
            'df_summary': df_summary,
            'df_comparison': df_comparison,
            'df_resources': df_resources,
            'checkpoints_registry': checkpoints_registry,
        }

    def _generate_plots(
        self,
        df_summary: pd.DataFrame,
        df_per_node: pd.DataFrame,
        df_resources: pd.DataFrame,
    ):
        """Generate high-resolution publication diagnostic plots for T6."""
        plots_dir = self.output_dir / 'plots'

        # 1. Bar Chart: MAE & RMSE across architectures on Canonical T1 Test
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        t1_sub = df_summary[df_summary['period'] == 'test_t1']
        labels = [T6_CONDITIONS[c]['label'] for c in t1_sub['condition_id']]
        maes = t1_sub['pooled_mae_mean_w'].values
        mae_errs = t1_sub['pooled_mae_std_w'].values
        rmses = t1_sub['pooled_rmse_mean_w'].values
        rmse_errs = t1_sub['pooled_rmse_std_w'].values

        colors = ['#1f77b4', '#2ca02c', '#d62728']

        # MAE
        bars1 = ax1.bar(labels, maes, yerr=mae_errs, capsize=5, color=colors, alpha=0.85, edgecolor='black', width=0.5)
        ax1.axhline(33.07, color='black', linestyle='--', linewidth=1.5, label='Persistence (33.07 W)')
        ax1.set_ylabel('Pooled Test MAE (Watts)', fontsize=11, fontweight='bold')
        ax1.set_title('Canonical T1 Held-Out Test: MAE by Architecture', fontsize=12, fontweight='bold')
        ax1.set_ylim(20, 36)
        ax1.grid(axis='y', linestyle=':', alpha=0.6)
        ax1.legend(loc='upper right', frameon=True)
        for bar, val in zip(bars1, maes):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        # RMSE
        bars2 = ax2.bar(labels, rmses, yerr=rmse_errs, capsize=5, color=colors, alpha=0.85, edgecolor='black', width=0.5)
        ax2.axhline(64.05, color='black', linestyle='--', linewidth=1.5, label='Persistence (64.05 W)')
        ax2.set_ylabel('Pooled Test RMSE (Watts)', fontsize=11, fontweight='bold')
        ax2.set_title('Canonical T1 Held-Out Test: RMSE by Architecture', fontsize=12, fontweight='bold')
        ax2.set_ylim(45, 68)
        ax2.grid(axis='y', linestyle=':', alpha=0.6)
        ax2.legend(loc='upper right', frameon=True)
        for bar, val in zip(bars2, rmses):
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.45, f"{val:.2f} W", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plot1_path = plots_dir / 't6_architecture_mae_rmse.png'
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot1_path}", flush=True)

        # 2. Temporal Generalization Comparison across evaluation windows
        fig, ax = plt.subplots(figsize=(10, 5.5))
        periods = ['test_t1', 'test_q1', 'test_march']
        period_labels = ['Canonical T1\n(Jan 28 - Feb 04)', 'Q1 Extended\n(Jan 28 - Apr 01)', 'March Window\n(Mar 01 - Apr 01)']
        x = np.arange(len(periods))
        width = 0.22

        for idx, (cond_id, col) in enumerate([
            ('condition_a_fedavg_gru', '#1f77b4'),
            ('condition_b_fedavg_lstm', '#2ca02c'),
            ('condition_c_centralized_xgboost', '#d62728'),
        ]):
            vals = [df_summary[(df_summary['period'] == p) & (df_summary['condition_id'] == cond_id)]['pooled_mae_mean_w'].values[0] for p in periods]
            ax.bar(x + (idx - 1.0) * width, vals, width, label=T6_CONDITIONS[cond_id]['label'], color=col, alpha=0.85, edgecolor='black')

        ax.set_ylabel('Pooled MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Temporal Generalization: Architectural Comparison Across Deployment Windows', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(period_labels, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plot2_path = plots_dir / 't6_temporal_generalization_daily.png'
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot2_path}", flush=True)

        # 3. Node Improvement Distribution: Delta vs Condition A (GRU)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        p_sub = df_per_node[(df_per_node['seed'] == 42) & (df_per_node['period'] == 'test_t1')]
        p_pivot = p_sub.pivot(index='node_idx', columns='condition_id', values='model_mae_w')

        if 'condition_a_fedavg_gru' in p_pivot.columns:
            ref = p_pivot['condition_a_fedavg_gru']
            for cond_id, col, lbl in [
                ('condition_b_fedavg_lstm', '#2ca02c', 'FedAvg-LSTM vs FedAvg-GRU'),
                ('condition_c_centralized_xgboost', '#d62728', 'Centralized XGBoost vs FedAvg-GRU'),
            ]:
                if cond_id in p_pivot.columns:
                    diff = p_pivot[cond_id] - ref
                    ax.hist(diff, bins=40, alpha=0.55, color=col, label=f"{lbl} (Mean: {diff.mean():+.3f} W)")

        ax.axvline(0.0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Per-Node MAE Difference vs FedAvg-GRU (Watts) [Left = Architecture Improves]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Node Count (out of 978 nodes)', fontsize=11, fontweight='bold')
        ax.set_title('Distribution of Per-Node Forecasting Differences vs FedAvg-GRU', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plot3_path = plots_dir / 't6_node_improvement_distribution.png'
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot3_path}", flush=True)

        # 4. Resource vs. Accuracy Frontier
        fig, ax = plt.subplots(figsize=(10, 5.5))
        res_summary = df_resources.groupby('condition_id').mean(numeric_only=True).reset_index()
        t1_summary = df_summary[df_summary['period'] == 'test_t1'].set_index('condition_id')

        for _, row in res_summary.iterrows():
            c_id = row['condition_id']
            mae = t1_summary.loc[c_id, 'pooled_mae_mean_w']
            comm = row['fl_total_comm_mb']
            lbl = T6_CONDITIONS[c_id]['label']
            color = {'condition_a_fedavg_gru': '#1f77b4', 'condition_b_fedavg_lstm': '#2ca02c', 'condition_c_centralized_xgboost': '#d62728'}[c_id]
            ax.scatter(comm, mae, s=180, color=color, edgecolor='black', linewidth=1.5, label=f"{lbl} ({comm:.1f} MB, {mae:.2f} W)")
            ax.annotate(lbl, (comm, mae), textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')

        ax.set_xlabel('Total FL Communication Volume Over 200 Rounds (MB) [0 = Centralized]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Canonical T1 Held-Out Test MAE (Watts)', fontsize=11, fontweight='bold')
        ax.set_title('Communication vs. Forecasting Accuracy Trade-off Across Architectures', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(frameon=True, fontsize=10, loc='upper right')
        plt.tight_layout()
        plot4_path = plots_dir / 't6_resource_vs_accuracy.png'
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {plot4_path}", flush=True)


if __name__ == '__main__':
    manager = T6ArchitectureBenchmarkManager(device_str='cuda:1')
    results = manager.run_all_benchmark(seeds=[42, 43, 44])
    print("\n[T6Manager] Benchmark completed successfully!", flush=True)
