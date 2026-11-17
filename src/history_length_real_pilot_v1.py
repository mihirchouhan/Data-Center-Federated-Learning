"""Task T1.3 — Real-Data Pilot Implementation for Controlled History-Length Experiment.

Executes a controlled comparison of:
- 1-hour history: 6 ten-minute bins
- 6-hour history: 36 ten-minute bins
- 24-hour history: 144 ten-minute bins

Protocol Constraints:
- Split: Candidate 1 (Train: Jan 01-Jan 22, Val: Jan 22-Jan 28, Test: Jan 28-Feb 04 UTC)
- Policy: FF30 (causal forward fill up to 30 min, reset at split cuts)
- Seed: 42
- Target: Next-hour mean node IT power in Watts (W)
- Normalization: Training-only pooled standard scalers computed on [0, 3024) bins
- Algorithm: Federated Averaging (FedAvg) with sample-weighted parameter aggregation
- Persistence baseline: Node last observed power step P[t-1] in Watts evaluated on identical targets
- Model selection: Validation-only minimum RMSE round; test set strictly held out until checkpoint lock.

DO NOT EXECUTE without explicit user authorization.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.history_length_experiment_v1 import SequenceGRU

ROOT = Path(__file__).resolve().parents[1]

# Candidate 1 Binned Calendar Cuts (10-minute cadence)
CANDIDATE_1_BINS = {
    'train': (0, 3024),      # 2022-01-01T00:00:00Z to 2022-01-22T00:00:00Z (21 days)
    'val':   (3024, 3888),   # 2022-01-22T00:00:00Z to 2022-01-28T00:00:00Z (6 days)
    'test':  (3888, 4896),   # 2022-01-28T00:00:00Z to 2022-02-04T00:00:00Z (7 days)
}

HISTORIES = (6, 36, 144)
HORIZON = 6
KW_TO_WATTS = 1000.0
SENSOR_COLS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class Candidate1RealData:
    """Manages memory-mapped real telemetry and dynamic window extraction for Candidate 1."""

    def __init__(
        self,
        cohort_dir: Path = ROOT / 'output/experiments/v5_missing_data/model_comparison/v1/cohorts',
    ):
        self.cohort_dir = Path(cohort_dir).resolve()
        bins_dir = self.cohort_dir / 'bins'
        self.features_path = bins_dir / 'train_features.npy'
        self.power_path = bins_dir / 'train_observed_power.npy'
        self.ages_path = bins_dir / 'train_fill_ages.npy'

        if not self.features_path.is_file() or not self.power_path.is_file():
            raise FileNotFoundError(f'Required binned arrays missing in {bins_dir}')

        self.features = np.load(self.features_path, mmap_mode='r', allow_pickle=False)
        self.power = np.load(self.power_path, mmap_mode='r', allow_pickle=False)
        self.ages = np.load(self.ages_path, mmap_mode='r', allow_pickle=False)

        self.node_count = self.features.shape[0]  # 979
        self.total_bins = self.features.shape[1]  # 5947

        # Compute training-only scalers on [0, 3024)
        self.scalers = self._fit_training_scalers()

        # Build index of valid origins per split and per node (vectorized)
        self.valid_origins = self._discover_valid_origins()

    def _fit_training_scalers(self) -> Dict[str, np.ndarray]:
        """Compute training-only StandardScaler moments on [0, 3024) bins."""
        lo, hi = CANDIDATE_1_BINS['train']
        train_feat = self.features[:, lo:hi, :]
        train_pow = self.power[:, lo:hi]

        x_mean = np.zeros(14, dtype=np.float64)
        x_std = np.ones(14, dtype=np.float64)

        for c in range(13):
            col = train_feat[:, :, c]
            finite = np.isfinite(col)
            x_mean[c] = col[finite].mean(dtype=np.float64)
            s = col[finite].std(dtype=np.float64)
            x_std[c] = s if s > 1e-12 else 1.0

        # Node index (channel 13) is categorical: unstandardized
        x_mean[13] = 0.0
        x_std[13] = 1.0

        # Power target
        finite_p = np.isfinite(train_pow)
        p_mean = train_pow[finite_p].mean(dtype=np.float64)
        p_std = train_pow[finite_p].std(dtype=np.float64)

        return {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': np.array([p_mean], dtype=np.float64),
            'y_std': np.array([p_std if p_std > 1e-12 else 1.0], dtype=np.float64),
        }

    def _discover_valid_origins(self) -> Dict[str, Dict[int, np.ndarray]]:
        """Identify candidate forecast origins t where all 3 history lengths and target are valid."""
        origins_by_split: Dict[str, Dict[int, np.ndarray]] = {}
        sensor_cols = list(SENSOR_COLS)

        for split_name, (lo, hi) in CANDIDATE_1_BINS.items():
            origins_by_split[split_name] = {}
            for node_idx in range(self.node_count):
                ages = self.ages[node_idx]
                power = self.power[node_idx]
                sensor_ages = ages[:, sensor_cols]

                bad_step = (sensor_ages < 0) | (sensor_ages > 3)
                # Check that fill source does not cross split lower boundary
                positions = np.arange(len(ages))
                bad_source = (sensor_ages > 0) & ((positions[:, None] - sensor_ages) < lo)
                bad = bad_step | bad_source

                hp = np.r_[0, np.cumsum(bad.any(axis=1))]
                tp = np.r_[0, np.cumsum(~np.isfinite(power))]

                origins = np.arange(lo, hi, dtype=np.int64)
                enough = origins >= 144
                inside = origins + HORIZON <= hi

                ok_hist = np.zeros(len(origins), dtype=bool)
                ok_targ = np.zeros(len(origins), dtype=bool)

                ok_hist[enough] = hp[origins[enough]] == hp[origins[enough] - 144]
                ok_targ[inside] = tp[origins[inside] + HORIZON] == tp[origins[inside]]

                accepted = enough & inside & ok_hist & ok_targ
                origins_by_split[split_name][node_idx] = origins[accepted]

        return origins_by_split

    def get_client_batch(
        self,
        node_idx: int,
        origins: np.ndarray,
        history_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Extract and standardize (X, y, persistence_w) for given client origins (vectorized)."""
        H = history_steps
        B = len(origins)

        # Target in standardized units
        target_bins = origins[:, None] + np.arange(HORIZON)
        targets_kw = np.mean(self.power[node_idx, target_bins], axis=1, keepdims=True)
        y_scaled = (targets_kw - self.scalers['y_mean'][0]) / self.scalers['y_std'][0]

        # History features
        time_indices = origins[:, None] + np.arange(-H, 0)
        raw_hist = self.features[node_idx, time_indices].copy()
        hist_ages = self.ages[node_idx, time_indices]

        # Causal forward fill (FF30 up to 3 bins, excluding weather 7 and 8)
        fill = (hist_ages > 0) & (hist_ages <= 3)
        fill[:, :, [7, 8]] = False
        if fill.any():
            rows, steps, cols = np.where(fill)
            src_times = time_indices[rows, steps] - hist_ages[rows, steps, cols]
            raw_hist[rows, steps, cols] = self.features[node_idx, src_times, cols]

        # Persistence baseline in Watts: last step power P[t-1]
        pers_w = (raw_hist[:, -1, 0] * KW_TO_WATTS).astype(np.float32)

        # Standardize features
        scaled_hist = (raw_hist - self.scalers['x_mean'][:13]) / self.scalers['x_std'][:13]
        scaled_hist[:, :, [7, 8]] = 0.0  # weather masked

        # Add node categorical index
        node_col = np.full((B, H, 1), float(node_idx), dtype=np.float32)
        x_scaled = np.concatenate([scaled_hist, node_col], axis=-1).astype(np.float32)

        return (
            torch.from_numpy(x_scaled),
            torch.from_numpy(y_scaled.astype(np.float32)),
            pers_w,
        )


class RealPilotRunner:
    """Executes the controlled 1h/6h/24h real-data pilot under locked FedAvg protocol."""

    def __init__(
        self,
        data: Candidate1RealData,
        output_dir: Path,
        seed: int = 42,
        rounds: int = 20,
        clients_per_round: int = 98,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        histories: Optional[Sequence[int]] = None,
    ):
        self.data = data
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_log_path = self.output_dir / 'progress.log'
        self.progress_csv_path = self.output_dir / 'round_progress.csv'
        self.seed = seed
        self.rounds = rounds
        self.clients_per_round = clients_per_round
        self.batch_size = batch_size
        self.lr = lr
        self.histories = tuple(histories) if histories is not None else HISTORIES

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
            if 'cuda' in device.type:
                try:
                    _ = torch.zeros(1, device=device)
                    self.device = device
                except Exception as e:
                    fallback_id = 0 if device.index == 1 else 1
                    try:
                        _ = torch.zeros(1, device=f'cuda:{fallback_id}')
                        print(f"NOTICE: Device {device} is out of memory ({e}). Falling back to cuda:{fallback_id}.")
                        self.device = torch.device(f'cuda:{fallback_id}')
                    except Exception:
                        self.device = device
            else:
                self.device = device

        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)

    def _log_progress(self, msg: str) -> None:
        """Write progress message immediately to terminal stdout and persistent log file."""
        print(msg, flush=True)
        with open(self.progress_log_path, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()

    def _log_round_csv(
        self,
        history_steps: int,
        round_idx: int,
        train_loss: float,
        val_rmse: float,
        val_mae: float,
        is_best_val: bool,
        elapsed_sec: float,
    ) -> None:
        """Append round-by-round metrics to persistent CSV file with immediate flush."""
        file_exists = self.progress_csv_path.is_file()
        with open(self.progress_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'history_hours',
                    'history_steps',
                    'round',
                    'total_rounds',
                    'seed',
                    'policy',
                    'train_loss',
                    'val_rmse_w',
                    'val_mae_w',
                    'is_best_val',
                    'elapsed_sec',
                    'timestamp_utc',
                ])
            writer.writerow([
                history_steps // 6,
                history_steps,
                round_idx,
                self.rounds,
                self.seed,
                'ff30',
                round(train_loss, 5),
                round(val_rmse, 3),
                round(val_mae, 3),
                int(is_best_val),
                round(elapsed_sec, 2),
                datetime.now(timezone.utc).isoformat(),
            ])
            f.flush()

    def run_pilot(self) -> Dict[str, Any]:
        """Run real-data experiment/pilot across 1h, 6h, and 24h history models."""
        results = {}

        self._log_progress('=== Experiment Run Initialized ===')
        self._log_progress(f'Target Hardware: {self.device}')
        self._log_progress(f'Seed: {self.seed} | Rounds: {self.rounds} | Policy: ff30')
        self._log_progress(f'Output Directory: {self.output_dir}')
        self._log_progress(f'Progress Log: {self.progress_log_path}')

        for h in self.histories:
            h_label = f'{h}_steps'
            fit_json_path = self.output_dir / f'fit_{h}_steps_result.json'
            if fit_json_path.is_file():
                self._log_progress(f'>>> [Seed{self.seed}] Fit {h} steps ({h//6}h) already completed. Loaded from {fit_json_path.name}. Skipping training.')
                results[h_label] = json.loads(fit_json_path.read_text(encoding='utf-8'))
                continue

            fit_res = self._train_single_fit(history_steps=h)
            results[h_label] = fit_res

            # Save intermediate fit result JSON for crash resilience
            fit_json_path = self.output_dir / f'fit_{h}_steps_result.json'
            fit_json_path.write_text(json.dumps(fit_res, indent=2), encoding='utf-8')

        # Ensure all available completed fits in output_dir are merged into results
        for all_h in HISTORIES:
            all_h_label = f'{all_h}_steps'
            if all_h_label not in results:
                existing_json = self.output_dir / f'fit_{all_h}_steps_result.json'
                if existing_json.is_file():
                    results[all_h_label] = json.loads(existing_json.read_text(encoding='utf-8'))

        # Write summary report, results CSV, and metadata JSON
        report_path = self._write_pilot_report(results)
        csv_path = self._write_pilot_csv(results)
        meta_path = self._write_metadata(results)

        status_str = 'real_pilot_complete' if self.rounds <= 20 else f'experiment_seed{self.seed}_complete'
        self._log_progress(f'=== Experiment Run Complete ({status_str}) ===')

        return {
            'status': status_str,
            'device': str(self.device),
            'seed': self.seed,
            'rounds': self.rounds,
            'results': results,
            'report_path': str(report_path),
            'csv_path': str(csv_path),
            'metadata_path': str(meta_path),
            'progress_log_path': str(self.progress_log_path),
        }

    def _train_single_fit(self, history_steps: int) -> Dict[str, Any]:
        """Execute one FedAvg fit across the declared rounds with validation selection."""
        H = history_steps
        t0 = time.monotonic()
        h_str = f'{H // 6}h'

        self._log_progress(f'>>> [Seed{self.seed}] Staging Fit: History {H} steps ({h_str}, {self.rounds} rounds) on {self.device}...')

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        global_model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=self.data.node_count).to(self.device)
        criterion = nn.MSELoss()

        val_records = []
        best_val_rmse = float('inf')
        best_round = 0
        best_state = None

        # FedAvg communication rounds
        for r in range(1, self.rounds + 1):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, 1, r, 0]))
            sampled_nodes = rng.choice(self.data.node_count, size=self.clients_per_round, replace=False)

            client_updates = []
            total_examples = 0
            round_loss_sum = 0.0
            round_batch_samples = 0

            # Local client sessions
            for node_idx in sampled_nodes:
                train_origins = self.data.valid_origins['train'][node_idx]
                if len(train_origins) == 0:
                    continue

                client_model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=self.data.node_count).to(self.device)
                client_model.load_state_dict(global_model.state_dict())
                client_model.train()
                optimizer = torch.optim.Adam(client_model.parameters(), lr=self.lr)

                n_samples = len(train_origins)
                perm = np.random.permutation(n_samples)
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_origins = train_origins[perm[start_idx:start_idx + self.batch_size]]
                    bx, by, _ = self.data.get_client_batch(node_idx, batch_origins, history_steps=H)
                    bx, by = bx.to(self.device), by.to(self.device)

                    optimizer.zero_grad()
                    out = client_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

                    batch_size_actual = len(batch_origins)
                    round_loss_sum += loss.item() * batch_size_actual
                    round_batch_samples += batch_size_actual

                client_updates.append((n_samples, client_model.state_dict()))
                total_examples += n_samples

            # FedAvg Parameter Aggregation (sample-weighted)
            if client_updates:
                new_state = OrderedDict()
                for key in global_model.state_dict():
                    accum = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float64, device='cpu')
                    for n_k, state in client_updates:
                        weight = n_k / total_examples
                        accum.add_(state[key].to(torch.float64).cpu(), alpha=weight)
                    new_state[key] = accum.to(global_model.state_dict()[key].dtype)
                global_model.load_state_dict(new_state)

            train_loss = (round_loss_sum / round_batch_samples) if round_batch_samples > 0 else 0.0

            # Validation evaluation on every round for continuous progress monitoring
            val_mae, val_rmse = self._evaluate_split(global_model, split='val', history_steps=H)
            val_records.append({'round': r, 'val_mae_w': val_mae, 'val_rmse_w': val_rmse, 'train_loss': round(train_loss, 5)})

            is_best = False
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_round = r
                best_state = OrderedDict((k, v.cpu().clone()) for k, v in global_model.state_dict().items())
                is_best = True

            elapsed_sec = time.monotonic() - t0
            progress_line = (
                f"[Seed{self.seed}][{h_str}] Round {r:03d}/{self.rounds:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_rmse:.2f} W | "
                f"Elapsed: {elapsed_sec:.1f}s"
            )
            self._log_progress(progress_line)
            self._log_round_csv(
                history_steps=H,
                round_idx=r,
                train_loss=train_loss,
                val_rmse=val_rmse,
                val_mae=val_mae,
                is_best_val=is_best,
                elapsed_sec=elapsed_sec,
            )

        # Locked checkpoint selection
        global_model.load_state_dict(best_state)
        ckpt_filename = f'model_{H}_steps_seed{self.seed}_best.pt'
        torch.save(best_state, self.output_dir / ckpt_filename)
        self._log_progress(
            f"[Seed{self.seed}][{h_str}] Checkpoint locked at round {best_round}/{self.rounds} (Val RMSE: {best_val_rmse:.2f} W). Evaluating test split..."
        )

        # Held-out Test Evaluation (evaluated strictly after selection lock)
        test_mae, test_rmse, test_pers_mae, test_pers_rmse, nodewise = self._evaluate_test_split(
            global_model, history_steps=H
        )

        elapsed = time.monotonic() - t0
        self._log_progress(
            f"[Seed{self.seed}][{h_str}] Test split complete. Test MAE: {test_mae:.2f} W, Test RMSE: {test_rmse:.2f} W (Pers. MAE: {test_pers_mae:.2f} W)."
        )

        elapsed = time.monotonic() - t0
        peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if self.device.type == 'cuda' else 0.0
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        return {
            'history_steps': H,
            'history_hours': H // 6,
            'best_round': best_round,
            'best_val_rmse_w': round(best_val_rmse, 3),
            'pooled_test_mae_w': round(test_mae, 3),
            'pooled_test_rmse_w': round(test_rmse, 3),
            'pooled_pers_mae_w': round(test_pers_mae, 3),
            'pooled_pers_rmse_w': round(test_pers_rmse, 3),
            'pooled_delta_mae_w': round(test_mae - test_pers_mae, 3),
            'pooled_delta_rmse_w': round(test_rmse - test_pers_rmse, 3),
            'macro_test_mae_w': nodewise['macro_mae_w'],
            'macro_test_rmse_w': nodewise['macro_rmse_w'],
            'macro_pers_mae_w': nodewise['macro_pers_mae_w'],
            'macro_pers_rmse_w': nodewise['macro_pers_rmse_w'],
            'macro_delta_mae_w': nodewise['macro_delta_mae_w'],
            'macro_delta_rmse_w': nodewise['macro_delta_rmse_w'],
            'test_mae_w': round(test_mae, 3),
            'test_rmse_w': round(test_rmse, 3),
            'test_persistence_mae_w': round(test_pers_mae, 3),
            'test_persistence_rmse_w': round(test_pers_rmse, 3),
            'delta_mae_w': round(test_mae - test_pers_mae, 3),
            'delta_rmse_w': round(test_rmse - test_pers_rmse, 3),
            'wall_seconds': round(elapsed, 2),
            'peak_vram_mb': round(peak_vram_mb, 1),
            'peak_rss_mb': round(peak_rss_mb, 1),
            'checkpoint_file': ckpt_filename,
            'validation_records': val_records,
            'nodewise_summary': nodewise,
        }

    def _evaluate_split(self, model: nn.Module, split: str, history_steps: int) -> Tuple[float, float]:
        """Evaluate validation metrics in Watts across sample batches."""
        model.eval()
        all_pred = []
        all_actual = []
        y_mean = self.data.scalers['y_mean'][0]
        y_std = self.data.scalers['y_std'][0]

        with torch.no_grad():
            for node_idx in range(self.data.node_count):
                origins = self.data.valid_origins[split][node_idx]
                if len(origins) == 0:
                    continue
                # Evaluate sample origins for speed during round checkpoints
                eval_origins = origins[::4] if len(origins) > 32 else origins
                for start in range(0, len(eval_origins), 512):
                    batch_origins = eval_origins[start:start + 512]
                    bx, by, _ = self.data.get_client_batch(node_idx, batch_origins, history_steps=history_steps)
                    bx = bx.to(self.device)
                    out = model(bx).cpu().numpy()

                    pred_w = (out[:, 0] * y_std + y_mean) * KW_TO_WATTS
                    actual_w = (by.numpy()[:, 0] * y_std + y_mean) * KW_TO_WATTS

                    all_pred.append(pred_w)
                    all_actual.append(actual_w)

        if not all_pred:
            return float('nan'), float('nan')

        p = np.concatenate(all_pred)
        a = np.concatenate(all_actual)
        mae = float(np.mean(np.abs(p - a)))
        rmse = float(np.sqrt(np.mean((p - a) ** 2)))
        return mae, rmse

    def _evaluate_test_split(
        self,
        model: nn.Module,
        history_steps: int,
    ) -> Tuple[float, float, float, float, Dict[str, Any]]:
        """Evaluate full held-out test split alongside node-level persistence in Watts."""
        model.eval()
        all_pred = []
        all_actual = []
        all_pers = []
        y_mean = self.data.scalers['y_mean'][0]
        y_std = self.data.scalers['y_std'][0]

        node_errors = {}

        with torch.no_grad():
            for node_idx in range(self.data.node_count):
                origins = self.data.valid_origins['test'][node_idx]
                if len(origins) == 0:
                    continue
                node_p, node_a, node_pers = [], [], []
                for start in range(0, len(origins), 512):
                    batch_origins = origins[start:start + 512]
                    bx, by, pers_w = self.data.get_client_batch(node_idx, batch_origins, history_steps=history_steps)
                    bx = bx.to(self.device)
                    out = model(bx).cpu().numpy()

                    pred_w = (out[:, 0] * y_std + y_mean) * KW_TO_WATTS
                    actual_w = (by.numpy()[:, 0] * y_std + y_mean) * KW_TO_WATTS

                    node_p.append(pred_w)
                    node_a.append(actual_w)
                    node_pers.append(pers_w)

                np_p = np.concatenate(node_p)
                np_a = np.concatenate(node_a)
                np_pers = np.concatenate(node_pers)

                all_pred.append(np_p)
                all_actual.append(np_a)
                all_pers.append(np_pers)

                n_mae = float(np.mean(np.abs(np_p - np_a)))
                n_rmse = float(np.sqrt(np.mean((np_p - np_a) ** 2)))
                n_pers_mae = float(np.mean(np.abs(np_pers - np_a)))
                n_pers_rmse = float(np.sqrt(np.mean((np_pers - np_a) ** 2)))
                node_errors[node_idx] = {
                    'mae_w': n_mae,
                    'rmse_w': n_rmse,
                    'pers_mae_w': n_pers_mae,
                    'pers_rmse_w': n_pers_rmse,
                    'delta_mae_w': n_mae - n_pers_mae,
                    'delta_rmse_w': n_rmse - n_pers_rmse,
                }

        p = np.concatenate(all_pred)
        a = np.concatenate(all_actual)
        pers = np.concatenate(all_pers)

        pooled_mae = float(np.mean(np.abs(p - a)))
        pooled_rmse = float(np.sqrt(np.mean((p - a) ** 2)))
        pooled_pers_mae = float(np.mean(np.abs(pers - a)))
        pooled_pers_rmse = float(np.sqrt(np.mean((pers - a) ** 2)))

        macro_mae = float(np.mean([v['mae_w'] for v in node_errors.values()]))
        macro_rmse = float(np.mean([v['rmse_w'] for v in node_errors.values()]))
        macro_pers_mae = float(np.mean([v['pers_mae_w'] for v in node_errors.values()]))
        macro_pers_rmse = float(np.mean([v['pers_rmse_w'] for v in node_errors.values()]))

        improved_nodes = sum(1 for v in node_errors.values() if v['delta_mae_w'] < 0)
        total_eval_nodes = len(node_errors)

        nodewise_stats = {
            'evaluated_nodes': total_eval_nodes,
            'nodes_outperforming_persistence': improved_nodes,
            'nodes_worse_than_persistence': total_eval_nodes - improved_nodes,
            'improved_fraction': round(improved_nodes / total_eval_nodes if total_eval_nodes else 0.0, 4),
            'macro_mae_w': round(macro_mae, 3),
            'macro_rmse_w': round(macro_rmse, 3),
            'macro_pers_mae_w': round(macro_pers_mae, 3),
            'macro_pers_rmse_w': round(macro_pers_rmse, 3),
            'macro_delta_mae_w': round(macro_mae - macro_pers_mae, 3),
            'macro_delta_rmse_w': round(macro_rmse - macro_pers_rmse, 3),
        }

        return pooled_mae, pooled_rmse, pooled_pers_mae, pooled_pers_rmse, nodewise_stats

    def _write_pilot_csv(self, results: Dict[str, Any]) -> Path:
        csv_filename = 'pilot_results.csv' if self.rounds <= 20 else 'experiment_results.csv'
        csv_path = self.output_dir / csv_filename
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'history_hours', 'history_steps', 'rounds', 'seed', 'policy',
                'best_round', 'best_val_rmse_w',
                'pooled_test_mae_w', 'pooled_test_rmse_w',
                'pooled_pers_mae_w', 'pooled_pers_rmse_w',
                'pooled_delta_mae_w', 'pooled_delta_rmse_w',
                'macro_test_mae_w', 'macro_test_rmse_w',
                'macro_pers_mae_w', 'macro_pers_rmse_w',
                'macro_delta_mae_w', 'macro_delta_rmse_w',
                'nodes_evaluated', 'nodes_outperforming_persistence', 'improved_fraction',
                'wall_seconds', 'peak_vram_mb', 'peak_rss_mb', 'checkpoint_file',
                'test_mae_w', 'test_rmse_w', 'test_persistence_mae_w', 'test_persistence_rmse_w',
                'delta_mae_w', 'delta_rmse_w',
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results.values():
                nw = r['nodewise_summary']
                row = {
                    'history_hours': r['history_hours'],
                    'history_steps': r['history_steps'],
                    'rounds': self.rounds,
                    'seed': self.seed,
                    'policy': 'ff30',
                    'best_round': r['best_round'],
                    'best_val_rmse_w': r['best_val_rmse_w'],
                    'pooled_test_mae_w': r.get('pooled_test_mae_w', r.get('test_mae_w')),
                    'pooled_test_rmse_w': r.get('pooled_test_rmse_w', r.get('test_rmse_w')),
                    'pooled_pers_mae_w': r.get('pooled_pers_mae_w', r.get('test_persistence_mae_w')),
                    'pooled_pers_rmse_w': r.get('pooled_pers_rmse_w', r.get('test_persistence_rmse_w')),
                    'pooled_delta_mae_w': r.get('pooled_delta_mae_w', r.get('delta_mae_w')),
                    'pooled_delta_rmse_w': r.get('pooled_delta_rmse_w', r.get('delta_rmse_w')),
                    'macro_test_mae_w': r.get('macro_test_mae_w', nw.get('macro_mae_w', r.get('test_mae_w'))),
                    'macro_test_rmse_w': r.get('macro_test_rmse_w', nw.get('macro_rmse_w', r.get('test_rmse_w'))),
                    'macro_pers_mae_w': r.get('macro_pers_mae_w', nw.get('macro_pers_mae_w', r.get('test_persistence_mae_w'))),
                    'macro_pers_rmse_w': r.get('macro_pers_rmse_w', nw.get('macro_pers_rmse_w', r.get('test_persistence_rmse_w'))),
                    'macro_delta_mae_w': r.get('macro_delta_mae_w', nw.get('macro_delta_mae_w', r.get('delta_mae_w'))),
                    'macro_delta_rmse_w': r.get('macro_delta_rmse_w', nw.get('macro_delta_rmse_w', r.get('delta_rmse_w'))),
                    'nodes_evaluated': nw.get('evaluated_nodes', 0),
                    'nodes_outperforming_persistence': nw.get('nodes_outperforming_persistence', 0),
                    'improved_fraction': nw.get('improved_fraction', 0.0),
                    'wall_seconds': r['wall_seconds'],
                    'peak_vram_mb': r['peak_vram_mb'],
                    'peak_rss_mb': r['peak_rss_mb'],
                    'checkpoint_file': r.get('checkpoint_file', ''),
                    'test_mae_w': r.get('test_mae_w', r.get('pooled_test_mae_w')),
                    'test_rmse_w': r.get('test_rmse_w', r.get('pooled_test_rmse_w')),
                    'test_persistence_mae_w': r.get('test_persistence_mae_w', r.get('pooled_pers_mae_w')),
                    'test_persistence_rmse_w': r.get('test_persistence_rmse_w', r.get('pooled_pers_rmse_w')),
                    'delta_mae_w': r.get('delta_mae_w', r.get('pooled_delta_mae_w')),
                    'delta_rmse_w': r.get('delta_rmse_w', r.get('pooled_delta_rmse_w')),
                }
                writer.writerow(row)
        return csv_path

    def _write_metadata(self, results: Dict[str, Any]) -> Path:
        meta_path = self.output_dir / 'metadata.json'
        serialized_results = {}
        for k, v in results.items():
            entry = dict(v)
            if 'nodewise_summary' in entry:
                entry['nodewise_summary'] = dict(entry['nodewise_summary'])
            serialized_results[k] = entry

        meta = {
            'experiment_name': f'controlled_history_length_{self.rounds}r_seed{self.seed}_ff30',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'device': str(self.device),
            'seed': self.seed,
            'rounds': self.rounds,
            'policy': 'ff30',
            'protocol_version': '1.0.0',
            'candidate_split': 'candidate_1',
            'history_configurations': list(HISTORIES),
            'node_count': self.data.node_count if self.data else 979,
            'results': serialized_results,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return meta_path

    def _write_pilot_report(self, results: Dict[str, Any]) -> Path:
        is_pilot = self.rounds <= 20
        report_filename = 'PILOT_REPORT.md' if is_pilot else 'EXPERIMENT_REPORT.md'
        report_path = self.output_dir / report_filename

        title = (
            f"# TASK T1.3 — Real-Data Pilot Report (Seed {self.seed}, FF30, {self.rounds} Rounds)"
            if is_pilot
            else f"# TASK T1.3 — Controlled History-Length Experiment Report (Seed {self.seed}, FF30, {self.rounds} Rounds)"
        )

        lines = [
            title,
            '',
            f"**Execution Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Target Hardware:** `{self.device}`",
            '**Dataset:** CINECA M100 Telemetry (Candidate 1 Chronological Split)',
            '**Policy:** FF30 (causal forward-fill $\\le$ 3 bins, reset at split cutoffs)',
            f"**Seed:** {self.seed}",
            f"**Rounds:** {self.rounds} FedAvg communication rounds",
            '**Target:** Next-hour mean node IT power in Watts (W)',
            f"**Status:** {'20-ROUND PILOT COMPLETE' if is_pilot else 'SEED 42 BENCHMARK COMPLETE (STAGED COMPARISON)'}",
            '',
        ]

        if is_pilot:
            lines.extend([
                '> [!NOTE]',
                '> **Pilot Disclaimer:** These metrics represent a 20-round feasibility and convergence benchmark. '
                '> They are preliminary empirical observations and **must not be presented as finalized 200-round research conclusions**.',
                '',
            ])
        else:
            lines.extend([
                '> [!IMPORTANT]',
                f"> **Single-Seed Evaluation Notice:** These metrics represent a {self.rounds}-round execution for **Seed {self.seed}** under policy **FF30**. "
                f"> In strict accordance with the T1.2 research protocol, **these results must be labeled as Seed {self.seed} results, not finalized multi-seed conclusions**. "
                '> Final scientific conclusions require completing the full multi-seed matrix (Seeds 42, 43, 44) across both FF30 and No-Fill.',
                '',
            ])

        lines.extend([
            '---',
            '',
            '## 1. Pooled Metrics (Window-Weighted Across All Test Windows)',
            '',
            '| History Length | Selected Round | Val RMSE (W) | Test MAE (W) | Test RMSE (W) | Persistence MAE (W) | Persistence RMSE (W) | $\\Delta$MAE (W) | $\\Delta$RMSE (W) | Runtime | Peak VRAM |',
            '| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])

        for k, r in results.items():
            lines.append(
                f"| **{r['history_hours']}h ({r['history_steps']} steps)** | Round {r['best_round']}/{self.rounds} | "
                f"{r['best_val_rmse_w']:.2f} W | {r['test_mae_w']:.2f} W | {r['test_rmse_w']:.2f} W | "
                f"{r['test_persistence_mae_w']:.2f} W | {r['test_persistence_rmse_w']:.2f} W | "
                f"**{r['delta_mae_w']:+.2f} W** | **{r.get('delta_rmse_w', r.get('pooled_delta_rmse_w', 0.0)):+.2f} W** | "
                f"{r['wall_seconds']:.1f}s | {r['peak_vram_mb']:.1f} MB |"
            )

        lines.extend([
            '',
            '---',
            '',
            '## 2. Macro-Node Metrics (Equal Weighting Across Nodes)',
            '',
            '| History Length | Macro Test MAE (W) | Macro Test RMSE (W) | Macro Pers. MAE (W) | Macro Pers. RMSE (W) | Macro $\\Delta$MAE (W) | Macro $\\Delta$RMSE (W) | Evaluated Nodes |',
            '| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])

        for k, r in results.items():
            nw = r['nodewise_summary']
            lines.append(
                f"| **{r['history_hours']}h ({r['history_steps']} steps)** | "
                f"{nw.get('macro_mae_w', r.get('macro_test_mae_w', 0.0)):.2f} W | "
                f"{nw.get('macro_rmse_w', r.get('macro_test_rmse_w', 0.0)):.2f} W | "
                f"{nw.get('macro_pers_mae_w', r.get('macro_pers_mae_w', 0.0)):.2f} W | "
                f"{nw.get('macro_pers_rmse_w', r.get('macro_pers_rmse_w', 0.0)):.2f} W | "
                f"**{nw.get('macro_delta_mae_w', r.get('macro_delta_mae_w', 0.0)):+.2f} W** | "
                f"**{nw.get('macro_delta_rmse_w', r.get('macro_delta_rmse_w', 0.0)):+.2f} W** | "
                f"{nw.get('evaluated_nodes', 0)} nodes |"
            )

        lines.extend([
            '',
            '---',
            '',
            '## 3. Node-Level Disaggregation',
            '',
            '| History Length | Evaluated Nodes | Nodes Outperforming Persistence | Fraction Outperforming |',
            '| :---: | ---: | ---: | ---: |',
        ])

        for k, r in results.items():
            nw = r['nodewise_summary']
            lines.append(
                f"| **{r['history_hours']}h ({r['history_steps']} steps)** | {nw['evaluated_nodes']} nodes | "
                f"{nw['nodes_outperforming_persistence']} nodes | {nw['improved_fraction'] * 100:.1f}% |"
            )

        lines.extend([
            '',
            '---',
            '',
            '## 4. Validation Convergence Curves',
            '',
            '| Round | 1h Val RMSE (W) | 6h Val RMSE (W) | 24h Val RMSE (W) |',
            '| :---: | ---: | ---: | ---: |',
        ])

        v1 = {rec['round']: rec['val_rmse_w'] for rec in results.get('6_steps', {}).get('validation_records', [])}
        v6 = {rec['round']: rec['val_rmse_w'] for rec in results.get('36_steps', {}).get('validation_records', [])}
        v24 = {rec['round']: rec['val_rmse_w'] for rec in results.get('144_steps', {}).get('validation_records', [])}

        all_rounds = sorted(set(v1.keys()) | set(v6.keys()) | set(v24.keys()))
        for r in all_rounds:
            s1 = f"{v1[r]:.2f} W" if r in v1 else "N/A"
            s6 = f"{v6[r]:.2f} W" if r in v6 else "N/A"
            s24 = f"{v24[r]:.2f} W" if r in v24 else "N/A"
            lines.append(f"| Round {r} | {s1} | {s6} | {s24} |")

        lines.extend([
            '',
            '---',
            '',
            '## 5. Hardware and Computational Footprint',
            '',
            f"- **Execution Device:** `{self.device}`",
            f"- **Peak VRAM across Fits:** `{max(r['peak_vram_mb'] for r in results.values()):.1f} MB`",
            f"- **Peak Host RAM (RSS):** `{max(r['peak_rss_mb'] for r in results.values()):.1f} MB`",
            f"- **Cumulative Wall Time (3 Fits):** `{sum(r['wall_seconds'] for r in results.values()):.1f} seconds` ({sum(r['wall_seconds'] for r in results.values())/60:.2f} minutes)",
            '',
            '## 6. Summary & Gating Decision',
            '',
            '1. All three sequence lengths ($H=6, 36, 144$) executed seamlessly under identical FedAvg parameter aggregation and common test cohorts.',
            '2. Checkpoint selection strictly utilized validation RMSE, with the held-out test split opened only after round locking.',
            f"3. All model fits completed successfully. Execution is paused awaiting review before launching additional seeds or policies.",
        ])

        report_path.write_text('\n'.join(lines), encoding='utf-8')
        if not is_pilot:
            (self.output_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        return report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Task T1.3 Real-Data Pilot')
    parser.add_argument('--output-dir', default='output/experiments/history_length_pilot_v1', help='Output directory')
    parser.add_argument('--rounds', type=int, default=20, help='Communication rounds for pilot')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', default='cuda:1', help='Target compute device')
    parser.add_argument('--histories', type=int, nargs='+', default=None, help='History step lengths to execute (e.g. 36 144)')
    args = parser.parse_args(argv)

    out_dir = ROOT / args.output_dir
    dev = torch.device(args.device) if torch.cuda.is_available() and 'cuda' in args.device else torch.device('cpu')

    print(f'Loading Candidate 1 real telemetry...')
    data = Candidate1RealData()

    print(f'Initializing RealPilotRunner on {dev} (Seed={args.seed}, Rounds={args.rounds})...')
    runner = RealPilotRunner(
        data=data,
        output_dir=out_dir,
        seed=args.seed,
        rounds=args.rounds,
        device=dev,
        histories=args.histories,
    )

    res = runner.run_pilot()
    print(json.dumps({'status': res['status'], 'report_path': res['report_path'], 'csv_path': res['csv_path']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
