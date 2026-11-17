"""
Pilot Demonstration for Mandatory Terminal Progress Logging.

Demonstrates:
1. Experiment stage tracking (preprocessing, training, evaluation, reporting).
2. Live Federated Learning (FedAvg) training with client-by-client progress,
   exact round completion blocks, accurate round duration, elapsed time, and ETA.
3. Live evaluation progress tracking with node/window progress bar and throughput rate.
4. Simultaneous unbuffered logging to stdout and persistent progress.log / round_progress.csv.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from src.progress_logging import (
    DualLogger,
    EvaluationProgressLogger,
    ExperimentStageTracker,
    FedAvgProgressLogger,
    TerminalProgressBar,
)


def run_pilot_demonstration(output_dir: Path, interactive: bool = True):
    print(f"\n================================================================================", flush=True)
    print(f"🎬 DEMONSTRATING MANDATORY TERMINAL PROGRESS LOGGING", flush=True)
    print(f"Destination: {output_dir}", flush=True)
    print(f"================================================================================\n", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Experiment Stage Tracking Demonstration
    stage_tracker = ExperimentStageTracker(
        suite_name="Mandatory Progress Logging Pilot",
        total_experiments=1,
        output_dir=output_dir,
    )
    stage_tracker.set_experiment(exp_idx=1, config_name="Pilot-FedAvg-GRU", seed=42)

    stage_tracker.set_stage("PREPROCESSING", "Initializing simulated client telemetry batches")
    time.sleep(0.3)
    stage_tracker.finish_stage()

    # 2. FedAvg Training Demonstration
    stage_tracker.set_stage("TRAINING", "3 rounds FedAvg with 10 clients/round")

    device_str = "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
    total_rounds = 3
    total_clients = 10

    fedavg_logger = FedAvgProgressLogger(
        experiment_name="T5-PILOT",
        config_name="Temp+Humidity-GRU",
        seed=42,
        total_rounds=total_rounds,
        total_clients=total_clients,
        device=device_str,
        output_dir=output_dir,
        interactive=interactive,
    )
    fedavg_logger.start_training()

    # Create dummy lightweight models for realistic loss simulation
    input_dim = 14
    hidden_dim = 32
    global_model = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
    linear = nn.Linear(hidden_dim, 1)

    for r in range(1, total_rounds + 1):
        fedavg_logger.start_round(r)
        round_losses = []

        for c in range(1, total_clients + 1):
            # Simulate client local epoch compute
            time.sleep(0.08)  # Real simulated time per client
            loss_val = 0.2500 - (0.02 * r) + (0.005 * np.random.randn())
            round_losses.append(loss_val)

            # Live client progress update (in-place)
            fedavg_logger.update_client_progress(
                client_idx=c,
                epoch=1,
                total_epochs=1,
                loss=float(loss_val),
            )

        mean_loss = float(np.mean(round_losses))
        # Simulated validation metrics in Watts
        val_mae = 32.50 - (0.85 * r) + float(0.1 * np.random.randn())
        val_rmse = 58.00 - (0.90 * r) + float(0.1 * np.random.randn())

        fedavg_logger.complete_round(
            train_loss=mean_loss,
            val_mae=val_mae,
            val_rmse=val_rmse,
            total_epochs=1,
        )

    fedavg_logger.finish_training()
    stage_tracker.finish_stage()

    # 3. Evaluation Demonstration
    stage_tracker.set_stage("EVALUATION", "Evaluating held-out test windows across 50 nodes")
    total_eval_windows = 25000
    total_eval_nodes = 50

    eval_logger = EvaluationProgressLogger(
        dataset_name="Pilot Test Set (Held-Out)",
        total_windows=total_eval_windows,
        total_nodes=total_eval_nodes,
        model_name="FedAvg-GRU-Pilot",
        output_dir=output_dir,
        interactive=interactive,
    )
    eval_logger.start_evaluation()

    windows_per_node = total_eval_windows // total_eval_nodes
    for node in range(1, total_eval_nodes + 1):
        time.sleep(0.02)  # Simulated evaluation time
        eval_logger.update_node(node_idx=node, windows_in_node=windows_per_node)

    eval_logger.complete_evaluation(
        mae=24.75,
        rmse=57.30,
        macro_mae=25.10,
        macro_rmse=52.40,
        nodes_beating_pers=47,
    )
    stage_tracker.finish_stage()

    # 4. Reporting Stage Demonstration
    stage_tracker.set_stage("REPORTING", "Sealing audit artifacts and verifying logs")
    time.sleep(0.2)
    stage_tracker.finish_stage()
    stage_tracker.finish_suite()

    # 5. Verification of Written Files
    log_file = output_dir / "progress.log"
    csv_file = output_dir / "round_progress.csv"

    print("\n--------------------------------------------------------------------------------", flush=True)
    print("📋 VERIFICATION OF GENERATED ARTIFACTS", flush=True)
    print(f"Log File: {log_file} (Exists: {log_file.exists()}, Size: {log_file.stat().st_size} bytes)", flush=True)
    print(f"CSV File: {csv_file} (Exists: {csv_file.exists()}, Size: {csv_file.stat().st_size} bytes)", flush=True)

    print("\n--- Last 15 lines of progress.log ---", flush=True)
    lines = log_file.read_text(encoding="utf-8").splitlines()
    for line in lines[-15:]:
        print(f"  {line}", flush=True)

    print("\n--- round_progress.csv content ---", flush=True)
    print(csv_file.read_text(encoding="utf-8"), flush=True)
    print("--------------------------------------------------------------------------------\n", flush=True)


if __name__ == "__main__":
    demo_dir = ROOT / "output" / "experiments" / "progress_logging_pilot"
    # Ensure unbuffered python execution
    os.environ["PYTHONUNBUFFERED"] = "1"
    run_pilot_demonstration(demo_dir, interactive=True)
