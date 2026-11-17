"""Task T1.3 — Master Orchestration Script for Multi-Seed History-Length Experiment (Seeds 43 & 44).

Executes:
1. Seed 43:
   - 1h (6 steps, 200 rounds)
   - 6h (36 steps, 200 rounds)
   - 24h (144 steps, 200 rounds)
   - Post-run single-seed analysis
2. Seed 44:
   - 1h (6 steps, 200 rounds)
   - 6h (36 steps, 200 rounds)
   - 24h (144 steps, 200 rounds)
   - Post-run single-seed analysis
3. Multi-seed consolidated analysis across Seeds 42, 43, and 44:
   - Generates MULTISEED_FINAL_REPORT.md
   - Generates comparison CSVs and plots in output/experiments/history_length_comparison_v1/multiseed_ff30_analysis/
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.history_length_real_pilot_v1 import Candidate1RealData, RealPilotRunner
from tools.analyze_history_length_generic import analyze_seed_directory
from tools.analyze_multiseed_ff30 import run_multiseed_analysis


def main():
    parser = argparse.ArgumentParser(description='Master Multi-Seed History Length Runner')
    parser.add_argument('--seeds', type=int, nargs='+', default=[43, 44], help='Seeds to execute')
    parser.add_argument('--rounds', type=int, default=200, help='FedAvg communication rounds')
    parser.add_argument('--device', type=str, default='cuda:1', help='Compute device')
    parser.add_argument('--histories', type=int, nargs='+', default=[6, 36, 144], help='History step lengths to execute')
    args = parser.parse_args()

    base_exp_dir = ROOT / 'output/experiments/history_length_comparison_v1'
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    print('=== Multi-Seed Reproducibility Pipeline Initialized ===', flush=True)
    print(f'Start Time UTC: {datetime.now(timezone.utc).isoformat()}', flush=True)
    print(f'Target Seeds: {args.seeds}', flush=True)
    print(f'History Steps: {args.histories}', flush=True)
    print(f'Rounds per Fit: {args.rounds}', flush=True)
    print(f'Requested Device: {args.device}', flush=True)

    # Determine device with fallback
    dev = torch.device(args.device) if torch.cuda.is_available() and 'cuda' in args.device else torch.device('cpu')
    if dev.type == 'cuda':
        try:
            _ = torch.zeros(1, device=dev)
        except Exception as e:
            fallback_id = 0 if dev.index == 1 else 1
            print(f'NOTICE: Device {dev} unavailable ({e}). Falling back to cuda:{fallback_id}', flush=True)
            dev = torch.device(f'cuda:{fallback_id}')

    print(f'Execution Device Confirmed: {dev}', flush=True)

    # Load telemetry data once
    print('Loading Candidate 1 real telemetry once for pipeline...', flush=True)
    t_data_0 = time.monotonic()
    data = Candidate1RealData()
    print(f'Telemetry loaded in {time.monotonic() - t_data_0:.1f}s. Total nodes: {data.node_count}', flush=True)

    # Execute each seed sequentially
    for seed in args.seeds:
        seed_dir = base_exp_dir / f'seed{seed}_ff30_{args.rounds}r'
        seed_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n========================================================', flush=True)
        print(f'>>> Commencing Experiment Matrix for Seed {seed}...', flush=True)
        print(f'Output Directory: {seed_dir}', flush=True)
        print(f'========================================================', flush=True)

        runner = RealPilotRunner(
            data=data,
            output_dir=seed_dir,
            seed=seed,
            rounds=args.rounds,
            device=dev,
            histories=args.histories,
        )

        res = runner.run_pilot()
        print(f'\n>>> Seed {seed} Training Matrix Complete! Status: {res["status"]}', flush=True)

        # Post-run single-seed analysis
        print(f'>>> Running Single-Seed Analysis for Seed {seed}...', flush=True)
        analyze_seed_directory(exp_dir=seed_dir, seed=seed, data=data, device=dev)
        print(f'>>> Single-Seed Analysis Complete for Seed {seed}!\n', flush=True)

    # Final consolidated multi-seed analysis
    print('\n========================================================', flush=True)
    print('>>> Commencing Final Consolidated Multi-Seed Analysis...', flush=True)
    print('========================================================', flush=True)
    res_multi = run_multiseed_analysis(base_dir=base_exp_dir)

    print('\n=== Multi-Seed Reproducibility Pipeline Complete ===', flush=True)
    print(f'End Time UTC: {datetime.now(timezone.utc).isoformat()}', flush=True)
    print(json.dumps(res_multi, indent=2), flush=True)


if __name__ == '__main__':
    main()
