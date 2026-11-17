"""Unit tests for Task T1.3 Controlled History-Length Experiment Preflight & Implementation.

Verifies:
- All 10 preflight checks pass cleanly.
- SequenceGRU adapts to history steps 6, 36, 144.
- Node-level persistence baseline computes correctly in Watts.
- Resource estimator accurately benchmarks forward/backward steps.
- Safe pilot benchmark executes cleanly without errors.
- Preflight report generation.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.history_length_experiment_v1 import (
    CANDIDATE_1_SPLITS,
    HISTORIES,
    KW_TO_WATTS,
    PreflightVerifier,
    ResourceEstimator,
    SequenceGRU,
    compute_persistence_prediction,
    run_synthetic_pilot_benchmark,
    write_t1_3_preflight_report,
)

ROOT = Path(__file__).resolve().parents[1]


class TestSequenceGRUArchitecture(unittest.TestCase):
    """Verify that SequenceGRU dynamically handles sequence lengths 6, 36, and 144."""

    def test_dynamic_sequence_lengths(self):
        """Model must process inputs of sequence lengths 6, 36, and 144 seamlessly."""
        model = SequenceGRU(input_dim=14, hidden_dim=64, num_layers=1, node_count=979)
        model.eval()

        for h in (6, 36, 144):
            batch_size = 8
            x = torch.randn(batch_size, h, 14)
            # Last channel is node_index (must be int in [0, 978])
            x[:, :, -1] = torch.randint(0, 979, (batch_size, h)).float()

            with torch.no_grad():
                out = model(x)

            self.assertEqual(out.shape, (batch_size, 1))
            self.assertTrue(torch.isfinite(out).all().item())


class TestPersistenceBaselineCalculation(unittest.TestCase):
    """Verify persistence baseline formulation and unit conversion to Watts."""

    def test_persistence_in_watts(self):
        """Persistence must take last power step P[t-1], unscale it, and convert to Watts."""
        batch_size = 4
        seq_len = 36
        inputs = np.zeros((batch_size, seq_len, 14), dtype=np.float32)

        # Set last time step (index -1) power (channel 0)
        # Scaled values: [0.0, 1.0, -1.0, 2.0]
        inputs[:, -1, 0] = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

        x_mean = 0.5  # kW
        x_std = 0.2   # kW

        # Unscaled kW:
        # [0.5, 0.7, 0.3, 0.9] kW
        # In Watts:
        # [500.0, 700.0, 300.0, 900.0] W
        expected_w = np.array([500.0, 700.0, 300.0, 900.0], dtype=np.float32)

        pers_w = compute_persistence_prediction(inputs, x_mean=x_mean, x_std=x_std)

        np.testing.assert_allclose(pers_w, expected_w, rtol=1e-5)


class TestPreflightVerificationChecks(unittest.TestCase):
    """Verify that all 10 Phase A preflight checks pass on repository contracts."""

    def test_preflight_verifier_passes_all_ten_checks(self):
        """PreflightVerifier must report PASSED with 10/10 checks."""
        verifier = PreflightVerifier()
        res = verifier.run_all_checks()

        self.assertEqual(res['status'], 'PASSED')
        self.assertEqual(res['checks_total'], 10)
        self.assertEqual(res['checks_passed'], 10)

        # Verify key individual checks
        checks = res['checks']
        self.assertTrue(checks['check_1_protocol_and_dataset_hashes']['passed'])
        self.assertTrue(checks['check_2_common_cohort_intersection']['passed'])
        self.assertTrue(checks['check_3_split_boundaries_and_no_leakage']['passed'])
        self.assertTrue(checks['check_4_identical_target_labels']['passed'])
        self.assertTrue(checks['check_5_training_only_normalization']['passed'])
        self.assertTrue(checks['check_6_policy_cohorts']['passed'])
        self.assertTrue(checks['check_7_client_eligibility']['passed'])
        self.assertTrue(checks['check_8_units_in_watts']['passed'])
        self.assertTrue(checks['check_9_validation_only_checkpoint_selection']['passed'])
        self.assertTrue(checks['check_10_test_set_isolation']['passed'])


class TestResourceEstimatorAndPilot(unittest.TestCase):
    """Verify resource benchmarking and pilot execution."""

    def test_resource_estimator_projection(self):
        """ResourceEstimator must project runtime across 18 fits."""
        estimator = ResourceEstimator(device=torch.device('cpu'))
        proj = estimator.estimate_full_matrix()

        self.assertEqual(proj['fits_total'], 18)
        self.assertIn('estimated_total_runtime_hours', proj)
        self.assertGreater(proj['estimated_total_runtime_hours'], 0)
        self.assertIn('estimated_peak_gpu_memory_mb', proj)
        self.assertIn('estimated_peak_ram_gb', proj)
        self.assertIn('estimated_disk_storage_gb', proj)

    def test_run_synthetic_pilot_benchmark(self):
        """run_synthetic_pilot_benchmark must execute cleanly on CPU or CUDA."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pilot_res = run_synthetic_pilot_benchmark(tmp_path, device=torch.device('cpu'))

            self.assertEqual(pilot_res['status'], 'pilot_benchmark_complete')
            self.assertEqual(len(pilot_res['pilot_fits']), 3)
            for pf in pilot_res['pilot_fits']:
                self.assertEqual(pf['status'], 'PILOT_PASSED')
                self.assertEqual(pf['rounds'], 2)

    def test_preflight_report_generation(self):
        """write_t1_3_preflight_report must write formatted markdown report."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            verifier = PreflightVerifier()
            preflight_res = verifier.run_all_checks()
            estimator = ResourceEstimator(device=torch.device('cpu'))
            resource_res = estimator.estimate_full_matrix()
            pilot_res = run_synthetic_pilot_benchmark(tmp_path, device=torch.device('cpu'))

            report_file = write_t1_3_preflight_report(tmp_path, preflight_res, resource_res, pilot_res)
            self.assertTrue(report_file.is_file())
            content = report_file.read_text(encoding='utf-8')

            self.assertIn('TASK T1.3 — Resource & Runtime Preflight Report', content)
            self.assertIn('10-Point Preflight Verification', content)
            self.assertIn('Estimated Runtime per Fit and Total Matrix', content)
            self.assertIn('Execution Gating and Operational Recommendation', content)


class TestRealPilotReporting(unittest.TestCase):
    """Verify that RealPilotRunner reporting and CSV writing functions operate cleanly."""

    def test_report_and_csv_generation(self):
        from src.history_length_real_pilot_v1 import RealPilotRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runner = RealPilotRunner(data=None, output_dir=tmp_path, seed=42, rounds=20, device=torch.device('cpu'))

            mock_results = {
                '6_steps': {
                    'history_steps': 6,
                    'history_hours': 1,
                    'best_round': 15,
                    'best_val_rmse_w': 18.5,
                    'test_mae_w': 14.2,
                    'test_rmse_w': 19.1,
                    'test_persistence_mae_w': 15.0,
                    'test_persistence_rmse_w': 20.0,
                    'delta_mae_w': -0.8,
                    'delta_rmse_w': -0.9,
                    'wall_seconds': 45.2,
                    'peak_vram_mb': 512.0,
                    'peak_rss_mb': 1024.0,
                    'validation_records': [{'round': 5, 'val_mae_w': 16.0, 'val_rmse_w': 21.0}, {'round': 15, 'val_mae_w': 14.5, 'val_rmse_w': 18.5}],
                    'nodewise_summary': {'evaluated_nodes': 979, 'nodes_outperforming_persistence': 600, 'improved_fraction': 0.613},
                },
                '36_steps': {
                    'history_steps': 36,
                    'history_hours': 6,
                    'best_round': 20,
                    'best_val_rmse_w': 17.5,
                    'test_mae_w': 13.8,
                    'test_rmse_w': 18.2,
                    'test_persistence_mae_w': 15.0,
                    'test_persistence_rmse_w': 20.0,
                    'delta_mae_w': -1.2,
                    'delta_rmse_w': -1.8,
                    'wall_seconds': 52.1,
                    'peak_vram_mb': 520.0,
                    'peak_rss_mb': 1040.0,
                    'validation_records': [{'round': 5, 'val_mae_w': 15.5, 'val_rmse_w': 20.0}, {'round': 20, 'val_mae_w': 13.9, 'val_rmse_w': 17.5}],
                    'nodewise_summary': {'evaluated_nodes': 979, 'nodes_outperforming_persistence': 650, 'improved_fraction': 0.664},
                },
                '144_steps': {
                    'history_steps': 144,
                    'history_hours': 24,
                    'best_round': 20,
                    'best_val_rmse_w': 17.0,
                    'test_mae_w': 13.5,
                    'test_rmse_w': 17.8,
                    'test_persistence_mae_w': 15.0,
                    'test_persistence_rmse_w': 20.0,
                    'delta_mae_w': -1.5,
                    'delta_rmse_w': -2.2,
                    'wall_seconds': 75.4,
                    'peak_vram_mb': 540.0,
                    'peak_rss_mb': 1060.0,
                    'validation_records': [{'round': 5, 'val_mae_w': 15.0, 'val_rmse_w': 19.5}, {'round': 20, 'val_mae_w': 13.5, 'val_rmse_w': 17.0}],
                    'nodewise_summary': {'evaluated_nodes': 979, 'nodes_outperforming_persistence': 680, 'improved_fraction': 0.695},
                },
            }

            csv_file = runner._write_pilot_csv(mock_results)
            self.assertTrue(csv_file.is_file())
            csv_content = csv_file.read_text(encoding='utf-8')
            self.assertIn('history_hours', csv_content)
            self.assertIn('delta_mae_w', csv_content)

            report_file = runner._write_pilot_report(mock_results)
            self.assertTrue(report_file.is_file())
            rep_content = report_file.read_text(encoding='utf-8')
            self.assertIn('TASK T1.3 — Real-Data Pilot Report', rep_content)
            self.assertIn('Pooled Metrics', rep_content)
            self.assertIn('Macro-Node Metrics', rep_content)
            self.assertIn('Validation Convergence Curves', rep_content)

    def test_progress_log_format_and_writing(self):
        from src.history_length_real_pilot_v1 import RealPilotRunner

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runner = RealPilotRunner(data=None, output_dir=tmp_path, seed=42, rounds=200, device=torch.device('cpu'))

            test_line = "[1h] Round 001/200 | Train Loss: 0.0842 | Val RMSE: 122.45 W | Elapsed: 4.8s"
            runner._log_progress(test_line)

            self.assertTrue(runner.progress_log_path.is_file())
            content = runner.progress_log_path.read_text(encoding='utf-8')
            self.assertIn("[1h] Round 001/200 | Train Loss: 0.0842 | Val RMSE: 122.45 W | Elapsed: 4.8s", content)

            runner._log_round_csv(
                history_steps=6,
                round_idx=1,
                train_loss=0.0842,
                val_rmse=122.45,
                val_mae=95.12,
                is_best_val=True,
                elapsed_sec=4.8,
            )

            self.assertTrue(runner.progress_csv_path.is_file())
            csv_content = runner.progress_csv_path.read_text(encoding='utf-8')
            self.assertIn('history_hours,history_steps,round,total_rounds,seed,policy,train_loss,val_rmse_w,val_mae_w,is_best_val,elapsed_sec,timestamp_utc', csv_content)
            self.assertIn('1,6,1,200,42,ff30,0.0842,122.45,95.12,1,4.8,', csv_content)


if __name__ == '__main__':
    unittest.main()
