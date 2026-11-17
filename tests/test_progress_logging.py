"""
Unit tests for the mandatory terminal progress logging module.
"""
from __future__ import annotations

import csv
import io
import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.progress_logging import (
    DualLogger,
    EvaluationProgressLogger,
    ExperimentStageTracker,
    FedAvgProgressLogger,
    TerminalProgressBar,
    calculate_eta,
    format_time,
)


class TestFormattingAndTiming(unittest.TestCase):
    def test_format_time_standard(self):
        self.assertEqual(format_time(0), "00:00:00")
        self.assertEqual(format_time(12.4), "00:00:12")
        self.assertEqual(format_time(65), "00:01:05")
        self.assertEqual(format_time(3600), "01:00:00")
        self.assertEqual(format_time(2468), "00:41:08")
        self.assertEqual(format_time(86400), "24:00:00")

    def test_format_time_edge_cases(self):
        self.assertEqual(format_time(-10), "00:00:00")
        self.assertEqual(format_time(float("nan")), "00:00:00")
        self.assertEqual(format_time(float("inf")), "00:00:00")
        self.assertEqual(format_time(None), "00:00:00")

    def test_calculate_eta(self):
        # 10 completed out of 100 in 100s -> 90 steps left * 10s/step = 900s
        eta = calculate_eta(elapsed_sec=100.0, completed_steps=10, total_steps=100)
        self.assertAlmostEqual(eta, 900.0, places=1)

        # Finished or exceeded steps -> 0s
        self.assertEqual(calculate_eta(elapsed_sec=50.0, completed_steps=100, total_steps=100), 0.0)
        self.assertEqual(calculate_eta(elapsed_sec=50.0, completed_steps=105, total_steps=100), 0.0)

        # 0 steps completed -> 0s
        self.assertEqual(calculate_eta(elapsed_sec=50.0, completed_steps=0, total_steps=100), 0.0)

        # Rolling window ETA: recent durations [5.0, 5.0, 5.0] -> avg 5.0s/step
        # Even if total elapsed was high earlier, ETA reflects current pace
        eta_recent = calculate_eta(
            elapsed_sec=1000.0,
            completed_steps=20,
            total_steps=30,
            recent_durations=[5.0, 5.0, 5.0],
            window_size=3,
        )
        self.assertAlmostEqual(eta_recent, 50.0, places=1)  # 10 steps * 5s


class TestTerminalProgressBar(unittest.TestCase):
    def test_iterator_progress_bar(self):
        items = list(range(5))
        collected = []
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            bar = TerminalProgressBar(items, desc="Testing", min_interval_sec=0.0)
            for item in bar:
                collected.append(item)

        self.assertEqual(collected, [0, 1, 2, 3, 4])
        output = captured_stdout.getvalue()
        self.assertIn("Testing:", output)
        self.assertIn("5/5", output)
        self.assertIn("100.0%", output)

    def test_manual_progress_bar(self):
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            with TerminalProgressBar(total=10, desc="Manual", min_interval_sec=0.0) as bar:
                bar.update(5)
                bar.update(5)

        output = captured_stdout.getvalue()
        self.assertIn("Manual:", output)
        self.assertIn("10/10", output)


class TestDualLogger(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_logger_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_dual_logger_writes_both(self):
        log_file = self.test_dir / "test.log"
        captured_stdout = io.StringIO()

        with patch("sys.stdout", captured_stdout):
            with DualLogger(log_file) as logger:
                logger.write_line("Test message line 1")
                logger.write_line("Test message line 2")

        stdout_text = captured_stdout.getvalue()
        self.assertIn("Test message line 1", stdout_text)
        self.assertIn("Test message line 2", stdout_text)

        self.assertTrue(log_file.exists())
        file_text = log_file.read_text(encoding="utf-8")
        self.assertIn("Test message line 1", file_text)
        self.assertIn("Test message line 2", file_text)


class TestFedAvgProgressLogger(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_fedavg_logger_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fedavg_progress_logger_exact_format(self):
        captured_stdout = io.StringIO()

        with patch("sys.stdout", captured_stdout):
            logger = FedAvgProgressLogger(
                experiment_name="T5",
                config_name="Temp+Humidity",
                seed=42,
                total_rounds=200,
                total_clients=98,
                device="cuda:1",
                output_dir=self.test_dir,
                interactive=False,
            )
            logger.start_training()
            logger.start_round(1)

            # Simulate client progress
            for c in range(1, 99):
                logger.update_client_progress(client_idx=c, epoch=1, total_epochs=1, loss=0.18342)

            # Complete round
            logger.complete_round(
                train_loss=0.18342,
                val_mae=31.82,
                val_rmse=58.10,
                total_epochs=1,
            )
            logger.finish_training()

        stdout_text = captured_stdout.getvalue()
        # Verify exact required tokens
        self.assertIn("[T5 | Temp+Humidity | Seed 42]", stdout_text)
        self.assertIn("Round 001/200 | Clients 098/098", stdout_text)
        self.assertIn("Local Epoch 1/1 | Loss: 0.18342", stdout_text)
        self.assertIn("Round completed | Val MAE: 31.82 W", stdout_text)
        self.assertIn("Elapsed:", stdout_text)
        self.assertIn("ETA:", stdout_text)

        # Verify progress.log
        log_file = self.test_dir / "progress.log"
        self.assertTrue(log_file.exists())
        log_content = log_file.read_text(encoding="utf-8")
        self.assertIn("[T5 | Temp+Humidity | Seed 42]", log_content)
        self.assertIn("Val MAE: 31.82 W", log_content)

        # Verify round_progress.csv
        csv_file = self.test_dir / "round_progress.csv"
        self.assertTrue(csv_file.exists())
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 1)
            row = reader[0]
            self.assertEqual(row["round"], "1")
            self.assertEqual(row["total_rounds"], "200")
            self.assertEqual(row["clients_sampled"], "98")
            self.assertAlmostEqual(float(row["train_loss"]), 0.18342, places=4)
            self.assertAlmostEqual(float(row["val_mae_w"]), 31.82, places=2)
            self.assertEqual(row["is_best"], "1")


class TestEvaluationAndStageLogger(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_eval_logger_"))

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_evaluation_progress_logger(self):
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            eval_logger = EvaluationProgressLogger(
                dataset_name="Canonical T1 Test",
                total_windows=243138,
                total_nodes=978,
                model_name="FedAvg-GRU-T4",
                output_dir=self.test_dir,
                interactive=False,
            )
            eval_logger.start_evaluation()
            eval_logger.update_node(1, 248)
            eval_logger.update_node(2, 248)
            eval_logger.complete_evaluation(
                mae=24.81,
                rmse=57.55,
                macro_mae=25.22,
                macro_rmse=52.60,
                nodes_beating_pers=898,
            )

        output = captured_stdout.getvalue()
        self.assertIn("Canonical T1 Test", output)
        self.assertIn("Pooled MAE: 24.81 W", output)
        self.assertIn("Pooled RMSE: 57.55 W", output)
        self.assertIn("Macro MAE: 25.22 W", output)
        self.assertIn("Beating Pers: 898/978", output)

    def test_experiment_stage_tracker(self):
        captured_stdout = io.StringIO()
        with patch("sys.stdout", captured_stdout):
            tracker = ExperimentStageTracker(
                suite_name="Reproducibility Benchmark",
                total_experiments=3,
                output_dir=self.test_dir,
            )
            tracker.set_experiment(1, "Config_A", 42)
            tracker.set_stage("PREPROCESSING", "Generating causal splits")
            tracker.finish_stage()
            tracker.set_stage("TRAINING", "200 rounds FedAvg")
            tracker.log_warning("High VRAM usage detected on GPU 1")
            tracker.finish_stage()
            tracker.finish_suite()

        output = captured_stdout.getvalue()
        self.assertIn("[Experiment 1/3]", output)
        self.assertIn("STAGE: PREPROCESSING", output)
        self.assertIn("STAGE: TRAINING", output)
        self.assertIn("[WARNING] High VRAM usage detected", output)
        self.assertIn("SUITE COMPLETED", output)


if __name__ == "__main__":
    unittest.main()
