"""Behavioral recovery tests on small, deterministic CPU runs."""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.io_utils import RunLock


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        defaults = json.loads((ROOT / "configs/default.json").read_text())
        defaults.update(epochs=2, batch_size=8, synthetic_samples=40, sequence_length=6,
                        input_size=3, hidden_size=4, num_threads=1, checkpoint_every_steps=1,
                        device="cpu")
        self.config.write_text(json.dumps(defaults))

    def tearDown(self):
        self.temporary.cleanup()

    def run_training(self, output, *arguments, success=True):
        # In-process CLI calls keep these tests fast while using real files and optimizers.
        import main
        code = main.main(["--config", str(self.config), "--output-dir", str(output), *arguments])
        self.assertEqual(code, 0 if success else 1)

    def weights(self, output):
        return torch.load(output / "models/final_weights.pt", map_location="cpu", weights_only=True)["model_state_dict"]

    def test_mid_epoch_resume_matches_uninterrupted_training(self):
        complete, resumed = self.root / "complete", self.root / "resumed"
        self.run_training(complete)
        self.run_training(resumed, "--stop-after-step", "2")
        checkpoint = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(checkpoint["training_state"]["cursor"], 16)
        self.assertIn("optimizer_state_dict", checkpoint)
        self.run_training(resumed)
        for key, value in self.weights(complete).items():
            self.assertTrue(torch.equal(value, self.weights(resumed)[key]), key)
        self.assertEqual(json.loads((complete / "results/history.json").read_text()),
                         json.loads((resumed / "results/history.json").read_text()))

    def test_stop_on_final_batch_then_resume(self):
        output = self.root / "last_batch"
        self.run_training(output, "--stop-after-step", "4")
        state = torch.load(output / "checkpoints/last.pt", weights_only=True)["training_state"]
        self.assertEqual(state["cursor"], 32)
        self.run_training(output)
        self.assertEqual(json.loads((output / "results/summary.json").read_text())["completed_epochs"], 2)

    def test_corrupted_last_falls_back_and_both_corrupt_fail(self):
        output = self.root / "corrupt"
        self.run_training(output, "--stop-after-step", "2")
        (output / "checkpoints/last.pt").write_bytes(b"truncated checkpoint")
        self.run_training(output)
        self.assertEqual(json.loads((output / "results/summary.json").read_text())["status"], "completed")
        for name in ("last.pt", "previous.pt"):
            (output / "checkpoints" / name).write_bytes(b"invalid")
        before = (output / "models/final_weights.pt").read_bytes()
        self.run_training(output, success=False)
        self.assertEqual(before, (output / "models/final_weights.pt").read_bytes())

    def test_changed_config_rejected_without_reset_and_fresh_never_erases(self):
        output = self.root / "settings"
        self.run_training(output, "--stop-after-step", "2")
        before = (output / "checkpoints/last.pt").read_bytes()
        self.run_training(output, "--batch-size", "4", success=False)
        self.assertEqual(before, (output / "checkpoints/last.pt").read_bytes())
        self.run_training(output, "--fresh", success=False)
        self.assertEqual(before, (output / "checkpoints/last.pt").read_bytes())

    def test_readable_but_damaged_model_checkpoint_uses_previous(self):
        output = self.root / "damaged_shape"
        self.run_training(output, "--stop-after-step", "2")
        path = output / "checkpoints/last.pt"
        saved = torch.load(path, weights_only=True)
        saved["model_state_dict"]["head.weight"] = torch.zeros(1, 99)
        torch.save(saved, path)
        self.run_training(output)
        self.assertEqual(json.loads((output / "results/summary.json").read_text())["status"], "completed")

    def test_completed_restart_and_epoch_extension(self):
        output = self.root / "completed"
        self.run_training(output)
        original = self.weights(output)
        old_step = json.loads((output / "results/summary.json").read_text())["global_step"]
        self.run_training(output)
        self.assertEqual(old_step, json.loads((output / "results/summary.json").read_text())["global_step"])
        for key, value in original.items():
            self.assertTrue(torch.equal(value, self.weights(output)[key]))
        self.run_training(output, "--epochs", "3")
        self.assertEqual(json.loads((output / "results/summary.json").read_text())["completed_epochs"], 3)

    def test_single_writer_lock_and_reuse_after_release(self):
        path = self.root / "lock"
        with RunLock(path):
            with self.assertRaises(RuntimeError):
                with RunLock(path):
                    pass
        with RunLock(path):
            pass

    def test_requested_gpu_never_starts_training_when_unavailable(self):
        import main
        for cuda_build, expected_message in ((None, "CPU-only PyTorch"),
                                             ("13.0", "cannot access an NVIDIA GPU")):
            with self.subTest(cuda_build=cuda_build):
                output = self.root / ("cpu_wheel" if cuda_build is None else "driver_failure")
                errors = io.StringIO()
                with patch.object(main.torch.version, "cuda", cuda_build), \
                        patch.object(main.torch.cuda, "is_available", return_value=False), \
                        patch.object(main, "SequenceRegressor") as model, \
                        contextlib.redirect_stderr(errors):
                    self.run_training(output, "--device", "cuda", success=False)
                self.assertIn(expected_message, errors.getvalue())
                model.assert_not_called()
                self.assertFalse((output / "checkpoints/last.pt").exists())

    def test_best_validation_export_uses_selected_weights_and_resume_keeps_last(self):
        import main
        config = json.loads(self.config.read_text())
        config['prediction_weights'] = 'best'
        self.config.write_text(json.dumps(config))
        original_evaluate = main.evaluate
        losses = iter([.1, .9, .1])
        def controlled_evaluate(*args):
            predictions, metrics = original_evaluate(*args)
            metrics['mse'] = next(losses)
            return predictions, metrics
        output = self.root / 'best_export'
        with patch.object(main, 'evaluate', side_effect=controlled_evaluate):
            self.run_training(output)
        saved = torch.load(output / 'checkpoints/last.pt', weights_only=True)
        exported = self.weights(output)
        for key, value in saved['training_state']['best_weights'].items():
            self.assertTrue(torch.equal(value, exported[key]), key)
        self.assertTrue(any(not torch.equal(value, exported[key]) for key, value in saved['model_state_dict'].items()))
        report = json.loads((output / 'results/summary.json').read_text())
        self.assertEqual(report['prediction_model'], 'best validation weights')
        self.assertEqual(report['best_epoch'], 1)

    def test_early_stopping_resume_and_epoch_extension(self):
        import main
        config = json.loads(self.config.read_text())
        config.update(epochs=10, prediction_weights='best', early_stopping_patience=2)
        self.config.write_text(json.dumps(config))
        original = main.evaluate
        losses = iter([.1, .9, 1., .1])
        def controlled(*args):
            predicted, metrics = original(*args)
            metrics['mse'] = next(losses)
            return predicted, metrics
        output = self.root / 'early_stop'
        with patch.object(main, 'evaluate', side_effect=controlled):
            self.run_training(output)
        report = json.loads((output / 'results/summary.json').read_text())
        self.assertTrue(report['early_stopped'])
        self.assertEqual(report['completed_epochs'], 3)
        step = report['global_step']
        self.run_training(output)
        self.assertEqual(json.loads((output / 'results/summary.json').read_text())['global_step'], step)
        losses = iter([.08, .09, .10, .08])
        with patch.object(main, 'evaluate', side_effect=controlled):
            self.run_training(output, '--epochs', '12')
        report = json.loads((output / 'results/summary.json').read_text())
        self.assertEqual(report['completed_epochs'], 6)
        self.assertEqual(report['best_epoch'], 4)
        self.assertTrue(report['early_stopped'])


if __name__ == "__main__":
    unittest.main()
