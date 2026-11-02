"""Joint-node boundaries, identity, missingness and baseline behavior."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.data import load_data
from src.model import SequenceRegressor
from src.multinode import weighted_bins
from src.preprocess import METRICS, NS_SECOND, _time_weighted_bins, prepare_m100


class MultinodeTests(unittest.TestCase):
    def fixture(self, root):
        base = datetime(2022, 1, 1, tzinfo=timezone.utc)
        minutes = np.arange(0, 31 * 24 * 60, 5)
        stamps = pa.array([base + timedelta(minutes=int(m)) for m in minutes], type=pa.timestamp('ms', tz='UTC'))
        for plugin, metric in METRICS:
            if metric.startswith('Gpu'):
                continue
            path = root / 'year_month=22-01' / f'plugin={plugin}_pub' / f'metric={metric}'
            path.mkdir(parents=True, exist_ok=True)
            if plugin == 'weather':
                values = 15 + np.sin(minutes / 1440) if metric == 'temp' else np.full(len(minutes), 50.)
                table = pa.table({'timestamp': stamps, 'value': values})
            else:
                nodes = ['a', 'b'] if metric == 'cpu_user' else ['a', 'b', 'missing_cpu']
                arrays = []
                for index, node in enumerate(nodes):
                    if metric == 'total_power':
                        values = 1000 + 1000 * index + minutes / 50 + 100 * np.sin(minutes / 300)
                    else:
                        constant = {'ambient': 23., 'cpu_user': 25., 'cpu_system': 10., 'cpu_idle': 60.,
                                    'cpu_wio': 5., 'mem_free': 500., 'mem_total': 1000.}[metric]
                        values = np.full(len(minutes), constant)
                    if node == 'b' and metric == 'cpu_system':
                        values[(minutes >= 1000) & (minutes < 1100)] = np.nan
                    arrays.append(pa.table({'timestamp': stamps, 'value': values, 'node': [node] * len(minutes)}))
                table = pa.concat_tables(arrays)
            pq.write_table(table, path / 'part.parquet')

    def settings(self):
        return {'nodes': 'all', 'months': ['22-01'], 'include_gpu_utilization': False,
                'resample_minutes': 10, 'history_hours': 1, 'horizon_minutes': 60,
                'train_fraction': .7, 'val_fraction': .15, 'min_bin_coverage': .5,
                'max_hold_seconds': 300, 'weather_max_age_minutes': 30,
                'min_windows_per_split': 10, 'time_features': True}

    def test_fast_bins_match_causal_reference_with_gaps_nulls_and_boundaries(self):
        rng = np.random.default_rng(2)
        times = np.cumsum(rng.integers(1, 800, 500)).astype(np.int64) * NS_SECOND
        values = rng.normal(size=len(times))
        values[::7] = np.nan
        edges = np.arange(600, 100000, 600, dtype=np.int64) * NS_SECOND
        for hold in (30, 300, 600, 900):
            actual = weighted_bins(times, values, edges, hold * NS_SECOND, .5)
            expected = _time_weighted_bins(times, values, edges, hold * NS_SECOND, .5)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, equal_nan=True, atol=1e-12)

    def test_joint_data_keep_nodes_separate_common_cuts_and_train_only_scaling(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / 'raw', Path(directory) / 'out'
            self.fixture(root)
            result = prepare_m100(root, output, self.settings())
            metadata = json.loads(Path(result['preprocessing_path']).read_text())
            audit = json.loads(Path(result['audit_path']).read_text())
            self.assertEqual(metadata['nodes'], ['a', 'b'])
            self.assertEqual(metadata['input_size'], 14)
            self.assertEqual(audit['nodes']['missing_cpu']['status'], 'excluded')
            self.assertEqual(audit['nodes']['missing_cpu']['missing_metrics'], ['cpu_user'])
            with np.load(result['data_path'], allow_pickle=False) as data:
                self.assertTrue(np.all(data['X_train'][:, :, -1] == data['X_train'][:, :1, -1]))
                np.testing.assert_allclose(data['X_train'][:, :, :-1].mean(axis=(0, 1)), 0, atol=2e-5)
                cut = np.datetime64(metadata['split_boundaries_utc']['train_end_val_start'].replace('+00:00', ''))
                test_cut = np.datetime64(metadata['split_boundaries_utc']['val_end_test_start'].replace('+00:00', ''))
                self.assertTrue(np.all(data['origin_train'] + np.timedelta64(60, 'm') <= cut))
                self.assertTrue(np.all(data['origin_val'] >= cut))
                self.assertTrue(np.all(data['origin_val'] + np.timedelta64(60, 'm') <= test_cut))
                self.assertTrue(np.all(data['origin_test'] >= test_cut))
                # The source gap in b's CPU history must invalidate windows.
                gap_start = np.datetime64('2022-01-01T16:40')
                gap_end = np.datetime64('2022-01-01T19:20')
                selected = data['origin_train'][data['node_train'] == 'b']
                self.assertFalse(np.any((selected > gap_start) & (selected < gap_end)))
            _, info = load_data({}, Path(result['data_path']))
            self.assertEqual(info['node_count'], 2)
            self.assertEqual(len(info['power_baseline_coefficients']), 2)

    def test_residual_model_starts_at_persistence_and_embedding_receives_gradient(self):
        model = SequenceRegressor(4, 1, hidden_size=8, node_count=2, baseline_coefficients=[2., -.5])
        x = torch.randn(3, 6, 4)
        x[:, :, -1] = torch.tensor([0., 1., 0.])[:, None]
        np.testing.assert_allclose(model(x).detach().numpy(), (2 * x[:, -1, 0:1] - .5).numpy())
        with torch.no_grad():
            model.head.weight.fill_(.1)
        model(x).square().mean().backward()
        self.assertGreater(float(model.node_embedding.weight.grad.abs().sum()), 0)

    def test_node_indices_reject_mixed_node_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.npz'
            x = np.zeros((2, 6, 2), dtype=np.float32)
            x[0, -1, -1] = 1
            np.savez(path, X_train=x, X_val=x, y_train=np.zeros((2, 1)), y_val=np.zeros((2, 1)), node_names=np.array(['a', 'b']))
            with self.assertRaisesRegex(ValueError, 'crossing node'):
                load_data({}, path)

    def test_joint_residual_training_resume_and_physical_per_node_export(self):
        import logging
        import main
        import pipeline
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory) / 'raw', Path(directory) / 'out'
            self.fixture(root)
            result = prepare_m100(root, output, self.settings())
            config = json.loads((Path(__file__).resolve().parents[1] / 'configs/all_nodes_gpu.json').read_text())
            config.update(epochs=2, hidden_size=4, num_layers=1, device='cpu', batch_size=512)
            config_path = Path(directory) / 'train.json'
            config_path.write_text(json.dumps(config))
            args = ['--config', str(config_path), '--data-path', result['data_path'], '--output-dir', str(output)]
            self.assertEqual(main.main(args), 0)
            checkpoint = torch.load(output / 'checkpoints/last.pt', weights_only=True)
            step = checkpoint['training_state']['global_step']
            pipeline.final_exports(output, Path(result['data_path']), Path(result['preprocessing_path']), logging.getLogger('test'))
            report = json.loads((output / 'results/test_summary.json').read_text())
            by_node = json.loads((output / 'results/test_by_node.json').read_text())
            self.assertEqual(report['evaluated_node_count'], 2)
            self.assertEqual(set(by_node), {'a', 'b'})
            self.assertEqual(report['weights'], 'best validation weights')
            self.assertAlmostEqual(report['macro_node_mae_kw'], np.mean([row['mae_kw'] for row in by_node.values()]))
            with np.load(output / 'results/test_predictions.npz', allow_pickle=False) as data:
                self.assertEqual(set(data['node']), {'a', 'b'})
                self.assertTrue(np.isfinite(data['predictions_kw']).all())
            self.assertEqual(main.main(args), 0)
            self.assertEqual(torch.load(output / 'checkpoints/last.pt', weights_only=True)['training_state']['global_step'], step)
