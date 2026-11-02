"""Research-contract regressions using synthetic arrays, never the real held-out set."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments
from src.experiment_data import FrozenData, FEATURES, immutable_json, weather_view, sha256
from src.experiment_models import lag_features, lag_mapping, select_if_better, fit_neural, fit_xgboost, predict_selected
from src.experiment_eval import evaluate
from src.forecast_plot import gap_series


def fixture(path):
    path.mkdir()
    rng = np.random.default_rng(7)
    names = np.array(['1', '2'])
    arrays = {'node_names': names, 'power_baseline_coefficients': np.array([1., 0.])}
    for split, day in (('train', '2022-02-10'), ('val', '2022-02-12'), ('test', '2022-02-21')):
        x = rng.normal(size=(12, 6, 14)).astype(np.float32)
        x[:, :, 13] = np.repeat([0, 1], 6)[:, None]
        y = (x[:, -1, :1] + 0.1 * x[:, -1, 1:2]).astype(np.float32)
        origins = np.tile(np.datetime64(day, 'ns') + np.arange(6) * np.timedelta64(10, 'm'), 2)
        arrays.update({f'X_{split}': x, f'y_{split}': y, f'origin_{split}': origins,
                       f'node_{split}': np.repeat(names, 6)})
    inventory = {}
    for name, value in arrays.items():
        np.save(path / f'{name}.npy', value, allow_pickle=False)
        inventory[f'{name}.npy'] = {'sha256': sha256(path / f'{name}.npy'), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    meta = {'feature_names': FEATURES, 'nodes': names.tolist(), 'x_mean': [0.5] * 14,
            'x_std': [0.2] * 14, 'y_mean': [0.5], 'y_std': [0.2],
            'settings': {'resample_minutes': 10, 'horizon_minutes': 60},
            'split_boundaries_utc': {'train_end_val_start': '2022-02-11T07:10:00+00:00',
                                     'val_end_test_start': '2022-02-20T03:30:00+00:00'}}
    immutable_json(path / 'preprocessing.json', meta)
    immutable_json(path / 'contract.json', {'arrays': inventory, 'metadata_sha256': sha256(path / 'preprocessing.json')})
    return FrozenData(path)


def config(model):
    value = experiments.read_json(ROOT / f'configs/experiments/v1/pilot_{model if model == "xgboost" else "gru"}.json')
    value.update(model=model, device='cpu', train_limit=None)
    if model == 'xgboost':
        value.update(trees=3, patience=3)
    else:
        value.update(hidden_size=4, epochs=2, batch_size=4)
    return value


class ContractTests(unittest.TestCase):
    def test_mask_preserves_original_other_channels_and_shape(self):
        x = np.random.default_rng(1).normal(size=(3, 6, 14)).astype(np.float32)
        before = x.copy()
        masked = weather_view(x, 'masked')
        self.assertEqual(masked.shape, x.shape)
        np.testing.assert_array_equal(masked[:, :, [7, 8]], 0)
        np.testing.assert_array_equal(masked[:, :, [i for i in range(14) if i not in (7, 8)]], x[:, :, [i for i in range(14) if i not in (7, 8)]])
        np.testing.assert_array_equal(x, before)
        self.assertIs(weather_view(x, 'enabled'), x)
        with self.assertRaises(ValueError): weather_view(x, 'off')

    def test_manifest_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'manifest.json'
            immutable_json(p, {'a': 1})
            old = p.read_bytes()
            with self.assertRaises(FileExistsError): immutable_json(p, {'a': 2})
            self.assertEqual(old, p.read_bytes())

    def test_frozen_integrity_detects_modified_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / 'frozen')
            data.verify()
            np.save(data.path / 'node_train.npy', np.full(12, 'changed'))
            with self.assertRaisesRegex(ValueError, 'Frozen array changed'): data.verify()

    def test_watts_baseline_alignment_and_gap_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / 'frozen')
            x, y, origins, nodes = data.split('val')
            prediction = np.asarray(y[:, 0]) + 0.1
            with self.assertRaisesRegex(ValueError, 'window IDs'):
                evaluate(data, 'val', prediction, nodes[::-1], origins, Path(temp) / 'bad')
            metrics = evaluate(data, 'val', prediction, nodes, origins, Path(temp) / 'result')
            self.assertAlmostEqual(metrics['mae_w'], 20, places=4)
            expected = ((x[:, -1, 0].astype(float) * .2 + .5) - (y[:, 0].astype(float) * .2 + .5)) * 1000
            self.assertAlmostEqual(metrics['persistence_rmse_w'], float(np.sqrt(np.mean(expected ** 2))))
            self.assertEqual(metrics['evaluated_nodes'], 2)
            self.assertEqual(json.loads((Path(temp) / 'result/coverage.json').read_text())['unique_origins'], 6)

    def test_lag_order_categorical_identity_and_mask(self):
        x = np.arange(2 * 6 * 14, dtype=np.float32).reshape(2, 6, 14)
        columns = lag_features(x, 'enabled')
        masked = lag_features(x, 'masked')
        self.assertEqual(columns.shape, (2, 79))
        for step in range(6):
            np.testing.assert_array_equal(columns[:, step * 13:(step + 1) * 13], x[:, step, :13])
        np.testing.assert_array_equal(columns[:, 78], x[:, -1, 13])
        indices = [step * 13 + j for step in range(6) for j in (7, 8)]
        np.testing.assert_array_equal(masked[:, indices], 0)
        np.testing.assert_array_equal(masked[:, [j for j in range(79) if j not in indices]], columns[:, [j for j in range(79) if j not in indices]])
        self.assertEqual(lag_mapping()[-1]['type'], 'c')
        self.assertEqual(lag_mapping()[0]['minutes_before_origin'], 60)
        self.assertEqual(lag_mapping()[65]['minutes_before_origin'], 10)

    def test_strict_validation_selection_keeps_baseline_and_earliest_tie(self):
        self.assertEqual(select_if_better(10, 10, 0, 1), (10, 0, False))
        self.assertEqual(select_if_better(9, 10, 0, 1), (9, 1, True))
        self.assertEqual(select_if_better(11, 9, 1, 2), (9, 1, False))

    def test_gap_series_real_timestamps_sorts_and_does_not_join_gaps(self):
        times = np.array(['2022-02-21T00:30', '2022-02-21T00:00', '2022-02-21T00:10'], dtype='datetime64[ns]')
        t, y = gap_series(times, [3, 1, 2], 10)
        np.testing.assert_array_equal(t, np.array(['2022-02-21T00:00', '2022-02-21T00:10', '2022-02-21T00:20', '2022-02-21T00:30'], dtype='datetime64[ns]'))
        self.assertTrue(np.isnan(y[2]))
        np.testing.assert_array_equal(y[[0, 1, 3]], [1, 2, 3])

    def test_unknown_config_and_comparison_subset_rejected(self):
        c = config('gru')
        experiments.validate_config(c)
        c['kind'] = 'comparison'
        with self.assertRaises(ValueError): experiments.validate_config(c)
        c = config('gru'); c['include_weather'] = False
        with self.assertRaises(ValueError): experiments.validate_config(c)

    def test_matrix_has_exactly_eighteen_unique_full_fits(self):
        entries = experiments.plan_entries(ROOT / 'configs/experiments/v1/matrix.json', Path('/tmp/unrun-matrix'))
        self.assertEqual(len(entries), 18)
        self.assertTrue(all(c['kind'] == 'comparison' and 'train_limit' not in c for _, c, _ in entries))


class AdapterTests(unittest.TestCase):
    def test_neural_both_families_and_masks_reload_without_test_access(self):
        for model in ('lstm', 'gru'):
            for weather in ('enabled', 'masked'):
                with self.subTest(model=model, weather=weather), tempfile.TemporaryDirectory() as temp:
                    data = fixture(Path(temp) / 'frozen')
                    original = data.split
                    def restricted(split):
                        if split == 'test': raise AssertionError('Training accessed held-out labels')
                        return original(split)
                    data.split = restricted
                    output = Path(temp) / 'fit'; output.mkdir()
                    c = config(model); c['weather'] = weather
                    result, prediction = fit_neural(data, c, output)
                    restored = predict_selected(data, c, output, result, 'val')
                    np.testing.assert_allclose(restored, prediction)
                    self.assertLessEqual(result['best_validation_rmse_w'], result['persistence_validation_rmse_w'])
                    history = json.loads((output / 'history.json').read_text())
                    scores = [h['validation_rmse_w'] for h in history]
                    self.assertEqual(result['selected_iteration'], int(np.argmin(scores)))

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'Optional XGBoost dependency not installed')
    def test_xgboost_both_masks_categorical_reload_and_no_test_access(self):
        for weather in ('enabled', 'masked'):
            with self.subTest(weather=weather), tempfile.TemporaryDirectory() as temp:
                data = fixture(Path(temp) / 'frozen')
                original = data.split
                def restricted(split):
                    if split == 'test': raise AssertionError('Training accessed held-out labels')
                    return original(split)
                data.split = restricted
                output = Path(temp) / 'fit'; output.mkdir()
                c = config('xgboost'); c['weather'] = weather
                result, prediction = fit_xgboost(data, c, output)
                restored = predict_selected(data, c, output, result, 'val')
                np.testing.assert_allclose(restored, prediction, atol=1e-6)
                import xgboost as xgb
                booster = xgb.Booster(model_file=output / 'selected.ubj')
                self.assertEqual(booster.feature_types[-1], 'c')
                self.assertLessEqual(result['best_validation_rmse_w'], result['persistence_validation_rmse_w'])
                scores = json.loads((output / 'history.json').read_text())
                all_scores = [scores['epoch_0_persistence_rmse_w']] + scores['validation']['rmse_w']
                self.assertEqual(result['selected_iteration'], int(np.argmin(all_scores)))

    def test_pilot_test_reporting_is_prohibited(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with mock.patch('experiments.check_provenance', return_value=({'config': {'kind': 'pilot'}}, {}, object())), mock.patch('experiments.predict_selected') as predict:
                with self.assertRaisesRegex(ValueError, 'Pilot runs cannot'): experiments.report(output)
                predict.assert_not_called()

    def test_full_fit_manifest_validation_and_separate_synthetic_test_report(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / 'frozen')
            c = config('gru'); c['kind'] = 'comparison'; c.pop('train_limit')
            path = Path(temp) / 'config.json'; path.write_text(json.dumps(c))
            output = Path(temp) / 'fit'
            experiments.fit(path, data.path, output)
            self.assertFalse((output / 'test').exists())
            self.assertFalse(json.loads((output / 'fit_completed.json').read_text())['test_evaluated'])
            manifest, selection, checked = experiments.check_provenance(output)
            self.assertEqual(checked.contract_sha256, data.contract_sha256)
            self.assertEqual(selection['common_evaluator_verified_on'], 'validation only')
            self.assertTrue((output / 'source/experiments.py').is_file())
            experiments.report(output)
            self.assertEqual(json.loads((output / 'test/metrics.json').read_text())['split'], 'test')
            with self.assertRaises(FileExistsError): experiments.report(output)
            with mock.patch('experiments.source_hashes', return_value={'changed': 'source'}):
                with self.assertRaisesRegex(ValueError, 'Execution source changed'): experiments.check_provenance(output)

    def test_matrix_report_requires_completed_validation_selections(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / 'frozen')
            with mock.patch('experiments.report') as report:
                with self.assertRaisesRegex(ValueError, 'Fit the matrix'):
                    experiments.matrix(ROOT / 'configs/experiments/v1/matrix.json', data.path, Path(temp) / 'matrix', 'report')
                report.assert_not_called()

    def test_fit_refuses_existing_outputs_without_replacing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp) / 'frozen')
            c = config('gru'); path = Path(temp) / 'config.json'; path.write_text(json.dumps(c))
            output = Path(temp) / 'fit'; output.mkdir(); (output / 'old.pt').write_bytes(b'keep')
            with self.assertRaises(FileExistsError): experiments.fit(path, data.path, output)
            self.assertEqual((output / 'old.pt').read_bytes(), b'keep')


if __name__ == '__main__':
    unittest.main()
