"""Small actual Parquet fixtures; no network or real telemetry is needed."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HAS_ARROW = importlib.util.find_spec("pyarrow") is not None
if HAS_ARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.preprocess import METRICS, NS_SECOND, _time_weighted_bins, _weather_asof, prepare_m100


@unittest.skipUnless(HAS_ARROW, "Install requirements.txt to run real Parquet fixture tests")
class PreprocessingTests(unittest.TestCase):
    def settings(self):
        return {"node": "0", "months": ["22-01"], "resample_minutes": 10,
                "history_hours": 1, "horizon_minutes": 20, "train_fraction": .70,
                "val_fraction": .15, "min_bin_coverage": .5,
                "weather_max_age_minutes": 30, "max_hold_seconds": 300}

    def fixture(self, root, *, gap_metric=None, duplicate_power=False, integer_nodes=False):
        base = datetime(2022, 1, 1, tzinfo=timezone.utc)
        for plugin, metric in METRICS:
            weather = plugin == "weather"
            minutes = np.arange(0, 1200, 10 if weather else 1)
            bins = minutes // 10
            if metric == "total_power":
                values = 1000 + bins * 100
            elif metric == "ambient":
                values = 23 + bins * .01
            elif metric == "cpu_user":
                values = 20 + bins * .1
            elif metric == "cpu_system":
                values = np.full(len(minutes), 10.)
            elif metric == "cpu_idle":
                values = np.full(len(minutes), 65.)
            elif metric == "cpu_wio":
                values = np.full(len(minutes), 5.)
            elif metric == "mem_free":
                values = np.full(len(minutes), 500.)
            elif metric == "mem_total":
                values = np.full(len(minutes), 1000.)
            elif metric.startswith("Gpu"):
                values = np.full(len(minutes), 10. * (int(metric[3]) + 1))
            elif metric == "temp":
                values = 10 + bins * .05
            else:
                values = np.full(len(minutes), 50.)
            if metric == gap_metric:
                # Twenty missing input bins in the middle; a 5-minute hold must
                # not fill a 200-minute source gap or compress the timeline.
                keep = (minutes < 400) | (minutes >= 600)
                minutes, values = minutes[keep], values[keep]
            timestamp = [base + timedelta(minutes=int(minute)) for minute in minutes]
            if duplicate_power and metric == "total_power":
                timestamp.append(base + timedelta(minutes=50))
                values = np.r_[values, 4321.]
            arrays = {"timestamp": pa.array(timestamp, type=pa.timestamp("ms", tz="UTC")),
                      "value": pa.array(values, type=pa.float64())}
            if not weather:
                # All fixtures include an unwanted node; it must not influence
                # audit, derived features, normalizers or targets.
                original_count = len(timestamp)
                arrays["timestamp"] = pa.concat_arrays([arrays["timestamp"], arrays["timestamp"]])
                arrays["value"] = pa.concat_arrays([arrays["value"], pa.array(np.full(original_count, 1e9))])
                node_type = pa.int32() if integer_nodes else pa.string()
                node_values = [0, 1] if integer_nodes else ["0", "1"]
                arrays["node"] = pa.array([node_values[0]] * original_count + [node_values[1]] * original_count,
                                          type=node_type).dictionary_encode()
            path = root / "enclosing" / "year_month=22-01" / f"plugin={plugin}_pub" / f"metric={metric}"
            path.mkdir(parents=True)
            pq.write_table(pa.table(arrays), path / "part-0.parquet")

    def test_windows_are_causal_and_scaling_uses_training_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            self.fixture(root, integer_nodes=True)
            result = prepare_m100(root, Path(temporary) / "output", self.settings())
            metadata = json.loads(Path(result["preprocessing_path"]).read_text())
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(metadata["input_size"], 10)
            self.assertEqual(metadata["sequence_length"], 6)
            self.assertEqual(audit["grid_bin_count"], 120)
            self.assertEqual(audit["metrics"]["total_power"]["selected_row_count_before_dedup"], 1200)
            with np.load(result["data_path"], allow_pickle=False) as data:
                self.assertEqual(data["X_train"].shape[1:], (6, 10))
                self.assertEqual(data["y_train"].shape[1:], (1,))
                raw_x = data["X_train"] * np.asarray(metadata["x_std"]) + np.asarray(metadata["x_mean"])
                raw_y = data["y_train"] * np.asarray(metadata["y_std"]) + np.asarray(metadata["y_mean"])
                # First origin = 01:00. Inputs are bins 0..5; target bins 6..7.
                np.testing.assert_allclose(raw_x[0, :, 0], 1 + np.arange(6) * .1, atol=1e-6)
                np.testing.assert_allclose(raw_y[0, 0], 1.65, atol=1e-6)
                self.assertEqual(data["origin_train"][0], "2022-01-01T01:00:00+00:00")
                # Train target intervals end before 14:00; val/test means cannot
                # enter this normalizer even though adjacent histories overlap.
                expected_x = np.stack([1 + np.arange(origin - 6, origin) * .1 for origin in range(6, 83)])
                self.assertAlmostEqual(metadata["x_mean"][0], expected_x.mean())
                self.assertAlmostEqual(metadata["y_mean"][0], np.mean([1 + (origin + .5) * .1 for origin in range(6, 83)]))
                self.assertLess(metadata["x_mean"][0], np.mean(1 + np.arange(120) * .1))
                self.assertEqual(data["origin_val"][0], "2022-01-01T14:00:00+00:00")
                self.assertEqual(data["origin_test"][0], "2022-01-01T17:00:00+00:00")
                self.assertTrue(all(data[key].dtype == np.float32 for key in data.files if key.startswith(("X_", "y_"))))

    def test_missing_source_gap_keeps_grid_and_invalidates_crossing_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            self.fixture(root, gap_metric="cpu_user", duplicate_power=True)
            result = prepare_m100(root, Path(temporary) / "output", self.settings())
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(audit["grid_bin_count"], 120)
            self.assertEqual(audit["metrics"]["total_power"]["duplicate_timestamp_count"], 1)
            self.assertGreater(audit["dropped_windows"]["history_incomplete"], 0)
            with np.load(result["data_path"], allow_pickle=False) as data:
                minutes = [int((datetime.fromisoformat(origin) - datetime(2022, 1, 1, tzinfo=timezone.utc)).total_seconds() / 60)
                           for origin in data["origin_train"]]
                self.assertFalse(any(410 <= minute <= 650 for minute in minutes))
                self.assertTrue(any(minute >= 660 for minute in minutes))

    def test_time_weighted_hold_and_weather_never_look_forward(self):
        second = NS_SECOND
        edges = np.array([0, 600, 1200, 1800], dtype=np.int64) * second
        # Only 100s at 10 and 300s at 20 are covered in bin 0; uncovered time
        # is not filled with zero and a 1400-second source gap is not bridged.
        times = np.array([0, 100, 1500], dtype=np.int64) * second
        values = np.array([10., 20., 40.])
        binned, coverage = _time_weighted_bins(times, values, edges, 300 * second, .5)
        self.assertAlmostEqual(binned[0], 17.5)
        self.assertAlmostEqual(coverage[0], 2 / 3)
        self.assertTrue(np.isnan(binned[1]))
        self.assertEqual(binned[2], 40.)
        weather = _weather_asof(np.array([600, 1200], dtype=np.int64) * second,
                                np.array([5., 9.]), edges, 600 * second)
        self.assertTrue(np.isnan(weather[0]))  # reading at 600 belongs to next bin
        self.assertEqual(weather[1], 5.)
        self.assertEqual(weather[2], 9.)

    def test_missing_metric_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Required metric partitions were not found"):
                prepare_m100(root, root / "output", self.settings())

    def test_gpu_omission_prepares_nine_features_without_gpu_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            self.fixture(root)
            for path in root.rglob("*.parquet"):
                if path.parent.name.startswith("metric=Gpu"):
                    path.unlink()
            settings = dict(self.settings(), include_gpu_utilization=False)
            result = prepare_m100(root, Path(temporary) / "output", settings)
            metadata = json.loads(Path(result["preprocessing_path"]).read_text())
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(metadata["input_size"], 9)
            self.assertEqual(len(metadata["feature_units"]), 9)
            self.assertFalse(metadata["include_gpu_utilization"])
            self.assertNotIn("gpu_utilization_mean", metadata["feature_names"])
            self.assertFalse(any(name.startswith("Gpu") for name in audit["metrics"]))
            with np.load(result["data_path"], allow_pickle=False) as data:
                self.assertEqual(data["X_train"].shape[1:], (6, 9))
                raw_x = data["X_train"] * np.asarray(metadata["x_std"]) + np.asarray(metadata["x_mean"])
                # Removing GPU utilization must preserve the identities/positions
                # of memory, inlet temperature, outdoor temperature and humidity.
                np.testing.assert_allclose(raw_x[0, :, 5], .5, atol=1e-6)
                np.testing.assert_allclose(raw_x[0, :, 6], 23 + np.arange(6) * .01, atol=1e-6)
                np.testing.assert_allclose(raw_x[0, :, 7], 10 + np.arange(6) * .05, atol=1e-6)
                np.testing.assert_allclose(raw_x[0, :, 8], 50, atol=1e-6)
            # GPU inclusion remains strict rather than silently substituting data.
            with self.assertRaisesRegex(ValueError, "Required metric partitions were not found"):
                prepare_m100(root, Path(temporary) / "strict", self.settings())

    def test_gpu_omission_preserves_missing_required_cpu_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            self.fixture(root)
            for path in root.rglob("*.parquet"):
                if path.parent.name == "metric=cpu_user":
                    path.unlink()
            with self.assertRaisesRegex(ValueError, "Required metric partitions were not found: cpu_user"):
                prepare_m100(root, Path(temporary) / "output",
                             dict(self.settings(), include_gpu_utilization=False))

    def test_gpu_option_requires_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in ("false", 0, None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "JSON boolean"):
                    prepare_m100(Path(temporary), Path(temporary) / "output",
                                 dict(self.settings(), include_gpu_utilization=value))

    def test_gpu_inclusion_reports_missing_rows_for_selected_node(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            self.fixture(root)
            path = next(root.rglob("metric=Gpu0_gpu_utilization/part-0.parquet"))
            table = pq.ParquetFile(path).read()
            table = table.filter(pa.compute.equal(pa.compute.cast(table["node"], pa.string()), "1"))
            pq.write_table(table, path)
            with self.assertRaisesRegex(ValueError, "include_gpu_utilization=false"):
                prepare_m100(root, Path(temporary) / "strict", self.settings())
            result = prepare_m100(root, Path(temporary) / "without_gpu",
                                  dict(self.settings(), include_gpu_utilization=False))
            self.assertGreater(result["sample_counts"]["test"], 0)


if __name__ == "__main__":
    unittest.main()
