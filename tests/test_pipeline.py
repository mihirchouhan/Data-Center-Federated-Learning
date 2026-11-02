"""Download-to-test-export integration using miniature schema-faithful telemetry."""
from __future__ import annotations

import functools
import hashlib
import http.server
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HAS_ARROW = importlib.util.find_spec("pyarrow") is not None
if HAS_ARROW:
    import test_preprocess as fixture_module


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DownloadOptionsTests(unittest.TestCase):
    def test_invalid_ca_bundle_reports_error_and_keeps_partial(self):
        import pipeline
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            partial = cache / "archives/22-01.tar.part"
            partial.parent.mkdir(parents=True)
            partial.write_bytes(b"already downloaded bytes")
            arguments = ["--config", str(ROOT / "configs/m100.json"),
                         "--data-dir", str(cache), "--output-dir", str(root / "output"),
                         "--stage", "download", "--ca-bundle", str(root / "missing-ca.pem")]
            with mock.patch("src.download.urllib.request.urlopen") as open_url, \
                    mock.patch("sys.stderr") as stderr:
                self.assertEqual(pipeline.main(arguments), 1)
            open_url.assert_not_called()
            self.assertIn("Cannot load trusted CA bundle", str(stderr.write.call_args_list))
            self.assertEqual(partial.read_bytes(), b"already downloaded bytes")


@unittest.skipUnless(HAS_ARROW, "Install requirements.txt for pipeline fixture test")
class PipelineTests(unittest.TestCase):
    def test_download_prepare_train_export_and_cached_restart(self):
        import pipeline
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            fixture = fixture_module.PreprocessingTests()
            fixture.fixture(raw)
            archive = root / "22-01.tar"
            with tarfile.open(archive, "w") as handle:
                for path in sorted(raw.rglob("*.parquet")):
                    handle.add(path, arcname=path.relative_to(raw).as_posix())
            expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
            handler = functools.partial(QuietHandler, directory=str(root))
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                training = json.loads((ROOT / "configs/default.json").read_text())
                training.update(epochs=1, hidden_size=4, batch_size=16, device="cpu")
                training_path = root / "training.json"
                training_path.write_text(json.dumps(training))
                config = {"dataset": {"record_url": "fixture only", "archives": [
                    {"month": "22-01", "url": f"http://127.0.0.1:{server.server_port}/22-01.tar", "md5": expected_md5}]},
                    "preprocessing": fixture.settings(), "training_config": str(training_path)}
                config_path = root / "pipeline.json"
                config_path.write_text(json.dumps(config))
                output = root / "output"
                cache = root / "cache"
                arguments = ["--config", str(config_path), "--data-dir", str(cache), "--output-dir", str(output)]
                self.assertEqual(pipeline.main(arguments), 0)
                required = ["prepared/dataset.npz", "prepared/preprocessing.json", "results/data_audit.json",
                            "checkpoints/last.pt", "checkpoints/best.pt", "models/final_weights.pt",
                            "results/test_summary.json", "results/test_predictions.npz", "results/loss_curve.png",
                            "results/test_forecast.png", "logs/train.log", "logs/pipeline.log", "stages/completed.json"]
                self.assertTrue(all((output / path).is_file() for path in required))
                report = json.loads((output / "results/test_summary.json").read_text())
                self.assertEqual(report["unit"], "kW")
                self.assertGreater(report["samples"], 0)
                old_step = json.loads((output / "results/summary.json").read_text())["global_step"]
                prepared_hash = hashlib.sha256((output / "prepared/dataset.npz").read_bytes()).hexdigest()
                self.assertEqual(pipeline.main(arguments), 0)
                self.assertEqual(old_step, json.loads((output / "results/summary.json").read_text())["global_step"])
                self.assertEqual(prepared_hash, hashlib.sha256((output / "prepared/dataset.npz").read_bytes()).hexdigest())
                config["preprocessing"]["node"] = "0"
                config["dataset"]["archives"][0]["md5"] = "0" * 32
                config_path.write_text(json.dumps(config))
                self.assertEqual(pipeline.main(arguments + ["--stage", "train"]), 1)
                self.assertEqual(prepared_hash, hashlib.sha256((output / "prepared/dataset.npz").read_bytes()).hexdigest())
                receipt = json.loads((output / "stages/prepare.json").read_text())
                self.assertTrue(all(not Path(path).is_absolute() for path in receipt["artifacts"]))
                # A different node must never overwrite a prepared/checkpointed experiment.
                config["dataset"]["archives"][0]["md5"] = expected_md5
                config["preprocessing"]["node"] = "1"
                config_path.write_text(json.dumps(config))
                self.assertEqual(pipeline.main(arguments), 1)
                self.assertEqual(prepared_hash, hashlib.sha256((output / "prepared/dataset.npz").read_bytes()).hexdigest())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
