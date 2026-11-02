"""M100 Dataset 5 download -> audit/prepare -> train/resume -> held-out export."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from src.io_utils import RunLock, atomic_write, write_json

ROOT = Path(__file__).resolve().parent
PIPELINE_VERSION = 1
METRICS = {("ipmi", "total_power"), ("ipmi", "ambient"),
           *(('ganglia', name) for name in ('cpu_user', 'cpu_system', 'cpu_idle', 'cpu_wio', 'mem_free', 'mem_total')),
           *(('ganglia', f'Gpu{i}_gpu_utilization') for i in range(4)),
           ("weather", "temp"), ("weather", "humidity")}


def portable_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def file_hash(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def extraction_is_complete(receipt_path, extracted, fingerprint):
    if not receipt_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return receipt["fingerprint"] == fingerprint and all(
            (extracted / item["path"]).is_file() and file_hash(extracted / item["path"]) == item["sha256"]
            for item in receipt["files"])
    except (OSError, KeyError, ValueError):
        return False


def final_exports(output, dataset_path, metadata_path, logger):
    """Held-out test evaluation in original physical units; all files stay under output."""
    import numpy as np
    import torch
    from src.model import SequenceRegressor
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads((output / "results/summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise RuntimeError("Training did not complete; restart before final test export")
    saved = torch.load(output / "models/final_weights.pt", map_location="cpu", weights_only=True)
    model = SequenceRegressor(saved["input_size"], saved["target_size"], saved["hidden_size"], saved["num_layers"], saved["model"], **saved.get("model_options", {}))
    model.load_state_dict(saved["model_state_dict"])
    device = torch.device(summary.get("device", "cpu"))
    model.to(device)
    model.eval()
    with np.load(dataset_path, allow_pickle=False) as dataset:
        x = dataset["X_test"]
        y = dataset["y_test"]
        origins = dataset["origin_test"]
        node_ids = dataset["node_test"] if "node_test" in dataset.files else np.full(len(y), str(metadata["node"]))
    predictions = []
    torch.set_num_threads(1)
    with torch.no_grad():
        batch_size = 1024 if device.type == "cuda" else 128
        for start in range(0, len(x), batch_size):
            predictions.append(model(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu().numpy())
    predicted_scaled = np.concatenate(predictions)
    mean, std = np.asarray(metadata["y_mean"]), np.asarray(metadata["y_std"])
    predicted, actual = predicted_scaled * std + mean, y * std + mean
    difference = predicted - actual
    # Operational feature zero is historical power. Persistence uses the final input bin.
    persistence = x[:, -1, 0:1] * metadata["x_std"][0] + metadata["x_mean"][0]
    baseline_difference = persistence - actual
    metrics = {"status": "completed", "split": "held_out_test", "unit": "kW", "samples": len(y),
               "model": saved["model"], "weights": saved.get("prediction_model", "final epoch"),
               "mae_kw": float(np.abs(difference).mean()), "rmse_kw": float(np.sqrt(np.square(difference).mean())),
               "persistence_mae_kw": float(np.abs(baseline_difference).mean()),
               "persistence_rmse_kw": float(np.sqrt(np.square(baseline_difference).mean()))}
    node_reports = {}
    for node in np.unique(node_ids):
        mask = node_ids == node
        error, baseline_error = difference[mask], baseline_difference[mask]
        node_reports[str(node)] = {"samples": int(mask.sum()), "mae_kw": float(np.abs(error).mean()),
                                  "rmse_kw": float(np.sqrt(np.square(error).mean())),
                                  "persistence_mae_kw": float(np.abs(baseline_error).mean()),
                                  "persistence_rmse_kw": float(np.sqrt(np.square(baseline_error).mean()))}
    metrics["evaluated_node_count"] = len(node_reports)
    metrics["macro_node_mae_kw"] = float(np.mean([item["mae_kw"] for item in node_reports.values()]))
    metrics["macro_node_persistence_mae_kw"] = float(np.mean([item["persistence_mae_kw"] for item in node_reports.values()]))
    write_json(output / "results/test_by_node.json", node_reports)
    write_json(output / "results/test_summary.json", metrics)
    atomic_write(output / "results/test_predictions.npz", lambda handle: np.savez_compressed(
        handle, predictions_kw=predicted, targets_kw=actual, persistence_kw=persistence, origin=origins, node=node_ids))
    # Headless figures are exported, never opened as a GUI on the server.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history = json.loads((output / "results/history.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Train")
    ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="MSE (scaled target units)", title="Training history")
    ax.legend()
    fig.tight_layout()
    atomic_write(output / "results/loss_curve.png", lambda handle: fig.savefig(handle, format="png", dpi=160))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    selected = np.flatnonzero(node_ids == node_ids[0])[:288]
    from src.forecast_plot import plot_forecast
    plot_forecast(ax, origins[selected], actual[selected, 0], predicted[selected, 0], persistence[selected, 0],
                  metadata["settings"]["resample_minutes"], metadata["settings"]["horizon_minutes"],
                  node_ids[0], unit="kW")
    fig.tight_layout()
    atomic_write(output / "results/test_forecast.png", lambda handle: fig.savefig(handle, format="png", dpi=160))
    plt.close(fig)
    logger.info("Held-out test: MAE=%.6f kW RMSE=%.6f kW (%d samples)", metrics["mae_kw"], metrics["rmse_kw"], len(y))
    return metrics


def execute(args):
    config_path = portable_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if set(config) != {"dataset", "preprocessing", "training_config"}:
        raise ValueError("Pipeline config requires dataset, preprocessing and training_config")
    settings = config["preprocessing"]
    archives = config["dataset"]["archives"]
    months = [item["month"] for item in archives]
    if not archives or len(set(months)) != len(months) or any(not re.fullmatch(r"\d{2}-(0[1-9]|1[0-2])", month) for month in months):
        raise ValueError("Archives require unique YY-MM months")
    if sorted(months) != sorted(settings["months"]):
        raise ValueError("Archive months and preprocessing months must match")
    if any(not re.fullmatch(r"[a-fA-F0-9]{32}", item["md5"]) for item in archives):
        raise ValueError("Every archive requires the publisher's 32-digit MD5")
    dataset_signature = digest({"version": PIPELINE_VERSION,
                                "archives": sorted((item["month"], item["md5"].lower()) for item in archives),
                                "metrics": sorted(METRICS)})
    data_dir = portable_path(args.data_dir or os.environ.get("M100_DATA_DIR") or "data")
    output = portable_path(args.output_dir or os.environ.get("TRAIN_OUTPUT_DIR") or "output/m100")
    archive_root = data_dir / "archives"
    # Content-specific caches cannot mix stale files from replacement archives.
    extracted = data_dir / "extracted" / dataset_signature
    with RunLock(output / ".pipeline.lock"):
        (output / "logs").mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("pipeline")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "logs/pipeline.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        cache_lock = None
        try:
            from src.download import download_file, extract_selected
            extraction_receipts = []
            if args.stage in ("all", "download", "prepare"):
                # Different runs may share one raw cache, but never write it concurrently.
                cache_lock = RunLock(data_dir / ".cache.lock")
                cache_lock.__enter__()
                for item in archives:
                    destination = archive_root / f"{item['month']}.tar"
                    if args.stage != "prepare":
                        download_file(item["url"], destination, item["md5"], logger=logger,
                                      ca_bundle=portable_path(args.ca_bundle) if args.ca_bundle else None)
                    elif not destination.is_file() or file_hash(destination, "md5") != item["md5"].lower():
                        raise RuntimeError(f"Missing or unverified {destination.name}; run --stage download first")
                    if args.stage == "download":
                        continue
                    fingerprint = digest({"version": PIPELINE_VERSION, "archive_md5": item["md5"], "metrics": sorted(METRICS)})
                    receipt_path = output / "stages" / f"extract_{item['month']}.json"
                    if extraction_is_complete(receipt_path, extracted, fingerprint):
                        logger.info("Reusing verified extraction for %s", item["month"])
                    else:
                        logger.info("Extracting selected metrics for %s", item["month"])
                        extract_selected(destination, extracted, METRICS, logger=logger)
                        files = sorted((extracted / f"year_month={item['month']}").rglob("*.parquet"))
                        if not files:
                            raise RuntimeError(f"No selected Parquet files found for {item['month']}")
                        receipt = {"fingerprint": fingerprint,
                                   "files": [{"path": p.relative_to(extracted).as_posix(), "sha256": file_hash(p)} for p in files]}
                        write_json(receipt_path, receipt)
                    extraction_receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
                if args.stage == "download":
                    logger.info("Downloads verified; next: --stage prepare")
                    return
            preparation_receipt = output / "stages/prepare.json"
            if args.stage in ("all", "prepare"):
                preparation_fingerprint = digest({"version": PIPELINE_VERSION, "settings": settings, "extraction": extraction_receipts})
                if preparation_receipt.is_file():
                    receipt = json.loads(preparation_receipt.read_text(encoding="utf-8"))
                    if receipt["fingerprint"] != preparation_fingerprint:
                        raise ValueError("Preprocessing/data changed in an existing output. Use original settings or a new --output-dir.")
                    if not all((output / p).is_file() and file_hash(output / p) == expected for p, expected in receipt["artifacts"].items()):
                        raise RuntimeError("Prepared artifacts are missing or changed; use a new --output-dir to prepare again")
                    logger.info("Reusing verified prepared data")
                else:
                    from src.preprocess import prepare_m100
                    prepared = prepare_m100(extracted, output, settings, logger)
                    artifact_paths = [Path(prepared[key]) for key in ("data_path", "preprocessing_path", "audit_path")]
                    receipt = {"fingerprint": preparation_fingerprint,
                               "dataset_signature": dataset_signature,
                               "settings_signature": digest(settings),
                               "artifacts": {p.relative_to(output).as_posix(): file_hash(p) for p in artifact_paths}}
                    write_json(preparation_receipt, receipt)
                if args.stage == "prepare":
                    logger.info("Preparation complete; next: --stage train")
                    return
            if cache_lock is not None:
                cache_lock.__exit__(None, None, None)
                cache_lock = None
            if not preparation_receipt.is_file():
                raise RuntimeError("Prepared data is unavailable; run --stage prepare first")
            receipt = json.loads(preparation_receipt.read_text(encoding="utf-8"))
            if receipt.get("dataset_signature") != dataset_signature:
                raise ValueError("Dataset archive configuration changed; choose a new --output-dir")
            if receipt["settings_signature"] != digest(settings):
                raise ValueError("Preprocessing settings changed; choose a new --output-dir")
            for path, expected in receipt["artifacts"].items():
                if not (output / path).is_file() or file_hash(output / path) != expected:
                    raise RuntimeError("Prepared data integrity check failed")
            dataset_path, metadata_path = output / "prepared/dataset.npz", output / "prepared/preprocessing.json"
            command = [sys.executable, str(ROOT / "main.py"), "--config", str(portable_path(config["training_config"])),
                       "--data-path", str(dataset_path), "--output-dir", str(output)]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.device is not None:
                command += ["--device", args.device]
            # The parent forwards graceful termination to the trainer so that its
            # own termination does not leave an unmonitored training child.
            process = subprocess.Popen(command, cwd=ROOT)
            previous_handlers = {}
            def forward_stop(signum, frame):
                if process.poll() is None:
                    process.send_signal(signum)
            try:
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.signal(signum, forward_stop)
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
            final_exports(output, dataset_path, metadata_path, logger)
            write_json(output / "stages/completed.json", {"status": "completed", "preparation_fingerprint": receipt["fingerprint"],
                                                          "test_summary": "results/test_summary.json"})
        finally:
            if cache_lock is not None:
                cache_lock.__exit__(None, None, None)
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            logger.handlers.clear()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m100.json")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ca-bundle", default=None,
                        help="PEM file containing additional trusted HTTPS CA certificates")
    parser.add_argument("--stage", choices=["all", "download", "prepare", "train"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    try:
        execute(parser.parse_args(argv))
        return 0
    except Exception as error:
        print(f"PIPELINE ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
