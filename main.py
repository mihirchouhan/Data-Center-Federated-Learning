"""Portable LSTM/GRU training with durable automatic mid-epoch recovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import platform
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch

from src.data import load_data
from src.io_utils import RunLock, atomic_copy, atomic_write, write_json
from src.model import SequenceRegressor

PROJECT_ROOT = Path(__file__).resolve().parent
FORMAT_VERSION = 1


def portable_path(value):
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def rng_state(shuffle_generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "shuffle": shuffle_generator.get_state(),
    }


def restore_rng(saved, shuffle_generator, logger):
    random.setstate(saved["python"])
    name, keys, position, has_gauss, cached = saved["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(saved["torch"].cpu())
    shuffle_generator.set_state(saved["shuffle"].cpu())
    if saved["cuda"] and torch.cuda.is_available():
        if len(saved["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(saved["cuda"])
        else:
            logger.warning("CUDA device count changed; progress resumes but exact RNG replay is unavailable")


def cpu_weights(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def save_checkpoint(path, model, optimizer, state, signature, shuffle_generator, *, rotate_previous=True):
    """Save weights, optimizer, epoch/loss/batch state and RNG as one durable file.

    Call only after a completed optimizer step or an epoch boundary. Serialization
    uses tensors/primitives so torch.load(weights_only=True) can restore the file.
    """
    payload = {
        "format_version": FORMAT_VERSION, "signature": signature,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": state["epoch"], "loss": state["loss"], "training_state": state,
        "rng_state": rng_state(shuffle_generator),
        "runtime": {"torch": str(torch.__version__), "python": platform.python_version()},
    }
    path = Path(path)
    if path.exists() and rotate_previous:
        atomic_copy(path, path.with_name("previous.pt"))
    atomic_write(path, lambda handle: torch.save(payload, handle))


def load_checkpoint(directory, model, optimizer, signature, shuffle_generator, logger=None):
    """Automatically restore last.pt; fall back to previous.pt only for damage.

    A readable incompatible checkpoint fails immediately rather than silently
    starting a different experiment or searching for an older compatible run.
    Returns (training_state, checkpoint_path), or (None, None) for a new run.
    """
    logger = logger or logging.getLogger("training")
    candidates = [Path(directory) / "last.pt", Path(directory) / "previous.pt"]
    failures = []
    initial_weights = cpu_weights(model)
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = rng_state(shuffle_generator)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict) or not {"format_version", "signature"}.issubset(saved):
                raise ValueError("Checkpoint header is incomplete")
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            logger.warning("Cannot read %s; trying the retained previous checkpoint: %s", path.name, error)
            continue
        if saved["format_version"] != FORMAT_VERSION or saved["signature"] != signature:
            raise ValueError("Checkpoint is incompatible with this model, dataset or training configuration. "
                             "Use the original settings or a new --output-dir.")
        try:
            required = {"format_version", "signature", "model_state_dict", "optimizer_state_dict",
                        "training_state", "rng_state", "epoch", "loss"}
            if not isinstance(saved, dict) or not required.issubset(saved):
                raise ValueError("Checkpoint payload is incomplete")
            state = saved["training_state"]
            if not isinstance(state, dict) or not {"epoch", "cursor", "permutation", "history", "global_step",
                                                    "loss_sum", "loss_count", "best_weights", "best_val_loss", "loss"}.issubset(state):
                raise ValueError("Checkpoint progress is incomplete")
            if any(not isinstance(state[key], int) or isinstance(state[key], bool) or state[key] < 0
                   for key in ("epoch", "cursor", "global_step", "loss_count")):
                raise ValueError("Checkpoint progress counters are invalid")
            if not isinstance(state["history"], list) or len(state["history"]) != state["epoch"]:
                raise ValueError("Checkpoint epoch/history counters disagree")
            if state["epoch"] != saved["epoch"] or state["loss_count"] != state["cursor"]:
                raise ValueError("Checkpoint epoch/batch counters disagree")
            permutation = state["permutation"]
            if permutation is None:
                if state["cursor"] != 0:
                    raise ValueError("Checkpoint has a batch cursor without an epoch permutation")
            elif (not isinstance(permutation, torch.Tensor) or permutation.ndim != 1
                  or permutation.dtype != torch.int64 or state["cursor"] > len(permutation)
                  or not torch.equal(permutation.sort().values, torch.arange(len(permutation)))):
                raise ValueError("Checkpoint shuffle permutation is invalid")
            if not math.isfinite(float(state["loss_sum"])) or state["loss"] is not None and not math.isfinite(float(state["loss"])):
                raise ValueError("Checkpoint loss is non-finite")
            for value in saved["model_state_dict"].values():
                if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
                    raise ValueError("Checkpoint model weights are invalid")
            # A failed restore may mutate some parameters or RNG states. Reset the
            # factory state before each attempt, then apply one coherent candidate.
            model.load_state_dict(initial_weights, strict=True)
            optimizer.load_state_dict(initial_optimizer)
            restore_rng(initial_rng, shuffle_generator, logger)
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            restore_rng(saved["rng_state"], shuffle_generator, logger)
        except Exception as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
            logger.warning("Cannot restore %s; trying the retained previous checkpoint: %s", path.name, error)
            continue
        logger.info("Resumed %s: completed_epochs=%s next_sample=%s global_step=%s",
                    path.name, state["epoch"], state["cursor"], state["global_step"])
        return state, path
    if failures:
        model.load_state_dict(initial_weights, strict=True)
        optimizer.load_state_dict(initial_optimizer)
        restore_rng(initial_rng, shuffle_generator, logger)
        raise RuntimeError("Existing checkpoints could not be restored; refusing to reset progress. " + "; ".join(failures))
    return None, None


def evaluate(model, data, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(data["X_val"]), batch_size):
            predictions.append(model(data["X_val"][start:start + batch_size].to(device)).cpu())
    predicted = torch.cat(predictions)
    actual = data["y_val"]
    difference = predicted.double() - actual.double()
    if not torch.isfinite(difference).all():
        raise FloatingPointError("Validation produced non-finite predictions")
    mse = float(difference.square().mean())
    return predicted.numpy(), {"mse": mse, "mae": float(difference.abs().mean()), "rmse": math.sqrt(mse)}


def configure_logger(output):
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output / "logs/train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default=None, help="Default: TRAIN_OUTPUT_DIR or output/")
    parser.add_argument("--data-path", default=None, help="External NPZ; default: TRAIN_DATA_PATH, otherwise synthetic")
    parser.add_argument("--fresh", action="store_true", help="Require a new/empty run directory; never erase a previous run")
    parser.add_argument("--stop-after-step", type=int, default=None, help="Controlled interruption for recovery testing")
    for name in ("epochs", "batch_size", "hidden_size", "num_layers", "num_threads", "seed",
                 "checkpoint_every_steps", "checkpoint_every_epochs", "synthetic_samples",
                 "sequence_length", "input_size", "target_size", "early_stopping_patience"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--model", choices=["lstm", "gru"], default=None)
    return parser.parse_args(argv)


def run(args):
    with portable_path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"seed", "epochs", "batch_size", "learning_rate", "model", "hidden_size", "num_layers", "num_threads",
                "device", "checkpoint_every_steps", "checkpoint_every_epochs", "synthetic_samples", "sequence_length", "input_size", "target_size"}
    optional = {"prediction_weights", "forecast_mode", "node_embedding_size", "early_stopping_patience"}
    if not isinstance(config, dict) or not required.issubset(config) or set(config) - required - optional:
        raise ValueError(f"Configuration requires {sorted(required)}; optional keys: {sorted(optional)}")
    if args.early_stopping_patience is not None:
        config['early_stopping_patience'] = args.early_stopping_patience
    patience = config.get('early_stopping_patience', 0)
    if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
        raise ValueError('early_stopping_patience must be a nonnegative integer')
    if config.get("prediction_weights", "final") not in ("final", "best"):
        raise ValueError("prediction_weights must be final or best")
    if config.get("forecast_mode", "direct") not in ("direct", "residual"):
        raise ValueError("forecast_mode must be direct or residual")
    embedding_size = config.get("node_embedding_size", 8)
    if not isinstance(embedding_size, int) or isinstance(embedding_size, bool) or embedding_size <= 0:
        raise ValueError("node_embedding_size must be a positive integer")
    for key in required:
        override = getattr(args, key, None)
        if override is not None:
            config[key] = override
    for key in required - {"learning_rate", "model", "device", "seed", "checkpoint_every_steps"}:
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (not isinstance(config["seed"], int) or not 0 <= config["seed"] < 2**32
            or not isinstance(config["checkpoint_every_steps"], int) or config["checkpoint_every_steps"] < 0
            or not isinstance(config["learning_rate"], (int, float))
            or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0):
        raise ValueError("Invalid seed, checkpoint step interval or learning rate")
    if config["synthetic_samples"] < 5 or config["model"] not in ("lstm", "gru") or config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Invalid synthetic sample count, model or device")
    if args.stop_after_step is not None and args.stop_after_step <= 0:
        raise ValueError("--stop-after-step must be positive")
    output = portable_path(args.output_dir or os.environ.get("TRAIN_OUTPUT_DIR") or "output")
    supplied_data = args.data_path or os.environ.get("TRAIN_DATA_PATH")
    data_path = portable_path(supplied_data) if supplied_data else None
    with RunLock(output / ".run.lock"):
        existing = [p for p in output.rglob("*") if p.is_file() and p.name != ".run.lock"]
        checkpoint_directory = output / "checkpoints"
        if args.fresh and existing:
            raise ValueError("--fresh requires a new/empty --output-dir; existing run files are preserved")
        training_artifacts = ((output / "run_config.json").exists()
                              or any((output / "models").glob("*.pt"))
                              or any((output / "results" / name).exists() for name in ("history.json", "summary.json", "val_predictions.npz")))
        if training_artifacts and not any((checkpoint_directory / name).exists() for name in ("last.pt", "previous.pt")):
            raise ValueError("Output contains artifacts but no resume checkpoint. Choose a new --output-dir.")
        for name in ("checkpoints", "logs", "models", "results"):
            (output / name).mkdir(parents=True, exist_ok=True)
        logger = configure_logger(output)
        torch.set_num_threads(config["num_threads"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if config["device"] == "cuda":
            if torch.version.cuda is None:
                raise RuntimeError(
                    "CUDA was requested, but this environment has CPU-only PyTorch installed. "
                    "Install a CUDA-enabled PyTorch wheel in this project's .venv; see README.md."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but PyTorch cannot access an NVIDIA GPU. "
                    "Check nvidia-smi in your server terminal; an NVIDIA driver/library "
                    "version mismatch needs the server administrator to repair the driver."
                )
        device = torch.device("cuda" if config["device"] == "cuda" or (config["device"] == "auto" and torch.cuda.is_available()) else "cpu")
        data, data_info = load_data(config, data_path)
        fingerprint_config = {key: config[key] for key in ("seed", "batch_size", "learning_rate", "model", "hidden_size", "num_layers")}
        fingerprint_config["data"] = data_info
        if any(key in config for key in optional - {'early_stopping_patience'}):
            fingerprint_config["forecast_mode"] = config.get("forecast_mode", "direct")
            fingerprint_config["prediction_weights"] = config.get("prediction_weights", "final")
            fingerprint_config["node_embedding_size"] = embedding_size
        signature = hashlib.sha256(json.dumps(fingerprint_config, sort_keys=True).encode()).hexdigest()
        residual = config.get("forecast_mode", "direct") == "residual"
        baseline_coefficients = data_info.get("power_baseline_coefficients") if residual else None
        if residual and (baseline_coefficients is None or data_info["target_size"] != 1):
            raise ValueError("Residual forecasting requires prepared node-power data with baseline scaling coefficients")
        model_options = {"node_count": data_info.get("node_count", 0),
                         "node_embedding_size": embedding_size,
                         "baseline_coefficients": baseline_coefficients}
        model = SequenceRegressor(data_info["input_size"], data_info["target_size"], config["hidden_size"], config["num_layers"], config["model"], **model_options).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        shuffle_generator = torch.Generator().manual_seed(config["seed"] + 1)
        state, restored_from = load_checkpoint(checkpoint_directory, model, optimizer, signature, shuffle_generator, logger)
        if state is not None and restored_from is not None:
            saved_runtime = torch.load(restored_from, map_location="cpu", weights_only=True).get("runtime", {})
            if saved_runtime.get("torch") != str(torch.__version__):
                logger.warning("PyTorch version changed; progress resumes, but exact numeric replay is not guaranteed")
        if state is None:
            state = {"epoch": 0, "global_step": 0, "cursor": 0, "permutation": None,
                     "loss": None, "loss_sum": 0.0, "loss_count": 0,
                     "history": [], "best_val_loss": None, "best_weights": None}
            if residual:
                _, initial_metrics = evaluate(model, data, config["batch_size"], device)
                state.update(best_val_loss=initial_metrics["mse"], best_weights=cpu_weights(model), best_epoch=0)
        if config["epochs"] < state["epoch"] or (state["permutation"] is not None and config["epochs"] <= state["epoch"]):
            raise ValueError("Requested epochs would truncate already completed/in-progress training")
        if state.get('early_stopped') and (patience == 0 or config['epochs'] > state.get('epoch_limit_at_stop', config['epochs'])):
            state.update(early_stopped=False, patience_start_epoch=state['epoch'])
        if patience and state.get('best_epoch') is None and state['history']:
            state['best_epoch'] = min(state['history'], key=lambda row: row['val_loss'])['epoch']
        if patience and state['epoch'] < config['epochs'] and state['epoch'] - max(state.get('best_epoch') or 0, state.get('patience_start_epoch', 0)) >= patience:
            state.update(early_stopped=True, epoch_limit_at_stop=config['epochs'])
            logger.info('Early stopping on resume: validation has not improved since epoch %s', state.get('best_epoch'))
        rotate_previous = restored_from is None or restored_from.name == "last.pt"
        logger.info("Device=%s model=%s dataset=%s output=%s", device, config["model"], data_info["kind"], output)
        write_json(output / "run_config.json", {"config": config, "device": str(device), "data": data_info,
                                               "signature": signature, "python": platform.python_version(),
                                               "torch": str(torch.__version__), "numpy": str(np.__version__)})
        # History is rebuilt from checkpoint state if power loss left JSON behind.
        write_json(output / "results/history.json", state["history"])
        stop = {"requested": False, "reason": None}
        def request_stop(signum, frame):
            stop.update(requested=True, reason=signal.Signals(signum).name)
        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        def persist():
            nonlocal rotate_previous
            save_checkpoint(checkpoint_directory / "last.pt", model, optimizer, state, signature,
                            shuffle_generator, rotate_previous=rotate_previous)
            rotate_previous = True
            write_json(output / "results/history.json", state["history"])
        try:
            if restored_from is None:
                persist()  # Initial checkpoint preserves even startup-only interruptions.
            n = len(data["X_train"])
            if state["cursor"] < 0 or state["cursor"] > n:
                raise ValueError("Checkpoint batch cursor is outside the training dataset")
            while state["epoch"] < config["epochs"] and not stop["requested"] and not state.get('early_stopped', False):
                if state["permutation"] is None:
                    state["permutation"] = torch.randperm(n, generator=shuffle_generator)
                    state.update(cursor=0, loss_sum=0.0, loss_count=0)
                model.train()
                while state["cursor"] < n and not stop["requested"]:
                    start = state["cursor"]
                    indices = state["permutation"][start:start + config["batch_size"]]
                    x = data["X_train"].index_select(0, indices).to(device)
                    y = data["y_train"].index_select(0, indices).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.mse_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss; retaining the last safe checkpoint")
                    loss.backward()
                    optimizer.step()
                    # Cursor advances only after the optimizer step is complete.
                    state["cursor"] = start + len(indices)
                    state["global_step"] += 1
                    state["loss"] = float(loss.detach().cpu())
                    state["loss_sum"] += state["loss"] * len(indices)
                    state["loss_count"] += len(indices)
                    if args.stop_after_step and state["global_step"] >= args.stop_after_step:
                        stop.update(requested=True, reason="stop_after_step")
                    if config["checkpoint_every_steps"] and state["global_step"] % config["checkpoint_every_steps"] == 0 and not stop["requested"]:
                        persist()
                if stop["requested"]:
                    break
                _, metrics = evaluate(model, data, config["batch_size"], device)
                epoch_loss = state["loss_sum"] / state["loss_count"]
                state["epoch"] += 1
                row = {"epoch": state["epoch"], "global_step": state["global_step"], "train_loss": epoch_loss,
                       "val_loss": metrics["mse"], "val_mae": metrics["mae"], "val_rmse": metrics["rmse"]}
                state["history"].append(row)
                if state["best_val_loss"] is None or metrics["mse"] < state["best_val_loss"]:
                    state["best_val_loss"] = metrics["mse"]
                    state["best_weights"] = cpu_weights(model)
                    state["best_epoch"] = state["epoch"]
                state.update(cursor=0, permutation=None, loss_sum=0.0, loss_count=0)
                logger.info("Epoch %d/%d train_loss=%.6f val_loss=%.6f global_step=%d",
                            state["epoch"], config["epochs"], epoch_loss, metrics["mse"], state["global_step"])
                if patience and state['epoch'] - max(state.get('best_epoch') or 0, state.get('patience_start_epoch', 0)) >= patience:
                    state.update(early_stopped=True, epoch_limit_at_stop=config['epochs'])
                    logger.info('Early stopping: %d epochs without validation improvement; best epoch=%s', patience, state.get('best_epoch'))
                if state["epoch"] % config["checkpoint_every_epochs"] == 0:
                    persist()
            # Always save on a graceful stop or completion, regardless of intervals.
            persist()
            completed = state["epoch"] >= config["epochs"] or state.get('early_stopped', False)
            summary = {"status": "completed" if completed else "interrupted", "completed_epochs": state["epoch"],
                       "global_step": state["global_step"], "next_sample": state["cursor"], "best_val_loss": state["best_val_loss"],
                       "data_kind": data_info["kind"], "device": str(device), "stop_reason": stop["reason"]}
            summary['early_stopped'] = state.get('early_stopped', False)
            summary['requested_epochs'] = config['epochs']
            if state["best_weights"] is not None:
                atomic_write(checkpoint_directory / "best.pt", lambda handle: torch.save(
                    {"model_state_dict": state["best_weights"], "val_loss": state["best_val_loss"], "signature": signature}, handle))
            if completed:
                prediction_model = "final epoch weights"
                if config.get("prediction_weights", "final") == "best" and state["best_weights"] is not None:
                    model.load_state_dict(state["best_weights"])
                    prediction_model = "best validation weights"
                predictions, metrics = evaluate(model, data, config["batch_size"], device)
                summary["final_validation"] = metrics
                summary["prediction_model"] = prediction_model
                summary["best_epoch"] = state.get("best_epoch")
                atomic_write(output / "models/final_weights.pt", lambda handle: torch.save(
                    {"model_state_dict": cpu_weights(model), "model": config["model"], "hidden_size": config["hidden_size"],
                     "num_layers": config["num_layers"], "input_size": data_info["input_size"],
                     "target_size": data_info["target_size"], "signature": signature,
                     "model_options": model_options, "prediction_model": prediction_model}, handle))
                atomic_write(output / "results/val_predictions.npz", lambda handle: np.savez_compressed(
                    handle, predictions=predictions, targets=data["y_val"].numpy()))
            write_json(output / "results/summary.json", summary)
            logger.info("Run %s: epochs=%d step=%d; artifacts in %s", summary["status"], state["epoch"], state["global_step"], output)
            return summary
        except Exception:
            # Do not write a possibly half-updated optimizer after an error. The previous
            # durable checkpoint is the recovery point (particularly for CUDA failures).
            logger.exception("Training failed; restart from the last durable checkpoint")
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            logger.handlers.clear()


def main(argv=None):
    try:
        run(parse_arguments(argv))
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    finally:
        # Also close handlers for startup failures before the training try/finally.
        logger = logging.getLogger("training")
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())
