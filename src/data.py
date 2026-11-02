"""Small synthetic smoke data or externally prepared chronological NPZ splits.

This intentionally does not load raw M100 archives or fit research preprocessing.
Prepared NPZ data must already use training-only scaling and valid time splits.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

KEYS = ("X_train", "y_train", "X_val", "y_val")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(config, data_path: Path | None):
    extras = {}
    if data_path is None:
        generator = np.random.default_rng(config["seed"])
        n, t, f, k = (config[key] for key in ("synthetic_samples", "sequence_length", "input_size", "target_size"))
        x = generator.normal(size=(n, t, f)).astype(np.float32)
        # A predictable toy relationship, not simulated M100 measurements.
        y = np.repeat((0.7 * x[:, -1, 0] + 0.3 * x[:, -min(6, t):, 0].mean(axis=1))[:, None], k, axis=1)
        y += generator.normal(scale=0.02, size=(n, k)).astype(np.float32)
        cut = int(n * 0.8)
        arrays = dict(zip(KEYS, (x[:cut], y[:cut], x[cut:], y[cut:])))
        digest = hashlib.sha256()
        for key in KEYS:
            digest.update(key.encode())
            digest.update(arrays[key].tobytes())
        fingerprint = digest.hexdigest()
        kind = "synthetic_smoke"
    else:
        if not data_path.is_file() or data_path.suffix.lower() != ".npz":
            raise ValueError("--data-path must point to an existing prepared .npz file")
        fingerprint = file_hash(data_path)
        with np.load(data_path, allow_pickle=False) as archive:
            missing = set(KEYS) - set(archive.files)
            if missing:
                raise ValueError(f"Prepared dataset is missing keys: {sorted(missing)}")
            arrays = {key: archive[key] for key in KEYS}
            if "node_names" in archive.files:
                names = archive["node_names"]
                if names.ndim != 1 or len(names) == 0 or names.dtype.kind != "U" or len(set(names.tolist())) != len(names):
                    raise ValueError("node_names must be unique nonempty string node IDs")
                extras["node_count"] = len(names)
                for key in ("X_train", "X_val"):
                    if arrays[key].ndim != 3 or arrays[key].shape[2] < 2:
                        raise ValueError(f"{key} needs operational features and a node index")
                    ids = arrays[key][:, :, -1]
                    if not np.isfinite(ids).all() or np.any(ids < 0) or np.any(ids >= len(names)) or np.any(ids != np.floor(ids)):
                        raise ValueError(f"{key} has invalid categorical node indices")
                    if np.any(ids != ids[:, :1]):
                        raise ValueError(f"{key} contains a window crossing node identities")
            if "power_baseline_coefficients" in archive.files:
                coefficients = archive["power_baseline_coefficients"]
                if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
                    raise ValueError("Invalid power baseline scaling coefficients")
                extras["power_baseline_coefficients"] = coefficients.tolist()
        kind = "prepared_npz"
    converted = {}
    for key, array in arrays.items():
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"{key} must contain finite numeric values")
        array = np.ascontiguousarray(array, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains values outside float32 range")
        converted[key] = array
    for split in ("train", "val"):
        x, y = converted[f"X_{split}"], converted[f"y_{split}"]
        if x.ndim != 3 or y.ndim != 2 or len(x) == 0 or len(x) != len(y):
            raise ValueError(f"{split} requires nonempty X [N,T,F] and y [N,K] with matching N")
        if min(x.shape) <= 0 or y.shape[1] <= 0:
            raise ValueError(f"{split} contains an empty sequence, feature or target dimension")
    if (converted["X_train"].shape[1:] != converted["X_val"].shape[1:]
            or converted["y_train"].shape[1:] != converted["y_val"].shape[1:]):
        raise ValueError("Training and validation feature/target dimensions must match")
    info = {"kind": kind, "sha256": fingerprint,
            "shapes": {key: list(value.shape) for key, value in converted.items()},
            "input_size": converted["X_train"].shape[2],
            "target_size": converted["y_train"].shape[1],
            "sequence_length": converted["X_train"].shape[1]}
    info.update(extras)
    return {key: torch.from_numpy(value) for key, value in converted.items()}, info
