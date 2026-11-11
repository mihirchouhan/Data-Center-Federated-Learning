"""Strict versioned specification; no real-data path or execution option."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import copy
import hashlib
import json
import re
from .schema import FEATURES, UNITS, TARGETS, TARGET_UNITS, WIDTH_NS

def utc_ns(value):
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",value) is None:
        raise ValueError("Require explicit UTC Z timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1_000_000_000

@dataclass(frozen=True)
class Spec:
    start: int
    train_end: int
    val_end: int
    end: int
    minimum_windows: int
    minimum_days: int
    hidden_size: int
    layers: int
    embedding_size: int
    seed: int
    profile: str
    digest: str
    history: int = 36
    horizon: int = 6
    stride: int = 6
    width: int = WIDTH_NS

    def bounds(self, split):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        return {"train": (self.start, self.train_end),
                "val": (self.train_end, self.val_end), "test": (self.val_end, self.end)}[split]

def parse_spec(document, profile="experiment"):
    d = copy.deepcopy(document)
    required = {"version", "phase", "execution", "features", "feature_units", "targets", "target_units",
        "history_steps", "horizon_steps", "origin_stride_steps", "missing", "target_aggregation",
        "split_rule", "normalization", "model", "client", "evaluation", "profiles"}
    if set(d) != required or d["version"] != 1 or d["phase"] != "phase1_3":
        raise ValueError("Unknown specification schema")
    expected = {
        "execution": "synthetic_only_cpu_no_training_no_aggregation_no_scoring",
        "features": list(FEATURES), "feature_units": list(UNITS),
        "targets": list(TARGETS), "target_units": list(TARGET_UNITS),
        "history_steps": 36, "horizon_steps": 6, "origin_stride_steps": 6,
        "missing": "no_fill_reject_unobserved_except_masked_weather_7_8",
        "target_aggregation": "six_original_observed_bins_arithmetic_mean_both_targets",
        "split_rule": "whole_history_and_target_inside_same_split",
        "normalization": "fit_accepted_training_windows_only_ddof0_mask_7_8_id13_identity_std_floor_1e-12",
    }
    for k, v in expected.items():
        if d[k] != v:
            raise ValueError("Specification differs: " + k)
    if d["model"].keys() != {"hidden_size", "layers", "embedding_size", "head", "loss"}:
        raise ValueError("Unknown model configuration")
    if d["model"]["head"] != "linear_hidden_to_2_no_residual" or d["model"]["loss"] != "future_equal_output_mean_standardized_mse":
        raise ValueError("Unknown architecture/loss")
    if d["client"] != "one_compute_node_one_simulated_client_training_only_eligibility":
        raise ValueError("Unknown client contract")
    if d["evaluation"] != "future_per_output_W_degC_pooled_and_equal_node_mae_rmse_no_scoring_implementation":
        raise ValueError("Unknown evaluation contract")
    if set(d["profiles"]) != {"experiment", "synthetic"} or profile not in d["profiles"]:
        raise ValueError("Unknown profile")
    p = d["profiles"][profile]
    if set(p) != {"start_utc", "train_end_utc", "val_end_utc", "end_utc", "minimum_windows", "minimum_days", "seed"}:
        raise ValueError("Unknown temporal/client settings")
    ints = [p["minimum_windows"], p["minimum_days"], *[d["model"][k] for k in ("hidden_size", "layers", "embedding_size")]]
    if any(type(v) is not int or v <= 0 for v in ints) or type(p["seed"]) is not int or p["seed"] < 0:
        raise ValueError("Dimensions/counts must be positive integers; seed nonnegative")
    edges = [utc_ns(p[k]) for k in ("start_utc", "train_end_utc", "val_end_utc", "end_utc")]
    if any(b <= a or (b-a) % WIDTH_NS for a,b in zip(edges, edges[1:])) or edges[0] % WIDTH_NS:
        raise ValueError("Unordered or off-grid split boundaries")
    if any((b-a) < 42*WIDTH_NS for a,b in zip(edges, edges[1:])):
        raise ValueError("Each split needs at least 36+6 bins")
    digest = hashlib.sha256(json.dumps({"document":d, "profile":profile},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return Spec(*edges, p["minimum_windows"],p["minimum_days"],d["model"]["hidden_size"],
        d["model"]["layers"],d["model"]["embedding_size"],p["seed"],profile,digest)

def load_spec(path, profile="experiment"):
    # Callers must provide only the new Phase 1.3 JSON. This is not a dataset loader.
    return parse_spec(json.loads(Path(path).read_text()), profile)
