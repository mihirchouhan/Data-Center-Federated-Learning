"""Explicit physical schema; already-binned synthetic telemetry only."""
from dataclasses import dataclass
import numpy as np

FEATURES = ("power_kw", "cpu_user", "cpu_system", "cpu_idle", "cpu_wio",
    "memory_used_fraction", "inlet_temperature_mean_c", "outdoor_temperature_c",
    "outdoor_humidity_percent", "hour_sin", "hour_cos", "week_sin", "week_cos", "node_index")
UNITS = ("kW", "%", "%", "%", "%", "fraction", "degC", "degC", "%",
         "unitless", "unitless", "unitless", "unitless", "categorical index")
TARGETS = ("next_hour_mean_node_it_power_w", "next_hour_mean_node_inlet_temperature_c")
TARGET_UNITS = ("W", "degC")
MASKED = (7, 8)
WIDTH_NS = 600_000_000_000

def frozen_array(value, dtype=None):
    a = np.ascontiguousarray(value, dtype=dtype)
    if a.dtype.hasobject:
        raise ValueError("Object arrays are forbidden")
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)

@dataclass(frozen=True)
class Telemetry:
    node: str
    times: np.ndarray  # UTC bin START timestamps, explicit datetime64[ns]
    values: np.ndarray  # thirteen original non-ID features
    observed: np.ndarray  # per-feature ORIGINAL observed status
    units: tuple = UNITS[:-1]
    source: str = "synthetic"

@dataclass(frozen=True)
class WindowBatch:
    x: np.ndarray
    y: np.ndarray
    nodes: np.ndarray
    origins: np.ndarray
    history_times: np.ndarray
    target_times: np.ndarray
    split: str
    mapping: tuple
    spec_digest: str
    source: str = "synthetic"

    def __len__(self):
        return len(self.y)
