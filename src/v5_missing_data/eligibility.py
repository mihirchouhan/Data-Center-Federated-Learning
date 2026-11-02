"""V5 missing-data eligibility helpers. No training or imputation."""

from __future__ import annotations

import numpy as np

FEATURES = (
    "power_kw",
    "cpu_user",
    "cpu_system",
    "cpu_idle",
    "cpu_wio",
    "memory_used_fraction",
    "inlet_temperature_mean_c",
    "outdoor_temperature_c",
    "outdoor_humidity_percent",
    "hour_sin",
    "hour_cos",
    "week_sin",
    "week_cos",
)

WEATHER_INDICES = (7, 8)


def history_is_eligible(features, origin, history_steps, masked_indices=WEATHER_INDICES):
    """Check history eligibility while ignoring explicitly masked input channels.

    features has shape (time_steps, 13).
    origin is the first time index after the history window.
    """
    x = np.asarray(features)

    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("features must have shape (time_steps, 13)")
    if history_steps <= 0 or origin < history_steps or origin > len(x):
        return False

    masked = set(masked_indices)
    if any(i < 0 or i >= len(FEATURES) for i in masked):
        raise ValueError("invalid masked feature index")

    start = origin - history_steps
    for channel in range(len(FEATURES)):
        if channel in masked:
            continue
        if not np.isfinite(x[start:origin, channel]).all():
            return False

    return True


def target_is_valid(power, origin, horizon_steps=6):
    """Require every future target bin to be observed and finite."""
    y = np.asarray(power)

    if y.ndim != 1 or horizon_steps <= 0:
        raise ValueError("power must be 1-D and horizon_steps positive")
    end = origin + horizon_steps
    if origin < 0 or end > len(y):
        return False
    return bool(np.isfinite(y[origin:end]).all())


def missing_counts(features, start, end):
    """Count missing values by feature in the half-open interval [start, end)."""
    x = np.asarray(features)

    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError("features must have shape (time_steps, 13)")
    if start < 0 or end < start or end > len(x):
        raise ValueError("invalid interval")

    section = x[start:end]
    return {
        name: int((~np.isfinite(section[:, i])).sum())
        for i, name in enumerate(FEATURES)
    }
