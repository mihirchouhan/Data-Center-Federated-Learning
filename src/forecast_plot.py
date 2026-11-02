"""Forecasts on UTC time axes; missing grid intervals are never joined."""
from datetime import timezone
import numpy as np


def gap_series(origins, values, interval_minutes):
    times = np.asarray(origins).astype('datetime64[ns]')
    values = np.asarray(values).reshape(-1)
    if len(times) != len(values) or np.isnat(times).any():
        raise ValueError('Invalid forecast timestamps')
    order = np.argsort(times, kind='stable')
    times, values = times[order], values[order]
    if len(times) > 1 and (np.diff(times) <= np.timedelta64(0, 'ns')).any():
        raise ValueError('Duplicate forecast timestamps')
    step = np.timedelta64(int(interval_minutes), 'm')
    if step <= np.timedelta64(0, 'm'):
        raise ValueError('Grid interval must be positive')
    breaks = np.flatnonzero(np.diff(times) > step) + 1
    # Insert NaN immediately after the preceding observation, on the real time axis.
    return np.insert(times, breaks, times[breaks - 1] + step), np.insert(values.astype(float), breaks, np.nan)


def plot_forecast(ax, origins, observed, predicted, persistence, interval_minutes, horizon_minutes, node, unit='W'):
    import matplotlib.dates as mdates
    for values, label in ((observed, 'Observed'), (predicted, 'Forecast'), (persistence, 'Persistence')):
        times, values = gap_series(origins, values, interval_minutes)
        ax.plot(times, values, label=label, alpha=0.65 if label == 'Persistence' else 1)
    locator = mdates.AutoDateLocator(tz=timezone.utc)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=timezone.utc))
    ax.set(xlabel='Forecast origin (UTC)', ylabel=f'Next-{horizon_minutes}-minute mean node power ({unit})',
           title=f'Forecast: node {node}')
    ax.legend()
