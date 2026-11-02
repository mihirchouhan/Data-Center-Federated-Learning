"""Single aligned evaluator for neural and boosted predictions, in physical watts."""
import os
from pathlib import Path
import numpy as np
from .experiment_data import immutable_json, sha256
from .io_utils import atomic_write
from .forecast_plot import plot_forecast


def metric_pair(error, baseline_error):
    return {'samples': len(error), 'mae_w': float(np.abs(error).mean()),
            'rmse_w': float(np.sqrt(np.mean(error ** 2))),
            'persistence_mae_w': float(np.abs(baseline_error).mean()),
            'persistence_rmse_w': float(np.sqrt(np.mean(baseline_error ** 2)))}


def evaluate(data, split, predicted_scaled, prediction_nodes, prediction_origins, output):
    x, y, origins, nodes = data.split(split)
    predicted_scaled = np.asarray(predicted_scaled).reshape(-1)
    if len(predicted_scaled) != len(y) or not np.isfinite(predicted_scaled).all():
        raise ValueError('Missing, mis-sized or nonfinite predictions')
    if not np.array_equal(nodes, prediction_nodes) or not np.array_equal(origins, prediction_origins):
        raise ValueError('Prediction window IDs differ from the frozen split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mean, std = data.meta['y_mean'][0], data.meta['y_std'][0]
    actual = (np.asarray(y[:, 0], dtype=np.float64) * std + mean) * 1000
    predicted = (predicted_scaled.astype(np.float64) * std + mean) * 1000
    persistence = (np.asarray(x[:, -1, 0], dtype=np.float64) * data.meta['x_std'][0] + data.meta['x_mean'][0]) * 1000
    error, baseline_error = predicted - actual, persistence - actual
    overall = metric_pair(error, baseline_error)
    by_node = {}
    for name in data.names:
        keep = nodes == name
        if keep.any():
            by_node[str(name)] = metric_pair(error[keep], baseline_error[keep])
    overall.update(unit='W', split=split, evaluated_nodes=len(by_node),
                   macro_node_mae_w=float(np.mean([v['mae_w'] for v in by_node.values()])),
                   macro_node_persistence_mae_w=float(np.mean([v['persistence_mae_w'] for v in by_node.values()])))
    days = origins.astype('datetime64[D]')
    coverage = {'unique_origins': len(np.unique(origins)), 'per_day': {
        str(day): {'windows': int((days == day).sum()), 'unique_origins': len(np.unique(origins[days == day]))}
        for day in np.unique(days)}}
    immutable_json(output / 'metrics.json', overall)
    immutable_json(output / 'per_node.json', by_node)
    immutable_json(output / 'coverage.json', coverage)
    atomic_write(output / 'predictions.npz', lambda f: np.savez_compressed(f, predictions_w=predicted,
                 targets_w=actual, persistence_w=persistence, node=nodes, origin=origins))
    os.environ.setdefault('MPLCONFIGDIR', str(output.parent / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = np.flatnonzero(nodes == nodes[0])[:288]
    fig, ax = plt.subplots(figsize=(11, 4))
    plot_forecast(ax, origins[selected], actual[selected], predicted[selected], persistence[selected],
                  data.meta['settings']['resample_minutes'], data.meta['settings']['horizon_minutes'], nodes[0])
    fig.tight_layout()
    atomic_write(output / 'forecast.png', lambda f: fig.savefig(f, format='png', dpi=160))
    plt.close(fig)
    receipt = {'frozen_contract_sha256': data.contract_sha256,
               'array_hashes': {key: data.contract['arrays'][f'{key}_{split}.npy']['sha256'] for key in ('X', 'y', 'node', 'origin')},
               'artifacts': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    immutable_json(output / 'evaluation.json', receipt)
    return overall
