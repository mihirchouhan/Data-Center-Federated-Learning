"""Immutable frozen arrays and an unchanged-shape standardized weather mask."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np

REVIEWED_SHA256 = 'b0b34c9b3af706d1f0062278ece1eb2a865c2e7f6dde03bb15c8257c4c7101e9'
FEATURES = ['power_kw', 'cpu_user', 'cpu_system', 'cpu_idle', 'cpu_wio', 'memory_used_fraction',
            'inlet_temperature_mean_c', 'outdoor_temperature_c', 'outdoor_humidity_percent',
            'hour_sin', 'hour_cos', 'week_sin', 'week_cos', 'node_index']
WEATHER = [7, 8]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def immutable_json(path, value):
    """Exclusive creation plus fsync and read-only permission; never replace a manifest."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def read_json(path):
    return json.loads(Path(path).read_text())


def weather_view(x, weather):
    if weather not in ('enabled', 'masked'):
        raise ValueError('weather must be enabled or masked')
    if x.ndim != 3 or x.shape[1:] != (6, 14):
        raise ValueError('Primary comparison requires six steps and fourteen features')
    if weather == 'enabled':
        return x
    view = x.copy()
    view[:, :, WEATHER] = 0
    return view


def freeze(source_run, output):
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    source = source_run / 'prepared/dataset.npz'
    meta_path = source_run / 'prepared/preprocessing.json'
    meta = read_json(meta_path)
    if sha256(source) != REVIEWED_SHA256:
        raise ValueError('Source differs from the reviewed M100 dataset')
    if meta['feature_names'] != FEATURES or meta['history_steps'] != 6 or meta['horizon_steps'] != 6:
        raise ValueError('Unexpected feature/history/horizon contract')
    if meta['settings']['resample_minutes'] != 10:
        raise ValueError('Unexpected time grid')
    # mkdir is exclusive: a partial freeze is retained for diagnosis, never overwritten.
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(meta_path, output / 'preprocessing.json')
    arrays = {}
    with zipfile.ZipFile(source) as archive:
        required = [f'{key}_{split}.npy' for split in ('train', 'val', 'test')
                    for key in ('X', 'y', 'origin', 'node')] + ['node_names.npy', 'power_baseline_coefficients.npy']
        for name in required:
            with archive.open(name) as incoming, (output / name).open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            path = output / name
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            arrays[name] = {'sha256': sha256(path), 'shape': list(array.shape), 'dtype': str(array.dtype)}
            path.chmod(0o444)
    (output / 'preprocessing.json').chmod(0o444)
    contract = {'version': 1, 'source_dataset_sha256': REVIEWED_SHA256,
                'source_run': str(source_run), 'arrays': arrays,
                'metadata_sha256': sha256(output / 'preprocessing.json'),
                'window_id': '(node string, forecast origin datetime64[ns] interpreted as UTC)',
                'chronological_cuts': meta['split_boundaries_utc'], 'feature_names': FEATURES,
                'weather_mask_indices': WEATHER, 'history_steps': 6, 'horizon_steps': 6,
                'scalers': {key: meta[key] for key in ('x_mean', 'x_std', 'y_mean', 'y_std')},
                'test_status': 'Previously examined development benchmark; not a new untouched period'}
    immutable_json(output / 'contract.json', contract)
    FrozenData(output).verify()
    return contract


class FrozenData:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.contract = read_json(self.path / 'contract.json')
        self.meta = read_json(self.path / 'preprocessing.json')
        self.contract_sha256 = sha256(self.path / 'contract.json')
        self.names = self.array('node_names')
        self.coefficients = self.array('power_baseline_coefficients')

    def array(self, key):
        return np.load(self.path / f'{key}.npy', mmap_mode='r', allow_pickle=False)

    def verify(self):
        if sha256(self.path / 'preprocessing.json') != self.contract['metadata_sha256']:
            raise ValueError('Frozen metadata changed')
        for name, expected in self.contract['arrays'].items():
            if sha256(self.path / name) != expected['sha256']:
                raise ValueError(f'Frozen array changed: {name}')
        if self.meta['feature_names'] != FEATURES or list(self.names) != self.meta['nodes']:
            raise ValueError('Feature/node mapping differs from contract')
        # Check alignment, finite values, stable categorical identities and chronological uniqueness.
        boundaries = self.meta['split_boundaries_utc']
        cuts = sorted(np.datetime64(v.replace('+00:00', ''), 'ns') for v in boundaries.values())
        for split in ('train', 'val', 'test'):
            x, y, origin, node = self.split(split)
            if x.shape != (len(y), 6, 14) or y.shape != (len(y), 1) or len(origin) != len(y) or len(node) != len(y):
                raise ValueError(f'Shape mismatch: {split}')
            if np.isnat(origin).any() or not len(y):
                raise ValueError(f'Invalid origins: {split}')
            for start in range(0, len(y), 65536):
                batch = x[start:start + 65536]
                indices = batch[:, :, 13]
                idx = indices[:, 0].astype(np.int64)
                if not np.isfinite(batch).all() or not np.isfinite(y[start:start + 65536]).all():
                    raise ValueError(f'Nonfinite data: {split}')
                if not np.array_equal(indices, np.broadcast_to(idx[:, None], indices.shape)) or (idx < 0).any() or (idx >= len(self.names)).any():
                    raise ValueError('Invalid node identity')
                if not np.array_equal(self.names[idx], node[start:start + 65536]):
                    raise ValueError('Node tags do not match embedded identity')
            same_node = node[1:] == node[:-1]
            blocks = np.r_[0, np.flatnonzero(~same_node) + 1]
            if not np.array_equal(node[blocks], self.names) or (np.diff(origin)[same_node] <= np.timedelta64(0, 'ns')).any():
                raise ValueError('Windows must be unique and ordered within each node')
            horizon = np.timedelta64(60, 'm')
            if split == 'train' and (origin + horizon > cuts[0]).any():
                raise ValueError('Training target crosses validation cut')
            if split == 'val' and ((origin < cuts[0]).any() or (origin + horizon > cuts[-1]).any()):
                raise ValueError('Validation target crosses chronological cuts')
            if split == 'test' and (origin < cuts[-1]).any():
                raise ValueError('Test origin precedes test cut')

    def split(self, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        return tuple(self.array(f'{key}_{split}') for key in ('X', 'y', 'origin', 'node'))

    def baseline_scaled(self, x):
        return np.asarray(x[:, -1, 0], dtype=np.float64) * self.coefficients[0] + self.coefficients[1]

    def selection_rmse_w(self, predictions, targets):
        difference = np.asarray(predictions).reshape(-1).astype(np.float64) - np.asarray(targets).reshape(-1)
        if not np.isfinite(difference).all():
            raise ValueError('Nonfinite validation prediction')
        return float(np.sqrt(np.mean(difference ** 2)) * self.meta['y_std'][0] * 1000)
