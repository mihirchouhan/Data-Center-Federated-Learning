"""Training-only node clients over an anchored frozen contract; no fitting or I/O writes.

The coordinator owns the registry. Workers receive only ClientTrainDataset objects,
which contain copies of their own training windows and no path or parent registry.
This is an API boundary for simulation, not an OS/process privacy sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .experiment_data import FEATURES, read_json, sha256, weather_view


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _readonly(array):
    """Immutable bytes backing prevents re-enabling the returned array's write flag."""
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _utc_time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('Chronological cuts must explicitly use UTC')
    return np.datetime64(parsed.replace(tzinfo=None), 'ns')


@dataclass(frozen=True)
class FrozenScalers:
    x_mean: tuple[float, ...]
    x_std: tuple[float, ...]
    y_mean: tuple[float, ...]
    y_std: tuple[float, ...]


@dataclass(frozen=True, eq=False)
class ClientPartition:
    client_id: str
    node_index: int
    row_indices: np.ndarray
    distinct_training_days: int

    @property
    def num_examples(self):
        return len(self.row_indices)


def build_client_index(node_train, origin_train, node_names, *, train_end_utc,
                       history_minutes=60, horizon_minutes=60, grid_minutes=10,
                       grid_start_utc=None, minimum_windows=1024, minimum_days=7):
    """Partition only explicitly supplied training IDs; preserve original row order.

    All mapped clients must qualify. No filtering of the frozen cohort is allowed.
    Chronological checks reject validation/test origins even if passed as training.
    The public disk loader additionally checks anchored file hashes and X node IDs.
    """
    minimum_windows = _positive_int(minimum_windows, 'minimum_windows')
    minimum_days = _positive_int(minimum_days, 'minimum_days')
    history_minutes = _positive_int(history_minutes, 'history_minutes')
    horizon_minutes = _positive_int(horizon_minutes, 'horizon_minutes')
    grid_minutes = _positive_int(grid_minutes, 'grid_minutes')
    nodes, origins = np.asarray(node_train), np.asarray(origin_train)
    names = np.asarray(node_names)
    if (names.ndim != 1 or not len(names) or names.dtype.kind != 'U'
            or any(not str(name) for name in names) or len(set(names.tolist())) != len(names)):
        raise ValueError('node_names must contain unique nonempty strings')
    if nodes.ndim != 1 or nodes.dtype.kind != 'U' or not len(nodes):
        raise ValueError('node_train must be a nonempty string vector')
    if origins.shape != nodes.shape or origins.dtype != np.dtype('datetime64[ns]') or np.isnat(origins).any():
        raise ValueError('Training origins must be aligned finite datetime64[ns] values')
    cut = _utc_time(train_end_utc)
    if (origins > cut - np.timedelta64(horizon_minutes, 'm')).any():
        raise ValueError('Training target crosses the training boundary')
    if (origins.astype(np.int64) % (grid_minutes * 60 * 10**9) != 0).any():
        raise ValueError('Training origin is off the frozen time grid')
    if grid_start_utc is not None:
        start = _utc_time(grid_start_utc)
        if (origins < start + np.timedelta64(history_minutes, 'm')).any():
            raise ValueError('Training history precedes the frozen calendar')
    # np.unique provides an efficient integer grouping, independent of ID spelling.
    labels, inverse = np.unique(nodes, return_inverse=True)
    if set(labels.tolist()) != set(names.tolist()):
        raise ValueError('Training node tags differ from the frozen node mapping')
    order = np.argsort(inverse, kind='stable')
    counts = np.bincount(inverse)
    edges = np.r_[0, np.cumsum(counts)]
    groups = {str(name): order[edges[i]:edges[i + 1]] for i, name in enumerate(labels)}
    partitions = []
    for index, name in enumerate(names):
        rows = groups[str(name)]
        times = origins[rows]
        if (np.diff(times) <= np.timedelta64(0, 'ns')).any():
            raise ValueError(f'Client {name}: duplicate or unordered training origins')
        days = len(np.unique(times.astype('datetime64[D]')))
        if len(rows) < minimum_windows or days < minimum_days:
            raise ValueError(f'Client {name}: insufficient local training data ({len(rows)} windows, {days} days)')
        partitions.append(ClientPartition(str(name), index, _readonly(rows.astype(np.int64)), days))
    # Stable grouping visits every input row once; enforce that invariant explicitly.
    all_rows = np.concatenate([part.row_indices for part in partitions])
    if len(all_rows) != len(nodes) or not np.array_equal(np.sort(all_rows), np.arange(len(nodes))):
        raise ValueError('Client membership is not disjoint and exhaustive')
    return tuple(partitions)


class _FrozenContext:
    def __init__(self, directory, expected_contract_sha256):
        self.path = Path(directory).resolve()
        if (not isinstance(expected_contract_sha256, str) or len(expected_contract_sha256) != 64
                or any(char not in '0123456789abcdef' for char in expected_contract_sha256)):
            raise ValueError('An explicit lowercase SHA-256 contract anchor is required')
        if sha256(self.path / 'contract.json') != expected_contract_sha256:
            raise ValueError('Frozen contract differs from the expected anchor')
        self.contract_sha256 = expected_contract_sha256
        self.contract = read_json(self.path / 'contract.json')
        self.meta = read_json(self.path / 'preprocessing.json')
        if sha256(self.path / 'preprocessing.json') != self.contract['metadata_sha256']:
            raise ValueError('Frozen preprocessing metadata changed')
        meta = self.meta
        if (meta['feature_names'] != FEATURES or self.contract['feature_names'] != FEATURES
                or meta['history_steps'] != 6 or meta['horizon_steps'] != 6
                or meta['settings']['resample_minutes'] != 10
                or meta['settings']['history_hours'] != 1 or meta['settings']['horizon_minutes'] != 60):
            raise ValueError('Expected the unchanged one-hour, fourteen-feature M100 contract')
        for name in ('x_mean', 'x_std', 'y_mean', 'y_std'):
            value = np.asarray(meta[name], dtype=np.float64)
            size = 14 if name.startswith('x_') else 1
            if value.shape != (size,) or not np.isfinite(value).all() or ('std' in name and (value <= 0).any()):
                raise ValueError(f'Invalid frozen scaler: {name}')
            if self.contract['scalers'][name] != meta[name]:
                raise ValueError(f'Scaler differs from the frozen contract: {name}')
        if meta['x_mean'][13] != 0 or meta['x_std'][13] != 1:
            raise ValueError('Categorical node indices must not be standardized')
        self.scalers = FrozenScalers(*(tuple(meta[name]) for name in ('x_mean', 'x_std', 'y_mean', 'y_std')))
        self.names = self.array('node_names')
        if self.names.ndim != 1 or self.names.dtype.kind != 'U' or self.names.tolist() != meta['nodes']:
            raise ValueError('Frozen node mapping differs from preprocessing')
        coefficients = self.array('power_baseline_coefficients')
        expected = np.array([meta['x_std'][0] / meta['y_std'][0],
                             (meta['x_mean'][0] - meta['y_mean'][0]) / meta['y_std'][0]])
        if coefficients.shape != (2,) or not np.isfinite(coefficients).all() or not np.allclose(coefficients, expected, rtol=1e-12, atol=1e-12):
            raise ValueError('Persistence coefficients do not match frozen scalers')
        self.coefficients = tuple(float(value) for value in coefficients)
        cuts = meta['split_boundaries_utc']
        if self.contract['chronological_cuts'] != cuts:
            raise ValueError('Chronological cuts differ from frozen preprocessing')
        self.train_end = cuts['train_end_val_start']
        self.val_end = cuts['val_end_test_start']
        self.grid_start = datetime.strptime(min(meta['months']), '%y-%m').strftime('%Y-%m-01T00:00:00+00:00')

    def array(self, name):
        # The caller supplies a constant allowlisted name, never a split argument.
        allowed = {'node_names', 'power_baseline_coefficients', 'X_train', 'y_train', 'node_train',
                   'origin_train', 'X_val', 'y_val', 'node_val', 'origin_val'}
        if name not in allowed:
            raise PermissionError('The Stage 1 interface never opens test arrays')
        path = self.path / f'{name}.npy'
        expected = self.contract['arrays'][path.name]
        if sha256(path) != expected['sha256']:
            raise ValueError(f'Frozen array changed: {path.name}')
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != expected['shape'] or str(value.dtype) != expected['dtype']:
            raise ValueError(f'Frozen array schema changed: {path.name}')
        return value

    def split_arrays(self, split):
        if split not in ('train', 'val'):
            raise PermissionError('Test data is unavailable in this interface')
        x, y, origins, nodes = (self.array(f'{key}_{split}') for key in ('X', 'y', 'origin', 'node'))
        if x.dtype != np.float32 or y.dtype != np.float32 or x.shape != (len(y), 6, 14) or y.shape != (len(y), 1) or not len(y):
            raise ValueError('Invalid frozen input/target shapes or dtypes')
        if nodes.shape != (len(y),) or nodes.dtype.kind != 'U' or origins.shape != (len(y),) or origins.dtype != np.dtype('datetime64[ns]'):
            raise ValueError('Invalid frozen window IDs')
        if np.isnat(origins).any():
            raise ValueError('Nonfinite forecast origins')
        lookup = {str(name): index for index, name in enumerate(self.names)}
        for start in range(0, len(y), 65536):
            batch = x[start:start + 65536]
            tags = nodes[start:start + 65536]
            try:
                identities = np.fromiter((lookup[str(name)] for name in tags), dtype=np.int64, count=len(tags))
            except KeyError as error:
                raise ValueError('Window contains an unknown node ID') from error
            if not np.isfinite(batch).all() or not np.isfinite(y[start:start + 65536]).all():
                raise ValueError('Nonfinite training/validation values')
            if not np.array_equal(batch[:, :, 13], np.broadcast_to(identities[:, None], (len(tags), 6))):
                raise ValueError('Input identity differs from node tags or crosses nodes')
        return x, y, origins, nodes


@dataclass(frozen=True)
class ClientBatch:
    inputs: np.ndarray
    targets: np.ndarray
    origins: np.ndarray
    row_indices: np.ndarray


def _positions(values, size):
    positions = np.asarray(values)
    if positions.ndim != 1 or positions.dtype.kind not in 'iu' or (positions < 0).any() or (positions >= size).any():
        raise IndexError('Batch positions must be local integer positions in range')
    return positions.astype(np.int64)


class ClientTrainDataset:
    """Map-style NumPy dataset; construct only with TrainingClients.client().

    __getitem__ returns independent (inputs, targets) arrays suitable for standard
    DataLoader collation. batch() also returns aligned origins and global row IDs.
    Neither this object nor its serialized state contains validation/test arrays.
    """
    def __init__(self, *args, **kwargs):
        raise TypeError('Construct client data with load_training_clients(...).client(client_id)')

    @classmethod
    def _from_training(cls, partition, arrays, weather, scalers, coefficients, node_count):
        result = object.__new__(cls)
        x, y, origins, _ = arrays
        rows = partition.row_indices
        result.client_id = partition.client_id
        result.node_index = partition.node_index
        result.node_count = node_count
        result.weather = weather
        result.scalers = scalers
        result.baseline_coefficients = coefficients
        result.__x = _readonly(weather_view(np.array(x[rows], copy=True), weather))
        result.__y = _readonly(y[rows])
        result.__origins = _readonly(origins[rows])
        result.__rows = _readonly(rows)
        return result

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, position):
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, Integral) or not 0 <= position < len(self):
            raise IndexError('Use a local training-window position')
        return self.__x[position].copy(), self.__y[position].copy()

    def batch(self, positions):
        positions = _positions(positions, len(self))
        return ClientBatch(self.__x[positions].copy(), self.__y[positions].copy(),
                           self.__origins[positions].copy(), self.__rows[positions].copy())


class TrainingClients:
    """Coordinator-owned registry. Pass client() results, not this registry, to workers."""
    def __init__(self, context, arrays, partitions, weather):
        self.__context = context
        self.__arrays = arrays
        self.partitions = partitions
        self.__lookup = {part.client_id: part for part in partitions}
        self.weather = weather
        self.contract_sha256 = context.contract_sha256
        self.scalers = context.scalers
        self.node_names = tuple(str(name) for name in context.names)
        self.baseline_coefficients = context.coefficients
        self.num_windows = len(arrays[1])
        self.training_counts = MappingProxyType({part.client_id: part.num_examples for part in partitions})

    @property
    def client_ids(self):
        return tuple(self.__lookup)

    def client(self, client_id):
        if not isinstance(client_id, str) or client_id not in self.__lookup:
            raise KeyError(f'Unknown frozen client: {client_id!r}')
        return ClientTrainDataset._from_training(self.__lookup[client_id], self.__arrays, self.weather,
                                                self.scalers, self.baseline_coefficients, len(self.node_names))


def load_training_clients(directory, *, expected_contract_sha256, weather='masked',
                          minimum_windows=1024, minimum_days=7):
    """Open only training and shared metadata; never open validation or test arrays."""
    if weather not in ('enabled', 'masked'):
        raise ValueError('weather must be enabled or masked')
    context = _FrozenContext(directory, expected_contract_sha256)
    arrays = context.split_arrays('train')
    partitions = build_client_index(arrays[3], arrays[2], context.names, train_end_utc=context.train_end,
                                    grid_start_utc=context.grid_start, minimum_windows=minimum_windows,
                                    minimum_days=minimum_days)
    return TrainingClients(context, arrays, partitions, weather)


class CoordinatorValidationData:
    """Coordinator-only view compatible with the common evaluator's val interface."""
    def __init__(self, context, arrays):
        self.__arrays = arrays
        self.names = tuple(str(name) for name in context.names)
        self.meta = _freeze(context.meta)
        self.contract = _freeze(context.contract)
        self.contract_sha256 = context.contract_sha256

    def split(self, split):
        if split != 'val':
            raise PermissionError('CoordinatorValidationData exposes validation only')
        return self.__arrays


def load_coordinator_validation(directory, *, expected_contract_sha256):
    """Explicit separate coordinator capability; no validation references in workers."""
    context = _FrozenContext(directory, expected_contract_sha256)
    arrays = context.split_arrays('val')
    _, _, origins, nodes = arrays
    if (origins < _utc_time(context.train_end)).any() or (origins > _utc_time(context.val_end) - np.timedelta64(60, 'm')).any():
        raise ValueError('Validation target crosses its chronological boundaries')
    # Enforce per-node unique chronological IDs, without imposing training thresholds.
    build_client_index(nodes, origins, context.names, train_end_utc=context.val_end,
                       minimum_windows=1, minimum_days=1)
    return CoordinatorValidationData(context, arrays)
