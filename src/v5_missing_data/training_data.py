"""Read-only v5 capabilities; train workers never receive file paths or eval bins."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from src.history_data_v4 import inside, sha256
from src.experiment_data import FEATURES
from src.v5_missing_data.build_training_cohorts import read_json, id_dtype

POLICIES = ('no_fill', 'ff10', 'ff20', 'ff30')
GAPS = dict(zip(POLICIES, (0, 1, 2, 3)))
HISTORY, HORIZON = 36, 6


def readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class CohortContext:
    """Coordinator metadata and byte-hash verification; no np.load of examples."""
    def __init__(self, directory, expected_receipt_sha256, *, verify=True):
        self.path = Path(directory).resolve()
        if sha256(self.path / 'completed.json') != expected_receipt_sha256:
            raise ValueError('Cohort completion receipt changed')
        self.receipt_sha256 = expected_receipt_sha256
        self.receipt = read_json(self.path / 'completed.json')
        if self.receipt['status'] != 'complete':
            raise ValueError('Cohort preparation is incomplete')
        if verify:
            self.verify_bytes()
        self.contract = read_json(self.path / 'contract.json')
        self.names = tuple(read_json(self.path / 'node_mapping.json'))
        self.scalers = read_json(self.path / 'scalers.json')
        self.budgets = read_json(self.path / 'client_budgets.json')
        c = self.contract
        if c['history_steps'] != 36 or c['horizon_steps'] != 6 or c['feature_names'] != FEATURES:
            raise ValueError('Expected unchanged fourteen-channel, six-hour v5 cohort')
        if len(set(self.names)) != len(self.names) or len(self.names) != len(self.budgets['clients']):
            raise ValueError('Invalid frozen node population')
        for p in POLICIES:
            if (c['policy_training_views'][p] != f'native/{p}/train'
                    or c['policy_validation_views'][p] != 'common/all_policies/val'
                    or c['policy_test_views'][p] != 'common/all_policies/test'):
                raise ValueError('Native training/common evaluation binding changed')
        for i, client in enumerate(self.budgets['clients']):
            if (client['node_index'] != i or client['node'] != self.names[i]
                    or client['m_k'] != c['cohorts']['native/no_fill/train']['per_node_counts'][i]
                    or any(client['native_training_counts'][p] != c['cohorts'][f'native/{p}/train']['per_node_counts'][i]
                           or client['m_k'] > client['native_training_counts'][p] for p in POLICIES)):
                raise ValueError('Frozen native counts/reference budgets/node mapping differ')
        for key in ('x_mean', 'x_std', 'y_mean', 'y_std'):
            a = np.asarray(self.scalers[key])
            if a.shape != ((14,) if key.startswith('x') else (1,)) or not np.isfinite(a).all() or ('std' in key and (a <= 0).any()):
                raise ValueError('Invalid immutable scaler')
        if self.scalers['x_mean'][13] != 0 or self.scalers['x_std'][13] != 1:
            raise ValueError('Node identity must remain categorical')
        self.coefficients = (self.scalers['x_std'][0] / self.scalers['y_std'][0],
            (self.scalers['x_mean'][0] - self.scalers['y_mean'][0]) / self.scalers['y_std'][0])

    def verify_bytes(self):
        actual = {str(p.relative_to(self.path)) for p in self.path.rglob('*')
                  if p.is_file() and p != self.path / 'completed.json'}
        if actual != set(self.receipt['artifacts']):
            raise ValueError('Cohort artifact inventory changed')
        for name, expected in self.receipt['artifacts'].items():
            if sha256(inside(self.path, name)) != expected:
                raise ValueError('Cohort artifact changed: ' + name)
        return len(actual)


class SplitData:
    """One explicitly chosen split capability. No fallback to another split."""
    def __init__(self, context, split, policy='no_fill'):
        if split not in ('train', 'val', 'test') or policy not in POLICIES:
            raise ValueError('Unknown split/policy')
        self.context, self.split_name, self.policy = context, split, policy
        c = context.contract
        self.view = f'native/{policy}/train' if split == 'train' else f'common/all_policies/{split}'
        self.spec = c['cohorts'][self.view]
        info = c['bin_contexts'][split]
        self.info = info
        def load(path):
            # These paths come solely from the explicit split capability.
            return np.load(inside(context.path, path), mmap_mode='r', allow_pickle=False)
        self.ids = load(self.spec['paths']['ids'])
        self.targets = load(self.spec['paths']['targets_scaled'])
        self.raw_targets = load(self.spec['paths']['targets_kw'])
        self.features = load(info['features'])
        self.ages = load(info['ages'])
        self.power = load(info['observed_power'])
        if (self.ids.dtype != id_dtype(context.names) or self.ids.shape != (self.spec['count'],)
                or self.targets.shape != (len(self.ids), 1) or self.targets.dtype != np.float32
                or self.raw_targets.shape != self.targets.shape or self.raw_targets.dtype != np.float32):
            raise ValueError('Invalid frozen cohort array schema')

    def __len__(self):
        return len(self.ids)

    def batch(self, positions):
        positions = np.asarray(positions)
        if positions.ndim != 1 or positions.dtype.kind not in 'iu' or (positions < 0).any() or (positions >= len(self)).any():
            raise ValueError('Invalid cohort row positions')
        ids = np.array(self.ids[positions], copy=True)
        nodes = ids['node_index'].astype(np.int64)
        c = self.context.contract
        origins = ids['origin'].view('i8')
        if (np.isnat(ids['origin']).any() or ((origins - c['grid_start_ns']) % c['width_ns']).any()
                or (nodes < 0).any() or (nodes >= len(self.context.names)).any()):
            raise ValueError('Malformed frozen window identity')
        if not np.array_equal(np.asarray(self.context.names)[nodes], ids['node']):
            raise ValueError('Node string/index identity mismatch')
        global_bins = (origins - c['grid_start_ns']) // c['width_ns']
        lo, hi = c['boundaries'][self.split_name]
        if (global_bins < lo).any() or (global_bins < HISTORY).any() or (global_bins + HORIZON > hi).any():
            raise ValueError('Target/history crosses frozen split boundaries')
        target_bins = global_bins[:, None] - lo + np.arange(HORIZON)
        if not np.isfinite(self.power[nodes[:, None], target_bins]).all():
            raise ValueError('Future target power is not observed')
        raw_y = np.array(self.raw_targets[positions], copy=True)
        y = np.array(self.targets[positions], copy=True)
        expected = ((raw_y.astype(np.float64) - self.context.scalers['y_mean'][0]) /
                    self.context.scalers['y_std'][0]).astype(np.float32)
        if not np.isfinite(y).all() or y.tobytes() != expected.tobytes():
            raise ValueError('Stored target/scaler provenance mismatch')
        times = global_bins[:, None] - self.info['feature_start'] + np.arange(-HISTORY, 0)
        raw = np.array(self.features[nodes[:, None], times], copy=True)
        age = self.ages[nodes[:, None], times]
        gap = GAPS[self.policy] if self.split_name == 'train' else 0
        if gap:
            for channel in range(13):
                if channel in (7, 8):
                    continue
                fill = (age[:, :, channel] > 0) & (age[:, :, channel] <= gap)
                rows, steps = np.where(fill)
                source_local = times[rows, steps] - age[rows, steps, channel]
                source_global = source_local + self.info['feature_start']
                destination_global = times[rows, steps] + self.info['feature_start']
                source_split = np.searchsorted([c['boundaries']['val'][0], c['boundaries']['test'][0]], source_global, side='right')
                dest_split = np.searchsorted([c['boundaries']['val'][0], c['boundaries']['test'][0]], destination_global, side='right')
                if (source_local < 0).any() or (source_global >= destination_global).any() or not np.array_equal(source_split, dest_split):
                    raise ValueError('Noncausal or cross-boundary fill source')
                raw[rows, steps, channel] = self.features[nodes[rows], source_local, channel]
        raw = np.concatenate((raw, np.broadcast_to(nodes[:, None, None], (len(nodes), HISTORY, 1))), axis=2).astype(np.float32)
        x = ((raw - np.asarray(self.context.scalers['x_mean'])) /
             np.asarray(self.context.scalers['x_std'])).astype(np.float32)
        x[:, :, [7, 8]] = 0
        if not np.isfinite(x).all() or not np.array_equal(x[:, :, 13], np.broadcast_to(nodes[:, None], (len(nodes), HISTORY))):
            raise ValueError('Invalid inputs after causal fill/mask/scaling')
        return x, y, ids

    def require_identity(self, ids):
        if ids.shape != self.ids.shape or ids.dtype != self.ids.dtype or ids.tobytes() != self.ids.tobytes():
            raise ValueError('Evaluation window identities/order differ from locked common cohort')


@dataclass(frozen=True)
class LocalData:
    client_id: str
    node_index: int
    inputs: np.ndarray
    targets: np.ndarray
    identities: np.ndarray
    row_positions: np.ndarray

    def batches(self, size):
        for first in range(0, len(self.targets), size):
            yield self.inputs[first:first+size].copy(), self.targets[first:first+size].copy()


class TrainingRegistry:
    """Coordinator-owned train capability; pass only LocalData to a worker."""
    def __init__(self, context, policy):
        self.data = SplitData(context, 'train', policy)
        self.context, self.policy = context, policy
        self.names = context.names
        self.counts = tuple(self.data.spec['per_node_counts'])
        self.offsets = np.r_[0, np.cumsum(self.counts)]
        self.m = tuple(int(x['m_k']) for x in context.budgets['clients'])
        for i, client in enumerate(context.budgets['clients']):
            if (client['node_index'] != i or client['node'] != self.names[i]
                    or not 0 < self.m[i] <= self.counts[i]):
                raise ValueError('Client sampling budget/mapping mismatch')

    def local(self, node_index, rows):
        if not 0 <= node_index < len(self.names):
            raise ValueError('Unknown node')
        rows = np.asarray(rows)
        if rows.ndim != 1 or rows.dtype.kind not in 'iu':
            raise ValueError('Sample rows must be integer positions')
        rows = rows.astype(np.int64)
        if (len(rows) != self.m[node_index] or len(np.unique(rows)) != len(rows)
                or (rows < self.offsets[node_index]).any() or (rows >= self.offsets[node_index+1]).any()):
            raise ValueError('Sample count, uniqueness or client membership mismatch')
        x, y, ids = self.data.batch(rows)
        if not np.all(ids['node_index'] == node_index):
            raise ValueError('Cross-client training rows')
        return LocalData(self.names[node_index], node_index, readonly(x), readonly(y), readonly(ids), readonly(rows))
