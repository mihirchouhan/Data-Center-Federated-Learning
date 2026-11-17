"""B1 in-memory observed-window bridge. No file loader, models, or test capability.

Raw preparation remains upstream: callers supply coverage-qualified original bins
and source evidence, not historical scalar targets or already-standardized inputs.
This API validates evidence but cannot authenticate an upstream producer's claims.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import numpy as np
from src import b1_target_validation_v1 as target_validation
from src.phase1_3.schema import FEATURES, UNITS, frozen_array
from src.phase2_multioutput.contracts import Windows, validate_windows, identity_digest

HISTORY, HORIZON, WIDTH_NS = 36, 6, 600_000_000_000
ORIGINAL_NAMES = tuple(str(i) for i in range(1, 980))
MAPPING_DIGEST = '6a763c3be11c4e40a10a7f6dea7dcc28a9c01c8e729b09f21642ca572a22ee87'
SPLITS = MappingProxyType({'train': ('2022-01-01T00:00:00', '2022-01-28T00:00:00'),
                          'val': ('2022-01-28T00:00:00', '2022-02-04T00:00:00')})
_PROTOCOL = {'splits': {k: tuple(v+'Z' for v in bounds) for k,bounds in SPLITS.items()}}
_TOKEN = object()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def initialization_record(root_seed):
    """Record the newly specified B1 convention; never initialize a model."""
    if type(root_seed) is not int or root_seed not in (42, 43, 44):
        raise ValueError('B1 root seed must be 42,43,44')
    coordinates = [root_seed, 0, 0, 0]
    derived = int(np.random.SeedSequence(coordinates).generate_state(1)[0])
    return {'schema': 'b1.initialization.stream.v1', 'root_seed': root_seed,
            'coordinates': coordinates, 'initialization_seed': derived,
            'numpy_version': np.__version__, 'historical_reproducibility_claim': False}


def _numeric(value, label, shape=None):
    if np.ma.isMaskedArray(value):
        raise ValueError(label + ' cannot be a MaskedArray, even without masked cells')
    a = np.asarray(value)
    if a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(label + ' must be finite real numeric values, not boolean')
    if shape is not None and a.shape != shape:
        raise ValueError(label + ' shape differs')
    return a


def _times(value, label, shape):
    if np.ma.isMaskedArray(value):
        raise ValueError(label + ' cannot be masked')
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.dtype('datetime64[ns]') or np.isnat(a).any():
        raise ValueError(label + ' requires complete UTC datetime64[ns] values')
    return a


def _evidence(value):
    if not isinstance(value, dict) or set(value) != {'source_kind', 'raw_contract_sha256',
            'bin_contract_sha256', 'producer_source_sha256'}:
        raise ValueError('Explicit raw/bin/producer provenance required')
    if value['source_kind'] not in ('synthetic', 'observed_prepared'):
        raise ValueError('Unknown provenance kind')
    for key in ('raw_contract_sha256', 'bin_contract_sha256', 'producer_source_sha256'):
        h = value[key]
        if type(h) is not str or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Invalid provenance SHA-256: ' + key)
    return json.loads(canonical(value))


def _source_bounds(earliest, latest, starts, lo, shape, label, weather=False):
    first = _times(earliest, label + ' earliest', shape).view('i8')
    last = _times(latest, label + ' latest', shape).view('i8')
    if (first < lo).any() or (first > last).any() or (last >= starts + WIDTH_NS).any():
        raise ValueError(label + ' future/cross-split/source ordering violation')
    if weather:
        if not np.array_equal(first, last) or (starts + WIDTH_NS - 1 - last > 1800_000_000_000).any():
            raise ValueError('Invalid weather as-of source or age')
    elif (first < starts - 300_000_000_000).any():
        raise ValueError(label + ' source exceeds the maximum causal hold')


def _support(value, shape, label):
    a = _numeric(value, label, shape)
    if (a < .5).any() or (a > 1).any():
        raise ValueError(label + ' needs 50 to 100 percent bin support')
    return frozen_array(a, np.float64)


class _ObservedWindows:
    __slots__ = ('__w', '__physical', '__evidence', '__extras', '__content_digest')
    def __init__(self, token, w, physical, evidence, extras):
        if token is not _TOKEN:
            raise TypeError('Use accept_windows to validate original containers')
        object.__setattr__(self, '_ObservedWindows__w', w)
        object.__setattr__(self, '_ObservedWindows__physical', physical)
        object.__setattr__(self, '_ObservedWindows__evidence', canonical(evidence))
        object.__setattr__(self, '_ObservedWindows__extras', MappingProxyType(extras))
        h = hashlib.sha256(canonical({'split': w.split, 'mapping': list(w.mapping),
                                     'evidence': evidence}))
        for key in ('x', 'observed', 'future', 'target_observed', 'history_times', 'target_times',
                    'history_nodes', 'target_nodes', 'source_earliest', 'source_latest', 'origins', 'nodes'):
            a = getattr(w, key); h.update(canonical([key, str(a.dtype), list(a.shape)])); h.update(a.tobytes())
        for key, a in sorted(extras.items()):
            h.update(canonical([key, str(a.dtype), list(a.shape)])); h.update(a.tobytes())
        object.__setattr__(self, '_ObservedWindows__content_digest', h.hexdigest())
    def __setattr__(self, name, value):
        raise AttributeError('Validated window capabilities are immutable')
    def __len__(self): return len(self.__w.nodes)
    @property
    def split(self): return self.__w.split
    @property
    def mapping(self): return self.__w.mapping
    @property
    def x(self): return self.__w.x
    @property
    def physical_targets(self): return self.__physical
    @property
    def future(self): return self.__w.future
    @property
    def target_observed(self): return self.__w.target_observed
    @property
    def nodes(self): return self.__w.nodes
    @property
    def origins(self): return self.__w.origins
    @property
    def identity_sha256(self): return identity_digest(self.nodes, self.origins)
    @property
    def target_sha256(self): return hashlib.sha256(self.physical_targets.astype('<f8').tobytes()).hexdigest()
    @property
    def content_sha256(self): return self.__content_digest
    @property
    def source_evidence(self): return json.loads(self.__evidence)
    def canonical_chunks(self, size):
        if type(size) is not int or size < 1:
            raise ValueError('Positive integer chunk size required')
        for first in range(0, len(self), size):
            yield self.x[first:first+size], self.physical_targets[first:first+size]
    def record(self):
        return {'split': self.split, 'windows': len(self), 'identity_sha256': self.identity_sha256,
                'target_sha256': self.target_sha256, 'content_sha256': self.content_sha256,
                'mapping_digest': MAPPING_DIGEST, 'source_evidence': self.source_evidence}


class TrainingWindows(_ObservedWindows):
    __slots__ = ()


class ValidationWindows(_ObservedWindows):
    __slots__ = ()


def accept_windows(*, split, mapping, x, observed, future, target_observed,
        nodes, origins, history_times, target_times, history_nodes, target_nodes,
        source_earliest, source_latest, target_source_earliest, target_source_latest,
        history_coverage, target_coverage, source_evidence, timezone='UTC',
        features=FEATURES, units=UNITS):
    """Validate physical, already-windowed ORIGINAL bins; no data or model I/O.

    Targets and their flags are checked by the hardened B1 helper before any
    mask-erasing constructor/conversion. There is deliberately no test split.
    Coverage/source fields are upstream evidence, not independent raw replay.
    """
    if split not in SPLITS:
        raise PermissionError('Only B1 train/val; March/test is inaccessible')
    # Deny March/out-of-scope dates before touching any target payload, even
    # when a caller falsely labels those dates as train or validation.
    if timezone != 'UTC' or np.ma.isMaskedArray(origins):
        raise ValueError('Explicit complete UTC origins required')
    origin_preview = np.asarray(origins)
    if origin_preview.ndim != 1 or not len(origin_preview):
        raise ValueError('Nonempty one-dimensional UTC origins required')
    _times(origin_preview, 'origins', (len(origin_preview),))
    o = origin_preview.view('i8')
    lo_ns,hi_ns = (int(np.datetime64(v, 'ns').view('i8')) for v in SPLITS[split])
    if ((o % (6*WIDTH_NS)).any() or (o-36*WIDTH_NS < lo_ns).any() or
            (o+6*WIDTH_NS > hi_ns).any()):
        raise ValueError('B1 whole-window split leakage or off-hour origin')
    physical = target_validation.target_means(future, target_observed)
    _numeric(future, 'future')  # Explicitly reject unsupported/object/complex values.
    n = len(physical)
    if isinstance(mapping, (str, bytes)) or tuple(mapping) != ORIGINAL_NAMES:
        raise ValueError('Original 979 literal node ordinals must remain unchanged')
    if timezone != 'UTC' or tuple(features) != FEATURES or tuple(units) != UNITS:
        raise ValueError('UTC/features/units must match the B1 schema')
    _numeric(x, 'physical inputs', (n, 36, 14))
    for label, value in [('observed', observed), ('nodes', nodes), ('origins', origins),
            ('history_times', history_times), ('target_times', target_times),
            ('history_nodes', history_nodes), ('target_nodes', target_nodes),
            ('source_earliest', source_earliest), ('source_latest', source_latest)]:
        if np.ma.isMaskedArray(value): raise ValueError(label + ' cannot be masked')
    w = Windows(x=np.asarray(x, dtype=np.float64), observed=observed,
        future=future, target_observed=target_observed, history_times=history_times,
        target_times=target_times, history_nodes=history_nodes, target_nodes=target_nodes,
        source_earliest=source_earliest, source_latest=source_latest,
        nodes=nodes, origins=origins, mapping=ORIGINAL_NAMES, split=split, history=36,
        timezone=timezone, features=tuple(features), units=tuple(units))
    checked = validate_windows(w, _PROTOCOL)
    seconds = (w.history_times.view('i8') + WIDTH_NS) / 1_000_000_000
    daily = 2*np.pi*(seconds % 86400)/86400
    weekly = 2*np.pi*((seconds+259200) % 604800)/604800
    calendar = np.stack((np.sin(daily),np.cos(daily),np.sin(weekly),np.cos(weekly)),axis=2)
    if not np.allclose(w.x[:,:,9:13],calendar,rtol=1e-12,atol=1e-12):
        raise ValueError('Calendar channels differ from UTC bin-right-edge definitions')
    if not np.array_equal(checked, physical): raise ValueError('Target conversion mismatch')
    lo = int(np.datetime64(SPLITS[split][0], 'ns').view('i8'))
    hs = w.history_times.view('i8')[:, :, None]; ts = w.target_times.view('i8')[:, :, None]
    _source_bounds(w.source_earliest[:, :, :7], w.source_latest[:, :, :7], hs, lo, (n,36,7), 'history')
    _source_bounds(w.source_earliest[:, :, 7:], w.source_latest[:, :, 7:], hs, lo, (n,36,2), 'weather', True)
    _source_bounds(target_source_earliest, target_source_latest, ts, lo, (n,6,2), 'target')
    extra = {'history_coverage': _support(history_coverage, (n,36,7), 'history coverage'),
             'target_coverage': _support(target_coverage, (n,6,2), 'target coverage'),
             'target_source_earliest': frozen_array(target_source_earliest),
             'target_source_latest': frozen_array(target_source_latest)}
    cls = TrainingWindows if split == 'train' else ValidationWindows
    return cls(_TOKEN, w, physical, _evidence(source_evidence), extra)


def training_eligibility(training):
    if type(training) is not TrainingWindows:
        raise PermissionError('Eligibility requires validated training-only windows')
    out = {}
    for i, name in enumerate(ORIGINAL_NAMES):
        keep = training.nodes == name; count = int(keep.sum())
        days = len(np.unique(training.origins[keep].astype('datetime64[D]')))
        if count >= 24 and days >= 7: out[name] = {'node_index': i, 'windows': count, 'positive_dates': days}
    return out


@dataclass(frozen=True)
class BridgeBatch:
    x: np.ndarray                 # standardized float32 [B,36,14]
    y: np.ndarray                 # physical float64 [B,2] W/degC
    y_standardized: np.ndarray    # float32 [B,2]
    future: np.ndarray            # ORIGINAL raw kW/degC [B,6,2]
    target_observed: np.ndarray
    nodes: np.ndarray
    origins: np.ndarray
    positions: np.ndarray
    def cpu_tensors(self):
        """Copies to CPU tensors only; no model, autograd or optimizer call."""
        import torch
        return {'x': torch.from_numpy(np.array(self.x, copy=True)),
                'y': torch.from_numpy(np.array(self.y, copy=True)),
                'y_standardized': torch.from_numpy(np.array(self.y_standardized, copy=True))}


class ClientDataset:
    """A copied single-client capability with no parent or other-split reference."""
    __slots__ = ('__batch', '__split', '__client', '__ordinal')
    def __init__(self, token, batch, split, client):
        if token is not _TOKEN: raise TypeError('Use training/validation dataset factories')
        self.__batch, self.__split, self.__client = batch, split, client
        self.__ordinal = ORIGINAL_NAMES.index(client)
    def __len__(self): return len(self.__batch.y)
    @property
    def split(self): return self.__split
    @property
    def client_id(self): return self.__client
    @property
    def node_index(self): return self.__ordinal
    def batch(self, positions):
        p = np.asarray(positions)
        if p.ndim != 1 or p.dtype.kind not in 'iu' or not len(p) or (p<0).any() or (p>=len(self)).any() or len(np.unique(p)) != len(p):
            raise ValueError('Client-local integer positions required')
        return BridgeBatch(*(frozen_array(getattr(self.__batch, k)[p]) for k in BridgeBatch.__dataclass_fields__))


class _Dataset:
    def __init__(self, token, windows, scaler, allowed):
        if token is not _TOKEN: raise TypeError('Use dataset factories')
        self._windows, self._scaler = windows, scaler
        self._allowed = frozenset(allowed)
    @property
    def clients(self): return tuple(n for n in ORIGINAL_NAMES if n in self._allowed)
    def client(self, name):
        if name not in self._allowed: raise KeyError('Client unavailable/ineligible in this capability')
        w = self._windows; p = np.flatnonzero(w.nodes == name)
        if not len(p): raise ValueError('No windows for this validation client; never substitute training rows')
        batch = BridgeBatch(self._scaler.transform_inputs(w.x[p]), frozen_array(w.physical_targets[p]),
            self._scaler.standardize_targets(w.physical_targets[p]), frozen_array(w.future[p]),
            frozen_array(w.target_observed[p]), frozen_array(w.nodes[p]), frozen_array(w.origins[p]), frozen_array(p))
        return ClientDataset(_TOKEN, batch, w.split, name)


class TrainingDataset(_Dataset):
    """Coordinator training registry; pass only client() copies to local workers."""


class CoordinatorValidationDataset(_Dataset):
    """Coordinator-only validation; never pass this capability to a local worker."""


def make_datasets(training, validation, scaler):
    from src.b1_bridge_scaling_v1 import FrozenScaler
    if type(training) is not TrainingWindows or type(validation) is not ValidationWindows:
        raise PermissionError('Separate validated training and validation capabilities required')
    if not isinstance(scaler, FrozenScaler): raise ValueError('Frozen B1 scaler required')
    scaler.require_training(training)
    eligible = training_eligibility(training)
    if set(training.nodes.tolist()) != set(eligible):
        raise ValueError('Ineligible training rows must not be silently filtered')
    if not set(validation.nodes.tolist()) <= set(eligible):
        raise ValueError('Validation cannot admit a client missing training eligibility')
    return (TrainingDataset(_TOKEN, training, scaler, eligible),
            CoordinatorValidationDataset(_TOKEN, validation, scaler, set(validation.nodes.tolist())))
