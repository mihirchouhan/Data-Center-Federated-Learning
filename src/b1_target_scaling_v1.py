"""Supplied two-output scaling only: no fitting, data loading or file I/O."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

TARGET_NAMES = ('next_hour_mean_node_it_power_w',
                'next_hour_mean_node_inlet_temperature_c')
TARGET_UNITS = ('W', 'degC')


def _pair(value, name):
    if np.ma.isMaskedArray(value):
        raise ValueError(name + ' cannot be masked')
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind not in 'iuf' or any(
            isinstance(v, (bool, np.bool_)) for v in value):
        raise ValueError(name + ' must contain two real numeric constants')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError(name + ' must be finite')
    return tuple(float(v) for v in a)


def _matrix(value):
    if np.ma.isMaskedArray(value):
        raise ValueError('Masked target matrices are not supported')
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or a.dtype.kind not in 'iuf':
        raise ValueError('Nonempty real numeric [B,2] targets required')
    a = a.astype(np.float64)
    if not np.isfinite(a).all():
        raise ValueError('Finite target matrix required')
    return a


def _immutable(a):
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class TargetScaling:
    """Immutable supplied means/stds in W and degC, not fitted statistics.

    Real B1 scaling-population/provenance approval belongs to a later data bridge.
    This object does not assert that the caller's constants were fitted correctly.
    """
    mean: tuple
    std: tuple

    def __post_init__(self):
        object.__setattr__(self, 'mean', _pair(self.mean, 'mean'))
        object.__setattr__(self, 'std', _pair(self.std, 'std'))
        if any(v < 1e-12 for v in self.std):
            raise ValueError('Constant/invalid target scale: stop for review')

    def standardize(self, physical):
        """Physical [W,degC] -> independent dimensionless target coordinates."""
        a = _matrix(physical)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            out = (a - self.mean) / self.std
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite standardized targets')
        return _immutable(out)

    def inverse(self, standardized):
        """Inverse once into W/degC; there is no additional kW-to-W factor."""
        a = _matrix(standardized)
        with np.errstate(over='raise', invalid='raise'):
            out = a * self.std + self.mean
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite inverse targets')
        return _immutable(out)

    def record(self):
        return {'schema': 'b1.supplied.target.scaling.v1',
                'target_names': list(TARGET_NAMES), 'target_units': list(TARGET_UNITS),
                'mean': list(self.mean), 'std': list(self.std),
                'fit_performed_by_this_object': False}

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
