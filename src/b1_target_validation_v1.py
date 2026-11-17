"""Versioned B1 target validation; historical Phase 2 contracts stay unchanged.

Future B1 callers must explicitly import this helper. It does not implement or
connect a model/data adapter, and it never loads files or fits statistics.
"""
import numpy as np

from src.phase2_multioutput.contracts import target_means as _historical_target_means


def target_means(future, observed):
    """Return [N,2] next-hour means in W, then recorded ambient degrees C.

    ``future`` contains six original power-kW/temperature-C bins per window.
    Reject every MaskedArray before conversion, even an unmasked instance: its
    mask must never be silently discarded. Plain boolean *observation flags*
    remain required, but boolean target values are not physical measurements.
    All existing numeric shape, observation, finiteness, physical-range and
    aggregation rules are delegated unchanged to the historical pure helper.
    """
    if np.ma.isMaskedArray(future):
        raise ValueError('Masked-array targets are not supported')
    if np.ma.isMaskedArray(observed):
        raise ValueError('Masked-array target observation flags are not supported')
    values = np.asarray(future)
    if values.dtype.kind == 'b':
        raise ValueError('Boolean target arrays are not supported')
    return _historical_target_means(values, observed)
