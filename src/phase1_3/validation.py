"""Reject schema/time defects; invalidate physical readings without imputation."""
import numpy as np
from .schema import Telemetry, WindowBatch, UNITS, MASKED, WIDTH_NS, frozen_array

def sanitize(record, spec):
    if record.source != "synthetic" or spec.profile != "synthetic":
        raise PermissionError("Phase 1.3 accepts synthetic telemetry only")
    if not isinstance(record.node, str) or not record.node:
        raise ValueError("Nonempty node string required")
    t, v, observed = np.asarray(record.times), np.asarray(record.values), np.asarray(record.observed)
    if t.ndim != 1 or t.dtype != np.dtype("datetime64[ns]") or np.isnat(t).any():
        raise ValueError("Explicit datetime64[ns] UTC bin starts required")
    ns = t.view("i8")
    if ((ns-spec.start) % spec.width).any() or (ns < spec.start).any() or (ns >= spec.end).any():
        raise ValueError("Out-of-range or off-grid timestamp")
    if (np.diff(ns) <= 0).any():
        raise ValueError("Duplicate or unordered timestamp")
    if v.shape != (len(t), 13) or v.dtype.kind not in "fi" or observed.shape != v.shape or observed.dtype != np.bool_:
        raise ValueError("Expected numeric [T,13] features and boolean observed mask")
    if tuple(record.units) != UNITS[:-1]:
        raise ValueError("Feature unit/order mismatch")
    clean = np.array(v, dtype=np.float64, copy=True)
    good = np.isfinite(clean) & observed
    good[:, 0] &= clean[:, 0] >= 0
    for k in (1,2,3,4,8):
        good[:, k] &= (clean[:, k] >= 0) & (clean[:, k] <= 100)
    good[:,5] &= (clean[:,5] >= 0) & (clean[:,5] <= 1)
    for k in (6,7):
        good[:,k] &= clean[:,k] >= -273.15  # physical lower bound; no guessed site upper cutoff
    for k in (9,10,11,12):
        good[:,k] &= (clean[:,k] >= -1) & (clean[:,k] <= 1)
    clean[~good] = np.nan
    result = Telemetry(record.node, frozen_array(t), frozen_array(clean), frozen_array(good),
                       tuple(record.units), record.source)
    return result, {"missing_timestamp_bins":int(np.maximum(np.diff(ns)//spec.width-1,0).sum()),
                    "invalid_or_missing_feature_values":int((~good).sum())}

def validate_batch(batch, spec):
    if not isinstance(batch, WindowBatch) or batch.source != "synthetic" or spec.profile != "synthetic":
        raise PermissionError("Only synthetic windows are accepted")
    if batch.spec_digest != spec.digest:
        raise ValueError("Window/spec identity mismatch")
    n = len(batch)
    if (batch.x.shape != (n,36,14) or batch.y.shape != (n,2) or batch.nodes.shape != (n,)
            or batch.origins.shape != (n,) or batch.history_times.shape != (n,36)
            or batch.target_times.shape != (n,6)):
        raise ValueError("Window shape mismatch")
    if len(set(batch.mapping)) != len(batch.mapping) or not batch.mapping:
        raise ValueError("Node mapping must be nonempty and unique")
    if not all(isinstance(v,str) and v for v in batch.mapping):
        raise ValueError("Node mapping requires strings")
    if batch.nodes.dtype.kind not in "iu" or (batch.nodes < 0).any() or (batch.nodes >= len(batch.mapping)).any():
        raise ValueError("Invalid node index")
    if batch.origins.dtype != np.dtype("datetime64[ns]") or np.isnat(batch.origins).any():
        raise ValueError("Invalid origin timestamps")
    o = batch.origins.view("i8")
    if ((o-spec.start) % (spec.stride*spec.width)).any():
        raise ValueError("Origin is not on the fixed hourly grid")
    lo, hi = spec.bounds(batch.split)
    if (o-36*spec.width < lo).any() or (o+6*spec.width > hi).any():
        raise ValueError("Whole window must stay inside its split")
    h = o[:,None] + np.arange(-36,0)*spec.width
    t = o[:,None] + np.arange(6)*spec.width
    if batch.history_times.dtype != np.dtype("datetime64[ns]") or batch.target_times.dtype != np.dtype("datetime64[ns]"):
        raise ValueError("Time array dtype mismatch")
    if not np.array_equal(batch.history_times.view("i8"),h) or not np.array_equal(batch.target_times.view("i8"),t):
        raise ValueError("Noncontiguous history or future leakage")
    keys = list(zip(batch.nodes.tolist(),o.tolist()))
    if keys != sorted(set(keys)):
        raise ValueError("Duplicate or unordered window identity")
    if not np.isfinite(batch.x).all() or not np.isfinite(batch.y).all():
        raise ValueError("Nonfinite window tensor")
    if not np.array_equal(batch.x[:,:,13], np.broadcast_to(batch.nodes[:,None],(n,36))):
        raise ValueError("Node identity channel differs")
    if (batch.x[:,:,list(MASKED)] != 0).any():
        raise ValueError("Masked-weather invariant differs")
    if (batch.y[:,0] < 0).any() or (batch.y[:,1] < -273.15).any():
        raise ValueError("Invalid physical targets")
    return True
