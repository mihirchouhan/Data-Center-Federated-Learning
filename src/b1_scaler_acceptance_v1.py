"""B1 CPU scaler acceptance: IEEE pointwise bounds, no fitting or file I/O.

The original 2e-6 diagnostic remains in b1_bridge_scaling_v1 unchanged.
This module verifies the declared float64 -> float32 -> float64 operation
sequence, not model quality, arbitrary devices, or an experiment approval.
"""
import numpy as np
from src import b1_bridge_scaling_v1 as scaling
from src import b1_real_data_bridge_v1 as bridge

RULE = 'b1.pointwise.ieee.rne.affine_quantization.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def environment_check():
    """Small probes of nearest-even and gradual underflow; no device probing."""
    require(np.finfo(np.float32).nmant == 23 and np.finfo(np.float64).nmant == 52,
            'IEEE binary32/binary64 required')
    require(np.float32(1. + 2.**-24) == np.float32(1.), 'Nearest-even cast required')
    require(np.float32(1. + 3*2.**-24) == np.float32(1. + 2.**-22), 'Odd tie cast differs')
    a = np.array(1., dtype=np.float64)
    require(np.add(a, np.float64(2.**-53)) == 1., 'Nearest-even binary64 required')
    tiny = np.nextafter(np.float32(0), np.float32(1))
    require(np.float32(float(tiny)*.75) == tiny, 'Float32 gradual underflow required')
    require(np.float32(float(tiny)*.5) == 0., 'Subnormal tie differs')
    smallest = np.nextafter(np.float64(0), np.float64(1))
    require(np.float64(np.finfo('f8').tiny)*.5 > 0. and smallest > 0.,
            'Float64 gradual underflow required')
    return {'rounding': 'nearest_even_probes_passed', 'gradual_underflow': True,
            'numpy_version': np.__version__, 'device_scope': 'CPU only'}


def half_cell(value, dtype):
    """Enclose half the larger neighboring gap; finite endpoint uses finite side.

    At a finite extreme this equals the virtual overflow-lattice half gap.
    Binary64 subnormal half-gaps round UP rather than disappear to zero.
    """
    dt = np.dtype(dtype)
    require(dt in (np.dtype('f4'), np.dtype('f8')), 'Only binary32/binary64 cells')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        a = np.asarray(value, dtype=dt)
    require(np.isfinite(a).all(), 'Nonfinite rounding-cell center')
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        hi = np.nextafter(a, np.array(np.inf, dtype=dt)).astype(np.float64)
        lo = np.nextafter(a, np.array(-np.inf, dtype=dt)).astype(np.float64)
        a64 = a.astype(np.float64)
        hi_gap = np.subtract(hi, a64, out=np.zeros_like(a64), where=np.isfinite(hi))
        lo_gap = np.subtract(a64, lo, out=np.zeros_like(a64), where=np.isfinite(lo))
        half = np.maximum(hi_gap, lo_gap)*.5
        if dt == np.dtype('f8'):
            half = np.nextafter(half, np.inf)
    require(np.isfinite(half).all() and (half > 0).all(), 'Invalid cell enclosure')
    return half


def up_add(a, b):
    require(np.isfinite(a).all() and np.isfinite(b).all() and
            (np.asarray(a) >= 0).all() and (np.asarray(b) >= 0).all(), 'Positive bound terms required')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        out = np.nextafter(np.add(a, b, dtype=np.float64), np.inf)
    require(np.isfinite(out).all(), 'Nonfinite bound sum')
    return out


def up_mul(a, b):
    require(np.isfinite(a).all() and np.isfinite(b).all() and
            (np.asarray(a) >= 0).all() and (np.asarray(b) >= 0).all(), 'Positive bound terms required')
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        out = np.nextafter(np.multiply(a, b, dtype=np.float64), np.inf)
    require(np.isfinite(out).all(), 'Nonfinite bound product')
    return out


def error_upper(a, b):
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        out = np.nextafter(np.abs(np.subtract(a, b, dtype=np.float64)), np.inf)
    require(np.isfinite(out).all(), 'Nonfinite residual')
    return out


def reference(x, mean, scale):
    """Declared separate operations and outward bounds, independent of scaler methods."""
    x = bridge._numeric(x, 'acceptance physical values').astype(np.float64)
    mean = bridge._numeric(mean, 'acceptance mean').astype(np.float64)
    scale = bridge._numeric(scale, 'acceptance scale').astype(np.float64)
    require((scale > 0).all(), 'Positive scales required')
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        t = np.subtract(x, mean, dtype=np.float64)
        z = np.divide(t, scale, dtype=np.float64)
        q = z.astype(np.float32)
        p = np.multiply(q.astype(np.float64), scale, dtype=np.float64)
        hat = np.add(p, mean, dtype=np.float64)
        p0 = np.multiply(z, scale, dtype=np.float64)
        hat0 = np.add(p0, mean, dtype=np.float64)
    require(all(np.isfinite(a).all() for a in (t, z, q, p, hat, p0, hat0)), 'Nonfinite affine operation')
    hs, hd, hq = half_cell(t, 'f8'), up_mul(scale, half_cell(z, 'f8')), half_cell(q, 'f4')
    arithmetic = up_add(up_add(hs, hd), up_add(half_cell(p, 'f8'), half_cell(hat, 'f8')))
    bound = up_add(arithmetic, up_mul(scale, hq))
    affine_bound = up_add(up_add(hs, hd), up_add(half_cell(p0, 'f8'), half_cell(hat0, 'f8')))
    return dict(t=t, z=z, q=q, p=p, hat=hat, p0=p0, hat0=hat0, h32=hq,
                bound=bound, affine_bound=affine_bound, arithmetic_bound=arithmetic)


def verify_affine(x, mean, scale, actual_q, actual_inverse):
    """Require exact declared operation bits AND pointwise error enclosures.

    A perturbed inverse is rejected even if it happens to fit in the bound.
    Returned arrays are diagnostics only; no empirical maxima enter acceptance.
    """
    x = bridge._numeric(x, 'physical values').astype(np.float64)
    aq = bridge._numeric(actual_q, 'actual float32 values')
    ai = bridge._numeric(actual_inverse, 'actual inverse')
    require(aq.shape == x.shape and aq.dtype == np.float32, 'Actual float32 shape/dtype differs')
    require(ai.shape == x.shape and ai.dtype == np.float64, 'Actual inverse shape/dtype differs')
    d = reference(x, mean, scale)
    require(np.array_equal(aq.view('u4'), d['q'].view('u4')), 'Forward operation bits differ')
    require(np.array_equal(ai.view('u8'), d['hat'].view('u8')), 'Inverse operation bits differ')
    # The binary64 subtraction is exact here (Sterbenz; q=0 is exact too).
    require((np.abs(aq.astype(np.float64)-d['z']) <= d['h32']).all(), 'Quantization cell violated')
    require((error_upper(ai, x) <= d['bound']).all(), 'Inverse representation bound violated')
    require((error_upper(d['hat0'], x) <= d['affine_bound']).all(), 'Float64 affine bound violated')
    require((np.abs(d['hat0']-x) <= scaling.FLOAT64_ATOL + scaling.FLOAT64_RTOL*np.abs(x)).all(),
            'Existing float64 affine criterion violated')
    d['quantization_residual'] = np.abs(aq.astype(np.float64)-d['z'])
    d['inverse_residual'] = np.abs(ai-x)
    d['affine_residual'] = np.abs(d['hat0']-x)
    d['historical_failures'] = d['inverse_residual'] > scaling.FLOAT32_ATOL + scaling.FLOAT32_RTOL*np.abs(x)
    return d


def verify_candidate(scaler, expected_record):
    """Caller supplies independently authenticated unchanged candidate record."""
    require(type(scaler) is scaling.FrozenScaler, 'Exact B1 FrozenScaler required')
    require(scaler.record() == expected_record, 'Candidate record changed')
    r = scaler.record(); p = r['provenance']
    require(tuple(p['features']) == bridge.FEATURES and tuple(p['feature_units']) == bridge.UNITS,
            'Feature order or units differ')
    require(tuple(p['mapping']) == bridge.ORIGINAL_NAMES and p['mapping_digest'] == bridge.MAPPING_DIGEST,
            'Original mapping differs')
    require(p['fit_split'] == 'train' and p['ddof'] == 0 and p['arithmetic'] == 'float64' and
            p['algorithm'] == scaling.ALGORITHM and p['threshold'] == scaling.EPSILON,
            'Fitting convention differs')
    require(p['train_bounds_utc'] == [t+'Z' for t in bridge.SPLITS['train']], 'Training bounds differ')
    require(r['target_units'] == ['W', 'degC'] and p['target_units'] == ['W', 'degC'], 'Target units differ')
    expected_std = np.where(np.array(scaler.original_x_std[:13]) < scaling.EPSILON, 1., scaler.original_x_std[:13])
    require(np.array_equal(expected_std, scaler.x_std[:13]), 'Constant-channel handling differs')
    require(p['tolerances'] == dict(float64_rtol=1e-12, float64_atol=1e-12, float32_rtol=2e-6, float32_atol=2e-6),
            'Historical tolerances changed')
    return scaler.fingerprint


def verify_inputs(scaler, physical):
    before = scaler.fingerprint
    actual = scaler.transform_inputs(physical)
    inverse = scaler.inverse_inputs(actual)
    require(actual.shape == np.shape(physical) and actual.dtype == np.float32, 'Model input shape/dtype differs')
    require(np.array_equal(actual[:,:,13], np.asarray(physical)[:,:,13]) and
            np.array_equal(inverse[:,:,13], np.asarray(physical)[:,:,13]), 'Categorical ordinal changed')
    d = verify_affine(np.asarray(physical)[:,:,:13], scaler.x_mean[:13], scaler.x_std[:13],
                      actual[:,:,:13], inverse[:,:,:13])
    require(before == scaler.fingerprint, 'Scaler mutated')
    return d


def verify_targets(scaler, physical):
    """Same IEEE accounting for the pre-existing independent W/degC target scaler."""
    return verify_affine(physical, scaler.y_mean, scaler.y_std,
                         scaler.standardize_targets(physical), scaler.inverse_targets(scaler.standardize_targets(physical)))
