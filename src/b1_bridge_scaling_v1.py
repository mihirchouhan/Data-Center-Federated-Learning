"""B1 float64 fixed-order two-pass TRAIN-only population scalers; no file I/O.

Tests may fit invented windows. No actual project scaler fit is authorized by
this module. Validation capabilities cannot enter fit_training_scaler.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_real_data_bridge_v1 import (TrainingWindows, ORIGINAL_NAMES, MAPPING_DIGEST,
    FEATURES, UNITS, SPLITS, training_eligibility, canonical, _numeric)
from src.phase1_3.schema import frozen_array

EPSILON = 1e-12
FLOAT64_RTOL, FLOAT64_ATOL = 1e-12, 1e-12
FLOAT32_RTOL, FLOAT32_ATOL = 2e-6, 2e-6
ALGORITHM = 'two_pass_f64_window_then_step_order_kahan_window_sums_ddof0_v1'


def _add(total, correction, value):
    adjusted = value - correction
    result = total + adjusted
    return result, (result - total) - adjusted


@dataclass(frozen=True)
class FrozenScaler:
    x_mean: tuple
    x_std: tuple
    original_x_std: tuple
    y_mean: tuple
    y_std: tuple
    provenance_json: bytes
    def __post_init__(self):
        for name, size in [('x_mean',14), ('x_std',14), ('original_x_std',14), ('y_mean',2), ('y_std',2)]:
            raw = getattr(self, name)
            a = _numeric(raw, name, (size,)).astype(np.float64)
            object.__setattr__(self, name, tuple(float(x) for x in a))
        if (min(self.x_std)<=0 or min(self.original_x_std)<0 or min(self.y_std)<EPSILON or
                self.x_mean[13]!=0 or self.x_std[13]!=1 or self.original_x_std[13]!=0):
            raise ValueError('Scaler roles/positive scales differ from B1')
        p = json.loads(self.provenance_json)
        if p.get('fit_split')!='train' or p.get('algorithm')!=ALGORITHM or p.get('mapping_digest')!=MAPPING_DIGEST:
            raise ValueError('Training scaler provenance missing/incompatible')
        object.__setattr__(self, 'provenance_json', canonical(p))
    @property
    def target_scaling(self):
        return TargetScaling(self.y_mean, self.y_std)
    def record(self):
        return {'schema':'b1.frozen.window.pooled.scaler.v1',
                'x_mean':list(self.x_mean), 'x_std':list(self.x_std),
                'original_x_std':list(self.original_x_std),
                'constant_input_channels':[i for i in range(13) if self.original_x_std[i]<EPSILON],
                'y_mean':list(self.y_mean), 'y_std':list(self.y_std),
                'target_units':['W','degC'], 'provenance':json.loads(self.provenance_json)}
    @property
    def fingerprint(self): return hashlib.sha256(canonical(self.record())).hexdigest()
    def require_training(self, training):
        if type(training) is not TrainingWindows:
            raise PermissionError('Exact training capability required')
        p = json.loads(self.provenance_json)
        if (p['fit_identity_sha256']!=training.identity_sha256 or
                p['fit_target_sha256']!=training.target_sha256 or
                p['fit_content_sha256']!=training.content_sha256 or p['fit_windows']!=len(training)):
            raise ValueError('Scaler fitted population/content differs from training')
        return True
    def transform_inputs(self, physical):
        a = _numeric(physical, 'physical inputs')
        if a.ndim!=3 or a.shape[1:]!=(36,14) or not len(a):
            raise ValueError('Nonempty [B,36,14] required')
        ids = a[:,:,13]
        if ((ids<0).any() or (ids>=979).any() or not np.array_equal(ids,ids.round()) or
                not np.array_equal(ids,np.broadcast_to(ids[:,:1],ids.shape))):
            raise ValueError('Invalid unscaled categorical identity')
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            scaled = ((a.astype(np.float64)-self.x_mean)/self.x_std).astype(np.float32)
        if not np.isfinite(scaled).all() or not np.array_equal(scaled[:,:,13],ids):
            raise ValueError('Invalid float32 normalized input or changed node ordinal')
        return frozen_array(scaled)
    def inverse_inputs(self, standardized):
        a = _numeric(standardized, 'standardized inputs')
        if a.ndim!=3 or a.shape[1:]!=(36,14) or not len(a):
            raise ValueError('Nonempty [B,36,14] required')
        with np.errstate(over='raise',invalid='raise'):
            out = a.astype(np.float64)*self.x_std+self.x_mean
        if not np.isfinite(out).all(): raise ValueError('Invalid inverse inputs')
        return frozen_array(out)
    def standardize_targets(self, physical):
        with np.errstate(over='raise',invalid='raise'):
            out = self.target_scaling.standardize(physical).astype(np.float32)
        if not np.isfinite(out).all(): raise ValueError('Target not representable in float32')
        return frozen_array(out)
    def inverse_targets(self, standardized):
        return self.target_scaling.inverse(standardized)


def fit_training_scaler(training, *, implementation_sha256, chunk_size=256):
    """Fit only validated eligible TRAIN windows, in canonical node/origin order.

    Two passes over identical immutable windows; chunking affects reads only.
    Each window's 36 steps reduce in a fixed NumPy float64 axis-0 sum, followed
    by sequential vector Kahan accumulation over windows. Targets count once
    per window; overlapping input bins count once per window-step occurrence.
    The caller must bind the actual implementation hash in future provenance.
    """
    if type(training) is not TrainingWindows:
        raise PermissionError('Validation/test cannot enter training scaler fit')
    if type(chunk_size) is not int or chunk_size<1:
        raise ValueError('Positive integer chunk size required')
    if (type(implementation_sha256) is not str or len(implementation_sha256)!=64 or
            any(c not in '0123456789abcdef' for c in implementation_sha256)):
        raise ValueError('Explicit implementation SHA-256 required')
    eligible = training_eligibility(training)
    if not eligible or set(training.nodes.tolist())!=set(eligible):
        raise ValueError('Only eligible training windows may enter fit; no silent filtering')
    before = training.content_sha256
    sx=np.zeros(13,dtype=np.float64);cx=sx.copy();sy=np.zeros(2,dtype=np.float64);cy=sy.copy()
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for x,y in training.canonical_chunks(chunk_size):
                for row,target in zip(x,y):
                    sx,cx=_add(sx,cx,np.sum(row[:,:13],axis=0,dtype=np.float64))
                    sy,cy=_add(sy,cy,target)
            xm=sx/(len(training)*36);ym=sy/len(training)
            vx=np.zeros(13,dtype=np.float64);cx=vx.copy();vy=np.zeros(2,dtype=np.float64);cy=vy.copy()
            for x,y in training.canonical_chunks(chunk_size):
                for row,target in zip(x,y):
                    dx=row[:,:13]-xm;dy=target-ym
                    vx,cx=_add(vx,cx,np.sum(dx*dx,axis=0,dtype=np.float64))
                    vy,cy=_add(vy,cy,dy*dy)
            raw_x=np.sqrt(vx/(len(training)*36));ys=np.sqrt(vy/len(training))
    except FloatingPointError as exc:
        raise ValueError('Nonfinite/overflow scaler arithmetic') from exc
    if not all(np.isfinite(a).all() for a in (xm,ym,raw_x,ys)) or (ys<EPSILON).any():
        raise ValueError('Nonfinite/constant target statistics require review')
    if training.content_sha256!=before: raise ValueError('Training content changed during fit')
    applied=np.where(raw_x<EPSILON,1.,raw_x)
    provenance={'fit_split':'train','fit_windows':len(training),'fit_input_step_occurrences':len(training)*36,
        'fit_target_count':len(training),'fit_identity_sha256':training.identity_sha256,
        'fit_target_sha256':training.target_sha256,'fit_content_sha256':training.content_sha256,
        'mapping_digest':MAPPING_DIGEST,'mapping':list(ORIGINAL_NAMES),
        'features':list(FEATURES),'feature_units':list(UNITS),'target_units':['W','degC'],
        'train_bounds_utc':[b+'Z' for b in SPLITS['train']],
        'weighting':'each accepted eligible training window-step; repeated bins retained; targets once/window',
        'algorithm':ALGORITHM,'arithmetic':'float64','ddof':0,'threshold':EPSILON,
        'reduction_order':'original node ordinal, increasing origin; fixed36step axis0 sum per window, then vector Kahan',
        'numpy_version':np.__version__,'implementation_sha256':implementation_sha256,
        'source_evidence':training.source_evidence,
        'tolerances':{'float64_rtol':FLOAT64_RTOL,'float64_atol':FLOAT64_ATOL,
                      'float32_rtol':FLOAT32_RTOL,'float32_atol':FLOAT32_ATOL},
        'chunk_size_is_io_only':True,'fit_authorization_granted_by_code':False}
    return FrozenScaler(tuple(xm)+(0.,),tuple(applied)+(1.,),tuple(raw_x)+(0.,),
                        tuple(ym),tuple(ys),canonical(provenance))
