"""Independent training-only scalers for two physical-unit targets."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .schema import frozen_array, TARGET_UNITS, MASKED
from .validation import validate_batch

@dataclass(frozen=True)
class Scaler:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    mapping: tuple
    spec_digest: str
    training_identity_sha256: str

    def __post_init__(self):
        for name,shape in (("x_mean",(14,)),("x_std",(14,)),("y_mean",(2,)),("y_std",(2,))):
            a = frozen_array(getattr(self,name),np.float64)
            if a.shape != shape or not np.isfinite(a).all() or ("std" in name and (a <= 0).any()):
                raise ValueError("Invalid scaler "+name)
            object.__setattr__(self,name,a)
        if not self.mapping or len(set(self.mapping)) != len(self.mapping):
            raise ValueError("Scaler mapping changed")
        if len(self.spec_digest) != 64 or len(self.training_identity_sha256) != 64:
            raise ValueError("Missing scaler provenance")
        for k in (*MASKED,13):
            if self.x_mean[k] != 0 or self.x_std[k] != 1:
                raise ValueError("Masked and categorical channels require identity scaling")

    def transform(self, batch, spec):
        validate_batch(batch,spec)
        if batch.mapping != self.mapping or batch.spec_digest != self.spec_digest:
            raise ValueError("Scaler mapping/spec mismatch")
        x = ((batch.x-self.x_mean)/self.x_std).astype(np.float32)
        y = ((batch.y-self.y_mean)/self.y_std).astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite scaled values")
        return frozen_array(x), frozen_array(y)

    def inverse_targets(self, values):
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
            raise ValueError("Two finite target channels required")
        result = a.astype(np.float64)*self.y_std+self.y_mean
        if not np.isfinite(result).all():
            raise ValueError("Inverse transform overflow")
        return result  # columns remain W, degC

    def to_json(self):
        d = {k:getattr(self,k).tolist() for k in ("x_mean","x_std","y_mean","y_std")}
        d.update(version=1,mapping=list(self.mapping),spec_digest=self.spec_digest,
                 training_identity_sha256=self.training_identity_sha256,target_units=list(TARGET_UNITS))
        return json.dumps(d,sort_keys=True,allow_nan=False)

    @classmethod
    def from_json(cls,text,spec):
        d = json.loads(text)
        if set(d) != {"version","mapping","spec_digest","training_identity_sha256","target_units","x_mean","x_std","y_mean","y_std"}:
            raise ValueError("Scaler serialization schema differs")
        if d["version"] != 1 or d["target_units"] != list(TARGET_UNITS) or d["spec_digest"] != spec.digest:
            raise ValueError("Scaler unit/spec provenance mismatch")
        return cls(*[np.array(d[k]) for k in ("x_mean","x_std","y_mean","y_std")],
                   tuple(d["mapping"]),d["spec_digest"],d["training_identity_sha256"])

def fit_training_scaler(batch,spec):
    validate_batch(batch,spec)
    if batch.split != "train" or not len(batch):
        raise ValueError("Fit scalers on nonempty training windows only")
    xm = batch.x.mean(axis=(0,1),dtype=np.float64)
    xs = batch.x.std(axis=(0,1),dtype=np.float64)
    ym = batch.y.mean(axis=0,dtype=np.float64)
    ys = batch.y.std(axis=0,dtype=np.float64)
    xs[xs < 1e-12] = 1
    ys[ys < 1e-12] = 1
    for k in (*MASKED,13):
        xm[k],xs[k] = 0,1
    identity = hashlib.sha256(batch.nodes.astype("<i8").tobytes()+batch.origins.view("i8").astype("<i8").tobytes()).hexdigest()
    return Scaler(xm,xs,ym,ys,batch.mapping,spec.digest,identity)
