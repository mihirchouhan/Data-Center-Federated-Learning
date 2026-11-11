"""Variable-history, joint-target and identity contracts; no data loading."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from src.phase1_3.schema import FEATURES, UNITS, TARGETS, TARGET_UNITS, WIDTH_NS, frozen_array
from src.phase1_3.config import utc_ns

HISTORIES = (6, 36, 144)

def identity_digest(nodes, origins):
    nodes, origins = np.asarray(nodes), np.asarray(origins)
    if nodes.ndim != 1 or nodes.dtype.kind != 'U' or origins.shape != nodes.shape or any(not x for x in nodes.tolist()):
        raise ValueError('Literal node strings and matching origins required')
    if origins.dtype != np.dtype('datetime64[ns]') or np.isnat(origins).any():
        raise ValueError('UTC datetime64[ns] origins required')
    h = hashlib.sha256()
    for node, origin in zip(nodes.tolist(),origins.view('i8').tolist()):
        row = json.dumps([node,origin],ensure_ascii=True,separators=(',',':')).encode()
        h.update(len(row).to_bytes(8,'big')); h.update(row)
    return h.hexdigest()

def bounds(protocol, split):
    if split not in ('train','val','test'): raise ValueError('Unknown split')
    return tuple(utc_ns(x) for x in protocol['splits'][split])

def split_for_origin(origin, history, protocol):
    if type(history) is not int or history not in HISTORIES: raise ValueError('Unsupported history')
    origin = int(origin)
    if origin % (6*WIDTH_NS): return None
    for split in ('train','val','test'):
        lo,hi = bounds(protocol,split)
        if origin-history*WIDTH_NS >= lo and origin+6*WIDTH_NS <= hi: return split
    return None

def target_means(future, observed):
    """future columns: original binned power kW and inlet temperature degC."""
    f, good = np.asarray(future), np.asarray(observed)
    if f.ndim != 3 or not len(f) or f.shape[1:] != (6,2) or good.shape != f.shape or good.dtype != np.bool_:
        raise ValueError('Six bins and two boolean observed target masks required')
    if not good.all() or not np.isfinite(f).all() or (f[:,:,0]<0).any() or (f[:,:,1]<-273.15).any():
        raise ValueError('Missing, imputed or invalid future target')
    out = f.mean(axis=1,dtype=np.float64); out[:,0] *= 1000
    return frozen_array(out)

@dataclass(frozen=True)
class Windows:
    x: np.ndarray                 # unscaled [N,H,14]
    observed: np.ndarray          # original-valid [N,H,13]
    future: np.ndarray            # original [N,6,2], kW/degC
    target_observed: np.ndarray   # [N,6,2]
    history_times: np.ndarray     # bin STARTS [N,H], UTC datetime64[ns]
    target_times: np.ndarray      # bin STARTS [N,6]
    history_nodes: np.ndarray     # mapping indices [N,H]
    target_nodes: np.ndarray      # mapping indices [N,6]
    source_latest: np.ndarray     # [N,H,9] latest contributing SENSOR timestamp
    source_earliest: np.ndarray   # [N,H,9] earliest contributing SENSOR timestamp
    nodes: np.ndarray             # literal strings [N]
    origins: np.ndarray           # UTC datetime64[ns] [N]
    mapping: tuple
    split: str
    history: int
    timezone: str = 'UTC'
    features: tuple = FEATURES
    units: tuple = UNITS

    def __post_init__(self):
        for name in ('x','observed','future','target_observed','history_times','target_times',
                     'history_nodes','target_nodes','source_latest','source_earliest','nodes','origins'):
            object.__setattr__(self,name,frozen_array(getattr(self,name)))

def validate_windows(w, protocol):
    if not isinstance(w,Windows) or w.history not in HISTORIES or type(w.history) is not int:
        raise ValueError('Versioned supported history required')
    n,h = len(w.nodes),w.history
    if not n: raise ValueError('Empty cohort requires scientific review')
    if w.timezone != 'UTC' or w.features != FEATURES or w.units != UNITS:
        raise ValueError('Timezone, feature order or units changed')
    if not w.mapping or len(set(w.mapping)) != len(w.mapping) or any(type(x) is not str or not x for x in w.mapping):
        raise ValueError('Invalid literal node mapping')
    shapes = {'x':(n,h,14),'observed':(n,h,13),'future':(n,6,2),'target_observed':(n,6,2),
              'history_times':(n,h),'target_times':(n,6),'history_nodes':(n,h),'target_nodes':(n,6),
              'source_latest':(n,h,9),'source_earliest':(n,h,9),'nodes':(n,),'origins':(n,)}
    for key,shape in shapes.items():
        if getattr(w,key).shape != shape: raise ValueError('Incompatible shape: '+key)
    if w.nodes.dtype.kind != 'U' or any(x not in w.mapping for x in w.nodes.tolist()):
        raise ValueError('Unmapped/nonliteral client')
    for key in ('origins','history_times','target_times','source_latest','source_earliest'):
        a = getattr(w,key)
        if a.dtype != np.dtype('datetime64[ns]') or np.isnat(a).any(): raise ValueError('Ambiguous time: '+key)
    origins = w.origins.view('i8'); lo,hi = bounds(protocol,w.split)
    if (origins%(6*WIDTH_NS)).any() or (origins-h*WIDTH_NS<lo).any() or (origins+6*WIDTH_NS>hi).any():
        raise ValueError('Off-hour origin or whole-window split leakage')
    expected_h = origins[:,None]+np.arange(-h,0)*WIDTH_NS
    expected_t = origins[:,None]+np.arange(6)*WIDTH_NS
    if not np.array_equal(w.history_times.view('i8'),expected_h) or not np.array_equal(w.target_times.view('i8'),expected_t):
        raise ValueError('Missing/gapped/future input or altered target bins')
    indices = np.array([w.mapping.index(x) for x in w.nodes.tolist()])
    for key,steps in [('history_nodes',h),('target_nodes',6)]:
        a=getattr(w,key)
        if a.dtype.kind not in 'iu' or not np.array_equal(a,np.broadcast_to(indices[:,None],(n,steps))):
            raise ValueError('Windows cannot cross node identities')
    keys=list(zip(indices.tolist(),origins.tolist()))
    if keys != sorted(set(keys)): raise ValueError('Duplicate or unordered node/origin identity')
    for idx in np.unique(indices):
        if (np.diff(origins[indices==idx])<6*WIDTH_NS).any(): raise ValueError('Overlapping targets')
    if w.observed.dtype != np.bool_ or not w.observed.all() or not np.isfinite(w.x).all():
        raise ValueError('Missing active input; no forward filling')
    if not np.array_equal(w.x[:,:,13],np.broadcast_to(indices[:,None],(n,h))):
        raise ValueError('Categorical channel inconsistent')
    if ((w.x[:,:,0]<0).any() or ((w.x[:,:,[1,2,3,4,8]]<0)|(w.x[:,:,[1,2,3,4,8]]>100)).any()
        or ((w.x[:,:,5]<0)|(w.x[:,:,5]>1)).any() or (w.x[:,:,[6,7]]<-273.15).any()
        or (np.abs(w.x[:,:,9:13])>1).any()): raise ValueError('Invalid physical input')
    earliest,latest = w.source_earliest.view('i8'),w.source_latest.view('i8')
    if (earliest<lo).any() or (latest>=expected_h[:,:,None]+WIDTH_NS).any() or (earliest>latest).any():
        raise ValueError('Future/raw cross-split contributor leakage')
    return target_means(w.future,w.target_observed)

def eligibility(training, protocol):
    validate_windows(training,protocol)
    if training.split != 'train': raise ValueError('Client admission must be training-only')
    result = {}
    for node in training.mapping:
        keep=training.nodes==node; count=int(keep.sum())
        days=len(np.unique(training.origins[keep].astype('datetime64[D]')))
        if count>=protocol['eligibility']['minimum_windows'] and days>=protocol['eligibility']['minimum_days']:
            result[node]=count
    return result

def validate_client_cohorts(training,validation,test,protocol):
    for w,split in [(training,'train'),(validation,'val'),(test,'test')]:
        validate_windows(w,protocol)
        if w.split!=split or w.mapping!=training.mapping: raise ValueError('Cohort split/mapping inconsistent')
    clients=eligibility(training,protocol)
    if len(clients)<protocol['federation']['clients_per_round']:
        raise ValueError('Insufficient eligible training clients; no sampling fallback')
    for w in (training,validation,test):
        if not set(w.nodes.tolist()).issubset(clients): raise ValueError('Ineligible/evaluation-only client')
    return clients

def assert_same_evaluation(a,b,protocol):
    ya,yb=validate_windows(a,protocol),validate_windows(b,protocol)
    if a.mapping!=b.mapping or a.split!=b.split or not np.array_equal(a.nodes,b.nodes) or not np.array_equal(a.origins,b.origins):
        raise ValueError('Unequal ordered evaluation cohort')
    if not np.array_equal(a.target_times,b.target_times) or not np.array_equal(a.future,b.future) or not np.array_equal(ya,yb):
        raise ValueError('Same identity must retain exact original targets')
    return identity_digest(a.nodes,a.origins)

def matched_indices(batches,protocol):
    if not batches: raise ValueError('No histories supplied')
    for w in batches: validate_windows(w,protocol)
    if len({w.history for w in batches})!=len(batches) or any(w.mapping!=batches[0].mapping or w.split!=batches[0].split for w in batches):
        raise ValueError('Mixed mapping/split or duplicate history')
    keys=[list(zip(w.nodes.tolist(),w.origins.view('i8').tolist())) for w in batches]
    common=set(keys[0]).intersection(*(set(k) for k in keys[1:]))
    if not common: raise ValueError('Empty history intersection; stop this study')
    order=sorted(common,key=lambda x:(batches[0].mapping.index(x[0]),x[1]))
    lookups=[dict(zip(k,range(len(k)))) for k in keys]
    result=[np.array([lookup[v] for v in order],dtype=np.int64) for lookup in lookups]
    reference=batches[0].future[result[0]]
    for w,idx in zip(batches,result):
        if not np.array_equal(reference,w.future[idx]): raise ValueError('History intersection target mismatch')
    return result

def mask_weather(standardized, policy):
    a=np.asarray(standardized)
    if a.ndim!=3 or a.shape[1] not in HISTORIES or a.shape[2]!=14 or not np.isfinite(a).all():
        raise ValueError('Finite standardized [N,H,14] required')
    if policy not in ('enabled','masked'): raise ValueError('Unknown weather policy')
    out=np.array(a,copy=True)
    if policy=='masked': out[:,:,[7,8]]=0
    return frozen_array(out)

def validate_scaler(record, training, protocol):
    validate_windows(training,protocol)
    if training.split!='train': raise ValueError('Scaler population must be training-only')
    if record['fit_identity_sha256']!=identity_digest(training.nodes,training.origins):
        raise ValueError('Scaler fitted population differs')
    if record['fit_split']!='train' or tuple(record['mapping'])!=training.mapping:
        raise ValueError('Scaler split/mapping differs')
    if record['reference_history_steps']!=36 or training.history!=36 or record['target_units']!=list(TARGET_UNITS):
        raise ValueError('Wrong reference history/target scaling')
    for k,shape in [('x_mean',(14,)),('x_std',(14,)),('y_mean',(2,)),('y_std',(2,))]:
        a=np.asarray(record[k],dtype=np.float64)
        if a.shape!=shape or not np.isfinite(a).all() or ('std' in k and (a<=0).any()): raise ValueError('Invalid scaler: '+k)
    if record['x_mean'][13]!=0 or record['x_std'][13]!=1: raise ValueError('Node index cannot be standardized')
    raw_std=np.asarray(record['original_y_std'],dtype=np.float64)
    if raw_std.shape!=(2,) or not np.isfinite(raw_std).all() or (raw_std<1e-12).any():
        raise ValueError('Constant/invalid target: stop and review')
    accepted=eligibility(training,protocol)
    if not accepted or set(training.nodes.tolist())!=set(accepted): raise ValueError('Filter eligible clients before scaler fit')
    return True
