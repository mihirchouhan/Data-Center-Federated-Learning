"""Pure two-output metric/selection functions. No model or data file loading."""
import numpy as np
import re
from .contracts import validate_windows, identity_digest

def persistence(w,protocol):
    validate_windows(w,protocol)
    result=w.x[:,-1,[0,6]].astype(np.float64,copy=True)
    result[:,0]*=1000
    return result

def _pair(errors):
    with np.errstate(over='raise',invalid='raise'):
        return {'mae':np.mean(np.abs(errors),axis=0,dtype=np.float64).tolist(),
                'rmse':np.sqrt(np.mean(errors*errors,axis=0,dtype=np.float64)).tolist()}

def metrics(actual,predicted,nodes,origins):
    a,p=np.asarray(actual,dtype=np.float64),np.asarray(predicted,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=2 or not len(a) or p.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Both complete finite physical-unit outputs required')
    nodes,origins=np.asarray(nodes),np.asarray(origins)
    if nodes.shape!=(len(a),) or origins.shape!=(len(a),): raise ValueError('Metric identities differ')
    digest=identity_digest(nodes,origins)
    if len(set(zip(nodes.tolist(),origins.view('i8').tolist())))!=len(a): raise ValueError('Duplicate prediction identity')
    with np.errstate(over='raise',invalid='raise'): e=p-a
    by_node={str(n):{'samples':int((nodes==n).sum()),**_pair(e[nodes==n])} for n in np.unique(nodes)}
    return {'units':['W','degC'],'samples':len(a),'identity_sha256':digest,'pooled':_pair(e),
            'per_node':by_node,'macro_node':{k:np.mean([v[k] for v in by_node.values()],axis=0).tolist() for k in ('mae','rmse')}}

def joint_score(rmse,training_std):
    r,s=np.asarray(rmse,dtype=np.float64),np.asarray(training_std,dtype=np.float64)
    if r.shape!=(2,) or s.shape!=(2,) or not np.isfinite(r).all() or not np.isfinite(s).all() or (r<0).any() or (s<1e-12).any():
        raise ValueError('Finite two-output RMSE and nonconstant training scales required')
    value=float(np.mean((r/s)**2))
    if not np.isfinite(value): raise ValueError('Nonfinite joint selection score')
    return value

def select_validation(trace,training_std,expected_indices):
    if not trace or [r['index'] for r in trace]!=list(expected_indices) or any(r['split']!='val' for r in trace):
        raise ValueError('Complete ordered validation-only selection trace required')
    if any(type(x) is not int or x<0 for x in expected_indices) or list(expected_indices)!=sorted(set(expected_indices)):
        raise ValueError('Duplicate or unordered candidates')
    best=None
    for record in trace:
        score=joint_score(record['rmse'],training_std)
        if best is None or score<best['score']:best={'index':record['index'],'score':score}
    return best

def seed_summary(records):
    if sorted(records)!=[42,43,44]: raise ValueError('Keep all three predeclared seeds')
    items=[records[s] for s in (42,43,44)]
    if len({(x['identity_sha256'],x['samples']) for x in items})!=1: raise ValueError('Seed cohorts differ')
    return {metric:{'mean':np.mean([x['pooled'][metric] for x in items],axis=0).tolist(),
                    'sample_sd':np.std([x['pooled'][metric] for x in items],axis=0,ddof=1).tolist()}
            for metric in ('mae','rmse')}

def lag_mapping(history):
    from .contracts import HISTORIES, FEATURES
    if type(history) is not int or history not in HISTORIES: raise ValueError('Unsupported lag history')
    return [{'column':13*step+j,'feature':name,'minutes_before_origin':10*(history-step),'type':'q'}
            for step in range(history) for j,name in enumerate(FEATURES[:-1])]+[
                {'column':13*history,'feature':'node_index','type':'c','encoding':'frozen literal node mapping'}]

def verify_selection_record(record,expected_fit,expected_bindings):
    """Metadata-only contract for a future lock; never chooses a test checkpoint."""
    if record['fit_id']!=expected_fit or record['selection_split']!='val': raise ValueError('Wrong identity or test-based selection')
    if record['bindings']!=expected_bindings: raise ValueError('Lock provenance mismatch')
    if not record['models'] or any(re.fullmatch('[a-f0-9]{64}',m.get('sha256','')) is None for m in record['models']):
        raise ValueError('Missing/invalid model hash binding')
    return True

def evaluate_validation_arrays(w,predicted,prediction_nodes,prediction_origins,protocol):
    """Common aligned validation contract; final locked-test adapter is future work."""
    if w.split!='val': raise PermissionError('This release does not implement the locked final-test adapter')
    actual=validate_windows(w,protocol)
    if not np.array_equal(w.nodes,prediction_nodes) or not np.array_equal(w.origins,prediction_origins):
        raise ValueError('Predictions differ from ordered evaluation windows')
    return {'split':'val','model':metrics(actual,predicted,w.nodes,w.origins),
            'persistence':metrics(actual,persistence(w,protocol),w.nodes,w.origins)}
