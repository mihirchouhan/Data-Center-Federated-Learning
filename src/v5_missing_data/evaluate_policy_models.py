"""V5 common-cohort validation adapter and separately locked final test reporting.

No inference at import, preflight, lock creation, or verify. Actual test inference
requires --stage evaluate plus an existing immutable 24-model selection lock.
Recovered-window reporting is deliberately outside this implementation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import statistics

import numpy as np
import torch

from src.experiment_data import immutable_json, read_json, sha256
from src.experiment_eval import metric_pair
from src.v5_missing_data.training_data import CohortContext, SplitData, POLICIES
from src.v5_missing_data import training_io as tio

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TestPermission:
    cohort_receipt_sha256: str
    common_identity_sha256: str
    selection_lock_sha256: str


def predict_and_score(model, data, device, *, test_permission=None):
    """Explicit future inference path. Never called by implementation tests."""
    if data.split_name == 'test':
        expected = data.spec['identity']['ordered_sha256']
        if (not isinstance(test_permission, TestPermission)
                or test_permission.cohort_receipt_sha256 != data.context.receipt_sha256
                or test_permission.common_identity_sha256 != expected):
            raise PermissionError('Test inference requires an existing verified selection lock')
    elif data.split_name != 'val':
        raise PermissionError('Evaluator cannot score training examples')
    predicted, actual, baseline, identities = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(data), 1024):
            x, y, ids = data.batch(np.arange(first, min(first+1024, len(data)), dtype=np.int64))
            p = model(torch.from_numpy(x).to(device)).detach().cpu().numpy().reshape(-1)
            if len(p) != len(y) or not np.isfinite(p).all():
                raise ValueError('Nonfinite or mis-sized prediction vector')
            mean, std = data.context.scalers['y_mean'][0], data.context.scalers['y_std'][0]
            predicted.append((p.astype(np.float64)*std+mean)*1000)
            actual.append((y[:,0].astype(np.float64)*std+mean)*1000)
            baseline.append((x[:,-1,0].astype(np.float64)*data.context.scalers['x_std'][0]
                             +data.context.scalers['x_mean'][0])*1000)
            identities.append(ids)
    ids = np.concatenate(identities)
    data.require_identity(ids)
    p, y, b = map(np.concatenate, (predicted, actual, baseline))
    if not all(np.isfinite(v).all() for v in (p,y,b)):
        raise ValueError('Nonfinite physical-unit predictions/targets/baseline')
    result = {**metric_pair(p-y,b-y), 'unit':'W', 'split':data.split_name,
        'finite_predictions':True,
        'evaluated_nodes':len(np.unique(ids['node'])),
        'predictions_sha256':hashlib.sha256(p.tobytes()).hexdigest(),
        'targets_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
        'persistence_sha256':hashlib.sha256(b.tobytes()).hexdigest(),
        'window_identity':data.spec['identity']}
    by_node = {}
    for node in data.context.names:
        keep = ids['node'] == node
        if keep.any():
            by_node[node] = metric_pair((p-y)[keep],(b-y)[keep])
    result['macro_node_mae_w'] = float(np.mean([v['mae_w'] for v in by_node.values()]))
    result['macro_node_rmse_w'] = float(np.mean([v['rmse_w'] for v in by_node.values()]))
    for metric in ('mae_w','rmse_w'):
        result['macro_node_persistence_'+metric]=float(np.mean([v['persistence_'+metric] for v in by_node.values()]))
    return result, {'predictions_w':p,'targets_w':y,'persistence_w':b,'ids':ids,'per_node':by_node}


def selections(plan, context, primary, sensitivity):
    from src.v5_missing_data.run_policy_experiments import ALGORITHMS, SEEDS, verify_run
    records = []
    for method in ALGORITHMS:
        for policy in POLICIES:
            parent = primary if policy in ('no_fill','ff20') else sensitivity
            for seed in SEEDS:
                run = parent / method / f'{policy}_seed{seed}'
                checked = verify_run(plan,context,run)
                manifest = read_json(run/'manifest.json')
                if (checked['status']!='verified_complete' or manifest['kind']!='main'
                        or manifest['cap']!=200 or (manifest['method'],manifest['policy'],manifest['seed'])!=(method,policy,seed)):
                    raise ValueError('Final test requires every declared complete 200-round main fit')
                receipt = read_json(run/'selected/receipt.json')
                records.append({'method':method,'policy':policy,'seed':seed,'selected_round':receipt['selected_round'],
                    'validation':receipt['validation'],'run':str(run.resolve()),
                    'model_sha256':sha256(run/'selected/model.pt'),
                    'selected_receipt_sha256':sha256(run/'selected/receipt.json'),
                    'completed_sha256':sha256(run/'completed.json'),'manifest_sha256':sha256(run/'manifest.json')})
    return records


def make_lock(plan, context, primary, sensitivity, output):
    # Train-only verification happens before any test SplitData construction.
    records = selections(plan,context,primary,sensitivity)
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'selection_lock.json',{'version':1,'plan_digest':tio.digest(plan),
        'cohort_receipt_sha256':context.receipt_sha256,
        'common_test_identity':context.contract['cohorts']['common/all_policies/test']['identity'],
        'fits':records,'selection':'Validation only; test period previously observed; no best-seed selection'})
    immutable_json(output/'completed.json',{'status':'locked','selection_lock_sha256':sha256(output/'selection_lock.json')})
    return {'status':'locked','models':len(records),'test_examples_opened':False,'model_forward_or_scoring':False}


def load_lock(plan, context, directory):
    receipt = read_json(directory/'completed.json')
    lock = read_json(directory/'selection_lock.json')
    checksum = sha256(directory/'selection_lock.json')
    if (receipt['status']!='locked' or receipt['selection_lock_sha256']!=checksum
            or lock['plan_digest']!=tio.digest(plan) or lock['cohort_receipt_sha256']!=context.receipt_sha256
            or lock['common_test_identity']!=context.contract['cohorts']['common/all_policies/test']['identity']):
        raise ValueError('Selection lock/data/configuration changed')
    from src.v5_missing_data.run_policy_experiments import ALGORITHMS, SEEDS
    expected={(m,p,s) for m in ALGORITHMS for p in POLICIES for s in SEEDS}
    actual={(x['method'],x['policy'],x['seed']) for x in lock['fits']}
    if len(lock['fits'])!=24 or actual!=expected:
        raise ValueError('Test lock must contain exactly the 24 declared fits')
    for item in lock['fits']:
        run=Path(item['run'])
        for file,key in (('selected/model.pt','model_sha256'),('selected/receipt.json','selected_receipt_sha256'),
                         ('completed.json','completed_sha256'),('manifest.json','manifest_sha256')):
            if sha256(run/file)!=item[key]:raise ValueError('Locked fit/export changed')
    permission=TestPermission(context.receipt_sha256,lock['common_test_identity']['ordered_sha256'],checksum)
    return lock,permission


def evaluate_locked(plan, context, lock_dir, output):
    lock,permission=load_lock(plan,context,lock_dir)
    if output.exists():raise ValueError('Evaluation output exists; never overwrite partial/completed results')
    # The lock already exists on disk before the first test example is mapped.
    data=SplitData(context,'test')
    from src.v5_missing_data.run_policy_experiments import initialize,runtime
    device=runtime(plan)
    output.mkdir(parents=True,exist_ok=False)
    immutable_json(output/'selection_lock.json',lock)
    immutable_json(output/'manifest.json',{'plan':plan,'lock_sha256':permission.selection_lock_sha256,
        'cohort_receipt_sha256':context.receipt_sha256,'software':tio.software(),
        'test_status':'Previously observed development period; not untouched confirmatory data'})
    results=[]; baseline_identity=None
    for item in lock['fits']:
        model=initialize(context,item['seed'])
        state=torch.load(Path(item['run'])/'selected/model.pt',map_location='cpu',weights_only=True)['model_state']
        model.load_state_dict(state,strict=True);model=model.to(device)
        score,arrays=predict_and_score(model,data,device,test_permission=permission)
        signature=(score['targets_sha256'],score['persistence_sha256'],score['window_identity']['ordered_sha256'])
        if baseline_identity is None:baseline_identity=signature
        if signature!=baseline_identity:raise ValueError('Test targets/baseline/windows differ across fits')
        folder=output/f"{item['method']}_{item['policy']}_seed{item['seed']}";folder.mkdir()
        immutable_json(folder/'metrics.json',score)
        immutable_json(folder/'per_node.json',arrays['per_node'])
        with (folder/'predictions.npz').open('xb') as handle:
            np.savez_compressed(handle,**{k:v for k,v in arrays.items() if k!='per_node'})
        results.append({**item,**score})
    groups=[]
    for method in ('fedavg','centralized_sgd'):
        for policy in POLICIES:
            group=[r for r in results if r['method']==method and r['policy']==policy]
            groups.append({'method':method,'policy':policy,
                **{f'{stat}_{metric}':fn([r[metric] for r in group])
                   for metric in ('mae_w','rmse_w') for stat,fn in (('mean',statistics.mean),('sample_sd',statistics.stdev))}})
    contrasts=[]
    for method in ('fedavg','centralized_sgd'):
        for policy in ('ff20','ff10','ff30'):
            paired=[]
            for seed in (42,43,44):
                filled=next(r for r in results if (r['method'],r['policy'],r['seed'])==(method,policy,seed))
                reference=next(r for r in results if (r['method'],r['policy'],r['seed'])==(method,'no_fill',seed))
                paired.append({'seed':seed,**{metric:filled[metric]-reference[metric] for metric in ('mae_w','rmse_w')}})
            contrasts.append({'method':method,'contrast':policy+' minus no_fill','primary':policy=='ff20','paired':paired,
                **{f'{stat}_{metric}':fn([r[metric] for r in paired])
                   for metric in ('mae_w','rmse_w') for stat,fn in (('mean',statistics.mean),('sample_sd',statistics.stdev))}})
    immutable_json(output/'results.json',{'fits':results,'groups':groups,'paired_contrasts':contrasts,
        'std_definition':'sample SD across three seeds; not independent datasets or a confidence interval',
        'aggregation':'Individual-score means, not an ensemble','test_based_model_selection':False})
    immutable_json(output/'completed.json',{'status':'complete','artifacts':{
        str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}})
    return {'status':'complete','fits':24,'samples_per_fit':len(data)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--data-dir',required=True)
    parser.add_argument('--stage',choices=('lock','evaluate','verify'),required=True)
    parser.add_argument('--primary-dir');parser.add_argument('--sensitivity-dir');parser.add_argument('--lock-dir')
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    try:
        from src.v5_missing_data.run_policy_experiments import load_plan,output_path
        plan=load_plan(Path(args.plan).resolve())
        context=CohortContext(args.data_dir,plan['cohort_receipt_sha256'])
        output=output_path(ROOT,args.output_dir)
        if args.stage=='lock':
            if not args.primary_dir or not args.sensitivity_dir:raise ValueError('Both completed matrix roots are required')
            result=make_lock(plan,context,Path(args.primary_dir).resolve(),Path(args.sensitivity_dir).resolve(),output)
        elif args.stage=='evaluate':
            if not args.lock_dir:raise ValueError('Existing lock directory is required')
            result=evaluate_locked(plan,context,Path(args.lock_dir).resolve(),output)
        else:
            receipt=read_json(output/'completed.json')
            if receipt['status']=='locked':
                load_lock(plan,context,output)
            else:
                if receipt['status']!='complete':raise ValueError('Evaluation has not completed')
                saved_manifest=read_json(output/'manifest.json')
                if (tio.digest(saved_manifest['plan'])!=tio.digest(plan)
                        or saved_manifest['cohort_receipt_sha256']!=context.receipt_sha256):
                    raise ValueError('Evaluation plan/cohort binding changed')
                actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p!=output/'completed.json'}
                if actual!=set(receipt['artifacts']):raise ValueError('Evaluation inventory differs')
                for name,checksum in receipt['artifacts'].items():
                    from src.history_data_v4 import inside
                    if sha256(inside(output,name))!=checksum:raise ValueError('Evaluation artifact changed: '+name)
            result={'status':'verified_complete','model_forward_or_scoring':False}
        import json
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        print('V5 EVALUATION ERROR: '+str(error));return 1


if __name__=='__main__':
    raise SystemExit(main())
