"""Locked B1 March inference with a mandatory separately issued receipt.

Defaults to description. No fit/selection/scaler-fit API. Prediction verification
uses saved arrays and independent arithmetic, never a second forward pass.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time

import numpy as np
from src import b1_march_integrity_v1 as integrity
from src import b1_march_data_v1 as data_module
from src import b1_preparation_io_v1 as io
from src.phase2_multioutput.contracts import identity_digest
from src.phase2_multioutput.evaluation import metrics as _metrics


def metrics(actual, predicted, nodes, origins):
    for a in (actual,predicted):
        if np.ma.isMaskedArray(a) or np.asarray(a).dtype.kind not in 'iuf':
            raise ValueError('Unmasked real numeric targets/predictions required')
    return _metrics(actual,predicted,nodes,origins)


def independent_metrics(actual,predicted,nodes,origins):
    """Separate math.fsum reductions of saved physical prediction residuals."""
    a,p=np.asarray(actual),np.asarray(predicted)
    if np.ma.isMaskedArray(actual) or np.ma.isMaskedArray(predicted) or a.dtype.kind not in 'iuf' or p.dtype.kind not in 'iuf' or a.shape!=p.shape or a.ndim!=2 or a.shape[1]!=2 or not len(a):
        raise ValueError('Invalid prediction arrays')
    if not np.isfinite(a).all() or not np.isfinite(p).all():raise ValueError('Nonfinite predictions/targets')
    nodes,origins=np.asarray(nodes),np.asarray(origins)
    digest=identity_digest(nodes,origins)
    if len(nodes)!=len(a) or len(set(zip(nodes.tolist(),origins.view('i8').tolist())))!=len(a):raise ValueError('Missing/duplicate identities')
    def pair(indices):
        e=[[float(p[i,j])-float(a[i,j]) for i in indices] for j in range(2)]
        return {'mae':[math.fsum(abs(x) for x in v)/len(v) for v in e],
                'rmse':[math.sqrt(math.fsum(x*x for x in v)/len(v)) for v in e]}
    by={str(n):{'samples':int((nodes==n).sum()),**pair(np.flatnonzero(nodes==n))} for n in np.unique(nodes)}
    return {'units':['W','degC'],'samples':len(a),'identity_sha256':digest,'pooled':pair(range(len(a))),
        'per_node':by,'macro_node':{k:[math.fsum(v[k][j] for v in by.values())/len(by) for j in range(2)] for k in ('mae','rmse')}}


def compare_metrics(a,b):
    if a.keys()!=b.keys():raise ValueError('Metric fields differ')
    for k in a:
        x,y=a[k],b[k]
        if isinstance(x,dict):compare_metrics(x,y)
        elif isinstance(x,list):
            if x and isinstance(x[0],str):
                if x!=y:raise ValueError('Metric units differ')
            else:
                if not np.allclose(x,y,rtol=1e-12,atol=1e-12):raise ValueError('Independent metric mismatch')
        elif x!=y:raise ValueError('Metric metadata differs')


def aggregate(scores):
    if set(scores)!={42,43,44}:raise ValueError('All seeds required')
    if len({(v['samples'],v['identity_sha256']) for v in scores.values()})!=1:raise ValueError('Seed test cohorts differ')
    return {group:{metric:{'mean':np.mean([scores[s][group][metric] for s in (42,43,44)],axis=0).tolist(),
        'sample_sd':np.std([scores[s][group][metric] for s in (42,43,44)],axis=0,ddof=1).tolist()}
        for metric in ('mae','rmse')} for group in ('pooled','macro_node')}


def device_contract(device):
    if device not in ('cpu','cuda:0'):raise ValueError('Explicit cpu or reviewed cuda:0 required')
    if device=='cuda:0' and (os.environ.get('CUDA_VISIBLE_DEVICES')!=integrity.GPU_UUID or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise ValueError('Reviewed physical GPU1 visibility/determinism required')
    return {'device':device,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')}


def receipt_body(plan,lock,data,device,operator):
    return {'schema':'b1.march.evaluation.authorization.v1','decision':'AUTHORIZED_TEST_INFERENCE_ONLY',
        'plan_sha256':integrity.digest(plan),'selection_lock_sha256':integrity.LOCK_SHA,
        'training_configuration':lock['configuration'],'training_dataset_manifest':lock['dataset_manifest'],
        'scaler_lock':lock['scaler_lock'],'scaler_statistics':lock['scaler_statistics'],
        'selected_models':lock['selected_models'],'sources':plan['sources'],'protocol':plan['protocol'],
        'test_dataset_manifest':data.manifest_binding,'test_reference':data.manifest['reference_file'],
        'test_results':data.manifest['results'],'output':str(integrity.ROOT/plan['evaluation_output']),
        'device_environment':device_contract(device),'operator_assertion':operator,
        'historical_preflight':'FAILED','semantic_reproduction':'UNKNOWN'}


def check_receipt(receipt,expected):
    if io.canonical(receipt)!=io.canonical(expected):raise PermissionError('Evaluation receipt does not bind exact verified models/data/source/protocol/device')
    if receipt['operator_assertion'].get('explicit_operator_authorization') is not True:
        raise PermissionError('No explicit operator authorization')


def authorize(plan,operator_path,device):
    lock,scaler,_=integrity.locked_context(plan)
    data=data_module.MarchData(integrity.ROOT/plan['dataset_output']/'dataset',plan,scaler)
    bindings={'plan_sha256':integrity.digest(plan),'selection_lock_sha256':integrity.LOCK_SHA,
        'dataset_manifest':data.manifest_binding,'output':str(integrity.ROOT/plan['evaluation_output']),'device':device}
    operator=integrity.statement(operator_path,'B1_March_test_evaluation_only',bindings)
    destination=integrity.ROOT/plan['receipt_output'];stage=integrity.stage_directory(destination)
    body=receipt_body(plan,lock,data,device,operator)
    io.exclusive(stage/'receipt.json',io.canonical(body))
    data.assert_unchanged()
    for n,b in plan['sources'].items():integrity.check(integrity.ROOT/n,b)
    integrity.publish(stage,destination)
    return {'status':'authorized_receipt_published','receipt':io.binding(destination/'receipt.json'),'inference':False}


def load_selected(model_record,lock,scaler,device):
    """Called only AFTER receipt verification by evaluate; no historical warm start."""
    path=integrity.ROOT/model_record['selected_model']['path']
    integrity.check(path,model_record['selected_model'])
    import torch
    from src.b1_device_adapter_v1 import B1DeviceAdapter
    from src.b1_real_fedavg_runner_v1 import state_contract
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    with io.safe_open(path) as f:saved=torch.load(f,map_location='cpu',weights_only=True)
    m=integrity.read_json(integrity.ROOT/model_record['run_manifest']['path'])
    if saved.get('schema')!='b1.production.selected.model.v1' or saved['manifest_sha256']!=integrity.digest(m):
        raise ValueError('Selected export envelope differs')
    roles=state_contract(saved['state'],scaler.target_scaling)
    snapshot=RoundSnapshot(saved['state'],roles,round_id=200)
    if snapshot.fingerprint!=model_record['selected_state_fingerprint']:raise ValueError('Selected state fingerprint differs')
    model=B1DeviceAdapter(target_scaling=scaler.target_scaling,root_seed=model_record['seed'],device=device)
    model.restore_round(snapshot);model.eval();integrity.check(path,model_record['selected_model'])
    return model


def _actual(reference):
    return np.column_stack((reference['power_w'],reference['temperature_c']))


def _read_predictions(path,n):
    with io.safe_open(path) as f:
        version=np.lib.format.read_magic(f)
        if version!=(1,0):raise ValueError('Prediction header version differs')
        shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=1024)
        if shape!=(n,2) or dtype!=np.dtype('float64') or fortran or os.fstat(f.fileno()).st_size!=f.tell()+n*16:raise ValueError('Prediction size/schema differs')
    with io.safe_open(path) as f:return np.load(f,allow_pickle=False)


def verify_saved_predictions(output,data):
    scores=integrity.read_json(output/'metrics.json');reference=data.reference
    if set(scores)!={'42','43','44'}:raise ValueError('Missing/extra evaluated seed')
    with io.safe_open(output/'reference.npy') as f:saved_ref=np.load(f,allow_pickle=False)
    if not np.array_equal(saved_ref,reference):raise ValueError('Saved test identities/targets differ')
    nodes=np.asarray(data_module.bridge.ORIGINAL_NAMES)[reference['node_index']];origins=reference['origin_ns'].astype('datetime64[ns]')
    results={}
    for seed in (42,43,44):
        predicted=_read_predictions(output/f'predictions_seed{seed}.npy',len(reference))
        independent=independent_metrics(_actual(reference),predicted,nodes,origins)
        compare_metrics(scores[str(seed)],independent);results[seed]=independent
    compare_metrics(integrity.read_json(output/'aggregate_metrics.json'),aggregate(results))
    # Independently reconcile report tables with the stored, independently checked scores.
    with (output/'seed_metrics.csv').open() as f:seedrows=list(csv.DictReader(f))
    expected={(s,g,name):(scores[str(s)][g]['mae'][j],scores[str(s)][g]['rmse'][j])
        for s in (42,43,44) for g in ('pooled','macro_node') for j,name in enumerate(('power_W','ambient_degC'))}
    if len(seedrows)!=12 or len({(int(r['seed']),r['aggregation'],r['output']) for r in seedrows})!=12:raise ValueError('Missing/duplicate seed table rows')
    for r in seedrows:
        if (float(r['MAE']),float(r['RMSE']))!=expected[int(r['seed']),r['aggregation'],r['output']]:raise ValueError('Seed table metrics differ')
    with (output/'node_metrics.csv').open() as f:noderows=list(csv.DictReader(f))
    keys={(int(r['seed']),r['node']) for r in noderows}
    if len(noderows)!=sum(len(v['per_node']) for v in scores.values()) or len(keys)!=len(noderows):raise ValueError('Missing/duplicate node table rows')
    for r in noderows:
        v=scores[r['seed']]['per_node'][r['node']]
        if int(r['samples'])!=v['samples'] or [float(r[k]) for k in ('power_MAE_W','ambient_MAE_degC')]!=v['mae'] or [float(r[k]) for k in ('power_RMSE_W','ambient_RMSE_degC')]!=v['rmse']:raise ValueError('Node table metrics differ')
    with (output/'aggregate_metrics.csv').open() as f:aggregate_rows=list(csv.DictReader(f))
    summary=aggregate(results)
    if len(aggregate_rows)!=8 or len({(r['aggregation'],r['output'],r['metric']) for r in aggregate_rows})!=8:raise ValueError('Missing/duplicate aggregate table rows')
    for r in aggregate_rows:
        j=('power_W','ambient_degC').index(r['output']);expected=summary[r['aggregation']][r['metric']]
        if not np.allclose([float(r['mean']),float(r['sample_sd'])],[expected['mean'][j],expected['sample_sd'][j]],rtol=1e-12,atol=1e-12):raise ValueError('Aggregate table metrics differ')
    data.assert_unchanged()
    return {'status':'verified_complete','seedset':[42,43,44],'windows_per_seed':len(reference),
        'identical_test_windows':True,'finite_complete_predictions':True,'independent_saved_prediction_metrics':True,
        'model_forward':False,'prediction_inference':False,'metrics_recomputed_from_saved_predictions':True,
        'verification_method':'math.fsum physical residual recomputation; no model forward'}


def verify_evaluation(output,plan,receipt,data):
    output=io.safe_path(output);completed=integrity.read_json(output/'completed.json')
    if completed['status']!='complete' or completed['plan_sha256']!=integrity.digest(plan) or completed['receipt_sha256']!=integrity.digest(receipt):raise ValueError('Evaluation completion identity differs')
    expected=completed['artifacts']
    if {p.name for p in output.iterdir()}!=set(expected)|{'completed.json'}:raise ValueError('Incomplete/extra evaluation files')
    for n,b in expected.items():integrity.check(integrity.child(output,n),b)
    verified=verify_saved_predictions(output,data)
    if integrity.read_json(output/'verification.json')!=verified:raise ValueError('Verification evidence differs')
    return verified


def evaluate(plan,receipt_path,operator_path,device,quota_path):
    lock,scaler,_=integrity.locked_context(plan)
    data=data_module.MarchData(integrity.ROOT/plan['dataset_output']/'dataset',plan,scaler)
    bindings={'plan_sha256':integrity.digest(plan),'selection_lock_sha256':integrity.LOCK_SHA,
        'dataset_manifest':data.manifest_binding,'output':str(integrity.ROOT/plan['evaluation_output']),'device':device}
    operator=integrity.statement(operator_path,'B1_March_test_evaluation_only',bindings)
    receipt=integrity.read_json(receipt_path)
    check_receipt(receipt,receipt_body(plan,lock,data,device,operator))
    destination=integrity.ROOT/plan['evaluation_output']
    budget=io.Budget(destination,io.Limits(**plan['limits']),integrity.read_json(quota_path))
    stage=integrity.stage_directory(destination);budget.charge(len(data.reference)*16*4+16*1024**2)
    allowed=[data.path/'reference.npy',*(data.path/v['path'] for v in data.manifest['nodes'])]
    integrity.guard_test_access(stage,allowed,[integrity.ROOT/m['selected_model']['path'] for m in lock['selected_models']])
    import torch
    from src.b1_real_fedavg_runner_v1 import runtime
    # First model/CUDA operations occur after exact receipt equality above.
    intended,env=runtime(device)
    if device=='cuda:0' and str(torch.cuda.get_device_properties(intended).uuid).removeprefix('GPU-')!=integrity.GPU_UUID.removeprefix('GPU-'):
        raise ValueError('Physical GPU differs')
    scores={};execution=[];started=[]
    try:
        io.exclusive(stage/'authorization_receipt.json',io.canonical(receipt))
        with (stage/'reference.npy').open('xb') as f:np.save(f,data.reference,allow_pickle=False)
        for record in lock['selected_models']:
            budget.check();seed=record['seed'];started.append(seed);begin=time.monotonic()
            model=load_selected(record,lock,scaler,device);fingerprint=model.snapshot(200).fingerprint
            predicted=np.lib.format.open_memmap(stage/f'predictions_seed{seed}.npy',mode='w+',dtype='f8',shape=(len(data.reference),2))
            cursor=0
            with torch.inference_mode():
                for x,y,nodes,origins in data.batches(plan['protocol']['evaluation_batch_size']):
                    budget.check();stop=cursor+len(x);ref=data.reference[cursor:stop]
                    if not np.array_equal(nodes,np.asarray(bridge_names())[ref['node_index']]) or not np.array_equal(origins.view('i8'),ref['origin_ns']) or not np.array_equal(y,_actual(ref)):raise ValueError('Batch identities/observed targets differ before forward')
                    result=model(torch.tensor(np.array(x,copy=True),dtype=torch.float32,device=intended)).cpu().numpy()
                    if result.shape!=y.shape or not np.isfinite(result).all():raise ValueError('Nonfinite/invalid predictions')
                    predicted[cursor:stop]=result;cursor=stop
            if cursor!=len(data.reference) or model.snapshot(200).fingerprint!=fingerprint:raise ValueError('Incomplete inference or modified model')
            predicted.flush();nodes=np.asarray(bridge_names())[data.reference['node_index']];origins=data.reference['origin_ns'].astype('datetime64[ns]')
            scores[seed]=metrics(_actual(data.reference),predicted,nodes,origins)
            execution.append({'seed':seed,'round':200,'seconds':time.monotonic()-begin,'state_unchanged':True,'model':record['selected_model'],
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(intended) if intended.type=='cuda' else None,
                'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(intended) if intended.type=='cuda' else None})
            del predicted,model
        io.exclusive(stage/'metrics.json',io.canonical({str(k):v for k,v in scores.items()}))
        io.exclusive(stage/'aggregate_metrics.json',io.canonical(aggregate(scores)))
        io.exclusive(stage/'execution.json',io.canonical({'environment':env,'seeds':execution,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
        write_tables(stage,scores)
        report='# B1 March test results\n\nEvaluation on the previously inspected retrospective March period; not untouched confirmatory data. All models were locked at round 200 before inference. No training, reselection, scaler fit or tuning.\n\nSee seed_metrics.csv, node_metrics.csv and aggregate_metrics.json for pooled/equal-node macro physical W/°C metrics. Across-seed sample SD uses ddof=1 and is not a confidence interval. Clients are simulated compute nodes at one facility. No centralized/local comparison, energy-saving or formal privacy claim.\n'
        report+='\n|Seed|Power MAE W|Power RMSE W|Ambient MAE °C|Ambient RMSE °C|\n|---|---:|---:|---:|---:|\n'
        for seed,v in scores.items():report+=f"|{seed}|{v['pooled']['mae'][0]:.9f}|{v['pooled']['rmse'][0]:.9f}|{v['pooled']['mae'][1]:.9f}|{v['pooled']['rmse'][1]:.9f}|\n"
        report+=f"\nCoverage: {len(data.reference)} windows on {data.manifest['results']['nodes']} nodes, {data.manifest['results']['origins']} origins and {data.manifest['results']['dates']} UTC dates. Full disjoint exclusion counts are in the frozen test manifest.\n\nTest dataset manifest SHA256: `{data.manifest_binding['sha256']}`. Selection lock SHA256: `{integrity.LOCK_SHA}`. Exact source, checkpoint and scaler bindings are in authorization_receipt.json. Independent verification is in verification.json; raw artifact hashes are in completed.json. Historical preflight FAILED; historical semantic reproduction UNKNOWN.\n"
        io.exclusive(stage/'REPORT.md',report.encode())
        data.assert_unchanged()
        for n,b in plan['sources'].items():integrity.check(integrity.ROOT/n,b)
        for record in lock['selected_models']:integrity.check(integrity.ROOT/record['selected_model']['path'],record['selected_model'])
        verification=verify_saved_predictions(stage,data)
        io.exclusive(stage/'verification.json',io.canonical(verification))
        artifacts=io.inventory(stage)
        io.exclusive(stage/'completed.json',io.canonical({'status':'complete','plan_sha256':integrity.digest(plan),'receipt_sha256':integrity.digest(receipt),'artifacts':artifacts}))
        verification=verify_evaluation(stage,plan,receipt,data)
        integrity.publish(stage,destination)
        return {**verification,'output':str(destination),'inference_completed':True}
    except BaseException as error:
        try:io.exclusive(stage/'failure.json',io.canonical({'status':'incomplete_preserved','models_started':started,'models_completed':sorted(scores),'error':str(error),'automatic_retry':False}))
        except OSError:pass
        raise


def bridge_names():
    return data_module.bridge.ORIGINAL_NAMES


def write_tables(stage,scores):
    seedrows=[];noderows=[]
    for seed,s in scores.items():
        for group in ('pooled','macro_node'):
            for j,name in enumerate(('power_W','ambient_degC')):
                seedrows.append({'seed':seed,'aggregation':group,'output':name,'MAE':s[group]['mae'][j],'RMSE':s[group]['rmse'][j]})
        for node,v in s['per_node'].items():
            noderows.append({'seed':seed,'node':node,'samples':v['samples'],'power_MAE_W':v['mae'][0],'power_RMSE_W':v['rmse'][0],'ambient_MAE_degC':v['mae'][1],'ambient_RMSE_degC':v['rmse'][1]})
    summary=aggregate(scores)
    aggregate_rows=[{'aggregation':group,'output':name,'metric':metric,'mean':summary[group][metric]['mean'][j],'sample_sd':summary[group][metric]['sample_sd'][j]}
        for group in ('pooled','macro_node') for j,name in enumerate(('power_W','ambient_degC')) for metric in ('mae','rmse')]
    for name,rows in [('seed_metrics.csv',seedrows),('node_metrics.csv',noderows),('aggregate_metrics.csv',aggregate_rows)]:
        with (stage/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True)
    p.add_argument('--stage',choices=('describe','authorize','evaluate','verify'),default='describe')
    p.add_argument('--device',choices=('cpu','cuda:0'));p.add_argument('--operator-authorization');p.add_argument('--receipt');p.add_argument('--quota-evidence')
    a=p.parse_args();plan=integrity.load_plan(a.plan)
    if a.stage=='describe':print(json.dumps({'status':'implementation_only','inference':False,'receipt_required':True,'protocol':plan['protocol']}));return
    if a.stage=='authorize':
        if not a.operator_authorization or not a.device:raise PermissionError('Explicit operator assertion/device required')
        print(json.dumps(authorize(plan,a.operator_authorization,a.device)));return
    if a.stage=='evaluate':
        if not all((a.operator_authorization,a.device,a.receipt,a.quota_evidence)):raise PermissionError('Operator assertion, receipt, device and fresh quota evidence required')
        print(json.dumps(evaluate(plan,a.receipt,a.operator_authorization,a.device,a.quota_evidence)));return
    if not a.receipt or not a.device:raise PermissionError('Original receipt and exact device required for verification')
    lock,scaler,_=integrity.locked_context(plan);data=data_module.MarchData(integrity.ROOT/plan['dataset_output']/'dataset',plan,scaler)
    receipt=integrity.read_json(a.receipt);operator=receipt['operator_assertion']
    check_receipt(receipt,receipt_body(plan,lock,data,a.device,operator))
    print(json.dumps(verify_evaluation(integrity.ROOT/plan['evaluation_output'],plan,receipt,data)))

if __name__=='__main__':main()
