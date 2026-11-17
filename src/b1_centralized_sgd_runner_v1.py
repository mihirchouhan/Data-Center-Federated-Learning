"""Additive B1 centralized-SGD control. Defaults to describe; no March loader.

A continuing central model replays B1 FedAvg client/batch exposures exactly.
Real execution requires a separately recorded exact implementation review.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,random,signal,time,traceback
import numpy as np
from src import b1_real_fedavg_runner_v1 as b1
from src import b1_production_data_v1 as data_module
from src import b1_preparation_io_v1 as io
from src.b1_production_io_v1 import RunStore,digest,source_bindings
from src.b1_real_data_bridge_v1 import ORIGINAL_NAMES,initialization_record

ROOT=b1.ROOT
SETTINGS={**b1.FIXED,'aggregation':'none_continuing_central_model',
          'schedule':'exact_B1_FedAvg_client_batch_replay','blocks':200}
SOURCES=[*b1.SOURCES,'src/b1_centralized_sgd_runner_v1.py']
OUTPUT='output/experiments/b1_no_ff/centralized_sgd_control_v1/runs'
LOCK='output/march_window_audit/b1_model_selection_lock_v1/selection_lock.json'
LOCK_SHA='ceb3f469e84fb4beb1de5fc34cef70d1cc626fbc4ae5b96711ce1a8c3e67ebc4'
BASE_PLAN='configs/experiments/b1_no_ff/fedavg_gpu_v1.json'
GPU='GPU-7ef6875b-ba62-88e7-d0b9-e301a25b687e'


def validate_plan(plan):
    keys={'schema','settings','sources','dataset','dataset_manifest_sha256','scaler_lock','scaler_lock_sha256',
          'reference','output','March_access','historical_preflight','semantic_reproduction','budget_decision'}
    if set(plan)!=keys or plan['schema']!='b1.centralized.control.config.v1':raise ValueError('Control config schema differs')
    if io.canonical(plan['settings'])!=io.canonical(SETTINGS):raise ValueError('B1 control protocol differs')
    if plan['dataset']!='output/experiments/b1_no_ff/preparation/train_val_v1/dataset' or plan['dataset_manifest_sha256']!=data_module.DATA_SHA:raise ValueError('Training/validation dataset differs')
    if plan['scaler_lock']!='output/march_window_audit/b1_scaler_approval_and_lock_v1/locked_scaler/lock.json' or plan['scaler_lock_sha256']!=data_module.LOCK_SHA:raise ValueError('Scaler differs')
    if plan['March_access'] is not False or plan['historical_preflight']!='FAILED' or plan['semantic_reproduction']!='UNKNOWN':raise ValueError('Forbidden access/historical status')
    if plan['output']!=OUTPUT or set(plan['sources'])!=set(SOURCES):raise ValueError('Output/source graph differs')
    if plan['budget_decision']!='exact_replay_subject_to_explicit_operator_confirmation':raise ValueError('Budget declaration differs')
    ref=plan['reference']
    if ref['selection_lock']['path']!=LOCK or ref['selection_lock']['sha256']!=LOCK_SHA or ref['plan']['path']!=BASE_PLAN or set(ref['seeds'])!={'42','43','44'}:raise ValueError('Reference identities differ')
    return plan


def load_config(path):
    p=validate_plan(data_module.read_json(path))
    if source_bindings(ROOT,SOURCES)!=p['sources']:raise ValueError('Execution source integrity failure')
    for item in (p['reference']['plan'],p['reference']['selection_lock']):
        if io.binding(ROOT/item['path'])!={k:item[k] for k in ('bytes','sha256')}:raise ValueError('Original B1 reference changed')
    original=b1.load_config(ROOT/BASE_PLAN)
    for key,value in b1.FIXED.items():
        if key!='aggregation' and p['settings'][key]!=value:raise ValueError('B1 hyperparameter differs')
    if original['sources']!={k:p['sources'][k] for k in b1.SOURCES}:raise ValueError('Original B1 source bindings differ')
    return p


def reference_ledgers(plan,data):
    """Read immutable JSON only. Never load or migrate a FedAvg checkpoint."""
    lock=data_module.read_json(ROOT/LOCK);out={}
    if [x['seed'] for x in lock['selected_models']]!=[42,43,44]:raise ValueError('Reference seedset differs')
    for seed in (42,43,44):
        ref=plan['reference']['seeds'][str(seed)];wanted=next(x for x in lock['selected_models'] if x['seed']==seed)
        for key in ('run_manifest','completion','selected_receipt','selected_model'):
            if ref[key]!=wanted[key]:raise ValueError('Locked B1 model/reference differs')
        for key in ('run_manifest','completion','selected_receipt'):
            if io.binding(ROOT/ref[key]['path'])!={k:ref[key][k] for k in ('bytes','sha256')}:raise ValueError('Reference receipt bytes differ')
        manifest=data_module.read_json(ROOT/ref['run_manifest']['path']);complete=data_module.read_json(ROOT/ref['completion']['path'])
        if manifest['seed']!=seed or manifest['settings']!=b1.FIXED or complete['status']!='training_complete' or complete['round']!=200:raise ValueError('Reference run not complete')
        if manifest['data']!=data.binding:raise ValueError('Central and FedAvg data/scaler/eligibility differ')
        if len(ref['round_receipts'])!=201:raise ValueError('Reference round inventory incomplete')
        records=[];previous=None
        for index,entry in enumerate(ref['round_receipts']):
            path=ROOT/entry['path']
            expected=Path(ref['run_manifest']['path']).parent/'rounds'/f'{index:04d}'/'receipt.json'
            if entry['path']!=str(expected) or io.binding(path)!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('Reference round receipt changed')
            row=data_module.read_json(path)
            if row['round']!=index or row['manifest_sha256']!=digest(manifest) or row['previous_receipt_sha256']!=previous:raise ValueError('Reference receipt chain differs')
            previous=entry['sha256'];expected_clients=[] if index==0 else list(b1.select_clients(data.names,seed,index,98))
            if row['selected_clients']!=expected_clients or [c['node'] for c in row['local']]!=expected_clients:raise ValueError('Reference sampling differs')
            for c in row['local']:
                n=data.counts[c['node']];ordinal=ORIGINAL_NAMES.index(c['node']);order,shuffle=b1.shuffled_positions(seed,index,ordinal,n)
                sizes=[min(128,n-first) for first in range(0,n,128)]
                if c['ordinal']!=ordinal or c['examples']!=n or c['batch_sizes']!=sizes or c['optimizer_steps']!=len(sizes) or c['shuffle_seed']!=shuffle or c['row_order_sha256']!=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest():raise ValueError('Reference batch exposure differs')
            if row['examples']!=sum(c['examples'] for c in row['local']) or row['optimizer_steps']!=sum(c['optimizer_steps'] for c in row['local']):raise ValueError('Reference round budgets differ')
            records.append(row)
        if complete['last_round_receipt']!={k:ref['round_receipts'][-1][k] for k in ('bytes','sha256')}:raise ValueError('Reference terminal receipt differs')
        out[seed]=records
    return out


def central_session(model,client,seed,block,batch_size=128):
    """One full training client pass on the SAME continuing central model."""
    import torch
    if batch_size!=128:raise ValueError('Approved batch128 required')
    model.observed_targets(client.future,client.observed)  # before optimizer/forward
    model._validate_state();model.train();model.zero_grad(set_to_none=True)
    start=model.snapshot(0);optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=0.,weight_decay=0.)
    if optimizer.state:raise ValueError('Fresh stateless SGD required')
    order,shuffle=b1.shuffled_positions(seed,block,client.ordinal,len(client));sizes=[];loss_sum=0.;begin=time.monotonic()
    for first in range(0,len(client),128):
        ids=order[first:first+128];x=torch.tensor(np.array(client.x[ids],copy=True),dtype=torch.float32,device=model.intended_device)
        optimizer.zero_grad(set_to_none=True);loss=model.loss(x,client.future[ids],client.observed[ids]);loss.backward()
        for p in model.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all().item():raise ValueError('Missing/nonfinite central gradient')
        grad=model.core.node_embedding.weight.grad
        if torch.count_nonzero(grad[:client.ordinal]).item() or torch.count_nonzero(grad[client.ordinal+1:]).item():raise ValueError('Foreign embedding row gradient')
        optimizer.step();model._validate_state();sizes.append(len(ids));loss_sum+=float(loss.detach().cpu())*len(ids)
    if sum(sizes)!=len(client) or optimizer.state:raise ValueError('Central optimizer/accounting mismatch')
    end=model.snapshot(0)
    return {'node':client.node,'ordinal':client.ordinal,'examples':len(client),'optimizer_steps':len(sizes),'batch_sizes':sizes,
            'shuffle_seed':shuffle,'row_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            'start_fingerprint':start.fingerprint,'end_fingerprint':end.fingerprint,'fresh_optimizer':True,
            'optimizer_state_carried':False,'loss_standardized_MSE':loss_sum/len(client),'seconds':time.monotonic()-begin}


def validate_payload(payload,manifest,data,records,reference):
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    cfg=manifest['settings'];r=payload['round'];seed=manifest['seed']
    if type(r) is not int or not 0<=r<=cfg['rounds'] or r!=len(records)-1:raise ValueError('Control cursor/round cap differs')
    if payload['schema']!='b1.centralized.checkpoint.v1' or payload['manifest_sha256']!=digest(manifest) or payload['root_seed']!=seed or payload['initialization']!=initialization_record(seed):raise ValueError('Control checkpoint identity differs')
    b1.validate_rng(payload['rng'],'cuda' if manifest['environment']['device'].startswith('cuda:') else 'cpu')
    current=RoundSnapshot(payload['state'],b1.state_contract(payload['state'],data.scaler.target_scaling),round_id=r)
    if current.fingerprint!=records[-1]['state_fingerprint'] or RoundSnapshot(payload['state'],current.trainable_keys,round_id=0).fingerprint!=records[-1]['weights_fingerprint'] or payload['optimizer_state'] is not None:raise ValueError('State/optimizer differs')
    if records[0]['state_fingerprint']!=reference[0]['state_fingerprint'] or payload['initial_state_fingerprint']!=reference[0]['state_fingerprint']:raise ValueError('Paired initialization differs')
    trace=payload['validation_history']
    if [v['round'] for v in trace]!=list(range(0,r+1,cfg['validation_every'])):raise ValueError('Validation schedule differs')
    for v in trace:
        ref=data.binding['results']['val']
        if v!=records[v['round']]['validation'] or v['split']!='val' or v['windows']!=ref['windows'] or v['identity_sha256']!=ref['identity_sha256'] or v['target_sha256']!=ref['target_sha256']:raise ValueError('Validation cohort/trace differs')
        if set(v['per_node'])!=set(data.names) or sum(x['samples'] for x in v['per_node'].values())!=v['windows']:raise ValueError('Validation node coverage differs')
        expected_counts=ref.get('per_node')
        if expected_counts and {n:x['samples'] for n,x in v['per_node'].items()}!=expected_counts:raise ValueError('Validation per-node count differs')
        for metric in ('mae','rmse'):
            values=np.asarray(v[metric]);macro=np.asarray(v['macro_node'][metric])
            if values.shape!=(2,) or macro.shape!=(2,) or not np.isfinite(values).all() or not np.isfinite(macro).all() or (values<0).any() or (macro<0).any():raise ValueError('Invalid validation metrics')
            for j in range(2):
                node=np.asarray([x[metric][j] for x in v['per_node'].values()]);weights=np.asarray([x['samples'] for x in v['per_node'].values()])
                derived=float(np.average(node,weights=weights)) if metric=='mae' else math.sqrt(float(np.average(node**2,weights=weights)))
                if not np.isfinite(node).all() or (node<0).any() or not math.isclose(derived,float(values[j]),rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(float(node.mean()),float(macro[j]),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Validation reduction differs')
        j=float(np.mean((np.asarray(v['rmse'])/np.asarray(data.scaler.y_std))**2))
        if not math.isfinite(j) or j!=v['joint_standardized_MSE_J'] or math.sqrt(j)!=v['joint_standardized_RMSE']:raise ValueError('Selection metric differs')
    best=b1.earliest_minimum(trace)
    if (payload['selected_round'],payload['selected_score'])!=(best['round'],best['joint_standardized_RMSE']):raise ValueError('Earliest minimum selection differs')
    chosen=RoundSnapshot(payload['selected_state'],b1.state_contract(payload['selected_state'],data.scaler.target_scaling),round_id=best['round'])
    if chosen.fingerprint!=records[best['round']]['state_fingerprint']:raise ValueError('Selected tensor state differs')
    participation={n:0 for n in data.names};examples=steps=0
    for index,rec in enumerate(records):
        wanted=reference[index];clients=wanted['selected_clients']
        if rec['selected_clients']!=clients or [v['node'] for v in rec['local']]!=clients:raise ValueError('Missing/duplicate central session')
        previous=records[index-1]['weights_fingerprint'] if index else records[0]['weights_fingerprint']
        for c,expected in zip(rec['local'],wanted['local']):
            for key in ('node','ordinal','examples','optimizer_steps','batch_sizes','shuffle_seed','row_order_sha256'):
                if c[key]!=expected[key]:raise ValueError('Central/FedAvg exposure differs: '+key)
            if c['start_fingerprint']!=previous or c['fresh_optimizer'] is not True or c['optimizer_state_carried'] is not False or not math.isfinite(c['loss_standardized_MSE']):raise ValueError('Central continuation/optimizer differs')
            previous=c['end_fingerprint'];participation[c['node']]+=1
        if previous!=rec['weights_fingerprint'] or rec['examples']!=wanted['examples'] or rec['optimizer_steps']!=wanted['optimizer_steps']:raise ValueError('Control block accounting/continuation differs')
        examples+=rec['examples'];steps+=rec['optimizer_steps']
        expected_val=next((v for v in trace if v['round']==index),None)
        if rec['validation']!=expected_val:raise ValueError('Unexpected/missing validation')
    if payload['examples_processed']!=examples or payload['optimizer_steps']!=steps or payload['participation']!=participation or payload['selected_clients']!=records[-1]['selected_clients']:raise ValueError('Cumulative control accounting differs')
    return current,chosen


def review_expected(plan,config_path,device):
    return {'schema':'b1.centralized.independent.review.v1','decision':'APPROVED_EXACT_REPLAY_CONTROL',
            'config':io.binding(config_path),'sources':plan['sources'],'dataset_manifest_sha256':data_module.DATA_SHA,
            'scaler_lock_sha256':data_module.LOCK_SHA,'reference_digest':digest(plan['reference']),
            'seedset':[42,43,44],'device':device,'physical_GPU_UUID':GPU if device=='cuda:0' else None,'March_access':False}


def require_review(path,plan,config_path,device):
    if not path or data_module.read_json(path)!=review_expected(plan,config_path,device):raise ValueError('Exact independent control review required before model initialization')
    return io.binding(path)


def run_seed(data,output,cfg,seed,reference,device='cpu',*,resume=False,stop_after_block=None,sources=None,review_binding=None,reference_binding=None):
    import torch
    from src.b1_device_adapter_v1 import B1DeviceAdapter
    if cfg['kind']=='production':
        if getattr(data,'_verified_token',None) is not data_module.TOKEN or io.canonical(cfg['settings'])!=io.canonical(SETTINGS) or review_binding is None or reference_binding is None or not cfg.get('plan_digest'):raise ValueError('Verified production control prerequisites required')
    elif cfg['kind']!='synthetic' or getattr(data,'scope',None)!='synthetic':raise ValueError('Invented fixture capability required')
    settings=cfg['settings']
    if seed not in (42,43,44) or len(reference)!=settings['rounds']+1 or (stop_after_block is not None and (type(stop_after_block) is not int or not 0<=stop_after_block<=settings['rounds'])):raise ValueError('Seed, cap or reference inventory differs')
    if os.path.lexists(output) and not resume:raise FileExistsError('Output occupied; no implicit restart')
    d,env=b1.runtime(device)
    if d.type=='cuda' and (os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or str(torch.cuda.get_device_properties(d).uuid).removeprefix('GPU-')!=GPU.removeprefix('GPU-')):raise ValueError('Reviewed physical GPU1 required')
    manifest=json.loads(io.canonical({'schema':'b1.centralized.training.manifest.v1','kind':cfg['kind'],'settings':settings,'seed':seed,
        'output':str(io.safe_path(output)),'data':data.binding,'sources':sources or {},'environment':env,'review':review_binding,
        'reference':reference_binding,'plan_digest':cfg.get('plan_digest'),'initialization':initialization_record(seed),'March_access':False,'historical_preflight':'FAILED','semantic_reproduction':'UNKNOWN'}))
    with RunStore(output,manifest,resume=resume) as store:
        try:
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            if d.type=='cuda':torch.cuda.manual_seed(seed)
            if resume:
                payload=store.load();current,selected=validate_payload(payload,manifest,data,store.records,reference)
                if (store.root/'completed.json').exists():
                    if not (store.root/'selected/model.pt').is_file() or payload['round']!=settings['rounds']:raise ValueError('Incomplete completed control')
                    receipt=store.publish_selected({'schema':'b1.centralized.selected.model.v1','state':selected.state_dict(),'manifest_sha256':digest(manifest)}, {'round':payload['selected_round'],'joint_standardized_RMSE':payload['selected_score']});return store.finish(receipt)
            model=B1DeviceAdapter(target_scaling=data.scaler.target_scaling,root_seed=seed,device=d)
            if resume:
                model.restore_round(current);b1.restore_rng(payload['rng'],d);cursor=payload['round'];trace=payload['validation_history'];best_state=selected.state_dict();best_round=payload['selected_round'];best_score=payload['selected_score'];examples=payload['examples_processed'];steps=payload['optimizer_steps'];participation=payload['participation']
            else:
                cursor=0;trace=[];best_state=None;best_round=None;best_score=math.inf;examples=steps=0;participation={n:0 for n in data.names}
                if model.snapshot(0).fingerprint!=reference[0]['state_fingerprint']:raise ValueError('Initial state does not match paired FedAvg seed')
            for block in range(cursor+1 if resume else 0,settings['rounds']+1):
                if stop_after_block is not None and block>stop_after_block:break
                begin=time.monotonic();local=[];clients=reference[block]['selected_clients']
                if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d);torch.cuda.synchronize(d)
                for node in clients:
                    client=data.client(node)
                    if len(client)!=data.counts[node]:raise ValueError('Training client count differs')
                    rec=central_session(model,client,seed,block);local.append(rec);participation[node]+=1;store.event('central_session',{'block':block,**rec})
                n=sum(v['examples'] for v in local);st=sum(v['optimizer_steps'] for v in local);examples+=n;steps+=st
                validation=None
                if block%settings['validation_every']==0:
                    validation=b1.evaluate_validation(model,data,block,settings['evaluation_batch_size']);trace.append(validation)
                    if validation['joint_standardized_RMSE']<best_score:best_round=block;best_score=validation['joint_standardized_RMSE'];best_state=model.snapshot(block).state_dict()
                current=model.snapshot(block)
                payload={'schema':'b1.centralized.checkpoint.v1','manifest_sha256':digest(manifest),'root_seed':seed,'round':block,'state':current.state_dict(),
                    'initial_state_fingerprint':reference[0]['state_fingerprint'],'initialization':initialization_record(seed),'selected_state':best_state,
                    'selected_round':best_round,'selected_score':best_score,'selected_clients':clients,'validation_history':trace,
                    'examples_processed':examples,'optimizer_steps':steps,'participation':participation.copy(),'rng':b1.rng_state(d),'optimizer_state':None}
                if d.type=='cuda':torch.cuda.synchronize(d)
                record={'state_fingerprint':current.fingerprint,'weights_fingerprint':model.snapshot(0).fingerprint,'selected_clients':clients,'local':local,'examples':n,'optimizer_steps':st,'validation':validation,
                    'seconds':time.monotonic()-begin,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(d) if d.type=='cuda' else None,
                    'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(d) if d.type=='cuda' else None}
                validate_payload(payload,manifest,data,[*store.records,{**record,'round':block}],reference);store.commit(block,payload,record);cursor=block
                print(json.dumps({'method':'centralized_sgd','seed':seed,'block':block,'selected_block':best_round,'examples':examples,'steps':steps,'seconds':record['seconds']}),flush=True)
            if cursor!=settings['rounds']:
                store.event('stopped_at_requested_boundary',{'block':cursor,'fit_complete':False});return {'status':'verified_partial','completed_round':cursor,'selected_round':best_round}
            payload=store.load();current,selected=validate_payload(payload,manifest,data,store.records,reference);data.assert_unchanged();store.verify_sources()
            receipt=store.publish_selected({'schema':'b1.centralized.selected.model.v1','state':selected.state_dict(),'manifest_sha256':digest(manifest)}, {'round':payload['selected_round'],'joint_standardized_RMSE':payload['selected_score']})
            result=store.finish(receipt);store.event('finished',result);return result
        except BaseException as error:
            try:store.event('failure',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'last_committed_block':len(store.records)-1,'recovery':'Explicit verified resume only; preserve pending artifacts'})
            except OSError:pass
            raise


def verify_run(data,output,plan,seed,reference):
    import fcntl,torch
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    store=RunStore.__new__(RunStore);store.root=io.safe_path(output);store.project_root=ROOT;store.manifest=data_module.read_json(store.root/'manifest.json');m=store.manifest
    kind='production' if getattr(data,'_verified_token',None) is data_module.TOKEN else 'synthetic' if getattr(data,'scope',None)=='synthetic' else None
    if not kind or m['schema']!='b1.centralized.training.manifest.v1' or m['kind']!=kind or m['seed']!=seed or m['output']!=str(store.root) or m['settings']!=plan['settings'] or m['sources']!=plan['sources'] or m['data']!=data.binding:raise ValueError('Control run identity differs')
    if kind=='production' and (m.get('reference')!=plan['reference']['seeds'][str(seed)] or m.get('plan_digest')!=digest(plan) or m.get('review') is None):raise ValueError('Control configuration/reference/review binding differs')
    if data_module.read_json(store.root/'owner.json')['uid']!=os.getuid():raise ValueError('Foreign run owner')
    fd=os.open(store.root/'writer.lock',os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);store.verify_sources();store.records=store.scan();payload=store.load();_,chosen=validate_payload(payload,m,data,store.records,reference);status='verified_partial'
        if (store.root/'completed.json').exists():
            c=data_module.read_json(store.root/'completed.json');r=data_module.read_json(store.root/'selected/receipt.json')
            if c['status']!='training_complete' or c['round']!=plan['settings']['rounds'] or payload['round']!=c['round'] or c['manifest_sha256']!=digest(m) or r['manifest_sha256']!=digest(m) or r['schema']!='b1.production.selected.v1':raise ValueError('Completion/control identity differs')
            if c['selected_receipt']!=io.binding(store.root/'selected/receipt.json') or c['last_round_receipt']!=io.binding(store.root/'rounds'/f"{c['round']:04d}"/'receipt.json') or r['model']!=io.binding(store.root/'selected/model.pt') or c['selection']!=r['selection'] or r['selection']!={'round':payload['selected_round'],'joint_standardized_RMSE':payload['selected_score']}:raise ValueError('Completion/selected receipt hashes differ')
            with io.safe_open(store.root/'selected/model.pt') as f:export=torch.load(f,map_location='cpu',weights_only=True)
            if io.binding(store.root/'selected/model.pt')!=r['model'] or export['schema']!='b1.centralized.selected.model.v1' or export['manifest_sha256']!=digest(m) or RoundSnapshot(export['state'],b1.state_contract(export['state'],data.scaler.target_scaling),round_id=payload['selected_round']).fingerprint!=chosen.fingerprint:raise ValueError('Selected tensor state differs')
            status='verified_complete'
        data.assert_unchanged();store.verify_sources();return {'status':status,'completed_round':payload['round'],'selected_round':payload['selected_round'],'model_forward_or_scoring':False,'method':'centralized_sgd'}
    finally:os.close(fd)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--stage',choices=('describe','preflight','fit','verify'),default='describe')
    p.add_argument('--device',choices=('cpu','cuda:0'));p.add_argument('--matrix',action='store_true');p.add_argument('--seed',type=int,choices=(42,43,44));p.add_argument('--review-receipt');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-block',type=int)
    a=p.parse_args(argv);plan=load_config(a.plan)
    if a.stage=='describe':print(json.dumps({'status':'implementation_only','settings':plan['settings'],'output':plan['output'],'training':False,'review_required':True}));return
    if a.stage=='fit':
        if not a.device:raise ValueError('Explicit device required')
        review=require_review(a.review_receipt,plan,a.plan,a.device)
        if a.device=='cuda:0' and (os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('Exact GPU1/determinism environment required')
    data=data_module.PreparedData(ROOT/plan['dataset'],ROOT/plan['scaler_lock']);output=ROOT/plan['output']
    # Install before reading references or any future model initialization.
    data_module.install_access_guard(data,output);refs=reference_ledgers(plan,data)
    if a.stage=='preflight':
        print(json.dumps({'status':'data_and_schedule_verified','binding':data.binding,'budgets':{str(s):{'examples':sum(r['examples'] for r in rs),'steps':sum(r['optimizer_steps'] for r in rs)} for s,rs in refs.items()},'model_initialized':False,'March_access':False}));return
    if (a.seed is None)==(not a.matrix):raise ValueError('Choose exactly one seed or explicit matrix')
    seeds=(42,43,44) if a.matrix else (a.seed,)
    if a.stage=='verify':
        if a.resume or a.stop_after_block is not None:raise ValueError('Verify cannot change execution')
        for seed in seeds:print(json.dumps(verify_run(data,output/f'seed{seed}',plan,seed,refs[seed])))
        return
    def interrupted(signum,frame):raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    if not output.parent.is_dir() or output.parent.stat().st_uid!=os.getuid():raise ValueError('Owned output parent required')
    import fcntl
    matrix={'schema':'b1.centralized.matrix.v1','plan':io.binding(a.plan),'review':review,'device':a.device,'seeds':list(seeds),'data':data.binding}
    if a.resume:
        if data_module.read_json(output/'matrix.json')!=matrix:raise ValueError('Matrix resume identity differs')
        fd=os.open(io.safe_path(output/'matrix.lock'),os.O_RDWR|os.O_NOFOLLOW)
    else:
        if os.path.lexists(output):raise FileExistsError('Control output occupied; no implicit resume')
        output.mkdir(mode=0o700);fd=os.open(output/'matrix.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not a.resume:io.exclusive(output/'matrix.json',io.canonical(matrix))
        existing=[seed for seed in seeds if (output/f'seed{seed}').exists()]
        if (output/'matrix_completed.json').exists() and (existing!=list(seeds) or any(not (output/f'seed{s}/completed.json').is_file() for s in seeds)):raise ValueError('Completed matrix missing seeds; never recreate')
        if existing!=list(seeds)[:len(existing)] or any(not (output/f'seed{s}/completed.json').exists() for s in existing[:-1]):raise ValueError('Missing earlier seed or later seed before completion')
        if set(x.name for x in output.iterdir())-{'matrix.json','matrix.lock','matrix_completed.json',*(f'seed{s}' for s in seeds)}:raise ValueError('Unexpected matrix artifacts')
        for seed in seeds:
            path=output/f'seed{seed}'
            result=run_seed(data,path,{'kind':'production','settings':SETTINGS,'plan_digest':digest(plan)},seed,refs[seed],a.device,resume=path.exists(),stop_after_block=a.stop_after_block,sources=plan['sources'],review_binding=review,reference_binding=plan['reference']['seeds'][str(seed)])
            if result['status']!='training_complete':print(json.dumps(result));break
            checked=verify_run(data,path,plan,seed,refs[seed])
            if checked['status']!='verified_complete':raise ValueError('Next seed blocked by verification')
            print(json.dumps(checked),flush=True)
        else:
            complete={'status':'verified_complete','seeds':{str(s):io.binding(output/f'seed{s}/completed.json') for s in seeds}}
            if (output/'matrix_completed.json').exists():
                if data_module.read_json(output/'matrix_completed.json')!=complete:raise ValueError('Matrix completion differs')
            else:io.exclusive(output/'matrix_completed.json',io.canonical(complete))
    finally:os.close(fd)

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as error:
        import sys
        print('B1 CENTRALIZED SAFETY STOP: '+type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(130 if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 1)
