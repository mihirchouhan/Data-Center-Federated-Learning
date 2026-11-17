"""B1 production FedAvg, additive to all synthetic/historical runners.

CLI defaults to no stage: explicit preflight/fit/verify required. Real fit needs
an independently supplied review record. No test loader or test evaluation API.
"""
from collections import Counter
from pathlib import Path
import argparse,copy,hashlib,json,math,os,platform,random,signal,time,traceback
import numpy as np
from src import b1_preparation_io_v1 as io
from src import b1_production_data_v1 as data_module
from src.b1_production_io_v1 import RunStore,digest,source_bindings
from src.b1_real_data_bridge_v1 import ORIGINAL_NAMES,MAPPING_DIGEST,initialization_record

FIXED={'history':36,'horizon':6,'features':14,'outputs':2,'hidden_size':64,'layers':2,
       'embedding_size':8,'residual':False,'clients':979,'clients_per_round':98,
       'batch_size':128,'local_epochs':1,'drop_last':False,'rounds':200,
       'seeds':[42,43,44],'learning_rate':0.01,'momentum':0.0,'weight_decay':0.0,
       'validation_every':10,'evaluation_batch_size':1024,'num_threads':1,
       'aggregation':'full_table_sample_weighted_CPU_float64',
       'selection':'earliest_strict_minimum_joint_standardized_RMSE_sqrt_J'}
SOURCES=['src/b1_device_adapter_v1.py','src/b1_production_data_v1.py','src/b1_production_io_v1.py','src/b1_real_fedavg_runner_v1.py',
 'src/b1_model_adapter_v1.py','src/b1_target_validation_v1.py','src/b1_target_scaling_v1.py','src/b1_bridge_scaling_v1.py',
 'src/b1_scaler_acceptance_v1.py','src/b1_fedavg_orchestration_v1.py','src/b1_real_data_bridge_v1.py','src/model.py',
 'src/b1_preparation_execution_v1.py','src/b1_preparation_io_v1.py','src/b1_raw_window_producer_v1.py',
 'src/history_data_v4.py','src/multinode.py','src/preprocess.py','src/phase2_multioutput/contracts.py','src/phase1_3/schema.py','src/phase1_3/config.py']
ROOT=Path(data_module.bridge.__file__).absolute().parents[1]


def load_config(path):
    plan=data_module.read_json(path)
    if set(plan)!={'schema','settings','sources','dataset','scaler_lock','dataset_manifest_sha256','scaler_lock_sha256','March_access','historical_preflight','semantic_reproduction'}:
        raise ValueError('Unexpected production configuration fields')
    if plan['schema']!='b1.production.fedavg.config.v1' or io.canonical(plan['settings'])!=io.canonical(FIXED) or plan['March_access'] is not False or plan['historical_preflight']!='FAILED' or plan['semantic_reproduction']!='UNKNOWN': raise ValueError('Production protocol differs')
    if plan['dataset']!='output/experiments/b1_no_ff/preparation/train_val_v1/dataset' or plan['scaler_lock']!='output/march_window_audit/b1_scaler_approval_and_lock_v1/locked_scaler/lock.json':raise ValueError('Foreign input path')
    if plan['dataset_manifest_sha256']!=data_module.DATA_SHA or plan['scaler_lock_sha256']!=data_module.LOCK_SHA: raise ValueError('Wrong immutable data/scaler')
    if set(plan['sources'])!=set(SOURCES) or source_bindings(ROOT,SOURCES)!=plan['sources']: raise ValueError('Execution source integrity failure')
    return plan


def require_review(path, plan, config_path, device):
    if not path: raise ValueError('Independent review receipt required before real model initialization or training')
    r=data_module.read_json(path)
    expected={'schema':'b1.production.independent.review.v1','decision':'APPROVED_FOR_B1_TRAIN_VAL_TRAINING',
              'config':io.binding(config_path),'sources':plan['sources'],'dataset_manifest_sha256':data_module.DATA_SHA,
              'scaler_lock_sha256':data_module.LOCK_SHA,'device':device,'March_access':False}
    if r!=expected: raise ValueError('Independent review does not match exact implementation/configuration/device/data/scaler')
    return io.binding(path)


def runtime(device):
    import torch
    from src.b1_device_adapter_v1 import explicit_device
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8')!=':4096:8': raise ValueError('Unexpected deterministic CUBLAS setting')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    d=explicit_device(device)
    if d.type=='cuda':torch.cuda.set_device(d)
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':str(torch.__version__),
         'device':str(d),'torch_cuda':torch.version.cuda,'deterministic_algorithms':True,
         'num_threads':1,'TF32':False,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
         'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}
    if d.type=='cuda':
        props=torch.cuda.get_device_properties(d)
        env.update(device_name=props.name,device_uuid=str(props.uuid),device_total_memory=props.total_memory,
                   cudnn=torch.backends.cudnn.version())
    else:env['machine']=platform.machine()
    return d,env


def rng_state(device):
    import torch
    ns=np.random.get_state()
    return {'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])],
            'torch_cpu':torch.random.get_rng_state(),'torch_device':torch.cuda.get_rng_state(device).cpu() if device.type=='cuda' else None}


def validate_rng(record, device_type):
    import torch
    random.Random().setstate(record['python'])
    ns=record['numpy'];probe=np.random.RandomState(0);probe.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    state=record['torch_cpu']
    if not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or state.device.type!='cpu' or state.ndim!=1:raise ValueError('Invalid CPU RNG state')
    torch.Generator(device='cpu').set_state(state)
    gpu=record['torch_device']
    if device_type=='cuda':
        if not isinstance(gpu,torch.Tensor) or gpu.dtype!=torch.uint8 or gpu.device.type!='cpu' or gpu.ndim!=1 or not len(gpu):raise ValueError('Invalid CUDA RNG state')
    elif gpu is not None:raise ValueError('CUDA RNG on CPU run')


def restore_rng(record, device):
    import torch
    random.setstate(record['python']);s=record['numpy'];np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),s[2],s[3],s[4]))
    torch.random.set_rng_state(record['torch_cpu'])
    if device.type=='cuda':torch.cuda.set_rng_state(record['torch_device'],device)
    elif record['torch_device'] is not None:raise ValueError('CUDA checkpoint cannot resume as CPU')


def select_clients(names, seed, round_id, count):
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1,round_id,0])))
    return tuple(names[int(i)] for i in rng.choice(len(names),size=count,replace=False))


def shuffled_positions(seed, round_id, ordinal, count):
    shuffle=int(np.random.SeedSequence([seed,2,round_id,ordinal]).generate_state(1)[0])
    return np.random.Generator(np.random.PCG64(shuffle)).permutation(count),shuffle


def local_update(global_model, client, seed, round_id, batch_size=128):
    import torch
    from src.b1_fedavg_orchestration_v1 import ClientStateUpdate,batch_sizes
    # Validate every original target BEFORE any optimizer or model execution.
    global_model.observed_targets(client.future,client.observed)
    base=global_model.snapshot(round_id-1);local=copy.deepcopy(global_model);local.train();local.zero_grad(set_to_none=True)
    if any(p.data_ptr()==q.data_ptr() for p,q in zip(local.parameters(),global_model.parameters())):raise ValueError('Client parameter alias')
    if local.snapshot(round_id-1).fingerprint!=base.fingerprint:raise ValueError('Client did not start from common global state')
    optimizer=torch.optim.SGD(local.parameters(),lr=.01,momentum=0.,weight_decay=0.)
    if optimizer.state:raise ValueError('Optimizer state must be fresh')
    order,shuffle=shuffled_positions(seed,round_id,client.ordinal,len(client))
    sizes=[];total=0;loss_sum=0.;start=time.monotonic()
    for first in range(0,len(client),batch_size):
        ids=order[first:first+batch_size]
        x=torch.tensor(np.array(client.x[ids],copy=True),dtype=torch.float32,device=local.intended_device)
        optimizer.zero_grad(set_to_none=True);loss=local.loss(x,client.future[ids],client.observed[ids]);loss.backward()
        for p in local.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all().item():raise ValueError('Missing/nonfinite gradient')
        gradient=local.core.node_embedding.weight.grad.detach()
        other=torch.cat((gradient[:client.ordinal],gradient[client.ordinal+1:]))
        if torch.count_nonzero(other).item():raise ValueError('Another client embedding row received gradient')
        optimizer.step();local._validate_state();sizes.append(len(ids));total+=len(ids);loss_sum+=float(loss.detach().cpu())*len(ids)
    if tuple(sizes)!=batch_sizes(len(client),batch_size) or total!=len(client) or optimizer.state:raise ValueError('Local budget or optimizer state mismatch')
    state=local.snapshot(round_id-1).state_dict()
    receipt={'node':client.node,'ordinal':client.ordinal,'examples':total,'optimizer_steps':len(sizes),'batch_sizes':sizes,
             'shuffle_seed':shuffle,'row_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
             'base_fingerprint':base.fingerprint,'loss_standardized_MSE':loss_sum/total,
             'fresh_optimizer':True,'optimizer_state_carried':False,'seconds':time.monotonic()-start,
             'embedding_row_delta_L2':float(torch.linalg.vector_norm(state['core.node_embedding.weight'][client.ordinal]-base.state_dict()['core.node_embedding.weight'][client.ordinal]))}
    if global_model.snapshot(round_id-1).fingerprint!=base.fingerprint:raise ValueError('Global state mutated by client')
    return ClientStateUpdate(client.node,total,base.fingerprint,state),receipt


def validation_batches(data, size):
    pending=[];n=0
    for batch in data.validation_batches():
        pending.append(batch);n+=len(batch[0])
        if n>=size:
            joined=tuple(np.concatenate([p[j] for p in pending]) for j in range(4))
            while len(joined[0])>=size:
                yield tuple(v[:size] for v in joined);joined=tuple(v[size:] for v in joined)
            pending=[joined] if len(joined[0]) else [];n=len(joined[0])
    if pending:yield tuple(np.concatenate([p[j] for p in pending]) for j in range(4))


def evaluate_validation(model, data, index, batch_size=1024):
    import torch
    before=model.snapshot(index).fingerprint;mode=model.training;model.eval()
    sse=np.zeros(2);sae=np.zeros(2);by={};ids=hashlib.sha256();targets=hashlib.sha256();n=0
    try:
        with torch.inference_mode():
            for x,y,nodes,origins in validation_batches(data,batch_size):
                predicted=model(torch.tensor(np.array(x,copy=True),dtype=torch.float32,device=model.intended_device)).cpu().numpy()
                if predicted.shape!=y.shape or not np.isfinite(predicted).all() or not np.isfinite(y).all():raise ValueError('Invalid complete validation outputs')
                with np.errstate(over='raise',invalid='raise'):
                    err=predicted-y;abs_err=np.abs(err);sq=err*err;sse+=sq.sum(axis=0,dtype=np.float64);sae+=abs_err.sum(axis=0,dtype=np.float64)
                for node in np.unique(nodes):
                    keep=nodes==node;entry=by.setdefault(str(node),{'samples':0,'sse':np.zeros(2),'sae':np.zeros(2)});entry['samples']+=int(keep.sum());entry['sse']+=sq[keep].sum(axis=0);entry['sae']+=abs_err[keep].sum(axis=0)
                for node,origin in zip(nodes.tolist(),origins.view('i8').tolist()):
                    row=json.dumps([node,origin],ensure_ascii=True,separators=(',',':')).encode();ids.update(len(row).to_bytes(8,'big'));ids.update(row)
                targets.update(y.astype('<f8').tobytes());n+=len(y)
    finally:model.train(mode)
    reference=data.binding['results']['val']
    if n!=reference['windows'] or ids.hexdigest()!=reference['identity_sha256'] or targets.hexdigest()!=reference['target_sha256']:raise ValueError('Validation cohort/identity/targets changed or incomplete')
    rmse=np.sqrt(sse/n);mae=sae/n;std=np.asarray(data.scaler.y_std);j=float(np.mean((rmse/std)**2));score=math.sqrt(j)
    if not math.isfinite(score):raise ValueError('Nonfinite joint validation score')
    if model.snapshot(index).fingerprint!=before:raise ValueError('Validation changed model weights')
    per={node:{'samples':v['samples'],'mae':(v['sae']/v['samples']).tolist(),'rmse':np.sqrt(v['sse']/v['samples']).tolist()} for node,v in by.items()}
    return {'round':index,'split':'val','windows':n,'identity_sha256':ids.hexdigest(),'target_sha256':targets.hexdigest(),
            'units':['W','degC'],'rmse':rmse.tolist(),'mae':mae.tolist(),'joint_standardized_MSE_J':j,
            'joint_standardized_RMSE':score,'per_node':per,'macro_node':{k:np.mean([v[k] for v in per.values()],axis=0).tolist() for k in ('mae','rmse')}}


def earliest_minimum(trace):
    if not trace:raise ValueError('Validation history required')
    best=trace[0]
    for row in trace[1:]:
        if row['joint_standardized_RMSE']<best['joint_standardized_RMSE']:best=row
    return best


def state_contract(state, target_scaling):
    import torch
    shape={'core.node_embedding.weight':(979,8),'core.head.weight':(2,64),'core.head.bias':(2,), 'target_mean':(2,), 'target_std':(2,)}
    for layer in (0,1):
        shape[f'core.recurrent.weight_ih_l{layer}']=(192,21 if layer==0 else 64)
        shape[f'core.recurrent.weight_hh_l{layer}']=(192,64)
        shape[f'core.recurrent.bias_ih_l{layer}']=(192,);shape[f'core.recurrent.bias_hh_l{layer}']=(192,)
    if set(state)!=set(shape):raise ValueError('Checkpoint state keys differ from B1 architecture')
    for k,t in state.items():
        expected=torch.float64 if k in ('target_mean','target_std') else torch.float32
        if not isinstance(t,torch.Tensor) or t.device.type!='cpu' or tuple(t.shape)!=shape[k] or t.dtype!=expected or not torch.isfinite(t).all():raise ValueError('Invalid checkpoint tensor '+k)
    if not torch.equal(state['target_mean'],torch.tensor(target_scaling.mean,dtype=torch.float64)) or not torch.equal(state['target_std'],torch.tensor(target_scaling.std,dtype=torch.float64)):raise ValueError('Checkpoint target scale changed')
    return tuple(k for k in state if k not in ('target_mean','target_std'))


def validate_payload(payload, manifest, data, records):
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    cfg=manifest['settings'];r=payload['round']
    validate_rng(payload['rng'],'cuda' if manifest['environment']['device'].startswith('cuda:') else 'cpu')
    if type(r) is not int or not 0<=r<=cfg['rounds']:raise ValueError('Round cursor outside configured cap')
    if payload['schema']!='b1.production.checkpoint.v1' or payload['manifest_sha256']!=digest(manifest) or payload['root_seed']!=manifest['seed'] or r!=len(records)-1:raise ValueError('Checkpoint schema/identity/cursor mismatch')
    if payload['initialization']!=initialization_record(manifest['seed']):raise ValueError('Initialization stream differs')
    roles=state_contract(payload['state'],data.scaler.target_scaling)
    current=RoundSnapshot(payload['state'],roles,round_id=r)
    if payload['initial_state_fingerprint']!=records[0]['state_fingerprint'] or payload['optimizer_state'] is not None or payload['selected_clients']!=records[r]['selected_clients']:raise ValueError('Initial state, optimizer-independent recovery or participants differ')
    if current.fingerprint!=records[r]['state_fingerprint']:raise ValueError('State fingerprint differs')
    indices=list(range(0,r+1,cfg['validation_every']))
    trace=payload['validation_history']
    if [v['round'] for v in trace]!=indices:raise ValueError('Incomplete validation schedule')
    for row in trace:
        if row!=records[row['round']]['validation']:raise ValueError('Historical validation trace differs from immutable receipt')
        for metric in ('mae','rmse'):
            values=np.asarray(row[metric])
            if values.shape!=(2,) or not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid physical validation metrics')
        if sum(v['samples'] for v in row['per_node'].values())!=row['windows']:raise ValueError('Validation per-node counts incomplete')
        expected_nodes=data.binding['results']['val'].get('per_node')
        if expected_nodes is not None and {k:v['samples'] for k,v in row['per_node'].items()}!=expected_nodes:raise ValueError('Validation node coverage differs')
        ref=data.binding['results']['val']
        if row['split']!='val' or row['windows']!=ref['windows'] or row['identity_sha256']!=ref['identity_sha256'] or row['target_sha256']!=ref['target_sha256']:raise ValueError('Validation provenance differs')
        score=math.sqrt(float(np.mean((np.asarray(row['rmse'])/np.asarray(data.scaler.y_std))**2)))
        if not math.isfinite(score) or score!=row['joint_standardized_RMSE'] or float(np.mean((np.asarray(row['rmse'])/np.asarray(data.scaler.y_std))**2))!=row['joint_standardized_MSE_J']:raise ValueError('Validation score differs')
    best=earliest_minimum(trace)
    if payload['selected_round']!=best['round'] or payload['selected_score']!=best['joint_standardized_RMSE']:raise ValueError('Selected checkpoint is not earliest validation minimum')
    broles=state_contract(payload['selected_state'],data.scaler.target_scaling)
    selected=RoundSnapshot(payload['selected_state'],broles,round_id=best['round'])
    if selected.fingerprint!=records[best['round']]['state_fingerprint']:raise ValueError('Selected weights differ from recorded validation round')
    part={n:0 for n in data.names};examples=steps=0
    for i,rec in enumerate(records):
        wanted=[] if i==0 else list(select_clients(data.names,manifest['seed'],i,cfg['clients_per_round']))
        if rec['selected_clients']!=wanted:raise ValueError('Round participation differs')
        local=rec['local'];count=0;st=0
        if [v['node'] for v in local]!=wanted:raise ValueError('Missing/duplicate local updates')
        for v in local:
            node=v['node'];n=data.counts[node];order,shuffle=shuffled_positions(manifest['seed'],i,ORIGINAL_NAMES.index(node),n)
            if v['examples']!=n or v['optimizer_steps']!=math.ceil(n/128) or v['shuffle_seed']!=shuffle or v['row_order_sha256']!=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() or not v['fresh_optimizer']:raise ValueError('Local budget/shuffle receipt differs')
            if v['base_fingerprint']!=records[i-1]['state_fingerprint']:raise ValueError('Client base state differs')
            part[node]+=1;count+=n;st+=math.ceil(n/128)
        if rec['examples']!=count or rec['optimizer_steps']!=st:raise ValueError('Round accounting differs')
        examples+=count;steps+=st
    if payload['examples_processed']!=examples or payload['optimizer_steps']!=steps or payload['participation']!=part:raise ValueError('Cumulative accounting differs')
    if records[r]['validation']!=(trace[-1] if r%cfg['validation_every']==0 else None):raise ValueError('Latest validation receipt differs')
    return current,selected


def run_seed(data, output, cfg, seed, device='cpu', *, resume=False, stop_after_round=None, sources=None, review_binding=None):
    """Engine tested on invented capabilities; CLI only supplies verified production data."""
    import torch
    from src.b1_device_adapter_v1 import B1DeviceAdapter
    from src.b1_fedavg_orchestration_v1 import aggregate_fedavg
    if cfg.get('kind')=='production':
        if getattr(data,'_verified_token',None) is not data_module.TOKEN or io.canonical(cfg['settings'])!=io.canonical(FIXED) or review_binding is None:raise ValueError('Verified production data, fixed protocol and independent review required')
    elif cfg.get('kind')!='synthetic' or getattr(data,'scope',None)!='synthetic':raise ValueError('Unknown execution capability')
    settings=cfg['settings'];data.names=tuple(data.names)
    if seed not in (42,43,44) or (stop_after_round is not None and (type(stop_after_round) is not int or not 0<=stop_after_round<=settings['rounds'])):raise ValueError('Invalid seed/stop boundary')
    d,env=runtime(device)
    manifest={'schema':'b1.production.training.manifest.v1','kind':cfg['kind'],'settings':settings,'seed':seed,
              'output':str(io.safe_path(output)),'data':data.binding,'sources':sources or {},'environment':env,'review':review_binding,
              'initialization':initialization_record(seed),'March_access':False,
              'historical_preflight':'FAILED','semantic_reproduction':'UNKNOWN'}
    manifest=json.loads(io.canonical(manifest))  # Freeze caller containers before binding checkpoints.
    with RunStore(output,manifest,resume=resume) as store:
        try:
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            if d.type=='cuda':torch.cuda.manual_seed(seed)
            if resume:
                payload=store.load();current,selected=validate_payload(payload,manifest,data,store.records)
                if (store.root/'completed.json').exists():
                    if not (store.root/'selected'/'receipt.json').is_file() or not (store.root/'selected'/'model.pt').is_file():raise ValueError('Completed run missing selected export; never reconstruct silently')
                    if payload['round']!=settings['rounds']:raise ValueError('Premature completed marker')
                    receipt=store.publish_selected({'schema':'b1.production.selected.model.v1','state':selected.state_dict(),'manifest_sha256':digest(manifest)}, {'round':payload['selected_round'],'joint_standardized_RMSE':payload['selected_score']})
                    return store.finish(receipt)
            # No model initialization occurs before review, data and resume byte checks.
            model=B1DeviceAdapter(target_scaling=data.scaler.target_scaling,root_seed=seed,device=d)
            if resume:
                model.restore_round(current);restore_rng(payload['rng'],d);cursor=payload['round']
                initial=payload['initial_state_fingerprint'];trace=payload['validation_history'];best_state=selected.state_dict();best_round=payload['selected_round'];best_score=payload['selected_score'];examples=payload['examples_processed'];steps=payload['optimizer_steps'];participation=payload['participation']
            else:
                cursor=0;initial=model.snapshot(0).fingerprint;trace=[];best_state=None;best_round=None;best_score=math.inf;examples=steps=0;participation={n:0 for n in data.names}
            first=cursor+1 if resume else 0
            for round_id in range(first,settings['rounds']+1):
                if stop_after_round is not None and round_id>stop_after_round:break
                if d.type=='cuda':torch.cuda.reset_peak_memory_stats(d);torch.cuda.synchronize(d)
                started=time.monotonic();local_receipts=[];selected_clients=[];nr=ns=0
                if round_id:
                    before=model.snapshot(round_id-1);selected_clients=list(select_clients(data.names,seed,round_id,settings['clients_per_round']));updates=[]
                    for node in selected_clients:
                        client=data.client(node)
                        if len(client)!=data.counts[node]:raise ValueError('Client population differs')
                        update,rec=local_update(model,client,seed,round_id,128);updates.append(update);local_receipts.append(rec)
                        store.event('client_complete',{'round':round_id,**rec})
                    merged=aggregate_fedavg(before,updates,expected_counts={n:data.counts[n] for n in selected_clients},client_order=selected_clients)
                    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
                    model.restore_round(RoundSnapshot(merged,before.trainable_keys,round_id=round_id))
                    nr=sum(v['examples'] for v in local_receipts);ns=sum(v['optimizer_steps'] for v in local_receipts)
                    examples+=nr;steps+=ns
                    for n in selected_clients:participation[n]+=1
                val=None
                if round_id%settings['validation_every']==0:
                    val=evaluate_validation(model,data,round_id,settings['evaluation_batch_size']);trace.append(val)
                    if val['joint_standardized_RMSE']<best_score:
                        best_round=round_id;best_score=val['joint_standardized_RMSE'];best_state=model.snapshot(round_id).state_dict()
                now=model.snapshot(round_id)
                payload={'schema':'b1.production.checkpoint.v1','manifest_sha256':digest(manifest),'root_seed':seed,'round':round_id,
                         'state':now.state_dict(),'initial_state_fingerprint':initial,'initialization':initialization_record(seed),
                         'selected_state':best_state,'selected_round':best_round,'selected_score':best_score,
                         'selected_clients':selected_clients,'validation_history':trace,'examples_processed':examples,
                         'optimizer_steps':steps,'participation':participation.copy(),'rng':rng_state(d),'optimizer_state':None}
                if d.type=='cuda':torch.cuda.synchronize(d)
                record={'state_fingerprint':now.fingerprint,'selected_clients':selected_clients,'examples':nr,'optimizer_steps':ns,'local':local_receipts,'validation':val,
                        'seconds':time.monotonic()-started,'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(d) if d.type=='cuda' else None,
                        'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(d) if d.type=='cuda' else None}
                # Validate next checkpoint semantics before committing it.
                provisional={**record,'round':round_id};validate_payload(payload,manifest,data,[*store.records,provisional])
                store.commit(round_id,payload,record);cursor=round_id
                print(json.dumps({'seed':seed,'round':round_id,'selected_round':best_round,'joint_standardized_RMSE':val['joint_standardized_RMSE'] if val else None,'seconds':record['seconds']}),flush=True)
            if cursor!=settings['rounds']:
                store.event('stopped_at_requested_boundary',{'round':cursor,'fit_complete':False})
                return {'status':'verified_partial','completed_round':cursor,'selected_round':best_round}
            payload=store.load();current,selected=validate_payload(payload,manifest,data,store.records);data.assert_unchanged();store.verify_sources()
            receipt=store.publish_selected({'schema':'b1.production.selected.model.v1','state':selected.state_dict(),'manifest_sha256':digest(manifest)}, {'round':payload['selected_round'],'joint_standardized_RMSE':payload['selected_score']})
            result=store.finish(receipt);store.event('finished',result);return result
        except BaseException as error:
            try:store.event('failure',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'last_committed_round':len(store.records)-1,'recovery':'explicit --resume from verified committed checkpoint; partial directories retained'})
            except OSError:pass
            raise


def verify_run(data, output, plan, seed):
    """Read-only byte/receipt/tensor/selection validation; no model/inference/GPU init."""
    import fcntl
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    store=RunStore.__new__(RunStore);store.root=io.safe_path(output);store.manifest=data_module.read_json(store.root/'manifest.json');store.project_root=ROOT
    m=store.manifest
    if m.get('output')!=str(store.root):raise ValueError('Run output identity differs; no implicit checkpoint migration')
    scope='production' if getattr(data,'_verified_token',None) is data_module.TOKEN else ('synthetic' if getattr(data,'scope',None)=='synthetic' else None)
    if scope is None:raise ValueError('Verified data or explicitly invented fixture required')
    if data_module.read_json(store.root/'owner.json')['uid']!=os.getuid():raise ValueError('Foreign run owner')
    if m['schema']!='b1.production.training.manifest.v1' or m['kind']!=scope or m['settings']!=plan['settings'] or m['seed']!=seed or m['data']!=data.binding or m['sources']!=plan['sources']:raise ValueError('Wrong training run identity')
    fd=os.open(store.root/'writer.lock',os.O_RDONLY|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_SH|fcntl.LOCK_NB);store.verify_sources();store.records=store.scan();p=store.load();_,selected=validate_payload(p,m,data,store.records)
        status='verified_partial'
        if (store.root/'completed.json').exists():
            if p['round']!=plan['settings']['rounds']:raise ValueError('Incomplete fit marked complete')
            receipt=data_module.read_json(store.root/'selected'/'receipt.json');completed=data_module.read_json(store.root/'completed.json')
            if io.binding(store.root/'selected'/'model.pt')!=receipt['model'] or io.binding(store.root/'selected'/'receipt.json')!=completed['selected_receipt'] or io.binding(store.root/'rounds'/f"{p['round']:04d}"/'receipt.json')!=completed['last_round_receipt']:raise ValueError('Selected/completion hashes differ')
            if completed['status']!='training_complete' or completed['round']!=p['round'] or completed['selection']!=receipt['selection'] or receipt['schema']!='b1.production.selected.v1' or receipt['manifest_sha256']!=digest(m):raise ValueError('Completion/selection schema or identity differs')
            if receipt['selection']!={'round':p['selected_round'],'joint_standardized_RMSE':p['selected_score']} or completed['manifest_sha256']!=digest(m):raise ValueError('Selected metadata differs')
            import torch
            with io.safe_open(store.root/'selected'/'model.pt') as f:export=torch.load(f,map_location='cpu',weights_only=True)
            if io.binding(store.root/'selected'/'model.pt')!=receipt['model']:raise ValueError('Selected model changed during read')
            roles=state_contract(export['state'],data.scaler.target_scaling)
            if export.get('schema')!='b1.production.selected.model.v1':raise ValueError('Unknown selected model schema')
            if export['manifest_sha256']!=digest(m) or RoundSnapshot(export['state'],roles,round_id=p['selected_round']).fingerprint!=selected.fingerprint:raise ValueError('Selected model differs from validation-selected state')
            status='verified_complete'
        data.assert_unchanged();store.verify_sources()
        return {'status':status,'completed_round':p['round'],'selected_round':p['selected_round'],'execution_scope':scope,'model_forward_or_scoring':False}
    finally:os.close(fd)


def verify_matrix_prefix(out):
    """Never recreate a lost earlier seed or continue past an incomplete seed."""
    allowed={'matrix.json','matrix.lock','matrix_completed.json',*(f'seed{s}' for s in FIXED['seeds'])}
    if any(p.name not in allowed for p in out.iterdir()):raise ValueError('Unexpected matrix artifacts')
    existing=[s for s in FIXED['seeds'] if (out/f'seed{s}').exists()]
    if existing!=FIXED['seeds'][:len(existing)]:raise ValueError('Missing earlier seed; never silently restart an existing matrix')
    for seed in existing[:-1]:
        if not (out/f'seed{seed}'/'completed.json').is_file():raise ValueError('Later seed exists before prior seed completed')
    if (out/'matrix_completed.json').exists() and (len(existing)!=3 or any(not (out/f'seed{s}'/'completed.json').is_file() for s in existing)):
        raise ValueError('Completed matrix has missing/incomplete seed artifacts')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--stage',choices=['preflight','fit','verify'],required=True)
    p.add_argument('--device');p.add_argument('--seed',type=int,choices=[42,43,44]);p.add_argument('--matrix',action='store_true')
    p.add_argument('--output-dir');p.add_argument('--review-receipt');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-round',type=int)
    args=p.parse_args(argv);plan=load_config(args.plan)
    if args.stage=='fit':
        if not args.device:raise ValueError('Explicit --device required')
        review=require_review(args.review_receipt,plan,args.plan,args.device)
    data=data_module.PreparedData(ROOT/plan['dataset'],ROOT/plan['scaler_lock']);data.names=ORIGINAL_NAMES
    data_module.install_access_guard(data,io.safe_path(args.output_dir) if args.output_dir else None)
    if args.stage=='preflight':
        print(json.dumps({'status':'data_preflight_passed','binding':data.binding,'model_initialized':False,'training_ready':False,'independent_review_required':True},sort_keys=True));return
    if not args.output_dir or (args.seed is None)==(not args.matrix):raise ValueError('Choose exactly one --seed or explicit --matrix plus --output-dir')
    out=io.safe_path(args.output_dir)
    allowed_root=ROOT/'output/experiments/b1_no_ff'
    if out==allowed_root or not out.is_relative_to(allowed_root) or out.is_relative_to(allowed_root/'preparation'):raise ValueError('New isolated B1 experiment output path required')
    if out.is_relative_to(data.path) or out.is_relative_to(data.lock_path.parent.parent):raise ValueError('Outputs must be separate from immutable inputs')
    if args.stage=='verify':
        if args.resume or args.stop_after_round is not None:raise ValueError('Verification cannot resume/fit')
        for seed in (FIXED['seeds'] if args.matrix else [args.seed]):print(json.dumps(verify_run(data,out/f'seed{seed}' if args.matrix else out,plan,seed)))
        return
    def terminate(signum,frame):raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,terminate)
    if args.matrix:
        import fcntl
        matrix={'schema':'b1.production.sequential.matrix.v1','plan':io.binding(args.plan),'device':args.device,'review':review,'seeds':FIXED['seeds'],'data':data.binding}
        if args.resume:
            if data_module.read_json(out/'matrix.json')!=matrix:raise ValueError('Matrix resume bindings changed')
            fd=os.open(out/'matrix.lock',os.O_RDWR|os.O_NOFOLLOW)
        else:
            if os.path.lexists(out):raise FileExistsError('Matrix root occupied; explicit verified resume required')
            if not out.parent.is_dir() or out.parent.stat().st_uid!=os.getuid():raise ValueError('Existing owned output parent required')
            out.mkdir(mode=0o700);fd=os.open(out/'matrix.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not args.resume:io.exclusive(out/'matrix.json',io.canonical(matrix));io.fsync_dir(out)
            verify_matrix_prefix(out)
            for seed in FIXED['seeds']:
                path=out/f'seed{seed}';exists=path.exists()
                if exists and not args.resume:raise FileExistsError('Unexpected occupied seed directory')
                result=run_seed(data,path,{'kind':'production','settings':FIXED},seed,args.device,resume=exists,stop_after_round=args.stop_after_round,sources=plan['sources'],review_binding=review)
                if result['status']!='training_complete':break
                verified=verify_run(data,path,plan,seed)
                if verified['status']!='verified_complete':raise ValueError('Seed failed verification; next seed blocked')
            else:
                completion={'status':'verified_complete','seeds':{str(seed):io.binding(out/f'seed{seed}'/'completed.json') for seed in FIXED['seeds']}}
                if (out/'matrix_completed.json').exists():
                    if data_module.read_json(out/'matrix_completed.json')!=completion:raise ValueError('Matrix completion changed')
                else:io.exclusive(out/'matrix_completed.json',io.canonical(completion));io.fsync_dir(out)
        finally:os.close(fd)
    else:
        print(json.dumps(run_seed(data,out,{'kind':'production','settings':FIXED},args.seed,args.device,resume=args.resume,stop_after_round=args.stop_after_round,sources=plan['sources'],review_binding=review)))

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as error:
        import sys
        print('B1 SAFETY STOP: '+type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(130 if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 1)
