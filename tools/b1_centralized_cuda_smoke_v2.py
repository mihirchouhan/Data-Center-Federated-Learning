"""Opt-in actual CUDA verification using invented arrays only; no data loader.

Fails with an exact operation/traceback. Evidence and synthetic checkpoints
are written exclusively under /tmp, never to production experiment paths.
"""
from pathlib import Path
import argparse,copy,hashlib,json,math,os,sys,tempfile,time,traceback
import numpy as np
import torch
from src import b1_centralized_sgd_runner_v2 as runner
from src import b1_centralized_sgd_runner_v1 as core
from src.b1_device_adapter_v1 import B1DeviceAdapter
from src.b1_target_scaling_v1 import TargetScaling
from src import b1_production_data_v1 as data_module
from src import b1_real_data_bridge_v1 as bridge,b1_bridge_scaling_v1 as scaling
from src.phase2_multioutput.contracts import identity_digest

def invented_client(node='1',count=4):
    ordinal=bridge.ORIGINAL_NAMES.index(node);rng=np.random.default_rng(11+ordinal)
    x=rng.normal(0,.1,(count,36,14)).astype('f4');x[:,:,13]=ordinal
    future=np.empty((count,6,2),'f8')
    future[:,:,0]=.4+np.arange(count)[:,None]*.0001;future[:,:,1]=23+np.arange(count)[:,None]*.01
    origins=np.datetime64('2022-01-10T06','ns')+np.arange(count)*np.timedelta64(1,'h')
    return data_module.ClientData(node,x,future,np.ones(future.shape,bool),origins)

class InventedData:
    scope='synthetic'
    def __init__(self):
        self.names=('1','2','979');self.clients={n:invented_client(n) for n in self.names}
        self.counts={n:len(c) for n,c in self.clients.items()}
        self.scaler=scaling.FrozenScaler((0.,)*14,(1.,)*14,(1.,)*13+(0.,),
            (400.,25.),(100.,5.),bridge.canonical({'fit_split':'train','algorithm':scaling.ALGORITHM,'mapping_digest':bridge.MAPPING_DIGEST}))
        self.val=[]
        for node,c in self.clients.items():
            origins=np.datetime64('2022-01-28T08','ns')+np.arange(len(c))*np.timedelta64(1,'h')
            self.val.append((c.x,bridge.target_validation.target_means(c.future,c.observed),np.array([node]*len(c),dtype='<U3'),origins))
        nodes=np.concatenate([b[2] for b in self.val]);origins=np.concatenate([b[3] for b in self.val]);y=np.concatenate([b[1] for b in self.val])
        self.binding={'synthetic':True,'results':{'val':{'windows':len(y),'identity_sha256':identity_digest(nodes,origins),'target_sha256':hashlib.sha256(y.astype('<f8').tobytes()).hexdigest()}}}
    def client(self,n):return self.clients[n]
    def validation_batches(self):yield from self.val
    def assert_unchanged(self):return True

def references(data,seed):
    model=B1DeviceAdapter(target_scaling=data.scaler.target_scaling,root_seed=seed,device='cpu')
    records=[{'state_fingerprint':model.snapshot(0).fingerprint,'selected_clients':[],'examples':0,'optimizer_steps':0,'local':[]}]
    for block in (1,2):
        names=list(core.b1.select_clients(data.names,seed,block,2));local=[]
        for n in names:
            client=data.client(n);order,shuffle=core.b1.shuffled_positions(seed,block,client.ordinal,len(client))
            local.append({'node':n,'ordinal':client.ordinal,'examples':len(client),'optimizer_steps':1,'batch_sizes':[len(client)],'shuffle_seed':shuffle,'row_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()})
        records.append({'selected_clients':names,'examples':sum(c['examples'] for c in local),'optimizer_steps':len(local),'local':local})
    return records

def probe():
    result={'scope':'invented_synthetic_tensors_only','device':'cuda:0','physical_GPU_UUID':core.GPU,
            'torch':str(torch.__version__),'torch_cuda_build':torch.version.cuda,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'real_training':False,'March_access':False,'approval_receipt_created':False,
            'completed_operations':[],'memory_available_bytes':None,'memory_total_bytes':None}
    stage='environment authorization/device validation';start=time.monotonic()
    try:
        if result['CUDA_VISIBLE_DEVICES']!=core.GPU or result['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':
            raise ValueError('Exact reviewed GPU1 and deterministic environment required')
        result['cuda_is_available']=torch.cuda.is_available();result['cuda_device_count']=torch.cuda.device_count()
        # Attempt actual initialization even when the availability probe is false.
        stage='torch.cuda.init()';torch.cuda.init();result['completed_operations'].append(stage)
        stage='torch.cuda.get_device_properties(cuda:0)';props=torch.cuda.get_device_properties(0)
        result.update(device_uuid=str(props.uuid),device_name=props.name)
        if str(props.uuid).removeprefix('GPU-')!=core.GPU.removeprefix('GPU-'):raise ValueError('Wrong physical GPU')
        stage='torch.cuda.mem_get_info(cuda:0)'
        free,total=torch.cuda.mem_get_info(0);result.update(memory_available_bytes=free,memory_total_bytes=total)
        if free<1024**3:raise RuntimeError('Synthetic CUDA smoke requires at least 1 GiB free device memory')
        stage='deterministic runtime and CUDA tensor allocation/computation'
        core.b1.runtime('cuda:0');torch.cuda.reset_peak_memory_stats(0)
        a=torch.arange(4,dtype=torch.float32,device='cuda:0');assert float((a*a).sum().cpu())==14.
        result['completed_operations'].append(stage)
        stage='B1 CUDA model initialization, forward, backward and local SGD'
        model=B1DeviceAdapter(target_scaling=TargetScaling((400.,25.),(100.,5.)),root_seed=42,device='cuda:0')
        before=model.snapshot(0).fingerprint
        session=core.central_session(model,invented_client(count=129),42,1)
        assert session['optimizer_steps']==2 and session['batch_sizes']==[128,1]
        assert math.isfinite(session['loss_standardized_MSE'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all().item() and torch.isfinite(p).all().item() for p in model.parameters())
        assert before!=model.snapshot(0).fingerprint
        result['session']=session;result['completed_operations'].append(stage)
        stage='synthetic CUDA atomic checkpoints, load and deterministic continuation'
        work=Path(tempfile.mkdtemp(prefix='b1-central-cuda-v2-',dir='/tmp'));result['synthetic_checkpoint_evidence']=str(work)
        data=InventedData();refs={s:references(data,s) for s in runner.SEEDS}
        settings=copy.deepcopy(core.SETTINGS);settings.update(clients=3,clients_per_round=2,rounds=2,blocks=2,validation_every=1)
        plan={'settings':settings,'sources':{},'reference':{'seeds':{str(s):{'synthetic':True} for s in runner.SEEDS}}}
        review={'synthetic_fixture':True};binding={'synthetic_fixture':True}
        clean=runner.execute_matrix(data,work/'clean',plan,refs,'cuda:0',review,binding,requested=(42,))
        partial=runner.execute_matrix(data,work/'recover',plan,refs,'cuda:0',review,binding,requested=(42,),stop_after_block=1)
        resumed=runner.execute_matrix(data,work/'recover',plan,refs,'cuda:0',review,binding,requested=(42,),resume=True)
        assert clean['completed_seeds']==resumed['completed_seeds']==[42]
        assert partial['completed_seeds']==[]
        fingerprints=[]
        for block in (0,1,2):
            left=data_module.read_json(work/f'clean/seed42/rounds/{block:04d}/receipt.json')['state_fingerprint']
            right=data_module.read_json(work/f'recover/seed42/rounds/{block:04d}/receipt.json')['state_fingerprint']
            assert left==right;fingerprints.append(left)
        torch.cuda.synchronize(0)
        result.update(continuation_fingerprints=fingerprints,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        result['completed_operations'].append(stage);result['status']='PASS'
    except BaseException as error:
        result.update(status='FAIL',failing_operation=stage,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    result['elapsed_seconds']=time.monotonic()-start
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--synthetic-cuda-authorized',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.synthetic_cuda_authorized:raise ValueError('Explicit synthetic CUDA scope required')
    dest=core.io.safe_path(a.output)
    if not dest.is_relative_to(Path('/tmp')) or not dest.parent.is_dir() or dest.parent.stat().st_uid!=os.getuid():raise ValueError('Owned /tmp evidence parent required')
    if os.path.lexists(dest):raise FileExistsError('Evidence destination occupied')
    result=probe();core.io.exclusive(dest,(json.dumps(result,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
