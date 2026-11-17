"""Isolated B1 train/validation preparation. No fitting/models/test capability.

Default CLI describes a plan. Real execution needs a separate, exact-scope
authorization record and fresh operator-confirmed quota evidence. Not approval.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,multiprocessing,os,resource,signal,time,traceback,zipfile,math,platform,uuid
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from src import b1_preparation_io_v1 as io
from src import b1_raw_window_producer_v1 as producer
from src import b1_real_data_bridge_v1 as bridge
from src.phase2_multioutput.contracts import identity_digest

SCHEMA='b1.preparation.execution.v1'
RECORD=np.dtype([('time','<i8'),('value','<f8')])
REFERENCE={
 'train':{'windows':280499,'identity_sha256':'835d24e0887550f0e28eb891d11ba55038e2d7ac3c708628265a6784155074d7','target_sha256':'ef4794e158d3aa9fb08bd49da283aefd02638380fb2902b63b3f84c918d8791a'},
 'val':{'windows':82993,'identity_sha256':'9493effdda844c100d462ac8f9f8eb5e231902e3d383c374264af4a3abac0f1c','target_sha256':'ff3652c9a06b065095a105f76dcbcd1745dc0d970a7e775169bb18f1acf54cf5'}}
FIELDS=('x','observed','future','target_observed','nodes','origins','history_times','target_times','history_nodes','target_nodes','source_earliest','source_latest','target_source_earliest','target_source_latest','history_coverage','target_coverage')

def implementation_bindings():
    root=Path(producer.__file__).parent
    names=('b1_raw_window_producer_v1.py','b1_real_data_bridge_v1.py','b1_target_validation_v1.py','b1_target_scaling_v1.py','multinode.py','preprocess.py','history_data_v4.py','phase1_3/schema.py','phase1_3/config.py','phase2_multioutput/contracts.py')
    paths=[root/n for n in names]+[Path(__file__),Path(io.__file__)]
    return {p.name:io.binding(p) for p in paths}

def contract(plan):return io.sha(io.canonical(plan))
def environment():return {'python':platform.python_version(),'numpy':np.__version__,'pyarrow':pa.__version__}

def validate_plan(plan,source_root,output,authorization=None,*,_inspection=False):
    # Copy caller containers so mutation cannot change the executing contract.
    plan=json.loads(io.canonical(plan));root=io.safe_path(source_root);output=io.safe_path(output)
    if plan.get('source_root')!=str(root):raise ValueError('Exact source root must be bound to the plan')
    if plan.get('output_parent')!=str(output.parent) or output.is_relative_to(root):raise ValueError('Bound isolated output parent required')
    if plan.get('schema')!=SCHEMA or plan.get('protocol_status')!='UNAPPROVED' or plan.get('historical_preflight')!='FAILED':raise ValueError('Explicit B1 unapproved/failed-provenance contract required')
    if plan.get('mapping')!=list(bridge.ORIGINAL_NAMES) or plan.get('features')!=list(bridge.FEATURES) or plan.get('units')!=list(bridge.UNITS) or plan.get('splits')!={k:list(v) for k,v in bridge.SPLITS.items()}:raise ValueError('B1 mapping/features/units/split changes forbidden')
    if plan.get('implementation')!=implementation_bindings():raise ValueError('Execution source changed')
    if plan.get('environment')!=environment():raise ValueError('Preparation software environment changed')
    if plan.get('window_contract')!={'history':36,'horizon':6,'grid_seconds':600,'hold_seconds':300,'min_coverage':0.5,'weather_max_age_seconds':1800,'input_fill':False,'target_fill':False,'target_units':['W','degC']}:raise ValueError('B1 preprocessing contract changed')
    limits=io.Limits(**plan['limits']);clients=plan['clients']
    if not clients or clients!=sorted(set(clients),key=lambda n:bridge.ORIGINAL_NAMES.index(n)):raise ValueError('Explicit canonical original client IDs required')
    if set(plan['expected'])!={'train','val'}:raise ValueError('Train/validation references only')
    for split in ('train','val'):
        e=plan['expected'][split]
        if set(e['per_node'])!=set(clients) or any(type(n) is not int or n<=0 or n>limits.max_client_windows for n in e['per_node'].values()) or sum(e['per_node'].values())!=e['windows']:raise ValueError('Exact complete per-client counts/capacity required')
        for name in ('identity_sha256','target_sha256'):
            h=e[name]
            if not isinstance(h,str) or len(h)!=64 or any(v not in '0123456789abcdef' for v in h):raise ValueError('Exact identity and target fingerprints required')
    # Entire source declaration validated without any raw open.
    entries=[producer.SourceEntry(**e) for e in plan['sources']];pairs=set()
    for e in entries:
        parts=Path(e.path).parts
        if len(parts)!=4 or parts[0] not in ('year_month=22-01','year_month=22-02') or e.metric not in producer.METRICS:raise PermissionError('Only explicit January/February sources')
        plugin='weather_pub' if e.metric in ('temp','humidity') else ('ipmi_pub' if e.metric in ('ambient','total_power') else 'ganglia_pub')
        if parts[1:3]!=(f'plugin={plugin}',f'metric={e.metric}') or parts[-1]!='a_0.parquet' or type(e.bytes) is not int or e.bytes<=0 or len(e.sha256)!=64:raise ValueError('Invalid source declaration')
        io.safe_path(root/e.path);pairs.add((parts[0],e.metric))
    if len(entries)!=20 or len(pairs)!=20:raise ValueError('Exactly20 unique month/metric sources required')
    if any('march' in p.lower() for p in root.parts):raise PermissionError('March source roots forbidden')
    scope=plan.get('scope')
    if scope=='synthetic':
        if not root.is_relative_to('/tmp') or not output.is_relative_to('/tmp'):raise PermissionError('Synthetic scope restricted to /tmp')
    elif scope=='production':
        if clients!=list(bridge.ORIGINAL_NAMES) or any({k:v for k,v in plan['expected'][s].items() if k!='per_node'}!=REFERENCE[s] for s in REFERENCE):raise ValueError('Production cannot change saved B1 counts/identity/targets')
        expected={'operation':'B1_train_val_preparation_only','contract_sha256':contract(plan),'output':str(output),'uid':os.getuid()}
        if not _inspection and (not isinstance(authorization,dict) or any(authorization.get(k)!=v for k,v in expected.items()) or authorization.get('explicit_operator_authorization') is not True or not 0<authorization.get('expires_at_unix',0)-time.time()<=86400):raise PermissionError('Separate explicit exact-scope preparation authorization missing/stale')
    else:raise ValueError('Unknown execution scope')
    if not _inspection and os.path.lexists(output):raise FileExistsError('Output attempt occupied; no in-place resume')
    return plan,root,output,limits

def _spool_name(split,metric,node):return f'spool/{split}_{metric}_{node or "weather"}.bin'

def _read_sources(verified,clients,attempt,limits):
    wanted={n:i for i,n in enumerate(clients)};counts={};rows_by_spool={}
    for entry in verified.entries:
        attempt.budget.check();month=Path(entry.path).parts[0][11:]
        lower=np.datetime64('2022-'+month[3:]+'-01','ns').view('i8');upper=np.datetime64('2022-02-01' if month=='22-01' else '2022-03-01','ns').view('i8')
        with producer._open_entry(verified.root,entry) as handle:
            before=os.fstat(handle.fileno());file=pq.ParquetFile(handle);schema=file.schema_arrow
            kind=schema.field('timestamp').type;value_kind=schema.field('value').type;local=entry.metric not in ('temp','humidity')
            if not pa.types.is_timestamp(kind) or kind.tz!='UTC' or not (pa.types.is_integer(value_kind) or pa.types.is_floating(value_kind)):raise ValueError('UTC/real numeric raw schema required')
            if local and not pa.types.is_string(schema.field('node').type):raise ValueError('Literal string node IDs required')
            if file.metadata.num_rows>limits.max_file_rows:raise ValueError('Raw file scan cap exceeded')
            # Never request value columns until every timestamp in this file is Jan/Feb-compatible.
            for batch in file.iter_batches(batch_size=limits.arrow_rows,columns=['timestamp'],use_threads=False):
                attempt.budget.check();arr=pc.filter(batch['timestamp'],pc.is_valid(batch['timestamp']))
                ticks=pc.cast(pc.cast(arr,pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy()
                if ((ticks<lower)|(ticks>=upper)).any():raise PermissionError('Misfiled source timestamp; no value column opened')
            quality={'scanned_rows':0,'null_timestamp_rows':0,'unmapped_or_unrequested_rows':0,'retained_train_rows':0,'retained_val_rows':0,'out_of_B1_rows':0}
            cols=['timestamp','node','value'] if local else ['timestamp','value']
            for batch in file.iter_batches(batch_size=limits.arrow_rows,columns=cols,use_threads=False):
                attempt.budget.check();quality['scanned_rows']+=len(batch)
                valid=pc.is_valid(batch['timestamp']);quality['null_timestamp_rows']+=len(batch)-int(pc.sum(pc.cast(valid,pa.int64())).as_py() or 0)
                batch=pc.filter(batch,valid)
                t=pc.cast(pc.cast(batch['timestamp'],pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy()
                if local:
                    d=pc.dictionary_encode(batch['node']);codes=np.array([wanted.get(n,-1) for n in d.dictionary.to_pylist()]+[-1],dtype='i4')[pc.fill_null(d.indices,-1).to_numpy()]
                else:codes=np.zeros(len(batch),dtype='i4')
                quality['unmapped_or_unrequested_rows']+=int((codes<0).sum())
                masks={s:(t>=producer._split(s)[0])&(t<producer._split(s)[1]) for s in ('train','val')}
                quality['out_of_B1_rows']+=int((~(masks['train']|masks['val'])).sum())
                for split,mask in masks.items():
                    rows=np.flatnonzero(mask&(codes>=0))
                    if not len(rows):continue
                    # Arrow filters values only for retained B1 nodes/times; original row order is preserved.
                    b=pc.take(batch,pa.array(rows));values=pc.cast(b['value'],pa.float64()).to_numpy(zero_copy_only=False)
                    if local:
                        order=np.argsort(codes[rows],kind='stable');group_codes=codes[rows][order];cuts=np.r_[0,np.flatnonzero(np.diff(group_codes))+1,len(order)]
                        groups=[(clients[int(group_codes[a])],order[a:z]) for a,z in zip(cuts[:-1],cuts[1:])]
                    else:groups=[(None,np.arange(len(rows)))]
                    for node,indices in groups:
                        name=_spool_name(split,entry.metric,node);rows_by_spool[name]=rows_by_spool.get(name,0)+len(indices)
                        if rows_by_spool[name]>limits.max_raw_rows:raise ValueError('Per-node raw-row cap exceeded; never truncate')
                        raw=np.empty(len(indices),dtype=RECORD);raw['time']=t[rows[indices]];raw['value']=values[indices]
                        attempt.append(name,raw.tobytes());quality['retained_'+split+'_rows']+=len(indices)
            after=os.fstat(handle.fileno())
            if any(getattr(before,k)!=getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_mode')) or producer._hash_file(handle)!=entry.sha256:raise ValueError('Source changed during preparation')
            if quality['scanned_rows']!=file.metadata.num_rows:raise ValueError('Raw scan incomplete')
            counts[entry.path]=quality;attempt.event('source_scanned',{'source':entry.path,'quality':quality})
    attempt.seal_spools();return counts

def _sorted_stream(attempt,split,metric,node,limits):
    path=attempt.root/_spool_name(split,metric,node)
    if not path.exists():return producer.RawSeries(np.empty(0,dtype='datetime64[ns]'),np.empty(0),limits=producer.Limits(max_raw_rows=limits.max_raw_rows))
    with io.safe_open(path) as f:
        size=os.fstat(f.fileno()).st_size
        if size%RECORD.itemsize or size//RECORD.itemsize>limits.max_raw_rows:raise ValueError('Spool size/count invalid')
        rows=np.frombuffer(f.read(),dtype=RECORD)
    order=np.argsort(rows['time'],kind='stable')
    # Explicit per-node time sorting; equal times retain original canonical-file/row order.
    return producer.RawSeries(rows['time'][order].astype('datetime64[ns]'),rows['value'][order],limits=producer.Limits(max_raw_rows=limits.max_raw_rows))

def _update_identity(h,nodes,origins):
    for node,origin in zip(nodes.tolist(),origins.view('i8').tolist()):
        b=json.dumps([node,origin],ensure_ascii=True,separators=(',',':')).encode();h.update(len(b).to_bytes(8,'big'));h.update(b)

def _write_chunks(attempt,produced,split,chunk_size,chunks,hashers):
    w=produced.windows;payload=produced._payload
    if tuple(payload)!=FIELDS:raise ValueError('Historical producer payload interface changed')
    _update_identity(hashers[0],w.nodes,w.origins);hashers[1].update(w.physical_targets.astype('<f8').tobytes())
    for first in range(0,len(w),chunk_size):
        stop=min(first+chunk_size,len(w));data={k:v[first:stop] for k,v in payload.items()};data['physical_targets']=w.physical_targets[first:stop]
        name=f'{split}_{bridge.ORIGINAL_NAMES.index(str(w.nodes[0])):04d}_{first:06d}.npz';path=attempt.dataset/name
        reserve=sum(v.nbytes for v in data.values())+65536;attempt.budget.charge(reserve)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
        details=io.binding(path)
        if details['bytes']>reserve:raise ValueError('Chunk exceeded reserved byte budget')
        chunks.append({'path':name,'split':split,'node':str(w.nodes[0]),'windows':stop-first,**details,
          'arrays':{k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in data.items()}})

def _prepare(plan,source_root,output,authorization,quota,*,_session=None):
    plan,source_root,output,limits=validate_plan(plan,source_root,output,authorization)
    budget=io.Budget(output,limits,quota);attempt=io.Attempt(output,contract(plan),budget,_session)
    try:
        attempt.write('contract.json',io.canonical(plan))
        attempt.write('execution_envelope.json',io.canonical({'scope':plan['scope'],'authorization':authorization,'quota_evidence':quota}))
        chunks=[];results={};diagnostics={}
        entries=[producer.SourceEntry(**x) for x in plan['sources']]
        verified=producer.verify_source_plan(source_root,entries)
        raw_quality=_read_sources(verified,plan['clients'],attempt,limits)
        plimits=producer.Limits(max_raw_rows=limits.max_raw_rows,max_scan_rows=limits.max_file_rows,arrow_batch_rows=limits.arrow_rows,max_windows=limits.max_client_windows,max_population_bytes=limits.max_client_windows*56000)
        evidence=producer._evidence('synthetic' if plan['scope']=='synthetic' else 'observed_prepared',verified.fingerprint)
        for split in ('train','val'):
            hashes=[hashlib.sha256(),hashlib.sha256()];count=0;pernode={};diagnostics[split]={}
            weather={m:_sorted_stream(attempt,split,m,None,limits) for m in ('temp','humidity')}
            for node in plan['clients']:
                budget.check();local={m:_sorted_stream(attempt,split,m,node,limits) for m in producer.LOCAL_METRICS}
                produced=producer._produce({node:local},weather,split,plimits,evidence)
                if len(produced.windows)!=plan['expected'][split]['per_node'][node]:raise ValueError('Exact per-node reference count differs: '+split+'/'+node)
                pernode[node]=len(produced.windows);count+=len(produced.windows);diagnostics[split][node]=produced.diagnostics
                days=len(np.unique(produced.windows.origins.astype('datetime64[D]')))
                diagnostics[split][node]['positive_UTC_dates']=days
                if split=='train' and plan['scope']=='production' and (len(produced.windows)<24 or days<7):raise ValueError('Training-only client eligibility failed; never filter or renumber')
                _write_chunks(attempt,produced,split,limits.chunk_windows,chunks,hashes);attempt.event('client_prepared',{'split':split,'node':node,'windows':len(produced.windows)})
                del produced,local
            results[split]={'windows':count,'identity_sha256':hashes[0].hexdigest(),'target_sha256':hashes[1].hexdigest(),'per_node':pernode}
            if results[split]!=plan['expected'][split]:raise ValueError('Exact split count/identity/target reference differs; no publication')
        # Recheck all executing dependency bytes before final validation/publication.
        if implementation_bindings()!=plan['implementation']:raise ValueError('Execution source changed during preparation')
        manifest={'schema':'b1.prepared.train_val.v1','status':'complete','contract_sha256':contract(plan),'contract':plan,'results':results,'source_evidence':evidence,'chunks':chunks,
            'raw_quality':raw_quality,'diagnostics':diagnostics,'protocol_status':'UNAPPROVED','historical_preflight':'FAILED','scalers_fitted':False,'training_authorized':False}
        attempt.write('dataset.partial/manifest.json',io.canonical(manifest))
        verify_dataset(attempt.dataset,plan,allow_staging=True)
        resources={'wall_seconds':time.monotonic()-budget.start,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'charged_bytes':budget.used,'limits':asdict(limits),'quota_evidence':quota,
          'worker_hard_limits':{'address_space':list(resource.getrlimit(resource.RLIMIT_AS)),'cpu_seconds':list(resource.getrlimit(resource.RLIMIT_CPU))}}
        attempt.write('resources.json',io.canonical(resources))
        budget.check();io.fsync_dir(attempt.dataset);io.rename_noreplace(attempt.dataset,attempt.root/'dataset')
        attempt.write('completed.json',io.canonical({'dataset_manifest':io.binding(attempt.root/'dataset/manifest.json'),'contract_sha256':contract(plan),'execution_envelope':io.binding(attempt.root/'execution_envelope.json')}));io.fsync_dir(attempt.root)
        return {'status':'verified_complete','results':results,'resources':resources}
    except BaseException as error:
        try:attempt.write('failure.json',io.canonical({'type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'status':'incomplete_preserved'}))
        except BaseException:pass
        raise
    finally:attempt.close()

def _check_archive(path,windows,limits):
    if type(windows) is not int or not 0<windows<=limits.chunk_windows:raise ValueError('Chunk window capacity invalid')
    n=windows
    specs={'x':('float64',(n,36,14)),'observed':('bool',(n,36,13)),'future':('float64',(n,6,2)),'target_observed':('bool',(n,6,2)),
      'nodes':('<U3',(n,)),'origins':('datetime64[ns]',(n,)),'history_times':('datetime64[ns]',(n,36)),'target_times':('datetime64[ns]',(n,6)),
      'history_nodes':('int64',(n,36)),'target_nodes':('int64',(n,6)),'source_earliest':('datetime64[ns]',(n,36,9)),'source_latest':('datetime64[ns]',(n,36,9)),
      'target_source_earliest':('datetime64[ns]',(n,6,2)),'target_source_latest':('datetime64[ns]',(n,6,2)),'history_coverage':('float64',(n,36,7)),
      'target_coverage':('float64',(n,6,2)),'physical_targets':('float64',(n,2))}
    with io.safe_open(path) as raw,zipfile.ZipFile(raw) as archive:
        entries=archive.infolist()
        if len(entries)!=len(specs) or {e.filename for e in entries}!={n+'.npy' for n in specs}:raise ValueError('Archive members differ')
        for entry in entries:
            if entry.compress_type!=zipfile.ZIP_STORED:raise ValueError('Only uncompressed bounded chunks permitted')
            with archive.open(entry) as f:
                version=np.lib.format.read_magic(f)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f,max_header_size=1024)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=1024)
                else:raise ValueError('Unsupported array header version')
                expected_dtype,expected_shape=specs[entry.filename[:-4]]
                if dtype!=np.dtype(expected_dtype) or shape!=expected_shape or fortran or dtype.hasobject or entry.file_size!=f.tell()+math.prod(shape)*dtype.itemsize:raise ValueError('Array header/size contract differs before allocation')

def verify_dataset(path,plan,*,allow_staging=False):
    path=io.safe_path(path)
    if path.name!='dataset' and not (allow_staging and path.name=='dataset.partial'):raise ValueError('Not a published dataset')
    limits=io.Limits(**plan['limits'])
    if (path/'manifest.json').stat().st_size>64*1024**2:raise ValueError('Manifest too large')
    with io.safe_open(path/'manifest.json') as f:m=json.load(f)
    if m.get('schema')!='b1.prepared.train_val.v1' or m.get('status')!='complete' or m.get('contract')!=plan or m.get('contract_sha256')!=contract(plan):raise ValueError('Dataset/contract mismatch')
    names=[x['path'] for x in m['chunks']]
    if len(set(names))!=len(names) or {x.name for x in path.iterdir()}!=set(names)|{'manifest.json'}:raise ValueError('Missing/duplicate/unexpected output artifacts')
    results={};evidence=m['source_evidence'];last={s:None for s in ('train','val')};h={s:[hashlib.sha256(),hashlib.sha256()] for s in last};counts={s:{} for s in last}
    for chunk in m['chunks']:
        name=chunk['path'];split=chunk['split']
        if Path(name).name!=name or split not in last or io.binding(path/name)!={k:chunk[k] for k in ('bytes','sha256')}:raise ValueError('Output chunk binding failure')
        _check_archive(path/name,chunk['windows'],limits)
        with io.safe_open(path/name) as f,np.load(f,allow_pickle=False) as archive:
            if set(archive.files)!=set(FIELDS)|{'physical_targets'}:raise ValueError('Changed array fields')
            data={n:archive[n] for n in archive.files}
        y=data.pop('physical_targets');w=bridge.accept_windows(split=split,mapping=bridge.ORIGINAL_NAMES,source_evidence=evidence,**data)
        if not np.array_equal(y,w.physical_targets) or len(w)!=chunk['windows'] or any(str(n)!=chunk['node'] for n in w.nodes):raise ValueError('Chunk target/count/client mismatch')
        if chunk['arrays']!={k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in {**data,'physical_targets':y}.items()}:raise ValueError('Array schema differs')
        key=(bridge.ORIGINAL_NAMES.index(str(w.nodes[0])),int(w.origins[0].view('i8')))
        if last[split] is not None and key<=last[split]:raise ValueError('Duplicate/unordered cross-chunk identities')
        last[split]=(bridge.ORIGINAL_NAMES.index(str(w.nodes[-1])),int(w.origins[-1].view('i8')))
        _update_identity(h[split][0],w.nodes,w.origins);h[split][1].update(y.astype('<f8').tobytes());counts[split][chunk['node']]=counts[split].get(chunk['node'],0)+len(w)
    for split in last:
        results[split]={'windows':sum(counts[split].values()),'identity_sha256':h[split][0].hexdigest(),'target_sha256':h[split][1].hexdigest(),'per_node':counts[split]}
    if results!=plan['expected'] or m['results']!=results:raise ValueError('Incomplete/changed full reference population')
    return {'status':'verified_complete','results':results,'model_or_scaler_executed':False}

def _worker(plan,source_root,output,authorization,quota,pipe,session):
    try:
        limits=io.Limits(**plan['limits']);resource.setrlimit(resource.RLIMIT_AS,(limits.max_address_space_bytes,limits.max_address_space_bytes))
        resource.setrlimit(resource.RLIMIT_CPU,(limits.max_wall_seconds+1,limits.max_wall_seconds+2))
        result=_prepare(plan,source_root,output,authorization,quota,_session=session);pipe.send({'ok':True,'result':result})
    except BaseException as error:
        try:pipe.send({'ok':False,'type':type(error).__name__,'error':str(error)})
        except BaseException:pass
    finally:pipe.close()

def verify_run(output,plan):
    """Read-only verification. Does not call preparation or open source telemetry."""
    output=io.safe_path(output);validate_plan(plan,plan['source_root'],output,_inspection=True)
    with io.safe_open(output/'owner.json') as f:owner=json.load(f)
    with io.safe_open(output/'contract.json') as f:saved=json.load(f)
    if owner.get('schema')!='b1.preparation.attempt.v1' or owner.get('uid')!=os.getuid() or owner.get('contract_sha256')!=contract(plan) or saved!=plan:raise ValueError('Attempt owner/contract differs')
    result=verify_dataset(output/'dataset',plan)
    with io.safe_open(output/'completed.json') as f:completion=json.load(f)
    if completion!={'dataset_manifest':io.binding(output/'dataset/manifest.json'),'contract_sha256':contract(plan),'execution_envelope':io.binding(output/'execution_envelope.json')}:raise ValueError('Completion receipt mismatch')
    with io.safe_open(output/'execution_envelope.json') as f:envelope=json.load(f)
    if envelope.get('scope')!=plan['scope']:raise ValueError('Execution scope evidence differs')
    if plan['scope']=='production':
        auth=envelope['authorization'];expected={'operation':'B1_train_val_preparation_only','contract_sha256':contract(plan),'output':str(output),'uid':os.getuid(),'explicit_operator_authorization':True}
        if not isinstance(auth,dict) or any(auth.get(k)!=v for k,v in expected.items()) or auth.get('expires_at_unix',0)<=owner['created_at_unix']:raise ValueError('Recorded preparation authorization invalid at attempt start')
    return result

def load_client_windows(output,plan,split,node):
    """Read-only bounded, unscaled bridge capability for exactly one client/split.

    Performs full dataset verification first. No scaler/training API is invoked.
    A future runner should separately review its verification/caching strategy.
    """
    if split not in ('train','val') or node not in plan['clients']:raise PermissionError('Declared train/val client only')
    verify_run(output,plan);root=Path(output)/'dataset'
    with io.safe_open(root/'manifest.json') as f:m=json.load(f)
    pieces=[]
    for entry in m['chunks']:
        if entry['split']==split and entry['node']==node:
            if io.binding(root/entry['path'])!={k:entry[k] for k in ('bytes','sha256')}:raise ValueError('Client chunk changed after dataset verification')
            with io.safe_open(root/entry['path']) as f,np.load(f,allow_pickle=False) as archive:pieces.append({k:archive[k] for k in FIELDS})
    if not pieces:raise ValueError('No client data; no fallback')
    payload={k:np.concatenate([x[k] for x in pieces]) for k in FIELDS}
    return bridge.accept_windows(split=split,mapping=bridge.ORIGINAL_NAMES,source_evidence=m['source_evidence'],**payload)

def run(plan,source_root,output,*,authorization=None,quota=None):
    """Supervised fresh attempt only. Never retries/resumes a failed directory."""
    plan,root,output,limits=validate_plan(plan,source_root,output,authorization);io.Budget(output,limits,quota)
    if not output.parent.is_dir() or output.parent.stat().st_uid!=os.getuid():raise ValueError('Existing owned output parent required')
    context=multiprocessing.get_context('spawn');parent,child=context.Pipe(duplex=False)
    session=uuid.uuid4().hex;process=context.Process(target=_worker,args=(plan,str(root),str(output),authorization,quota,child,session));started=time.monotonic();message=None;stop=None
    saved={n:os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}
    try:
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';process.start()
    finally:
        for n,v in saved.items():
            if v is None:os.environ.pop(n,None)
            else:os.environ[n]=v
    child.close()
    try:
        while process.is_alive():
            if parent.poll(.1):
                try:message=parent.recv()
                except EOFError:pass
            if time.monotonic()-started>limits.max_wall_seconds:stop='wall budget exceeded';break
            status=Path(f'/proc/{process.pid}/status')
            try:
                metrics={line.split(':',1)[0]:line.split(':',1)[1] for line in status.read_text().splitlines() if ':' in line}
                if int(metrics.get('VmHWM','0 kB').split()[0])*1024>limits.max_RSS_bytes:stop='worker peak RSS exceeded';break
            except FileNotFoundError:pass
            if output.exists():
                allocated=0
                for base,dirs,files in os.walk(output,followlinks=False):
                    for name in dirs:io.safe_path(Path(base)/name)
                    for name in files:
                        path=io.safe_path(Path(base)/name)
                        try:allocated+=path.stat().st_blocks*512
                        except FileNotFoundError:pass
                if allocated>limits.max_disk_bytes:stop='allocated attempt disk budget exceeded';break
        if stop:
            process.terminate();process.join(2)
            if process.is_alive():process.kill()
        process.join()
        if message is None and parent.poll():
            try:message=parent.recv()
            except EOFError:pass
        if stop or process.exitcode!=0 or not message or not message['ok']:
            owned=False
            try:
                with io.safe_open(output/'owner.json') as f:owner=json.load(f)
                owned=owner.get('attempt')==session and owner.get('uid')==os.getuid() and owner.get('contract_sha256')==contract(plan)
            except (OSError,ValueError):pass
            if owned and not (output/'supervisor_failure.json').exists():io.exclusive(output/'supervisor_failure.json',io.canonical({'error':stop or message or 'worker exited without completion','exitcode':process.exitcode,'status':'requires_read_only_review'}))
            raise RuntimeError('Preparation incomplete; artifacts preserved: '+str(stop or message))
        verified=verify_run(output,plan)
        return {**message['result'],'verification':verified}
    except BaseException:
        if process.is_alive():
            process.terminate();process.join(2)
            if process.is_alive():process.kill();process.join()
        raise
    finally:parent.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--stage',choices=('describe','prepare','verify','status'),default='describe');parser.add_argument('--source-root');parser.add_argument('--output-dir');parser.add_argument('--authorization');parser.add_argument('--quota-evidence')
    args=parser.parse_args()
    with io.safe_open(args.plan) as f:plan=json.load(f)
    if args.stage=='describe':result={'status':'description_only','contract_sha256':contract(plan),'scope':plan['scope'],'preparation_authorized_by_this_command':False,'sources_opened':False}
    elif args.stage=='status':result=io.status(args.output_dir)
    elif args.stage=='verify':result=verify_run(args.output_dir,plan)
    else:
        authorization=None;quota=None
        if args.authorization:
            with io.safe_open(args.authorization) as f:authorization=json.load(f)
        if args.quota_evidence:
            with io.safe_open(args.quota_evidence) as f:quota=json.load(f)
        result=run(plan,args.source_root,args.output_dir,authorization=authorization,quota=quota)
    print(json.dumps(result,sort_keys=True,indent=2))
if __name__=='__main__':main()
