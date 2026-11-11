"""Independent, no-fill coverage audit; no models, scalers or historical replay.

Raw Arrow batches are spooled to bounded node shards. Only a single node's raw
metric rows are sorted at once. Historical weighted bins and feature definitions
are reused, with the frozen Phase 2 raw-contributor split reset applied before
binning. Windows are validity-mask calculations, not dense training datasets.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from src.history_data_v4 import LOCAL_METRICS, RAW_DTYPE, node_features
from src.multinode import weighted_bins
from src.preprocess import _range_mask, _weather_asof, NS_SECOND
from src.phase1_3.config import utc_ns
from src.phase2_multioutput.contracts import identity_digest, split_for_origin
from src.phase2_multioutput.protocol import verify_design, validate_protocol

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 600 * NS_SECOND
HISTORY, HORIZON = 36, 6
SCENARIOS = ('january_february', 'march_only', 'january_march')
SENSITIVITY = (('windows_ge4',4,0), ('windows_ge24',24,0), ('days_ge7',1,7),
               ('windows_ge4_days_ge7',4,7), ('windows_ge24_days_ge7',24,7))
WINDOW_DTYPE = np.dtype([('node_index','<i4'),('origin_ns','<i8'),('power_w','<f8'),('temperature_c','<f8')])


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, relative):
    root=Path(root).absolute(); relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts: raise ValueError('Unsafe relative path')
    p=root/relative
    if any(x.is_symlink() for x in (p,*p.parents)): raise ValueError('Symlink path prohibited')
    return p


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def write_csv(path, rows, columns=None):
    rows=list(rows)
    if columns is None:
        if not rows: raise ValueError('Empty table needs explicit columns')
        columns=list(rows[0])
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def canonical_rows(times,values,metric):
    times=np.asarray(times,dtype=np.int64); values=np.asarray(values,dtype=np.float64)
    if times.shape!=values.shape or times.ndim!=1: raise ValueError('Row alignment')
    order=np.argsort(times,kind='stable');times=times[order];values=values[order].copy()
    keep=np.r_[times[1:]!=times[:-1],True] if len(times) else np.array([],dtype=bool)
    duplicates=int(len(times)-keep.sum());times,values=times[keep],values[keep]
    valid,_=_range_mask(metric,values)
    quality={'rows_after_dedup':len(times),'duplicates_keep_last':duplicates,
        'nonfinite_after_dedup':int((~np.isfinite(values)).sum()),
        'out_of_range_after_dedup':int((np.isfinite(values)&~valid).sum())}
    values[~valid]=np.nan
    return times,values,quality


def bin_split(times,values,metric,edges):
    """No hold/weather observation may originate before this split's lower edge."""
    edges=np.asarray(edges,dtype=np.int64)
    if edges.ndim!=1 or len(edges)<2 or not np.all(np.diff(edges)==WIDTH):raise ValueError('Ten-minute grid required')
    inside=(times>=edges[0])&(times<edges[-1]);t,v=times[inside],values[inside]
    if not len(t):return np.full(len(edges)-1,np.nan),np.zeros(len(edges)-1)
    if metric in ('temp','humidity'):
        result=_weather_asof(t,v,edges,1800*NS_SECOND)
        return result,np.isfinite(result).astype(np.float64)
    return weighted_bins(t,v,edges,300*NS_SECOND,0.5)


def aligned_memory(free,total):
    """Protocol ratio of separate time-weighted means on the SAME UTC bins.

    This does not replace it with raw timestamp joins or mean-of-ratios. Each
    source separately meets the bin-coverage requirement; simultaneous support
    within a bin is not guaranteed by this historical feature definition.
    """
    free,total=np.asarray(free),np.asarray(total)
    if free.shape!=total.shape:raise ValueError('Memory bin axes differ')
    good=np.isfinite(free)&np.isfinite(total)&(free>=0)&(total>0)&(free<=total)
    memory=np.full(free.shape,np.nan,dtype=np.float64)
    memory[good]=1-free[good]/total[good]
    return memory,{'bins':free.size,'both_finite':int((np.isfinite(free)&np.isfinite(total)).sum()),
        'invalid_denominator':int((np.isfinite(total)&(total<=0)).sum()),
        'free_exceeds_total':int((np.isfinite(free)&np.isfinite(total)&(free>total)).sum()),
        'valid_fraction_bins':int(good.sum())}


def window_masks(features,edges):
    """Return disjoint reasons plus diagnostics; never materialize H-by-F windows."""
    x=np.asarray(features,dtype=np.float64); edges=np.asarray(edges,dtype=np.int64)
    if x.shape!=(len(edges)-1,13) or not np.all(np.diff(edges)==WIDTH):raise ValueError('Features/time axes differ')
    if edges[0]%WIDTH:raise ValueError('Off-grid UTC split')
    origins=np.arange(len(x),dtype=np.int64);origins=origins[edges[:-1]%(6*WIDTH)==0]
    insufficient=origins<HISTORY
    crossing=(~insufficient)&(origins+HORIZON>len(x))
    contained=(origins>=HISTORY)&(origins+HORIZON<=len(x))
    target_contained=origins+HORIZON<=len(x)
    target_good=np.isfinite(x[:,[0,6]])&(x[:,[0,6]]>=np.array([0.,-273.15]))
    prefix=np.vstack((np.zeros((1,2),dtype=np.int64),np.cumsum(~target_good,axis=0)))
    complete=np.zeros((len(origins),2),dtype=bool)
    z=origins[target_contained]
    complete[target_contained]=prefix[z+HORIZON]==prefix[z]
    valid_input=np.isfinite(x)
    valid_input[:,0]&=x[:,0]>=0
    valid_input[:,[1,2,3,4,8]]&=(x[:,[1,2,3,4,8]]>=0)&(x[:,[1,2,3,4,8]]<=100)
    valid_input[:,5]&=(x[:,5]>=0)&(x[:,5]<=1)
    valid_input[:,[6,7]]&=x[:,[6,7]]>=-273.15
    valid_input[:,9:13]&=np.abs(x[:,9:13])<=1
    hp=np.vstack((np.zeros((1,13),dtype=np.int64),np.cumsum(~valid_input,axis=0)))
    missing_features=np.zeros((len(origins),13),dtype=bool)
    z=origins[contained];missing_features[contained]=hp[z]!=hp[z-HISTORY]
    joint=complete.all(axis=1)
    missing_targets=contained&~joint
    missing_history=contained&joint&missing_features.any(axis=1)
    accepted=contained&joint&~missing_features.any(axis=1)
    assert np.all(insufficient.astype(int)+crossing+missing_targets+missing_history+accepted==1)
    selected=origins[accepted]
    y=np.empty((len(selected),2),dtype=np.float64)
    for i,o in enumerate(selected):y[i]=x[o:o+6,[0,6]].mean(axis=0,dtype=np.float64)
    y[:,0]*=1000
    counts={'candidate_hourly_origins':len(origins),'insufficient_whole_history':int(insufficient.sum()),
        'target_crosses_split_end':int(crossing.sum()),'unobserved_joint_target':int(missing_targets.sum()),
        'incomplete_input_history':int(missing_history.sum()),'valid_windows':len(selected),
        'complete_power_target_hours':int(complete[:,0].sum()),
        'complete_temperature_target_hours':int(complete[:,1].sum()),'complete_joint_target_hours':int(joint.sum()),
        'complete_history_origins':int((contained&~missing_features.any(axis=1)).sum())}
    return selected,y,counts,missing_features[contained&joint].sum(axis=0).astype(int)


def eligibility(counts,days,minimum_windows,minimum_days):
    counts,days=np.asarray(counts),np.asarray(days)
    if counts.shape!=days.shape or minimum_windows<1 or minimum_days<0:raise ValueError('Invalid eligibility')
    return (counts>=minimum_windows)&(days>=minimum_days)


def scenario_has(scenario,split):
    if scenario not in SCENARIOS or split not in ('train','val','test'):raise ValueError('Unknown scenario/split')
    return scenario=='january_march' or (scenario=='march_only')==(split=='test')


def spool(files,names,scratch,intervals,limits):
    """Bounded Arrow batches; stable declaration/file/row order retained on disk."""
    size=limits['nodes_per_shard'];number=math.ceil(len(names)/size)
    paths=[scratch/f'{i:04d}.bin' for i in range(number)];counts=np.zeros(number,dtype=np.int64)
    lookup=pa.array(names,type=pa.string());quality=[]
    with ExitStack() as stack:
        handles=[stack.enter_context(p.open('xb')) for p in paths]
        for path in files:
            source=pq.ParquetFile(path); schema=source.schema_arrow
            if not {'timestamp','value','node'}<=set(schema.names):raise ValueError('Raw schema missing columns: '+str(path))
            ts=schema.field('timestamp').type
            if not pa.types.is_timestamp(ts) or ts.tz!='UTC':raise ValueError('Explicit UTC timestamps required: '+str(path))
            q={'path':str(path),'raw_rows':0,'null_time':0,'null_node':0,'unmapped_rows':0,'outside_declared_splits':0,'selected_rows':0,'unknown_nodes':set(),'first_utc_ns':None,'last_utc_ns':None}
            for batch in source.iter_batches(batch_size=limits['arrow_batch_rows'],columns=['timestamp','value','node'],use_threads=False):
                q['raw_rows']+=len(batch);good=pc.is_valid(batch['timestamp']);q['null_time']+=len(batch)-int(pc.sum(pc.cast(good,pa.int64())).as_py() or 0)
                nodes=pc.cast(pc.filter(batch['node'],good),pa.string());q['null_node']+=nodes.null_count
                ids=pc.fill_null(pc.index_in(nodes,value_set=lookup),-1).to_numpy(zero_copy_only=False)
                times=pc.cast(pc.cast(pc.filter(batch['timestamp'],good),pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy(zero_copy_only=False)
                values=pc.cast(pc.filter(batch['value'],good),pa.float64()).to_numpy(zero_copy_only=False)
                if len(times):
                    q['first_utc_ns']=int(times.min()) if q['first_utc_ns'] is None else min(q['first_utc_ns'],int(times.min()))
                    q['last_utc_ns']=int(times.max()) if q['last_utc_ns'] is None else max(q['last_utc_ns'],int(times.max()))
                known=ids>=0;q['unmapped_rows']+=int((~known).sum());q['unknown_nodes'].update(v for v in pc.unique(pc.filter(nodes,pa.array(~known))).to_pylist() if v is not None)
                if len(q['unknown_nodes'])>limits['max_unknown_nodes']:raise ValueError('Unknown-ID evidence bound exceeded')
                in_scope=np.zeros(len(times),dtype=bool)
                for lo,hi in intervals:in_scope|=(times>=lo)&(times<hi)
                q['outside_declared_splits']+=int((known&~in_scope).sum());keep=known&in_scope
                rows=np.empty(int(keep.sum()),dtype=RAW_DTYPE);rows['node']=ids[keep];rows['time']=times[keep];rows['value']=values[keep]
                shards=rows['node']//size
                for shard in np.unique(shards):
                    part=rows[shards==shard];counts[shard]+=len(part)
                    if int(counts.sum())*RAW_DTYPE.itemsize>limits.get('max_spool_bytes',12*1024**3):raise ValueError('Total spool budget exceeded')
                    if counts[shard]>limits['max_shard_rows']:raise ValueError('Raw shard row limit exceeded')
                    handles[shard].write(part.tobytes())
                q['selected_rows']+=len(rows)
            q['unknown_nodes']=sorted(q['unknown_nodes']);quality.append(q)
    return paths,counts,quality


def bin_local_metric(files,metric,names,edges_by_split,scratch,limits):
    result={s:np.lib.format.open_memmap(scratch/f'{metric}_{s}.npy',mode='w+',dtype=np.float64,
            shape=(len(names),len(edges)-1)) for s,edges in edges_by_split.items()}
    for a in result.values():a[:]=np.nan
    quality=[]
    with tempfile.TemporaryDirectory(prefix=metric+'-',dir=scratch) as temporary:
        paths,counts,sources=spool(files,names,Path(temporary),[(int(e[0]),int(e[-1])) for e in edges_by_split.values()],limits)
        for shard,path in enumerate(paths):
            if not counts[shard]:continue
            rows=np.memmap(path,mode='r',dtype=RAW_DTYPE)
            for idx in range(shard*limits['nodes_per_shard'],min((shard+1)*limits['nodes_per_shard'],len(names))):
                chosen=np.flatnonzero(rows['node']==idx)
                if len(chosen)>limits['max_node_rows']:raise ValueError('Raw node row limit exceeded')
                raw=rows[chosen];t,v,q=canonical_rows(raw['time'],raw['value'],metric)
                for split,edges in edges_by_split.items():
                    b,c=bin_split(t,v,metric,edges);result[split][idx]=b
                    quality.append({'node':names[idx],'node_index':idx,'metric':metric,'split':split,**{'all_splits_'+k:v for k,v in q.items()},
                        'split_raw_rows':int(((t>=edges[0])&(t<edges[-1])).sum()),'valid_bins':int(np.isfinite(b).sum()),
                        'below_coverage_bins':int((c<0.5).sum())})
            del rows
    for a in result.values():a.flush()
    return result,quality,sources


def weather_bins(files,metric,edges_by_split,limit):
    ts=[];vs=[];total=0
    for path in files:
        f=pq.ParquetFile(path);kind=f.schema_arrow.field('timestamp').type
        if not pa.types.is_timestamp(kind) or kind.tz!='UTC':raise ValueError('UTC weather required')
        for b in f.iter_batches(batch_size=65536,columns=['timestamp','value'],use_threads=False):
            total+=len(b)
            if total>limit:raise ValueError('Weather row bound exceeded')
            good=pc.is_valid(b['timestamp']);ts.append(pc.cast(pc.cast(pc.filter(b['timestamp'],good),pa.timestamp('ns',tz='UTC')),pa.int64()).to_numpy());vs.append(pc.cast(pc.filter(b['value'],good),pa.float64()).to_numpy())
    t,v,q=canonical_rows(np.concatenate(ts),np.concatenate(vs),metric)
    return {s:bin_split(t,v,metric,e)[0] for s,e in edges_by_split.items()},q


def check_bindings(root,config):
    if config['schema']!='march_window_audit.v1' or set(config['metric_files'])!=set(LOCAL_METRICS)|{'temp','humidity'}:raise ValueError('Audit metric declaration differs')
    for paths in config['metric_files'].values():
        if not paths or len(paths)!=len(set(paths)) or any(p not in config['inputs'] for p in paths):raise ValueError('Unbound or duplicate raw inputs')
    count=0
    for relative,expected in config['inputs'].items():
        p=safe_path(root,relative)
        if not p.is_file() or p.stat().st_size!=expected['bytes'] or sha256(p)!=expected['sha256']:raise ValueError('Audit input changed: '+relative)
        count+=1
    manifest=json.loads(safe_path(root,config['design_manifest']).read_text());verify_design(root,manifest)
    protocol=json.loads(safe_path(root,config['protocol']).read_text());validate_protocol(protocol)
    if protocol['primary_history_steps']!=36 or protocol['horizon_steps']!=6:raise ValueError('Wrong window protocol')
    names=json.loads(safe_path(root,protocol['mapping']['source']).read_text())
    if len(names)!=979 or len(set(names))!=979 or not all(isinstance(n,str) for n in names):raise ValueError('Original mapping required')
    return protocol,names,count


def validate_records(records,names,split,protocol):
    if records.dtype!=WINDOW_DTYPE or records.ndim!=1:raise ValueError('Audit window schema changed')
    idx=records['node_index'];origin=records['origin_ns']
    if (idx<0).any() or (idx>=len(names)).any():raise ValueError('Unmapped audit row')
    if len(records)>1 and not np.all((idx[1:]>idx[:-1])|((idx[1:]==idx[:-1])&(origin[1:]>origin[:-1]))):raise ValueError('Duplicate/unordered node origins')
    lo,hi=map(utc_ns,protocol['splits'][split])
    if ((origin%(6*WIDTH))!=0).any() or (origin-HISTORY*WIDTH<lo).any() or (origin+HORIZON*WIDTH>hi).any():raise ValueError('Audit split containment failed')
    if not np.isfinite(records['power_w']).all() or not np.isfinite(records['temperature_c']).all() or (records['power_w']<0).any() or (records['temperature_c']<-273.15).any():raise ValueError('Invalid observed targets')
    return True


def verify_audit(output):
    receipt=json.loads((output/'completion.json').read_text())
    actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p.name!='completion.json'}
    if receipt['status']!='coverage_audit_complete' or actual!=set(receipt['artifacts']):raise ValueError('Audit inventory differs')
    for n,info in receipt['artifacts'].items():
        p=safe_path(output,n)
        if p.stat().st_size!=info['bytes'] or sha256(p)!=info['sha256']:raise ValueError('Audit artifact changed: '+n)
    p=json.loads((output/'protocol_snapshot.json').read_text());cfg=json.loads((output/'configuration.json').read_text())
    names=json.loads((output/'node_mapping.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    arrays={}
    if len(summary['split_coverage'])!=18:raise ValueError('Expected 18 scenario/split/cohort summaries')
    for row in summary['split_coverage']:
        name=f"{row['scenario']}_{row['split']}_{row['cohort']}_windows.npy"
        a=np.load(output/name,allow_pickle=False);validate_records(a,names,row['split'],p)
        if len(a)!=row['valid_windows'] or identity_digest(np.asarray(names)[a['node_index']],a['origin_ns'].astype('datetime64[ns]'))!=row['ordered_identity_sha256']:raise ValueError('Saved window count/identity mismatch')
        if hashlib.sha256(np.column_stack((a['power_w'],a['temperature_c'])).tobytes()).hexdigest()!=row['target_bytes_sha256']:raise ValueError('Saved observed target mismatch')
        arrays[(row['scenario'],row['split'],row['cohort'])]=a
    for split in ('train','val','test'):
        present=[arrays[(scenario,split,'native_mapped')] for scenario in SCENARIOS if scenario_has(scenario,split)]
        if any(not np.array_equal(present[0],v) for v in present[1:]):raise ValueError('Scenario shared identities/targets differ')
    return {'status':'verified_coverage_audit','artifacts':len(actual),'window_arrays_checked':len(arrays),'model_operations':False}


def execute(root,config,output,config_path):
    started=time.monotonic();protocol,names,checks=check_bindings(root,config)
    if output.exists() or output.is_symlink():raise ValueError('Output exists; preserve it and use a new approved version')
    parent=output.parent
    while not parent.exists():parent=parent.parent
    space=os.statvfs(parent)
    if space.f_bavail*space.f_frsize<config['limits']['minimum_free_bytes']:raise ValueError('Insufficient scratch capacity')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'execution_log.txt').open('x') as log: log.write('Coverage-only audit started; no models/scalers.\n')
    def log(message):
        print(message,flush=True)
        with (output/'execution_log.txt').open('a') as f:f.write(message+'\n')
    write_json(output/'configuration.json',config);write_json(output/'protocol_snapshot.json',protocol);write_json(output/'node_mapping.json',names)
    edges={s:np.arange(utc_ns(b[0]),utc_ns(b[1])+WIDTH,WIDTH,dtype=np.int64) for s,b in protocol['splits'].items()}
    qualities=[];source_quality=[];node_rows=[];exclusions=[];missing_rows=[];memory_rows=[];all_records={s:[] for s in edges};counts={s:np.zeros(len(names),dtype=np.int64) for s in edges};days={s:np.zeros(len(names),dtype=np.int64) for s in edges}
    with tempfile.TemporaryDirectory(prefix='.owned-scratch-',dir=output) as tmp:
        scratch=Path(tmp);local={}
        for metric in LOCAL_METRICS:
            log('AUDIT binning '+metric+' (bounded raw scan)')
            inputs=[safe_path(root,n) for n in config['metric_files'][metric]]
            local[metric],q,src=bin_local_metric(inputs,metric,names,edges,scratch,config['limits']);qualities.extend(q);source_quality.extend(src)
        weather={}
        for metric in ('temp','humidity'):
            weather[metric],q=weather_bins([safe_path(root,n) for n in config['metric_files'][metric]],metric,edges,config['limits']['max_weather_rows'])
            write_json(output/f'weather_{metric}_quality.json',q)
        for idx,node in enumerate(names):
            for split,e in edges.items():
                raw=np.column_stack([local[m][split][idx] for m in LOCAL_METRICS])
                x=node_features(raw,{m:weather[m][split] for m in weather},e)
                mem,mc=aligned_memory(local['mem_free'][split][idx],local['mem_total'][split][idx]);assert np.array_equal(x[:,5],mem,equal_nan=True)
                selected,targets,cnt,miss=window_masks(x,e)
                origins=e[selected]
                if len(origins) and any(split_for_origin(int(o),36,protocol)!=split for o in (origins[0],origins[-1])):raise ValueError('Frozen split contract mismatch')
                counts[split][idx]=len(selected);days[split][idx]=len(np.unique(origins.astype('datetime64[ns]').astype('datetime64[D]')))
                row={'node':node,'node_index':idx,'split':split,**cnt,'origin_days':int(days[split][idx]),'valid_power_bins':int(np.isfinite(x[:,0]).sum()),'valid_temperature_bins':int(np.isfinite(x[:,6]).sum()),'valid_joint_target_bins':int(np.isfinite(x[:,[0,6]]).all(axis=1).sum()),'valid_full_input_bins':int(np.isfinite(x).all(axis=1).sum())};node_rows.append(row)
                memory_rows.append({'node':node,'node_index':idx,'split':split,**mc})
                for reason in ['insufficient_whole_history','target_crosses_split_end','unobserved_joint_target','incomplete_input_history','valid_windows']:
                    exclusions.append({'node':node,'split':split,'reason':reason,'count':cnt[reason],'disjoint_priority_order':True})
                for channel,total in enumerate(miss):missing_rows.append({'node':node,'split':split,'channel':channel,'feature':protocol['features'][channel],'incomplete_histories_with_observed_joint_targets':int(total),'overlapping_reasons':True})
                records=np.empty(len(selected),dtype=WINDOW_DTYPE);records['node_index']=idx;records['origin_ns']=origins;records['power_w']=targets[:,0];records['temperature_c']=targets[:,1];all_records[split].append(records)
        del local
    records={s:np.concatenate(v) for s,v in all_records.items()}
    for split,arr in records.items(): validate_records(arr,names,split,protocol)
    write_csv(output/'input_inventory.csv',[{'path':k,'bytes':v['bytes'],'sha256':v['sha256']} for k,v in config['inputs'].items()])
    for relative in config['new_implementation_files']:
        source=safe_path(root,relative);dest=output/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(source.read_bytes())
    write_csv(output/'coverage_by_node.csv',node_rows);write_csv(output/'window_exclusions.csv',exclusions);write_csv(output/'input_feature_exclusions.csv',missing_rows);write_csv(output/'memory_alignment.csv',memory_rows);write_csv(output/'raw_metric_quality.csv',qualities);write_json(output/'source_quality.json',source_quality)
    thresholds=[];summaries=[];scenario_nodes=[];origin_rows=[];date_rows=[];overlaps=[]
    for scenario in SCENARIOS:
        traincounts=counts['train'] if scenario_has(scenario,'train') else np.zeros(len(names),dtype=np.int64)
        traindays=days['train'] if scenario_has(scenario,'train') else np.zeros(len(names),dtype=np.int64)
        elig=eligibility(traincounts,traindays,24,7)
        for label,mindata,mindays in SENSITIVITY:
            accepted=eligibility(traincounts,traindays,mindata,mindays)
            thresholds.append({'scenario':scenario,'threshold':label,'minimum_windows':mindata,'minimum_origin_dates':mindays,'eligible_training_clients':int(accepted.sum()),'excluded_training_clients':len(names)-int(accepted.sum()),**{split+'_retained_windows':int(accepted[records[split]['node_index']].sum()) if scenario_has(scenario,split) else 0 for split in records},**{split+'_retained_nodes':len(np.unique(records[split]['node_index'][accepted[records[split]['node_index']]])) if scenario_has(scenario,split) else 0 for split in records},'training_only':True,'frozen_threshold':label=='windows_ge24_days_ge7'})
        for split,full in records.items():
            rec=full if scenario_has(scenario,split) else full[:0]
            common=rec[elig[rec['node_index']]]
            for label,arr in [('native_mapped',rec),('training_eligible_common',common)]:
                ids=np.asarray(names)[arr['node_index']];org=arr['origin_ns'].astype('datetime64[ns]')
                signatures=identity_digest(ids,org)
                summaries.append({'scenario':scenario,'split':split,'cohort':label,'valid_windows':len(arr),'nodes_with_windows':len(np.unique(arr['node_index'])),'unique_origins':len(np.unique(org)),'dates':len(np.unique(org.astype('datetime64[D]'))),'eligible_training_clients':int(elig.sum()),'ordered_identity_sha256':signatures,'target_bytes_sha256':hashlib.sha256(np.column_stack((arr['power_w'],arr['temperature_c'])).tobytes()).hexdigest()})
                out=output/f'{scenario}_{split}_{label}_windows.npy'
                with out.open('xb') as f:np.save(f,arr,allow_pickle=False)
                u,n=np.unique(arr['origin_ns'],return_counts=True)
                origin_rows.extend({'scenario':scenario,'split':split,'cohort':label,'origin_utc':str(np.datetime64(int(t),'ns'))+'Z','windows':int(k)} for t,k in zip(u,n))
                day,n=np.unique(org.astype('datetime64[D]'),return_counts=True)
                date_rows.extend({'scenario':scenario,'split':split,'cohort':label,'date_utc':str(d),'windows':int(k),'origins':len(np.unique(org[org.astype('datetime64[D]')==d]))} for d,k in zip(day,n))
            for i,name in enumerate(names):scenario_nodes.append({'scenario':scenario,'split':split,'node':name,'node_index':i,'valid_windows':int(counts[split][i]) if scenario_has(scenario,split) else 0,'training_windows':int(traincounts[i]),'training_origin_dates':int(traindays[i]),'eligible_training_client':bool(elig[i])})
    for split,rec in records.items():
        for i,a in enumerate(SCENARIOS):
            for b in SCENARIOS[i+1:]:
                shared=len(rec) if scenario_has(a,split) and scenario_has(b,split) else 0
                overlaps.append({'split':split,'scenario_a':a,'scenario_b':b,'shared_native_window_ids':shared,'identical_observed_targets_and_split':True,'explanation':'Same strictly-contained split rows; absent input months yield empty cohort'})
    write_csv(output/'split_coverage.csv',summaries);write_csv(output/'scenario_coverage_by_node.csv',scenario_nodes);write_csv(output/'client_eligibility_sensitivity.csv',thresholds);write_csv(output/'origin_coverage.csv',origin_rows,['scenario','split','cohort','origin_utc','windows']);write_csv(output/'date_coverage.csv',date_rows,['scenario','split','cohort','date_utc','windows','origins']);write_csv(output/'scenario_overlap.csv',overlaps)
    write_json(output/'summary.json',{'status':'coverage_only','history_steps':36,'horizon_steps':6,'mapped_clients':len(names),'split_coverage':summaries,'eligibility':thresholds,'weather_variants_use_identical_ids':True,'historical_preflight':'FAILED unchanged; not executed/bypassed','scalers_fitted':False,'model_operations':False,'runtime_seconds':time.monotonic()-started,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    log('All node/window counts calculated; verifying input bindings before completion.')
    # Raw byte bindings are rechecked after scans; no source file writes occur.
    check_bindings(root,config)
    report=build_report(summaries,thresholds,started,checks)
    with (output/'REPORT.md').open('x',encoding='utf-8') as f:f.write(report)
    write_json(output/'completion.json',{'status':'coverage_audit_complete','config_path':str(config_path),'config_sha256':sha256(config_path),'protocol_modified':False,'model_operations':False,'artifacts':{str(p.relative_to(output)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in output.rglob('*') if p.is_file()}})
    return verify_audit(output)


def build_report(summaries,thresholds,started,checks):
    text='''# March protocol-valid streaming window audit\n\nCoverage only: no training, model initialization/inference/scoring, scaler fitting, downloads or extraction. Historical preflight remains FAILED; this independent scope does not replay or repair it.\n\n## Protocol and interpretation\n\nAll 14 slots retained. UTC 600-second bins; causal holds <=300 seconds; each local bin requires >=50% coverage. No policy forward-fill or target imputation. Stable keep-last deduplication, finite/range checks and original calendar features are reused. Weather is latest strictly before bin right edge, max age 1800 seconds. Raw holds/weather reset at every declared split lower edge. Histories and six joint target bins are wholly split-contained; origins are hourly and future target intervals do not overlap within a node.\n\nMemory is 1 - mean_binned(mem_free)/mean_binned(mem_total), aligned by node and UTC bin. Both independent bins must be valid, total >0, and 0<=free<=total. This preserves the feature; it does not assert identical raw sample timestamps or simultaneous within-bin support. Invalid memory remains missing. Input weather enabled/masked variants use identical full-weather-valid window IDs; masking standardized slots7/8 is tested synthetically only, since no real scaler is fitted.\n\nTrain Jan1--Feb11 07:10 UTC; validation through Feb20 03:30; old Feb20--Mar1 test interval excluded; March1--Apr1 is test. These boundaries are never recomputed as data sources change. March-only therefore has no training eligibility; Jan--Feb-only has no March test. Comparisons are availability diagnostics, not alternative trained experiments. Original979 mapping retained; unknown raw IDs are reported/excluded without reindexing.\n\nTargets are next-hour arithmetic means of six coverage-qualified original bins: power in W and documented IPMI ambient/inlet temperature in degC. Continuous observation, calibrated placement, privacy, savings and predictive superiority are not established.\n\n## Counts\n\n| Scenario | Split | Cohort | Windows | Nodes | Origins | Dates | Eligible training clients |\n| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |\n'''
    for r in summaries:text+='| '+ ' | '.join(str(r[k]) for k in ['scenario','split','cohort','valid_windows','nodes_with_windows','unique_origins','dates','eligible_training_clients'])+' |\n'
    text+='\n## Outputs, reconciliation and limitations\n\nPer-node, memory, raw quality, disjoint exclusions, overlapping feature exclusions, origin/date, training-only threshold and source-overlap tables are saved separately. Each scenario/split/cohort has ordered structured NPY audit IDs plus observed two-target means, not dense model tensors or a training freeze. Empty arrays are explicit. Primary common test restricts to training-eligible nodes; weather masks do not enlarge that cohort. Dates/origins and gaps are visible in the CSVs. Reason priority is insufficient history, crossing target, missing joint target, missing input, accepted; these counts sum exactly to all hourly candidates per node. Feature reasons overlap and must not be summed.\n\nThe preliminary audit only counted timestamp occupancy, ignored sensor values/hold coverage/joint windows/weather and used bin counts rather than training-window counts for eligibility. Its percentages cannot substitute for this audit. No raw-to-historical frozen semantic reproduction or actual scaler/model validation is claimed. All local hash receipts prove consistency, not independent authenticity. Original mapping had retrospective admission bias; client IDs do not prove hardware continuity. The test period is retrospective, not automatically untouched confirmatory data.\n\n'
    primary=next(r for r in summaries if r['scenario']=='january_march' and r['split']=='test' and r['cohort']=='training_eligible_common')
    text+='## Readiness decision\n\n'
    if primary['valid_windows']==0 or primary['eligible_training_clients']<98:text+='BLOCKED: the declared complete-input/eligibility protocol has insufficient support. Preserve this result; no feature removal, imputation or split change is authorized. Review alternatives explicitly before any further experiment.\n'
    else:text+='Coverage supports further review only; it does not authorize preparation, scaling, pilot or training. Review temporal concentration, sparse per-node coverage, provenance acceptance, and new real-data adapters separately.\n'
    text+=f'\nInput bindings checked before/after: {checks}; elapsed {time.monotonic()-started:.2f} seconds. Process lifetime peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024} bytes (includes imports/scans/page cache). Exact input/config hashes are in configuration.json; artifact hashes in completion.json. Historical artifacts are checked separately by the task preservation record.\n'
    return text


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/audits/march_window_audit_v1.json')
    parser.add_argument('--output-dir',default='output/march_window_audit/v1')
    group=parser.add_mutually_exclusive_group();group.add_argument('--execute-audit',action='store_true');group.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv);cfg_path=safe_path(ROOT,args.config);cfg=json.loads(cfg_path.read_text());out=safe_path(ROOT,args.output_dir)
    if args.verify_only:result=verify_audit(out)
    elif args.execute_audit:result=execute(ROOT,cfg,out,cfg_path)
    else:
        protocol,names,checks=check_bindings(ROOT,cfg)
        result={'status':'read_only_preflight','output_exists':out.exists(),'bindings_checked':checks,'mapped_clients':len(names),'archive_requests':False,'model_operations':False,'writes':False}
    print(json.dumps(result,indent=2));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,KeyError,AssertionError) as e:
        print('MARCH WINDOW AUDIT ERROR: '+str(e));raise SystemExit(2)
