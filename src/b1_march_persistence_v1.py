"""Additive, CPU-only persistence comparison on frozen B1 March windows.

Never constructs a neural model or fits statistics. Existing source/locks unchanged.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,statistics,time
import numpy as np
from src import b1_march_data_v1 as d
from src import b1_march_integrity_v1 as integrity
from src import b1_march_evaluate_v1 as evaluation
from src.phase2_multioutput.contracts import Windows
from src.phase2_multioutput.evaluation import persistence as established_persistence

OUTPUT='output/experiments/b1_no_ff/march_persistence_comparison_v1'
COMPLETION_SHA='2075bb1a14ebb8d192f6cfd8c5e4f8e82ac010023633769db926d1e2bbee36d3'
RECEIPT_SHA='9512af4d81048b80a024fbfcf529fa602a3283fd1f6cb4e328649ca212ab5c78'
LABELS=('power_W','ambient_degC')
GROUPS=('pooled','macro_node')
METRICS=('mae','rmse')


def alignment(reference,p,actual):
    """Exact observed labels and original identities BEFORE scoring anything."""
    if np.ma.isMaskedArray(reference) or reference.dtype!=d.audit.WINDOW_DTYPE or reference.ndim!=1:
        raise ValueError('Original structured reference required')
    if np.ma.isMaskedArray(actual) or np.asarray(actual).dtype.kind not in 'iuf':
        raise ValueError('Unmasked nonboolean physical targets required')
    if len(reference)==0 or (reference['node_index']<0).any() or (reference['node_index']>=979).any():raise ValueError('Invalid client mapping ordinal')
    keys=list(zip(reference['node_index'].tolist(),reference['origin_ns'].tolist()))
    if keys!=sorted(set(keys)):raise ValueError('Duplicate or reordered reference identities')
    y=np.column_stack((reference['power_w'],reference['temperature_c']))
    if not np.isfinite(y).all() or not np.array_equal(actual,y):raise ValueError('Observed targets changed/missing')
    nodes=np.asarray(d.bridge.ORIGINAL_NAMES)[reference['node_index']]
    if not np.array_equal(p['nodes'],nodes) or not np.array_equal(p['origins'].view('i8'),reference['origin_ns']):
        raise ValueError('Node/origin alignment differs')


def packet_persistence(p):
    actual=d.validate_packet(p)  # hardened mask/boolean validation BEFORE Windows conversion
    keys=('x','observed','future','target_observed','history_times','target_times',
          'history_nodes','target_nodes','source_latest','source_earliest','nodes','origins')
    w=Windows(**{k:p[k] for k in keys},mapping=d.bridge.ORIGINAL_NAMES,split='test',history=36)
    predicted=established_persistence(w,d.TEST_PROTOCOL)
    direct=p['x'][:,-1,[0,6]].astype('f8',copy=True);direct[:,0]*=1000
    if not np.array_equal(predicted,direct) or not np.isfinite(predicted).all():
        raise ValueError('Persistence definition/units differ')
    if not np.array_equal(p['history_times'][:,-1].view('i8')+d.WIDTH,p['origins'].view('i8')):
        raise ValueError('Last completed ten-minute input bin required')
    return predicted,actual


def change(baseline,model):
    if not all(math.isfinite(float(x)) and x>=0 for x in (baseline,model)):
        raise ValueError('Finite nonnegative errors required')
    delta=float(baseline)-float(model)
    return {'improvement_abs':delta,'improvement_percent':100*delta/baseline if baseline>0 else None,
            'direction':'improved' if delta>0 else 'degraded' if delta<0 else 'equal',
            'percent_defined':baseline>0}


def distribution(values):
    a=np.asarray(values,dtype='f8')
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Finite nonempty distribution required')
    q=np.percentile(a,[0,5,25,50,75,95,100],method='linear')
    return dict(zip(('min','p05','p25','median','p75','p95','max'),map(float,q)),mean=float(a.mean()))


def tables(baseline,scores,names):
    if set(scores)!={42,43,44}:raise ValueError('All prescribed seeds required')
    for s in scores.values():
        if (s['samples'],s['identity_sha256'])!=(baseline['samples'],baseline['identity_sha256']) or set(s['per_node'])!=set(baseline['per_node']):
            raise ValueError('Unequal evaluation cohort')
        if any(s['per_node'][n]['samples']!=baseline['per_node'][n]['samples'] for n in baseline['per_node']):raise ValueError('Node sample counts differ')
    baseline_rows=[];comparison=[];nodes=[];node_comparisons=[];summaries=[];consistency=[]
    for group in GROUPS:
        for j,label in enumerate(LABELS):
            baseline_rows.append({'aggregation':group,'output':label,'MAE':baseline[group]['mae'][j],'RMSE':baseline[group]['rmse'][j],'windows':baseline['samples'],'nodes':len(baseline['per_node'])})
            for metric in METRICS:
                for model in (42,43,44,'seed_mean'):
                    values=[scores[s][group][metric][j] for s in (42,43,44)]
                    value=statistics.mean(values) if model=='seed_mean' else scores[model][group][metric][j]
                    b=baseline[group][metric][j]
                    comparison.append({'model':str(model),'aggregation':group,'output':label,'metric':metric,'persistence':b,'B1':value,'B1_sample_sd':statistics.stdev(values) if model=='seed_mean' else None,**change(b,value)})
    for ordinal,name in enumerate(names):
        if name not in baseline['per_node']:continue
        b=baseline['per_node'][name]
        nodes.append({'node_index':ordinal,'node':name,'windows':b['samples'],'power_MAE_W':b['mae'][0],'power_RMSE_W':b['rmse'][0],'ambient_MAE_degC':b['mae'][1],'ambient_RMSE_degC':b['rmse'][1]})
        for j,label in enumerate(LABELS):
            for metric in METRICS:
                for model in (42,43,44,'seed_mean'):
                    values=[scores[s]['per_node'][name][metric][j] for s in (42,43,44)]
                    value=statistics.mean(values) if model=='seed_mean' else scores[model]['per_node'][name][metric][j]
                    node_comparisons.append({'model':str(model),'node_index':ordinal,'node':name,'windows':b['samples'],'output':label,'metric':metric,'persistence':b[metric][j],'B1':value,**change(b[metric][j],value)})
    for model in ('42','43','44','seed_mean'):
        for label in LABELS:
            for metric in METRICS:
                rows=[r for r in node_comparisons if (r['model'],r['output'],r['metric'])==(model,label,metric)]
                pct=[r['improvement_percent'] for r in rows if r['percent_defined']]
                summary={'model':model,'output':label,'metric':metric,'nodes':len(rows),
                    **{key:sum(r['direction']==key for r in rows) for key in ('improved','degraded','equal')},'undefined_percentage_nodes':len(rows)-len(pct)}
                summary.update({'absolute_'+k:v for k,v in distribution([r['improvement_abs'] for r in rows]).items()})
                summary.update({'percent_'+k:v for k,v in (distribution(pct).items() if pct else {k:None for k in ('min','p05','p25','median','p75','p95','max','mean')}.items())})
                summaries.append(summary)
    for j,label in enumerate(LABELS):
        for metric in METRICS:
            delta=np.array([[baseline['per_node'][n][metric][j]-scores[s]['per_node'][n][metric][j] for s in (42,43,44)] for n in baseline['per_node']])
            better=int((delta>0).all(axis=1).sum());worse=int((delta<0).all(axis=1).sum());equal=int((delta==0).all(axis=1).sum())
            consistency.append({'output':label,'metric':metric,'nodes':len(delta),'improved_all_three':better,'degraded_all_three':worse,'equal_all_three':equal,'mixed_or_partial_ties':len(delta)-better-worse-equal})
    return {'persistence_metrics.csv':baseline_rows,'comparison_metrics.csv':comparison,'persistence_by_node.csv':nodes,
        'node_comparisons.csv':node_comparisons,'node_improvement_summary.csv':summaries,'seed_consistency.csv':consistency}


def write_json(path,value):
    integrity.io.exclusive(path,integrity.io.canonical(value))


def write_csv(path,rows):
    if not rows:raise ValueError('Empty result table')
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def figures(output,tab):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'savefig.dpi':220,'figure.constrained_layout.use':True})
    colors=['#616161','#4477AA','#228833','#CC6677','#AA3377']
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,group in enumerate(GROUPS):
        for j,label in enumerate(LABELS):
            ax=axes[row,j];x=np.arange(2);width=.15
            for k,model in enumerate(('persistence','42','43','44','seed_mean')):
                ys=[];err=[]
                for metric in METRICS:
                    rows=[r for r in tab['comparison_metrics.csv'] if r['aggregation']==group and r['output']==label and r['metric']==metric]
                    r=next(r for r in rows if r['model']==('42' if model=='persistence' else model))
                    ys.append(r['persistence'] if model=='persistence' else r['B1']);err.append(r['B1_sample_sd'] if model=='seed_mean' else 0)
                ax.bar(x+(k-2)*width,ys,width,label='Persistence' if model=='persistence' else 'B1 seed mean ± SD' if model=='seed_mean' else 'B1 seed '+model,color=colors[k],yerr=err,capsize=3)
            ax.set_xticks(x,['MAE','RMSE']);ax.set_ylabel('W' if j==0 else '°C');ax.set_title(('Pooled' if row==0 else 'Equal-node macro')+' · '+('IT power' if j==0 else 'Recorded ambient'))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('B1 vs persistence · identical March windows\nPreviously inspected retrospective evaluation');fig.savefig(output/'metric_comparison.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,label in enumerate(LABELS):
        for k,metric in enumerate(METRICS):
            ax=axes[j,k]
            for seed,color in zip(('42','43','44'),colors[1:4]):
                vals=np.sort([r['improvement_abs'] for r in tab['node_comparisons.csv'] if (r['model'],r['output'],r['metric'])==(seed,label,metric)])
                ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label='Seed '+seed,color=color)
            ax.axvline(0,color='black',linestyle='--',linewidth=1);ax.set_xlabel('Persistence error − B1 error ('+('W' if j==0 else '°C')+')');ax.set_ylabel('Fraction of represented nodes');ax.set_title(label+' · '+metric.upper());ax.grid(alpha=.2)
    axes[0,0].legend();fig.suptitle('Node-level change distribution · positive means B1 improvement\n976 equally weighted nodes; same retrospective March windows');fig.savefig(output/'node_improvement_ecdf.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for j,label in enumerate(LABELS):
        ax=axes[j];rows=[r for r in tab['seed_consistency.csv'] if r['output']==label];left=np.zeros(2)
        for key,color,name in [('improved_all_three','#228833','Improved in all seeds'),('degraded_all_three','#CC3311','Degraded in all seeds'),('equal_all_three','#BBBBBB','Equal in all seeds'),('mixed_or_partial_ties','#4477AA','Mixed / partial ties')]:
            v=np.array([r[key] for r in rows]);ax.barh(np.arange(2),v,left=left,label=name,color=color);left+=v
        ax.set_yticks(np.arange(2),[r['metric'].upper() for r in rows]);ax.set_xlabel('Nodes (total 976)');ax.set_title(label)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=2);fig.suptitle('Consistency across the three prescribed seeds');fig.savefig(output/'seed_consistency.png');plt.close(fig)


def analyze(plan_path,stage):
    """Write to a caller-owned unused stage; final publication is separate."""
    stage=integrity.io.safe_path(stage)
    if stage.exists():raise FileExistsError('Analysis stage occupied')
    plan=integrity.load_plan(plan_path);lock,scaler,lockcheck=integrity.locked_context(plan)
    data=d.MarchData(integrity.ROOT/plan['dataset_output']/'dataset',plan,scaler)
    old=integrity.ROOT/plan['evaluation_output'];receiptpath=integrity.ROOT/plan['receipt_output']/'receipt.json'
    if integrity.io.binding(old/'completed.json')['sha256']!=COMPLETION_SHA or integrity.io.binding(receiptpath)['sha256']!=RECEIPT_SHA:raise ValueError('Completed evaluation/receipt changed')
    receipt=integrity.read_json(receiptpath)
    checked=evaluation.verify_evaluation(old,plan,receipt,data)
    stage.mkdir(mode=0o700);start=time.monotonic();reference=data.reference
    predictions=np.empty((len(reference),2),dtype='f8');cursor=0
    for entry in data.manifest['nodes']:
        path=data.path/entry['path'];integrity.check(path,entry);bins=d.read_node(path)
        positions,y,_,_=d.audit.window_masks(bins['features'],d.EDGES)
        for first in range(0,len(positions),128):
            p=d.packet(entry['node'],bins,positions[first:first+128]);predicted,actual=packet_persistence(p)
            stop=cursor+len(actual);alignment(reference[cursor:stop],p,actual);predictions[cursor:stop]=predicted;cursor=stop
        integrity.check(path,entry)
    if cursor!=len(reference):raise ValueError('Incomplete persistence population')
    nodes=np.asarray(d.bridge.ORIGINAL_NAMES)[reference['node_index']];origins=reference['origin_ns'].astype('datetime64[ns]');actual=evaluation._actual(reference)
    base=evaluation.metrics(actual,predictions,nodes,origins)
    evaluation.compare_metrics(base,evaluation.independent_metrics(actual,predictions,nodes,origins))
    scores={s:v for s,v in ((int(s),v) for s,v in integrity.read_json(old/'metrics.json').items())}
    tab=tables(base,scores,d.bridge.ORIGINAL_NAMES)
    for name,rows in tab.items():write_csv(stage/name,rows)
    for name,arr in (('persistence_predictions.npy',predictions),('reference.npy',reference)):
        with (stage/name).open('xb') as f:np.save(f,arr,allow_pickle=False)
    write_json(stage/'persistence_metrics.json',base)
    # Independent direct extraction from the immutable bins, not packet/Windows.
    cursor=0;target_difference=[0.,0.]
    for entry in data.manifest['nodes']:
        n=entry['windows'];rows=reference[cursor:cursor+n];b=d.read_node(data.path/entry['path']);positions=((rows['origin_ns']-d.LO)//d.WIDTH).astype('i8')
        direct=b['features'][positions-1][:,[0,6]].copy();direct[:,0]*=1000
        if not np.array_equal(direct,predictions[cursor:cursor+n]):raise ValueError('Independent last-bin prediction mismatch')
        for row,pos in zip(rows,positions):
            target=[math.fsum(b['features'][pos:pos+6,0])/6*1000,math.fsum(b['features'][pos:pos+6,6])/6]
            truth=[float(row['power_w']),float(row['temperature_c'])]
            for j in range(2):
                target_difference[j]=max(target_difference[j],abs(target[j]-truth[j]))
                if not math.isclose(target[j],truth[j],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Independent observed target mismatch')
        cursor+=n
    if cursor!=194212:raise ValueError('Unexpected test population')
    with (stage/'persistence_predictions.npy').open('rb') as f:saved=np.load(f,allow_pickle=False)
    if not np.array_equal(saved,predictions):raise ValueError('Saved persistence differs')
    evaluation.compare_metrics(base,evaluation.independent_metrics(actual,saved,nodes,origins))
    figures(stage,tab)
    data.assert_unchanged()
    for rec in lock['selected_models']:integrity.check(integrity.ROOT/rec['selected_model']['path'],rec['selected_model'])
    for name,b in plan['sources'].items():integrity.check(integrity.ROOT/name,b)
    source_names=sorted(set(plan['sources'])|{'src/b1_march_persistence_v1.py','tests/test_b1_march_persistence_v1.py','docs/experiments/phase2_multioutput/v1/EVALUATION_SPEC.md','output/march_window_audit/b1_protocol_proposal_v1/PROTOCOL_PROPOSAL.md'})
    write_json(stage/'source_inventory.json',{'sources':{n:integrity.io.binding(integrity.ROOT/n) for n in source_names},'plan':integrity.io.binding(plan_path),'selection_lock_sha256':integrity.LOCK_SHA,'dataset_manifest':data.manifest_binding,'dataset_nodes':data.manifest['nodes'],'scaler_lock':lock['scaler_lock'],'scaler_statistics':lock['scaler_statistics'],'selected_models':lock['selected_models'],'old_evaluation_completion':integrity.io.binding(old/'completed.json'),'old_evaluation_artifacts':integrity.read_json(old/'completed.json')['artifacts'],'receipt':integrity.io.binding(receiptpath)})
    write_json(stage/'verification.json',{'status':'PASS','windows':len(reference),'represented_nodes':976,'mapping_nodes':979,'results':data.manifest['results'],'identity_target_reference_unchanged':True,'exact_independent_persistence_match':True,'saved_predictions_verified':True,'observed_target_independent_max_difference_W_degC':target_difference,'metric_comparison_rtol':1e-12,'metric_comparison_atol':1e-12,'existing_evaluation_verification':checked,'selection_verification':lockcheck,'baseline_definition':'Last completed ten-minute power kW*1000 and IPMI ambient degC; constant next-hour mean forecast','baseline_seed':None,'B1_neural_forward':False,'GPU_operations':False,'training':False,'scaler_fitting':False,'new_predictions':'Seedless persistence only','seconds_after_preflight':time.monotonic()-start})
    return tab,base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--stage-dir',required=True)
    a=p.parse_args();analyze(Path(a.plan),Path(a.stage_dir));print(json.dumps({'status':'analysis_verified','stage':a.stage_dir,'neural_inference':False}))
if __name__=='__main__':main()
