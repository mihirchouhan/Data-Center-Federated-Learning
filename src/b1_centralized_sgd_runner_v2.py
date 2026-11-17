"""B1 exact-replay control with a fixed three-seed, explicitly partial lifecycle.

Reuses the immutable v1 optimization, reference, checkpoint and metric code.
No March API. Describe is the default; no approval receipt is ever generated.
"""
from pathlib import Path
import argparse,fcntl,json,os,signal
from src import b1_centralized_sgd_runner_v1 as core
from src import b1_production_data_v1 as data_module
from src import b1_preparation_io_v1 as io
from src.b1_production_io_v1 import digest,source_bindings

ROOT=core.ROOT
SEEDS=(42,43,44)
SETTINGS=core.SETTINGS
SOURCES=[*core.SOURCES,'src/b1_centralized_sgd_runner_v2.py']
OUTPUT='output/experiments/b1_no_ff/centralized_sgd_control_v2/runs'
BASE_PLAN='configs/experiments/b1_no_ff/centralized_sgd_control_v1.json'
BASE_SHA='1b2a212774620767cf0ad537e0f872db714c40b309fb7d4f09434c5c34e07d32'

def validate_plan(plan):
    expected={'schema','settings','sources','dataset','dataset_manifest_sha256',
              'scaler_lock','scaler_lock_sha256','reference','output','March_access',
              'historical_preflight','semantic_reproduction','budget_decision','lifecycle','base_plan'}
    if set(plan)!=expected or plan['schema']!='b1.centralized.control.config.v2':
        raise ValueError('Version2 configuration schema differs')
    if io.canonical(plan['settings'])!=io.canonical(SETTINGS):raise ValueError('B1 protocol differs')
    if (plan['output']!=OUTPUT or set(plan['sources'])!=set(SOURCES) or
        plan['budget_decision']!='exact_replay_explicitly_confirmed' or
        plan['lifecycle']!='fixed_seedset_explicit_partial_v2'):
        raise ValueError('Version2 lifecycle/output/source declaration differs')
    if plan['base_plan']['path']!=BASE_PLAN or plan['base_plan']['sha256']!=BASE_SHA:
        raise ValueError('Original control configuration differs')
    # Validate unchanged scientific fields through the immutable v1 contract.
    old={k:v for k,v in plan.items() if k not in ('base_plan','lifecycle')}
    old.update(schema='b1.centralized.control.config.v1',output=core.OUTPUT,
               budget_decision='exact_replay_subject_to_explicit_operator_confirmation',
               sources={n:plan['sources'][n] for n in core.SOURCES})
    core.validate_plan(old)
    return plan

def load_config(path):
    plan=validate_plan(data_module.read_json(path))
    if source_bindings(ROOT,SOURCES)!=plan['sources']:raise ValueError('Execution source integrity failure')
    base=plan['base_plan']
    if io.binding(ROOT/base['path'])!={k:base[k] for k in ('bytes','sha256')}:
        raise ValueError('Immutable v1 plan bytes changed')
    old=core.load_config(ROOT/base['path'])
    for key in ('settings','dataset','dataset_manifest_sha256','scaler_lock','scaler_lock_sha256','reference'):
        if plan[key]!=old[key]:raise ValueError('Scientific/reference field differs: '+key)
    if {n:plan['sources'][n] for n in core.SOURCES}!=old['sources']:
        raise ValueError('Original execution source graph differs')
    return plan

def review_expected(plan,config_path,device):
    record=core.review_expected(plan,config_path,device)
    record.update(schema='b1.centralized.independent.review.v2',
                  lifecycle=plan['lifecycle'],seedset=list(SEEDS))
    return record

def require_review(path,plan,config_path,device):
    if not path or data_module.read_json(path)!=review_expected(plan,config_path,device):
        raise ValueError('Exact independent version2 review required before model initialization')
    return io.binding(path)

def verify_seed(data,path,plan,seed,reference,device,review):
    """No forward: all v1 checks plus the explicit current review/device binding."""
    checked=core.verify_run(data,path,plan,seed,reference)
    manifest=data_module.read_json(path/'manifest.json')
    if manifest['review']!=review or manifest['environment']['device']!=device:
        raise ValueError('Verified seed review/device differs from matrix')
    if device=='cuda:0' and (manifest['environment'].get('CUDA_VISIBLE_DEVICES')!=core.GPU or
        manifest['environment'].get('device_uuid','').removeprefix('GPU-')!=core.GPU.removeprefix('GPU-')):
        raise ValueError('Verified seed physical GPU differs')
    return checked

def _completion(output,matrix):
    return {'schema':'b1.centralized.matrix.complete.v2','status':'verified_complete',
            'matrix_sha256':digest(matrix),
            'seeds':{str(s):io.binding(output/f'seed{s}/completed.json') for s in SEEDS}}

def inspect_matrix(data,output,plan,references,device,review,matrix):
    """Verify every existing seed before any continuation; never modify artifacts."""
    allowed={'matrix.json','matrix.lock','matrix_completed.json',*(f'seed{s}' for s in SEEDS)}
    if set(p.name for p in output.iterdir())-allowed:raise ValueError('Unexpected matrix artifacts')
    if data_module.read_json(output/'matrix.json')!=matrix:raise ValueError('Matrix resume identity differs')
    for p in output.iterdir():io.safe_path(p)
    existing=[s for s in SEEDS if os.path.lexists(output/f'seed{s}')]
    if existing!=list(SEEDS)[:len(existing)]:raise ValueError('Missing predecessor seed')
    checked={}
    for seed in existing:
        path=output/f'seed{seed}'
        if not path.is_dir():raise ValueError('Invalid seed directory')
        checked[seed]=verify_seed(data,path,plan,seed,references[seed],device,review)
        if seed!=existing[-1] and checked[seed]['status']!='verified_complete':
            raise ValueError('Incomplete predecessor before later seed')
    if os.path.lexists(output/'matrix_completed.json'):
        if any(checked.get(s,{}).get('status')!='verified_complete' for s in SEEDS):
            raise ValueError('Completed matrix missing or incomplete seeds')
        if data_module.read_json(output/'matrix_completed.json')!=_completion(output,matrix):
            raise ValueError('Matrix completion differs')
    return checked

def execute_matrix(data,output,plan,references,device,review,plan_binding,*,requested=SEEDS,resume=False,stop_after_block=None,verify_only=False):
    """A seed invocation is a scope limit, never a change to experiment membership.

    Complete seeds are content-verified and reused WITHOUT run_seed/model init.
    Only the final existing incomplete seed can resume. All predecessors must
    verify complete before a later seed may start. No production approval API.
    """
    kind='production' if getattr(data,'_verified_token',None) is data_module.TOKEN else 'synthetic' if getattr(data,'scope',None)=='synthetic' else None
    if kind is None:raise ValueError('Verified training data or invented fixture required')
    if kind=='production':validate_plan(plan)
    requested=tuple(requested)
    if requested not in ((42,),(43,),(44,),SEEDS):raise ValueError('Explicit one seed or full matrix required')
    if device not in ('cpu','cuda:0') or review is None:raise ValueError('Explicit reviewed device required')
    output=io.safe_path(output)
    if kind=='production' and output!=ROOT/OUTPUT:raise ValueError('Fixed owned output required')
    matrix={'schema':'b1.centralized.matrix.v2','plan':plan_binding,'review':review,
            'device':device,'seeds':list(SEEDS),'data':data.binding}
    if not resume and requested[0]!=42:raise ValueError('Missing predecessor: begin with seed42')
    if verify_only and (not resume or stop_after_block is not None):raise ValueError('Verify cannot create or advance execution')
    if resume:
        if data_module.read_json(output/'matrix.json')!=matrix:raise ValueError('Matrix resume identity differs')
        if output.stat().st_uid!=os.getuid():raise ValueError('Foreign matrix owner')
        fd=os.open(io.safe_path(output/'matrix.lock'),(os.O_RDONLY if verify_only else os.O_RDWR)|os.O_NOFOLLOW)
    else:
        if os.path.lexists(output):raise FileExistsError('Matrix occupied; explicit resume required')
        if not output.parent.is_dir() or output.parent.stat().st_uid!=os.getuid():raise ValueError('Owned output parent required')
        output.mkdir(mode=0o700)
        fd=os.open(output/'matrix.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,(fcntl.LOCK_SH if verify_only else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        if not resume:
            io.exclusive(output/'matrix.json',io.canonical(matrix));io.fsync_dir(output)
        checked=inspect_matrix(data,output,plan,references,device,review,matrix)
        completed={s for s,v in checked.items() if v['status']=='verified_complete'}
        if verify_only:
            if any(s not in checked for s in requested):raise ValueError('Requested seed not present')
        else:
            for seed in requested:
                if seed in completed:
                    print(json.dumps({'seed':seed,'status':'verified_complete','action':'reused_without_model_initialization'}),flush=True)
                    continue
                if any(s not in completed for s in SEEDS[:SEEDS.index(seed)]):raise ValueError('Missing or incomplete predecessor seed')
                path=output/f'seed{seed}'
                result=core.run_seed(data,path,{'kind':kind,'settings':plan['settings'],'plan_digest':digest(plan)},
                    seed,references[seed],device,resume=path.exists(),stop_after_block=stop_after_block,
                    sources=plan['sources'],review_binding=review,reference_binding=plan['reference']['seeds'][str(seed)])
                checked[seed]=verify_seed(data,path,plan,seed,references[seed],device,review)
                if result['status']=='verified_partial':
                    if checked[seed]['status']!='verified_partial':raise ValueError('Partial fit/completion status differs')
                    break
                if result['status']!='training_complete' or checked[seed]['status']!='verified_complete':
                    raise ValueError('Next seed blocked by completion verification')
                completed.add(seed)
            if completed==set(SEEDS):
                complete=_completion(output,matrix)
                if (output/'matrix_completed.json').exists():
                    if data_module.read_json(output/'matrix_completed.json')!=complete:raise ValueError('Matrix completion differs')
                else:io.exclusive(output/'matrix_completed.json',io.canonical(complete));io.fsync_dir(output)
        published=(output/'matrix_completed.json').is_file()
        status=('verified_complete' if published else 'verified_complete_seeds_pending_matrix_receipt') if completed==set(SEEDS) else 'verified_partial_matrix'
        return {'status':status,'matrix_completion_published':published,
                'expected_seedset':list(SEEDS),'requested_seedset':list(requested),
                'completed_seeds':sorted(completed),'pending_seeds':[s for s in SEEDS if s not in completed],
                'seeds':{str(s):v for s,v in checked.items()},'model_forward_or_scoring':False if verify_only else 'training_only_in_incomplete_requested_seeds'}
    finally:os.close(fd)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--stage',choices=('describe','preflight','fit','verify'),default='describe')
    g=p.add_mutually_exclusive_group();g.add_argument('--seed',type=int,choices=SEEDS);g.add_argument('--matrix',action='store_true')
    p.add_argument('--device',choices=('cpu','cuda:0'));p.add_argument('--review-receipt');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-block',type=int)
    a=p.parse_args(argv);plan=load_config(a.plan)
    if a.stage=='describe':print(json.dumps({'status':'implementation_only','seedset':list(SEEDS),'partial_seed_supported':True,'output':OUTPUT,'training':False,'review_required':True}));return
    if a.stage in ('fit','verify'):
        if a.device is None or (a.seed is None and not a.matrix):raise ValueError('Explicit device and seed/matrix required')
        review=require_review(a.review_receipt,plan,a.plan,a.device)
        if a.stage=='fit' and a.device=='cuda:0' and (os.environ.get('CUDA_VISIBLE_DEVICES')!=core.GPU or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
            raise ValueError('Exact GPU1/determinism environment required')
        if a.stage=='verify' and (a.resume or a.stop_after_block is not None):raise ValueError('Verify cannot advance execution')
        if a.stop_after_block is not None and not 0<=a.stop_after_block<=200:raise ValueError('Invalid block cap')
    data=data_module.PreparedData(ROOT/plan['dataset'],ROOT/plan['scaler_lock'])
    output=ROOT/plan['output'];data_module.install_access_guard(data,output)
    refs=core.reference_ledgers(plan,data)
    if a.stage=='preflight':
        print(json.dumps({'status':'data_and_schedule_verified','binding':data.binding,'budgets':{str(s):{'examples':sum(r['examples'] for r in rows),'steps':sum(r['optimizer_steps'] for r in rows)} for s,rows in refs.items()},'model_initialized':False,'March_access':False}));return
    if a.stage=='fit':
        def interrupted(signum,frame):raise InterruptedError('Signal '+str(signum))
        signal.signal(signal.SIGTERM,interrupted)
    result=execute_matrix(data,output,plan,refs,a.device,review,io.binding(a.plan),
        requested=SEEDS if a.matrix else (a.seed,),resume=True if a.stage=='verify' else a.resume,
        stop_after_block=a.stop_after_block,verify_only=a.stage=='verify')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except (Exception,KeyboardInterrupt) as error:
        import sys
        print('B1 CENTRALIZED V2 SAFETY STOP: '+type(error).__name__+': '+str(error),file=sys.stderr)
        raise SystemExit(130 if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 1)
