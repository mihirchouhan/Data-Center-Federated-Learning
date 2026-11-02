"""Locked final-test adapter for completed v3 centralized-SGD GRU controls.

Preflight and postflight are read-only and never run a model. Actual inference
requires an explicit invocation into a new directory. No training, retuning,
resume, overwrite, split switch, checkpoint override or seed ranking is exposed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
import torch

from . import centralized_io as cio
from . import federated_evaluate as fed
from .experiment_data import read_json, immutable_json, sha256
from .experiment_models import neural_predict
from .experiment_resources import Resources
from .federated_io import digest_json, environment
from .federated_rounds import DEFAULTS, configure_runtime, initialize_model, sample_clients, stream_seed
from .federated_train import GlobalModelState
from .io_utils import sync_directory

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
BLOCKS = (190, 190, 180)


def execution_sources(root=ROOT):
    names = [*cio.execution_sources(), 'src/federated_evaluate.py',
             'src/experiment_models.py', 'src/centralized_evaluate.py']
    return {name: sha256(Path(root)/name) for name in names}


def verify_header(directory, declaration):
    contract, meta = fed.verify_header(directory, declaration)
    # Full byte integrity, but no np.load of examples during preflight.
    for name, value in contract['arrays'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe frozen array name')
        fed._hash(Path(directory)/name, value['sha256'])
    return contract, meta


class TestCohort:
    """Read-only test capability. Training/validation arrays are never mapped."""
    def __init__(self, directory, declaration):
        self.directory = Path(directory)
        self.contract, self.meta = verify_header(directory, declaration)
        self.contract_sha256 = declaration['contract_sha256']
        def load(name):
            value = np.load(self.directory/(name+'.npy'), mmap_mode='r', allow_pickle=False)
            spec = self.contract['arrays'][name+'.npy']
            if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype']:
                raise ValueError('Frozen test schema differs: '+name)
            return value
        self.names = load('node_names')
        self.node_names = tuple(str(n) for n in self.names)
        self.coefficients = load('power_baseline_coefficients')
        self.baseline_coefficients = self.coefficients
        expected = np.array([self.meta['x_std'][0]/self.meta['y_std'][0],
                             (self.meta['x_mean'][0]-self.meta['y_mean'][0])/self.meta['y_std'][0]])
        if (list(self.names)!=self.meta['nodes'] or self.coefficients.shape!=(2,)
                or not np.allclose(self.coefficients,expected,rtol=0,atol=1e-12)):
            raise ValueError('Frozen node mapping/persistence coefficients differ')
        self.__arrays = tuple(load(k+'_test') for k in ('X','y','origin','node'))
        x,y,origins,nodes = self.__arrays
        if x.dtype!=np.float32 or y.dtype!=np.float32:
            raise ValueError('Test inputs/targets must remain float32')
        lookup={name:i for i,name in enumerate(self.node_names)}
        try:ids=np.array([lookup[str(n)] for n in nodes],dtype=np.int64)
        except KeyError as error:raise ValueError('Unknown test node') from error
        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.array_equal(x[:,:,13],np.broadcast_to(ids[:,None],(len(ids),6)))):
            raise ValueError('Nonfinite test values or inconsistent embedding IDs')
        start=np.datetime64(self.meta['split_boundaries_utc']['val_end_test_start'].replace('+00:00',''),'ns')
        if (origins<start).any():raise ValueError('Test origin precedes chronological test boundary')
        same=nodes[1:]==nodes[:-1];blocks=np.r_[0,np.flatnonzero(~same)+1]
        if (not np.array_equal(nodes[blocks],self.names)
                or (np.diff(origins)[same]<=np.timedelta64(0,'ns')).any()):
            raise ValueError('Test windows must retain frozen node/time order')
        self.identity=fed.identity_record(nodes,origins,declaration['expected_windows'])
        self.__nodes=nodes.copy();self.__origins=origins.copy()

    def split(self, split):
        if split!='test':raise PermissionError('Final evaluator exposes test only')
        return self.__arrays

    def check_identity(self,nodes,origins):
        value=fed.identity_record(nodes,origins,self.identity['count'])
        if (value!=self.identity or not np.array_equal(nodes,self.__nodes)
                or not np.array_equal(origins,self.__origins)):
            raise ValueError('Test window identities/order differ')
        return value

    def verify_unchanged(self):
        verify_header(self.directory,{'contract_sha256':self.contract_sha256,
                      'expected_windows':self.identity['count'],'expected_nodes':len(self.node_names)})
        self.check_identity(self.__arrays[3],self.__arrays[2])


@dataclass(frozen=True)
class Selection:
    entry: dict
    manifest: dict
    exported: dict
    record: dict


def verify_prior(root, declaration, directory, cohort):
    """Inspect earlier completed FedAvg outputs; never infer or rerun reporting."""
    prior=fed._inside(root,declaration['directory'])
    for name,key in [('completed.json','completed_sha256'),('plan.json','plan_sha256'),
                     ('selection_lock.json','selection_lock_sha256')]:
        fed._hash(prior/name,declaration[key])
    receipt=read_json(prior/'completed.json');oldplan=read_json(prior/'plan.json')
    lock=read_json(prior/'selection_lock.json');results=read_json(prior/'results.json')
    if (receipt['status']!='complete' or receipt['split']!='test' or oldplan['cohort']!=cohort
            or lock['cohort']!=cohort or lock['plan_sha256']!=declaration['plan_sha256']
            or lock['plan_digest']!=digest_json(oldplan)
            or receipt['selection_lock_sha256']!=declaration['selection_lock_sha256']):
        raise ValueError('Prior FedAvg test completion/plan/lock differs')
    for name, expected in receipt['artifacts'].items():fed._hash(fed._inside(prior,name),expected)
    selected=fed.preflight(root,oldplan,directory)  # metadata/weights only, no examples.
    if len(lock['models'])!=3 or [r['seed'] for r in results['fits']]!=list(SEEDS):
        raise ValueError('Prior FedAvg models/seeds differ')
    identity=read_json(prior/'test_window_identity.json')
    if identity!=declaration['window_identity'] or results['window_identity']!=identity:
        raise ValueError('Prior FedAvg test window identity differs')
    contract=read_json(Path(directory)/'contract.json')
    expected_arrays={k:contract['arrays'][k+'_test.npy']['sha256'] for k in ('X','y','node','origin')}
    for index,s in enumerate(selected):
        if lock['models'][index]['model_sha256']!=s.entry['model_sha256']:
            raise ValueError('Prior lock model differs from authoritative run')
        row=results['fits'][index];score=read_json(prior/f"seed{s.entry['seed']}/comparison.json")
        evaluation=read_json(prior/f"seed{s.entry['seed']}/evaluation.json")
        if (row['model_sha256']!=s.entry['model_sha256'] or row['selected_round']!=s.entry['selected_round']
                or evaluation['array_hashes']!=expected_arrays
                or evaluation['frozen_contract_sha256']!=cohort['contract_sha256']
                or any(row.get(k)!=v for k,v in score.items())):
            raise ValueError('Prior model/score/frozen-array binding differs')
    if fed.summarize(results['fits'])!=results:
        raise ValueError('Prior aggregate differs from its individual stored scores')
    return selected,results


def verify_selection(root,entry,cohort,meta,sources,proof,paired):
    run=fed._inside(root,entry['run_dir'])
    if (not entry['run_dir'].startswith('output/experiments/v3_centralized_sgd/main/')
            or not entry['config_file'].startswith('configs/experiments/v3_centralized_sgd/')):
        raise ValueError('Selected control must use the v3 main/config namespaces')
    for name,key in [('manifest.json','manifest_sha256'),('configuration.json','configuration_sha256'),
                     ('completed.json','completed_sha256'),('final/receipt.json','final_receipt_sha256'),
                     ('final/model.pt','model_sha256')]:fed._hash(run/name,entry[key])
    fed._hash(fed._inside(root,entry['config_file']),entry['config_file_sha256'])
    manifest=read_json(run/'manifest.json');config=manifest['config']
    if (manifest['algorithm']!='Centralized-SGD-GRU schedule-replay' or config['version']!=3
            or config['kind']!='main' or config['rounds']!=200 or config['seed']!=entry['seed']
            or config!=read_json(run/'configuration.json')
            or config!=read_json(fed._inside(root,entry['config_file']))
            or manifest['config_digest']!=digest_json(config)
            or config['frozen_contract_sha256']!=cohort['contract_sha256']
            or manifest['frozen_contract_sha256']!=cohort['contract_sha256']):
        raise ValueError('Selected centralized configuration/provenance differs')
    for key in ('model','weather','local_epochs','batch_size','drop_last','optimizer','learning_rate',
                'momentum','weight_decay','hidden_size','num_layers','node_embedding_size',
                'validation_every','eval_batch_size','num_threads'):
        if type(config[key]) is not type(DEFAULTS[key]) or config[key]!=DEFAULTS[key]:
            raise ValueError('Centralized evaluation protocol differs: '+key)
    if cohort['expected_nodes']==979 and config['clients_per_round']!=98:
        raise ValueError('Original M100 participation required')
    if (manifest['source_hashes']!={k:sources[k] for k in cio.execution_sources()}
            or manifest['clients']!=paired.manifest['clients']
            or manifest['training_windows']!=paired.manifest['training_windows']
            or manifest['validation_windows']!=paired.manifest['validation_windows']
            or config['reference_run']!=paired.entry['run_dir']
            or config['reference_manifest_sha256']!=paired.entry['manifest_sha256']
            or manifest['reference']['manifest_sha256']!=paired.entry['manifest_sha256']):
        raise ValueError('Training source, partitions or paired reference differs')
    for name,expected in manifest['source_hashes'].items():fed._hash(fed._inside(run/'source',name),expected)
    for key,value in cio.software().items():
        if manifest['environment'][key]!=value:raise ValueError('Recorded software differs: '+key)
    if read_json(run/'reference_lock.json')!=manifest['reference']:
        raise ValueError('Training reference lock differs')
    previous=fed._inside(root,paired.entry['run_dir'])
    reference_hashes=[sha256(previous/'rounds'/f'{i:04d}'/'receipt.json') for i in range(201)]
    if manifest['reference']['round_receipt_sha256']!=reference_hashes:
        raise ValueError('Paired FedAvg receipt chain differs')
    payload,records=cio.load_blocks(run);exported,final=cio.verify_final(run)
    if payload is None or len(records)!=201 or payload['block']!=200:
        raise ValueError('Complete 200-block centralized run required')
    counts={p['id']:p['count'] for p in manifest['clients']};names=tuple(counts)
    if (tuple(meta['nodes'])!=names or len(names)!=cohort['expected_nodes']
            or [p['node_index'] for p in manifest['clients']]!=list(range(len(names)))
            or any(type(n) is not int or n<=0 for n in counts.values())
            or sum(counts.values())!=manifest['training_windows']):
        raise ValueError('Frozen centralized membership/mapping differs')
    evidence=next(p for p in proof['seeds'] if p['seed']==entry['seed'])
    total=steps=0
    keys=['client_id','shuffle_seed','training_windows','examples_processed','optimizer_steps','batch_sizes','row_order_sha256']
    for index,record in enumerate(records):
        expected=list(sample_clients(names,config['clients_per_round'],config['seed'],index)) if index else []
        if (record['selected_clients']!=expected or [p['client_id'] for p in record['clients']]!=expected
                or (record['validation'] is not None)!=(index%10==0)):
            raise ValueError('Recorded centralized participation/validation schedule differs')
        if index:
            plan=[{k:p[k] for k in keys} for p in record['clients']]
            if digest_json(plan)!=evidence['blocks'][index-1]['schedule_sha256']:
                raise ValueError('Recorded centralized batch schedule differs')
        for p in record['clients']:
            n=counts[p['client_id']];node_index=names.index(p['client_id'])
            sizes=[min(512,n-i) for i in range(0,n,512)]
            if (p['training_windows']!=n or p['examples_processed']!=n or p['optimizer_steps']!=len(sizes)
                    or p['batch_sizes']!=sizes or p['shuffle_seed']!=stream_seed(config['seed'],2,index,node_index)
                    or p['optimizer_state_entries']!=0):raise ValueError('Recorded centralized counts/shuffle differ')
            total+=n;steps+=len(sizes)
        if record['validation'] is not None:
            v=record['validation']
            if (v['samples']!=manifest['validation_windows'] or v['evaluated_nodes']!=len(names)
                    or v['split']!='val' or v['unit']!='W' or v['finite_predictions'] is not True
                    or not all(math.isfinite(v[k]) and v[k]>=0 for k in ('rmse_w','mae_w','persistence_rmse_w','persistence_mae_w'))):
                raise ValueError('Invalid centralized validation cohort/score')
    best=min((p for p in records if p['validation'] is not None),key=lambda p:(p['validation']['rmse_w'],p['block']))
    if (best['block']!=entry['selected_block'] or payload['best_block']!=best['block']
            or exported['selected_block']!=best['block'] or exported['validation_rmse_w']!=best['validation']['rmse_w']
            or payload['best_rmse_w']!=best['validation']['rmse_w']
            or payload['examples_processed']!=total or payload['optimizer_steps']!=steps
            or total!=evidence['examples'] or steps!=evidence['optimizer_steps']):
        raise ValueError('Centralized selection/budget differs from locked earliest validation minimum')
    selected=torch.load(run/'blocks'/f"{best['block']:04d}"/'state.pt',map_location='cpu',weights_only=True)
    for state in (selected['state'],payload['best_state']):
        if set(state)!=set(exported['state']) or any(not torch.equal(v,exported['state'][k]) for k,v in state.items()):
            raise ValueError('Selected export differs from validation checkpoint')
    factory=SimpleNamespace(node_names=names,baseline_coefficients=(meta['x_std'][0]/meta['y_std'][0],
                            (meta['x_mean'][0]-meta['y_mean'][0])/meta['y_std'][0]))
    model=initialize_model(factory,config);anchor=GlobalModelState(model,round_id=0)
    if (anchor.fingerprint!=manifest['reference']['initial_fingerprint']
            or anchor.fingerprint!=read_json(previous/'rounds/0000/receipt.json')['global_fingerprint']
            or anchor.fingerprint!=records[0]['global_fingerprint']):
        raise ValueError('Fresh initialization differs from paired reference')
    zero=torch.load(run/'blocks/0000/state.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(v,zero['state'][k]) for k,v in anchor.state_dict().items()):
        raise ValueError('Centralized round-zero weights differ from fresh initialization')
    anchor._validate(exported['state'],label='locked selected checkpoint')
    model.load_state_dict(exported['state'],strict=True)
    if GlobalModelState(model,round_id=best['block']).fingerprint!=best['global_fingerprint']:
        raise ValueError('Selected fingerprint differs')
    complete=read_json(run/'completed.json')
    if (complete['status']!='complete' or complete['completed_block']!=200 or complete['selected_block']!=best['block']
            or complete['validation_rmse_w']!=best['validation']['rmse_w'] or complete['examples_processed']!=total
            or complete['optimizer_steps']!=steps or complete['export_reload_exact'] is not True):
        raise ValueError('Centralized completion/selection differs')
    score=read_json(run/'final/validation/metrics.json')
    if score['rmse_w']!=best['validation']['rmse_w'] or score['mae_w']!=best['validation']['mae_w']:
        raise ValueError('Final validation scores differ from selection')
    record={**entry,'selected_model':entry['run_dir']+'/final/model.pt',
            'selected_validation_rmse_w':best['validation']['rmse_w'],
            'examples_processed':total,'optimizer_steps':steps,'verified_block_checkpoints':201}
    return Selection(entry,manifest,exported,record)


def preflight(root,plan,directory):
    root=Path(root)
    if (set(plan)!={'version','kind','cohort','execution_sources','runs','schedule_proof','prior_fedavg'}
            or plan['version']!=1 or plan['kind']!='frozen_centralized_sgd_final_test'
            or [v['seed'] for v in plan['runs']]!=list(SEEDS)
            or [v['selected_block'] for v in plan['runs']]!=list(BLOCKS)):
        raise ValueError('Expected fixed seeds 42/43/44 and blocks 190/190/180')
    sources=execution_sources(root)
    if sources!=plan['execution_sources']:raise ValueError('Evaluation execution-source hashes differ')
    _,meta=verify_header(directory,plan['cohort'])
    proof_path=fed._inside(root,plan['schedule_proof']['path'])
    fed._hash(proof_path,plan['schedule_proof']['sha256']);proof=read_json(proof_path)
    prior,prior_results=verify_prior(root,plan['prior_fedavg'],directory,plan['cohort'])
    selections=[verify_selection(root,p,plan['cohort'],meta,sources,proof,q) for p,q in zip(plan['runs'],prior)]
    normalized=[{k:v for k,v in s.manifest['config'].items() if k not in ('seed','reference_run','reference_manifest_sha256')} for s in selections]
    if any(v!=normalized[0] for v in normalized[1:]):raise ValueError('Centralized nonseed protocols differ')
    return selections,prior_results


def load_model(cohort,selection,device):
    if tuple(v['id'] for v in selection.manifest['clients'])!=cohort.node_names:
        raise ValueError('Frozen node embedding mapping differs')
    model=initialize_model(cohort,selection.manifest['config'])
    checked=GlobalModelState(model)._validate(selection.exported['state'],label='test model')
    model.load_state_dict(checked,strict=True)
    return model.to(device).eval()


def infer_test(cohort,selection,device):
    model=load_model(cohort,selection,device)
    before=GlobalModelState(model,round_id=selection.entry['selected_block']).fingerprint
    x,_,origins,nodes=cohort.split('test')
    prediction=neural_predict(model,x,{**selection.manifest['config'],'device':str(device)})
    if GlobalModelState(model,round_id=selection.entry['selected_block']).fingerprint!=before:
        raise ValueError('Inference changed selected parameters/buffers')
    result=fed.TestPredictions(prediction,nodes.copy(),origins.copy())
    fed.validate_predictions(cohort,result)
    return result


def paired_comparison(current,prior):
    if (current['window_identity']!=prior['window_identity']
            or current['samples_per_seed']!=prior['samples_per_seed']
            or any(current['persistence_'+k+'_w']!=prior['persistence_'+k+'_w'] for k in ('rmse','mae'))):
        raise ValueError('Centralized and historical FedAvg test cohorts/baselines differ')
    rows=[]
    for c,f in zip(current['fits'],prior['fits']):
        rows.append({'seed':c['seed'],'centralized_model_sha256':c['model_sha256'],
                     'fedavg_model_sha256':f['model_sha256'],
                     **{k+'_centralized_minus_fedavg_w':c[k+'_w']-f[k+'_w'] for k in ('rmse','mae')}})
    return {'split':'test','unit':'W','paired_seeds':rows,
            'centralized_mean_rmse_w':current['mean_rmse_w'],'fedavg_mean_rmse_w':prior['mean_rmse_w'],
            'centralized_mean_mae_w':current['mean_mae_w'],'fedavg_mean_mae_w':prior['mean_mae_w'],
            'difference_sign':'negative means lower centralized error',
            'prior_fedavg_recomputed':False,'test_based_model_selection':False,
            'test_status':'Previously observed development benchmark; not untouched confirmatory data'}


def make_lock(plan,selections,plan_sha256):
    return {'version':1,'algorithm':'Centralized-SGD-GRU','plan_sha256':plan_sha256,
            'plan_digest':digest_json(plan),'cohort':plan['cohort'],
            'models':[s.record for s in selections],'prior_fedavg':plan['prior_fedavg'],
            'selection':'earliest validation RMSE minimum, including persistence; locked before test arrays',
            'test_based_model_selection':False}


def evaluate_locked(root,plan,directory,output,*,device='cuda:0',plan_sha256=None):
    root=Path(root);output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('Evaluation output exists; preserve completed/partial artifacts')
    selections,prior=preflight(root,plan,directory)
    runtime={**selections[0].manifest['config'],'device':device}
    target=configure_runtime(runtime)
    output.mkdir(parents=True,exist_ok=False)
    sync_directory(output.parent)
    # Immutable ALL-model manifest precedes the first test example or prediction.
    immutable_json(output/'selection_lock.json',make_lock(plan,selections,plan_sha256))
    immutable_json(output/'plan.json',plan)
    immutable_json(output/'manifest.json',{'algorithm':'Centralized-SGD-GRU','split':'test','weather':'masked',
        'dtype':'float32','device':str(target),'eval_batch_size':runtime['eval_batch_size'],
        'environment':environment(target),'execution_sources':plan['execution_sources'],
        'selection_lock_sha256':sha256(output/'selection_lock.json'),'plan_sha256':plan_sha256,
        'data_directory':str(Path(directory).resolve()),'project_directory':str(root.resolve()),
        'contract_sha256':plan['cohort']['contract_sha256'],'scalers':'unchanged frozen training scalers',
        'test_status':'Previously observed benchmark; test results cannot select or retune models'})
    try:
        os.environ['MPLCONFIGDIR']=str(output/'.matplotlib')
        for name,expected in plan['execution_sources'].items():
            source=fed._inside(root,name);fed._hash(source,expected)
            dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:f.write(source.read_bytes());f.flush();os.fsync(f.fileno())
            dest.chmod(0o444);fed._hash(dest,expected);sync_directory(dest.parent)
        for directory_path in sorted((output/'source').rglob('*'),reverse=True):
            if directory_path.is_dir():sync_directory(directory_path)
        sync_directory(output/'source')
        sync_directory(output)
        with Resources() as resources:
            cohort=TestCohort(directory,plan['cohort'])
            if cohort.identity!=plan['prior_fedavg']['window_identity']:
                raise ValueError('Frozen test identity differs from prior benchmark')
            immutable_json(output/'test_window_identity.json',cohort.identity)
            rows=[]
            for s in selections:
                predictions=infer_test(cohort,s,target)
                scores=fed.evaluate_predictions(cohort,predictions,output/f"seed{s.entry['seed']}")
                rows.append({'seed':s.entry['seed'],'selected_model':s.record['selected_model'],
                             'model_sha256':s.entry['model_sha256'],'selected_block':s.entry['selected_block'],**scores})
            results=fed.summarize(rows)
            comparison=paired_comparison(results,prior)
            preflight(root,plan,directory);cohort.verify_unchanged()
            immutable_json(output/'results.json',results)
            immutable_json(output/'comparison_with_fedavg.json',comparison)
        immutable_json(output/'resources.json',resources.result)
        files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and '.matplotlib' not in p.parts}
        immutable_json(output/'completed.json',{'version':1,'status':'complete','split':'test',
                       'selection_lock_sha256':sha256(output/'selection_lock.json'),'artifacts':files})
        sync_directory(output)
        return results
    except BaseException as error:
        immutable_json(output/'failed.json',{'status':'failed','error':f'{type(error).__name__}: {error}',
            'recovery':'Preserve this directory; use a fresh output suffix with the same locked plan. No automatic resume.'})
        raise


def verify_results(root,plan,directory,output,*,plan_sha256=None):
    selections,prior=preflight(root,plan,directory)
    output=Path(output);receipt=read_json(output/'completed.json')
    if receipt['status']!='complete' or receipt['split']!='test':raise ValueError('Evaluation is incomplete')
    if (read_json(output/'plan.json')!=plan
            or read_json(output/'selection_lock.json')!=make_lock(plan,selections,plan_sha256)
            or receipt['selection_lock_sha256']!=sha256(output/'selection_lock.json')):
        raise ValueError('Evaluation selection manifest differs')
    required={'selection_lock.json','plan.json','manifest.json','test_window_identity.json',
              'results.json','resources.json','comparison_with_fedavg.json'}
    required.update('source/'+n for n in plan['execution_sources'])
    required.update(f'seed{s}/{n}' for s in SEEDS for n in ('metrics.json','per_node.json',
                    'coverage.json','predictions.npz','forecast.png','evaluation.json',
                    'per_node_comparison.json','comparison.json'))
    if set(receipt['artifacts'])!=required:raise ValueError('Completed evaluation artifact inventory differs')
    for name,expected in receipt['artifacts'].items():fed._hash(fed._inside(output,name),expected)
    manifest=read_json(output/'manifest.json')
    if (manifest['algorithm']!='Centralized-SGD-GRU' or manifest['split']!='test'
            or manifest['weather']!='masked' or manifest['dtype']!='float32'
            or manifest['execution_sources']!=plan['execution_sources']
            or manifest['selection_lock_sha256']!=sha256(output/'selection_lock.json')
            or manifest['plan_sha256']!=plan_sha256):raise ValueError('Evaluation manifest differs')
    result=read_json(output/'results.json')
    if fed.summarize(result['fits'])!=result:raise ValueError('Test summary differs from individual records')
    for s,row in zip(selections,result['fits']):
        scores=read_json(output/f"seed{s.entry['seed']}/comparison.json")
        if (row['model_sha256']!=s.entry['model_sha256'] or row['selected_block']!=s.entry['selected_block']
                or any(row.get(k)!=v for k,v in scores.items())):raise ValueError('Test model/result binding differs')
        evaluation=read_json(output/f"seed{s.entry['seed']}/evaluation.json")
        contract=read_json(Path(directory)/'contract.json')
        if (evaluation['frozen_contract_sha256']!=plan['cohort']['contract_sha256']
                or evaluation['array_hashes']!={k:contract['arrays'][k+'_test.npy']['sha256'] for k in ('X','y','node','origin')}):
            raise ValueError('Test evaluator array binding differs')
    if (result['window_identity']!=plan['prior_fedavg']['window_identity']
            or read_json(output/'comparison_with_fedavg.json')!=paired_comparison(result,prior)):
        raise ValueError('Historical comparison differs')
    return {'status':'verified_complete','test_evaluation_run':False,'model_inference_executed':False,
            'artifact_hashes_verified':len(receipt['artifacts']),'seeds':list(SEEDS)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--verify-only',action='store_true',help='Preflight, or postflight with output-dir; no inference or writes')
    args=parser.parse_args(argv)
    try:
        path=Path(args.plan).resolve()
        if not path.is_relative_to(ROOT/'configs/experiments/v3_centralized_sgd'):
            raise ValueError('Plan must be under configs/experiments/v3_centralized_sgd/')
        plan=read_json(path)
        if plan['cohort']!={'contract_sha256':DEFAULTS['frozen_contract_sha256'],'expected_windows':75566,'expected_nodes':979}:
            raise ValueError('CLI requires original M100 75,566-window / 979-node cohort')
        output=None
        if args.output_dir:
            output=Path(args.output_dir).resolve();allowed=ROOT/'output/experiments/v3_centralized_sgd/test_evaluation'
            if output==allowed or not output.is_relative_to(allowed):raise ValueError('Output must be a distinct v3 test_evaluation directory')
        if args.verify_only:
            if output is not None:result=verify_results(ROOT,plan,args.data_dir,output,plan_sha256=sha256(path))
            else:
                selections,_=preflight(ROOT,plan,args.data_dir)
                result={'status':'verified_only','test_evaluation_run':False,'model_inference_executed':False,
                        'models':[s.record for s in selections]}
        else:
            if output is None:raise ValueError('--output-dir is required for actual test inference')
            if args.device!='cpu' and not args.device.startswith('cuda:'):raise ValueError('Explicit CPU/CUDA device required')
            result=evaluate_locked(ROOT,plan,args.data_dir,output,device=args.device,plan_sha256=sha256(path))
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,StopIteration) as error:
        parser.exit(1,f'CENTRALIZED EVALUATION ERROR: {error}\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
