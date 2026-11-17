"""CPU invented-data control tests: continuation, exposure, locks and recovery."""
from pathlib import Path
import contextlib,copy,hashlib,importlib.util,io,json,os,sys,tempfile,unittest
from unittest import mock
import numpy as np
import torch
from src import b1_real_fedavg_runner_v1 as fed,b1_production_data_v1 as data,b1_preparation_io_v1 as fs
from src.b1_device_adapter_v1 import B1DeviceAdapter
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_fedavg_orchestration_v1 import RoundSnapshot
from test_b1_production_runner_v1 import InventedData,invented_client
if os.environ.get('B1_CONTROL_SOURCE'):
 spec=importlib.util.spec_from_file_location('b1_control_under_test',os.environ['B1_CONTROL_SOURCE']);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
else:from src import b1_centralized_sgd_runner_v1 as control

def model(seed=42):return B1DeviceAdapter(target_scaling=TargetScaling((400.,25.),(100.,5.)),root_seed=seed,device='cpu')
def config(rounds=2):
 settings=copy.deepcopy(control.SETTINGS);settings.update(clients=3,clients_per_round=2,rounds=rounds,blocks=rounds,validation_every=1)
 return {'kind':'synthetic','settings':settings}
def references(d,seed=42,rounds=2):
 rows=[{'state_fingerprint':model(seed).snapshot(0).fingerprint,'selected_clients':[],'examples':0,'optimizer_steps':0,'local':[]}]
 for r in range(1,rounds+1):
  names=list(fed.select_clients(d.names,seed,r,2));local=[]
  for n in names:
   c=d.client(n);order,shuffle=fed.shuffled_positions(seed,r,c.ordinal,len(c));sizes=[min(128,len(c)-i) for i in range(0,len(c),128)]
   local.append({'node':n,'ordinal':c.ordinal,'examples':len(c),'optimizer_steps':len(sizes),'batch_sizes':sizes,'shuffle_seed':shuffle,'row_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()})
  rows.append({'selected_clients':names,'examples':sum(v['examples'] for v in local),'optimizer_steps':sum(v['optimizer_steps'] for v in local),'local':local})
 return rows

def run(d,path,**kw):
 with contextlib.redirect_stdout(io.StringIO()):return control.run_seed(d,path,config(),42,references(d),**kw)

class OptimizationTests(unittest.TestCase):
 def test_exact_B1_architecture_and_seed(self):
  m=model();self.assertEqual(m.snapshot(0).fingerprint,model().snapshot(0).fingerprint);self.assertFalse(m.core.residual);self.assertEqual(m.core.head.out_features,2);self.assertEqual(m.core.node_embedding.num_embeddings,979);self.assertEqual(m.core.recurrent.num_layers,2)
  self.assertEqual(m.seed,int(np.random.SeedSequence([42,0,0,0]).generate_state(1)[0]))
 def test_partial_batches_and_real_parameter_updates(self):
  for count in (17,128,129,257):
   m=model();c=invented_client(count=count);before=m.snapshot(0);r=control.central_session(m,c,42,1)
   self.assertEqual(r['optimizer_steps'],math_ceil(count/128));self.assertEqual(sum(r['batch_sizes']),count);self.assertNotEqual(before.fingerprint,m.snapshot(0).fingerprint)
 def test_exact_manual_continuing_SGD(self):
  a=model();b=model();c=invented_client(count=129)
  rec=control.central_session(a,c,42,1);order,_=fed.shuffled_positions(42,1,c.ordinal,len(c));opt=torch.optim.SGD(b.parameters(),lr=.01,momentum=0.,weight_decay=0.)
  b.train()
  for start in range(0,len(c),128):
   ids=order[start:start+128];opt.zero_grad(set_to_none=True);loss=b.loss(torch.tensor(c.x[ids].copy()),c.future[ids],c.observed[ids]);loss.backward();opt.step()
  self.assertEqual(a.snapshot(0).fingerprint,b.snapshot(0).fingerprint);self.assertEqual(rec['end_fingerprint'],b.snapshot(0).fingerprint)
 def test_client_boundary_carries_weights_and_fresh_optimizer(self):
  m=model();original=torch.optim.SGD;objects=[]
  def create(*a,**kw):o=original(*a,**kw);objects.append(o);return o
  with mock.patch('torch.optim.SGD',side_effect=create):
   a=control.central_session(m,invented_client('1'),42,1);b=control.central_session(m,invented_client('2'),42,1)
  self.assertEqual(a['end_fingerprint'],b['start_fingerprint']);self.assertIsNot(objects[0],objects[1]);self.assertIsNot(objects[0].state,objects[1].state);self.assertEqual(objects[0].param_groups[0]['params'],objects[1].param_groups[0]['params'])
 def test_embedding_rows_not_averaged_or_diluted(self):
  m=model();before=m.core.node_embedding.weight.detach().clone();control.central_session(m,invented_client('1'),42,1);after=m.core.node_embedding.weight.detach().clone()
  self.assertFalse(torch.equal(before[0],after[0]));self.assertTrue(torch.equal(before[1:],after[1:]));control.central_session(m,invented_client('2'),42,1);self.assertTrue(torch.equal(after[0],m.core.node_embedding.weight[0]))
 def test_invalid_masked_and_boolean_targets_before_optimizer(self):
  c=invented_client();m=model()
  for value in (np.ma.array(c.future,mask=False),c.future.astype(bool),np.full(c.future.shape,np.nan)):
   class Bad:pass
   bad=Bad();bad.future=value;bad.observed=c.observed
   with mock.patch('torch.optim.SGD',side_effect=AssertionError('optimizer forbidden')):
    with self.assertRaises(ValueError):control.central_session(m,bad,42,1)
 def test_deterministic_updates(self):
  a,b=model(),model();c=invented_client(count=129);ra=control.central_session(a,c,42,1);rb=control.central_session(b,c,42,1)
  self.assertEqual(ra['row_order_sha256'],rb['row_order_sha256']);self.assertEqual(a.snapshot(0).fingerprint,b.snapshot(0).fingerprint)
 def test_sampling_replay_all200_blocks(self):
  from src.b1_real_data_bridge_v1 import ORIGINAL_NAMES
  for seed in (42,43,44):
   for r in range(1,201):
    chosen=fed.select_clients(ORIGINAL_NAMES,seed,r,98);self.assertEqual(len(set(chosen)),98);self.assertEqual(chosen,fed.select_clients(ORIGINAL_NAMES,seed,r,98))

class RecoveryTests(unittest.TestCase):
 def test_resume_matches_uninterrupted(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=InventedData();root=Path(tmp);a=run(d,root/'clean');partial=run(d,root/'recover',stop_after_block=1);self.assertEqual(partial['status'],'verified_partial');b=run(d,root/'recover',resume=True)
   self.assertEqual(a['selection'],b['selection'])
   for r in (0,1,2):self.assertEqual(data.read_json(root/f'clean/rounds/{r:04d}/receipt.json')['state_fingerprint'],data.read_json(root/f'recover/rounds/{r:04d}/receipt.json')['state_fingerprint'])
   check=control.verify_run(d,root/'recover',{'settings':config()['settings'],'sources':{}},42,references(d));self.assertEqual(check['status'],'verified_complete')
 def test_no_reset_or_aggregation(self):
  from src import b1_fedavg_orchestration_v1 as agg
  with tempfile.TemporaryDirectory() as tmp,mock.patch.object(agg,'aggregate_fedavg',side_effect=AssertionError('No aggregation')):
   d=InventedData();root=Path(tmp)/'run';run(d,root)
   for index in (1,2):
    row=data.read_json(root/f'rounds/{index:04d}/receipt.json');previous=data.read_json(root/f'rounds/{index-1:04d}/receipt.json')['weights_fingerprint']
    for c in row['local']:self.assertEqual(c['start_fingerprint'],previous);previous=c['end_fingerprint']
    self.assertEqual(previous,row['weights_fingerprint'])
 def test_wrong_reference_initialization_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=InventedData();ref=references(d);ref[0]['state_fingerprint']='0'*64
   with self.assertRaisesRegex(ValueError,'Initial state'):control.run_seed(d,Path(tmp)/'run',config(),42,ref)
 def test_checkpoint_hash_failure_precedes_model_init(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();run(d,root,stop_after_block=0);p=root/'rounds/0000/checkpoint.pt';p.write_bytes(p.read_bytes()+b'bad')
   with mock.patch('src.b1_device_adapter_v1.B1DeviceAdapter',side_effect=AssertionError('no init')):
    with self.assertRaisesRegex(ValueError,'hash mismatch'):run(d,root,resume=True)
 def test_wrong_reference_batch_hash_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();run(d,root,stop_after_block=1);ref=references(d);ref[1]['local'][0]['row_order_sha256']='bad'
   with self.assertRaisesRegex(ValueError,'exposure differs'):control.run_seed(d,root,config(),42,ref,resume=True)
 def test_existing_run_no_implicit_restart(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();run(d,root,stop_after_block=0)
   with self.assertRaises(FileExistsError):run(d,root)
 def test_failed_block_preserved_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData()
   with mock.patch.object(control,'central_session',side_effect=RuntimeError('injected')):
    with self.assertRaisesRegex(RuntimeError,'injected'):run(d,root)
   self.assertTrue((root/'rounds/0000/receipt.json').exists());self.assertFalse((root/'rounds/0001').exists());self.assertEqual(run(d,root,resume=True)['status'],'training_complete')
 def test_validation_no_optimizer_and_tie_rule(self):
  d=InventedData();m=model();before=m.snapshot(0).fingerprint
  with mock.patch('torch.optim.SGD',side_effect=AssertionError('No optimizer')):v=fed.evaluate_validation(m,d,0)
  self.assertEqual(m.snapshot(0).fingerprint,before);self.assertEqual(v['windows'],12)
  self.assertEqual(fed.earliest_minimum([{'round':0,'joint_standardized_RMSE':1.},{'round':10,'joint_standardized_RMSE':1.}])['round'],0)
 def test_stopping_does_not_publish_complete(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();self.assertEqual(run(d,root,stop_after_block=1)['status'],'verified_partial');self.assertFalse((root/'completed.json').exists());self.assertFalse((root/'selected').exists())
 def test_completion_tamper_detected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();run(d,root);p=root/'completed.json';v=data.read_json(p);v['selection']['round']=0;p.write_text(json.dumps(v))
   with self.assertRaisesRegex(ValueError,'receipt hashes'):control.verify_run(d,root,{'settings':config()['settings'],'sources':{}},42,references(d))
 def test_config_changes_rejected(self):
  self.assertEqual(control.SETTINGS['batch_size'],128);self.assertEqual(control.SETTINGS['learning_rate'],.01);self.assertEqual(control.SETTINGS['outputs'],2);self.assertEqual(control.SETTINGS['history'],36);self.assertFalse(control.SETTINGS['residual'])
  with self.assertRaises(ValueError):control.validate_plan({'schema':'bad'})
 def test_all_protocol_mutations_rejected(self):
  plan={'schema':'b1.centralized.control.config.v1','settings':copy.deepcopy(control.SETTINGS),'sources':{n:{} for n in control.SOURCES},'dataset':'output/experiments/b1_no_ff/preparation/train_val_v1/dataset','dataset_manifest_sha256':data.DATA_SHA,'scaler_lock':'output/march_window_audit/b1_scaler_approval_and_lock_v1/locked_scaler/lock.json','scaler_lock_sha256':data.LOCK_SHA,'reference':{'selection_lock':{'path':control.LOCK,'sha256':control.LOCK_SHA},'plan':{'path':control.BASE_PLAN},'seeds':{str(s):{} for s in (42,43,44)}},'output':control.OUTPUT,'March_access':False,'historical_preflight':'FAILED','semantic_reproduction':'UNKNOWN','budget_decision':'exact_replay_subject_to_explicit_operator_confirmation'}
  self.assertIs(control.validate_plan(plan),plan)
  for key,value in [('history',6),('outputs',1),('batch_size',512),('learning_rate',.02),('seeds',[42]),('rounds',201),('residual',True),('schedule','full_epochs')]:
   bad=copy.deepcopy(plan);bad['settings'][key]=value
   with self.assertRaisesRegex(ValueError,'protocol'):control.validate_plan(bad)
  for key,value in [('March_access',True),('historical_preflight','PASS'),('scaler_lock_sha256','bad'),('dataset','data/march'),('output','output/experiments/b1_no_ff/real_fedavg_training_v2')]:
   bad=copy.deepcopy(plan);bad[key]=value
   with self.assertRaises(ValueError):control.validate_plan(bad)
 def test_March_client_rejected_before_targets(self):
  c=invented_client()
  with self.assertRaisesRegex(ValueError,'chronology'):data.ClientData(c.node,c.x,np.ma.array(c.future,mask=False),c.observed,np.array(['2022-03-01T08']*len(c),dtype='datetime64[ns]'))
 def test_source_mismatch_rejected(self):
  with mock.patch.object(control,'validate_plan',side_effect=lambda p:p),mock.patch.object(data,'read_json',return_value={'sources':{}}),mock.patch.object(control,'source_bindings',return_value={'changed':{}}):
   with self.assertRaisesRegex(ValueError,'source integrity'):control.load_config('synthetic-plan.json')
 def test_wrong_method_checkpoint_schema_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'run';d=InventedData();run(d,root,stop_after_block=0)
   manifest=data.read_json(root/'manifest.json');receipt=data.read_json(root/'rounds/0000/receipt.json')
   with (root/'rounds/0000/checkpoint.pt').open('rb') as f:p=torch.load(f,map_location='cpu',weights_only=True)
   p['schema']='b1.production.checkpoint.v1'
   with self.assertRaisesRegex(ValueError,'identity'):control.validate_payload(p,manifest,d,[receipt],references(d))
 def test_production_requires_verified_data_before_init(self):
  d=InventedData()
  with tempfile.TemporaryDirectory() as tmp,mock.patch('src.b1_device_adapter_v1.B1DeviceAdapter',side_effect=AssertionError('no init')):
   with self.assertRaisesRegex(ValueError,'prerequisites'):control.run_seed(d,Path(tmp)/'run',{'kind':'production','settings':control.SETTINGS},42,references(d))
 def test_missing_review_fails_closed(self):
  with self.assertRaisesRegex(ValueError,'independent control review'):control.require_review(None,{},Path('absent'),'cpu')

@unittest.skipUnless(os.environ.get('B1_SYNTHETIC_CUDA_SMOKE')=='1' and torch.cuda.is_available(),'CUDA control smoke requires explicit opt-in after implementation review; not executed')
class OptionalCUDATests(unittest.TestCase):
 def test_synthetic_CUDA_resume(self):
  with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
   root=Path(tmp);d=InventedData();ref=references(d)
   control.run_seed(d,root/'clean',config(),42,ref,'cuda:0')
   control.run_seed(d,root/'recover',config(),42,ref,'cuda:0',stop_after_block=1)
   result=control.run_seed(d,root/'recover',config(),42,ref,'cuda:0',resume=True)
   self.assertEqual(result['status'],'training_complete')
   for r in (0,1,2):self.assertEqual(data.read_json(root/f'clean/rounds/{r:04d}/receipt.json')['state_fingerprint'],data.read_json(root/f'recover/rounds/{r:04d}/receipt.json')['state_fingerprint'])

def math_ceil(v):return int(np.ceil(v))
if __name__=='__main__':unittest.main(verbosity=2)
