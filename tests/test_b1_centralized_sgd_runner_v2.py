"""Invented CPU lifecycle regressions; no real data, model or approval receipt."""
from pathlib import Path
import contextlib,copy,fcntl,hashlib,importlib.util,io,json,os,sys,tempfile,unittest
from unittest import mock
from test_b1_centralized_sgd_runner_v1 import config,references
from test_b1_production_runner_v1 import InventedData
from src import b1_centralized_sgd_runner_v1 as core
if os.environ.get('B1_CONTROL_V2_SOURCE'):
 spec=importlib.util.spec_from_file_location('src.b1_centralized_sgd_runner_v2',os.environ['B1_CONTROL_V2_SOURCE'])
 runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
else:from src import b1_centralized_sgd_runner_v2 as runner

def plan():
 return {'settings':config()['settings'],'sources':{},'reference':{'seeds':{str(s):{'synthetic':True} for s in runner.SEEDS}}}
def snapshot(path):
 return {str(p.relative_to(path)):(core.io.binding(p),p.stat().st_mtime_ns) for p in path.rglob('*') if p.is_file()}

class LifecycleTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='b1-v2-test-',dir='/tmp');self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.out=self.root/'runs';self.data=InventedData();self.plan=plan()
  self.refs={s:references(self.data,s) for s in runner.SEEDS};self.review={'synthetic_fixture':True};self.binding={'synthetic_config_fixture':True}
 def run_matrix(self,**kw):
  with contextlib.redirect_stdout(io.StringIO()):return runner.execute_matrix(self.data,self.out,self.plan,self.refs,'cpu',self.review,self.binding,**kw)
 def test_seed42_partial_then_full_resume_reuses_complete_seed(self):
  first=self.run_matrix(requested=(42,));self.assertEqual(first['status'],'verified_partial_matrix');self.assertEqual(first['completed_seeds'],[42]);self.assertFalse(first['matrix_completion_published'])
  self.assertEqual(core.data_module.read_json(self.out/'matrix.json')['seeds'],[42,43,44]);before=snapshot(self.out/'seed42')
  original=core.run_seed;called=[]
  def invoke(*args,**kw):called.append(args[3]);return original(*args,**kw)
  with mock.patch.object(core,'run_seed',side_effect=invoke):last=self.run_matrix(resume=True)
  self.assertEqual(called,[43,44]);self.assertEqual(last['status'],'verified_complete');self.assertEqual(before,snapshot(self.out/'seed42'))
 def test_separate_seed43_and_seed44_after_seed42(self):
  self.run_matrix(requested=(42,));middle=self.run_matrix(requested=(43,),resume=True)
  self.assertEqual(middle['completed_seeds'],[42,43]);self.assertFalse((self.out/'seed44').exists());self.assertFalse(middle['matrix_completion_published'])
  self.assertEqual(self.run_matrix(requested=(44,),resume=True)['status'],'verified_complete')
 def test_complete_matrix_reuses_all_seeds_without_run_or_model(self):
  self.run_matrix();before=snapshot(self.out)
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No completed fit restart')):
   self.assertEqual(self.run_matrix(resume=True)['status'],'verified_complete')
  self.assertEqual(before,snapshot(self.out))
 def test_partial_seed42_checkpoint_then_full_resume_matches_clean(self):
  first=self.run_matrix(requested=(42,),stop_after_block=1);self.assertEqual(first['completed_seeds'],[])
  self.assertEqual(first['seeds']['42']['completed_round'],1)
  self.assertEqual(self.run_matrix(resume=True)['status'],'verified_complete')
  clean=self.root/'clean'
  with contextlib.redirect_stdout(io.StringIO()):runner.execute_matrix(self.data,clean,self.plan,self.refs,'cpu',self.review,self.binding)
  for seed in runner.SEEDS:
   for block in (0,1,2):
    relative=f'seed{seed}/rounds/{block:04d}/receipt.json'
    self.assertEqual(core.data_module.read_json(self.out/relative)['state_fingerprint'],core.data_module.read_json(clean/relative)['state_fingerprint'])
 def test_missing_predecessor_fresh_rejected_without_creation(self):
  with self.assertRaisesRegex(ValueError,'predecessor'):self.run_matrix(requested=(43,))
  self.assertFalse(self.out.exists())
 def test_missing_predecessor_existing_rejected(self):
  self.run_matrix();(self.out/'seed42').rename(self.root/'held-seed42')
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   with self.assertRaisesRegex(ValueError,'predecessor'):self.run_matrix(resume=True)
 def test_incomplete_predecessor_rejects_later_request(self):
  self.run_matrix(requested=(42,),stop_after_block=1)
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   with self.assertRaisesRegex(ValueError,'predecessor'):self.run_matrix(requested=(43,),resume=True)
 def test_existing_later_seed_after_incomplete_predecessor_rejected(self):
  self.run_matrix(requested=(42,),stop_after_block=1);(self.out/'seed43').mkdir()
  with self.assertRaisesRegex(ValueError,'Incomplete predecessor'):self.run_matrix(resume=True)
 def test_invalid_seed_completion_blocks_reuse_and_new_fit(self):
  self.run_matrix(requested=(42,));p=self.out/'seed42/completed.json';v=core.data_module.read_json(p);v['round']=1;p.write_text(json.dumps(v))
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   with self.assertRaisesRegex(ValueError,'Completion'):self.run_matrix(resume=True)
 def test_tampered_checkpoint_blocks_completed_reuse(self):
  self.run_matrix(requested=(42,));p=self.out/'seed42/rounds/0001/checkpoint.pt';p.write_bytes(p.read_bytes()+b'bad')
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_matrix(resume=True)
 def test_invalid_matrix_completion_rejected(self):
  self.run_matrix();p=self.out/'matrix_completed.json';v=core.data_module.read_json(p);v['status']='invalid';p.write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'Matrix completion'):self.run_matrix(resume=True)
 def test_missing_matrix_receipt_requires_verified_publication(self):
  self.run_matrix();(self.out/'matrix_completed.json').rename(self.root/'held-matrix-completed.json')
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No completed fit restart')):
   before=self.run_matrix(resume=True,verify_only=True)
   self.assertEqual(before['status'],'verified_complete_seeds_pending_matrix_receipt');self.assertFalse(before['matrix_completion_published'])
   self.assertEqual(self.run_matrix(resume=True)['status'],'verified_complete')
 def test_missing_selected_model_rejects_completion(self):
  self.run_matrix(requested=(42,));(self.out/'seed42/selected/model.pt').rename(self.root/'held-selected.pt')
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   with self.assertRaises(FileNotFoundError):self.run_matrix(resume=True)
 def test_matrix_identity_membership_not_editable(self):
  self.run_matrix(requested=(42,));p=self.out/'matrix.json';v=core.data_module.read_json(p);v['seeds']=[42];p.write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'identity'):self.run_matrix(resume=True)
 def test_occupied_root_requires_explicit_resume(self):
  self.run_matrix(requested=(42,))
  with self.assertRaises(FileExistsError):self.run_matrix(requested=(42,))
 def test_verify_only_is_non_mutating_and_cannot_start_seeds(self):
  self.run_matrix(requested=(42,));before=snapshot(self.out)
  with mock.patch.object(core,'run_seed',side_effect=AssertionError('No fit')):
   self.assertEqual(self.run_matrix(requested=(42,),resume=True,verify_only=True)['status'],'verified_partial_matrix')
   with self.assertRaisesRegex(ValueError,'not present'):self.run_matrix(resume=True,verify_only=True)
  self.assertEqual(before,snapshot(self.out))
 def test_matrix_lock_blocks_second_writer(self):
  self.run_matrix(requested=(42,));fd=os.open(self.out/'matrix.lock',os.O_RDWR)
  try:
   fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.run_matrix(resume=True)
  finally:os.close(fd)
 def test_wrong_review_binding_rejected_before_new_seed(self):
  self.run_matrix(requested=(42,));self.review={'different_synthetic_fixture':True}
  with self.assertRaisesRegex(ValueError,'identity'):self.run_matrix(resume=True)
 def test_invalid_requested_scope_rejected(self):
  for requested in ((),(43,44),(44,43),(42,42),(41,)):
   with self.assertRaises(ValueError):self.run_matrix(requested=requested)

class BindingTests(unittest.TestCase):
 def fixture(self):
  p=core.data_module.read_json(runner.ROOT/runner.BASE_PLAN);p=copy.deepcopy(p)
  p.update(schema='b1.centralized.control.config.v2',output=runner.OUTPUT,budget_decision='exact_replay_explicitly_confirmed',lifecycle='fixed_seedset_explicit_partial_v2',base_plan={'path':runner.BASE_PLAN,**core.io.binding(runner.ROOT/runner.BASE_PLAN)})
  p['sources']['src/b1_centralized_sgd_runner_v2.py']={'bytes':0,'sha256':'0'*64};return p
 def test_v2_protocol_mutations_rejected(self):
  p=self.fixture();self.assertIs(runner.validate_plan(p),p)
  for field,value in [('lifecycle','old'),('output',core.OUTPUT),('March_access',True),('budget_decision','epochs')]:
   bad=copy.deepcopy(p);bad[field]=value
   with self.assertRaises(ValueError):runner.validate_plan(bad)
  bad=copy.deepcopy(p);bad['settings']['batch_size']=512
  with self.assertRaises(ValueError):runner.validate_plan(bad)
 def test_source_hash_mismatch_rejected(self):
  p=self.fixture()
  with mock.patch.object(core.data_module,'read_json',return_value=p),mock.patch.object(runner,'source_bindings',return_value={}):
   with self.assertRaisesRegex(ValueError,'source integrity'):runner.load_config('invented-plan')
 def test_missing_review_precedes_real_data_loader(self):
  with mock.patch.object(runner,'load_config',return_value=self.fixture()),mock.patch.object(core.data_module,'PreparedData',side_effect=AssertionError('Real data forbidden')):
   with self.assertRaisesRegex(ValueError,'version2 review'):runner.main(['--plan','invented-plan','--stage','fit','--device','cpu','--seed','42'])
 def test_v1_review_schema_is_incompatible(self):
  p=self.fixture();expected=runner.review_expected(p,runner.ROOT/runner.BASE_PLAN,'cpu');old=copy.deepcopy(expected);old['schema']='b1.centralized.independent.review.v1'
  with mock.patch.object(core.data_module,'read_json',return_value=old):
   with self.assertRaisesRegex(ValueError,'version2 review'):runner.require_review('invented',p,runner.ROOT/runner.BASE_PLAN,'cpu')
 def test_verify_checks_exact_review_and_device(self):
  with mock.patch.object(core,'verify_run',return_value={'status':'verified_complete'}),mock.patch.object(core.data_module,'read_json',return_value={'review':{'wrong':True},'environment':{'device':'cpu'}}):
   with self.assertRaisesRegex(ValueError,'review/device'):runner.verify_seed(None,Path('/tmp/invented'),{},42,[],'cpu',{'synthetic':True})

class SmokeSafetyTests(unittest.TestCase):
 def module(self):
  p=os.environ.get('B1_CUDA_SMOKE_SOURCE',str(runner.ROOT/'tools/b1_centralized_cuda_smoke_v2.py'))
  spec=importlib.util.spec_from_file_location('cuda_smoke_under_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def test_actual_cuda_init_failure_has_operation_and_traceback(self):
  m=self.module()
  with mock.patch.dict(os.environ,CUDA_VISIBLE_DEVICES=core.GPU,CUBLAS_WORKSPACE_CONFIG=':4096:8'),mock.patch.object(m.torch.cuda,'is_available',return_value=False),mock.patch.object(m.torch.cuda,'device_count',return_value=0),mock.patch.object(m.torch.cuda,'init',side_effect=RuntimeError('invented CUDA initialization failure')),mock.patch.object(m,'B1DeviceAdapter',side_effect=AssertionError('No model')):
   r=m.probe()
  self.assertEqual(r['status'],'FAIL');self.assertEqual(r['failing_operation'],'torch.cuda.init()');self.assertIn('RuntimeError: invented CUDA initialization failure',r['traceback']);self.assertIsNone(r['memory_available_bytes'])
 def test_wrong_device_environment_stops_before_cuda_initialization(self):
  m=self.module()
  with mock.patch.dict(os.environ,CUDA_VISIBLE_DEVICES='wrong',CUBLAS_WORKSPACE_CONFIG=':4096:8'),mock.patch.object(m.torch.cuda,'init',side_effect=AssertionError('No CUDA')):
   r=m.probe()
  self.assertEqual(r['status'],'FAIL');self.assertEqual(r['failing_operation'],'environment authorization/device validation')

if __name__=='__main__':unittest.main(verbosity=2)
