"""CPU-only synthetic tests. Never load the real M100 arrays or run a GPU pilot."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_federated_data import fixture, clients
from src import centralized_sgd as central
from src import centralized_io as cio
from src.experiment_data import read_json, sha256, weather_view
from src.federated_data import ClientTrainDataset, load_coordinator_validation
from src.federated_io import load_commits
from src.federated_rounds import (DEFAULTS as FED, configure_runtime, initialize_model,
                                  run_rounds, train_client)
from src.federated_train import GlobalModelState


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root/'data'
        self.digest,self.arrays,_ = fixture(self.data,include_test=False)
        self.clients = clients(self.data,self.digest)
        self.validation = load_coordinator_validation(self.data,expected_contract_sha256=self.digest)
        self.refconfig = {**FED,'kind':'smoke','device':'cpu','rounds':3,'clients_per_round':2,
                          'validation_every':10,'minimum_windows':4,'minimum_days':2,
                          'frozen_contract_sha256':self.digest}
        self.reference = self.root/'reference'
        # Real protected reference implementation, but tiny synthetic arrays/CPU only.
        run_rounds(self.clients,self.validation,self.refconfig,self.reference)
        self.config = {**self.refconfig,'version':3,'reference_run':'synthetic-reference',
                       'reference_manifest_sha256':sha256(self.reference/'manifest.json')}
        self.out = self.root/'control'

    def run_control(self, output=None, **kwargs):
        return central.run_control(self.clients,self.validation,self.config,output or self.out,
                                   self.reference,**kwargs)

    def plan(self, block=1):
        model = initialize_model(self.clients,self.config)
        records,_ = central.verify_reference(self.reference,self.clients,self.config,model)
        return central.plan_block(self.clients,self.config,records,block)

    def test_initialization_exactly_matches_original_reference_and_preserves_rng(self):
        state = torch.get_rng_state().clone()
        model = initialize_model(self.clients,self.config)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        records,lock = central.verify_reference(self.reference,self.clients,self.config,model)
        zero = torch.load(self.reference/'rounds/0000/state.pt',weights_only=True)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,zero['state'][k]))
        self.assertEqual(lock['initial_fingerprint'],records[0]['global_fingerprint'])

    def test_different_seed_cannot_match_reference(self):
        config = {**self.config,'seed':43}
        with self.assertRaisesRegex(ValueError,'seed'):
            central.verify_reference(self.reference,self.clients,config,initialize_model(self.clients,config))

    def test_modified_initial_weights_cannot_match_recorded_initialization(self):
        model=initialize_model(self.clients,self.config)
        with torch.no_grad():model.head.bias.add_(1)
        with self.assertRaisesRegex(ValueError,'initialization'):
            central.verify_reference(self.reference,self.clients,self.config,model)

    def test_optimizer_protocol_is_fixed(self):
        for key,value in [('learning_rate',.001),('optimizer','adam'),('momentum',.9),
                          ('weight_decay',.01),('drop_last',True),('local_epochs',2),
                          ('batch_size',128),('num_threads',2),('weather','enabled'),
                          ('seed',True),('rounds',4),('new_key',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                central.validate_config({**self.config,key:value})
        self.assertEqual(central.validate_config(self.config),self.config)

    def test_batch_replay_matches_original_train_client_row_by_row(self):
        plan = self.plan(); model = initialize_model(self.clients,self.config)
        anchor = GlobalModelState(model,round_id=0)
        original = ClientTrainDataset.batch
        def trace(storage):
            def wrapped(dataset,positions):
                batch = original(dataset,positions)
                storage.append((dataset.client_id,batch.row_indices.copy(),batch.inputs.copy(),batch.targets.copy()))
                return batch
            return wrapped
        reference_batches=[];central_batches=[]
        with mock.patch.object(ClientTrainDataset,'batch',trace(reference_batches)):
            for p in plan:
                train_client(self.clients.client(p['client_id']),anchor,model,self.refconfig,
                             torch.device('cpu'),round_id=1)
        with mock.patch.object(ClientTrainDataset,'batch',trace(central_batches)):
            central.train_block(model,self.clients,plan,self.config,torch.device('cpu'),anchor,block=1)
        self.assertEqual(len(reference_batches),len(central_batches))
        for a,b in zip(reference_batches,central_batches):
            self.assertEqual(a[0],b[0])
            for x,y in zip(a[1:],b[1:]):np.testing.assert_array_equal(x,y)

    def test_continuing_model_matches_manual_sequential_sgd_without_resets(self):
        model = initialize_model(self.clients,self.config); manual=copy.deepcopy(model)
        plan=self.plan(); anchor=GlobalModelState(model)
        optimizer=torch.optim.SGD(manual.parameters(),lr=.01,momentum=0.,weight_decay=0.)
        for p in plan:
            data=self.clients.client(p['client_id'])
            order=np.random.default_rng(p['shuffle_seed']).permutation(len(data))
            for first in range(0,len(order),512):
                batch=data.batch(order[first:first+512]);optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(manual(torch.from_numpy(batch.inputs)),torch.from_numpy(batch.targets))
                loss.backward();optimizer.step()
        records=central.train_block(model,self.clients,plan,self.config,torch.device('cpu'),anchor,block=1)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,manual.state_dict()[k]),k)
        self.assertNotEqual(records[0]['initial_fingerprint'],records[1]['initial_fingerprint'])
        self.assertTrue(torch.equal(model.baseline_coefficients,anchor.state_dict()['baseline_coefficients']))

    def test_zero_momentum_optimizer_is_stateless_and_settings_recorded(self):
        original=torch.optim.SGD; seen=[]
        def constructor(parameters,**options):
            seen.append(options);return original(parameters,**options)
        model=initialize_model(self.clients,self.config)
        with mock.patch('src.centralized_sgd.torch.optim.SGD',side_effect=constructor):
            records=central.train_block(model,self.clients,self.plan(),self.config,torch.device('cpu'),GlobalModelState(model),block=1)
        self.assertEqual(seen,[{'lr':.01,'momentum':0.,'weight_decay':0.}])
        self.assertTrue(all(r['optimizer_state_entries']==0 for r in records))

    def test_accounting_and_full_partial_last_batches(self):
        summary=self.run_control()
        payload,receipts=cio.load_blocks(self.out)
        self.assertEqual(summary['examples_processed'],72)
        self.assertEqual(summary['optimizer_steps'],6)
        self.assertEqual(payload['examples_processed'],72)
        for record in receipts[1:]:
            self.assertEqual(len(record['clients']),2)
            for e in record['clients']:
                self.assertEqual(e['batch_sizes'],[12]);self.assertEqual(e['examples_processed'],12)
        exported,receipt=cio.verify_final(self.out)
        self.assertTrue(receipt['export_reload_exact'])
        self.assertTrue(all(torch.isfinite(x).all() for x in exported['state'].values()))
        self.assertEqual(read_json(self.out/'final/validation/metrics.json')['samples'],12)

    def test_more_than_one_batch_keeps_the_remainder(self):
        # New synthetic training fixture with 513 rows per client; no real arrays.
        def expand(arrays,meta):
            ids=np.tile(np.arange(3),513);rng=np.random.default_rng(91)
            arrays['X_train']=rng.normal(size=(1539,6,14)).astype('float32')
            arrays['X_train'][:,:,13]=ids[:,None]
            arrays['y_train']=np.zeros((1539,1),dtype='float32')
            arrays['node_train']=arrays['node_names'][ids]
            arrays['origin_train']=np.repeat(np.datetime64('2022-01-01T01:00','ns')+np.arange(513)*np.timedelta64(10,'m'),3)
        path=self.root/'large';digest,_,_=fixture(path,mutate=expand,include_test=False)
        registry=clients(path,digest);model=initialize_model(registry,self.config)
        seed=central.stream_seed(self.config['seed'],2,1,1)
        item={'client_id':'47','shuffle_seed':seed,'training_windows':513,'examples_processed':513,
              'optimizer_steps':2,'batch_sizes':[512,1],
              'row_order_sha256':hashlib.sha256(registry.partitions[1].row_indices[np.random.default_rng(seed).permutation(513)].tobytes()).hexdigest()}
        original=ClientTrainDataset.batch;observed=[]
        def traced(dataset,positions):
            batch=original(dataset,positions);observed.append(batch.row_indices.copy());return batch
        anchor=GlobalModelState(model)
        with mock.patch.object(ClientTrainDataset,'batch',traced):
            train_client(registry.client('47'),anchor,model,self.refconfig,torch.device('cpu'),round_id=1)
            result=central.train_block(model,registry,[item],self.config,torch.device('cpu'),anchor,block=1)
        self.assertEqual(len(observed),4)
        np.testing.assert_array_equal(observed[0],observed[2])
        np.testing.assert_array_equal(observed[1],observed[3])
        self.assertEqual(result[0]['batch_sizes'],[512,1])
        self.assertEqual(result[0]['optimizer_steps'],2)

    def test_resume_is_bit_identical_to_uninterrupted_on_cpu(self):
        full=self.root/'full';self.run_control(full)
        stopped=self.run_control(stop_after_block=1)
        self.assertEqual(stopped['status'],'stopped')
        self.assertFalse((self.out/'completed.json').exists())
        torch.manual_seed(199);torch.rand(87);np.random.seed(198);np.random.rand(77)
        self.run_control(resume=True)
        a,ar=cio.load_blocks(full);b,br=cio.load_blocks(self.out)
        for key in ('block','examples_processed','optimizer_steps','best_block','best_rmse_w'):
            self.assertEqual(a[key],b[key])
        for name in ('state','best_state'):
            for k,v in a[name].items():self.assertTrue(torch.equal(v,b[name][k]),k)
        self.assertEqual([r['global_fingerprint'] for r in ar],[r['global_fingerprint'] for r in br])
        self.assertEqual([r['validation'] for r in ar],[r['validation'] for r in br])

    def test_exception_mid_block_replays_full_block_without_advancing(self):
        self.run_control(stop_after_block=1)
        before=sha256(self.out/'blocks/0001/state.pt');original=ClientTrainDataset.batch
        calls=0
        def fail(dataset,positions):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt('synthetic interruption')
            return original(dataset,positions)
        with mock.patch.object(ClientTrainDataset,'batch',fail),self.assertRaises(KeyboardInterrupt):
            self.run_control(resume=True)
        self.assertFalse((self.out/'blocks/0002').exists())
        self.assertEqual(before,sha256(self.out/'blocks/0001/state.pt'))
        self.run_control(resume=True)
        full=self.root/'full';self.run_control(full)
        a,_=cio.load_blocks(full);b,_=cio.load_blocks(self.out)
        for k,v in a['state'].items():self.assertTrue(torch.equal(v,b['state'][k]))

    def test_resume_in_a_fresh_python_process_matches_uninterrupted(self):
        full=self.root/'full';self.run_control(full);self.run_control(stop_after_block=1)
        args=self.root/'arguments.json'
        args.write_text(json.dumps({'config':self.config,'data':str(self.data),'output':str(self.out),
                                    'reference':str(self.reference),'digest':self.digest}))
        code='''import json,sys
from pathlib import Path
from src.centralized_sgd import run_control
from src.federated_data import load_training_clients,load_coordinator_validation
v=json.loads(Path(sys.argv[1]).read_text())
c=load_training_clients(v['data'],expected_contract_sha256=v['digest'],minimum_windows=4,minimum_days=2)
w=load_coordinator_validation(v['data'],expected_contract_sha256=v['digest'])
run_control(c,w,v['config'],v['output'],v['reference'],resume=True)
'''
        result=subprocess.run([sys.executable,'-B','-c',code,str(args)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        a,_=cio.load_blocks(full);b,_=cio.load_blocks(self.out)
        for key in ('state','best_state'):
            for k,v in a[key].items():self.assertTrue(torch.equal(v,b[key][k]),k)
        self.assertEqual(cio.verify_final(full)[1]['predictions_sha256'],cio.verify_final(self.out)[1]['predictions_sha256'])

    def test_interrupted_atomic_commit_retains_pending_and_resumes(self):
        self.run_control(stop_after_block=1)
        original=cio.os.rename
        def fail(source,target):
            if Path(target).name=='0002':raise KeyboardInterrupt('before commit rename')
            return original(source,target)
        with mock.patch('src.centralized_io.os.rename',side_effect=fail),self.assertRaises(KeyboardInterrupt):
            self.run_control(resume=True)
        pending=list((self.out/'blocks').glob('.pending-0002-*'));self.assertEqual(len(pending),1)
        self.run_control(resume=True)
        self.assertTrue(pending[0].is_dir())
        self.assertEqual(len(cio.load_blocks(self.out)[1]),4)

    def test_interrupted_final_export_resumes_without_overwriting_commits(self):
        original=central.os.rename
        def fail(source,target):
            if Path(target).name=='final':raise KeyboardInterrupt('final rename')
            return original(source,target)
        with mock.patch('src.centralized_sgd.os.rename',side_effect=fail),self.assertRaises(KeyboardInterrupt):
            self.run_control()
        before=sha256(self.out/'blocks/0003/state.pt')
        self.run_control(resume=True)
        self.assertEqual(before,sha256(self.out/'blocks/0003/state.pt'))
        self.assertTrue(read_json(self.out/'completed.json')['export_reload_exact'])

    def test_validation_schedule_and_earliest_minimum_selection(self):
        c={**self.config,'rounds':200}
        self.assertEqual([i for i in range(201) if central.due(i,c)],list(range(0,201,10)))
        self.assertEqual([i for i in range(4) if central.due(i,self.config)],[0,3])
        self.assertEqual(central.choose(5.,5.,10,0),(5.,0))
        self.assertEqual(central.choose(4.,5.,10,0),(4.,10))
        with self.assertRaises(ValueError):central.choose(float('nan'),5.,10,0)
        self.run_control()
        payload,records=cio.load_blocks(self.out)
        best=min((r for r in records if r['validation']),key=lambda r:(r['validation']['rmse_w'],r['block']))
        self.assertEqual(payload['best_block'],best['block'])

    def test_persistence_can_remain_selected(self):
        def persistence_targets(arrays,meta):
            for split in ('train','val'):
                arrays[f'y_{split}']=(arrays[f'X_{split}'][:,-1,0:1]*2+.5).astype(np.float32)
        self.data=self.root/'persistence-data'
        self.digest,self.arrays,_=fixture(self.data,mutate=persistence_targets,include_test=False)
        self.clients=clients(self.data,self.digest)
        self.validation=load_coordinator_validation(self.data,expected_contract_sha256=self.digest)
        self.refconfig={**self.refconfig,'frozen_contract_sha256':self.digest}
        self.reference=self.root/'persistence-reference'
        run_rounds(self.clients,self.validation,self.refconfig,self.reference)
        self.config={**self.config,'frozen_contract_sha256':self.digest,
                     'reference_manifest_sha256':sha256(self.reference/'manifest.json')}
        result=self.run_control()
        value,records=cio.load_blocks(self.out)
        self.assertEqual(value['best_block'],0)
        self.assertEqual(value['best_rmse_w'],0.)
        self.assertEqual(result['selected_block'],0)
        self.assertEqual(cio.verify_final(self.out)[0]['selected_block'],0)
        for r in records:
            if r['validation'] is not None:self.assertEqual(r['validation']['rmse_w'],0.)

    def test_no_test_access_and_training_cannot_obtain_validation(self):
        original=np.load
        def deny(filename,*args,**kwargs):
            self.assertNotIn('_test.',str(filename));return original(filename,*args,**kwargs)
        with mock.patch('src.federated_data.np.load',side_effect=deny):
            self.run_control()
        with self.assertRaises(PermissionError):self.validation.split('train')
        with self.assertRaises(PermissionError):self.validation.split('test')
        model=initialize_model(self.clients,self.config)
        with self.assertRaises(ValueError):
            central.train_block(model,self.validation,self.plan(),self.config,torch.device('cpu'),GlobalModelState(model),block=1)
        for item in self.plan():
            dataset=self.clients.client(item['client_id'])
            self.assertFalse(hasattr(dataset,'split'))
            self.assertFalse(np.any(dataset.batch(np.arange(len(dataset))).targets==777))

    def test_training_loader_opens_no_validation_arrays_and_ignores_validation_targets(self):
        from src.federated_data import load_training_clients
        original=np.load;opened=[]
        def training_only(filename,*args,**kwargs):
            name=Path(filename).name;opened.append(name)
            self.assertNotIn('_val.',name);self.assertNotIn('_test.',name)
            return original(filename,*args,**kwargs)
        with mock.patch('src.federated_data.np.load',side_effect=training_only):
            registry=load_training_clients(self.data,expected_contract_sha256=self.digest,minimum_windows=4,minimum_days=2)
            model=initialize_model(registry,self.config)
            central.train_block(model,registry,self.plan(),self.config,torch.device('cpu'),GlobalModelState(model),block=1)
        self.assertEqual(set(opened),{'node_names.npy','power_baseline_coefficients.npy','X_train.npy',
                                     'y_train.npy','origin_train.npy','node_train.npy'})

    def test_weather_scalers_rows_and_original_arrays_are_unchanged(self):
        before=cio.capture_inventory(self.data)
        self.run_control()
        self.assertEqual(cio.check_inventory(self.data,before)['failed'],0)
        p=self.clients.partitions[0];batch=self.clients.client(p.client_id).batch(np.arange(p.num_examples))
        np.testing.assert_array_equal(batch.inputs,weather_view(self.arrays['X_train'][p.row_indices],'masked'))
        np.testing.assert_array_equal(batch.targets,self.arrays['y_train'][p.row_indices])

    def test_nonfinite_prediction_aborts_before_block_commit(self):
        self.run_control(stop_after_block=0)
        original=central.train_block
        def invalid(model,*args,**kwargs):
            with torch.no_grad():model.head.weight.fill_(float('nan'))
            return original(model,*args,**kwargs)
        with mock.patch('src.centralized_sgd.train_block',side_effect=invalid),self.assertRaisesRegex(ValueError,'finite'):
            self.run_control(resume=True)
        self.assertFalse((self.out/'blocks/0001').exists())

    def test_nonfinite_validation_prediction_is_rejected(self):
        with mock.patch('src.centralized_sgd.predict_validation',return_value=np.full(12,np.nan)),self.assertRaises(ValueError):
            self.run_control()
        self.assertEqual(len(cio.load_blocks(self.out)[1]),0)

    def test_nonfinite_gradient_and_poststep_parameter_are_rejected(self):
        model=initialize_model(self.clients,self.config);anchor=GlobalModelState(model)
        hook=model.head.weight.register_hook(lambda x:torch.full_like(x,float('nan')))
        with self.assertRaisesRegex(ValueError,'gradient'):
            central.train_block(model,self.clients,self.plan(),self.config,torch.device('cpu'),anchor,block=1)
        hook.remove()
        original=torch.optim.SGD.step
        def broken(optimizer,*args,**kwargs):
            result=original(optimizer,*args,**kwargs)
            with torch.no_grad():optimizer.param_groups[0]['params'][0].fill_(float('inf'))
            return result
        with mock.patch('torch.optim.SGD.step',broken),self.assertRaisesRegex(ValueError,'parameter'):
            central.train_block(model,self.clients,self.plan(),self.config,torch.device('cpu'),anchor,block=1)

    def test_fixed_coefficients_cannot_change(self):
        model=initialize_model(self.clients,self.config);anchor=GlobalModelState(model)
        with torch.no_grad():model.baseline_coefficients.add_(1)
        with self.assertRaisesRegex(ValueError,'fixed'):
            central.train_block(model,self.clients,self.plan(),self.config,torch.device('cpu'),anchor,block=1)

    def test_bad_batch_accounting_is_rejected(self):
        model=initialize_model(self.clients,self.config);plan=self.plan();plan[0]['optimizer_steps']+=1
        with self.assertRaisesRegex(ValueError,'batches/counts'):
            central.train_block(model,self.clients,plan,self.config,torch.device('cpu'),GlobalModelState(model),block=1)

    def test_checkpoint_and_receipt_corruption_are_rejected(self):
        self.run_control(stop_after_block=1)
        path=self.out/'blocks/0001/state.pt';path.chmod(0o644)
        with path.open('ab') as f:f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity'):self.run_control(resume=True)

    def test_source_config_reference_and_environment_changes_reject_resume(self):
        self.run_control(stop_after_block=1)
        with mock.patch('src.centralized_io.execution_sources',return_value={'changed':'0'*64}),self.assertRaisesRegex(ValueError,'manifest'):
            self.run_control(resume=True)
        saved=self.config;self.config={**saved,'eval_batch_size':512}
        with self.assertRaises(ValueError):self.run_control(resume=True)
        self.config=saved
        with mock.patch('src.centralized_io.software',return_value={'python':'wrong'}),self.assertRaisesRegex(ValueError,'software'):
            self.run_control(resume=True)
        path=self.reference/'rounds/0001/receipt.json';path.chmod(0o644)
        value=read_json(path);value['selected_clients'].reverse();path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.run_control(resume=True)

    def test_existing_output_is_preserved_and_verify_only_never_predicts(self):
        self.run_control()
        before=cio.capture_inventory(self.out)
        with self.assertRaises(FileExistsError):self.run_control()
        with mock.patch('src.centralized_sgd.predict_validation',side_effect=AssertionError('inference forbidden')):
            result=self.run_control(verify_only=True)
        self.assertEqual(result['status'],'verified_complete')
        self.assertFalse(result['model_inference_executed'])
        self.assertEqual(cio.capture_inventory(self.out),before)

    def test_final_export_and_completion_corruption_are_rejected(self):
        self.run_control()
        path=self.out/'final/model.pt';path.chmod(0o644)
        with path.open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'artifact'):self.run_control(verify_only=True)

    def test_bad_shape_dtype_and_fixed_state_are_rejected(self):
        self.run_control(stop_after_block=1)
        payload,records=cio.load_blocks(self.out);model=initialize_model(self.clients,self.config)
        _,reference=load_commits(self.reference)
        for change in ('shape','dtype','finite','fixed','count'):
            value=copy.deepcopy(payload)
            if change=='shape':value['state']['head.weight']=torch.zeros(1,2)
            if change=='dtype':value['state']['head.weight']=value['state']['head.weight'].double()
            if change=='finite':value['state']['head.weight'].fill_(float('nan'))
            if change=='fixed':value['state']['baseline_coefficients'].add_(1)
            if change=='count':value['optimizer_steps']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):
                central.validate_payload(value,records,model,self.clients,self.config,reference,GlobalModelState(initialize_model(self.clients,self.config)))

    def test_completion_counter_corruption_and_missing_final_are_rejected(self):
        self.run_control()
        p=self.out/'completed.json';saved=p.read_bytes();p.chmod(0o644)
        value=json.loads(saved);value['optimizer_steps']+=1;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'accounting'):
            self.run_control(verify_only=True)
        p.write_bytes(saved)
        (self.out/'final').rename(self.out/'preserved-final')
        with self.assertRaisesRegex(ValueError,'Missing final'):
            self.run_control(verify_only=True)

    def test_invalid_cursor_and_wrong_capabilities_do_not_create_output(self):
        for cursor in (-1,4,True):
            with self.subTest(cursor=cursor),self.assertRaises(ValueError):
                self.run_control(stop_after_block=cursor)
        with self.assertRaises(ValueError):
            central.run_control(self.validation,self.clients,self.config,self.out,self.reference)
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()
