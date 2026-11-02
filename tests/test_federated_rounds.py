"""Stage 2 transaction, protocol, deterministic replay and capability tests."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_federated_data import fixture, clients
from src.experiment_data import read_json
from src.federated_data import ClientTrainDataset, load_coordinator_validation
from src.federated_io import (commit_round, digest_json, export_selected, load_commits, load_selected,
                              make_manifest, open_run, source_inventory)
from src.federated_rounds import (DEFAULTS, configure_runtime, initialize_model, predict_validation,
                                 run_from_paths, run_rounds, sample_clients, stream_seed, train_client,
                                 validate_config, validation_due, validation_metrics)
from src.federated_train import GlobalModelState


class RoundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.digest, self.arrays, self.meta = fixture(self.data, include_test=False)
        self.registry = clients(self.data, self.digest)
        self.validation = load_coordinator_validation(self.data, expected_contract_sha256=self.digest)
        self.config = {**DEFAULTS, 'kind': 'smoke', 'device': 'cpu', 'rounds': 3,
                       'clients_per_round': 2, 'validation_every': 1, 'minimum_windows': 4,
                       'minimum_days': 2, 'frozen_contract_sha256': self.digest}
        configure_runtime(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def run_fit(self, name='run', **kwargs):
        return run_rounds(self.registry, self.validation, self.config, self.root / name, **kwargs)

    def test_fresh_seeded_initialization_is_persistence_and_independent_of_global_rng(self):
        before = torch.random.get_rng_state().clone()
        first = initialize_model(self.registry, self.config)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.rand(27); np.random.rand(43)
        second = initialize_model(self.registry, self.config)
        for name, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[name]))
        other = initialize_model(self.registry, {**self.config, 'seed': 43})
        self.assertFalse(torch.equal(first.node_embedding.weight, other.node_embedding.weight))
        self.assertTrue(torch.equal(first.head.weight, torch.zeros_like(first.head.weight)))
        prediction = predict_validation(first, self.validation, self.config, torch.device('cpu'))
        x = self.arrays['X_val']
        np.testing.assert_array_equal(prediction, x[:, -1, 0] * 2 + .5)
        self.assertEqual(len(list(first.recurrent.all_weights)), 2)

    def test_main_protocol_and_pilot_limits_fail_closed(self):
        self.assertEqual(validate_config(DEFAULTS), DEFAULTS)
        for key, value in [('weather', 'enabled'), ('optimizer', 'adam'), ('learning_rate', .001),
                           ('momentum', .9), ('weight_decay', .1), ('drop_last', True), ('local_epochs', 2),
                           ('batch_size', 128), ('rounds', 201), ('clients_per_round', 97), ('validation_every', 1),
                           ('minimum_windows', 1), ('seed', True), ('unknown', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**DEFAULTS, key: value})
        for field in ('rounds', 'clients_per_round'):
            with self.assertRaises(ValueError):validate_config({**self.config, field: 4})
        with self.assertRaisesRegex(ValueError, 'contract differs'):
            run_rounds(self.registry, self.validation, {**DEFAULTS, 'device': 'cpu',
                       'frozen_contract_sha256': DEFAULTS['frozen_contract_sha256']}, self.root/'invalid')

    def test_uniform_sampling_without_replacement_is_independent_of_shuffle_and_init(self):
        population = tuple(map(str, range(979)))
        expected = sample_clients(population, 98, 42, 1)
        self.assertEqual(len(expected), 98); self.assertEqual(len(set(expected)), 98)
        initialize_model(self.registry, self.config)
        np.random.default_rng(stream_seed(42, 2, 1, 0)).permutation(5000)
        self.assertEqual(sample_clients(population, 98, 42, 1), expected)
        self.assertNotEqual(sample_clients(population, 98, 43, 1), expected)
        self.assertNotEqual(sample_clients(population, 98, 42, 2), expected)
        self.assertEqual(len({stream_seed(42, d, 1, 0) for d in (0,1,2)}), 3)
        with self.assertRaises(ValueError):sample_clients(population, 980, 42, 1)

    def test_validation_schedule_has_initial_candidate_every_ten_and_final(self):
        self.assertEqual([r for r in range(201) if validation_due(r, DEFAULTS)], [0] + list(range(10,201,10)))
        self.assertTrue(validation_due(3, {**self.config,'validation_every':10}))
        self.assertFalse(validation_due(1, {**self.config,'validation_every':10}))

    def test_worker_receives_training_only_masked_rows_and_identical_global_start(self):
        original = train_client
        fingerprints=[]
        def checked(dataset, snapshot, template, config, device, **kwargs):
            self.assertIsInstance(dataset, ClientTrainDataset)
            self.assertFalse(hasattr(dataset, 'split'))
            batch=dataset.batch(np.arange(len(dataset)))
            self.assertTrue((batch.inputs[:,:,7:9] == 0).all())
            np.testing.assert_array_equal(batch.targets, self.arrays['y_train'][batch.row_indices])
            self.assertFalse((batch.targets == 777).any())
            fingerprints.append((kwargs['round_id'], snapshot.fingerprint))
            before = snapshot.state_dict()
            update, record = original(dataset, snapshot, template, config, device, **kwargs)
            for k,v in before.items():self.assertTrue(torch.equal(v,snapshot.state_dict()[k]))
            return update, record
        with mock.patch('src.federated_rounds.train_client', side_effect=checked):self.run_fit()
        for round_id in (1,2,3):
            self.assertEqual(len({f for r,f in fingerprints if r == round_id}),1)

    def test_new_optimizer_for_every_session_and_full_last_batch(self):
        dataset=self.registry.client('47')
        model=initialize_model(self.registry,self.config);snapshot=GlobalModelState(model)
        original=torch.optim.SGD;optimizers=[]
        def fresh(*args,**kwargs):
            optimizer=original(*args,**kwargs);optimizers.append(optimizer)
            self.assertEqual(len(optimizer.state),0)
            self.assertEqual(kwargs,{'lr':.01,'momentum':0.,'weight_decay':0.})
            return optimizer
        with mock.patch('src.federated_rounds.torch.optim.SGD',side_effect=fresh):
            first,record=train_client(dataset,snapshot,model,self.config,torch.device('cpu'),round_id=1)
            second,_=train_client(dataset,snapshot,model,self.config,torch.device('cpu'),round_id=1)
        self.assertIsNot(optimizers[0],optimizers[1])
        self.assertEqual(record['examples_processed'],12);self.assertEqual(record['optimizer_steps'],1)
        for key,value in first.state.items():self.assertTrue(torch.equal(value,second.state[key]))
        with self.assertRaises(ValueError):train_client(self.validation,snapshot,model,self.config,torch.device('cpu'),round_id=1)

    def test_validation_formula_matches_common_evaluator_physical_units(self):
        prediction=np.arange(12,dtype=np.float32)/2
        metrics=validation_metrics(prediction,self.validation)
        actual=(self.arrays['y_val'][:,0].astype(np.float64)*.2+.5)*1000
        predicted=(prediction.astype(np.float64)*.2+.5)*1000
        baseline=(self.arrays['X_val'][:,-1,0].astype(np.float64)*.4+.6)*1000
        self.assertEqual(metrics['rmse_w'],float(np.sqrt(np.mean((predicted-actual)**2))))
        self.assertEqual(metrics['persistence_rmse_w'],float(np.sqrt(np.mean((baseline-actual)**2))))
        self.assertEqual(metrics['samples'],12);self.assertEqual(metrics['evaluated_nodes'],3)
        with self.assertRaises(ValueError):validation_metrics(prediction[:-1],self.validation)
        with self.assertRaises(ValueError):validation_metrics(prediction*np.nan,self.validation)

    def test_513_window_client_processes_full_and_partial_batch(self):
        def larger(arrays,meta):
            count=513;ids=np.tile(np.arange(3),count);rng=np.random.default_rng(19)
            arrays['X_train']=rng.normal(size=(count*3,6,14)).astype(np.float32)
            arrays['X_train'][:,:,13]=ids[:,None]
            arrays['y_train']=rng.normal(size=(count*3,1)).astype(np.float32)
            arrays['node_train']=arrays['node_names'][ids]
            arrays['origin_train']=np.repeat(np.datetime64('2022-01-01T01:00','ns')+np.arange(count)*np.timedelta64(10,'m'),3)
        path=self.root/'larger';digest,_,_=fixture(path,mutate=larger,include_test=False)
        registry=clients(path,digest)
        model=initialize_model(registry,self.config);snapshot=GlobalModelState(model)
        _,record=train_client(registry.client('47'),snapshot,model,self.config,torch.device('cpu'),round_id=1)
        self.assertEqual(record['examples_processed'],513)
        self.assertEqual(record['optimizer_steps'],2)

    def test_uninterrupted_and_stop_resume_states_are_bit_identical(self):
        full=self.run_fit('full')
        stopped=self.run_fit('resumed',stop_after_round=1)
        self.assertEqual(stopped['completed_round'],1);self.assertEqual(stopped['status'],'stopped')
        self.assertFalse((self.root/'resumed/selected').exists())
        torch.rand(50);np.random.rand(50)
        resumed=self.run_fit('resumed',resume=True)
        self.assertEqual(resumed['started_from_round'],1)
        a,ar=load_commits(self.root/'full');b,br=load_commits(self.root/'resumed')
        for key in ('round','best_round','best_rmse_w','optimizer_steps','examples_processed','manifest_digest'):
            self.assertEqual(a[key],b[key])
        for field in ('state','best_state'):
            for key in a[field]:self.assertTrue(torch.equal(a[field][key],b[field][key]))
        for x,y in zip(ar,br):
            self.assertEqual(x['global_fingerprint'],y['global_fingerprint'])
            self.assertEqual(x['validation'],y['validation'])
            self.assertEqual(x['selected_clients'],y['selected_clients'])
        self.assertTrue(full['export_reload_check']['exact_predictions_equal'])
        self.assertTrue(resumed['export_reload_check']['finite_parameters'])
        self.assertEqual(resumed['examples_processed'],72);self.assertEqual(resumed['optimizer_steps'],6)

    def test_exception_mid_round_aborts_all_updates_and_replays_same_participants(self):
        original=train_client;calls=[]
        def interrupted(dataset,*args,**kwargs):
            calls.append((kwargs['round_id'],dataset.client_id))
            if len(calls)==4:raise RuntimeError('simulated interruption')
            return original(dataset,*args,**kwargs)
        with mock.patch('src.federated_rounds.train_client',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):self.run_fit('interrupted')
        payload,receipts=load_commits(self.root/'interrupted')
        self.assertEqual(payload['round'],1);self.assertEqual(len(receipts),2)
        self.assertFalse((self.root/'interrupted/rounds/0002').exists())
        replay=[]
        def resumed(dataset,*args,**kwargs):
            replay.append((kwargs['round_id'],dataset.client_id))
            return original(dataset,*args,**kwargs)
        with mock.patch('src.federated_rounds.train_client',side_effect=resumed):self.run_fit('interrupted',resume=True)
        self.assertEqual(calls[2:4],replay[:2])
        self.run_fit('control')
        a,_=load_commits(self.root/'interrupted');b,_=load_commits(self.root/'control')
        for key in a['state']:self.assertTrue(torch.equal(a['state'][key],b['state'][key]))

    def test_invalid_update_is_fatal_with_no_participant_substitution(self):
        original=train_client
        def invalid(dataset,*args,**kwargs):
            update,record=original(dataset,*args,**kwargs)
            state=dict(update.state);state['head.bias']=torch.full_like(state['head.bias'],float('nan'))
            return replace(update,state=state),record
        with mock.patch('src.federated_rounds.train_client',side_effect=invalid) as worker:
            with self.assertRaisesRegex(ValueError,'nonfinite'):self.run_fit('invalid')
        self.assertEqual(worker.call_count,2)
        payload,receipts=load_commits(self.root/'invalid')
        self.assertEqual(payload['round'],0);self.assertEqual(len(receipts),1)

    def test_invalid_shape_dtype_and_fixed_buffer_abort_round(self):
        original=train_client
        for kind in ('shape','dtype','fixed'):
            def invalid(dataset,*args,**kwargs):
                update,record=original(dataset,*args,**kwargs)
                state=dict(update.state)
                if kind=='shape':state['head.bias']=torch.zeros(2)
                elif kind=='dtype':state['head.bias']=state['head.bias'].double()
                else:state['baseline_coefficients']=state['baseline_coefficients']+1
                return replace(update,state=state),record
            with self.subTest(kind=kind),mock.patch('src.federated_rounds.train_client',side_effect=invalid):
                with self.assertRaisesRegex(ValueError,'shape differs|dtype differs|fixed state changed'):self.run_fit('invalid-'+kind)
            payload,_=load_commits(self.root/('invalid-'+kind));self.assertEqual(payload['round'],0)

    def test_validation_failure_discards_completed_local_updates(self):
        original=predict_validation;calls=0
        def fail(model,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise RuntimeError('validation interrupted')
            return original(model,*args,**kwargs)
        with mock.patch('src.federated_rounds.predict_validation',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'validation interrupted'):self.run_fit('val-failure')
        payload,_=load_commits(self.root/'val-failure');self.assertEqual(payload['round'],0)
        self.run_fit('val-failure',resume=True)
        payload,_=load_commits(self.root/'val-failure');self.assertEqual(payload['round'],3)

    def test_wrong_returned_client_or_count_aborts_before_commit(self):
        original=train_client
        for field,value in (('client_id','intruder'),('num_examples',1)):
            def invalid(dataset,*args,**kwargs):
                update,record=original(dataset,*args,**kwargs)
                return replace(update,**{field:value}),record
            with self.subTest(field=field),mock.patch('src.federated_rounds.train_client',side_effect=invalid):
                with self.assertRaisesRegex(ValueError,'round aborted'):self.run_fit('bad-'+field)
            payload,_=load_commits(self.root/('bad-'+field));self.assertEqual(payload['round'],0)

    def test_atomic_commit_failure_keeps_previous_round_and_preserves_orphan(self):
        self.run_fit('atomic',stop_after_round=1)
        import src.federated_io as io
        original=io.os.rename
        def fail_round(source,destination):
            if Path(destination).name=='0002':raise OSError('simulated crash before commit rename')
            return original(source,destination)
        with mock.patch('src.federated_io.os.rename',side_effect=fail_round):
            with self.assertRaisesRegex(OSError,'simulated crash'):self.run_fit('atomic',resume=True)
        payload,_=load_commits(self.root/'atomic');self.assertEqual(payload['round'],1)
        orphans=list((self.root/'atomic/rounds').glob('.pending-0002-*'))
        self.assertEqual(len(orphans),1)
        self.run_fit('atomic',resume=True)
        self.assertTrue(orphans[0].is_dir())
        payload,_=load_commits(self.root/'atomic');self.assertEqual(payload['round'],3)

    def test_resume_rejects_source_config_checkpoint_or_receipt_tampering(self):
        for kind in ('source','config','checkpoint','receipt'):
            name='tamper-'+kind;self.run_fit(name,stop_after_round=1)
            with self.subTest(kind=kind):
                if kind=='source':
                    with mock.patch('src.federated_io.source_inventory',return_value={'changed':'0'*64}):
                        with self.assertRaisesRegex(ValueError,'manifest differs'):self.run_fit(name,resume=True)
                elif kind=='config':
                    with self.assertRaisesRegex(ValueError,'manifest differs'):
                        run_rounds(self.registry,self.validation,{**self.config,'seed':43},self.root/name,resume=True)
                else:
                    path=self.root/name/'rounds/0001'/('state.pt' if kind=='checkpoint' else 'receipt.json')
                    path.chmod(0o644)
                    if kind=='checkpoint':
                        with path.open('ab') as f:f.write(b'changed')
                    else:
                        record=read_json(path);record['previous_receipt_sha256']='0'*64;path.write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError,'hash mismatch|chain mismatch'):self.run_fit(name,resume=True)

    def test_initial_and_completed_runs_cannot_be_overwritten(self):
        self.run_fit()
        inventory=lambda:{str(p.relative_to(self.root/'run')):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (self.root/'run').rglob('*') if p.is_file() and p.name!='.run.lock' and p.parent.name!='events'}
        before=inventory()
        with self.assertRaises(FileExistsError):self.run_fit()
        self.run_fit(resume=True)
        self.assertEqual(before,inventory())
        payload,_=load_commits(self.root/'run')
        with self.assertRaises(FileExistsError):commit_round(self.root/'run',payload,{'round':3})

    def test_earliest_validation_tie_retains_initial_persistence(self):
        original=validation_metrics
        def constant(prediction,validation):return {**original(prediction,validation),'rmse_w':123.}
        with mock.patch('src.federated_rounds.validation_metrics',side_effect=constant):result=self.run_fit('ties')
        self.assertEqual(result['selected_round'],0)
        exported=load_selected(self.root/'ties')
        initial=initialize_model(self.registry,self.config)
        for key in exported['state']:self.assertTrue(torch.equal(exported['state'][key],initial.state_dict()[key]))

    def test_earliest_equal_minimum_after_round_zero_is_selected(self):
        original=validation_metrics;scores=iter([1000.,100.,100.,200.,100.])
        def controlled(prediction,validation):return {**original(prediction,validation),'rmse_w':next(scores)}
        with mock.patch('src.federated_rounds.validation_metrics',side_effect=controlled):result=self.run_fit('later-tie')
        self.assertEqual(result['selected_round'],1)

    def test_resume_rejects_snapshot_tampering(self):
        self.run_fit('snapshot',stop_after_round=1)
        path=self.root/'snapshot/source/src/model.py';path.chmod(0o644)
        with path.open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'source snapshot differs'):self.run_fit('snapshot',resume=True)

    def test_end_to_end_path_loader_never_opens_test_and_worker_never_opens_files(self):
        original=np.load;opened=[]
        def no_test(filename,*args,**kwargs):
            name=Path(filename).name;opened.append(name)
            if '_test.' in name:raise AssertionError('Test split accessed')
            return original(filename,*args,**kwargs)
        worker=train_client
        def no_io(dataset,*args,**kwargs):
            with mock.patch('src.federated_data.np.load',side_effect=AssertionError('Worker opened a dataset file')):
                return worker(dataset,*args,**kwargs)
        with mock.patch('src.federated_data.np.load',side_effect=no_test),mock.patch('src.federated_rounds.train_client',side_effect=no_io):
            result=run_from_paths(self.config,self.data,self.root/'path-run')
        self.assertEqual(result['completed_round'],3)
        self.assertFalse(any('_test.' in name for name in opened))
        self.assertTrue({'X_train.npy','X_val.npy'}.issubset(opened))

    def test_nonfinite_validation_aborts_without_commit(self):
        with mock.patch('src.federated_rounds.predict_validation',return_value=np.full(12,np.nan)):
            with self.assertRaisesRegex(ValueError,'entire cohort'):self.run_fit('nan')
        payload,records=load_commits(self.root/'nan');self.assertIsNone(payload);self.assertEqual(records,[])

    def test_actual_run_honors_sparse_validation_schedule(self):
        config={**self.config,'validation_every':10}
        run_rounds(self.registry,self.validation,config,self.root/'sparse')
        _,records=load_commits(self.root/'sparse')
        self.assertEqual([r['round'] for r in records if r['validation'] is not None],[0,3])


if __name__=='__main__':unittest.main()
