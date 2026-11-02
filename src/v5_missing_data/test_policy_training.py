"""Data/planning/state tests only. No forward pass, SGD step, or scoring allowed."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from src.experiment_data import read_json, sha256
from src.federated_train import GlobalModelState, aggregate_fedavg
from src.v5_missing_data import test_build_training_cohorts as prepared_fixture
from src.v5_missing_data.build_training_cohorts import execute
from src.v5_missing_data.forward_fill_audit import causal_forward_fill
from src.v5_missing_data.training_data import CohortContext, TrainingRegistry, SplitData, POLICIES
from src.v5_missing_data.sampling import round_plan, metadata, stream_seed, due, select, verify_budget_totals
from src.v5_missing_data import training_io as tio
from src.v5_missing_data.run_policy_experiments import initialize, validate_payload, output_path
from src.v5_missing_data.evaluate_policy_models import predict_and_score, make_lock, load_lock

GUARDS=[]


def setUpModule():
    for target in ('src.model.SequenceRegressor.forward','torch.optim.SGD.step',
                   'src.v5_missing_data.evaluate_policy_models.metric_pair'):
        patch=mock.patch(target,side_effect=AssertionError('Training/inference/scoring is forbidden in these tests'))
        GUARDS.append((patch,patch.start()))


def tearDownModule():
    for patch,guard in GUARDS:
        assert guard.call_count==0, 'A forbidden operation was attempted'
        patch.stop()


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.f=prepared_fixture.PreparationTests('test_full_synthetic_preparation_replays_reference_and_saves_common_ids')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.patch_raw()
        with mock.patch('src.v5_missing_data.build_training_cohorts.load_inputs',return_value=self.f.context):
            execute(self.f.root,self.f.plan_path,self.f.root/'audit',self.f.frozen,
                    self.f.root/'unused_raw',self.f.output)
        self.directory=self.f.output
        self.context=CohortContext(self.directory,sha256(self.directory/'completed.json'))


class DataTests(FixtureTests):
    def test_exact_split_end_and_prior_split_histories(self):
        for split in ('train','val','test'):
            data=SplitData(self.context,split)
            lower,upper=self.context.contract['boundaries'][split]
            origin=self.f.edges[upper-6].astype('datetime64[ns]')
            positions=np.flatnonzero((data.ids['node_index']==0)&(data.ids['origin']==origin))
            self.assertEqual(len(positions),1)
            data.batch(positions)
            if split!='train':
                first=np.flatnonzero((data.ids['node_index']==0)&(data.ids['origin']==self.f.edges[lower].astype('datetime64[ns]')))
                self.assertEqual(len(first),1)
                x,_,_=data.batch(first)
                self.assertEqual(x.shape,(1,36,14))
            path=self.directory/data.spec['paths']['ids'];path.chmod(0o644)
            ids=np.load(path,mmap_mode='r+',allow_pickle=False)
            ids['origin'][positions[0]]=self.f.edges[upper-5].astype('datetime64[ns]');ids.flush()
            with self.assertRaisesRegex(ValueError,'split boundaries'):data.batch(positions)
            ids._mmap.close()

    def test_training_registry_never_maps_validation_or_test_arrays(self):
        original=np.load;opened=[]
        def train_only(path,*args,**kwargs):
            name=str(path);opened.append(name)
            if '/val/' in name or '/test/' in name or '/bins/val_' in name or '/bins/test_' in name:
                raise AssertionError('Training opened an evaluation array')
            return original(path,*args,**kwargs)
        with mock.patch('numpy.load',side_effect=train_only):
            context=CohortContext(self.directory,self.context.receipt_sha256)
            registry=TrainingRegistry(context,'ff20')
            plan=round_plan(registry,42,1,clients_per_round=2)
            local=registry.local(plan[0]['node_index'],plan[0]['rows'])
        self.assertTrue(opened)
        self.assertFalse(hasattr(local,'context'))
        self.assertFalse(hasattr(local,'path'))
        with self.assertRaises(ValueError):local.inputs.setflags(write=True)

    def test_reconstructed_inputs_match_audited_fill_cast_mask_and_scalers(self):
        for policy,gap in zip(POLICIES,(0,1,2,3)):
            data=SplitData(self.context,'train',policy)
            positions=np.arange(min(20,len(data)),dtype=np.int64)
            x,y,ids=data.batch(positions)
            expected=[]
            for row in ids:
                node=int(row['node_index'])
                origin=int((row['origin'].view('i8')-self.f.edges[0])//600_000_000_000)
                original=self.f.fixture.originals[node]
                filled=causal_forward_fill(original,gap,self.f.boundaries.values()) if gap else original
                raw=np.column_stack((filled[origin-36:origin],np.full(36,node))).astype(np.float32)
                scaled=((raw-np.asarray(self.context.scalers['x_mean']))/np.asarray(self.context.scalers['x_std'])).astype(np.float32)
                scaled[:,[7,8]]=0;expected.append(scaled)
            self.assertEqual(x.tobytes(),np.array(expected).tobytes())
            self.assertEqual(y.tobytes(),data.targets[positions].tobytes())
            self.assertTrue(np.isfinite(x).all())

    def test_all_policies_use_byte_identical_common_evaluation_inputs_and_ids(self):
        for split in ('val','test'):
            reference=SplitData(self.context,split)
            positions=np.arange(len(reference),dtype=np.int64)
            x,y,ids=reference.batch(positions)
            for policy in POLICIES:
                data=SplitData(self.context,split,policy)
                a,b,c=data.batch(positions)
                self.assertEqual(a.tobytes(),x.tobytes());self.assertEqual(b.tobytes(),y.tobytes())
                self.assertEqual(c.tobytes(),ids.tobytes())

    def test_bad_evaluation_window_order_is_rejected_without_inference(self):
        data=SplitData(self.context,'val')
        ids=np.array(data.ids,copy=True)
        ids[[0,1]]=ids[[1,0]]
        with self.assertRaisesRegex(ValueError,'identities/order'):
            data.require_identity(ids)

    def test_unobserved_future_target_is_rejected_even_with_fill_enabled(self):
        data=SplitData(self.context,'train','ff20')
        row=data.ids[0];node=int(row['node_index'])
        origin=int((row['origin'].view('i8')-self.context.contract['grid_start_ns'])//600_000_000_000)
        path=self.directory/data.info['observed_power'];path.chmod(0o644)
        power=np.load(path,mmap_mode='r+',allow_pickle=False)
        power[node,origin-data.info['lower']]=np.nan;power.flush()
        with self.assertRaisesRegex(ValueError,'not observed'):
            data.batch(np.array([0],dtype=np.int64))
        power._mmap.close()

    def test_changed_cohort_receipt_or_artifact_fails_hash_verification(self):
        with self.assertRaisesRegex(ValueError,'receipt changed'):
            CohortContext(self.directory,'0'*64)
        path=self.directory/'scalers.json';path.chmod(0o644)
        with path.open('ab') as handle:handle.write(b' ')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            CohortContext(self.directory,self.context.receipt_sha256)

    def test_window_node_origin_and_boundary_tampering_fail(self):
        data=SplitData(self.context,'train')
        path=self.directory/data.spec['paths']['ids'];path.chmod(0o644)
        ids=np.load(path,mmap_mode='r+',allow_pickle=False)
        original=ids[0].copy()
        for field,value in (('node','wrong'),('origin',np.datetime64('2022-03-01','ns'))):
            ids[0]=original;ids[field][0]=value;ids.flush()
            with self.assertRaises(ValueError):data.batch(np.array([0],dtype=np.int64))
        ids._mmap.close()

    def test_local_sample_cannot_duplicate_or_cross_client_rows(self):
        registry=TrainingRegistry(self.context,'no_fill')
        item=round_plan(registry,42,1,clients_per_round=2)[0]
        rows=item['rows'].copy();rows[1]=rows[0]
        with self.assertRaises(ValueError):registry.local(item['node_index'],rows)
        with self.assertRaises(ValueError):registry.local(item['node_index'],item['rows'].astype(float))

    def test_test_and_training_scoring_requirements_reject_before_model_calls(self):
        model=mock.Mock()
        for split in ('train','test'):
            with self.assertRaises(PermissionError):predict_and_score(model,SplitData(self.context,split),'cpu')
        model.eval.assert_not_called()


class SamplingTests(FixtureTests):
    def test_fedavg_weights_processed_counts_and_clients_share_immutable_start(self):
        registry=TrainingRegistry(self.context,'ff20')
        model=initialize(self.context,42);snapshot=GlobalModelState(model,round_id=0)
        fingerprint=snapshot.fingerprint;updates=[]
        for i in range(2):
            client=snapshot.initialize_client(model)
            self.assertEqual(GlobalModelState(client,round_id=0).fingerprint,fingerprint)
            with torch.no_grad():client.head.bias.fill_(float(i))  # synthetic tensor update, never a training step
            updates.append(snapshot.capture_update(registry.names[i],client,registry.m[i]))
        counts={registry.names[i]:registry.m[i] for i in range(2)}
        result=aggregate_fedavg(snapshot,updates,expected_counts=counts)
        expected=registry.m[1]/sum(registry.m)
        self.assertAlmostEqual(result['head.bias'].item(),expected,places=6)
        self.assertEqual(snapshot.fingerprint,fingerprint)
        wrong={registry.names[i]:registry.counts[i] for i in range(2)}
        self.assertNotEqual(wrong,counts)
        with self.assertRaises(ValueError):aggregate_fedavg(snapshot,updates,expected_counts=wrong)

    def test_independent_client_budget_replay_matches_and_rejects_drift(self):
        self.assertEqual(verify_budget_totals(self.context),self.context.budgets['budget_totals'])
        self.context.budgets['budget_totals']['42']['examples_processed_per_method_policy']+=1
        with self.assertRaisesRegex(ValueError,'budgets differ'):verify_budget_totals(self.context)

    def test_sampling_is_reproducible_without_replacement_at_exact_reference_budget(self):
        for policy in POLICIES:
            registry=TrainingRegistry(self.context,policy)
            a=round_plan(registry,42,1,clients_per_round=2)
            b=round_plan(registry,42,1,clients_per_round=2)
            self.assertEqual(metadata(a),metadata(b))
            for item,other in zip(a,b):
                self.assertTrue(np.array_equal(item['rows'],other['rows']))
                self.assertEqual(len(np.unique(item['rows'])),item['m_k'])
                self.assertEqual(sum(item['batch_sizes']),item['m_k'])
                self.assertEqual(item['m_k'],registry.m[item['node_index']])

    def test_algorithm_independence_and_equal_candidate_pools_produce_same_plan(self):
        registry=TrainingRegistry(self.context,'no_fill')
        a=round_plan(registry,43,2,clients_per_round=2)
        registry.policy='ff30'  # The label is deliberately not a random-stream input.
        b=round_plan(registry,43,2,clients_per_round=2)
        self.assertEqual(metadata(a),metadata(b))
        self.assertTrue(all(np.array_equal(x['rows'],y['rows']) for x,y in zip(a,b)))

    def test_initialization_sampling_and_shuffle_domains_are_separate(self):
        self.assertEqual(len({stream_seed(42,d,1,0) for d in range(4)}),4)
        registry=TrainingRegistry(self.context,'ff20')
        a=round_plan(registry,42,1,clients_per_round=2)
        b=round_plan(registry,42,2,clients_per_round=2)
        self.assertNotEqual(metadata(a),metadata(b))

    def test_all_policy_sample_passes_have_equal_examples_and_step_budgets(self):
        budgets=[]
        for policy in POLICIES:
            plan=round_plan(TrainingRegistry(self.context,policy),44,3,clients_per_round=2)
            budgets.append((sum(p['m_k'] for p in plan),sum(p['optimizer_steps'] for p in plan)))
        self.assertEqual(len(set(budgets)),1)

    def test_validation_schedule_and_earliest_tie_selection(self):
        self.assertEqual([r for r in range(201) if due(r,200)],list(range(0,201,10)))
        self.assertEqual([r for r in range(4) if due(r,3)],[0,3])
        self.assertEqual(select(2.,2.,10,0),(2.,0))
        self.assertEqual(select(1.,2.,10,0),(1.,10))
        with self.assertRaises(ValueError):select(float('nan'),2.,10,0)

    def test_matched_cpu_initialization_is_exact_without_any_forward_pass(self):
        for seed in (42,43,44):
            fingerprints=[]
            for _ in range(8):
                model=initialize(self.context,seed)
                fingerprints.append(GlobalModelState(model,round_id=0).fingerprint)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertFalse(model.baseline_coefficients.requires_grad)
            self.assertEqual(len(set(fingerprints)),1)


class LedgerTests(FixtureTests):
    def test_nonfinite_client_parameters_are_rejected_without_training(self):
        model=initialize(self.context,42);snapshot=GlobalModelState(model,round_id=0)
        client=snapshot.initialize_client(model)
        with torch.no_grad():client.head.bias.fill_(float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            snapshot.capture_update(self.context.names[0],client,1)

    def test_source_snapshot_resume_and_preservation_checks(self):
        source=self.f.root/'original.py';source.write_bytes(b'# protected source\n')
        before=sha256(source)
        output=self.f.root/'snapshot_ledger'
        manifest={'plan':{},'sources':{'original.py':before}}
        tio.open_run(self.f.root,output,manifest)
        tio.open_run(self.f.root,output,manifest,resume=True)
        self.assertEqual(sha256(source),before)
        snapshot=output/'source/original.py';snapshot.chmod(0o644)
        snapshot.write_bytes(b'# changed snapshot\n')
        with self.assertRaisesRegex(ValueError,'snapshot changed'):
            tio.open_run(self.f.root,output,manifest,resume=True)
        self.assertEqual(sha256(source),before)

    def fake_ledger(self,cap=3,last=1):
        """No-op control-flow fixture: weights never trained, scalar logs are synthetic."""
        registry=TrainingRegistry(self.context,'no_fill')
        output=self.f.root/'ledger'
        tio.open_run(self.f.root,output,{'plan':{},'sources':{}},resume=False)
        model=initialize(self.context,42)
        initial=GlobalModelState(model,round_id=0).state_dict()
        examples=steps=0;records=[]
        for r in range(last+1):
            plans=round_plan(registry,42,r,clients_per_round=2) if r else []
            examples+=sum(p['m_k'] for p in plans);steps+=sum(p['optimizer_steps'] for p in plans)
            state=GlobalModelState(model,round_id=r)
            val={'rmse_w':2.,'synthetic_selection_metadata':True} if due(r,cap) else None
            payload={'round':r,'state':state.state_dict(),'best_state':initial,'best_round':0,'best_score':2.,
                     'examples_processed':examples,'optimizer_steps':steps}
            record={'round':r,'clients':metadata(plans),'validation':val,'examples_processed':examples,
                    'optimizer_steps':steps,'global_fingerprint':state.fingerprint}
            records.append(tio.commit(output,payload,record,plans))
        return output,registry,payload,records

    def test_checkpoint_resume_replays_same_sample_rows_and_accounts_exactly(self):
        output,registry,_,_=self.fake_ledger()
        payload,records=tio.load_commits(output)
        replay=lambda registry,seed,r:round_plan(registry,seed,r,clients_per_round=2)
        with mock.patch('src.v5_missing_data.run_policy_experiments.round_plan',side_effect=replay):
            validate_payload(output,payload,records,registry,initialize(self.context,42),42,3)
            bad=dict(payload,examples_processed=payload['examples_processed']+1)
            with self.assertRaisesRegex(ValueError,'cursor/counts'):
                validate_payload(output,bad,records,registry,initialize(self.context,42),42,3)

    def test_interrupted_pending_round_is_retained_and_not_treated_as_commit(self):
        output,_,_,_=self.fake_ledger()
        pending=output/'rounds/.pending-0002-interrupted';pending.mkdir()
        (pending/'state.pt').write_bytes(b'partial')
        payload,records=tio.load_commits(output)
        self.assertEqual(payload['round'],1);self.assertEqual(len(records),2)
        self.assertTrue(pending.exists())

    def test_checkpoint_tampering_and_committed_overwrite_are_rejected(self):
        output,_,payload,records=self.fake_ledger()
        with self.assertRaises(FileExistsError):tio.commit(output,payload,records[-1],[])
        path=output/'rounds/0001/state.pt';path.chmod(0o644)
        with path.open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):tio.load_commits(output)

    def test_export_reload_is_exact_and_fixed_buffer_tampering_is_rejected(self):
        output,_,payload,_=self.fake_ledger()
        receipt=tio.export(output,payload,{'validation':{'rmse_w':2.,'synthetic_selection_metadata':True}})
        self.assertTrue(receipt['export_reload_tensor_exact'])
        state={k:v.clone() for k,v in payload['state'].items()}
        state['baseline_coefficients'][0]+=1
        with self.assertRaisesRegex(ValueError,'fixed state changed'):
            GlobalModelState(initialize(self.context,42),round_id=0)._validate(state,label='bad')

    def test_no_existing_fit_or_frozen_cohort_path_can_be_overwritten(self):
        with self.assertRaises(ValueError):output_path(self.f.root,self.directory)
        output,_,_,_=self.fake_ledger()
        with self.assertRaises(FileExistsError):tio.open_run(self.f.root,output,{'plan':{},'sources':{}},resume=False)


class SelectionLockTests(FixtureTests):
    def synthetic_exports(self):
        records=[]
        for method in ('fedavg','centralized_sgd'):
            for policy in POLICIES:
                for seed in (42,43,44):
                    run=self.f.root/f'{method}_{policy}_{seed}';(run/'selected').mkdir(parents=True)
                    record={'method':method,'policy':policy,'seed':seed,'run':str(run),'selected_round':10,
                            'validation':{'rmse_w':2.,'synthetic_selection_metadata':True}}
                    for file,key in (('selected/model.pt','model_sha256'),('selected/receipt.json','selected_receipt_sha256'),
                                     ('completed.json','completed_sha256'),('manifest.json','manifest_sha256')):
                        (run/file).write_bytes(b'synthetic hash fixture; not a model')
                        record[key]=sha256(run/file)
                    records.append(record)
        return records

    def test_selection_lock_precedes_test_array_access_and_binds_all_exports(self):
        plan={'synthetic':True};records=self.synthetic_exports();output=self.f.root/'locked'
        with mock.patch('src.v5_missing_data.evaluate_policy_models.selections',return_value=records), \
             mock.patch('src.v5_missing_data.evaluate_policy_models.SplitData',side_effect=AssertionError('Test examples opened')):
            result=make_lock(plan,self.context,self.f.root,self.f.root,output)
            lock,permission=load_lock(plan,self.context,output)
        self.assertFalse(result['test_examples_opened']);self.assertEqual(len(lock['fits']),24)
        self.assertEqual(permission.common_identity_sha256,self.context.contract['cohorts']['common/all_policies/test']['identity']['ordered_sha256'])
        model=Path(records[0]['run'])/'selected/model.pt'
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'export changed'):load_lock(plan,self.context,output)
        with self.assertRaises(FileExistsError):
            with mock.patch('src.v5_missing_data.evaluate_policy_models.selections',return_value=records):
                make_lock(plan,self.context,self.f.root,self.f.root,output)

    def test_changed_selection_configuration_is_rejected_before_test_access(self):
        records=self.synthetic_exports();output=self.f.root/'locked'
        with mock.patch('src.v5_missing_data.evaluate_policy_models.selections',return_value=records):
            make_lock({'synthetic':True},self.context,self.f.root,self.f.root,output)
        with self.assertRaisesRegex(ValueError,'lock/data/configuration changed'):
            load_lock({'synthetic':False},self.context,output)


if __name__=='__main__':unittest.main()
