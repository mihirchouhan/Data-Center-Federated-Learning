"""Invented CPU/CUDA models, prepared-window fixtures, recovery and isolation."""
from pathlib import Path
import contextlib,copy,hashlib,io,json,os,tempfile,unittest
from unittest import mock
import numpy as np
import torch
from src import b1_real_fedavg_runner_v1 as runner,b1_production_data_v1 as data,b1_production_io_v1 as disk
from src.b1_device_adapter_v1 import B1DeviceAdapter,explicit_device
from src import b1_real_data_bridge_v1 as bridge,b1_bridge_scaling_v1 as scaling,b1_preparation_io_v1 as fs
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_fedavg_orchestration_v1 import RoundSnapshot,ClientStateUpdate,aggregate_fedavg
from src.phase2_multioutput.contracts import identity_digest
from test_b1_real_data_bridge_v1 import fixture


def invented_client(node='1', count=4):
    ordinal=bridge.ORIGINAL_NAMES.index(node);rng=np.random.default_rng(11+ordinal)
    x=rng.normal(0,.1,(count,36,14)).astype('f4');x[:,:,13]=ordinal
    f=np.empty((count,6,2),'f8');f[:,:,0]=.4+np.arange(count)[:,None]*.0001;f[:,:,1]=23+np.arange(count)[:,None]*.01
    origins=np.datetime64('2022-01-10T06','ns')+np.arange(count)*np.timedelta64(1,'h')
    return data.ClientData(node,x,f,np.ones(f.shape,bool),origins)


class InventedData:
    scope='synthetic'
    def __init__(self,names=('1','2','979'),count=4):
        self.names=names;self.clients={n:invented_client(n,count) for n in names};self.counts={n:count for n in names}
        self.scaler=scaling.FrozenScaler((0.,)*14,(1.,)*14,(1.,)*13+(0.,),(400.,25.),(100.,5.),bridge.canonical({'fit_split':'train','algorithm':scaling.ALGORITHM,'mapping_digest':bridge.MAPPING_DIGEST}))
        self._val=[]
        for node in names:
            c=self.clients[node];y=bridge.target_validation.target_means(c.future,c.observed)
            origins=np.datetime64('2022-01-28T08','ns')+np.arange(count)*np.timedelta64(1,'h')
            self._val.append((c.x,y,np.array([node]*count,dtype='<U3'),origins))
        ns=np.concatenate([x[2] for x in self._val]);ts=np.concatenate([x[3] for x in self._val]);ys=np.concatenate([x[1] for x in self._val])
        self.binding={'synthetic':True,'results':{'val':{'windows':len(ys),'identity_sha256':identity_digest(ns,ts),'target_sha256':hashlib.sha256(ys.astype('<f8').tobytes()).hexdigest()}}}
    def client(self,n):return self.clients[n]
    def validation_batches(self):yield from self._val
    def assert_unchanged(self):return True


def test_config(rounds=2):
    settings=copy.deepcopy(runner.FIXED);settings.update(rounds=rounds,clients=3,clients_per_round=2,validation_every=1)
    return {'kind':'synthetic','settings':settings}


def model(device='cpu', seed=42):return B1DeviceAdapter(target_scaling=TargetScaling((400.,25.),(100.,5.)),root_seed=seed,device=device)


def run(d,out,**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):return runner.run_seed(d,out,test_config(),42,**kwargs)


class AdapterTests(unittest.TestCase):
    def test_exact_architecture_and_state_roles(self):
        m=model();self.assertFalse(m.core.residual);self.assertEqual(tuple(m.core.node_embedding.weight.shape),(979,8));self.assertEqual(m.core.recurrent.hidden_size,64);self.assertEqual(m.core.recurrent.num_layers,2);self.assertEqual(tuple(m.core.head.weight.shape),(2,64))
        self.assertEqual(runner.state_contract(m.snapshot(0).state_dict(),m.target_scaling),m.snapshot(0).trainable_keys)
    def test_derived_initialization_is_deterministic(self):
        a=model();b=model();self.assertEqual(a.snapshot(0).fingerprint,b.snapshot(0).fingerprint)
        self.assertNotEqual(a.snapshot(0).fingerprint,model(seed=43).snapshot(0).fingerprint)
        expected=int(np.random.SeedSequence([42,0,0,0]).generate_state(1)[0]);self.assertEqual(a.seed,expected)
    def test_matches_historical_architecture_with_derived_seed(self):
        from src.b1_model_adapter_v1 import B1TwoOutputAdapter
        m=model();old=B1TwoOutputAdapter(node_names=bridge.ORIGINAL_NAMES,expected_mapping_digest=bridge.MAPPING_DIGEST,target_scaling=m.target_scaling,seed=m.seed)
        for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]))
    def test_initialization_preserves_external_cpu_rng(self):
        state=torch.random.get_rng_state().clone();model();self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
    def test_forward_shape_order_and_inverse(self):
        m=model();c=invented_client();x=torch.tensor(c.x.copy())
        with torch.no_grad():
            m.core.head.weight.zero_();m.core.head.bias.copy_(torch.tensor([1.,-1.]))
        y=m(x);torch.testing.assert_close(y,torch.tensor([[500.,20.]]*len(c),dtype=torch.float64))
    def test_loss_equal_output_standardized(self):
        m=model();c=invented_client();x=torch.tensor(c.x.copy());y=m.target_scaling.standardize(m.observed_targets(c.future,c.observed))
        expected=((m.forward_standardized(x)-torch.tensor(y,dtype=torch.float32))**2).mean()
        torch.testing.assert_close(m.loss(x,c.future,c.observed),expected)
    def test_masked_boolean_invalid_targets_rejected_before_forward(self):
        m=model();c=invented_client();x=torch.tensor(c.x.copy())
        for f in (np.ma.array(c.future,mask=False),c.future.astype(bool),np.full(c.future.shape,np.nan)):
            with mock.patch.object(m,'forward_standardized',side_effect=AssertionError('forward forbidden')):
                with self.assertRaises(ValueError):m.loss(x,f,c.observed)
    def test_exact_device_dtype_and_ordinal_validation(self):
        m=model();x=torch.tensor(invented_client().x.copy())
        with self.assertRaises(ValueError):m(x.double())
        x[:,:,13]=979
        with self.assertRaises(ValueError):m(x)
        for d in ('cuda','meta','cpu:0'):
            with self.assertRaises((ValueError,RuntimeError)):explicit_device(d)
    def test_snapshot_round_transfer_and_fixed_buffer_rejection(self):
        m=model();snap=m.snapshot(0);changed=snap.state_dict();changed['core.head.bias']+=1
        target=RoundSnapshot(changed,snap.trainable_keys,round_id=1);m.restore_round(target);self.assertEqual(m.snapshot(1).fingerprint,target.fingerprint)
        bad=target.state_dict();bad['target_std']+=1
        with self.assertRaises(ValueError):m.restore_round(RoundSnapshot(bad,target.trainable_keys,round_id=1))
    def test_loose_checkpoint_loading_stays_disabled(self):
        m=model()
        with self.assertRaises(PermissionError):m.load_state_dict(m.state_dict())
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable: driver/runtime probe failed; no GPU readiness claim')
    def test_cuda_synthetic_forward_backward_optimizer_and_transfer(self):
        m=model('cuda:0');c=invented_client();update,rec=runner.local_update(m,c,42,1)
        self.assertEqual(rec['optimizer_steps'],1);self.assertTrue(all(t.device.type=='cpu' for t in update.state.values()))
        m.restore_round(RoundSnapshot(update.state,m.snapshot(0).trainable_keys,round_id=1))
        self.assertTrue(all(p.device==torch.device('cuda:0') for p in m.parameters()))
        with self.assertRaises(ValueError):m(torch.tensor(c.x.copy()))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable: no CPU/CUDA comparison executed')
    def test_cpu_cuda_initial_weights_and_synthetic_outputs(self):
        cpu=model();gpu=model('cuda:0');self.assertEqual(cpu.snapshot(0).fingerprint,gpu.snapshot(0).fingerprint)
        x=torch.tensor(invented_client().x.copy());torch.testing.assert_close(cpu(x),gpu(x.cuda(0)).cpu(),rtol=2e-5,atol=2e-5)


class LocalAndAggregationTests(unittest.TestCase):
    def test_actual_local_steps_and_partial_batches(self):
        m=model()
        for count in (17,128,129,257):
            c=invented_client(count=count);u,r=runner.local_update(m,c,42,1)
            self.assertEqual(r['examples'],count);self.assertEqual(r['optimizer_steps'],math_ceil(count/128));self.assertEqual(sum(r['batch_sizes']),count)
            self.assertTrue(any(not torch.equal(u.state[k],m.snapshot(0).state_dict()[k]) for k in m.snapshot(0).trainable_keys))
    def test_fresh_optimizer_and_independent_clients(self):
        m=model();before=m.snapshot(0).fingerprint;original=torch.optim.SGD;objects=[]
        def create(*a,**kw):
            o=original(*a,**kw);objects.append(o);return o
        with mock.patch('torch.optim.SGD',side_effect=create):
            runner.local_update(m,invented_client('1'),42,1);runner.local_update(m,invented_client('2'),42,1)
        self.assertIsNot(objects[0],objects[1]);self.assertIsNot(objects[0].state,objects[1].state)
        self.assertTrue(set(map(id,objects[0].param_groups[0]['params'])).isdisjoint(map(id,objects[1].param_groups[0]['params'])))
        self.assertEqual(m.snapshot(0).fingerprint,before)
    def test_deterministic_local_shuffle_and_updates(self):
        m=model();c=invented_client(count=129);a,ra=runner.local_update(m,c,42,1);b,rb=runner.local_update(m,c,42,1)
        self.assertEqual(ra['row_order_sha256'],rb['row_order_sha256'])
        for k in a.state:self.assertTrue(torch.equal(a.state[k],b.state[k]))
    def test_full_table_weighting_and_embedding_delta_dilution(self):
        m=model();base=m.snapshot(0);a=base.state_dict();b=base.state_dict();a['core.node_embedding.weight'][0]+=4;b['core.head.bias']+=2
        merged=aggregate_fedavg(base,[ClientStateUpdate('1',1,base.fingerprint,a),ClientStateUpdate('2',3,base.fingerprint,b)],expected_counts={'1':1,'2':3},client_order=['1','2'])
        torch.testing.assert_close(merged['core.node_embedding.weight'][0],base.state_dict()['core.node_embedding.weight'][0]+1)
        self.assertTrue(torch.equal(merged['core.node_embedding.weight'][978],base.state_dict()['core.node_embedding.weight'][978]))
        torch.testing.assert_close(merged['core.head.bias'],base.state_dict()['core.head.bias']+1.5)
    def test_sampling_matches_existing_98_of_979(self):
        from src.b1_fedavg_orchestration_v1 import ClientRegistry,sample_clients
        registry=ClientRegistry(bridge.ORIGINAL_NAMES,{n:24 for n in bridge.ORIGINAL_NAMES},expected_mapping_digest=bridge.MAPPING_DIGEST)
        for seed in (42,43,44):
            wanted=tuple(bridge.ORIGINAL_NAMES[i] for i in sample_clients(registry,seed,1));actual=runner.select_clients(bridge.ORIGINAL_NAMES,seed,1,98)
            self.assertEqual(wanted,actual);self.assertEqual(len(set(actual)),98)
    def test_invalid_client_before_optimizer(self):
        with self.assertRaises(ValueError):data.ClientData('1',invented_client().x,np.ma.array(invented_client().future,mask=False),invented_client().observed,invented_client().origins)
    def test_March_origins_rejected_before_targets(self):
        c=invented_client();origins=np.array(['2022-03-01T08']*len(c),dtype='datetime64[ns]')
        with self.assertRaisesRegex(ValueError,'chronology'):data.ClientData('1',c.x,np.ma.array(c.future,mask=False),c.observed,origins)


def math_ceil(x):return int(np.ceil(x))


class ValidationAndRecoveryTests(unittest.TestCase):
    def test_validation_is_complete_no_grad_and_no_parameter_updates(self):
        d=InventedData();m=model();before=m.snapshot(0).fingerprint
        with mock.patch('torch.optim.SGD',side_effect=AssertionError('optimizer forbidden')):v=runner.evaluate_validation(m,d,0,5)
        self.assertEqual(v['windows'],12);self.assertEqual(v['identity_sha256'],d.binding['results']['val']['identity_sha256']);self.assertTrue(all(p.grad is None for p in m.parameters()));self.assertEqual(before,m.snapshot(0).fingerprint)
        self.assertAlmostEqual(v['joint_standardized_RMSE']**2,v['joint_standardized_MSE_J'])
    def test_earliest_strict_minimum_with_exact_ties(self):
        trace=[{'round':r,'joint_standardized_RMSE':v} for r,v in [(0,2.),(10,1.),(20,1.),(30,1.2)]]
        self.assertEqual(runner.earliest_minimum(trace)['round'],10)
    def test_validation_missing_identity_is_rejected(self):
        d=InventedData();d.binding['results']['val']['windows']+=1
        with self.assertRaisesRegex(ValueError,'cohort'):runner.evaluate_validation(model(),d,0)
    def test_resume_matches_uninterrupted_two_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();a=run(d,root/'clean');partial=run(d,root/'resume',stop_after_round=1)
            self.assertEqual(partial['status'],'verified_partial');self.assertFalse((root/'resume/completed.json').exists())
            b=run(d,root/'resume',resume=True)
            self.assertEqual(a['status'],b['status']);self.assertEqual(a['selection'],b['selection'])
            for i in (0,1,2):self.assertEqual(data.read_json(root/'clean/rounds'/f'{i:04d}'/'receipt.json')['state_fingerprint'],data.read_json(root/'resume/rounds'/f'{i:04d}'/'receipt.json')['state_fingerprint'])
    def test_failed_round_replays_from_prior_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();original=runner.local_update
            def fail(m,c,seed,r,*a):
                if r==1:raise RuntimeError('injected client failure')
                return original(m,c,seed,r,*a)
            with mock.patch.object(runner,'local_update',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'injected'):run(d,root/'failed')
            self.assertTrue((root/'failed/rounds/0000/receipt.json').exists());self.assertFalse((root/'failed/rounds/0001').exists())
            self.assertEqual(run(d,root/'failed',resume=True)['status'],'training_complete')
    def test_pending_round_preserved_and_not_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0)
            pending=root/'run/rounds'/('.pending-0001-'+'a'*32);pending.mkdir();(pending/'checkpoint.pt').write_bytes(b'incomplete')
            run(d,root/'run',resume=True);self.assertEqual((pending/'checkpoint.pt').read_bytes(),b'incomplete')
    def test_occupied_directory_cannot_restart_from_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0)
            with self.assertRaises(FileExistsError):run(d,root/'run')
    def test_checkpoint_tampering_rejected_before_model_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0)
            p=root/'run/rounds/0000/checkpoint.pt';p.write_bytes(p.read_bytes()+b'tampered')
            with mock.patch('src.b1_device_adapter_v1.B1DeviceAdapter',side_effect=AssertionError('initialization forbidden')):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):run(d,root/'run',resume=True)
    def test_incompatible_configuration_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0);cfg=test_config();cfg['settings']['rounds']=3
            with self.assertRaisesRegex(ValueError,'mismatch'):runner.run_seed(d,root/'run',cfg,42,resume=True)
    def test_missing_round_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');(root/'run/rounds/0001').rename(root/'run/rounds/.pending-0001-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            with self.assertRaisesRegex(ValueError,'noncontiguous'):run(d,root/'run',resume=True)
    def test_uninitialized_run_never_silently_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m={'sources':{},'schema':'synthetic'}
            with disk.RunStore(root/'empty',m):pass
            with self.assertRaisesRegex(ValueError,'No committed'):disk.RunStore(root/'empty',m,resume=True)
    def test_concurrent_writer_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);m={'sources':{}}
            with disk.RunStore(root/'run',m):
                with self.assertRaises(BlockingIOError):disk.RunStore(root/'run',m,resume=True)
    def test_source_snapshot_and_live_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'source.py').write_text('first');m={'sources':{'source.py':fs.binding(root/'source.py')}}
            with disk.RunStore(root/'run',m,project_root=root):pass
            (root/'source.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source'):disk.RunStore(root/'run',m,resume=True,project_root=root)
    def test_completed_run_does_not_initialize_another_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run')
            with mock.patch('src.b1_device_adapter_v1.B1DeviceAdapter',side_effect=AssertionError('must skip completed')):self.assertEqual(run(d,root/'run',resume=True)['status'],'training_complete')
    def test_tampered_validation_selection_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');m=data.read_json(root/'run/manifest.json')
            with disk.RunStore(root/'run',m,resume=True) as store:
                payload=store.load();payload['selected_round']=199
                with self.assertRaisesRegex(ValueError,'earliest'):runner.validate_payload(payload,m,d,store.records)
    def test_symlink_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'target').mkdir();(root/'link').symlink_to(root/'target',target_is_directory=True)
            with self.assertRaises(ValueError):run(InventedData(),root/'link')
    def test_production_cannot_use_test_fixture_or_short_config(self):
        d=InventedData()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Verified production'):runner.run_seed(d,Path(folder)/'bad',{'kind':'production','settings':runner.FIXED},42)
    def test_missing_review_blocks_before_data_loader_or_model(self):
        fake={'sources':{}}
        with mock.patch.object(runner,'load_config',return_value=fake),mock.patch.object(data,'PreparedData',side_effect=AssertionError('loader forbidden')):
            with self.assertRaisesRegex(ValueError,'review receipt'):runner.main(['--plan','unused','--stage','fit','--device','cpu','--seed','42','--output-dir','unused'])
    def test_signal_or_failure_never_marks_fit_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData()
            with mock.patch.object(runner,'evaluate_validation',side_effect=InterruptedError('signal')):
                with self.assertRaises(InterruptedError):run(d,root/'run')
            self.assertFalse((root/'run/completed.json').exists());self.assertTrue(list((root/'run/attempts').rglob('*failure.json')))

    def test_invalid_rng_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0);m=data.read_json(root/'run/manifest.json')
            with disk.RunStore(root/'run',m,resume=True) as store:
                payload=store.load();payload['rng']['torch_cpu']=torch.zeros(3,dtype=torch.uint8)
                with self.assertRaises((ValueError,RuntimeError)):runner.validate_payload(payload,m,d,store.records)
    def test_historical_validation_rows_are_bound_to_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');m=data.read_json(root/'run/manifest.json')
            with disk.RunStore(root/'run',m,resume=True) as store:
                payload=store.load();payload['validation_history'][0]['mae'][0]+=.001
                with self.assertRaisesRegex(ValueError,'Historical validation'):runner.validate_payload(payload,m,d,store.records)
    def test_copied_run_cannot_resume_at_another_path(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=0);shutil.copytree(root/'run',root/'copy')
            with self.assertRaisesRegex(ValueError,'mismatch'):run(d,root/'copy',resume=True)
    def test_low_storage_stops_without_round_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with disk.RunStore(root/'run',{'sources':{}}) as store:
                v=type('V',(),{'f_bavail':1,'f_frsize':4096})()
                with mock.patch.object(os,'statvfs',return_value=v):
                    with self.assertRaisesRegex(OSError,'reserve'):store.commit(0,{}, {})
                self.assertFalse((root/'run/rounds/0000').exists())
    def test_production_validation_schedule_has_21_candidates(self):
        self.assertEqual(list(range(0,runner.FIXED['rounds']+1,runner.FIXED['validation_every'])),list(range(0,201,10)))

    def test_read_only_verifier_complete_and_partial_no_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run',stop_after_round=1);plan={'settings':test_config()['settings'],'sources':{}}
            before={str(p.relative_to(root)):fs.binding(p) for p in root.rglob('*') if p.is_file()}
            with mock.patch('src.b1_device_adapter_v1.B1DeviceAdapter',side_effect=AssertionError('model forbidden')):
                result=runner.verify_run(d,root/'run',plan,42)
            self.assertEqual(result['status'],'verified_partial')
            self.assertEqual(before,{str(p.relative_to(root)):fs.binding(p) for p in root.rglob('*') if p.is_file()})
            run(d,root/'run',resume=True)
            self.assertEqual(runner.verify_run(d,root/'run',plan,42)['status'],'verified_complete')
    def test_selected_export_corruption_fails_verification_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');p=root/'run/selected/model.pt';p.write_bytes(p.read_bytes()+b'bad')
            with self.assertRaises(ValueError):runner.verify_run(d,root/'run',{'settings':test_config()['settings'],'sources':{}},42)
            with self.assertRaises(ValueError):run(d,root/'run',resume=True)
    def test_completed_missing_selected_is_not_recreated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');(root/'run/selected/model.pt').rename(root/'run/selected/model.saved')
            with self.assertRaisesRegex(ValueError,'never reconstruct'):run(d,root/'run',resume=True)
            self.assertFalse((root/'run/selected/model.pt').exists())

    def test_review_record_binds_exact_config_sources_device_and_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=root/'config.json';config.write_text('{}');plan={'sources':{}}
            receipt={'schema':'b1.production.independent.review.v1','decision':'APPROVED_FOR_B1_TRAIN_VAL_TRAINING','config':fs.binding(config),'sources':{},'dataset_manifest_sha256':data.DATA_SHA,'scaler_lock_sha256':data.LOCK_SHA,'device':'cuda:0','March_access':False}
            path=root/'review.json';path.write_text(json.dumps(receipt))
            self.assertEqual(runner.require_review(path,plan,config,'cuda:0'),fs.binding(path))
            with self.assertRaisesRegex(ValueError,'review does not match'):runner.require_review(path,plan,config,'cuda:1')
            receipt['March_access']=True;path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):runner.require_review(path,plan,config,'cuda:0')
    def test_stale_checkpoint_receipt_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d=InventedData();run(d,root/'run');receipt=root/'run/rounds/0001/receipt.json';row=data.read_json(receipt);row['previous_receipt_sha256']='0'*64;receipt.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,'chain'):run(d,root/'run',resume=True)


class PreparedInterfaceTests(unittest.TestCase):
    def test_exact_synthetic_prepared_chunk_and_bridge_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);f=fixture(ordinals=(0,));split=f.pop('split');f.pop('mapping');e=f.pop('source_evidence');w=bridge.accept_windows(split=split,mapping=bridge.ORIGINAL_NAMES,source_evidence=e,**f)
            path=root/'train_0000_000000.npz';np.savez(path,**f,physical_targets=w.physical_targets)
            ch={'path':path.name,'split':'train','node':'1','windows':len(w),**fs.binding(path)}
            obj=data.PreparedData.__new__(data.PreparedData);obj.path=root;obj.manifest={'source_evidence':e,'contract':{'limits':vars(fs.Limits())}}
            got=obj._chunk(ch);self.assertEqual(got.identity_sha256,w.identity_sha256);self.assertEqual(got.target_sha256,w.target_sha256)
    def test_foreign_March_dataset_path_rejected_before_read(self):
        with mock.patch.object(fs,'binding',side_effect=AssertionError('foreign bytes forbidden')):
            with self.assertRaisesRegex(ValueError,'Only the bound'):data.PreparedData('/tmp/March/dataset','/tmp/March/lock.json')
    def test_prepared_manifest_hash_failure_stops_before_decoding(self):
        root=runner.ROOT
        with mock.patch.object(fs,'binding',return_value={'bytes':1,'sha256':'bad'}):
            with self.assertRaisesRegex(ValueError,'binding mismatch'):data.PreparedData(root/'output/experiments/b1_no_ff/preparation/train_val_v1/dataset',root/'output/march_window_audit/b1_scaler_approval_and_lock_v1/locked_scaler/lock.json')
    def test_configuration_cannot_change_batch_or_round_count(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'config.json';p.write_text(json.dumps({'settings':{'batch_size':32}}))
            with self.assertRaises(ValueError):runner.load_config(p)
    def test_no_scaler_fitter_in_production_execution_code(self):
        for module in (runner,data):self.assertNotIn('fit_training_scaler(',Path(module.__file__).read_text())


class MatrixControlTests(unittest.TestCase):
    def test_missing_earlier_seed_never_restarts(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'seed43').mkdir()
            with self.assertRaisesRegex(ValueError,'Missing earlier seed'):runner.verify_matrix_prefix(out)
    def test_later_seed_requires_completed_previous_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'seed42').mkdir();(out/'seed43').mkdir()
            with self.assertRaisesRegex(ValueError,'prior seed completed'):runner.verify_matrix_prefix(out)
    def test_partial_matrix_prefix_is_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'seed42').mkdir();runner.verify_matrix_prefix(out)
    def test_complete_marker_rejects_missing_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'matrix_completed.json').write_text('{}')
            with self.assertRaises(ValueError):runner.verify_matrix_prefix(out)
    def test_matrix_runs_sequentially_and_verifies_each_seed(self):self._matrix('complete')
    def test_matrix_stops_after_partial_seed(self):self._matrix('partial')
    def test_matrix_stops_after_failed_verification(self):self._matrix('bad_verify')
    def _matrix(self,case):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'output/experiments/b1_no_ff';parent.mkdir(parents=True)
            plan_path=root/'plan.json';plan_path.write_text('{}');plan={'sources':{},'settings':runner.FIXED,'dataset':'unused','scaler_lock':'unused'}
            fake=InventedData();fake.path=root/'immutable/dataset';fake.lock_path=root/'immutable/scaler/lock.json';order=[]
            def fit(d,path,cfg,seed,*a,**kw):
                order.append(('fit',seed));path.mkdir();(path/'completed.json').write_text('{}')
                return {'status':'verified_partial' if case=='partial' else 'training_complete'}
            def verify(d,path,p,seed):
                order.append(('verify',seed));return {'status':'verified_partial' if case=='bad_verify' else 'verified_complete'}
            with mock.patch.object(runner,'ROOT',root),mock.patch.object(runner,'load_config',return_value=plan),mock.patch.object(runner,'require_review',return_value={'test_fixture_only':True}),mock.patch.object(data,'PreparedData',return_value=fake),mock.patch.object(data,'install_access_guard'),mock.patch.object(runner,'run_seed',side_effect=fit),mock.patch.object(runner,'verify_run',side_effect=verify),mock.patch.object(runner.signal,'signal'):
                args=['--plan',str(plan_path),'--stage','fit','--matrix','--device','cpu','--review-receipt','invented','--output-dir',str(parent/'runs')]
                if case=='bad_verify':
                    with self.assertRaisesRegex(ValueError,'next seed blocked'):runner.main(args)
                else:runner.main(args)
            expected=[('fit',42)] if case=='partial' else ([('fit',42),('verify',42)] if case=='bad_verify' else [(kind,seed) for seed in (42,43,44) for kind in ('fit','verify')])
            self.assertEqual(order,expected);self.assertEqual((parent/'runs/matrix_completed.json').exists(),case=='complete')

if __name__=='__main__':unittest.main()
