"""Synthetic CPU evaluator tests; all checkpoints/history are fabricated fixtures.

No optimizer, real example arrays, GPU inference or historical report is run.
The legacy evaluator is exercised only on temporary synthetic data to construct
a synthetic prior-result fixture, never on the existing real FedAvg evaluation.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from test_federated_evaluate import synthetic_project
from src import centralized_evaluate as ce
from src import centralized_io as cio
from src import federated_evaluate as fed
from src.experiment_data import read_json,immutable_json,sha256
from src.federated_io import digest_json,environment
from src.federated_rounds import initialize_model,sample_clients,stream_seed,configure_runtime
from src.federated_train import GlobalModelState


def synthetic_control_project(root):
    oldplan=synthetic_project(root)
    for name in ce.execution_sources(ROOT):
        dest=root/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    prior=root/'output/experiments/v2_fedavg/test_evaluation/complete'
    plan_file=root/'old_plan.json';immutable_json(plan_file,oldplan)
    fed.evaluate_locked(root,oldplan,root/'frozen',prior,device='cpu',plan_sha256=sha256(plan_file))
    cohort=oldplan['cohort'];data=fed.FrozenTestCohort(root/'frozen',cohort)
    entries=[];proof=[]
    for seed,selected_block,oldentry in zip(ce.SEEDS,ce.BLOCKS,oldplan['runs']):
        old=root/oldentry['run_dir'];oldmanifest=read_json(old/'manifest.json')
        config={**oldmanifest['config'],'version':3,'reference_run':oldentry['run_dir'],
                'reference_manifest_sha256':oldentry['manifest_sha256']}
        config_file=f'configs/experiments/v3_centralized_sgd/gru_masked_seed{seed}_schedule_matched.json'
        (root/config_file).parent.mkdir(parents=True,exist_ok=True);immutable_json(root/config_file,config)
        relative=f'output/experiments/v3_centralized_sgd/main/gru_masked_seed{seed}_schedule_matched'
        run=root/relative;run.mkdir(parents=True);(run/'blocks').mkdir();(run/'final/validation').mkdir(parents=True)
        model=initialize_model(data,config);anchor=GlobalModelState(model)
        reference={'manifest_sha256':oldentry['manifest_sha256'],'initial_fingerprint':anchor.fingerprint,
                   'round_receipt_sha256':[sha256(old/'rounds'/f'{i:04d}'/'receipt.json') for i in range(201)]}
        manifest={'algorithm':'Centralized-SGD-GRU schedule-replay','config':config,'config_digest':digest_json(config),
                  'frozen_contract_sha256':cohort['contract_sha256'],'clients':oldmanifest['clients'],
                  'training_windows':36,'validation_windows':12,'source_hashes':cio.execution_sources(),
                  'environment':environment('cpu'),'reference':reference}
        immutable_json(run/'manifest.json',manifest);immutable_json(run/'configuration.json',config)
        immutable_json(run/'reference_lock.json',reference)
        for name in manifest['source_hashes']:
            dest=run/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((root/name).read_bytes())
        best_score=float('inf');best_block=0;best_state=anchor.state_dict();examples=steps=0;blocks=[]
        for index in range(201):
            if index:
                with torch.no_grad():model.head.bias.fill_(.01*(seed-41)+index/10000)
            snapshot=GlobalModelState(model,round_id=index)
            chosen=list(sample_clients(data.node_names,2,seed,index)) if index else []
            plan=[]
            for name in chosen:
                node=data.node_names.index(name);shuffle=stream_seed(seed,2,index,node)
                order=np.random.default_rng(shuffle).permutation(12)
                import hashlib
                plan.append({'client_id':name,'shuffle_seed':shuffle,'training_windows':12,'examples_processed':12,
                             'optimizer_steps':1,'batch_sizes':[12],
                             'row_order_sha256':hashlib.sha256((np.arange(node*12,(node+1)*12)[order]).tobytes()).hexdigest()})
            clients=[{**p,'optimizer_state_entries':0} for p in plan]
            examples+=24 if index else 0;steps+=2 if index else 0
            if index:blocks.append({'block':index,'schedule_sha256':digest_json(plan)})
            validation=None
            if index%10==0:
                score=300-index if index<=selected_block else 301-selected_block
                validation={'samples':12,'mae_w':score/2,'rmse_w':float(score),'persistence_mae_w':200.,
                            'persistence_rmse_w':300.,'split':'val','unit':'W','evaluated_nodes':3,'finite_predictions':True}
                if score<best_score:best_score=score;best_block=index;best_state=snapshot.state_dict()
            value={'block':index,'state':snapshot.state_dict(),'best_state':best_state,'best_block':best_block,
                   'best_rmse_w':float(best_score),'examples_processed':examples,'optimizer_steps':steps,
                   'manifest_digest':digest_json(manifest)}
            cio.commit_block(run,value,{'block':index,'selected_clients':chosen,'clients':clients,
                             'validation':validation,'global_fingerprint':snapshot.fingerprint,'seconds':.1})
        cio.save_tensor_file(run/'final/model.pt',{'state':best_state,'selected_block':best_block,
                            'validation_rmse_w':float(best_score),'manifest_digest':digest_json(manifest)})
        immutable_json(run/'final/validation/metrics.json',{'rmse_w':float(best_score),'mae_w':best_score/2})
        immutable_json(run/'final/receipt.json',{'manifest_sha256':sha256(run/'manifest.json'),
            'last_block_receipt_sha256':sha256(run/'blocks/0200/receipt.json'),'selected_block':best_block,
            'validation_rmse_w':float(best_score),'export_reload_exact':True,'finite_predictions':True,
            'artifacts':{'model.pt':sha256(run/'final/model.pt'),'validation/metrics.json':sha256(run/'final/validation/metrics.json')}})
        immutable_json(run/'completed.json',{'status':'complete','completed_block':200,'selected_block':best_block,
            'validation_rmse_w':float(best_score),'examples_processed':examples,'optimizer_steps':steps,
            'export_reload_exact':True,'manifest_sha256':sha256(run/'manifest.json'),
            'final_receipt_sha256':sha256(run/'final/receipt.json')})
        entry={'seed':seed,'selected_block':best_block,'run_dir':relative,'config_file':config_file,
               'config_file_sha256':sha256(root/config_file)}
        for name,key in [('manifest.json','manifest_sha256'),('configuration.json','configuration_sha256'),
                         ('completed.json','completed_sha256'),('final/receipt.json','final_receipt_sha256'),('final/model.pt','model_sha256')]:
            entry[key]=sha256(run/name)
        entries.append(entry);proof.append({'seed':seed,'examples':examples,'optimizer_steps':steps,'blocks':blocks})
    path=root/'schedule_proof.json';immutable_json(path,{'seeds':proof})
    return {'version':1,'kind':'frozen_centralized_sgd_final_test','runs':entries,'cohort':cohort,
            'execution_sources':ce.execution_sources(root),'schedule_proof':{'path':'schedule_proof.json','sha256':sha256(path)},
            'prior_fedavg':{'directory':str(prior.relative_to(root)),'completed_sha256':sha256(prior/'completed.json'),
               'plan_sha256':sha256(prior/'plan.json'),'selection_lock_sha256':sha256(prior/'selection_lock.json'),
               'window_identity':read_json(prior/'test_window_identity.json')}}


@contextmanager
def changed(path,data):
    path=Path(path);before=path.read_bytes();mode=path.stat().st_mode
    path.chmod(0o644);path.write_bytes(data)
    try:yield
    finally:path.write_bytes(before);path.chmod(mode)


class CentralizedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'project'
        with mock.patch.dict(os.environ,{'MPLCONFIGDIR':str(Path(cls.temp.name)/'mpl')}):
            cls.plan=synthetic_control_project(cls.root)
        cls.data=cls.root/'frozen';cls.cohort=ce.TestCohort(cls.data,cls.plan['cohort'])
        cls.selections,cls.prior=ce.preflight(cls.root,cls.plan,cls.data)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        self.tempdir=tempfile.TemporaryDirectory();self.addCleanup(self.tempdir.cleanup)
        self.output=Path(self.tempdir.name)/'evaluation'
        patch=mock.patch.dict(os.environ,{'MPLCONFIGDIR':str(Path(self.tempdir.name)/'mpl')});patch.start();self.addCleanup(patch.stop)

    def prediction(self,offset=0.):
        _,y,origin,node=self.cohort.split('test')
        return fed.TestPredictions(np.asarray(y[:,0],dtype=np.float64)+offset,node.copy(),origin.copy())

    def test_correct_three_checkpoint_selections_are_locked(self):
        self.assertEqual([s.entry['selected_block'] for s in self.selections],[190,190,180])
        for s in self.selections:
            self.assertEqual(s.record['examples_processed'],4800);self.assertEqual(s.record['optimizer_steps'],400)
            self.assertEqual(s.record['verified_block_checkpoints'],201)

    def test_preflight_never_loads_example_arrays_or_predicts(self):
        with (mock.patch('numpy.load',side_effect=AssertionError('array mapped')),
              mock.patch('src.centralized_evaluate.infer_test',side_effect=AssertionError('inference'))):
            selected,_=ce.preflight(self.root,self.plan,self.data)
        self.assertEqual(len(selected),3)

    def test_wrong_selection_seed_order_or_missing_seed_fails_before_output(self):
        for change in ('selection','order','missing'):
            plan=copy.deepcopy(self.plan)
            if change=='selection':plan['runs'][0]['selected_block']=200
            if change=='order':plan['runs'].reverse()
            if change=='missing':plan['runs'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):
                ce.evaluate_locked(self.root,plan,self.data,self.output,device='cpu')
            self.assertFalse(self.output.exists())

    def test_hash_substitutions_and_paths_are_rejected(self):
        for key in ('model_sha256','manifest_sha256','configuration_sha256','completed_sha256',
                    'final_receipt_sha256','config_file_sha256','run_dir'):
            plan=copy.deepcopy(self.plan);plan['runs'][0][key]='../escape' if key=='run_dir' else '0'*64
            with self.subTest(key=key),self.assertRaises((ValueError,OSError)):ce.preflight(self.root,plan,self.data)

    def test_source_proof_prior_and_contract_substitutions_are_rejected(self):
        for kind in ('source','proof','prior','contract','count'):
            plan=copy.deepcopy(self.plan)
            if kind=='source':plan['execution_sources']['src/model.py']='0'*64
            if kind=='proof':plan['schedule_proof']['sha256']='0'*64
            if kind=='prior':plan['prior_fedavg']['completed_sha256']='0'*64
            if kind=='contract':plan['cohort']['contract_sha256']='0'*64
            if kind=='count':plan['cohort']['expected_windows']=13
            with self.subTest(kind=kind),self.assertRaises((ValueError,OSError)):ce.preflight(self.root,plan,self.data)

    def test_actual_checkpoint_byte_corruption_is_rejected(self):
        p=self.root/self.plan['runs'][0]['run_dir']/'blocks/0010/state.pt'
        with changed(p,p.read_bytes()+b'corrupt'),self.assertRaises(ValueError):ce.preflight(self.root,self.plan,self.data)

    def test_prior_report_byte_corruption_is_rejected_without_rerun(self):
        p=self.root/self.plan['prior_fedavg']['directory']/'seed42/metrics.json'
        with changed(p,b'{}'),self.assertRaises(ValueError):ce.preflight(self.root,self.plan,self.data)

    def test_test_capability_never_maps_training_or_validation(self):
        original=np.load;seen=[]
        def guard(path,*args,**kwargs):
            name=Path(path).name;seen.append(name)
            self.assertNotIn('_train.',name);self.assertNotIn('_val.',name)
            return original(path,*args,**kwargs)
        with mock.patch('numpy.load',side_effect=guard):cohort=ce.TestCohort(self.data,self.plan['cohort'])
        self.assertEqual(set(seen),{'node_names.npy','power_baseline_coefficients.npy','X_test.npy','y_test.npy','node_test.npy','origin_test.npy'})
        for split in ('train','val'):
            with self.assertRaises(PermissionError):cohort.split(split)

    def test_validation_predictions_and_reordered_ids_cannot_enter_test(self):
        p=self.prediction()
        for value in [replace(p,split='val'),replace(p,nodes=p.nodes[::-1]),replace(p,origins=p.origins[::-1]),
                      replace(p,scaled=p.scaled[:-1])]:
            with self.assertRaises(ValueError):fed.validate_predictions(self.cohort,value)

    def test_nonfinite_and_missing_predictions_fail(self):
        for v in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(v=v),self.assertRaises(ValueError):fed.validate_predictions(self.cohort,self.prediction(v))

    def test_nonfinite_model_output_is_rejected_before_metric_reporting(self):
        with mock.patch('src.centralized_evaluate.neural_predict',return_value=np.full(12,np.nan)):
            with self.assertRaises(ValueError):ce.infer_test(self.cohort,self.selections[0],torch.device('cpu'))

    def test_frozen_model_shape_dtype_fixed_buffer_and_finiteness(self):
        for kind in ('shape','dtype','fixed','nan'):
            s=self.selections[0];state={k:v.clone() for k,v in s.exported['state'].items()}
            if kind=='shape':state['head.bias']=torch.zeros(2)
            if kind=='dtype':state['head.bias']=state['head.bias'].double()
            if kind=='fixed':state['baseline_coefficients'].add_(1)
            if kind=='nan':state['head.bias'].fill_(float('nan'))
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                ce.load_model(self.cohort,replace(s,exported={**s.exported,'state':state}),torch.device('cpu'))

    def test_weather_mask_only_changes_two_standardized_channels(self):
        s=self.selections[0];model=ce.load_model(self.cohort,s,torch.device('cpu'));seen=[]
        hook=model.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().cpu().numpy().copy()))
        with mock.patch('src.centralized_evaluate.load_model',return_value=model):ce.infer_test(self.cohort,s,torch.device('cpu'))
        hook.remove();actual=np.concatenate(seen);x=self.cohort.split('test')[0]
        np.testing.assert_array_equal(actual[:,:,7:9],0)
        keep=[k for k in range(14) if k not in (7,8)]
        np.testing.assert_array_equal(actual[:,:,keep],x[:,:,keep]);self.assertEqual(actual.shape,(12,6,14))

    def test_inference_cannot_mutate_selected_parameters(self):
        original=ce.neural_predict
        def altered(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            with torch.no_grad():model.head.bias.add_(1)
            return result
        with mock.patch('src.centralized_evaluate.neural_predict',side_effect=altered),self.assertRaisesRegex(ValueError,'changed'):
            ce.infer_test(self.cohort,self.selections[0],torch.device('cpu'))

    def test_watts_metrics_and_persistence_are_the_common_contract(self):
        offsets=np.tile([0.,1.,-2.,3.],3)
        score=fed.evaluate_predictions(self.cohort,self.prediction(offsets),self.output)
        self.assertAlmostEqual(score['mae_w'],300.)
        self.assertAlmostEqual(score['rmse_w'],np.sqrt(3.5)*200)
        with np.load(self.output/'predictions.npz',allow_pickle=False) as p:
            x,y,_,_=self.cohort.split('test')
            np.testing.assert_array_equal(p['targets_w'],(y[:,0].astype('float64')*.2+.5)*1000)
            np.testing.assert_array_equal(p['persistence_w'],(x[:,-1,0].astype('float64')*.4+.6)*1000)

    def test_mean_and_sample_standard_deviation_not_ensemble(self):
        rows=[{'seed':seed,'rmse_w':float(i),'mae_w':2.*i,'persistence_rmse_w':4.,'persistence_mae_w':5.,
               'samples':12,'window_identity':self.cohort.identity} for seed,i in zip(ce.SEEDS,(1,2,3))]
        result=fed.summarize(rows)
        self.assertEqual(result['mean_rmse_w'],2.);self.assertEqual(result['std_rmse_w'],1.)
        self.assertEqual(result['std_mae_w'],2.);self.assertIn('not an ensemble',result['aggregation'])

    def test_full_synthetic_evaluation_locks_before_test_arrays_and_preserves_inputs(self):
        before=cio.capture_inventory(self.root);original=ce.TestCohort
        def locked(*args,**kwargs):
            lock=read_json(self.output/'selection_lock.json')
            self.assertEqual([s['selected_block'] for s in lock['models']],[190,190,180])
            self.assertTrue((self.output/'manifest.json').is_file())
            self.assertFalse((self.output/'selection_lock.json').stat().st_mode&0o222)
            return original(*args,**kwargs)
        with (mock.patch('src.centralized_evaluate.TestCohort',side_effect=locked),
              mock.patch('torch.optim.SGD',side_effect=AssertionError('optimizer forbidden'))):
            result=ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu',plan_sha256='fixture')
        self.assertEqual(result['samples_per_seed'],12)
        self.assertEqual(cio.check_inventory(self.root,before)['mismatches'],[])
        self.assertFalse(result['test_based_model_selection'])
        for seed in ce.SEEDS:
            nodes=read_json(self.output/f'seed{seed}/per_node.json')
            self.assertEqual(len(nodes),3);self.assertEqual(sum(v['samples'] for v in nodes.values()),12)
        with mock.patch('src.centralized_evaluate.infer_test',side_effect=AssertionError('postflight inference')):
            verified=ce.verify_results(self.root,self.plan,self.data,self.output,plan_sha256='fixture')
        self.assertEqual(verified['status'],'verified_complete')
        self.assertFalse(verified['model_inference_executed'])
        self.assertEqual(verified['artifact_hashes_verified'],48)

    def test_existing_or_partial_output_is_never_overwritten(self):
        self.output.mkdir();(self.output/'keep').write_text('preserve')
        with self.assertRaises(FileExistsError):ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')
        self.assertEqual((self.output/'keep').read_text(),'preserve')

    def test_interruption_retains_lock_and_requires_fresh_retry(self):
        with mock.patch('src.centralized_evaluate.infer_test',side_effect=KeyboardInterrupt('fixture')):
            with self.assertRaises(KeyboardInterrupt):ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')
        self.assertTrue((self.output/'selection_lock.json').exists());self.assertTrue((self.output/'failed.json').exists())
        self.assertFalse((self.output/'completed.json').exists())
        with self.assertRaises(FileExistsError):ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')

    def test_completed_output_hash_corruption_is_rejected(self):
        ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')
        p=self.output/'seed43/metrics.json'
        with changed(p,b'{}'),self.assertRaises(ValueError):ce.verify_results(self.root,self.plan,self.data,self.output)

    def test_missing_artifact_entries_cannot_pass_postflight(self):
        ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')
        p=self.output/'completed.json';value=read_json(p);value['artifacts'].pop('seed42/predictions.npz')
        with changed(p,json.dumps(value).encode()),self.assertRaisesRegex(ValueError,'inventory'):
            ce.verify_results(self.root,self.plan,self.data,self.output)

    def test_semantically_better_late_checkpoint_cannot_replace_locked_selection(self):
        original=cio.load_blocks
        def altered(path):
            payload,records=original(path);records[-1]['validation']['rmse_w']=0.
            return payload,records
        with mock.patch('src.centralized_evaluate.cio.load_blocks',side_effect=altered),self.assertRaisesRegex(ValueError,'selection/budget'):
            ce.preflight(self.root,self.plan,self.data)

    def test_validation_or_duplicate_origins_cannot_enter_test_even_with_new_fixture_hash(self):
        for kind in ('duplicate','validation'):
            dest=Path(self.tempdir.name)/kind;shutil.copytree(self.data,dest)
            origins=np.load(dest/'origin_test.npy').copy()
            origins[1]=origins[0] if kind=='duplicate' else np.datetime64('2022-02-12T01:00','ns')
            np.save(dest/'origin_test.npy',origins,allow_pickle=False)
            contract=read_json(dest/'contract.json');contract['arrays']['origin_test.npy']['sha256']=sha256(dest/'origin_test.npy')
            (dest/'contract.json').write_text(json.dumps(contract))
            declaration={**self.plan['cohort'],'contract_sha256':sha256(dest/'contract.json')}
            with self.subTest(kind=kind),self.assertRaises(ValueError):ce.TestCohort(dest,declaration)

    def test_test_array_hash_tampering_is_rejected_before_output(self):
        p=self.data/'y_test.npy'
        with changed(p,p.read_bytes()+b'corrupt'),self.assertRaises(ValueError):
            ce.evaluate_locked(self.root,self.plan,self.data,self.output,device='cpu')
        self.assertFalse(self.output.exists())

    def test_prior_identity_mismatch_stops_comparison(self):
        altered=copy.deepcopy(self.prior);altered['window_identity']={'different':True}
        with self.assertRaises(ValueError):ce.paired_comparison(self.prior,altered)

    def test_embedding_order_mismatch_is_rejected(self):
        s=self.selections[0];m={**s.manifest,'clients':list(reversed(s.manifest['clients']))}
        with self.assertRaises(ValueError):ce.load_model(self.cohort,replace(s,manifest=m),torch.device('cpu'))

    def test_cli_rejects_synthetic_cohort_before_inference(self):
        p=self.root/'configs/experiments/v3_centralized_sgd/test_plan.json';p.write_text(json.dumps(self.plan))
        with mock.patch('src.centralized_evaluate.ROOT',self.root),mock.patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:ce.main(['--plan',str(p),'--data-dir',str(self.data),'--verify-only'])
        self.assertEqual(error.exception.code,1)


if __name__=='__main__':unittest.main()
