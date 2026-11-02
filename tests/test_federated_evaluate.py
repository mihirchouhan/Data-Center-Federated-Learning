"""Synthetic-only final-evaluator tests; no real held-out inference is authorized."""
from collections import Counter
from dataclasses import replace
import hashlib
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
from test_federated_data import fixture
from src.experiment_data import immutable_json,read_json,sha256
from src.federated_io import commit_round,digest_json,environment,source_inventory
from src.federated_rounds import DEFAULTS,configure_runtime,initialize_model,sample_clients,validation_due
from src.federated_train import GlobalModelState
from src.federated_evaluate import (FrozenTestCohort,TestPredictions,comparison_metrics,evaluate_locked,
    evaluate_predictions,execution_sources,identity_record,improvement_percent,infer_test,load_frozen_model,
    main,preflight,summarize,validate_predictions,verify_header,verify_selection)


def synthetic_data(path):
    digest,arrays,meta=fixture(path)
    inventory={}
    for name,value in arrays.items():
        for split in ('train','val','test'):
            if name.endswith('_'+split):
                nodes=arrays['node_'+split]
                mapping={name:i for i,name in enumerate(arrays['node_names'])}
                order=np.argsort([mapping[n] for n in nodes],kind='stable')
                value=value[order]
        with (path/(name+'.npy')).open('wb') as f:np.save(f,value,allow_pickle=False)
        inventory[name+'.npy']={'sha256':sha256(path/(name+'.npy')),'shape':list(value.shape),'dtype':str(value.dtype)}
    contract=read_json(path/'contract.json')
    contract.update(arrays=inventory,history_steps=6,horizon_steps=6,weather_mask_indices=[7,8])
    (path/'contract.json').write_text(json.dumps(contract))
    return {'contract_sha256':sha256(path/'contract.json'),'expected_windows':12,'expected_nodes':3}


def synthetic_project(root):
    """Fabricated 200-round receipts/state files, never optimizer steps or fitting.

    The small cohort and controlled validation scores exist only in a temporary
    fixture. The public CLI requires the original real M100 contract and sizes.
    """
    root.mkdir()
    cohort=synthetic_data(root/'frozen')
    sources=execution_sources(ROOT)
    for name in sources:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,path)
    data=FrozenTestCohort(root/'frozen',cohort)
    entries=[]
    for seed in (42,43,44):
        config={**DEFAULTS,'seed':seed,'device':'cpu','clients_per_round':2,
                'frozen_contract_sha256':cohort['contract_sha256']}
        configure_runtime(config)
        config_file=f'configs/experiments/v2_fedavg/gru_masked_seed{seed}.json'
        path=root/config_file;path.parent.mkdir(parents=True,exist_ok=True);immutable_json(path,config)
        dirname='gru_masked_seed42b' if seed==42 else f'gru_masked_seed{seed}'
        relative='output/experiments/v2_fedavg/main/'+dirname
        run=root/relative;run.mkdir(parents=True);(run/'rounds').mkdir();(run/'selected').mkdir();(run/'events').mkdir()
        manifest={'config':config,'config_digest':digest_json(config),
            'frozen_contract_sha256':cohort['contract_sha256'],'source_hashes':{name:sources[name] for name in source_inventory()},
            'environment':environment('cpu'),'training_windows':36,'validation_windows':12,
            'clients':[{'id':str(name),'count':12,'node_index':i,'days':12,'row_ids_sha256':'fixture'} for i,name in enumerate(data.names)]}
        immutable_json(run/'manifest.json',manifest)
        for name in manifest['source_hashes']:
            path=run/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,path)
        model=initialize_model(data,config);initial=GlobalModelState(model).state_dict()
        previous=None;best_state=initial;best_round=0;best_score=300.;examples=steps=0
        counts={str(n):12 for n in data.names};names=tuple(counts)
        for r in range(201):
            if r:
                with torch.no_grad():model.head.bias.fill_(.01*(seed-41))
            snapshot=GlobalModelState(model,round_id=r)
            chosen=list(sample_clients(names,2,seed,r)) if r else []
            clients=[{'client_id':n,'training_windows':12,'examples_processed':12,'optimizer_steps':1,
                      'initial_fingerprint':previous} for n in chosen]
            examples+=24 if r else 0;steps+=2 if r else 0
            validation=None
            if validation_due(r,config):
                validation={'samples':12,'mae_w':200.-r/2,'rmse_w':300.-r,
                            'persistence_mae_w':200.,'persistence_rmse_w':300.,'unit':'W','split':'val',
                            'evaluated_nodes':3,'finite_predictions':True,'predictions_sha256':'synthetic-validation'}
                best_state=snapshot.state_dict();best_round=r;best_score=validation['rmse_w']
            payload={'round':r,'state':snapshot.state_dict(),'best_state':best_state,'best_round':best_round,
                     'best_rmse_w':best_score,'examples_processed':examples,'optimizer_steps':steps,'manifest_digest':digest_json(manifest)}
            record={'round':r,'selected_clients':chosen,'clients':clients,'validation':validation,
                    'global_fingerprint':snapshot.fingerprint,
                    'previous_receipt_sha256':sha256(run/'rounds'/f'{r-1:04d}'/'receipt.json') if r else None}
            commit_round(run,payload,record);previous=snapshot.fingerprint
        exported={'state':payload['best_state'],'selected_round':200,'validation_rmse_w':100.,'manifest_digest':digest_json(manifest)}
        torch.save(exported,run/'selected/model.pt')
        immutable_json(run/'selected/receipt.json',{'model_sha256':sha256(run/'selected/model.pt'),
                       'manifest_sha256':sha256(run/'manifest.json'),'selected_round':200,'validation_rmse_w':100.,
                       'final_round_receipt_sha256':sha256(run/'rounds/0200/receipt.json')})
        event={'status':'complete','completed_round':200,'selected_round':200,'selected_validation_rmse_w':100.,
               'examples_processed':examples,'optimizer_steps':steps,
               'export_reload_check':{'exact_predictions_equal':True,'finite_parameters':True,'validation':validation}}
        immutable_json(run/'events/completed.json',event)
        entries.append({'seed':seed,'run_dir':relative,'selected_round':200,'manifest_sha256':sha256(run/'manifest.json'),
                        'selected_receipt_sha256':sha256(run/'selected/receipt.json'),'model_sha256':sha256(run/'selected/model.pt'),
                        'config_file':config_file,'config_file_sha256':sha256(root/config_file),
                        'completion_event':'events/completed.json','completion_event_sha256':sha256(run/'events/completed.json')})
    return {'version':1,'kind':'frozen_fedavg_final_test','cohort':cohort,'execution_sources':sources,'runs':entries}


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master=tempfile.TemporaryDirectory()
        cls.project=Path(cls.master.name)/'project'
        cls.plan=synthetic_project(cls.project)
        cls.cohort=FrozenTestCohort(cls.project/'frozen',cls.plan['cohort'])
        cls.selections=preflight(cls.project,cls.plan,cls.project/'frozen')

    @classmethod
    def tearDownClass(cls):cls.master.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.output=Path(self.temp.name)/'evaluation'
        self.env=mock.patch.dict(os.environ,{'MPLCONFIGDIR':str(Path(self.temp.name)/'.matplotlib')})
        self.env.start()

    def tearDown(self):self.env.stop();self.temp.cleanup()

    def prediction(self,offset=None):
        x,y,origins,nodes=self.cohort.split('test')
        values=np.asarray(y[:,0],dtype=np.float64)
        if offset is not None:values=values+np.asarray(offset)
        return TestPredictions(values,nodes.copy(),origins.copy())

    def test_known_errors_inverse_watts_and_common_persistence(self):
        offsets=np.tile([0.,1.,-2.,3.],3)
        metrics=evaluate_predictions(self.cohort,self.prediction(offsets),self.output)
        self.assertAlmostEqual(metrics['mae_w'],300.)
        self.assertAlmostEqual(metrics['rmse_w'],np.sqrt(3.5)*200)
        with np.load(self.output/'predictions.npz',allow_pickle=False) as result:
            x,y,origin,node=self.cohort.split('test')
            actual=(y[:,0].astype(np.float64)*.2+.5)*1000
            persistence=(x[:,-1,0].astype(np.float64)*.4+.6)*1000
            np.testing.assert_array_equal(result['targets_w'],actual)
            np.testing.assert_array_equal(result['persistence_w'],persistence)
            np.testing.assert_array_equal(result['origin'],origin)
            np.testing.assert_array_equal(result['node'],node)
            np.testing.assert_array_equal(result['predictions_w'],actual+offsets*200)
            self.assertAlmostEqual(metrics['persistence_rmse_w'],np.sqrt(np.mean((persistence-actual)**2)))
        self.assertEqual(metrics['samples'],12)

    def test_per_node_counts_metrics_keep_even_poor_nodes(self):
        offsets=np.repeat([0.,1.,100.],4)
        metrics=evaluate_predictions(self.cohort,self.prediction(offsets),self.output)
        per_node=read_json(self.output/'per_node_comparison.json')
        self.assertEqual(set(per_node),set(self.cohort.node_names))
        for node,expected in zip(self.cohort.node_names,[0.,200.,20000.]):
            self.assertEqual(per_node[node]['samples'],4)
            self.assertAlmostEqual(per_node[node]['rmse_w'],expected)
            self.assertAlmostEqual(per_node[node]['mae_w'],expected)
        self.assertEqual(metrics['nodes_beating_persistence_rmse'],sum(v['rmse_w']<v['persistence_rmse_w'] for v in per_node.values()))
        self.assertEqual(metrics['nodes_beating_persistence_mae'],sum(v['mae_w']<v['persistence_mae_w'] for v in per_node.values()))

    def test_percentage_sign_and_zero_baseline_are_explicit(self):
        self.assertAlmostEqual(improvement_percent(8.,10.),20.)
        self.assertAlmostEqual(improvement_percent(12.,10.),-20.)
        self.assertEqual(improvement_percent(0.,0.),0.)
        self.assertIsNone(improvement_percent(1.,0.))
        for values in ((np.nan,1.),(1.,np.inf),(-1.,1.)):
            with self.assertRaises(ValueError):improvement_percent(*values)

    def test_missing_duplicate_reordered_and_unknown_identities_rejected(self):
        p=self.prediction()
        for changed in (replace(p,scaled=p.scaled[:-1],nodes=p.nodes[:-1],origins=p.origins[:-1]),
                        replace(p,nodes=p.nodes[::-1],origins=p.origins[::-1]),
                        replace(p,nodes=np.full(12,'absent'))):
            with self.subTest(changed=changed.nodes.shape),self.assertRaises(ValueError):
                validate_predictions(self.cohort,changed)
        nodes=p.nodes.copy();origins=p.origins.copy();nodes[1]=nodes[0];origins[1]=origins[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):identity_record(nodes,origins,12)
        with self.assertRaises(ValueError):identity_record(p.nodes,p.origins.astype('datetime64[s]'),12)
        with self.assertRaises(ValueError):identity_record(p.nodes,np.full(12,np.datetime64('NaT','ns')),12)

    def test_invalid_prediction_lengths_shapes_and_finiteness_rejected_before_output(self):
        p=self.prediction()
        for values in (np.zeros(11),np.zeros((12,1)),np.full(12,np.nan),np.full(12,np.inf)):
            with self.subTest(shape=values.shape),self.assertRaises(ValueError):
                evaluate_predictions(self.cohort,replace(p,scaled=values),self.output)
            self.assertFalse(self.output.exists())

    def test_validation_cannot_be_requested_or_mislabeled_as_test(self):
        with self.assertRaises(PermissionError):self.cohort.split('val')
        with self.assertRaises(PermissionError):self.cohort.split('train')
        with self.assertRaisesRegex(ValueError,'validation is forbidden'):
            validate_predictions(self.cohort,replace(self.prediction(),split='val'))
        x,y,origins,nodes=np.load(self.project/'frozen/X_val.npy'),np.load(self.project/'frozen/y_val.npy'),np.load(self.project/'frozen/origin_val.npy'),np.load(self.project/'frozen/node_val.npy')
        with self.assertRaisesRegex(ValueError,'identities/order differ'):
            validate_predictions(self.cohort,TestPredictions(y[:,0],nodes,origins))

    def test_checkpoint_loading_and_inference_preserve_parameters_and_frozen_arrays(self):
        selected=self.selections[0]
        before={k:v.clone() for k,v in selected.exported['state'].items()}
        files={p.name:sha256(p) for p in (self.project/'frozen').iterdir()}
        model=load_frozen_model(self.cohort,selected,torch.device('cpu'))
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,before[key]))
        prediction=infer_test(self.cohort,selected,torch.device('cpu'))
        self.assertEqual(validate_predictions(self.cohort,prediction),self.cohort.identity)
        for key,value in selected.exported['state'].items():self.assertTrue(torch.equal(value,before[key]))
        self.assertEqual(files,{p.name:sha256(p) for p in (self.project/'frozen').iterdir()})

    def test_weather_masking_keeps_shape_node_ids_and_all_other_features(self):
        selected=self.selections[0];model=load_frozen_model(self.cohort,selected,torch.device('cpu'))
        seen=[]
        handle=model.register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().cpu().numpy().copy()))
        with mock.patch('src.federated_evaluate.load_frozen_model',return_value=model):infer_test(self.cohort,selected,torch.device('cpu'))
        handle.remove();actual=np.concatenate(seen);original=self.cohort.split('test')[0]
        self.assertEqual(actual.shape,(12,6,14))
        np.testing.assert_array_equal(actual[:,:,7:9],0.)
        keep=[j for j in range(14) if j not in (7,8)]
        np.testing.assert_array_equal(actual[:,:,keep],original[:,:,keep])

    def test_invalid_checkpoint_shape_dtype_fixed_buffer_and_nonfinite_rejected(self):
        selected=self.selections[0]
        for kind in ('shape','dtype','fixed','nan'):
            state={k:v.clone() for k,v in selected.exported['state'].items()}
            if kind=='shape':state['head.bias']=torch.zeros(2)
            elif kind=='dtype':state['head.bias']=state['head.bias'].double()
            elif kind=='fixed':state['baseline_coefficients']+=1
            else:state['head.bias'][:]=float('nan')
            changed=replace(selected,exported={**selected.exported,'state':state})
            with self.subTest(kind=kind),self.assertRaises(ValueError):load_frozen_model(self.cohort,changed,torch.device('cpu'))

    def test_node_embedding_mapping_cannot_be_permuted(self):
        selected=self.selections[0]
        manifest={**selected.manifest,'clients':list(reversed(selected.manifest['clients']))}
        with self.assertRaisesRegex(ValueError,'node-index mapping differs'):
            load_frozen_model(self.cohort,replace(selected,manifest=manifest),torch.device('cpu'))

    def test_preflight_never_opens_test_examples_or_infers(self):
        with (mock.patch('src.federated_evaluate.FrozenTestCohort',side_effect=AssertionError('examples loaded')),
              mock.patch('src.federated_evaluate.infer_test',side_effect=AssertionError('inference'))):
            result=preflight(self.project,self.plan,self.project/'frozen')
        self.assertEqual([s.entry['seed'] for s in result],[42,43,44])
        self.assertTrue(result[0].entry['run_dir'].endswith('gru_masked_seed42b'))

    def test_preflight_rejects_hash_source_count_and_seed_substitution(self):
        for kind in ('model','config','source','count','seed','path','event'):
            plan=json.loads(json.dumps(self.plan))
            if kind=='model':plan['runs'][0]['model_sha256']='0'*64
            elif kind=='config':plan['runs'][0]['config_file_sha256']='0'*64
            elif kind=='source':plan['execution_sources']['src/model.py']='0'*64
            elif kind=='count':plan['cohort']['expected_windows']=13
            elif kind=='seed':plan['runs'][1]['seed']=42
            elif kind=='path':plan['runs'][0]['run_dir']='output/experiments/v2_fedavg/main/gru_masked_seed42'
            else:plan['runs'][0]['completion_event_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises((ValueError,OSError)):preflight(self.project,plan,self.project/'frozen')

    def test_model_identity_mismatch_prevents_any_test_prediction(self):
        plan=json.loads(json.dumps(self.plan));plan['runs'][0]['model_sha256']='0'*64
        with mock.patch('src.federated_evaluate.infer_test',side_effect=AssertionError('premature prediction')):
            with self.assertRaises(ValueError):evaluate_locked(self.project,plan,self.project/'frozen',self.output,device='cpu')
        self.assertFalse(self.output.exists())

    def test_complete_synthetic_evaluation_locks_all_models_first_and_preserves_inputs(self):
        before={str(p.relative_to(self.project)):sha256(p) for p in self.project.rglob('*') if p.is_file()}
        original=FrozenTestCohort
        def locked_first(*args,**kwargs):
            lock=read_json(self.output/'selection_lock.json')
            self.assertEqual([r['seed'] for r in lock['models']],[42,43,44])
            self.assertEqual(lock['models'][0]['model_sha256'],self.plan['runs'][0]['model_sha256'])
            return original(*args,**kwargs)
        with (mock.patch('src.federated_evaluate.FrozenTestCohort',side_effect=locked_first),
              mock.patch('torch.optim.SGD',side_effect=AssertionError('training optimizer used'))):
            result=evaluate_locked(self.project,self.plan,self.project/'frozen',self.output,device='cpu')
        self.assertEqual(result['samples_per_seed'],12)
        self.assertFalse(result['test_based_model_selection'])
        self.assertEqual(result['window_identity'],self.cohort.identity)
        self.assertEqual(before,{str(p.relative_to(self.project)):sha256(p) for p in self.project.rglob('*') if p.is_file()})
        receipt=read_json(self.output/'completed.json')
        for name,expected in receipt['artifacts'].items():self.assertEqual(sha256(self.output/name),expected)
        for seed in (42,43,44):
            per_node=read_json(self.output/f'seed{seed}/per_node_comparison.json')
            self.assertEqual(set(per_node),set(self.cohort.node_names))
            self.assertEqual(sum(row['samples'] for row in per_node.values()),12)
        with self.assertRaises(FileExistsError):evaluate_locked(self.project,self.plan,self.project/'frozen',self.output,device='cpu')

    def test_summary_sample_sd_and_equal_window_baselines_are_enforced(self):
        rows=[]
        for seed,rmse,mae in ((42,1.,2.),(43,2.,4.),(44,3.,6.)):
            rows.append({'seed':seed,'rmse_w':rmse,'mae_w':mae,'persistence_rmse_w':4.,'persistence_mae_w':5.,
                         'samples':12,'window_identity':self.cohort.identity})
        result=summarize(rows)
        self.assertEqual(result['mean_rmse_w'],2.)
        self.assertEqual(result['std_rmse_w'],1.)
        self.assertEqual(result['mean_mae_w'],4.)
        self.assertEqual(result['std_mae_w'],2.)
        for changed in (rows[:-1],list(reversed(rows)),[rows[0],rows[1],{**rows[2],'window_identity':{'different':True}}],
                        [rows[0],rows[1],{**rows[2],'persistence_rmse_w':5.}]):
            with self.assertRaises(ValueError):summarize(changed)

    def test_inference_failure_preserves_partial_lock_without_completed_receipt(self):
        with mock.patch('src.federated_evaluate.infer_test',side_effect=ValueError('nonfinite prediction')):
            with self.assertRaisesRegex(ValueError,'nonfinite prediction'):
                evaluate_locked(self.project,self.plan,self.project/'frozen',self.output,device='cpu')
        self.assertTrue((self.output/'selection_lock.json').is_file())
        self.assertTrue((self.output/'failed.json').is_file())
        self.assertFalse((self.output/'completed.json').exists())
        with self.assertRaises(FileExistsError):evaluate_locked(self.project,self.plan,self.project/'frozen',self.output,device='cpu')

    def test_tampered_test_array_is_detected_without_predictions(self):
        changed=Path(self.temp.name)/'frozen';shutil.copytree(self.project/'frozen',changed)
        with (changed/'y_test.npy').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen array changed'):FrozenTestCohort(changed,self.plan['cohort'])

    def test_duplicate_or_validation_origin_inside_test_rejected_even_with_matching_fixture_hashes(self):
        for kind in ('duplicate','validation'):
            path=Path(self.temp.name)/kind;shutil.copytree(self.project/'frozen',path)
            origins=np.load(path/'origin_test.npy').copy()
            origins[1]=origins[0] if kind=='duplicate' else np.datetime64('2022-02-12T01:00','ns')
            np.save(path/'origin_test.npy',origins,allow_pickle=False)
            contract=read_json(path/'contract.json')
            contract['arrays']['origin_test.npy']['sha256']=sha256(path/'origin_test.npy')
            (path/'contract.json').write_text(json.dumps(contract))
            declaration={**self.plan['cohort'],'contract_sha256':sha256(path/'contract.json')}
            with self.subTest(kind=kind),self.assertRaises(ValueError):FrozenTestCohort(path,declaration)

    def test_inference_that_changes_parameters_is_rejected(self):
        import src.federated_evaluate as evaluator
        original=evaluator.neural_predict
        def mutate(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            with torch.no_grad():model.head.bias.add_(1)
            return result
        with mock.patch('src.federated_evaluate.neural_predict',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed frozen model'):
                infer_test(self.cohort,self.selections[0],torch.device('cpu'))

    def test_cli_rejects_wrong_benchmark_before_evaluation(self):
        path=self.project/'configs/experiments/v2_fedavg/test_plan.json'
        path.write_text(json.dumps(self.plan))
        with mock.patch('src.federated_evaluate.ROOT',self.project),mock.patch('sys.stderr',new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:main(['--plan',str(path),'--data-dir',str(self.project/'frozen'),'--verify-only'])
        self.assertEqual(error.exception.code,1)


@unittest.skipUnless(os.environ.get('M100_FEDAVG_EVALUATOR_BASELINE'),'Set M100_FEDAVG_EVALUATOR_BASELINE for preservation verification')
class EvaluatorPreservationTests(unittest.TestCase):
    def test_all_pre_evaluator_files_remain_byte_identical(self):
        baseline=read_json(os.environ['M100_FEDAVG_EVALUATOR_BASELINE'])
        root=Path(baseline['project_root'])
        self.assertGreaterEqual(len(baseline['protected_files']),2114)
        for name,expected in baseline['protected_files'].items():
            with self.subTest(path=name):self.assertEqual(sha256(root/name),expected)


if __name__=='__main__':unittest.main()
