"""Synthetic arrays and /tmp only; no models, archive access or real scoring."""
import copy
from dataclasses import replace
import hashlib
import importlib
import io
import contextlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
from src.phase2_multioutput import contracts as c, evaluation as e, protocol as p, readiness

ROOT=Path(__file__).absolute().parents[2]
CONFIG='configs/experiments/phase2_multioutput/v1'

def fixture(history=36,split='train',nodes=('a','a','b'),origins=None):
    protocol,matrix,methods=p.load_bundle(ROOT,CONFIG)
    protocol=copy.deepcopy(protocol)
    protocol['splits']={'train':['2030-01-01T00:00:00Z','2030-01-15T00:00:00Z'],
                        'val':['2030-01-15T00:00:00Z','2030-01-22T00:00:00Z'],
                        'test':['2030-02-01T00:00:00Z','2030-02-08T00:00:00Z']}
    protocol['eligibility']={'minimum_windows':1,'minimum_days':1,'training_only':True}
    protocol['federation']['clients_per_round']=2
    day={'train':'2030-01-05','val':'2030-01-18','test':'2030-02-04'}[split]
    origins=np.array(origins or [day+'T00:00',day+'T01:00',day+'T00:00'],dtype='datetime64[ns]')
    nodes=np.array(nodes);n=len(nodes);index=np.array([('a','b').index(x) for x in nodes])
    hist=(origins.view('i8')[:,None]+np.arange(-history,0)*c.WIDTH_NS).astype('datetime64[ns]')
    future_t=(origins.view('i8')[:,None]+np.arange(6)*c.WIDTH_NS).astype('datetime64[ns]')
    x=np.zeros((n,history,14));x[:,:,0]=2.;x[:,:,1:5]=[30,10,55,5];x[:,:,5]=.5
    x[:,:,6:9]=[22,15,60];x[:,:,9:13]=[0,1,0,1];x[:,:,13]=index[:,None]
    future=np.broadcast_to([2.,22.],(n,6,2)).copy()
    source=np.broadcast_to(hist[:,:,None],(n,history,9)).copy()
    w=c.Windows(x,np.ones((n,history,13),dtype=bool),future,np.ones((n,6,2),dtype=bool),hist,future_t,
        np.broadcast_to(index[:,None],(n,history)).copy(),np.broadcast_to(index[:,None],(n,6)).copy(),
        source,source,nodes,origins,('a','b'),split,history)
    return w,protocol

class ContractTests(unittest.TestCase):
    def setUp(self):self.w,self.p=fixture()
    def test_three_histories_supported(self):
        for h in (6,36,144):
            w,p0=fixture(h);self.assertEqual(c.validate_windows(w,p0).shape,(3,2))
    def test_target_means_exact_units(self):
        f=np.stack([np.arange(1.,7.),np.arange(10.,16.)],axis=1)[None]
        np.testing.assert_array_equal(c.target_means(f,np.ones(f.shape,dtype=bool)),[[3500.,12.5]])
    def test_no_target_imputation(self):
        obs=self.w.target_observed.copy();obs[0,0,1]=False
        with self.assertRaisesRegex(ValueError,'target'):c.validate_windows(replace(self.w,target_observed=obs),self.p)
    def test_invalid_target_power(self):
        f=self.w.future.copy();f[0,0,0]=-1
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,future=f),self.p)
    def test_invalid_target_temperature(self):
        f=self.w.future.copy();f[0,0,1]=-300
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,future=f),self.p)
    def test_nan_target_rejected(self):
        f=self.w.future.copy();f[0,0,0]=np.nan
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,future=f),self.p)
    def test_missing_feature_rejected(self):
        obs=self.w.observed.copy();obs[0,1,3]=False
        with self.assertRaisesRegex(ValueError,'input'):c.validate_windows(replace(self.w,observed=obs),self.p)
    def test_nan_feature_rejected(self):
        x=self.w.x.copy();x[0,0,7]=np.nan
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,x=x),self.p)
    def test_weather_missing_not_expanded_by_mask(self):
        obs=self.w.observed.copy();obs[0,0,7]=False
        masked=c.mask_weather(self.w.x,'masked')
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,x=masked,observed=obs),self.p)
    def test_missing_time_gap_rejected(self):
        t=self.w.history_times.copy();t[0,3]+=np.timedelta64(10,'m')
        with self.assertRaisesRegex(ValueError,'gapped'):c.validate_windows(replace(self.w,history_times=t),self.p)
    def test_future_input_rejected(self):
        t=self.w.history_times.copy();t[0,-1]=self.w.origins[0]
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,history_times=t),self.p)
    def test_changed_target_interval_rejected(self):
        t=self.w.target_times.copy();t[0,0]+=np.timedelta64(10,'m')
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,target_times=t),self.p)
    def test_cross_node_history_rejected(self):
        n=self.w.history_nodes.copy();n[0,4]=1
        with self.assertRaisesRegex(ValueError,'node identities'):c.validate_windows(replace(self.w,history_nodes=n),self.p)
    def test_cross_node_future_rejected(self):
        n=self.w.target_nodes.copy();n[0,4]=1
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,target_nodes=n),self.p)
    def test_node_channel_not_free_numeric(self):
        x=self.w.x.copy();x[0,0,13]=.3
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,x=x),self.p)
    def test_timezone_missing_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,timezone='naive'),self.p)
    def test_datetime_precision_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,origins=self.w.origins.astype('datetime64[s]')),self.p)
    def test_feature_order_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,features=tuple(reversed(c.FEATURES))),self.p)
    def test_units_rejected(self):
        units=list(c.UNITS);units[0]='W'
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,units=tuple(units)),self.p)
    def test_invalid_percent_rejected(self):
        x=self.w.x.copy();x[0,0,1]=101
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,x=x),self.p)
    def test_invalid_memory_fraction(self):
        x=self.w.x.copy();x[0,0,5]=1.1
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,x=x),self.p)
    def test_future_sensor_contributor_rejected(self):
        t=self.w.source_latest.copy();t[0,0,0]=self.w.origins[0]
        with self.assertRaisesRegex(ValueError,'contributor'):c.validate_windows(replace(self.w,source_latest=t),self.p)
    def test_sensor_hold_across_split_rejected(self):
        t=self.w.source_earliest.copy();t[0,0,0]=np.datetime64('2029-12-31','ns')
        with self.assertRaisesRegex(ValueError,'contributor'):c.validate_windows(replace(self.w,source_earliest=t),self.p)
    def test_split_crossing_history_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,split='val'),self.p)
    def test_exact_split_edges_and_unused_interval(self):
        real,_,_=p.load_bundle(ROOT,CONFIG)
        lo,hi=c.bounds(real,'test')
        self.assertEqual(c.split_for_origin(lo+36*c.WIDTH_NS,36,real),'test')
        self.assertIsNone(c.split_for_origin(lo+30*c.WIDTH_NS,36,real))
        self.assertEqual(c.split_for_origin(hi-6*c.WIDTH_NS,144,real),'test')
        self.assertIsNone(c.split_for_origin(hi,36,real))
        self.assertIsNone(c.split_for_origin(c.utc_ns('2022-02-25T12:00:00Z'),36,real))
    def test_hourly_origins_only(self):
        real,_,_=p.load_bundle(ROOT,CONFIG)
        self.assertIsNone(c.split_for_origin(c.utc_ns('2022-03-03T00:10:00Z'),36,real))
    def test_duplicate_identity_rejected(self):
        o=self.w.origins.copy();o[1]=o[0]
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,origins=o),self.p)
    def test_mapping_order_changes_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,mapping=('b','a')),self.p)
    def test_training_only_eligibility(self):
        w,p0=fixture(split='val')
        with self.assertRaisesRegex(ValueError,'training-only'):c.eligibility(w,p0)
    def test_invalid_client_count_stops(self):
        val,_=fixture(split='val');test,_=fixture(split='test');pr=copy.deepcopy(self.p);pr['federation']['clients_per_round']=98
        with self.assertRaisesRegex(ValueError,'Insufficient'):c.validate_client_cohorts(self.w,val,test,pr)
    def test_valid_client_cohorts(self):
        val,_=fixture(split='val');test,_=fixture(split='test')
        self.assertEqual(c.validate_client_cohorts(self.w,val,test,self.p),{'a':2,'b':1})
    def test_identical_history_targets_and_persistence(self):
        batches=[fixture(h)[0] for h in (6,36,144)]
        for b in batches:
            c.assert_same_evaluation(batches[0],b,self.p)
            np.testing.assert_array_equal(e.persistence(b,self.p),[[2000,22]]*3)
    def test_changed_common_target_rejected(self):
        f=self.w.future.copy();f[0,0,1]+=1
        with self.assertRaisesRegex(ValueError,'targets'):c.assert_same_evaluation(self.w,replace(self.w,future=f),self.p)
    def test_equal_counts_different_origins_rejected(self):
        other,_=fixture(origins=['2030-01-05T02:00','2030-01-05T03:00','2030-01-05T02:00'])
        with self.assertRaisesRegex(ValueError,'cohort'):c.assert_same_evaluation(self.w,other,self.p)
    def test_intersection_indices_deterministic(self):
        batches=[fixture(h)[0] for h in (6,36,144)]
        for a in c.matched_indices(batches,self.p):np.testing.assert_array_equal(a,[0,1,2])
    def test_intersection_rejects_duplicate_histories(self):
        with self.assertRaises(ValueError):c.matched_indices([self.w,self.w],self.p)
    def test_weather_mask_preserves_arrays(self):
        before=self.w.x.copy();out=c.mask_weather(before,'masked')
        np.testing.assert_array_equal(out[:,:,[7,8]],0)
        keep=[j for j in range(14) if j not in (7,8)]
        np.testing.assert_array_equal(out[:,:,keep],before[:,:,keep]);np.testing.assert_array_equal(before,self.w.x)
        self.assertFalse(out.flags.writeable)
    def test_unsupported_history_rejected(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,history=12),self.p)
    def test_empty_cohort_stops(self):
        with self.assertRaises(ValueError):c.validate_windows(replace(self.w,nodes=np.array([],dtype='U1')),self.p)
    def scaler(self):
        return {'fit_split':'train','mapping':['a','b'],'fit_identity_sha256':c.identity_digest(self.w.nodes,self.w.origins),
            'reference_history_steps':36,'target_units':['W','degC'],'x_mean':[0]*14,'x_std':[1]*14,
            'y_mean':[2000,22],'y_std':[20,1],'original_y_std':[20,1]}
    def test_scaler_contract_no_fitting(self):self.assertTrue(c.validate_scaler(self.scaler(),self.w,self.p))
    def test_scaler_wrong_fit_population(self):
        s=self.scaler();s['fit_identity_sha256']='0'*64
        with self.assertRaises(ValueError):c.validate_scaler(s,self.w,self.p)
    def test_scaler_cannot_use_validation(self):
        val,_=fixture(split='val')
        with self.assertRaises(ValueError):c.validate_scaler(self.scaler(),val,self.p)
    def test_constant_target_requires_review(self):
        s=self.scaler();s['original_y_std'][0]=0
        with self.assertRaisesRegex(ValueError,'Constant'):c.validate_scaler(s,self.w,self.p)

class EvaluationTests(unittest.TestCase):
    def setUp(self):self.w,self.p=fixture()
    def test_pooled_vs_equal_node_metrics(self):
        actual=np.zeros((3,2));pred=np.array([[0,0],[0,0],[6,2]])
        m=e.metrics(actual,pred,self.w.nodes,self.w.origins)
        self.assertEqual(m['pooled']['mae'],[2.,2/3]);self.assertEqual(m['macro_node']['mae'],[3.,1.])
        self.assertAlmostEqual(m['pooled']['rmse'][0],np.sqrt(12))
    def test_nonfinite_predictions_rejected(self):
        pred=np.zeros((3,2));pred[0,0]=np.nan
        with self.assertRaises(ValueError):e.metrics(np.zeros((3,2)),pred,self.w.nodes,self.w.origins)
    def test_single_output_incompatible(self):
        with self.assertRaises(ValueError):e.metrics(np.zeros((3,1)),np.zeros((3,1)),self.w.nodes,self.w.origins)
    def test_no_prediction_dropping(self):
        with self.assertRaises(ValueError):e.metrics(np.zeros((3,2)),np.zeros((2,2)),self.w.nodes,self.w.origins)
    def test_sample_sd_all_three_seeds(self):
        records={s:e.metrics(np.zeros((3,2)),np.full((3,2),i+1),self.w.nodes,self.w.origins) for i,s in enumerate([42,43,44])}
        summary=e.seed_summary(records);np.testing.assert_allclose(summary['mae']['mean'],[2,2]);np.testing.assert_allclose(summary['mae']['sample_sd'],[1,1])
    def test_seed_cohort_mismatch(self):
        records={s:e.metrics(np.zeros((3,2)),np.ones((3,2)),self.w.nodes,self.w.origins) for s in [42,43,44]};records[43]['identity_sha256']='other'
        with self.assertRaises(ValueError):e.seed_summary(records)
    def test_no_best_seed_selection(self):
        with self.assertRaises(ValueError):e.seed_summary({42:{}})
    def test_selection_earliest_exact_joint_minimum(self):
        trace=[{'index':i,'split':'val','rmse':r} for i,r in [(0,[2,4]),(10,[1,2]),(20,[1,2])]]
        self.assertEqual(e.select_validation(trace,[2,4],[0,10,20]),{'index':10,'score':.25})
    def test_test_score_cannot_select_checkpoint(self):
        with self.assertRaises(ValueError):e.select_validation([{'index':0,'split':'test','rmse':[1,1]}],[1,1],[0])
    def test_missing_selection_candidate_rejected(self):
        with self.assertRaises(ValueError):e.select_validation([{'index':0,'split':'val','rmse':[1,1]}],[1,1],[0,10])
    def test_target_scales_cannot_mix_physical_units(self):
        self.assertEqual(e.joint_score([100,2],[100,2]),1.)
        with self.assertRaises(ValueError):e.joint_score([100,2],[100,0])
    def test_generalized_lag_mapping(self):
        for h in (6,36,144):
            m=e.lag_mapping(h);self.assertEqual(len(m),13*h+1);self.assertEqual(m[-1]['type'],'c')
            self.assertEqual(m[0]['minutes_before_origin'],10*h);self.assertEqual(m[-2]['minutes_before_origin'],10)
    def test_future_lock_rejects_test_selection(self):
        r={'fit_id':'f','selection_split':'test','bindings':{},'models':[{'sha256':'a'*64}]}
        with self.assertRaises(ValueError):e.verify_selection_record(r,'f',{})
    def test_future_lock_rejects_binding_drift(self):
        r={'fit_id':'f','selection_split':'val','bindings':{'cohort':'a'},'models':[{'sha256':'a'*64}]}
        with self.assertRaises(ValueError):e.verify_selection_record(r,'f',{'cohort':'b'})
    def test_validation_common_evaluator_aligns_baseline(self):
        w,p0=fixture(split='val');actual=c.validate_windows(w,p0)
        result=e.evaluate_validation_arrays(w,actual,w.nodes,w.origins,p0)
        self.assertEqual(result['model']['pooled']['mae'],[0.,0.])
        self.assertEqual(result['model']['identity_sha256'],result['persistence']['identity_sha256'])
    def test_validation_evaluator_rejects_reordered_predictions(self):
        w,p0=fixture(split='val')
        with self.assertRaises(ValueError):e.evaluate_validation_arrays(w,np.zeros((3,2)),w.nodes[::-1],w.origins[::-1],p0)
    def test_final_test_adapter_not_silently_enabled(self):
        w,p0=fixture(split='test')
        with self.assertRaises(PermissionError):e.evaluate_validation_arrays(w,np.zeros((3,2)),w.nodes,w.origins,p0)
    def test_invalid_checkpoint_hash_rejected(self):
        r={'fit_id':'f','selection_split':'val','bindings':{},'models':[{'sha256':'z'*64}]}
        with self.assertRaises(ValueError):e.verify_selection_record(r,'f',{})

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.p,self.m,self.methods=p.load_bundle(ROOT,CONFIG)
    def test_explicit_matrix_counts(self):self.assertEqual(len(self.m['fits']),36);self.assertEqual(len(self.m['baselines']),4)
    def test_missing_fit_rejected(self):
        m=copy.deepcopy(self.m);m['fits'].pop()
        with self.assertRaises(ValueError):p.validate_matrix(m,self.methods)
    def test_duplicate_fit_rejected(self):
        m=copy.deepcopy(self.m);m['fits'][-1]=m['fits'][0]
        with self.assertRaises(ValueError):p.validate_matrix(m,self.methods)
    def test_existing_output_paths_not_reused(self):
        for row in self.m['fits']+self.m['baselines']:self.assertFalse((ROOT/row['output_relative']).exists())
    def test_march_counts_cannot_be_invented(self):
        pr=copy.deepcopy(self.p);pr['march_counts']={'test':6590}
        with self.assertRaises(ValueError):p.validate_protocol(pr)
    def test_training_disabled(self):
        pr=copy.deepcopy(self.p);pr['execution_enabled']=True
        with self.assertRaises(ValueError):p.validate_protocol(pr)
    def test_split_cannot_move_silently(self):
        pr=copy.deepcopy(self.p);pr['splits']['train'][1]='2022-02-01T00:00:00Z'
        with self.assertRaises(ValueError):p.validate_protocol(pr)
    def test_xgboost_single_joint_iteration(self):self.assertEqual(self.methods['xgboost_enabled']['multioutput_strategy'],'two_scalar_boosters_shared_selected_iteration')
    def test_actual_hash_content_verification(self):
        with tempfile.TemporaryDirectory(prefix='m100-p20-hash-') as td:
            r=Path(td);(r/'x').write_bytes(b'a')
            manifest={'schema':'phase2.0.design.v1','artifacts':{'x':{'bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}},'historical_dependencies':{'x':{'bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}},'self_hash_policy':'external'}
            self.assertEqual(p.verify_design(r,manifest),2)
            (r/'x').write_bytes(b'b')
            with self.assertRaises(ValueError):p.verify_design(r,manifest)
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):p.safe_read(ROOT,'../other')
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory(prefix='m100-p20-link-') as td:
            r=Path(td);(r/'a').write_bytes(b'a');(r/'b').symlink_to(r/'a')
            with self.assertRaises(ValueError):p.safe_read(r,'b')
    def test_data_readiness_remains_blocked(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:self.assertEqual(readiness.main(['--stage','data']),2)
        self.assertFalse(json.loads(out.getvalue())['data_frozen'])
    def test_protocol_command_offline_no_writes(self):
        with mock.patch.object(socket.socket,'connect',side_effect=AssertionError('network')),mock.patch.object(Path,'write_text',side_effect=AssertionError('write')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(readiness.main(['--stage','protocol']),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'protocol_verified_data_not_frozen')
    def test_no_model_framework_imported(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('xgboost',sys.modules)
    def test_archive_not_read_as_evidence(self):
        roles={k:'not_a_manifest.tar' for k in ['acquisition_receipt','coverage_report','schema_units_report','cohort_manifest','mapping','scalers','provenance_acceptance','semantic_window_verification']}
        evidence={'schema':'phase2.0.data_evidence.v1','protocol_digest':p.digest(self.p),'matrix_digest':p.digest(self.m),'roles':roles,'artifacts':{'not_a_manifest.tar':{'bytes':1,'sha256':'a'*64}}}
        with mock.patch.object(readiness,'safe_read',side_effect=AssertionError('archive read')):
            with self.assertRaisesRegex(ValueError,'metadata only'):readiness.verify_evidence_hashes(ROOT,evidence,self.p,self.m)
    def test_training_stage_not_present(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:readiness.main(['--stage','fit'])
        self.assertEqual(error.exception.code,2)

if __name__=='__main__':unittest.main(verbosity=2)
