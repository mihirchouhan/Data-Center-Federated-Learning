"""Invented telemetry, prediction arrays and /tmp files only; no models or March I/O."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from src import b1_march_data_v1 as d
from src import b1_march_integrity_v1 as k
from src import b1_march_evaluate_v1 as e
from src import b1_bridge_scaling_v1 as scaling
from src import b1_preparation_io_v1 as io


def streams(start='2022-03-01',steps=144):
    t=np.datetime64(start,'ns')+np.arange(steps)*np.timedelta64(10,'m')
    stamps=np.stack([t,t+np.timedelta64(300,'s')],axis=1).ravel()
    values={'total_power':500+np.arange(steps)*.125,'ambient':20+np.arange(steps)*.001,
        'cpu_user':np.full(steps,10.),'cpu_system':np.full(steps,20.),'cpu_idle':np.full(steps,60.),
        'cpu_wio':np.full(steps,10.),'mem_free':np.full(steps,40.),'mem_total':np.full(steps,100.)}
    local={m:(stamps,np.repeat(v,2)) for m,v in values.items()}
    weather={m:(t+np.timedelta64(120,'s'),np.full(steps,v)) for m,v in [('temp',12.),('humidity',55.)]}
    return local,weather


def synth_scaler():
    provenance={'fit_split':'train','algorithm':scaling.ALGORITHM,'mapping_digest':d.bridge.MAPPING_DIGEST}
    return scaling.FrozenScaler(tuple([0.]*14),tuple([2.]*13+[1.]),tuple([2.]*13+[0.]),(600.,20.),(100.,2.),io.canonical(provenance))


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local,cls.weather=streams();cls.bins,_=d.build_node(cls.local,cls.weather)
        cls.positions,cls.targets,cls.reasons,_=d.audit.window_masks(cls.bins['features'],d.EDGES)
        cls.p=d.packet('3',cls.bins,cls.positions)
    def test_shapes_units_and_order(self):
        self.assertEqual(self.p['x'].shape,(18,36,14));np.testing.assert_allclose(self.p['x'][0,0,:9],[.5,10,20,60,10,.6,20,12,55])
        np.testing.assert_allclose(d.validate_packet(self.p)[0],[500+np.arange(36,42).mean()*.125,20+np.arange(36,42).mean()*.001])
    def test_original_ordinals(self):
        p=d.packet('979',self.bins,self.positions);self.assertTrue((p['x'][:,:,13]==978).all());self.assertTrue((self.p['x'][:,:,13]==2).all())
    def test_unknown_node(self):
        with self.assertRaises(ValueError):d.packet('0',self.bins,self.positions)
    def test_first_origin_six_hours(self):self.assertEqual(self.p['origins'][0],np.datetime64('2022-03-01T06:00','ns'))
    def test_exact_upper_boundary(self):
        a,w=streams('2022-03-31T17:00',42);b,_=d.build_node(a,w);r,_=d.records('3',b)
        self.assertEqual(len(r),1);self.assertEqual(r['origin_ns'][0]+6*d.WIDTH,d.HI)
    def test_history_boundary_crossing(self):
        with self.assertRaises(ValueError):d.packet('3',self.bins,np.array([30]))
    def test_target_boundary_crossing(self):
        with self.assertRaises(ValueError):d.packet('3',self.bins,np.array([4464]))
    def test_off_hour(self):
        with self.assertRaises(ValueError):d.packet('3',self.bins,np.array([37]))
    def test_missing_input_gap(self):
        a,w=streams();t,v=a['cpu_user'];v=v.copy();v[36:38]=np.nan;a['cpu_user']=(t,v)
        b,_=d.build_node(a,w);r,c=d.records('3',b);self.assertNotIn(d.LO+36*d.WIDTH,r['origin_ns']);self.assertGreater(c['incomplete_input_history'],0)
    def test_missing_target_never_filled(self):
        a,w=streams();t,v=a['ambient'];v=v.copy();v[72:74]=np.nan;a['ambient']=(t,v)
        b,_=d.build_node(a,w);r,c=d.records('3',b);self.assertNotIn(d.LO+36*d.WIDTH,r['origin_ns']);self.assertGreater(c['unobserved_joint_target'],0)
    def test_no_before_split_hold(self):
        t=np.array([d.LO-100_000_000_000],dtype='datetime64[ns]');b,_=d.qualified_bins(t,np.array([500.]),'total_power');self.assertTrue(np.isnan(b.values).all())
    def test_weather_strict_before_right_edge(self):
        t=np.array([d.LO+d.WIDTH],dtype='datetime64[ns]');b,_=d.qualified_bins(t,np.array([12.]),'temp');self.assertTrue(np.isnan(b.values[0]));self.assertEqual(b.values[1],12.)
    def test_weather_stale(self):
        t=np.array([d.LO],dtype='datetime64[ns]');b,_=d.qualified_bins(t,np.array([12.]),'temp');self.assertTrue(np.isnan(b.values[3]))
    def test_half_coverage(self):
        b,_=d.qualified_bins(np.array([d.LO],dtype='datetime64[ns]'),np.array([500.]),'total_power');self.assertEqual(b.values[0],500.);self.assertEqual(b.coverage[0],.5)
    def test_less_half_coverage(self):
        t=np.array([d.LO+400_000_000_000],dtype='datetime64[ns]');b,_=d.qualified_bins(t,np.array([500.]),'total_power');self.assertTrue(np.isnan(b.values[0]))
    def test_duplicate_invalid_last(self):
        t=np.array([d.LO,d.LO],dtype='datetime64[ns]');b,q=d.qualified_bins(t,np.array([500.,np.nan]),'total_power');self.assertTrue(np.isnan(b.values[0]));self.assertEqual(q['duplicate_rows'],1)
    def test_input_boolean_rejection(self):
        with self.assertRaises(ValueError):d.qualified_bins(np.array([d.LO],dtype='datetime64[ns]'),np.array([True]),'total_power')
    def test_masked_raw_rejection(self):
        with self.assertRaises(ValueError):d.qualified_bins(np.array([d.LO],dtype='datetime64[ns]'),np.ma.array([1.],mask=False),'total_power')
    def test_masked_packet_before_coercion(self):
        for field in ('future','target_observed','x'):
            p=copy.deepcopy(self.p);p[field]=np.ma.array(p[field],mask=False)
            with self.assertRaises(ValueError):d.validate_packet(p)
    def test_boolean_targets(self):
        p=copy.deepcopy(self.p);p['future']=np.ones(p['future'].shape,bool)
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_future_input_leakage(self):
        p=copy.deepcopy(self.p);p['source_latest'][0,-1,0]=p['origins'][0]
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_missing_history_time(self):
        p=copy.deepcopy(self.p);p['history_times'][0,1]=p['history_times'][0,0]
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_cross_node_history(self):
        p=copy.deepcopy(self.p);p['history_nodes'][0,1]=978
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_calendar_changed(self):
        p=copy.deepcopy(self.p);p['x'][0,0,9]=0.
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_low_target_coverage(self):
        p=copy.deepcopy(self.p);p['target_coverage'][0,0,0]=.49
        with self.assertRaises(ValueError):d.validate_packet(p)
    def test_memory_invalid(self):
        a,w=streams();t,v=a['mem_free'];a['mem_free']=(t,np.full_like(v,200.));b,_=d.build_node(a,w);r,_=d.records('3',b);self.assertEqual(len(r),0)
    def test_stable_sort_identity(self):
        a,w=streams();a={m:(t[::-1],v[::-1]) for m,(t,v) in a.items()};b,_=d.build_node(a,w);np.testing.assert_array_equal(d.records('3',b)[0],d.records('3',self.bins)[0])
    def test_future_does_not_change_input(self):
        a,w=streams();t,v=a['total_power'];v=v.copy();v[72:84]+=100.;a['total_power']=(t,v);b,_=d.build_node(a,w)
        p=d.packet('3',b,np.array([36]));np.testing.assert_array_equal(p['x'],self.p['x'][:1]);self.assertNotEqual(d.validate_packet(p)[0,0],self.targets[0,0])
    def test_reference_order_and_target_change(self):
        a,_=d.records('3',self.bins);expected={'windows':18,'nodes':1,'origins':18,'dates':1,'identity_sha256':d.identity_digest(np.full(18,'3'),a['origin_ns'].astype('datetime64[ns]')),'target_sha256':hashlib.sha256(np.column_stack((a['power_w'],a['temperature_c'])).tobytes()).hexdigest()}
        d.validate_reference(a,expected)
        for changed in (a[::-1].copy(),a.copy()):
            if changed[0]['origin_ns']==a[0]['origin_ns']:changed['power_w'][0]+=1
            with self.assertRaises(ValueError):d.validate_reference(changed,expected)
    def test_frozen_scaler_no_fit(self):
        scaler=synth_scaler();before=scaler.record()
        with mock.patch.object(scaling,'fit_training_scaler',side_effect=AssertionError('fitter called')):
            x=scaler.transform_inputs(self.p['x']);self.assertEqual(x.dtype,np.float32);self.assertTrue((x[:,:,13]==2).all());self.assertAlmostEqual(x[0,0,0],.25)
        self.assertEqual(before,scaler.record())
    def test_mask_policy_cannot_change(self):
        self.assertEqual(k.PROTOCOL['weather'],'enabled');self.assertFalse(k.PROTOCOL['input_fill']);self.assertFalse(k.PROTOCOL['target_fill'])
    def test_node_file_roundtrip(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'node.npz';np.savez(p,**{k:np.ascontiguousarray(v) for k,v in self.bins.items()});b=d.read_node(p)
            for k in b:np.testing.assert_array_equal(self.bins[k],b[k])
    def test_bad_node_header(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'node.npz';np.savez(p,features=np.zeros((1,13)))
            with self.assertRaises(ValueError):d.read_node(p)


class IntegrityMetricTests(unittest.TestCase):
    def arrays(self):
        return np.array([[10.,20.],[20.,21.],[30.,22.]]),np.array([[11.,18.],[18.,23.],[33.,20.]]),np.array(['3','3','979']),np.array(['2022-03-01T06','2022-03-01T07','2022-03-01T06'],dtype='datetime64[ns]')
    def test_independent_metrics(self):
        a,p,n,o=self.arrays();m=e.metrics(a,p,n,o);e.compare_metrics(m,e.independent_metrics(a,p,n,o));self.assertEqual(m['pooled']['mae'],[2.,2.]);self.assertNotEqual(m['macro_node']['mae'][0],m['pooled']['mae'][0])
    def test_duplicate_prediction_identity(self):
        a,p,n,o=self.arrays();o[1]=o[0]
        with self.assertRaises(ValueError):e.independent_metrics(a,p,n,o)
    def test_nonfinite_prediction(self):
        a,p,n,o=self.arrays();p[0,0]=np.inf
        with self.assertRaises(ValueError):e.independent_metrics(a,p,n,o)
    def test_boolean_or_masked_predictions(self):
        a,p,n,o=self.arrays()
        for bad in (np.ma.array(p,mask=False),np.ones(p.shape,bool)):
            with self.assertRaises(ValueError):e.metrics(a,bad,n,o)
    def test_sample_sd(self):
        a,p,n,o=self.arrays();scores={s:e.metrics(a,p+i,n,o) for i,s in enumerate((42,43,44))};m=e.aggregate(scores)
        np.testing.assert_array_equal(m['pooled']['mae']['sample_sd'],np.std([scores[s]['pooled']['mae'] for s in (42,43,44)],axis=0,ddof=1))
    def test_missing_seed(self):
        with self.assertRaises(ValueError):e.aggregate({42:{},43:{}})
    def test_source_hash_mismatch(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'f';p.write_text('invented');b=io.binding(p);p.write_text('changed')
            with self.assertRaises(ValueError):k.check(p,b)
    def test_occupied_output(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'out';p.mkdir()
            with self.assertRaises(FileExistsError):k.stage_directory(p)
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'out';p.symlink_to(Path(t))
            with self.assertRaises(ValueError):k.stage_directory(p)
    def test_missing_receipt(self):
        with self.assertRaises(PermissionError):e.check_receipt({}, {'required':True})
    def test_changed_receipt_bindings(self):
        expected={'operator_assertion':{'explicit_operator_authorization':True},'test_target_hash':'known'}
        with self.assertRaises(PermissionError):e.check_receipt({**expected,'test_target_hash':'changed'},expected)
    def test_operator_authorization_required(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'op.json';p.write_text(json.dumps({'operation':'test','explicit_operator_authorization':False,'uid':os.getuid(),'bindings':{},'authorization_text':'invented statement only','expires_at_unix':time.time()+60}))
            with self.assertRaises(PermissionError):k.statement(p,'test',{})
    def test_stale_authorization(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'op.json';p.write_text(json.dumps({'operation':'test','explicit_operator_authorization':True,'uid':os.getuid(),'bindings':{},'authorization_text':'invented statement only','expires_at_unix':time.time()-1}))
            with self.assertRaises(PermissionError):k.statement(p,'test',{})
    def test_numeric_true_is_not_operator_authorization(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'op.json';p.write_text(json.dumps({'operation':'test','explicit_operator_authorization':1,'uid':os.getuid(),'bindings':{},'authorization_text':'invented statement only','expires_at_unix':time.time()+60}))
            with self.assertRaises(PermissionError):k.statement(p,'test',{})
    def test_incomplete_staging_blocks_duplicate_launch(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            destination=Path(t)/'out';stage=k.stage_directory(destination)
            with self.assertRaises(FileExistsError):k.stage_directory(destination)
            self.assertTrue((stage/'owner.json').is_file())
    def test_prediction_file_wrong_shape(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'p.npy';np.save(p,np.zeros((3,3)))
            with self.assertRaises(ValueError):e._read_predictions(p,3)
    def test_no_test_model_before_receipt(self):
        with mock.patch.object(k,'locked_context',return_value=({'configuration':{},'dataset_manifest':{},'scaler_lock':{},'scaler_statistics':{},'selected_models':[]},None,None)),mock.patch.object(d,'MarchData') as data,mock.patch.object(k,'statement',side_effect=PermissionError('no approval')),mock.patch.object(e,'load_selected',side_effect=AssertionError('model initialized')) as model:
            data.return_value.manifest_binding={}
            with self.assertRaises(PermissionError):e.evaluate({'dataset_output':'not-opened','evaluation_output':'not-opened'},'/tmp/missing','/tmp/missing','cpu','/tmp/missing')
            model.assert_not_called()
    def test_checkpoint_bytes_checked_before_model_import(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as t:
            p=Path(t)/'model.pt';p.write_bytes(b'invented invalid checkpoint')
            model={'selected_model':{'path':'model.pt','bytes':len(p.read_bytes()),'sha256':'0'*64}}
            with mock.patch.object(k,'ROOT',Path(t)):
                with self.assertRaises(ValueError):e.load_selected(model,{},None,'cpu')
    def test_zero_coverage_nodes_retain_original_mapping(self):
        a,w=streams();a={m:(t,np.full_like(v,np.nan)) for m,(t,v) in a.items()};bins,_=d.build_node(a,w)
        r,reasons=d.records('979',bins);self.assertEqual(len(r),0);self.assertEqual(len(d.bridge.ORIGINAL_NAMES),979);self.assertEqual(sum(reasons[n] for n in ('insufficient_whole_history','target_crosses_split_end','unobserved_joint_target','incomplete_input_history','valid_windows')),reasons['candidate_hourly_origins'])
    def test_old_bridge_still_rejects_test(self):
        with self.assertRaises(PermissionError):d.bridge.accept_windows(split='test',mapping=(),source_evidence={},**{k:None for k in ('x','observed','future','target_observed','nodes','origins','history_times','target_times','history_nodes','target_nodes','source_earliest','source_latest','target_source_earliest','target_source_latest','history_coverage','target_coverage')})


class StreamingAndSavedEvaluationTests(unittest.TestCase):
    def test_synthetic_parquet_spool_and_source_preservation(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from dataclasses import replace
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);sources={};a,w=streams(steps=48)
            for metric in (*d.LOCAL_METRICS,'temp','humidity'):
                plugin='weather_pub' if metric in w else ('ipmi_pub' if metric in ('ambient','total_power') else 'ganglia_pub')
                name=f'data/march_2022_manual/extracted/year_month=22-03/plugin={plugin}/metric={metric}/a_0.parquet'
                p=root/name;p.parent.mkdir(parents=True)
                times,values=w[metric] if metric in w else a[metric]
                raw={'timestamp':pa.array(times,type=pa.timestamp('ns',tz='UTC')),'value':pa.array(values)}
                if metric not in w:raw['node']=pa.array(['3']*len(times))
                pq.write_table(pa.table(raw),p);sources[name]=io.binding(p)
            before={n:(io.binding(root/n),(root/n).stat().st_mtime_ns) for n in sources}
            limits=io.Limits(max_disk_bytes=32*1024**2,reserve_bytes=1024**2)
            quota={'uid':os.getuid(),'confirmed':True,'headroom_bytes':33*1024**2,'checked_at_unix':time.time(),'filesystem_device':root.stat().st_dev,'evidence_source':'invented /tmp fixture, not project quota'}
            attempt=io.Attempt(root/'attempt','synthetic',io.Budget(root/'attempt',limits,quota))
            try:
                q=d.spool_sources(root,sources,attempt,limits)
                local={m:d._read_spool(attempt,m,2,limits) for m in d.LOCAL_METRICS}
                weather={m:d._read_spool(attempt,m,0,limits) for m in w}
                b,_=d.build_node(local,weather);expected,_=d.build_node(a,w)
                for field in b:np.testing.assert_array_equal(b[field],expected[field])
                self.assertEqual(len(q),10);self.assertEqual(d._read_spool(attempt,'total_power',978,limits)[0].size,0)
                self.assertEqual(before,{n:(io.binding(root/n),(root/n).stat().st_mtime_ns) for n in sources})
            finally:attempt.close()
    def make_evaluation(self,root):
        a,w=streams();b,_=d.build_node(a,w);reference,_=d.records('3',b)
        data=type('InventedData',(),{'reference':reference,'assert_unchanged':lambda self:None})()
        actual=e._actual(reference);nodes=np.full(len(reference),'3');origins=reference['origin_ns'].astype('datetime64[ns]')
        with (root/'reference.npy').open('xb') as f:np.save(f,reference,allow_pickle=False)
        scores={}
        for i,s in enumerate((42,43,44)):
            pred=actual+np.array([float(i+1),.25*(i+1)])
            np.save(root/f'predictions_seed{s}.npy',pred)
            scores[s]=e.metrics(actual,pred,nodes,origins)
        io.exclusive(root/'metrics.json',io.canonical({str(k):v for k,v in scores.items()}))
        io.exclusive(root/'aggregate_metrics.json',io.canonical(e.aggregate(scores)));e.write_tables(root,scores)
        return data
    def test_independent_saved_prediction_verification_and_completion(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);data=self.make_evaluation(root);verification=e.verify_saved_predictions(root,data)
            io.exclusive(root/'verification.json',io.canonical(verification));plan={'synthetic_fixture':True};receipt={'synthetic_fixture_only':True}
            io.exclusive(root/'completed.json',io.canonical({'status':'complete','plan_sha256':k.digest(plan),'receipt_sha256':k.digest(receipt),'artifacts':io.inventory(root)}))
            result=e.verify_evaluation(root,plan,receipt,data);self.assertEqual(result['windows_per_seed'],18);self.assertFalse(result['model_forward'])
    def test_changed_saved_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);data=self.make_evaluation(root);p=root/'predictions_seed42.npy';a=np.load(p);a[0,0]+=1;np.save(p,a)
            with self.assertRaises(ValueError):e.verify_saved_predictions(root,data)
    def test_changed_saved_test_targets_are_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);data=self.make_evaluation(root);a=data.reference.copy();a['power_w'][0]+=1;np.save(root/'reference.npy',a)
            with self.assertRaises(ValueError):e.verify_saved_predictions(root,data)
    def test_duplicate_table_rows_are_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);data=self.make_evaluation(root);p=root/'seed_metrics.csv';text=p.read_text();p.write_text(text+text.splitlines()[1]+'\n')
            with self.assertRaises(ValueError):e.verify_saved_predictions(root,data)
    def test_incomplete_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);plan={'synthetic':True}
            io.exclusive(root/'manifest.json',io.canonical({'schema':'b1.march.prepared.bins.v1','status':'complete','plan':plan,'plan_sha256':k.digest(plan),'mapping':list(d.bridge.ORIGINAL_NAMES),'nodes':[]}))
            with self.assertRaises(ValueError):d.verify_dataset(root,plan)
    def test_reference_header_rejects_oversize_shape(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            p=Path(tmp)/'r.npy';np.save(p,np.zeros(1))
            with self.assertRaises(ValueError):d.read_reference(p)

if __name__=='__main__':unittest.main(verbosity=2)
