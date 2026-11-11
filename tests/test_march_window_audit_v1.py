"""Synthetic-only no-model tests; all file fixtures live in TemporaryDirectory."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from src import march_window_audit_v1 as a
from src.preprocess import _time_weighted_bins
from src.phase2_multioutput.contracts import mask_weather, split_for_origin


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.start=a.utc_ns('2022-03-01T00:00:00Z');self.W=a.WIDTH
        self.edges=self.start+np.arange(73)*self.W
        self.x=np.ones((72,13));self.x[:,0]=.2;self.x[:,5]=.5;self.x[:,6:8]=25;self.x[:,8]=50
    def bins(self,seconds,values,metric='total_power',start=None):
        start=self.start if start is None else start
        t=np.array(seconds,dtype=np.int64)*a.NS_SECOND+start
        t,v,q=a.canonical_rows(t,values,metric)
        return a.bin_split(t,v,metric,np.array([start,start+self.W])),q
    def test_exact_half_coverage_valid(self):
        (v,c),q=self.bins([0],[200]);self.assertEqual(v[0],200);self.assertEqual(c[0],.5)
    def test_less_than_half_invalid(self):
        (v,c),q=self.bins([301],[200]);self.assertTrue(np.isnan(v[0]));self.assertLess(c[0],.5)
    def test_hold_cap_not_implicit_long_fill(self):
        t,v,_=a.canonical_rows([self.start],[200],'total_power');b,c=a.bin_split(t,v,'total_power',self.edges)
        self.assertEqual(np.isfinite(b).sum(),1);self.assertEqual(c[0],.5)
    def test_invalid_record_breaks_hold(self):
        (v,c),q=self.bins([0,100],[200,np.nan]);self.assertTrue(np.isnan(v[0]))
    def test_negative_power_is_invalid(self):
        (v,c),q=self.bins([0],[-1]);self.assertTrue(np.isnan(v[0]));self.assertEqual(q['out_of_range_after_dedup'],1)
    def test_keep_last_null_duplicate(self):
        (v,c),q=self.bins([0,0],[200,np.nan]);self.assertTrue(np.isnan(v[0]));self.assertEqual(q['duplicates_keep_last'],1)
    def test_stable_duplicate_order(self):
        t,v,q=a.canonical_rows([2,1,1],[20,10,11],'cpu_user');np.testing.assert_array_equal(t,[1,2]);np.testing.assert_array_equal(v,[11,20])
    def test_reference_weighted_binning_exact(self):
        t=self.start+np.array([0,100,200,600,800])*a.NS_SECOND;v=np.array([10.,20.,30.,40.,50.]);e=self.edges[:4]
        b,c=a.bin_split(t,v,'cpu_user',e);expected,ec=_time_weighted_bins(t,v,e,300*a.NS_SECOND,.5)
        np.testing.assert_allclose(b,expected,equal_nan=True,atol=1e-12);np.testing.assert_allclose(c,ec)
    def test_pre_split_hold_is_discarded(self):
        (v,c),q=self.bins([-1],[200]);self.assertTrue(np.isnan(v[0]));self.assertEqual(c[0],0)
    def test_future_values_do_not_change_completed_bin(self):
        (v,c),q=self.bins([0,200,600],[100,200,1000]);(v2,c2),q=self.bins([0,200,600],[100,200,9000]);np.testing.assert_array_equal(v,v2)
    def test_weather_strict_right_edge(self):
        (v,c),q=self.bins([600],[25],'temp');self.assertTrue(np.isnan(v[0]))
    def test_weather_age_expires(self):
        t=np.array([self.start]);v=np.array([25.]);b,_=a.bin_split(t,v,'temp',self.edges);self.assertTrue(np.isfinite(b[2]));self.assertTrue(np.isnan(b[3]))
    def test_weather_pre_split_observation_rejected(self):
        (v,c),q=self.bins([-60],[25],'temp');self.assertTrue(np.isnan(v[0]))
    def test_weather_invalid_latest_blocks_old_value(self):
        (v,c),q=self.bins([0,100],[25,np.nan],'temp');self.assertTrue(np.isnan(v[0]))
    def test_humidity_range(self):
        (v,c),q=self.bins([0],[101],'humidity');self.assertTrue(np.isnan(v[0]))
    def test_memory_alignment_and_invalid_values(self):
        m,q=a.aligned_memory([25,101,0,np.nan,5],[100,100,0,100,-1]);np.testing.assert_allclose(m,[.75,np.nan,np.nan,np.nan,np.nan],equal_nan=True);self.assertEqual(q['free_exceeds_total'],2)
    def test_memory_axes_rejected(self):
        with self.assertRaises(ValueError):a.aligned_memory([1],[2,3])
    def test_complete_windows_hourly_and_targets(self):
        selected,y,c,miss=a.window_masks(self.x,self.edges);np.testing.assert_array_equal(selected,[36,42,48,54,60,66]);np.testing.assert_allclose(y,[[200,25]]*6);self.assertEqual(c['candidate_hourly_origins'],12)
    def test_memory_gap_not_silently_dropped(self):
        self.x[35,5]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(len(s),0);self.assertEqual(c['incomplete_input_history'],6)
    def test_cpu_channel_kept(self):
        self.x[:,1]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(len(s),0);self.assertEqual(m[1],6)
    def test_weather_required_before_masking(self):
        self.x[:,7]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(len(s),0)
    def test_target_missing_is_not_filled(self):
        self.x[71,6]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges);self.assertNotIn(66,s);self.assertEqual(c['unobserved_joint_target'],1)
    def test_no_averaging_over_missing_target(self):
        self.x[71,0]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(len(y),5)
    def test_no_off_grid_or_short_axes(self):
        with self.assertRaises(ValueError):a.window_masks(self.x,self.edges+1)
        with self.assertRaises(ValueError):a.window_masks(self.x[:-1],self.edges)
    def test_exact_upper_endpoint_allowed(self):
        s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(s[-1]+6,72)
    def test_non_hour_split_start(self):
        e=self.edges+10*60*a.NS_SECOND;s,y,c,m=a.window_masks(self.x,e);self.assertTrue(((e[s]%(6*self.W))==0).all());self.assertTrue((s>=36).all())
    def test_exclusions_reconcile(self):
        self.x[50,0]=np.nan;self.x[42,5]=np.nan;s,y,c,m=a.window_masks(self.x,self.edges)
        self.assertEqual(sum(c[k] for k in ['insufficient_whole_history','target_crosses_split_end','unobserved_joint_target','incomplete_input_history','valid_windows']),c['candidate_hourly_origins'])
    def test_target_arithmetic_mean_not_last(self):
        self.x[36:42,0]=np.arange(6)/1000;self.x[36:42,6]=np.arange(6)+20;s,y,c,m=a.window_masks(self.x,self.edges);np.testing.assert_allclose(y[0],[2.5,22.5])
    def test_physical_ranges_not_missing_features(self):
        self.x[:,1]=101;s,y,c,m=a.window_masks(self.x,self.edges);self.assertEqual(len(s),0)
    def test_absolute_zero_target_rejected(self):
        self.x[71,6]=-274;s,y,c,m=a.window_masks(self.x,self.edges);self.assertNotIn(66,s)
    def test_mask_preserves_other_channels_and_input(self):
        x=np.arange(2*36*14,dtype=float).reshape(2,36,14);old=x.copy();z=mask_weather(x,'masked');np.testing.assert_array_equal(x,old);self.assertTrue((z[:,:,[7,8]]==0).all());np.testing.assert_array_equal(z[:,:,list(set(range(14))-{7,8})],old[:,:,list(set(range(14))-{7,8})])
    def test_enabled_mask_no_change(self):
        x=np.ones((1,36,14));np.testing.assert_array_equal(mask_weather(x,'enabled'),x)
    def test_training_only_thresholds(self):
        np.testing.assert_array_equal(a.eligibility([4,24,24],[7,6,7],24,7),[False,False,True])
    def test_scenarios_do_not_move_splits(self):
        self.assertFalse(a.scenario_has('march_only','train'));self.assertFalse(a.scenario_has('january_february','test'));self.assertTrue(a.scenario_has('january_march','test'))
    def test_existing_split_function_agreement(self):
        p=json.loads((a.ROOT/'configs/experiments/phase2_multioutput/v1/protocol.json').read_text());s,y,c,m=a.window_masks(self.x,self.edges)
        self.assertTrue(all(split_for_origin(int(self.edges[i]),36,p)=='test' for i in s))
    def test_spooling_node_separation_and_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'sample.parquet'
            pq.write_table(pa.table({'timestamp':pa.array([self.start,self.start,self.start],type=pa.timestamp('ns',tz='UTC')),'value':[10.,20.,30.],'node':['1','2','unknown']}),f)
            limits={'nodes_per_shard':1,'arrow_batch_rows':2,'max_shard_rows':10,'max_unknown_nodes':10}
            paths,c,q=a.spool([f],['1','2'],root,[(self.start,self.start+self.W)],limits)
            self.assertEqual(c.tolist(),[1,1]);self.assertEqual(q[0]['unknown_nodes'],['unknown']);self.assertEqual(np.fromfile(paths[0],dtype=a.RAW_DTYPE)['value'][0],10)
    def test_naive_timestamps_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'x.parquet';pq.write_table(pa.table({'timestamp':pa.array([self.start],type=pa.timestamp('ns')),'value':[1.],'node':['1']}),f)
            with self.assertRaises(ValueError):a.spool([f],['1'],root,[(self.start,self.start+self.W)],{'nodes_per_shard':1,'arrow_batch_rows':2,'max_shard_rows':10,'max_unknown_nodes':10})
    def test_symlink_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'alias').symlink_to(p,target_is_directory=True)
            with self.assertRaises(ValueError):a.safe_path(p,'alias/out')
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):a.safe_path(a.ROOT,'../other')
    def test_occupied_destination_never_written(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);old=p.stat().st_mtime_ns
            with patch.object(a,'check_bindings',return_value=({},[],0)):
                with self.assertRaises(ValueError):a.execute(p,{},p,'cfg')
            self.assertEqual(p.stat().st_mtime_ns,old);self.assertEqual(list(p.iterdir()),[])
    def test_verify_hash_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'data.txt').write_text('one');a.write_json(p/'completion.json',{'status':'coverage_audit_complete','artifacts':{'data.txt':{'bytes':3,'sha256':a.sha256(p/'data.txt')}}});(p/'data.txt').write_text('two')
            with self.assertRaises(ValueError):a.verify_audit(p)

    def test_duplicate_records_and_split_leakage_rejected(self):
        protocol={'splits':{'test':['2022-03-01T00:00:00Z','2022-04-01T00:00:00Z']}}
        r=np.zeros(2,dtype=a.WINDOW_DTYPE);r['origin_ns']=self.start+36*self.W;r['power_w']=200;r['temperature_c']=25
        with self.assertRaises(ValueError):a.validate_records(r,['1'],'test',protocol)
        r=r[:1];r['origin_ns']=self.start+30*self.W
        with self.assertRaises(ValueError):a.validate_records(r,['1'],'test',protocol)
    def test_synthetic_streaming_end_to_end(self):
        protocol=json.loads((a.ROOT/'configs/experiments/phase2_multioutput/v1/protocol.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);files={};names=['1','2']
            for metric in (*a.LOCAL_METRICS,'temp','humidity'):
                times=[];values=[];nodes=[]
                for lo,hi in protocol['splits'].values():
                    t=a.utc_ns(lo)
                    for node in names:
                        for sec in range(0,48*3600,300):
                            times.append(t+sec*a.NS_SECOND);nodes.append(node)
                            values.append({'total_power':200.,'ambient':25.,'mem_free':25.,'mem_total':100.,'temp':20.,'humidity':50.}.get(metric,10.))
                cols={'timestamp':pa.array(times,type=pa.timestamp('ns',tz='UTC')),'value':values}
                if metric not in ('temp','humidity'):cols['node']=nodes
                path=root/(metric+'.parquet');pq.write_table(pa.table(cols),path);files[metric]=[path.name]
            cfg={'metric_files':files,'limits':{'nodes_per_shard':1,'arrow_batch_rows':20,'max_shard_rows':10000,'max_node_rows':10000,'max_unknown_nodes':100,'max_weather_rows':10000,'minimum_free_bytes':1},'new_implementation_files':[],'inputs':{path.name:{'bytes':path.stat().st_size,'sha256':a.sha256(path)} for path in root.glob('*.parquet')}}
            cp=root/'config.json';a.write_json(cp,cfg)
            with patch.object(a,'check_bindings',return_value=(protocol,names,0)):
                result=a.execute(root,cfg,root/'out',cp)
            self.assertEqual(result['status'],'verified_coverage_audit');self.assertEqual(result['window_arrays_checked'],18)
            original=a.sha256(root/'total_power.parquet');a.verify_audit(root/'out');self.assertEqual(original,a.sha256(root/'total_power.parquet'))
            summary=json.loads((root/'out/summary.json').read_text())
            march_train=next(r for r in summary['split_coverage'] if (r['scenario'],r['split'],r['cohort'])==('march_only','train','native_mapped'))
            self.assertEqual(march_train['valid_windows'],0)
            self.assertTrue(any(r['valid_windows']>0 for r in summary['split_coverage']))

if __name__=='__main__':unittest.main()
