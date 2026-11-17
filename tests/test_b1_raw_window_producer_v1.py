"""Invented raw telemetry and /tmp Parquet fixtures only; no fitting/models."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import numpy as np
from src import b1_raw_window_producer_v1 as p
from src import b1_real_data_bridge_v1 as b


def streams(split='train',nodes=('3','979'),days=1):
    start=np.datetime64('2022-01-10' if split=='train' else '2022-01-28','ns')
    steps=days*144;t=start+np.arange(steps)*np.timedelta64(10,'m')
    times=np.stack((t,t+np.timedelta64(300,'s')),axis=1).ravel()
    local={}
    for node in nodes:
        k=b.ORIGINAL_NAMES.index(node)
        vals={'total_power':500+k+np.arange(steps)*.125,'ambient':20+np.arange(steps)*.001,
          'cpu_user':np.full(steps,10.),'cpu_system':np.full(steps,20.),'cpu_idle':np.full(steps,60.),
          'cpu_wio':np.full(steps,10.),'mem_free':np.full(steps,40.),'mem_total':np.full(steps,100.)}
        local[node]={m:p.RawSeries(times,np.repeat(vals[m],2)) for m in p.LOCAL_METRICS}
    weather={m:p.RawSeries(t+np.timedelta64(120,'s'),np.full(steps,v)) for m,v in [('temp',12.),('humidity',55.)]}
    return local,weather


def altered(series,indices,values):
    v=series.values.copy();v[indices]=values
    return p.RawSeries(series.times,v)


class RawWindowTests(unittest.TestCase):
    def test_shape_units_and_feature_order(self):
        a,w=streams();r=p.produce_from_series(a,w,split='train');x=r.windows.x
        self.assertEqual(x.shape,(36,36,14));self.assertEqual(r.windows.physical_targets.shape,(36,2))
        self.assertEqual(tuple(b.FEATURES),('power_kw','cpu_user','cpu_system','cpu_idle','cpu_wio','memory_used_fraction','inlet_temperature_mean_c','outdoor_temperature_c','outdoor_humidity_percent','hour_sin','hour_cos','week_sin','week_cos','node_index'))
        self.assertEqual(b.UNITS,('kW','%','%','%','%','fraction','degC','degC','%','unitless','unitless','unitless','unitless','categorical index'))
        np.testing.assert_allclose(x[0,0,:9],[.502,10,20,60,10,.6,20,12,55])
    def test_observed_six_bin_targets_and_single_W_conversion(self):
        a,w=streams(nodes=('3',));r=p.produce_from_series(a,w,split='train').windows
        np.testing.assert_allclose(r.physical_targets[0],[502+np.arange(36,42).mean()*.125,20+np.arange(36,42).mean()*.001])
        np.testing.assert_array_equal(r.physical_targets,b.target_validation.target_means(r.future,r.target_observed))
        self.assertTrue(r.target_observed.all());self.assertEqual(r.future.shape,(18,6,2))
    def test_original_ordinal_preserved_in_subsets(self):
        a,w=streams();r=p.produce_from_series(a,w,split='train')
        self.assertEqual(r.clients,('3','979'));self.assertEqual(r.windows.mapping,b.ORIGINAL_NAMES)
        self.assertEqual(set(r.windows.x[:,:,13].ravel()),{2.,978.})
        self.assertTrue((r.client('979').x[:,:,13]==978).all())
        with self.assertRaises(KeyError):r.client('1')
    def test_identity_determinism_dictionary_order_independent(self):
        a,w=streams();x=p.produce_from_series(a,w,split='train').windows
        y=p.produce_from_series(dict(reversed(list(a.items()))),dict(reversed(list(w.items()))),split='train').windows
        self.assertEqual(x.identity_sha256,y.identity_sha256);self.assertEqual(x.content_sha256,y.content_sha256)
    def test_client_window_copies_independent_and_immutable(self):
        a,w=streams();r=p.produce_from_series(a,w,split='train');c=r.client('3')
        self.assertFalse(np.shares_memory(c.x,r.windows.x));self.assertTrue((c.nodes=='3').all())
        with self.assertRaises(ValueError):c.x.setflags(write=True)
        with self.assertRaises(AttributeError):r.windows=c
    def test_missing_input_bin_is_not_bridged(self):
        a,w=streams(nodes=('3',));a['3']['cpu_user']=altered(a['3']['cpu_user'],[36,37],np.nan)
        r=p.produce_from_series(a,w,split='train');origin=np.datetime64('2022-01-10T06:00','ns')
        self.assertNotIn(origin,r.windows.origins);self.assertGreater(r.diagnostics['per_client']['3']['incomplete_input_history'],0)
    def test_invalid_target_nan_inf_and_negative_rejected_by_window_selection(self):
        for v in (np.nan,np.inf,-1.):
            a,w=streams(nodes=('3',));a['3']['total_power']=altered(a['3']['total_power'],[72,73],v)
            r=p.produce_from_series(a,w,split='train')
            self.assertNotIn(np.datetime64('2022-01-10T06:00','ns'),r.windows.origins)
            self.assertGreater(r.diagnostics['per_client']['3']['invalid_observed_target'],0)
            self.assertTrue(np.isfinite(r.windows.physical_targets).all())
    def test_invalid_ambient_target_below_absolute_zero(self):
        a,w=streams(nodes=('3',));a['3']['ambient']=altered(a['3']['ambient'],[72,73],-300.)
        r=p.produce_from_series(a,w,split='train');self.assertGreater(r.diagnostics['per_client']['3']['invalid_observed_target'],0)
    def test_masked_even_nomask_and_boolean_raw_targets_rejected(self):
        t=np.array(['2022-01-01'],dtype='datetime64[ns]')
        for value in (np.ma.array([1.],mask=False),np.ma.array([1.],mask=True),np.array([True])):
            with self.assertRaises(ValueError):p.RawSeries(t,value)
    def test_null_timestamps_dropped_not_time_compressed(self):
        a,w=streams(nodes=('3',));s=a['3']['cpu_user'];times=s.times.copy();times[36:38]=np.datetime64('NaT','ns')
        a['3']['cpu_user']=p.RawSeries(times,s.values);r=p.produce_from_series(a,w,split='train')
        self.assertEqual(r.diagnostics['raw_quality']['3']['cpu_user']['null_timestamp_rows'],2)
        self.assertNotIn(np.datetime64('2022-01-10T06:00','ns'),r.windows.origins)
    def test_missing_weather_not_masked_or_removed(self):
        a,w=streams(nodes=('3',));s=w['temp'];w['temp']=altered(s,np.arange(12,18),np.nan)
        r=p.produce_from_series(a,w,split='train')
        self.assertNotIn(np.datetime64('2022-01-10T06:00','ns'),r.windows.origins)
        self.assertTrue((r.windows.x[:,:,7]==12).all())
    def test_invalid_memory_denominator_and_ratio_reject_windows(self):
        for m,value in [('mem_total',0.),('mem_free',101.)]:
            a,w=streams(nodes=('3',));a['3'][m]=altered(a['3'][m],[36,37],value)
            r=p.produce_from_series(a,w,split='train');self.assertNotIn(np.datetime64('2022-01-10T06:00','ns'),r.windows.origins)
    def test_missing_required_feature_declaration_aborts(self):
        a,w=streams();del a['3']['cpu_wio']
        with self.assertRaises(ValueError):p.produce_from_series(a,w,split='train')
    def test_unknown_and_renamed_node_rejected(self):
        a,w=streams(nodes=('3',))
        for name in ('0','03',3,'980'):
            with self.assertRaises(ValueError):p.produce_from_series({name:a['3']},w,split='train')
    def test_UTC_and_timestamp_dtype_must_be_explicit(self):
        for time,zone in [(np.array([1]),'UTC'),(np.array(['2022-01-01'],dtype='datetime64[s]'),'UTC'),(np.array(['2022-01-01'],dtype='datetime64[ns]'),'local')]:
            with self.assertRaises(ValueError):p.RawSeries(time,[1.],timezone=zone)
    def test_March_date_rejected_before_value_access(self):
        class Forbidden:
            def __array__(self,*a,**k):raise AssertionError('March values accessed')
        with self.assertRaises(PermissionError):p.RawSeries(np.array(['2022-03-01'],dtype='datetime64[ns]'),Forbidden())
    def test_test_split_rejected_before_payload_access(self):
        class Forbidden(dict):
            def __iter__(self):raise AssertionError('Data touched')
        with self.assertRaises(PermissionError):p.produce_from_series(Forbidden(),Forbidden(),split='test')
        with self.assertRaises(PermissionError):p.produce_from_plan(None,split='test',clients=('3',))
    def test_split_boundary_history_and_last_target_exact(self):
        a,w=streams('val',nodes=('3',),days=7);produced=p.produce_from_series(a,w,split='val');r=produced.windows
        self.assertEqual(r.origins[0],np.datetime64('2022-01-28T06:00','ns'))
        self.assertEqual(r.origins[-1],np.datetime64('2022-02-03T23:00','ns'))
        self.assertTrue((produced._payload['history_times']>=np.datetime64('2022-01-28','ns')).all())
        self.assertEqual(produced._payload['target_times'][-1,-1]+np.timedelta64(10,'m'),np.datetime64('2022-02-04','ns'))
    def test_validation_prefix_weather_cannot_use_previous_training_split(self):
        a,w=streams('val',nodes=('3',));s=w['temp'];t=s.times.copy();v=s.values.copy()
        # Add a valid23:50 prior-split record, remove first validation weather record.
        w['temp']=p.RawSeries(np.r_[np.datetime64('2022-01-27T23:50','ns'),t[1:]],np.r_[12.,v[1:]])
        r=p.produce_from_series(a,w,split='val')
        self.assertNotIn(np.datetime64('2022-01-28T06:00','ns'),r.windows.origins)
        self.assertIn(np.datetime64('2022-01-28T07:00','ns'),r.windows.origins)
    def test_no_later_rows_recycled_into_training(self):
        a,w=streams(nodes=('3',));extra=np.array(['2022-01-28T00:00'],dtype='datetime64[ns]')
        s=a['3']['total_power'];a['3']['total_power']=p.RawSeries(np.r_[s.times,extra],np.r_[s.values,1e99])
        r=p.produce_from_series(a,w,split='train')
        self.assertEqual(r.windows.physical_targets.shape,(18,2));self.assertTrue((r.windows.physical_targets[:,0]<1000).all())
        self.assertEqual(r.diagnostics['raw_quality']['3']['total_power']['out_of_split_rows'],1)
    def test_target_intervals_hourly_nonoverlapping_and_id_unique(self):
        a,w=streams();r=p.produce_from_series(a,w,split='train').windows
        for n in np.unique(r.nodes):self.assertTrue((np.diff(r.origins[r.nodes==n])>=np.timedelta64(1,'h')).all())
        self.assertTrue((r.origins.view('i8')%(6*p.WIDTH)==0).all())
        self.assertEqual(len(set(zip(r.nodes.tolist(),r.origins.view('i8').tolist()))),len(r))
    def test_rejection_accounting_reconciles(self):
        a,w=streams();r=p.produce_from_series(a,w,split='train')
        for q in r.diagnostics['per_client'].values():self.assertEqual(q['candidates'],sum(q[k] for k in ('accepted','history_before_split','target_after_split','invalid_observed_target','incomplete_input_history')))
    def test_limits_abort_without_silent_population_truncation(self):
        a,w=streams()
        for lim in (p.Limits(max_windows=1),p.Limits(max_population_bytes=1)):
            with self.assertRaises(ValueError):p.produce_from_series(a,w,split='train',limits=lim)
        with self.assertRaises(ValueError):p.Limits(max_windows=True)
        with self.assertRaises(ValueError):p.RawSeries(np.repeat(np.datetime64('2022-01-01','ns'),2),[1,2],limits=p.Limits(max_raw_rows=1))
    def test_no_input_mutation_and_original_bins_consistent_across_windows(self):
        a,w=streams(nodes=('3',));before=a['3']['total_power'].fingerprint;r=p.produce_from_series(a,w,split='train').windows
        self.assertEqual(before,a['3']['total_power'].fingerprint)
        np.testing.assert_array_equal(r.x[0,6:,:],r.x[1,:30,:])
        np.testing.assert_array_equal(r.future[0,:,0],r.x[1,-6:,0])
    def test_supplied_frozen_scaler_application_without_any_fit(self):
        from src import b1_bridge_scaling_v1 as s
        a,w=streams(nodes=('3',),days=8);train=p.produce_from_series(a,w,split='train')
        a,w=streams('val',nodes=('3',),days=7);val=p.produce_from_series(a,w,split='val')
        record={'fit_split':'train','algorithm':s.ALGORITHM,'mapping_digest':b.MAPPING_DIGEST,
          'fit_identity_sha256':train.windows.identity_sha256,'fit_target_sha256':train.windows.target_sha256,
          'fit_content_sha256':train.windows.content_sha256,'fit_windows':len(train.windows)}
        scaler=s.FrozenScaler((0.,)*14,(1.,)*14,(1.,)*13+(0.,),(0.,20.),(100.,5.),b.canonical(record))
        with mock.patch.object(s,'fit_training_scaler',side_effect=AssertionError('No fit')) as fit:
            td,vd=p.bind_frozen_datasets(train,val,scaler);batch=vd.client('3').batch(np.array([0,1]));fit.assert_not_called()
        np.testing.assert_allclose(scaler.inverse_targets(batch.y_standardized),batch.y,rtol=2e-6,atol=2e-6)
        self.assertEqual(td.clients,('3',));self.assertEqual(vd.clients,('3',))
        with self.assertRaises(PermissionError):p.bind_frozen_datasets(val,train,scaler)

    def test_cross_node_missing_bin_never_borrowed(self):
        a,w=streams();a['3']['cpu_user']=altered(a['3']['cpu_user'],[36,37],np.nan)
        r=p.produce_from_series(a,w,split='train');o=np.datetime64('2022-01-10T06:00','ns')
        self.assertNotIn(o,r.client('3').origins);self.assertIn(o,r.client('979').origins)
    def test_no_complete_windows_aborts_without_repair(self):
        a,w=streams(nodes=('3',));s=a['3']['cpu_user'];a['3']['cpu_user']=p.RawSeries(s.times,np.full(len(s.times),np.nan))
        with self.assertRaisesRegex(ValueError,'Empty accepted split'):p.produce_from_series(a,w,split='train')
    def test_missing_weather_declaration_cannot_drop_feature(self):
        a,w=streams();del w['humidity']
        with self.assertRaises(ValueError):p.produce_from_series(a,w,split='train')


class BinningTests(unittest.TestCase):
    def series(self,t,v):return p.RawSeries(np.array(t,dtype='datetime64[ns]'),v)
    def test_exact_half_support_weighted_mean_and_contributors(self):
        s=self.series(['2022-01-01T00:00','2022-01-01T00:02:30'],[10.,20.]);z,q=p.bin_series(s,'total_power','train')
        self.assertAlmostEqual(z.coverage[0],.75);self.assertAlmostEqual(z.values[0],(150*10+300*20)/450)
        self.assertEqual(z.contributors[0],2);self.assertEqual(z.earliest[0],s.times[0]);self.assertEqual(z.latest[0],s.times[1])
    def test_less_than_half_support_invalid_and_no_hold_across_gap(self):
        s=self.series(['2022-01-01T00:00','2022-01-01T00:04'],[10.,np.nan]);z,q=p.bin_series(s,'total_power','train')
        self.assertAlmostEqual(z.coverage[0],.4);self.assertTrue(np.isnan(z.values[:3]).all())
    def test_exact300second_hold_threshold_is_valid(self):
        z,_=p.bin_series(self.series(['2022-01-01T00:00'],[10.]),'total_power','train')
        self.assertEqual(z.coverage[0],.5);self.assertEqual(z.values[0],10.);self.assertTrue(np.isnan(z.values[1]))
    def test_duplicate_invalid_last_breaks_hold(self):
        z,q=p.bin_series(self.series(['2022-01-01T00:00','2022-01-01T00:00'],[10.,np.nan]),'total_power','train')
        self.assertEqual(q['duplicate_rows'],1);self.assertTrue(np.isnan(z.values[0]));self.assertEqual(z.coverage[0],0)
    def test_unsorted_rows_stably_canonicalized(self):
        t=['2022-01-01T00:05','2022-01-01T00:00','2022-01-01T00:00']
        z,_=p.bin_series(self.series(t,[20.,10.,30.]),'total_power','train');self.assertEqual(z.values[0],25.)
    def test_weather_right_edge_excluded_and_latest_invalid_not_backfilled(self):
        s=self.series(['2022-01-01T00:00','2022-01-01T00:10','2022-01-01T00:11'],[12.,99.,np.nan]);z,_=p.bin_series(s,'temp','train')
        self.assertEqual(z.values[0],12.);self.assertTrue(np.isnan(z.values[1]));self.assertEqual(z.earliest[0],s.times[0])
    def test_weather_maximum_age_expires_without_future_fill(self):
        z,_=p.bin_series(self.series(['2022-01-01T00:00'],[12.]),'temp','train')
        self.assertTrue(np.isfinite(z.values[:3]).all());self.assertTrue(np.isnan(z.values[3:]).all())
    def test_no_cross_split_local_hold(self):
        z,_=p.bin_series(self.series(['2022-01-27T23:59'],[500.]),'total_power','val')
        self.assertTrue(np.isnan(z.values).all());self.assertEqual(z.coverage.sum(),0)
    def test_empty_series_preserves_grid_as_invalid(self):
        z,_=p.bin_series(p.RawSeries(np.empty(0,dtype='datetime64[ns]'),[]),'cpu_user','val')
        self.assertEqual(len(z.values),1008);self.assertTrue(np.isnan(z.values).all())
    def test_bin_values_reuse_existing_reference_exactly(self):
        from src.preprocess import _time_weighted_bins
        s=self.series(['2022-01-01T00:01','2022-01-01T00:05','2022-01-01T00:09'],[10.,20.,30.]);z,_=p.bin_series(s,'total_power','train')
        lo,hi=p._split('train');e=np.arange(lo,hi+p.WIDTH,p.WIDTH,dtype='i8');v,c=_time_weighted_bins(s.times.view('i8'),s.values,e,300*10**9,.5)
        np.testing.assert_array_equal(z.values,v);np.testing.assert_array_equal(z.coverage,c)


def parquet_fixture(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    local,weather=streams(nodes=('3',));vlocal,vweather=streams('val',nodes=('3',));entries=[]
    for month in ('22-01','22-02'):
        for metric in p.METRICS:
            plugin='weather_pub' if metric in ('temp','humidity') else ('ipmi_pub' if metric in ('total_power','ambient') else 'ganglia_pub')
            rel=f'year_month={month}/plugin={plugin}/metric={metric}/a_0.parquet';path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
            series=weather[metric] if metric in weather else local['3'][metric]
            vseries=vweather[metric] if metric in weather else vlocal['3'][metric]
            times=np.r_[series.times,vseries.times] if month=='22-01' else np.empty(0,dtype='datetime64[ns]');values=np.r_[series.values,vseries.values] if month=='22-01' else np.empty(0)
            table={'timestamp':pa.array(times,type=pa.timestamp('ns',tz='UTC')),'value':pa.array(values,type=pa.float64())}
            if metric not in weather:table['node']=pa.array(['3']*len(times),type=pa.string())
            pq.write_table(pa.table(table),path);raw=path.read_bytes();entries.append(p.SourceEntry(rel,metric,len(raw),hashlib.sha256(raw).hexdigest()))
    return entries


class SourcePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic-source-');self.root=Path(self.temp.name);self.entries=parquet_fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def test_20_source_plan_and_Parquet_bridge_match_in_memory(self):
        plan=p.verify_source_plan(self.root,self.entries);r=p.produce_from_plan(plan,split='train',clients=('3',))
        a,w=streams(nodes=('3',));expected=p.produce_from_series(a,w,split='train').windows
        np.testing.assert_array_equal(r.windows.x,expected.x);np.testing.assert_array_equal(r.windows.future,expected.future)
        self.assertEqual(r.windows.identity_sha256,expected.identity_sha256);self.assertEqual(r.windows.source_evidence['source_kind'],'observed_prepared')
    def test_plan_order_is_deterministic_and_immutable(self):
        a=p.verify_source_plan(self.root,self.entries);z=p.verify_source_plan(self.root,reversed(self.entries));self.assertEqual(a.fingerprint,z.fingerprint)
        with self.assertRaises(AttributeError):a.root=Path('/data')
    def test_March_declaration_rejected_before_any_file_open(self):
        e=list(self.entries);e[-1]=replace(e[-1],path=e[-1].path.replace('22-02','22-03'))
        with mock.patch.object(p,'_open_entry',side_effect=AssertionError('File opened')):
            with self.assertRaises(PermissionError):p.verify_source_plan(self.root,e)
    def test_March_root_rejected_before_open(self):
        with mock.patch.object(p,'_open_entry',side_effect=AssertionError('File opened')):
            with self.assertRaises(PermissionError):p.verify_source_plan(self.root/'march_2022_manual',self.entries)
    def test_duplicate_overlapping_or_missing_source_plan(self):
        with self.assertRaises(ValueError):p.verify_source_plan(self.root,self.entries[:-1])
        e=list(self.entries);e[-1]=e[0]
        with self.assertRaises(ValueError):p.verify_source_plan(self.root,e)
    def test_path_traversal_wrong_metric_and_plugin_rejected_before_open(self):
        for name in ('../outside.parquet','/tmp/file.parquet',self.entries[0].path.replace('ipmi_pub','ganglia_pub')):
            e=list(self.entries);e[0]=replace(e[0],path=name)
            with mock.patch.object(p,'_open_entry',side_effect=AssertionError('File opened')):
                with self.assertRaises(ValueError):p.verify_source_plan(self.root,e)
    def test_hash_and_size_mismatches_fail(self):
        for kw in ({'sha256':'0'*64},{'bytes':1},{'bytes':True}):
            e=list(self.entries);e[0]=replace(e[0],**kw)
            with self.assertRaises(ValueError):p.verify_source_plan(self.root,e)
    def test_symlink_source_rejected(self):
        e=self.entries[0];file=self.root/e.path;backup=file.with_suffix('.original');file.rename(backup);file.symlink_to(backup.name)
        with self.assertRaises(ValueError):p.verify_source_plan(self.root,self.entries)
    def test_file_tampering_after_plan_verification_detected(self):
        plan=p.verify_source_plan(self.root,self.entries);file=self.root/self.entries[0].path
        with file.open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):p.produce_from_plan(plan,split='train',clients=('3',))
    def test_nonUTC_schema_and_boolean_values_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for timestamp,values in ((pa.timestamp('ns'),pa.float64()),(pa.timestamp('ns',tz='UTC'),pa.bool_())):
            e=self.entries[0];path=self.root/e.path
            table=pa.table({'timestamp':pa.array(np.array(['2022-01-10'],dtype='datetime64[ns]'),type=timestamp),'value':pa.array([True] if pa.types.is_boolean(values) else [1.],type=values),'node':['3']})
            pq.write_table(table,path);raw=path.read_bytes();entries=list(self.entries);entries[0]=replace(e,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            plan=p.verify_source_plan(self.root,entries)
            with self.assertRaises(ValueError):p.produce_from_plan(plan,split='train',clients=('3',))
    def test_misfiled_March_timestamp_stops_before_requesting_values(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        e=self.entries[0];path=self.root/e.path
        pq.write_table(pa.table({'timestamp':pa.array(np.array(['2022-03-01'],dtype='datetime64[ns]'),type=pa.timestamp('ns',tz='UTC')),'value':[500.],'node':['3']}),path)
        raw=path.read_bytes();entries=list(self.entries);entries[0]=replace(e,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());plan=p.verify_source_plan(self.root,entries)
        original=pq.ParquetFile.iter_batches
        def guard(f,*args,**kw):
            if 'value' in kw.get('columns',[]) and f.schema_arrow.field('timestamp').type.tz=='UTC' and f.metadata.num_rows==1:raise AssertionError('March value columns requested')
            return original(f,*args,**kw)
        with mock.patch.object(pq.ParquetFile,'iter_batches',guard):
            with self.assertRaises(PermissionError):p.produce_from_plan(plan,split='train',clients=('3',))
    def test_unknown_node_count_reported_not_admitted(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        e=self.entries[0];path=self.root/e.path;table=pq.ParquetFile(path).read();first=table.slice(0,1).set_column(2,'node',pa.array(['0']))
        pq.write_table(pa.concat_tables([table,first]),path);raw=path.read_bytes();entries=list(self.entries);entries[0]=replace(e,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        r=p.produce_from_plan(p.verify_source_plan(self.root,entries),split='train',clients=('3',));q=r.diagnostics['raw_quality']['3'][e.metric]['reader_evidence'][e.path]
        self.assertEqual(q['unknown_node_ids'],['0']);self.assertEqual(q['unknown_node_rows'],1);self.assertEqual(r.clients,('3',))
    def test_scan_budget_aborts_and_no_implicit_all_client_run(self):
        plan=p.verify_source_plan(self.root,self.entries)
        with self.assertRaises(ValueError):p.produce_from_plan(plan,split='train',clients=('3',),limits=p.Limits(max_scan_rows=1))
        for clients in ((),('3','3'),('980',)):
            with self.assertRaises(ValueError):p.produce_from_plan(plan,split='train',clients=clients)
    def test_no_side_effects_to_fixture_files(self):
        before={e.path:(self.root/e.path).read_bytes() for e in self.entries};plan=p.verify_source_plan(self.root,self.entries)
        p.produce_from_plan(plan,split='train',clients=('3',))
        self.assertEqual(before,{e.path:(self.root/e.path).read_bytes() for e in self.entries})

    def test_Parquet_validation_isolated_and_matches_in_memory(self):
        plan=p.verify_source_plan(self.root,self.entries);r=p.produce_from_plan(plan,split='val',clients=('3',))
        a,w=streams('val',nodes=('3',));expected=p.produce_from_series(a,w,split='val').windows
        np.testing.assert_array_equal(r.windows.x,expected.x);np.testing.assert_array_equal(r.windows.future,expected.future)
        self.assertEqual(r.windows.identity_sha256,expected.identity_sha256)
        self.assertTrue((r.windows.origins>=np.datetime64('2022-01-28','ns')).all())
    def test_file_raw_rows_read_per_client_before_next_client(self):
        plan=p.verify_source_plan(self.root,self.entries);events=[];read=p._read_plan_series;binning=p.bin_series
        def rr(plan,metric,node,split,limits):
            events.append(('read',node));return read(plan,metric,node,split,limits)
        def bb(*args,**kw):events.append(('bin',None));return binning(*args,**kw)
        with mock.patch.object(p,'_read_plan_series',rr),mock.patch.object(p,'bin_series',bb):
            result=p.produce_from_plan(plan,split='train',clients=('3','979'))
        first=events.index(('read','3'));next_client=events.index(('read','979'))
        self.assertIn(('bin',None),events[first:next_client]);self.assertEqual(result.clients,('3',))
        self.assertEqual(result.diagnostics['per_client']['979']['accepted'],0)
