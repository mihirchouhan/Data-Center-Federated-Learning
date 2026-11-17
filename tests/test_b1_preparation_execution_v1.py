"""Synthetic-only B1 execution integration tests; real project data denied."""
from pathlib import Path
from dataclasses import asdict,replace
import copy,hashlib,importlib.util,json,os,sys,tempfile,time,unittest,zipfile,io as stdio,shutil,types
from unittest import mock
R=Path('/media/csedept/cse2014/Project/m100_training_template')
sys.path.insert(0,str(R))
import src
# The staging path is a test-only import location, never a production fallback.
if not os.environ.get('B1_EXECUTION_TEST_INSTALLED') and (Path(__file__).parent/'b1_preparation_execution_v1.py').is_file():src.__path__.insert(0,str(Path(__file__).parent))
from src import b1_preparation_execution_v1 as e
from src import b1_preparation_io_v1 as io
from src import b1_raw_window_producer_v1 as p
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def deny_real(event,args):
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No network/subprocess; multiprocessing spawn uses its audited Python path')
    if event=='import' and args[0].split('.')[0] in ('torch','tensorflow','sklearn','xgboost'):raise PermissionError('No model/scoring imports')
    if event=='open' and not isinstance(args[0],int):
        path=Path(os.fsdecode(args[0])).absolute();flags=args[2]
        if path.is_relative_to(R/'data') or (path.is_relative_to(R) and path.suffix in ('.npy','.npz','.pt','.tar')):raise PermissionError('No actual telemetry/arrays/checkpoints')
        if type(flags) is int and flags&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC|os.O_APPEND) and not path.is_relative_to(Path('/tmp')):raise PermissionError('Synthetic fixture writes only')
sys.addaudithook(deny_real)

def quota(root,limits):return {'confirmed':True,'uid':os.getuid(),'headroom_bytes':limits.max_disk_bytes+limits.reserve_bytes+1024**3,'checked_at_unix':time.time(),'filesystem_device':Path(root).stat().st_dev,'evidence_source':'INVENTED synthetic quota only'}
def sleep_worker(*args):time.sleep(5)
def foreign_worker(plan,root,output,authorization,quota,pipe,session):
    out=Path(output);out.mkdir();io.exclusive(out/'owner.json',io.canonical({'attempt':'foreign-session','uid':os.getuid(),'contract_sha256':e.contract(plan)}));pipe.send({'ok':False,'error':'competitor won destination'});pipe.close()

class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='b1-execution-synthetic-');cls.root=Path(cls.temp.name);cls.raw=cls.root/'raw';cls.raw.mkdir();cls.clients=['1','979'];cls.streams={};entries=[]
        cls.limits=io.Limits(arrow_rows=37,chunk_windows=2,max_disk_bytes=256*1024**2,reserve_bytes=1024**2,max_RSS_bytes=2*1024**3,max_address_space_bytes=8*1024**3,max_wall_seconds=60)
        # Twelve hours per burst, five-minute observed measurements; Jan+Feb val bursts.
        def burst(date):
            start=np.datetime64(date,'ns').view('i8');return (start+np.arange(0,12*3600,300,dtype='i8')*10**9).astype('datetime64[ns]')
        dates={'22-01':['2022-01-01','2022-01-28'],'22-02':['2022-02-01']}
        cls.rows={}
        for metric in p.METRICS:
            local=metric in p.LOCAL_METRICS;cls.rows[metric]={}
            for month in dates:
                columns={'timestamp':[],'value':[]}
                if local:columns['node']=[]
                for node in cls.clients if local else [None]:
                    t=np.concatenate([burst(d) for d in dates[month]])
                    val={'total_power':400. if node=='1' else 500.,'ambient':20. if node=='1' else 22.,'cpu_user':20.,'cpu_system':10.,'cpu_idle':65.,'cpu_wio':5.,'mem_free':4.,'mem_total':10.,'temp':18.,'humidity':50.}[metric]
                    v=np.full(len(t),val);order=np.arange(len(t))[::-1]
                    columns['timestamp'].extend(t[order].view('i8').tolist());columns['value'].extend(v[order].tolist())
                    if local:columns['node'].extend([node]*len(t))
                    cls.rows[metric].setdefault(node,[]).append((t[order],v[order]))
                table={'timestamp':pa.array(columns['timestamp'],type=pa.timestamp('ns',tz='UTC')),'value':pa.array(columns['value'])}
                if local:table['node']=pa.array(columns['node'])
                plugin='weather_pub' if not local else ('ipmi_pub' if metric in ('total_power','ambient') else 'ganglia_pub')
                relative=f'year_month={month}/plugin={plugin}/metric={metric}/a_0.parquet';path=cls.raw/relative;path.parent.mkdir(parents=True);pq.write_table(pa.table(table),path)
                entries.append({'path':relative,'metric':metric,**io.binding(path)})
        weather={m:cls.series(m,None) for m in ('temp','humidity')};local={node:{m:cls.series(m,node) for m in p.LOCAL_METRICS} for node in cls.clients};expected={};cls.references={}
        for split in ('train','val'):
            reference=p.produce_from_series(local,weather,split=split);w=reference.windows;cls.references[split]=reference
            expected[split]={'windows':len(w),'identity_sha256':w.identity_sha256,'target_sha256':w.target_sha256,'per_node':{n:int((w.nodes==n).sum()) for n in cls.clients}}
        cls.plan={'schema':e.SCHEMA,'scope':'synthetic','protocol_status':'UNAPPROVED','historical_preflight':'FAILED','source_root':str(cls.raw),'output_parent':str(cls.root),'mapping':list(e.bridge.ORIGINAL_NAMES),'features':list(e.bridge.FEATURES),'units':list(e.bridge.UNITS),'splits':{k:list(v) for k,v in e.bridge.SPLITS.items()},'clients':cls.clients,'sources':entries,'expected':expected,'limits':asdict(cls.limits),'implementation':e.implementation_bindings(),'environment':e.environment(),
          'window_contract':{'history':36,'horizon':6,'grid_seconds':600,'hold_seconds':300,'min_coverage':.5,'weather_max_age_seconds':1800,'input_fill':False,'target_fill':False,'target_units':['W','degC']}}
    @classmethod
    def series(cls,metric,node):
        pieces=cls.rows[metric][node];return p.RawSeries(np.concatenate([a for a,b in pieces]),np.concatenate([b for a,b in pieces]))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def output(self):return self.root/('attempt-'+os.urandom(6).hex())
    def prepare(self,plan=None,out=None):return e._prepare(plan or self.plan,self.raw,out or self.output(),None,quota(self.root,self.limits))
    def test_hand_computed_counts_and_units(self):
        self.assertEqual(self.plan['expected']['train']['windows'],12);self.assertEqual(self.plan['expected']['val']['windows'],24)
        for split in self.references:
            w=self.references[split].windows;np.testing.assert_allclose(w.physical_targets[w.nodes=='1'],np.broadcast_to([400.,20.],w.physical_targets[w.nodes=='1'].shape),rtol=0,atol=1e-12);self.assertTrue(np.all(w.x[w.nodes=='979',:,13]==978))
    def test_streamed_output_matches_existing_producer(self):
        out=self.output();result=self.prepare(out=out);self.assertEqual(result['results'],self.plan['expected']);self.assertEqual(io.status(out)['status'],'published_requires_verification')
        m=json.loads((out/'dataset/manifest.json').read_bytes())
        for split in ('train','val'):
            data=[]
            for chunk in m['chunks']:
                if chunk['split']==split:
                    with np.load(out/'dataset'/chunk['path'],allow_pickle=False) as f:data.append({n:f[n] for n in f.files})
            for n in e.FIELDS:np.testing.assert_array_equal(np.concatenate([d[n] for d in data]),self.references[split]._payload[n])
    def test_supervised_spawn_execution(self):
        out=self.output();result=e.run(self.plan,self.raw,out,quota=quota(self.root,self.limits));self.assertEqual(result['verification']['status'],'verified_complete');self.assertTrue((out/'completed.json').is_file())
        self.assertEqual(result['resources']['worker_hard_limits']['address_space'],[self.limits.max_address_space_bytes]*2)
    def test_original_ordinal_978_retained(self):
        out=self.output();self.prepare(out=out)
        for file in (out/'dataset').glob('*0978*.npz'):
            with np.load(file,allow_pickle=False) as a:self.assertTrue((a['x'][:,:,13]==978).all());self.assertTrue((a['nodes']=='979').all())
    def test_capacity_preflight_rejects_small_client_limit(self):
        plan=copy.deepcopy(self.plan);plan['limits']['max_client_windows']=5;plan['limits']['chunk_windows']=2
        with self.assertRaises(ValueError):e.validate_plan(plan,self.raw,self.output())
    def test_changed_expected_count_never_publishes(self):
        plan=copy.deepcopy(self.plan);plan['expected']['train']['per_node']['1']+=1;plan['expected']['train']['windows']+=1;out=self.output()
        with self.assertRaises(ValueError):self.prepare(plan,out)
        self.assertFalse((out/'dataset').exists());self.assertEqual(io.status(out)['status'],'incomplete_preserved');self.assertTrue((out/'failure.json').is_file())
    def test_target_digest_failure_preserves_attempt(self):
        plan=copy.deepcopy(self.plan);plan['expected']['val']['target_sha256']='0'*64;out=self.output()
        with self.assertRaises(ValueError):self.prepare(plan,out)
        self.assertFalse((out/'dataset').exists());self.assertTrue((out/'spool').is_dir())
    def test_identity_digest_failure_never_publishes(self):
        plan=copy.deepcopy(self.plan);plan['expected']['train']['identity_sha256']='0'*64;out=self.output()
        with self.assertRaises(ValueError):self.prepare(plan,out)
        self.assertFalse((out/'dataset').exists())
    def test_reproducible_chunk_bytes_and_manifest(self):
        a=self.output();b=self.output();self.prepare(out=a);self.prepare(out=b)
        self.assertEqual((a/'dataset/manifest.json').read_bytes(),(b/'dataset/manifest.json').read_bytes())
        self.assertEqual(io.inventory(a/'dataset'),io.inventory(b/'dataset'))
    def test_no_overwrite_existing_attempt(self):
        out=self.output();out.mkdir();(out/'precious').write_bytes(b'old')
        with self.assertRaises(FileExistsError):self.prepare(out=out)
        self.assertEqual((out/'precious').read_bytes(),b'old')
    def test_source_hash_mismatch(self):
        plan=copy.deepcopy(self.plan);plan['sources'][0]['sha256']='0'*64;out=self.output()
        with self.assertRaises(ValueError):self.prepare(plan,out)
        self.assertFalse((out/'dataset').exists())
    def test_source_month_rejected_before_open(self):
        plan=copy.deepcopy(self.plan);plan['sources'][0]['path']=plan['sources'][0]['path'].replace('22-01','22-03')
        with mock.patch.object(p,'verify_source_plan',side_effect=AssertionError('must not open')),self.assertRaises(PermissionError):self.prepare(plan)
    def test_no_test_capability(self):
        plan=copy.deepcopy(self.plan);plan['expected']['test']=plan['expected']['val']
        with self.assertRaises(ValueError):self.prepare(plan)
    def test_split_change_rejected(self):
        plan=copy.deepcopy(self.plan);plan['splits']['train'][1]='2022-02-11T07:10:00'
        with self.assertRaises(ValueError):self.prepare(plan)
    def test_node_renumbering_rejected(self):
        plan=copy.deepcopy(self.plan);plan['mapping'][978]='other'
        with self.assertRaises(ValueError):self.prepare(plan)
    def test_feature_drop_rejected(self):
        plan=copy.deepcopy(self.plan);plan['features'].pop()
        with self.assertRaises(ValueError):self.prepare(plan)
    def test_scaler_fitter_never_called(self):
        from src import b1_bridge_scaling_v1 as scaling
        with mock.patch.object(scaling,'fit_training_scaler',side_effect=AssertionError('forbidden')):self.prepare()
    def test_source_symlink_rejected(self):
        link=self.root/('symlink-'+os.urandom(4).hex());link.symlink_to(self.raw,target_is_directory=True)
        with self.assertRaises(ValueError):e.validate_plan(self.plan,link,self.output())
    def test_production_no_approval_rejected(self):
        plan=copy.deepcopy(self.plan);plan['scope']='production';plan['clients']=list(e.bridge.ORIGINAL_NAMES)
        # Production reference validation fails closed, before any source file opens.
        with self.assertRaises((ValueError,PermissionError,KeyError)):e.validate_plan(plan,self.raw,self.output())
    def test_unknown_quota_rejected_before_attempt(self):
        out=self.output()
        with self.assertRaises(ValueError):e.run(self.plan,self.raw,out)
        self.assertFalse(out.exists())
    def test_stale_quota_rejected(self):
        q=quota(self.root,self.limits);q['checked_at_unix']=0
        with self.assertRaises(ValueError):io.Budget(self.output(),self.limits,q)
    def test_insufficient_quota_rejected(self):
        q=quota(self.root,self.limits);q['headroom_bytes']=1
        with self.assertRaises(ValueError):io.Budget(self.output(),self.limits,q)
    def test_filesystem_quota_identity_rejected(self):
        q=quota(self.root,self.limits);q['filesystem_device']=-1
        with self.assertRaises(ValueError):io.Budget(self.output(),self.limits,q)
    def test_disk_cap_exceeded_no_charge(self):
        b=io.Budget(self.output(),self.limits,quota(self.root,self.limits))
        with self.assertRaises(OSError):b.charge(self.limits.max_disk_bytes+1)
        self.assertEqual(b.used,0)
    def test_supervisor_wall_timeout(self):
        plan=copy.deepcopy(self.plan);plan['limits']['max_wall_seconds']=1;out=self.output()
        with mock.patch.object(e,'_worker',sleep_worker),self.assertRaises(RuntimeError):e.run(plan,self.raw,out,quota=quota(self.root,self.limits))
        self.assertFalse((out/'dataset').exists())
    def test_incomplete_inspection_read_only(self):
        out=self.output();budget=io.Budget(out,self.limits,quota(self.root,self.limits));attempt=io.Attempt(out,'a'*64,budget);attempt.close();before=io.inventory(out)
        self.assertEqual(io.status(out)['status'],'incomplete_preserved');self.assertEqual(before,io.inventory(out))
    def test_active_attempt_reported_running(self):
        out=self.output();attempt=io.Attempt(out,'a'*64,io.Budget(out,self.limits,quota(self.root,self.limits)))
        try:self.assertEqual(io.status(out)['status'],'running')
        finally:attempt.close()
    def test_wrong_chunk_hash_detected(self):
        out=self.output();self.prepare(out=out);path=next((out/'dataset').glob('*.npz'));path.write_bytes(b'bad')
        with self.assertRaises(ValueError):e.verify_dataset(out/'dataset',self.plan)
    def test_missing_chunk_detected(self):
        out=self.output();self.prepare(out=out);next((out/'dataset').glob('*.npz')).rename(out/'removed')
        with self.assertRaises(ValueError):e.verify_dataset(out/'dataset',self.plan)
    def test_extra_artifact_detected(self):
        out=self.output();self.prepare(out=out);(out/'dataset/unexpected').write_bytes(b'x')
        with self.assertRaises(ValueError):e.verify_dataset(out/'dataset',self.plan)
    def test_compressed_archive_refused_before_array_load(self):
        path=self.root/'compressed.npz';np.savez_compressed(path,x=np.zeros((1,36,14)))
        with self.assertRaises(ValueError):e._check_archive(path,1,self.limits)
    def test_atomic_publication_race_does_not_replace(self):
        source=self.root/('partial-'+os.urandom(4).hex());destination=self.root/('final-'+os.urandom(4).hex());source.mkdir();destination.mkdir();inode=destination.stat().st_ino
        with self.assertRaises(OSError):io.rename_noreplace(source,destination)
        self.assertEqual(destination.stat().st_ino,inode);self.assertTrue(source.exists())
    def test_source_and_targets_are_preserved(self):
        before={x['path']:io.binding(self.raw/x['path']) for x in self.plan['sources']};self.prepare()
        self.assertEqual(before,{x['path']:io.binding(self.raw/x['path']) for x in self.plan['sources']})
    def production_metadata(self):
        plan=copy.deepcopy(self.plan);plan['scope']='production';plan['clients']=list(e.bridge.ORIGINAL_NAMES)
        for split in ('train','val'):
            ref=e.REFERENCE[split];q,r=divmod(ref['windows'],979);plan['expected'][split]={**ref,'per_node':{n:q+(i<r) for i,n in enumerate(plan['clients'])}}
        return plan
    def test_production_exact_gate_before_raw_open(self):
        plan=self.production_metadata();out=self.output()
        with mock.patch.object(p,'verify_source_plan',side_effect=AssertionError('must not open')),self.assertRaises(PermissionError):e.run(plan,self.raw,out,quota=quota(self.root,self.limits))
        self.assertFalse(out.exists())
    def test_auth_binds_output_and_contract(self):
        plan=self.production_metadata();out=self.output();auth={'operation':'B1_train_val_preparation_only','contract_sha256':'0'*64,'output':str(out),'uid':os.getuid(),'explicit_operator_authorization':True,'expires_at_unix':time.time()+60}
        with self.assertRaises(PermissionError):e.validate_plan(plan,self.raw,out,auth)
    def test_truthy_nonboolean_authorization_rejected(self):
        plan=self.production_metadata();out=self.output();auth={'operation':'B1_train_val_preparation_only','contract_sha256':e.contract(plan),'output':str(out),'uid':os.getuid(),'explicit_operator_authorization':'yes','expires_at_unix':time.time()+60}
        with self.assertRaises(PermissionError):e.validate_plan(plan,self.raw,out,auth)
    def test_bound_output_parent(self):
        with self.assertRaises(ValueError):e.validate_plan(self.plan,self.raw,self.root/'raw'/'new-output')
    def test_parent_traversal_denied(self):
        with self.assertRaises(ValueError):io.safe_path(self.root/'../bad')
    def test_duplicate_source_identity_rejected(self):
        plan=copy.deepcopy(self.plan);plan['sources'][1]=plan['sources'][0]
        with self.assertRaises(ValueError):e.validate_plan(plan,self.raw,self.output())
    def test_implementation_hash_mismatch(self):
        plan=copy.deepcopy(self.plan);next(iter(plan['implementation'].values()))['sha256']='0'*64
        with self.assertRaises(ValueError):e.validate_plan(plan,self.raw,self.output())
    def test_default_cli_only_describes(self):
        path=self.root/'description-plan.json';path.write_bytes(io.canonical(self.plan));stream=stdio.StringIO()
        with mock.patch.object(sys,'argv',['program','--plan',str(path)]),mock.patch.object(sys,'stdout',stream),mock.patch.object(p,'verify_source_plan',side_effect=AssertionError('no raw open')):e.main()
        self.assertFalse(json.loads(stream.getvalue())['sources_opened'])
    def test_worker_failure_not_reported_success(self):
        plan=copy.deepcopy(self.plan);plan['sources'][0]['sha256']='0'*64;out=self.output()
        with self.assertRaises(RuntimeError):e.run(plan,self.raw,out,quota=quota(self.root,self.limits))
        self.assertFalse((out/'dataset').exists());self.assertTrue((out/'supervisor_failure.json').is_file())
    def test_RSS_supervisor_stop(self):
        out=self.output()
        with mock.patch.object(e,'_worker',sleep_worker),mock.patch.object(Path,'read_text',return_value='VmHWM: 99999999 kB\n'),self.assertRaises(RuntimeError):e.run(self.plan,self.raw,out,quota=quota(self.root,self.limits))
        self.assertFalse((out/'dataset').exists())
    def variant(self,edit):
        root=self.root/('raw-variant-'+os.urandom(4).hex());shutil.copytree(self.raw,root);plan=copy.deepcopy(self.plan);plan['source_root']=str(root)
        entry=plan['sources'][0];path=root/entry['path'];table=pq.read_table(path);table=edit(table);pq.write_table(table,path);entry.update(io.binding(path));return plan,root
    def test_misfiled_invented_March_time_before_values(self):
        def edit(table):
            values=table['timestamp'].to_pylist();values[0]=np.datetime64('2022-03-01','ns').astype('datetime64[us]').astype(object)
            return table.set_column(0,'timestamp',pa.array(values,type=pa.timestamp('ns',tz='UTC')))
        plan,root=self.variant(edit);out=self.output();real=pq.ParquetFile
        class Guard:
            def __init__(self,h):self.f=real(h);self.schema_arrow=self.f.schema_arrow;self.metadata=self.f.metadata
            def iter_batches(self,**kw):
                if 'value' in kw['columns']:raise AssertionError('value request before month validation')
                return self.f.iter_batches(**kw)
        with mock.patch.object(pq,'ParquetFile',Guard),self.assertRaises(PermissionError):e._prepare(plan,root,out,None,quota(self.root,self.limits))
        self.assertFalse((out/'dataset').exists())
    def test_invalid_last_raw_duplicate_changes_reference_and_stops(self):
        def edit(table):
            # Append an invalid duplicate to an observed target bin; preserve raw row order.
            one=table.slice(0,2);one=one.set_column(one.schema.get_field_index('value'),'value',pa.array([float('nan'),float('nan')]))
            return pa.concat_tables([table,one])
        plan,root=self.variant(edit);out=self.output()
        with self.assertRaises(ValueError):e._prepare(plan,root,out,None,quota(self.root,self.limits))
        self.assertFalse((out/'dataset').exists())
    def test_verify_never_reads_raw_sources(self):
        out=self.output();self.prepare(out=out)
        with mock.patch.object(p,'verify_source_plan',side_effect=AssertionError('raw source access')),mock.patch.object(e,'_read_sources',side_effect=AssertionError('raw parse')):self.assertEqual(e.verify_run(out,self.plan)['status'],'verified_complete')
    def test_completion_receipt_tamper_rejected(self):
        out=self.output();self.prepare(out=out);(out/'completed.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):e.verify_run(out,self.plan)
    def test_software_change_rejected(self):
        plan=copy.deepcopy(self.plan);plan['environment']['numpy']='changed'
        with self.assertRaises(ValueError):e.validate_plan(plan,self.raw,self.output())
    def test_client_load_preserves_capability_roles(self):
        out=self.output();self.prepare(out=out)
        train=e.load_client_windows(out,self.plan,'train','979');val=e.load_client_windows(out,self.plan,'val','979')
        self.assertIs(type(train),e.bridge.TrainingWindows);self.assertIs(type(val),e.bridge.ValidationWindows);self.assertTrue((train.x[:,:,13]==978).all())
        with self.assertRaises(PermissionError):e.load_client_windows(out,self.plan,'test','979')
    def test_losing_supervisor_never_writes_foreign_attempt(self):
        out=self.output()
        with mock.patch.object(e,'_worker',foreign_worker),self.assertRaises(RuntimeError):e.run(self.plan,self.raw,out,quota=quota(self.root,self.limits))
        self.assertEqual({x.name for x in out.iterdir()},{'owner.json'})
    def test_unclaimed_partial_status_preserved(self):
        out=self.output();out.mkdir();self.assertEqual(io.status(out)['status'],'incomplete_unclaimed');self.assertEqual(list(out.iterdir()),[])
    def test_full_filesystem_capacity_required(self):
        with mock.patch.object(os,'statvfs',return_value=types.SimpleNamespace(f_bavail=self.limits.reserve_bytes+1,f_frsize=1)),self.assertRaises(OSError):io.Budget(self.output(),self.limits,quota(self.root,self.limits))
    def test_missing_source_metric_refused(self):
        plan=copy.deepcopy(self.plan);plan['sources'].pop()
        with self.assertRaises(ValueError):e.validate_plan(plan,self.raw,self.output())
    def test_execution_envelope_preserved_and_bound(self):
        out=self.output();self.prepare(out=out);envelope=json.loads((out/'execution_envelope.json').read_bytes());self.assertEqual(envelope['scope'],'synthetic');self.assertTrue(envelope['quota_evidence']['confirmed'])
        (out/'execution_envelope.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):e.verify_run(out,self.plan)

if __name__=='__main__':unittest.main(verbosity=2)
