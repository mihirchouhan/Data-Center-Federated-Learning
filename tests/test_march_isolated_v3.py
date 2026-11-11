"""Phase 1.9.5: isolated /tmp fixtures and literal-loopback transport only."""
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path('/media/csedept/cse2014/Project/m100_training_template')
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'tests'))
import march_isolated_v3 as iso
import test_march_validators_v2 as prior_tests

class Interrupted(RuntimeError): pass

class IsolatedSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix='m100-p195-tests-'))
        cls.cfg = json.loads(iso.read_owned(iso.CONFIG))
        cls.saved = {p:iso.read_owned(ROOT/p) for p in cls.cfg['prior_session']['files']}
        connect = socket.socket.connect
        def local(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
                raise AssertionError('Non-loopback connection forbidden')
            return connect(sock,address)
        cls.network_guard = mock.patch.object(socket.socket,'connect',local)
        cls.network_guard.start()
    @classmethod
    def tearDownClass(cls): cls.network_guard.stop()
    def setUp(self):
        self.root = self.work/self._testMethodName; self.root.mkdir()
        self.cfg = copy.deepcopy(type(self).cfg); self.servers = []
        for relative,raw in self.saved.items():
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix=='.json':raw=raw.replace(str(ROOT).encode(),str(self.root).encode())
            path.write_bytes(raw)
            s=path.stat();self.cfg['prior_session']['files'][relative]={
                'bytes':s.st_size,'sha256':hashlib.sha256(raw).hexdigest(),
                'mtime_ns':s.st_mtime_ns,'mode':stat.S_IMODE(s.st_mode),
                'inode':s.st_ino,'device':s.st_dev}
        self.before=self.old_snapshot()
    def tearDown(self):
        for server in self.servers:server.close()
    def old_snapshot(self):
        return {p:(self.root/p).read_bytes() for p in self.cfg['prior_session']['files']}
    def engine(self,modes=None,**changes):
        server=prior_tests.Server(modes);self.servers.append(server)
        raw=b'{"synthetic":"no-live-metadata"}'
        spec=iso.acq.Spec(server.url,len(prior_tests.BODY),hashlib.md5(prior_tests.BODY).hexdigest(),hashlib.sha256(raw).hexdigest(),'synthetic','1')
        limits=iso.acq.Limits(max_requests=changes.pop('max_requests',4),max_storage_bytes=2000000,reserve_free_bytes=0,evidence_reserve_bytes=65536,timeout=2,chunk_bytes=2048)
        e=iso.IsolatedEngine(spec,self.root/self.cfg['data_relative'],self.root/self.cfg['output_relative'],
            {'helper_sha256':'a'*64,'config_sha256':'b'*64,'prior_archive_requests':1},limits,
            synthetic=True,sleep=lambda n:None,metadata_bytes=raw,history_root=self.root,declaration=self.cfg,**changes)
        return e,server
    def prefix(self,e):
        def halt(n):raise Interrupted('synthetic pause')
        e.sleep=halt
        with self.assertRaises(Interrupted):e.run()
        e.sleep=lambda n:None
        return e.owner
    def spec(self):
        binding={'config_sha256':'b'*64}
        return iso.make_spec(self.root,self.cfg,iso.ENTRY,sys.executable,{},binding)
    @contextlib.contextmanager
    def pure(self):
        with (mock.patch.object(Path,'mkdir',side_effect=AssertionError('mkdir')),
              mock.patch.object(iso.acq,'fetch_official_metadata',side_effect=AssertionError('network'))):
            yield
    def test_recorded_prior_request_and_budget(self):
        with self.pure():r=iso.verify_lineage(self.root,self.cfg)
        self.assertEqual(r['prior_archive_requests'],1);self.assertEqual(r['new_request_cap'],4)
        self.assertEqual(r['lifetime_request_cap'],5);self.assertEqual(r['historical_files_verified'],18)
    def test_fresh_isolated_success_and_old_preservation(self):
        e,s=self.engine();result=e.run()
        self.assertEqual(result['lifetime_archive_requests'],2)
        self.assertEqual(result['remaining_archive_requests'],3)
        self.assertEqual((e.data/iso.acq.NAME).read_bytes(),prior_tests.BODY)
        self.assertEqual(self.old_snapshot(),self.before)
        session=json.loads((e.data/'session.json').read_text())
        self.assertEqual(session['limits']['max_requests'],4)
        self.assertEqual(len(list(e.output.glob('event-*.json')))>=2,True)
    def test_no_old_validator_migration_new_last_modified(self):
        e,s=self.engine();e.run()
        self.assertEqual(e.latest()[1]['validator']['kind'],'last_modified')
        oldstate=next((self.root/self.cfg['prior_session']['output_relative']).glob('state-*.json'))
        self.assertIsNone(json.loads(oldstate.read_text())['validator'])
        self.assertIsNone(s.seen[0]['range']);self.assertIsNone(s.seen[0]['if_range'])
    def test_new_strong_etag_preferred(self):
        e,s=self.engine([{'etag':'"new-strong"'}]);e.run()
        self.assertEqual(e.latest()[1]['validator'],{'kind':'etag','value':'"new-strong"'})
    def test_fresh_validator_missing_zero_body_saved(self):
        e,s=self.engine([{'etag':None,'lm':None,'date':None}])
        with self.assertRaisesRegex(iso.acq.SafetyError,'Strong ETag'):e.run()
        self.assertEqual((e.data/'payload.part').stat().st_size,0)
        self.assertEqual(iso.request_budget(self.root,self.cfg)['lifetime_archive_requests'],2)
    def test_lifetime_cap_one_old_plus_four_new(self):
        e,s=self.engine([{'short':True}]*5)
        with self.assertRaisesRegex(iso.acq.SafetyError,'budget'):e.run()
        self.assertEqual(len(s.seen),4)
        budget=iso.request_budget(self.root,self.cfg)
        self.assertEqual(budget['lifetime_archive_requests'],5);self.assertEqual(budget['remaining_archive_requests'],0)
        before=iso.acq.hashes(e.data/'payload.part')
        with self.assertRaisesRegex(iso.acq.SafetyError,'budget'):e.run(e.owner)
        self.assertEqual(len(s.seen),4);self.assertEqual(before,iso.acq.hashes(e.data/'payload.part'))
    def test_resume_retains_budget_and_exact_if_range(self):
        e,s=self.engine([{'short':True},{}]);owner=self.prefix(e)
        self.assertEqual(iso.request_budget(self.root,self.cfg)['lifetime_archive_requests'],2)
        result=e.run(owner)
        self.assertEqual(result['lifetime_archive_requests'],3)
        self.assertEqual(s.seen[-1]['if_range'],prior_tests.LM)
        self.assertEqual(s.seen[-1]['range'],'bytes=8192-')
        self.assertEqual(self.old_snapshot(),self.before)
    def test_changed_validator_preserves_partial(self):
        e,s=self.engine([{'short':True},{'lm':prior_tests.LM_CHANGED}]);owner=self.prefix(e);h=iso.acq.hashes(e.data/'payload.part')
        with self.assertRaisesRegex(iso.acq.SafetyError,'Last-Modified changed'):e.run(owner)
        self.assertEqual(iso.acq.hashes(e.data/'payload.part'),h)
    def test_range_ignored_preserves_partial(self):
        e,s=self.engine([{'short':True},{'status':200}]);owner=self.prefix(e);h=iso.acq.hashes(e.data/'payload.part')
        with self.assertRaisesRegex(iso.acq.SafetyError,'Range'):e.run(owner)
        self.assertEqual(iso.acq.hashes(e.data/'payload.part'),h)
    def test_malformed_range_preserves_partial(self):
        e,s=self.engine([{'short':True},{'range':'bytes 8193-16383/16384'}]);owner=self.prefix(e)
        with self.assertRaisesRegex(iso.acq.SafetyError,'Range'):e.run(owner)
    def test_416_preserves_partial(self):
        e,s=self.engine([{'short':True},{'status':416}]);owner=self.prefix(e);h=iso.acq.hashes(e.data/'payload.part')
        with self.assertRaisesRegex(iso.acq.SafetyError,'416'):e.run(owner)
        self.assertEqual(iso.acq.hashes(e.data/'payload.part'),h)
    def test_checksum_failure_no_publication(self):
        e,s=self.engine([{'corrupt':True}])
        with self.assertRaisesRegex(iso.acq.SafetyError,'checksum'):e.run()
        self.assertFalse((e.data/iso.acq.NAME).exists())
    def test_occupied_new_data_rejected(self):
        e,s=self.engine();e.data.mkdir();(e.data/'unexpected').write_bytes(b'preserve')
        with self.assertRaisesRegex(iso.acq.SafetyError,'existing'):e.run()
        self.assertEqual(s.seen,[]);self.assertEqual((e.data/'unexpected').read_bytes(),b'preserve')
    def test_occupied_new_evidence_rejected(self):
        e,s=self.engine();e.output.mkdir();(e.output/'unexpected').write_bytes(b'preserve')
        with self.assertRaisesRegex(iso.acq.SafetyError,'existing'):e.run()
        self.assertEqual(s.seen,[])
    def test_existing_final_never_overwritten(self):
        e,s=self.engine();e.data.mkdir();(e.data/iso.acq.NAME).write_bytes(b'old')
        with self.assertRaisesRegex(iso.acq.SafetyError,'Final'):e.run()
        self.assertEqual((e.data/iso.acq.NAME).read_bytes(),b'old');self.assertEqual(s.seen,[])
    def test_wrong_resume_owner_rejected(self):
        e,s=self.engine([{'short':True}]);self.prefix(e)
        with self.assertRaisesRegex(iso.acq.SafetyError,'ownership'):e.run('f'*32)
        self.assertEqual(len(s.seen),1)
    def test_changed_new_binding_rejected(self):
        e,s=self.engine([{'short':True}]);owner=self.prefix(e);e.binding={'changed':'source'}
        with self.assertRaisesRegex(iso.acq.SafetyError,'ownership'):e.run(owner)
        self.assertEqual(len(s.seen),1)
    def test_old_destination_cannot_be_reused(self):
        e,s=self.engine()
        with self.assertRaisesRegex(iso.Stop,'Unsafe isolated'):
            iso.IsolatedEngine(e.spec,self.root/iso.OLD_PATHS[0],e.output,e.binding,e.limits,
                synthetic=True,history_root=self.root,declaration=self.cfg)
        self.assertEqual(self.old_snapshot(),self.before)
    def test_old_owner_cannot_resume_new_session(self):
        e,s=self.engine([{'short':True}]);self.prefix(e)
        with self.assertRaisesRegex(iso.acq.SafetyError,'ownership'):e.run(self.cfg['prior_session']['owner'])
    def test_five_new_requests_rejected(self):
        with self.assertRaisesRegex(iso.Stop,'four new requests'):self.engine(max_requests=5)
    def test_budget_reset_declaration_rejected(self):
        for key,value in [('max_requests',5),('lifetime_max_requests',6)]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(iso.Stop):iso.validate_declaration(cfg)
    def test_old_and_new_evidence_share_retained_budget(self):
        e,s=self.engine();lineage=iso.verify_lineage(self.root,self.cfg)
        with mock.patch.object(iso.acq.Engine,'space') as space:
            e.space(4096)
        space.assert_called_once_with(4096+lineage['prior_retained_bytes'])
    def test_exhausted_budget_rejected_by_wrapper_guard(self):
        e,s=self.engine([{'short':True}]*4)
        with self.assertRaisesRegex(iso.acq.SafetyError,'budget'):e.run()
        with (mock.patch.object(iso,'ROOT',self.root),
              mock.patch.object(iso,'load_declaration',return_value=self.cfg)):
            with self.assertRaisesRegex(iso.Stop,'No remaining'):iso.check_resume(self.spec(),True)
        self.assertEqual(len(s.seen),4)
    def test_full_composed_preflight_offline(self):
        good=types.SimpleNamespace(f_flag=0,f_bavail=100000000000,f_frsize=1)
        with self.pure(),mock.patch.object(iso.bg.os,'fstatvfs',return_value=good),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(iso.main(['preflight']),0)
        result=json.loads(out.getvalue())
        self.assertEqual(result['session_action'],'start')
        self.assertEqual(result['remaining_archive_requests'],4)
        self.assertFalse(result['download_authorized']);self.assertEqual(result['network_requests'],0)
    def test_engine_reuses_original_validator_and_publication_methods(self):
        self.assertIs(iso.IsolatedEngine.publish,iso.acq.Engine.publish)
        self.assertIs(iso.IsolatedEngine.checkpoint,iso.acq.Engine.checkpoint)
        self.assertIs(iso.IsolatedEngine.latest,iso.acq.Engine.latest)
        self.assertTrue(iso.acq.tls_context().check_hostname)
    def test_prior_consumed_zero_rejected(self):
        self.cfg['prior_session']['archive_requests_consumed']=0
        with self.assertRaisesRegex(iso.Stop,'reset'):iso.verify_lineage(self.root,self.cfg)
    def test_boolean_request_count_rejected(self):
        self.cfg['max_requests']=True
        with self.assertRaises(iso.Stop):iso.validate_declaration(self.cfg)
    def test_added_old_request_blocks_before_transport(self):
        e,s=self.engine()
        p=self.root/self.cfg['prior_session']['output_relative']/('request-'+'f'*32+'.json');p.write_text('{}')
        with self.assertRaisesRegex(iso.Stop,'set changed'):e.run()
        self.assertEqual(s.seen,[])
    def test_deleted_old_request_blocks(self):
        req=next((self.root/iso.OLD_PATHS[1]).glob('request-*.json'));req.unlink()
        with self.assertRaisesRegex(iso.Stop,'set changed'):iso.verify_lineage(self.root,self.cfg)
    def test_old_metadata_content_tamper_rejected(self):
        p=self.root/iso.OLD_PATHS[1]/'official_metadata.json';p.write_text('{}')
        with self.assertRaisesRegex(iso.Stop,'mismatch'):iso.verify_lineage(self.root,self.cfg)
    def test_old_payload_nonempty_rejected(self):
        (self.root/iso.OLD_PATHS[0]/'payload.part').write_bytes(b'x')
        with self.assertRaisesRegex(iso.Stop,'mismatch'):iso.verify_lineage(self.root,self.cfg)
    def test_old_file_timestamp_change_rejected(self):
        p=self.root/iso.OLD_PATHS[0]/'payload.part';s=p.stat();os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1))
        with self.assertRaisesRegex(iso.Stop,'mismatch'):iso.verify_lineage(self.root,self.cfg)
    def test_historical_writer_lock_blocks(self):
        e,s=self.engine();p=self.root/iso.OLD_PATHS[0]/'.acquisition.lock'
        with p.open('rb') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(iso.Stop,'writer is active'):e.run()
        self.assertEqual(s.seen,[])
    def test_read_lock_preserves_old_bytes_and_inode(self):
        p=self.root/iso.OLD_PATHS[0]/'.acquisition.lock';s=p.stat()
        with iso.history_read_lock(self.root,self.cfg):
            with p.open('rb') as f:
                with self.assertRaises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertEqual((p.stat().st_ino,p.stat().st_mtime_ns,p.read_bytes()),(s.st_ino,s.st_mtime_ns,b''))
    def test_symlink_new_destination_rejected(self):
        e,s=self.engine();e.data.parent.mkdir(parents=True,exist_ok=True);e.data.symlink_to(self.root/iso.OLD_PATHS[0],target_is_directory=True)
        with self.assertRaises(iso.acq.SafetyError):e.run()
        self.assertEqual(s.seen,[])
    def test_symlink_old_evidence_rejected(self):
        p=self.root/iso.OLD_PATHS[0]/'payload.part';p.unlink();p.symlink_to(self.root/'missing')
        with self.assertRaises((iso.Stop,iso.bg.Stop)):iso.verify_lineage(self.root,self.cfg)
    def test_pure_preflight_does_not_create_roots(self):
        spec=self.spec();good=types.SimpleNamespace(f_flag=0,f_bavail=100000000000,f_frsize=1)
        with self.pure(),mock.patch.object(iso.bg.os,'fstatvfs',return_value=good):
            iso.verify_lineage(self.root,self.cfg)
            self.assertIsNone(iso._original_check_resume(spec,False))
            r=iso.bg.inspect_capacity(spec['data'])
        self.assertFalse(r['directories_created']);self.assertFalse(Path(spec['root']).exists())
    def test_insufficient_capacity_no_roots(self):
        spec=self.spec();bad=types.SimpleNamespace(f_flag=0,f_bavail=1,f_frsize=1)
        with self.pure(),mock.patch.object(iso.bg.os,'fstatvfs',return_value=bad):
            with self.assertRaisesRegex(iso.bg.Stop,'80 GB'):iso.bg.inspect_capacity(spec['data'])
        self.assertFalse(Path(spec['root']).exists())
    def test_dry_run_zero_network_no_roots(self):
        spec=self.spec()
        with self.pure(),mock.patch.object(iso.bg,'check_sources'):
            result=iso.bg.launch(spec)
        self.assertEqual(result['status'],'dry_run');self.assertFalse(result['transfer_enabled'])
    def test_quota_gate_before_local_writes(self):
        spec=self.spec()
        with self.pure(),mock.patch.object(iso.bg,'check_sources'):
            with self.assertRaisesRegex(iso.bg.Stop,'quota'):iso.bg.launch(spec,authorized=True)
    def test_wrong_config_hash_rejected_offline(self):
        with mock.patch.object(iso,'CONFIG_SHA','0'*64):
            with self.assertRaisesRegex(iso.Stop,'configuration hash'):iso.load_declaration()
    def test_original_source_pin_failure_rejected(self):
        pins=dict(iso.PINS);pins['tools/acquire_march_2022_v2.py']='0'*64
        with mock.patch.object(iso,'PINS',pins):
            with self.assertRaisesRegex(iso.Stop,'Protected source'):iso.load_declaration()
    def test_production_spec_source_binding_and_paths(self):
        with self.pure():spec=iso.production_spec()
        self.assertEqual(spec['data'],str(ROOT/iso.NEW_PATHS[0]))
        self.assertEqual(spec['root'],str(ROOT/iso.NEW_PATHS[2]))
        self.assertEqual(spec['limits']['max_requests'],4)
        self.assertEqual(spec['binding']['prior_archive_requests'],1)
        for p in iso.PINS:self.assertIn(str(ROOT/p),spec['source_sha256'])
        self.assertEqual(spec['command'][-3:],['_acquire','--authorize-transfer','--quota-confirmed'])
    def test_actual_cli_dry_run_is_offline(self):
        with self.pure(),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(iso.main(['start']),0)
        self.assertEqual(json.loads(output.getvalue())['status'],'dry_run')
        self.assertFalse((ROOT/iso.NEW_PATHS[2]).exists())
    def test_worker_requires_quota_before_metadata(self):
        spec=self.spec()
        with mock.patch.object(iso,'production_spec',return_value=spec),mock.patch.object(iso,'load_declaration',return_value=self.cfg),self.pure():
            with self.assertRaisesRegex(iso.Stop,'quota'):iso.acquire(['--authorize-transfer'])
    def test_immutable_existing_completed_run_cannot_start(self):
        e,s=self.engine();e.run()
        with self.assertRaisesRegex(iso.acq.SafetyError,'Final'):e.run()
        self.assertEqual(len(s.seen),1)
    def test_new_receipt_owner_tamper_rejected(self):
        e,s=self.engine([{'short':True}]);self.prefix(e);p=next(e.output.glob('request-*.json'))
        d=json.loads(p.read_text());d['owner']='0'*32;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(iso.Stop,'request receipt'):iso.request_budget(self.root,self.cfg)
    def test_new_receipt_source_tamper_rejected(self):
        e,s=self.engine([{'short':True}]);self.prefix(e);p=next(e.output.glob('request-*.json'))
        d=json.loads(p.read_text());d['url']='https://unrelated.invalid/archive';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(iso.Stop,'request receipt'):iso.request_budget(self.root,self.cfg)
    def test_existing_fresh_guard_rejects_without_resume(self):
        spec=self.spec();Path(spec['data']).mkdir()
        with self.assertRaisesRegex(iso.bg.Stop,'Existing acquisition roots'):iso._original_check_resume(spec,False)
    def test_no_old_paths_in_worker_command(self):
        spec=self.spec();self.assertEqual(spec['command'][3],str(iso.ENTRY))
        self.assertNotIn('acquire_march_2022_v1.py',' '.join(spec['command']))
    def test_published_verifier_rehashes_contents_and_budget(self):
        e,s=self.engine();e.run();spec=self.spec()
        spec.update(identity=e.spec.identity(),binding=e.binding,limits=e.limits.__dict__)
        result=iso._original_verify_archive(spec)
        self.assertTrue(result['full_archive_rehashed_now'])
        self.assertEqual(result['hashes']['sha256'],hashlib.sha256(prior_tests.BODY).hexdigest())
        self.assertEqual(iso.request_budget(self.root,self.cfg)['lifetime_archive_requests'],2)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IsolatedSessionTests))
    print(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                      'skipped':len(result.skipped),'fixture_root':str(IsolatedSessionTests.work),
                      'loopback_requests':len(prior_tests.REQUESTS),'official_requests':0,
                      'production_transfer_started':False},sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
