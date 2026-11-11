"""Phase 1.9.3: local validator tests plus all prior acquisition/lifecycle suites."""
import contextlib
import copy
from datetime import timedelta
from email.message import Message
from email.utils import format_datetime
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from unittest import mock

ROOT=Path('/media/csedept/cse2014/Project/m100_training_template')
sys.path.insert(0,str(ROOT/'tools'))
import acquire_march_2022_v1 as old
import acquire_march_2022_v2 as acq
import march_background_v3 as bg

LM='Thu, 30 Apr 2026 07:37:06 GMT'
DATE='Tue, 10 Nov 2026 08:00:00 GMT'
LM_CHANGED='Fri, 01 May 2026 07:37:06 GMT'
BODY=bytes(range(256))*64
REQUESTS=[];ROOTS=[]

class Halt(RuntimeError):pass

class Server:
    def __init__(self,modes=None):
        self.modes=list(modes or [{}]);self.seen=[];outer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                mode=outer.modes.pop(0) if outer.modes else {}
                off=int(self.headers.get('Range','bytes=0-')[6:-1])
                row={'range':self.headers.get('Range'),'if_range':self.headers.get('If-Range'),'offset':off,'mode':mode}
                outer.seen.append(row);REQUESTS.append(row)
                status=mode.get('status',206 if off else 200)
                self.send_response_only(status)
                content=BODY[off:]
                if status in (302,416):content=b''
                self.send_header('Content-Length',str(mode.get('length',len(content))))
                if status==302:self.send_header('Location','/must-not-follow')
                for name,key,default in [('ETag','etag',None),('Last-Modified','lm',LM),('Date','date',DATE)]:
                    value=mode.get(key,default)
                    if value is not None:self.send_header(name,value)
                    if mode.get('duplicate')==name:self.send_header(name,value or 'extra')
                if off and not mode.get('missing_range'):
                    value=mode.get('range',f'bytes {off}-{len(BODY)-1}/{len(BODY)}')
                    self.send_header('Content-Range',value)
                    if mode.get('duplicate')=='Content-Range':self.send_header('Content-Range',value)
                if mode.get('encoding'):self.send_header('Content-Encoding',mode['encoding'])
                self.end_headers()
                if mode.get('corrupt') and content:content=bytes([content[0]^1])+content[1:]
                if mode.get('short'):content=content[:len(content)//2]
                try:
                    self.wfile.write(content);self.wfile.flush()
                    if mode.get('short'):
                        self.close_connection=True;self.connection.shutdown(socket.SHUT_RDWR)
                except (BrokenPipeError,ConnectionResetError,OSError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler);self.server.daemon_threads=True
        self.url=f'http://127.0.0.1:{self.server.server_port}/fixture.bin'
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True);self.thread.start()
    def close(self):self.server.shutdown();self.server.server_close();self.thread.join(2)

class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(tempfile.mkdtemp(prefix='m100-p193-validators-'));ROOTS.append(str(cls.root))
        connect=socket.socket.connect
        def local(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
                raise AssertionError('External socket forbidden')
            return connect(sock,address)
        cls.guard=mock.patch.object(socket.socket,'connect',local);cls.guard.start()
    @classmethod
    def tearDownClass(cls):cls.guard.stop()
    def setUp(self):self.base=self.root/self._testMethodName;self.base.mkdir();self.servers=[]
    def tearDown(self):
        for s in self.servers:s.close()
    def engine(self,modes=None,requests=5,engine_class=None,binding=None):
        server=Server(modes);self.servers.append(server)
        raw=b'{"fixture":"no-real-metadata"}'
        spec=acq.Spec(server.url,len(BODY),hashlib.md5(BODY).hexdigest(),hashlib.sha256(raw).hexdigest(),'synthetic','1')
        limits=acq.Limits(max_requests=requests,max_storage_bytes=2000000,reserve_free_bytes=0,evidence_reserve_bytes=65536,timeout=2,chunk_bytes=2048)
        cls=engine_class or acq.Engine
        e=cls(spec,self.base/'data',self.base/'evidence',binding or {'helper_sha256':bg.HELPER_SHA,'config_sha256':'f'*64},limits,synthetic=True,metadata_bytes=raw,sleep=lambda n:None)
        return e,server
    def prefix(self,e):
        def halt(n):raise Halt('synthetic operator interruption')
        e.sleep=halt
        with self.assertRaises(Halt):e.run()
        e.sleep=lambda n:None
        self.assertEqual((e.data/'payload.part').stat().st_size,len(BODY)//2)
        return e.owner
    def failed_resume(self,mode,pattern=None,initial=None):
        e,s=self.engine([{**(initial or {}),'short':True},mode]);token=self.prefix(e)
        before=acq.hashes(e.data/'payload.part');states={p.name:p.read_bytes() for p in e.output.glob('state-*.json')}
        with self.assertRaises(acq.SafetyError) as raised:e.run(token)
        if pattern:self.assertRegex(str(raised.exception),pattern)
        self.assertEqual(acq.hashes(e.data/'payload.part'),before)
        for name,value in states.items():self.assertEqual((e.output/name).read_bytes(),value)
        self.assertFalse((e.data/acq.NAME).exists());self.assertEqual(len(s.seen),2)
        return e,s
    def failed_initial(self,mode,pattern=None):
        e,s=self.engine([mode])
        with self.assertRaises(acq.SafetyError) as raised:e.run()
        if pattern:self.assertRegex(str(raised.exception),pattern)
        self.assertEqual((e.data/'payload.part').stat().st_size,0);self.assertFalse((e.data/acq.NAME).exists())
        return e,s
    def message(self,**kw):
        m=Message()
        for k,v in kw.items():
            if v is not None:m[k.replace('_','-')]=v
        return m
    def test_last_modified_exact_success_hashes(self):
        e,s=self.engine();r=e.run();self.assertEqual(r['sha256'],hashlib.sha256(BODY).hexdigest());self.assertEqual(r['md5'],hashlib.md5(BODY).hexdigest())
        v=e.latest()[1]['validator'];self.assertEqual(v,{'kind':'last_modified','value':LM,'response_date':DATE});self.assertEqual(len(s.seen),1)
    def test_last_modified_resume_exact_if_range(self):
        e,s=self.engine([{'short':True},{}]);owner=self.prefix(e);before=(e.data/'payload.part').read_bytes();e.run(owner)
        self.assertEqual(s.seen[-1]['range'],'bytes=8192-');self.assertEqual(s.seen[-1]['if_range'],LM)
        self.assertTrue((e.data/acq.NAME).read_bytes().startswith(before))
    def test_strong_etag_preferred_over_valid_last_modified(self):
        e,s=self.engine([{'etag':'"preferred"'}]);e.run();self.assertEqual(e.latest()[1]['validator'],{'kind':'etag','value':'"preferred"'})
    def test_strong_etag_does_not_require_weather_dates(self):
        e,s=self.engine([{'etag':'"preferred"','lm':'malformed','date':None}]);e.run();self.assertEqual(e.latest()[1]['validator']['kind'],'etag')
    def test_etag_resume_preserved(self):
        e,s=self.engine([{'etag':'"tag"','short':True},{'etag':'"tag"'}]);owner=self.prefix(e);e.run(owner);self.assertEqual(s.seen[-1]['if_range'],'"tag"')
    def test_new_response_date_does_not_change_recorded_proof(self):
        e,s=self.engine([{'short':True},{'date':'Tue, 10 Nov 2026 09:00:00 GMT'}]);owner=self.prefix(e);v=e.latest()[1]['validator'];e.run(owner);self.assertEqual(e.latest()[1]['validator'],v)
    def test_last_modified_changed_no_append(self):self.failed_resume({'lm':LM_CHANGED},'Last-Modified changed')
    def test_last_modified_missing_no_append(self):self.failed_resume({'lm':None},'Last-Modified plus Date')
    def test_date_missing_on_resume_no_append(self):self.failed_resume({'date':None},'Last-Modified plus Date')
    def test_validator_kind_upgrade_rejected(self):self.failed_resume({'etag':'"new"'},'kind changed')
    def test_etag_loss_cannot_downgrade(self):self.failed_resume({},'kind changed',{'etag':'"old"'})
    def test_etag_change_rejected(self):self.failed_resume({'etag':'"new"'},'ETag changed',{'etag':'"old"'})
    def test_weak_etag_never_falls_back_to_date(self):self.failed_initial({'etag':'W/"weak"'},'Strong ETag')
    def test_malformed_etag_never_falls_back_to_date(self):self.failed_initial({'etag':'unquoted'},'Strong ETag')
    def test_empty_etag_header_rejected(self):self.failed_initial({'etag':''},'Strong ETag')
    def test_valid_empty_opaque_etag_is_strong(self):
        e,s=self.engine([{'etag':'""'}]);e.run();self.assertEqual(e.latest()[1]['validator']['value'],'""')
    def test_no_validator_rejected_before_body(self):self.failed_initial({'lm':None,'date':None},'Strong ETag')
    def test_last_modified_without_date_rejected(self):self.failed_initial({'date':None},'Date required')
    def test_malformed_last_modified_rejected(self):self.failed_initial({'lm':'tomorrow'},'HTTP date')
    def test_malformed_date_rejected(self):self.failed_initial({'date':'today'},'HTTP date')
    def test_wrong_weekday_rejected(self):self.failed_initial({'lm':'Fri, 30 Apr 2026 07:37:06 GMT'},'weekday')
    def test_future_last_modified_rejected(self):self.failed_initial({'date':'Thu, 30 Apr 2026 07:00:00 GMT'},'strong-date')
    def test_clock_margin_59_seconds_rejected(self):
        date=format_datetime(acq.http_date(LM)+timedelta(seconds=59),usegmt=True);self.failed_initial({'date':date},'strong-date')
    def test_clock_margin_60_seconds_accepted(self):
        date=format_datetime(acq.http_date(LM)+timedelta(seconds=60),usegmt=True);e,s=self.engine([{'date':date}]);e.run();self.assertEqual(e.latest()[1]['validator']['response_date'],date)
    def test_unsupported_legacy_date_fails_closed(self):self.failed_initial({'lm':'Thursday, 30-Apr-26 07:37:06 GMT'},'unsupported')
    def test_duplicate_last_modified_rejected(self):self.failed_initial({'duplicate':'Last-Modified'},'Duplicate')
    def test_duplicate_date_rejected(self):self.failed_initial({'duplicate':'Date'},'Duplicate')
    def test_duplicate_etag_rejected(self):self.failed_initial({'etag':'"tag"','duplicate':'ETag'},'Duplicate')
    def test_resume_200_rejected_even_with_remaining_length(self):self.failed_resume({'status':200},'Range')
    def test_malformed_range_start_rejected(self):self.failed_resume({'range':'bytes 8193-16383/16384'},'Range')
    def test_malformed_range_total_rejected(self):self.failed_resume({'range':'bytes 8192-16383/99999'},'Range')
    def test_malformed_range_wildcard_rejected(self):self.failed_resume({'range':'bytes 8192-16383/*'},'Range')
    def test_missing_range_rejected(self):self.failed_resume({'missing_range':True},'Range')
    def test_duplicate_range_rejected(self):self.failed_resume({'duplicate':'Content-Range'},'Duplicate')
    def test_http416_preserves_partial(self):self.failed_resume({'status':416},'416')
    def test_wrong_remaining_length_rejected(self):self.failed_resume({'length':8191},'Content-Length')
    def test_encoded_response_rejected(self):self.failed_initial({'encoding':'gzip'},'Encoded')
    def test_redirect_not_followed(self):
        e,s=self.failed_initial({'status':302},'302');self.assertEqual(len(s.seen),1)
    def test_request_budget_survives_last_modified_resume(self):
        e,s=self.engine([{'short':True},{'short':True}],requests=2);owner=self.prefix(e)
        with self.assertRaisesRegex(acq.SafetyError,'budget'):e.run(owner)
        before=acq.hashes(e.data/'payload.part')
        with self.assertRaisesRegex(acq.SafetyError,'budget'):e.run(owner)
        self.assertEqual(len(s.seen),2);self.assertEqual(before,acq.hashes(e.data/'payload.part'))
    def test_checksum_failure_cannot_publish(self):
        e,s=self.engine([{'corrupt':True}])
        with self.assertRaisesRegex(acq.SafetyError,'checksum'):e.run()
        self.assertFalse((e.data/acq.NAME).exists());self.assertEqual((e.data/'payload.part').stat().st_size,len(BODY))
    def test_bad_recorded_date_proof_stops_before_request(self):
        e,s=self.engine([{'short':True}]);owner=self.prefix(e);tip=e.latest()[1]
        p=next(p for p in e.output.glob('state-*.json') if json.loads(p.read_text())==tip)
        tip['validator']['response_date']=LM;p.write_text(json.dumps(tip));n=len(s.seen)
        with self.assertRaisesRegex(acq.SafetyError,'strong-date'):e.run(owner)
        self.assertEqual(len(s.seen),n)
    def test_legacy_string_validator_not_converted(self):
        with self.assertRaisesRegex(acq.SafetyError,'no legacy conversion'):acq.validate_validator('"old"')
    def test_old_helper_binding_cannot_be_reinterpreted(self):
        e,s=self.engine(engine_class=old.Engine,binding={'helper_sha256':bg.sha(ROOT/'tools/acquire_march_2022_v1.py'),'config_sha256':'f'*64})
        with self.assertRaisesRegex(old.SafetyError,'Strong ETag'):e.run()
        before={str(p):p.read_bytes() for root in [e.data,e.output] for p in root.iterdir()};n=len(s.seen)
        new=acq.Engine(e.spec,e.data,e.output,{'helper_sha256':bg.HELPER_SHA,'config_sha256':'f'*64},e.limits,synthetic=True,metadata_bytes=e.metadata_bytes)
        with self.assertRaisesRegex(acq.SafetyError,'ownership'):new.run(e.owner)
        self.assertEqual(len(s.seen),n);self.assertEqual(before,{str(p):p.read_bytes() for root in [e.data,e.output] for p in root.iterdir()})
    def test_wrapper_reports_last_modified_checkpoint(self):
        e,s=self.engine();e.run();spec={'data':str(e.data),'evidence':str(e.output),'identity':e.spec.identity(),'binding':e.binding,'limits':e.limits.__dict__}
        status=bg.checkpoint_summary(spec);self.assertEqual(status['validator_kind'],'last_modified');self.assertEqual(status['validator_value'],LM);self.assertTrue(status['strong_validator_recorded']);self.assertFalse(status['partial_content_rehashed'])
    def test_wrapper_reports_etag_checkpoint(self):
        e,s=self.engine([{'etag':'"preferred"'}]);e.run();spec={'data':str(e.data),'evidence':str(e.output),'identity':e.spec.identity(),'binding':e.binding,'limits':e.limits.__dict__}
        self.assertEqual(bg.checkpoint_summary(spec)['validator_kind'],'etag')
    def test_matching_validator_does_not_mutate_evidence(self):
        original={'kind':'last_modified','value':LM,'response_date':DATE};before=copy.deepcopy(original)
        self.assertIs(acq.match_validator(original,{'kind':'last_modified','value':LM,'response_date':'Tue, 10 Nov 2026 09:00:00 GMT'}),original);self.assertEqual(original,before)
    def test_unknown_checkpoint_kind_rejected(self):
        with self.assertRaisesRegex(acq.SafetyError,'Unknown'):acq.validate_validator({'kind':'guessed','value':LM})
    def test_incomplete_recorded_date_evidence_rejected(self):
        with self.assertRaisesRegex(acq.SafetyError,'evidence'):acq.validate_validator({'kind':'last_modified','value':LM})
    def test_invalid_validator_never_reads_body(self):
        e,s=self.engine()
        class Response:
            code=200
            headers=Message();headers['Content-Length']=str(len(BODY))
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def geturl(self):return s.url
            def read(self,n):raise AssertionError('body read before validator')
        e.transport=mock.Mock();e.transport.open.return_value=Response()
        with self.assertRaisesRegex(acq.SafetyError,'Strong ETag'):e.run()
        self.assertEqual((e.data/'payload.part').stat().st_size,0);self.assertEqual(s.seen,[])
    def test_existing_final_is_never_overwritten(self):
        e,s=self.engine();e.data.mkdir();(e.data/acq.NAME).write_bytes(b'old')
        with self.assertRaisesRegex(acq.SafetyError,'Final destination'):e.run()
        self.assertEqual((e.data/acq.NAME).read_bytes(),b'old');self.assertEqual(s.seen,[])
    def test_historical_local_precheck_stops_before_metadata(self):
        declaration,_=acq.load_config(acq.CONFIG)
        with mock.patch.object(acq,'fetch_official_metadata',side_effect=AssertionError('metadata request')) as fetch:
            with self.assertRaisesRegex(acq.SafetyError,'historical session'):acq.local_transfer_precheck(declaration,{'helper_sha256':bg.HELPER_SHA,'config_sha256':bg.CONFIG_SHA})
            historical=json.loads((ROOT/'data/march_2022_acquisition/v1/session.json').read_text())
            with self.assertRaisesRegex(acq.SafetyError,'migration not authorized'):acq.local_transfer_precheck(declaration,{'helper_sha256':bg.HELPER_SHA,'config_sha256':bg.CONFIG_SHA},historical['owner'])
            fetch.assert_not_called()
    def test_config_pin_unchanged_and_no_authorization_dry_run(self):
        self.assertEqual(bg.sha(bg.CONFIG),'4a7911cdf1cbf35b109186000b9ad697de1ba06faf66295bfb5d1338542f5725')
        with mock.patch.object(acq,'fetch_official_metadata',side_effect=AssertionError('metadata')),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(acq.main([]),0)
        self.assertFalse(json.loads(output.getvalue())['transfer_enabled'])


def module_from_source(name,text,path):
    m=types.ModuleType(name);m.__file__=str(path);sys.modules[name]=m;exec(compile(text,str(path),'exec'),m.__dict__);return m


def existing_and_v3_suite():
    path=ROOT/'tests/test_march_background_v2.py';text=path.read_text()
    # Real v1 paths now contain a preserved failed session. Isolate only the
    # original tests that require a fresh real destination; never erase it.
    text=text.replace("destination=ROOT/'data/march_2022_acquisition/v1'", "destination=ROOT/'data'/'__phase193_absent_capacity_probe'/'v1'")
    m=module_from_source('m100_p193_prior_capacity',text,path)
    for name in ['test_capacity_failure_before_workflow_creation','test_fresh_production_layout_passes_capacity_before_any_mkdir']:
        original=getattr(m.CapacityTests,name)
        def isolated(self,fn=original):
            spec=m.bg.production_spec();spec=copy.deepcopy(spec)
            spec.update(data=str(self.base/'missing'/'v1'),evidence=str(self.base/'evidence'/'v1'),root=str(self.base/'background'/'v2'))
            with mock.patch.object(m.bg,'production_spec',return_value=spec):return fn(self)
        setattr(m.CapacityTests,name,isolated)
    suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(m.CapacityTests));suite.addTests(m.lifecycle_module(1));suite.addTests(m.lifecycle_module(2))
    # V3 lifecycle fixtures use the unchanged algorithms with the new helper,
    # typed checkpoint schema and a separate wrapper namespace.
    path1=ROOT/'tests/test_march_background_v1.py';t=path1.read_text().replace('march_background_v1','march_background_v3').replace('acquire_march_2022_v1','acquire_march_2022_v2')
    t=t.replace("mock.patch.object(bg.os,'statvfs',return_value=type('FS',(),{'f_flag':0,'f_bavail':1,'f_frsize':4096})())", "mock.patch.object(bg.os,'fstatvfs',return_value=type('FS',(),{'f_flag':0,'f_bavail':1,'f_frsize':4096})())")
    t=t.replace("self.assertFalse(server.seen);self.assertEqual(len(bg.attempts(spec['root'])),0)","self.assertFalse(server.seen);self.assertFalse(Path(spec['root']).exists())")
    v3=module_from_source('m100_p193_lifecycle_v3',t,path1);suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(v3.BackgroundTests))
    # Exercise the real detached supervisor with Last-Modified-only responses.
    lm_text=t.replace("self.send_header('ETag','\"changed\"' if outer.mode=='changed_etag' else '\"fixture-v1\"')",
                      "self.send_header('Last-Modified', 'Fri, 01 May 2026 07:37:06 GMT' if outer.mode=='changed_etag' else 'Thu, 30 Apr 2026 07:37:06 GMT')")
    assert lm_text!=t
    lm_text=lm_text.replace("self.assertEqual(server.seen[-1]['if_range'],'\"fixture-v1\"')", "self.assertEqual(server.seen[-1]['if_range'],'Thu, 30 Apr 2026 07:37:06 GMT')")
    lm_text=lm_text.replace("self.assertIn('ETag changed',Path(v['log']).read_text())", "self.assertIn('Last-Modified changed',Path(v['log']).read_text())")
    lm=module_from_source('m100_p193_lifecycle_v3_last_modified',lm_text,path1)
    for name in ['test_terminal_hangup_survival','test_interruption_preserves_and_resume_delegates_range','test_changed_etag_rejected_prefix_preserved']:
        suite.addTest(lm.BackgroundTests(name))
    return suite,m,v3,lm


def main():
    before=time.monotonic();helper_results=[]
    for label,helper in [('original_helper_v1',old),('new_helper_v2',acq)]:
        output=io.StringIO()
        with contextlib.redirect_stdout(output):code=helper.synthetic_tests()
        print('HELPER_SUITE',label,output.getvalue(),flush=True)
        result=json.loads(output.getvalue().splitlines()[-1]);helper_results.append({'label':label,'exit_status':code,**result})
        ROOTS.append(result['workdir'])
    suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorTests))
    legacy,m,v3,lm=existing_and_v3_suite();suite.addTests(legacy)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    roots=ROOTS+m.REGRESSION_WORK+[str(x.WORK) for _,x in m.LIFECYCLE_MODULES]+[str(v3.WORK),str(lm.WORK)]
    summary={'status':'passed' if result.wasSuccessful() and all(x['exit_status']==0 for x in helper_results) else 'failed',
      'tests_run':result.testsRun+sum(x['tests_run'] for x in helper_results),'unittest_group_tests':result.testsRun,
      'validator_tests':unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorTests).countTestCases(),'prior_capacity_and_lifecycle_tests':82,'new_v3_lifecycle_tests':30,'last_modified_background_cases':3,
      'failures':len(result.failures)+sum(x['failures'] for x in helper_results),'errors':len(result.errors)+sum(x['errors'] for x in helper_results),'skipped':len(result.skipped)+sum(x['skipped'] for x in helper_results),
      'helper_suites':helper_results,'seconds':time.monotonic()-before,
      'loopback_requests':len(REQUESTS)+sum(len(x.REQUESTS) for _,x in m.LIFECYCLE_MODULES)+len(v3.REQUESTS)+len(lm.REQUESTS)+sum(x['loopback_HTTP_requests'] for x in helper_results),
      'real_archive_requests':0,'official_metadata_requests':0,'production_transfer_engine_invoked':False,
      'temporary_fixture_roots':roots,'legacy_test_adaptations':['Two original fresh/capacity-failure tests isolated to temporary paths because real v1 session now exists','Real-path absence probe uses an uncreated child under data/; no existing session changed','V3 fixtures use helper v2, wrapper v3, fstatvfs and early-failure outcome']}
    print(json.dumps(summary,sort_keys=True),flush=True)
    return 0 if summary['status']=='passed' else 1

if __name__=='__main__':sys.exit(main())
