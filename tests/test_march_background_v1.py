"""Lifecycle tests: only /tmp fixtures and literal loopback HTTP, no production CLI transfer."""
import copy
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import pty
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

ROOT = Path('/media/csedept/cse2014/Project/m100_training_template')
WRAPPER = ROOT/'tools/march_background_v1.py'
HELPER = ROOT/'tools/acquire_march_2022_v1.py'
sys.path.insert(0, str(ROOT/'tools'))
import march_background_v1 as bg
import acquire_march_2022_v1 as acq

WORK = Path(os.environ.get('M100_BG_TEST_ROOT') or tempfile.mkdtemp(prefix='m100-p191-tests-'))
WORK.mkdir(exist_ok=True)
BODY = bytes(range(256))*256
REQUESTS = []
PRODUCTION_GUARDS = []

WORKER = '''import sys,json,socket,ipaddress
from pathlib import Path
sys.path.insert(0,TOOLS)
import acquire_march_2022_v1 as a
# Guard all worker sockets against non-loopback destinations.
real_connect=socket.socket.connect
def connect(s,address):
 if s.family in (socket.AF_INET,socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
  raise AssertionError('External network forbidden')
 return real_connect(s,address)
socket.socket.connect=connect
c=json.loads(Path(sys.argv[1]).read_text())
if c.get('mode')=='exit_zero': print('zero is not verified',flush=True);sys.exit(0)
if c.get('mode')=='exit_bad': print('fixture failure',flush=True);sys.exit(7)
spec=a.Spec(**c['source'])
engine=a.Engine(spec,Path(c['data']),Path(c['evidence']),c['binding'],a.Limits(**c['limits']),synthetic=True,metadata_bytes=c['metadata'].encode())
owner=None if '--resume-owner' not in sys.argv else sys.argv[sys.argv.index('--resume-owner')+1]
print('fixture helper starts',flush=True)
try: print(json.dumps(engine.run(owner)),flush=True)
except (a.SafetyError,OSError,ValueError) as error:
 print('fixture safety stop: '+str(error),flush=True);sys.exit(2)
'''.replace('TOOLS',repr(str(ROOT/'tools')))

SUPERVISOR = '''import sys,json
from pathlib import Path
sys.path.insert(0,TOOLS)
import march_background_v1 as bg
spec=json.loads(Path(sys.argv[1]).read_text())
sys.exit(bg.supervise(spec,sys.argv[2],int(sys.argv[3])))
'''.replace('TOOLS',repr(str(ROOT/'tools')))

class Server:
    def __init__(self, slow=False, mode='ok'):
        self.slow=slow;self.mode=mode;self.seen=[];outer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                offset=int(self.headers.get('Range','bytes=0-')[6:-1])
                row={'range':self.headers.get('Range'),'if_range':self.headers.get('If-Range'),'offset':offset,'mode':outer.mode}
                outer.seen.append(row);REQUESTS.append(row)
                if outer.mode=='ignore_range': offset=0
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length',str(len(BODY)-offset))
                self.send_header('ETag','"changed"' if outer.mode=='changed_etag' else '"fixture-v1"')
                if offset:self.send_header('Content-Range',f'bytes {offset}-{len(BODY)-1}/{len(BODY)}')
                self.end_headers()
                data=BODY[offset:]
                if outer.mode=='corrupt':data=bytes([data[0]^1])+data[1:]
                try:
                    for i in range(0,len(data),512):
                        self.wfile.write(data[i:i+512]);self.wfile.flush()
                        if outer.slow:time.sleep(.012)
                except (BrokenPipeError,ConnectionResetError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.daemon_threads=True
        self.url=f'http://127.0.0.1:{self.server.server_port}/fixture.bin'
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True);self.thread.start()
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2)

class BackgroundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_connect=socket.socket.connect
        def loopback_only(sock,address):
            if sock.family in (socket.AF_INET,socket.AF_INET6) and not ipaddress.ip_address(address[0]).is_loopback:
                raise AssertionError('External network forbidden by lifecycle tests')
            return real_connect(sock,address)
        cls.socket_guard=mock.patch.object(socket.socket,'connect',loopback_only)
        cls.socket_guard.start()
    @classmethod
    def tearDownClass(cls):cls.socket_guard.stop()
    def setUp(self):
        self.base=WORK/(self._testMethodName+'-'+os.urandom(4).hex());self.base.mkdir()
        self.servers=[];self.specs=[]
    def tearDown(self):
        for spec in self.specs:
            try:
                value=bg.status(spec)
                if value.get('running'):
                    bg.stop(spec);self.wait_done(spec)
            except (bg.Stop,OSError,ValueError,KeyError):pass
        for server in self.servers:server.close()
    def fixture(self,**kwargs):
        server=Server(**kwargs);self.servers.append(server)
        path=self.base/str(len(self.specs));path.mkdir()
        worker=path/'worker.py';worker.write_text(WORKER)
        supervisor=path/'supervise.py';supervisor.write_text(SUPERVISOR)
        metadata='{"fixture":"loopback-only"}'
        source=acq.Spec(server.url,len(BODY),hashlib.md5(BODY).hexdigest(),hashlib.sha256(metadata.encode()).hexdigest()).identity()
        limits=acq.Limits(chunk_bytes=512,reserve_free_bytes=0,max_storage_bytes=10000000).__dict__
        binding={'helper_sha256':bg.sha(HELPER),'config_sha256':'f'*64}
        config={'source':source,'metadata':metadata,'data':str(path/'data'),'evidence':str(path/'evidence'),'limits':limits,'binding':binding}
        cfg=path/'fixture.json';cfg.write_text(json.dumps(config))
        spec={'version':1,'mode':'synthetic','root':str(path/'background'),'data':config['data'],'evidence':config['evidence'],
              'cwd':str(path),'python':sys.executable,'command':[sys.executable,'-B','-u',str(worker),str(cfg)],
              'identity':source,'binding':binding,'limits':limits,
              'source_sha256':{str(HELPER):bg.sha(HELPER),str(WRAPPER):bg.sha(WRAPPER),str(worker):bg.sha(worker),str(cfg):bg.sha(cfg),str(supervisor):bg.sha(supervisor)}}
        expected=path/'spec.json';expected.write_text(json.dumps(spec))
        self.specs.append(spec)
        def spawn(attempt,fd):return [sys.executable,'-B','-u',str(supervisor),str(expected),attempt.name,str(fd)]
        return spec,server,spawn
    def launch(self,spec,spawn,**kw):
        return bg.launch(spec,authorized=True,quota_confirmed=True,supervisor_command=spawn,**kw)
    def wait_for(self,fn,timeout=15):
        until=time.monotonic()+timeout
        while time.monotonic()<until:
            result=fn()
            if result:return result
            time.sleep(.03)
        self.fail('Timed out: '+str(fn))
    def wait_done(self,spec):
        return self.wait_for(lambda:(v if not v.get('running') and not v.get('guard_held') and 'last_exit' in v else None) if (v:=bg.status(spec)) else None)
    def interrupted(self):
        spec,server,spawn=self.fixture(slow=True)
        self.launch(spec,spawn)
        self.wait_for(lambda:Path(spec['data'],'payload.part').exists() and Path(spec['data'],'payload.part').stat().st_size>=2048)
        self.assertEqual(bg.stop(spec)['status'],'interrupt_requested')
        stopped=self.wait_done(spec)
        self.assertFalse(stopped['archive_verified_and_published'])
        return spec,server,spawn,stopped
    def test_no_authorization_no_directories_or_requests(self):
        spec,server,spawn=self.fixture()
        r=bg.launch(spec,supervisor_command=spawn)
        self.assertEqual(r['status'],'dry_run');self.assertFalse(Path(spec['root']).exists());self.assertEqual(server.seen,[])
    def test_quota_gate_before_launch(self):
        spec,server,spawn=self.fixture()
        with self.assertRaisesRegex(bg.Stop,'quota'):bg.launch(spec,authorized=True,supervisor_command=spawn)
        self.assertFalse(Path(spec['root']).exists());self.assertFalse(server.seen)
    def test_success_logs_receipt_hashes_and_fast_status(self):
        spec,server,spawn=self.fixture();r=self.launch(spec,spawn);v=self.wait_done(spec)
        self.assertEqual(v['status'],'verified_published');self.assertTrue(v['archive_verified_and_published'])
        self.assertFalse(v['full_archive_rehashed_now']);self.assertTrue(bg.verify_archive(spec)['full_archive_rehashed_now'])
        self.assertIn('fixture helper starts',Path(r['log']).read_text())
        self.assertIn('verified_download_complete',Path(r['log']).read_text())
        self.assertEqual(v['checkpoint']['partial_bytes'],len(BODY));self.assertEqual(v['last_exit']['worker_returncode'],0)
        self.assertTrue(v['supervisor']['started_at']);self.assertTrue(v['last_exit']['finished_at'])
    def test_duplicate_launch_blocked_no_second_request(self):
        spec,server,spawn=self.fixture(slow=True);self.launch(spec,spawn)
        self.wait_for(lambda:bool(server.seen))
        before=len(server.seen)
        with self.assertRaises(bg.Stop):self.launch(spec,spawn)
        with self.assertRaises(bg.Stop):self.launch(spec,spawn,resume=True)
        self.assertEqual(len(server.seen),before);bg.stop(spec);self.wait_done(spec)
    def test_terminal_hangup_survival(self):
        spec,server,spawn=self.fixture(slow=True)
        driver=self.base/'terminal.py';expected=Path(spec['cwd'])/'spec.json';sup=Path(spec['cwd'])/'supervise.py'
        text='import os,sys,json,time\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT/'tools'))+')\nimport march_background_v1 as bg\ns=json.loads(Path('+repr(str(expected))+').read_text())\ndef spawn(a,fd):return [sys.executable,"-B","-u",'+repr(str(sup))+','+repr(str(expected))+',a.name,str(fd)]\nr=bg.launch(s,authorized=True,quota_confirmed=True,supervisor_command=spawn)\nprint(json.dumps(r),flush=True)\ntime.sleep(30)\n'
        driver.write_text(text)
        master,slave=pty.openpty()
        def controlling_terminal():
            os.setsid()
            import termios
            fcntl.ioctl(0,termios.TIOCSCTTY,0)
        parent=subprocess.Popen([sys.executable,'-B','-u',str(driver)],stdin=slave,stdout=slave,stderr=slave,preexec_fn=controlling_terminal)
        os.close(slave)
        try:
            self.wait_for(lambda:bool(server.seen))
            before=bg.status(spec);self.assertTrue(before['running'])
            supervisor=before['supervisor']['identity'];self.assertNotEqual(os.getsid(supervisor['pid']),parent.pid)
            os.close(master);master=None  # real controlling-terminal hangup
            parent.wait(timeout=5);self.assertEqual(parent.returncode,-signal.SIGHUP)
            self.assertTrue(bg.status(spec)['running'])
            result=self.wait_done(spec);self.assertTrue(result['archive_verified_and_published'])
            self.assertTrue(Path(result['log']).exists())
        finally:
            if master is not None:os.close(master)
            if parent.poll() is None:parent.kill();parent.wait()
    def test_interruption_preserves_and_resume_delegates_range(self):
        spec,server,spawn,old=self.interrupted()
        partial=Path(spec['data'])/'payload.part';prefix=partial.read_bytes()
        old_states={p.name:p.read_bytes() for p in Path(spec['evidence']).glob('state-*.json')}
        old_log=Path(old['log']).read_bytes();inode=(Path(spec['data'])/'.acquisition.lock').stat().st_ino
        server.slow=False;self.launch(spec,spawn,resume=True);v=self.wait_done(spec)
        self.assertTrue(v['archive_verified_and_published']);self.assertTrue(partial.read_bytes().startswith(prefix))
        self.assertEqual((Path(spec['data'])/'.acquisition.lock').stat().st_ino,inode)
        self.assertEqual(Path(old['log']).read_bytes(),old_log)
        for name,data in old_states.items():self.assertEqual((Path(spec['evidence'])/name).read_bytes(),data)
        self.assertEqual(server.seen[-1]['range'],f'bytes={len(prefix)}-');self.assertEqual(server.seen[-1]['if_range'],'"fixture-v1"')
        self.assertEqual(len(bg.attempts(spec['root'])),2)
    def test_zero_exit_without_completion_is_failure(self):
        spec,server,spawn=self.fixture();cfg=Path(spec['cwd'])/'fixture.json';d=json.loads(cfg.read_text());d['mode']='exit_zero';cfg.write_text(json.dumps(d))
        spec['source_sha256'][str(cfg)]=bg.sha(cfg);(Path(spec['cwd'])/'spec.json').write_text(json.dumps(spec))
        self.launch(spec,spawn);v=self.wait_done(spec)
        self.assertFalse(v['archive_verified_and_published']);self.assertEqual(v['status'],'failed');self.assertEqual(v['last_exit']['worker_returncode'],0)
    def test_failed_exit_not_success(self):
        spec,server,spawn=self.fixture();cfg=Path(spec['cwd'])/'fixture.json';d=json.loads(cfg.read_text());d['mode']='exit_bad';cfg.write_text(json.dumps(d))
        spec['source_sha256'][str(cfg)]=bg.sha(cfg);(Path(spec['cwd'])/'spec.json').write_text(json.dumps(spec))
        self.launch(spec,spawn);v=self.wait_done(spec)
        self.assertEqual(v['last_exit']['worker_returncode'],7);self.assertFalse(v['archive_verified_and_published'])
    def test_changed_etag_rejected_prefix_preserved(self):
        spec,server,spawn,old=self.interrupted();prefix=Path(spec['data'],'payload.part').read_bytes();server.mode='changed_etag';server.slow=False
        self.launch(spec,spawn,resume=True);v=self.wait_done(spec)
        self.assertEqual(v['status'],'failed');self.assertEqual(Path(spec['data'],'payload.part').read_bytes(),prefix)
        self.assertIn('ETag changed',Path(v['log']).read_text())
    def test_ignored_range_rejected_prefix_preserved(self):
        spec,server,spawn,old=self.interrupted();prefix=Path(spec['data'],'payload.part').read_bytes();server.mode='ignore_range';server.slow=False
        self.launch(spec,spawn,resume=True);v=self.wait_done(spec)
        self.assertEqual(v['status'],'failed');self.assertEqual(Path(spec['data'],'payload.part').read_bytes(),prefix)
    def test_tampered_partial_helper_rejects_before_request(self):
        spec,server,spawn,old=self.interrupted();p=Path(spec['data'])/'payload.part';data=p.read_bytes();p.write_bytes(bytes([data[0]^1])+data[1:]);before=len(server.seen)
        self.launch(spec,spawn,resume=True);v=self.wait_done(spec)
        self.assertEqual(v['status'],'failed');self.assertEqual(len(server.seen),before);self.assertIn('tampered partial',Path(v['log']).read_text())
    def test_invalid_checkpoint_blocks_before_launch(self):
        spec,server,spawn,old=self.interrupted();p=next(Path(spec['evidence']).glob('state-*.json'));v=json.loads(p.read_text());v['owner']='0'*32;p.write_text(json.dumps(v));before=len(server.seen)
        with self.assertRaisesRegex(bg.Stop,'owner'):self.launch(spec,spawn,resume=True)
        self.assertEqual(len(server.seen),before);self.assertEqual(len(bg.attempts(spec['root'])),1)
    def test_acquisition_lock_not_bypassed(self):
        spec,server,spawn,old=self.interrupted();before=len(server.seen)
        with acq.acquire_lock(Path(spec['data'])/'.acquisition.lock'):
            with self.assertRaisesRegex(bg.Stop,'lock is held'):self.launch(spec,spawn,resume=True)
        self.assertEqual(len(server.seen),before)
    def test_bad_checksum_never_success(self):
        spec,server,spawn=self.fixture(mode='corrupt');self.launch(spec,spawn);v=self.wait_done(spec)
        self.assertEqual(v['status'],'failed');self.assertFalse(v['archive_verified_and_published']);self.assertTrue(Path(spec['data'],'payload.part').exists());self.assertFalse(Path(spec['data'],'22-03.tar').exists())
    def test_sources_changed_block_before_request(self):
        spec,server,spawn=self.fixture();path=Path(spec['cwd'])/'worker.py';path.write_text(path.read_text()+'\n#changed\n')
        with self.assertRaisesRegex(bg.Stop,'changed'):self.launch(spec,spawn)
        self.assertFalse(Path(spec['root']).exists());self.assertFalse(server.seen)
        self.specs.remove(spec)  # intentionally invalid binding; no child was started
    def test_existing_final_not_overwritten(self):
        spec,server,spawn=self.fixture();data=Path(spec['data']);data.mkdir();p=data/'22-03.tar';p.write_bytes(b'old')
        with self.assertRaises(bg.Stop):self.launch(spec,spawn)
        self.assertEqual(p.read_bytes(),b'old');self.assertFalse(server.seen)
    def test_symlink_destination_rejected(self):
        spec,server,spawn=self.fixture();other=self.base/'old';other.mkdir();Path(spec['root']).symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(bg.Stop,'Symlink'):self.launch(spec,spawn)
        self.assertEqual(list(other.iterdir()),[])
    def test_stale_pid_and_boot_not_running(self):
        i=bg.process_identity(os.getpid());self.assertTrue(bg.alive(i))
        for field in ('boot_id','start_ticks'):
            changed=dict(i);changed[field]='other' if field=='boot_id' else changed[field]+1;self.assertFalse(bg.alive(changed))
    def test_guard_is_held_by_worker_if_supervisor_killed(self):
        spec,server,spawn=self.fixture(slow=True);self.launch(spec,spawn)
        self.wait_for(lambda:bool(server.seen));v=bg.status(spec);os.kill(v['supervisor']['identity']['pid'],signal.SIGKILL)
        self.wait_for(lambda:not bg.alive(v['supervisor']['identity']))
        self.assertTrue(bg.guard_held(spec['root']));self.assertTrue(bg.status(spec)['worker']['running'])
        with self.assertRaises(bg.Stop):self.launch(spec,spawn,resume=True)
        bg.stop(spec);self.wait_for(lambda:not bg.guard_held(spec['root']))
        self.assertNotIn('last_exit',bg.status(spec));self.assertFalse(bg.status(spec)['archive_verified_and_published'])
    def test_completed_archive_never_relaunched(self):
        spec,server,spawn=self.fixture();self.launch(spec,spawn);self.wait_done(spec);before=len(server.seen)
        with self.assertRaises(bg.Stop):self.launch(spec,spawn,resume=True)
        self.assertEqual(len(server.seen),before)
    def test_changed_published_bytes_status_and_verify_fail(self):
        spec,server,spawn=self.fixture();self.launch(spec,spawn);self.wait_done(spec)
        p=Path(spec['data'])/'22-03.tar';p.write_bytes(b'bad')
        self.assertFalse(bg.status(spec)['archive_verified_and_published'])
        with self.assertRaises(bg.Stop):bg.verify_archive(spec)
    def test_status_no_network_or_writes(self):
        spec,server,spawn=self.fixture();self.launch(spec,spawn);self.wait_done(spec)
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.base.rglob('*') if p.is_file()}
        n=len(server.seen)
        with mock.patch.object(bg,'write_new',side_effect=AssertionError('write')):
            bg.status(spec)
        self.assertEqual(len(server.seen),n)
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_production_defaults_and_fixed_command_no_launch(self):
        spec=bg.production_spec()
        self.assertEqual(spec['command'],[str(ROOT/'.venv/bin/python'),'-B','-u',str(HELPER),'--config',str(bg.CONFIG),'--authorize-transfer','--quota-confirmed'])
        with mock.patch.object(bg.subprocess,'Popen',side_effect=AssertionError('production launch forbidden')):
            self.assertEqual(bg.launch(spec)['status'],'dry_run')
            with self.assertRaises(bg.Stop):bg.launch(spec,authorized=True)
        PRODUCTION_GUARDS.append('production dry-run and quota gate: no child launch')
    def test_supervisor_sigterm_preserves_partial(self):
        spec,server,spawn=self.fixture(slow=True);self.launch(spec,spawn)
        self.wait_for(lambda:Path(spec['data'],'payload.part').exists() and Path(spec['data'],'payload.part').stat().st_size>=2048)
        v=bg.status(spec);os.kill(v['supervisor']['identity']['pid'],signal.SIGTERM)
        v=self.wait_done(spec)
        self.assertFalse(v['archive_verified_and_published']);self.assertTrue(Path(spec['data'],'payload.part').exists())
        self.assertTrue(any(x['signal']==signal.SIGTERM for x in v['last_exit']['received_signals']))
    def test_no_tmux_or_nohup_required(self):
        spec,server,spawn=self.fixture()
        with mock.patch.dict(os.environ,{'PATH':'/nonexistent'}):self.launch(spec,spawn)
        self.assertTrue(self.wait_done(spec)['archive_verified_and_published'])
    def test_changed_receipt_detected(self):
        spec,server,spawn=self.fixture();self.launch(spec,spawn);self.wait_done(spec)
        p=Path(spec['evidence'])/'completed.json';value=json.loads(p.read_text());value['owner']='0'*32;p.write_text(json.dumps(value))
        self.assertFalse(bg.status(spec)['archive_verified_and_published'])
        with self.assertRaisesRegex(bg.Stop,'provenance'):bg.verify_archive(spec)
    def test_changed_session_binding_blocks_resume(self):
        spec,server,spawn,old=self.interrupted();p=Path(spec['evidence'])/'session.json';value=json.loads(p.read_text());value['binding']['config_sha256']='0'*64;p.write_text(json.dumps(value));n=len(server.seen)
        with self.assertRaisesRegex(bg.Stop,'ownership/binding'):self.launch(spec,spawn,resume=True)
        self.assertEqual(len(server.seen),n)
    def test_insufficient_storage_blocks_before_child(self):
        spec,server,spawn=self.fixture();spec['mode']='production'
        with mock.patch.object(bg.os,'statvfs',return_value=type('FS',(),{'f_flag':0,'f_bavail':1,'f_frsize':4096})()), mock.patch.object(bg.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            with self.assertRaisesRegex(bg.Stop,'90|retained budget'):self.launch(spec,spawn)
        self.assertFalse(server.seen);self.assertEqual(len(bg.attempts(spec['root'])),0)
    def test_pidfd_unavailable_fails_without_unpinned_signal(self):
        identity=bg.process_identity(os.getpid())
        def available(obj,name):
            return False if obj is os or obj is signal else hasattr(obj,name)
        with mock.patch.object(bg.ctypes,'CDLL',return_value=object()), mock.patch.object(bg,'hasattr',side_effect=available,create=True), mock.patch.object(bg.os,'kill',side_effect=AssertionError('unpinned signal forbidden')):
            with self.assertRaisesRegex(bg.Stop,'unavailable'):bg.interrupt_identity(identity)
    def test_protected_helper_and_config_pins(self):
        self.assertEqual(bg.sha(HELPER),bg.HELPER_SHA);self.assertEqual(bg.sha(bg.CONFIG),bg.CONFIG_SHA)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BackgroundTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'loopback_requests':len(REQUESTS),'real_archive_requests':0,'production_cli_transfer_branch_invoked':False,
             'workdir':str(WORK),'production_guards':PRODUCTION_GUARDS}
    print(json.dumps(summary,sort_keys=True),flush=True)
    (WORK/'test_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
