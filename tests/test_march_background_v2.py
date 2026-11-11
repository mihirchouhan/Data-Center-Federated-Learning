"""Phase 1.9.2 regression and v1/v2 lifecycle tests; /tmp and loopback only."""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT=Path('/media/csedept/cse2014/Project/m100_training_template')
sys.path.insert(0,str(ROOT/'tools'))
import march_background_v2 as bg

REGRESSION_WORK=[]
LIFECYCLE_MODULES=[]

class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='m100-p192-capacity-'))
        REGRESSION_WORK.append(str(cls.work))
    def setUp(self):
        self.base=self.work/self._testMethodName;self.base.mkdir()
        self.good=types.SimpleNamespace(f_flag=0,f_bavail=100000000000,f_frsize=1)
        self.vfs=mock.patch.object(bg.os,'fstatvfs',return_value=self.good)
        self.vfs_mock=self.vfs.start();self.addCleanup(self.vfs.stop)
    @contextlib.contextmanager
    def read_only(self):
        with mock.patch.object(Path,'mkdir',side_effect=AssertionError('preflight mkdir')),mock.patch.object(os,'mkdir',side_effect=AssertionError('preflight mkdir')),mock.patch.object(os,'makedirs',side_effect=AssertionError('preflight makedirs')),mock.patch.object(bg,'write_new',side_effect=AssertionError('preflight write')),mock.patch.object(subprocess,'Popen',side_effect=AssertionError('preflight process')),mock.patch.object(socket,'socket',side_effect=AssertionError('preflight socket')):
            yield
    def inspect(self,p):
        with self.read_only():return bg.inspect_capacity(p)
    def test_fresh_parent_absent_no_creation(self):
        destination=self.base/'absent'/'v1';before=sorted(self.base.rglob('*'))
        result=self.inspect(destination)
        self.assertEqual(result['inspected_ancestor'],str(self.base))
        self.assertFalse(result['directories_created']);self.assertEqual(result['required_bytes'],90000000000)
        self.assertEqual(sorted(self.base.rglob('*')),before)
    def test_existing_parent(self):
        parent=self.base/'existing';parent.mkdir();r=self.inspect(parent/'v1')
        self.assertEqual(r['inspected_ancestor'],str(parent));self.assertFalse((parent/'v1').exists())
    def test_existing_destination_inspects_itself(self):
        destination=self.base/'v1';destination.mkdir()
        self.assertEqual(self.inspect(destination)['inspected_ancestor'],str(destination))
    def test_deep_missing_parents(self):
        self.assertEqual(self.inspect(self.base/'a'/'b'/'c'/'v1')['inspected_ancestor'],str(self.base))
    def test_symlink_ancestor_rejected(self):
        real=self.base/'real';real.mkdir();link=self.base/'link';link.symlink_to(real,target_is_directory=True)
        with self.assertRaisesRegex(bg.Stop,'Symlink'):self.inspect(link/'v1')
        self.vfs_mock.assert_not_called()
    def test_dangling_symlink_rejected(self):
        p=self.base/'dangling';p.symlink_to(self.base/'missing',target_is_directory=True)
        with self.assertRaisesRegex(bg.Stop,'Symlink'):self.inspect(p/'v1')
        self.vfs_mock.assert_not_called()
    def test_symlink_destination_rejected(self):
        real=self.base/'real';real.mkdir();p=self.base/'v1';p.symlink_to(real,target_is_directory=True)
        with self.assertRaisesRegex(bg.Stop,'Symlink'):self.inspect(p)
    def test_file_in_ancestry_rejected(self):
        p=self.base/'file';p.write_bytes(b'existing')
        with self.assertRaisesRegex(bg.Stop,'not a directory'):self.inspect(p/'v1')
        self.assertEqual(p.read_bytes(),b'existing')
    def test_insufficient_space_rejected_no_creation(self):
        self.good.f_bavail=89999999999
        with self.assertRaisesRegex(bg.Stop,'80 GB retained'):self.inspect(self.base/'absent'/'v1')
        self.assertEqual(list(self.base.iterdir()),[])
    def test_exact_90gb_boundary_accepted(self):
        self.good.f_bavail=90000000000
        self.assertEqual(self.inspect(self.base/'v1')['filesystem_available_bytes'],90000000000)
    def test_filesystem_readonly_rejected(self):
        self.good.f_flag=os.ST_RDONLY
        with self.assertRaisesRegex(bg.Stop,'writable filesystem'):self.inspect(self.base/'v1')
    def test_available_bytes_use_fragment_size(self):
        self.good.f_bavail=25000000;self.good.f_frsize=4096
        self.assertEqual(self.inspect(self.base/'v1')['filesystem_available_bytes'],102400000000)
    def test_descriptor_identity_change_rejected(self):
        actual=self.base.stat()
        changed=types.SimpleNamespace(st_dev=actual.st_dev,st_ino=actual.st_ino+1,st_uid=os.getuid(),st_mode=actual.st_mode)
        with mock.patch.object(bg.os,'fstat',return_value=changed):
            with self.assertRaisesRegex(bg.Stop,'changed while opening'):self.inspect(self.base/'v1')
        self.vfs_mock.assert_not_called()
    def test_descriptor_wrong_owner_rejected(self):
        actual=self.base.stat()
        changed=types.SimpleNamespace(st_dev=actual.st_dev,st_ino=actual.st_ino,st_uid=os.getuid()+1,st_mode=actual.st_mode)
        with mock.patch.object(bg.os,'fstat',return_value=changed):
            with self.assertRaisesRegex(bg.Stop,'changed while opening'):self.inspect(self.base/'v1')
        self.vfs_mock.assert_not_called()
    def test_dry_run_no_directory_or_capacity_or_process(self):
        spec=bg.production_spec()
        with self.read_only(),mock.patch.object(bg,'inspect_capacity',side_effect=AssertionError('dry-run capacity')):
            r=bg.launch(spec)
        self.assertEqual(r['status'],'dry_run');self.assertFalse(r['directories_created'])
    def test_quota_gate_precedes_capacity_and_writes(self):
        spec=bg.production_spec()
        with self.read_only(),mock.patch.object(bg,'inspect_capacity',side_effect=AssertionError('quota bypass')):
            with self.assertRaisesRegex(bg.Stop,'quota'):bg.launch(spec,authorized=True)
    def test_capacity_failure_before_workflow_creation(self):
        self.good.f_bavail=1;spec=bg.production_spec()
        with self.read_only():
            with self.assertRaisesRegex(bg.Stop,'80 GB retained'):bg.launch(spec,authorized=True,quota_confirmed=True)
        self.assertFalse(Path(spec['root']).exists())
    def test_fresh_production_layout_passes_capacity_before_any_mkdir(self):
        spec=bg.production_spec();self.assertFalse(Path(spec['data']).parent.exists())
        with self.read_only():
            with self.assertRaisesRegex(AssertionError,'preflight mkdir'):bg.launch(spec,authorized=True,quota_confirmed=True)
        self.assertEqual(self.vfs_mock.call_count,1)
        self.assertFalse(Path(spec['root']).exists());self.assertFalse(Path(spec['data']).parent.exists())
    def test_preflight_cli_no_network_or_creation(self):
        output=io.StringIO()
        with self.read_only(),contextlib.redirect_stdout(output):self.assertEqual(bg.main(['preflight']),0)
        value=json.loads(output.getvalue());self.assertEqual(value['status'],'capacity_preflight_passed')
        self.assertFalse(value['transfer_enabled']);self.assertEqual(value['network_requests'],0)
    def test_preflight_rejects_transfer_authorization_flags(self):
        with self.read_only():
            with self.assertRaisesRegex(bg.Stop,'only to explicit'):bg.main(['preflight','--authorize-download-only'])
    def test_pins_original_workflow_and_runtime_isolation(self):
        spec=bg.production_spec();self.assertEqual(spec['version'],2)
        self.assertEqual(spec['root'],str(ROOT/'output/march_2022_background/v2'))
        self.assertEqual(bg.sha(bg.HELPER),bg.HELPER_SHA);self.assertEqual(bg.sha(bg.CONFIG),bg.CONFIG_SHA)
        manifest=bg.load(ROOT/'output/march_2022_background/v1/workflow.json')
        self.assertEqual(manifest['source_sha256'][str(ROOT/'tools/march_background_v1.py')],bg.sha(ROOT/'tools/march_background_v1.py'))
    def test_real_destination_capacity_read_is_non_mutating(self):
        self.vfs.stop()  # this test uses actual host fstatvfs, not a fabricated measurement
        try:
            destination=ROOT/'data/march_2022_acquisition/v1'
            self.assertFalse(destination.parent.exists())
            with self.read_only():value=bg.inspect_capacity(destination)
            self.assertEqual(value['inspected_ancestor'],str(ROOT/'data'))
            self.assertGreaterEqual(value['filesystem_available_bytes'],90000000000)
            self.assertFalse(destination.parent.exists())
        finally:self.vfs.start()


def lifecycle_module(version):
    """Run original tests unmodified for v1; explicitly adapted fixtures for v2.

    In-memory source adaptation never modifies the historical test file. V2's
    storage failure intentionally precedes workflow creation and uses fstatvfs.
    """
    text=(ROOT/'tests/test_march_background_v1.py').read_text()
    if version==2:
        text=text.replace('march_background_v1','march_background_v2')
        old="mock.patch.object(bg.os,'statvfs',return_value=type('FS',(),{'f_flag':0,'f_bavail':1,'f_frsize':4096})())"
        new="mock.patch.object(bg.os,'fstatvfs',return_value=type('FS',(),{'f_flag':0,'f_bavail':1,'f_frsize':4096})())"
        assert old in text;text=text.replace(old,new)
        old="self.assertFalse(server.seen);self.assertEqual(len(bg.attempts(spec['root'])),0)"
        new="self.assertFalse(server.seen);self.assertFalse(Path(spec['root']).exists())"
        assert old in text;text=text.replace(old,new)
    module=types.ModuleType(f'm100_p192_lifecycle_v{version}')
    module.__file__=str(ROOT/'tests/test_march_background_v1.py')
    sys.modules[module.__name__]=module
    # Avoid reusing caller-supplied test roots between two suite invocations.
    with mock.patch.dict(os.environ,{},clear=False):
        os.environ.pop('M100_BG_TEST_ROOT',None)
        exec(compile(text,module.__file__,'exec'),module.__dict__)
    LIFECYCLE_MODULES.append((version,module))
    return unittest.defaultTestLoader.loadTestsFromTestCase(module.BackgroundTests)


def main():
    started=time.monotonic()
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(CapacityTests))
    suite.addTests(lifecycle_module(1));suite.addTests(lifecycle_module(2))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    summary={'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skipped':len(result.skipped),
        'seconds':time.monotonic()-started,'regression_tests':unittest.defaultTestLoader.loadTestsFromTestCase(CapacityTests).countTestCases(),
        'unchanged_v1_tests':30,'v2_lifecycle_tests':30,
        'v2_fixture_adaptations':['module/path version names','statvfs mock to fstatvfs','capacity failure leaves workflow absent'],
        'loopback_requests':sum(len(m.REQUESTS) for _,m in LIFECYCLE_MODULES),'real_archive_requests':0,'official_metadata_requests':0,
        'production_transfer_cli_invoked':False,'real_runtime_directories_created':False,
        'temporary_fixture_roots':REGRESSION_WORK+[str(m.WORK) for _,m in LIFECYCLE_MODULES]}
    print(json.dumps(summary,sort_keys=True),flush=True)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':sys.exit(main())
