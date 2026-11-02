"""Optional real-project preservation check against the pre-Stage-2 inventory."""
import hashlib
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('M100_FEDAVG_STAGE2_BASELINE'), 'Set M100_FEDAVG_STAGE2_BASELINE')
class Stage2IntegrityTests(unittest.TestCase):
    def test_all_pre_stage2_files_including_stage1_remain_unchanged(self):
        baseline=json.loads(Path(os.environ['M100_FEDAVG_STAGE2_BASELINE']).read_text())
        root=Path('/media/csedept/cse2014/Project/m100_training_template')
        self.assertGreaterEqual(len(baseline),797)
        for name in ('src/federated_data.py','src/federated_train.py','tests/test_fedavg.py',
                     'tests/test_federated_data.py','tests/test_federated_integrity.py'):
            self.assertIn(name,baseline)
        for name,expected in baseline.items():
            with self.subTest(path=name):
                digest=hashlib.sha256()
                with (root/name).open('rb') as handle:
                    for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
                self.assertEqual(digest.hexdigest(),expected)


if __name__=='__main__':unittest.main()
