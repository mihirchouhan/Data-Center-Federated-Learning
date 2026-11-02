"""Optional read-only before/after check of the baseline captured before Stage 1."""
import hashlib
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('M100_FEDAVG_BASELINE'),'Set M100_FEDAVG_BASELINE to the recorded external baseline')
class V1IntegrityTests(unittest.TestCase):
    def test_every_pre_existing_protected_file_retains_its_recorded_hash(self):
        baseline=json.loads(Path(os.environ['M100_FEDAVG_BASELINE']).read_text())
        root=Path(baseline['project_root'])
        self.assertGreater(len(baseline['protected_files']),0)
        for name in ('experiments.py','src/model.py','output/experiments/v1/frozen/contract.json',
                     'output/experiments/v1/comparison/comparison.json'):
            self.assertIn(name,baseline['protected_files'])
        for name,expected in baseline['protected_files'].items():
            with self.subTest(path=name):
                digest=hashlib.sha256()
                with (root/name).open('rb') as handle:
                    for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
                self.assertEqual(digest.hexdigest(),expected)
        recorded=json.loads((root/'output/experiments/v1/comparison/matrix_manifest.json').read_text())
        current=[root/'experiments.py',root/'src/model.py',root/'src/io_utils.py',root/'src/forecast_plot.py',
                 *sorted((root/'src').glob('experiment_*.py'))]
        self.assertEqual({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in current},recorded['source_hashes'])


if __name__=='__main__':unittest.main()
