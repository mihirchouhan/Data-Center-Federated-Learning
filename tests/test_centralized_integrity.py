"""Synthetic preservation tests; optional saved real inventory is read-only."""
import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.centralized_io import capture_inventory,check_inventory,execution_sources


class PreservationTests(unittest.TestCase):
    def test_inventory_detects_changes_and_missing_files_but_allows_new_additions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'old').write_text('protected')
            baseline=capture_inventory(root);(root/'new').write_text('additive')
            self.assertEqual(check_inventory(root,baseline)['failed'],0)
            (root/'old').write_text('changed')
            self.assertEqual(check_inventory(root,baseline)['mismatches'],['old'])
            (root/'old').unlink()
            self.assertEqual(check_inventory(root,baseline)['failed'],1)

    def test_new_sources_do_not_match_v1_execution_glob_and_are_snapshotted(self):
        files=['src/centralized_sgd.py','src/centralized_io.py']
        for name in files:
            self.assertFalse(Path(name).match('src/experiment_*.py'))
            self.assertIn(name,execution_sources())

    def test_runner_has_no_test_split_or_federated_evaluator_import(self):
        tree=ast.parse((ROOT/'src/centralized_sgd.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):self.assertNotEqual(node.module,'federated_evaluate')
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='split':
                for arg in node.args:
                    if isinstance(arg,ast.Constant):self.assertNotEqual(arg.value,'test')

    def test_saved_preservation_inventory_if_supplied(self):
        path=os.environ.get('M100_CENTRALIZED_BASELINE')
        if path is None:
            # This remains a real executed synthetic check, never a skipped test.
            with tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'protected';p.write_text('fixture')
                self.assertEqual(check_inventory(temp,capture_inventory(temp))['failed'],0)
        else:
            value=json.loads(Path(path).read_text())
            baseline={k:v['sha256'] if isinstance(v,dict) else v for k,v in value['files'].items()}
            result=check_inventory(value['project_root'],baseline)
            self.assertEqual(result['mismatches'],[])


if __name__=='__main__':unittest.main()
