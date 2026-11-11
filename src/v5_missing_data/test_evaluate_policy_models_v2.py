"""Synthetic discovery/provenance tests; never fit, score, or create a test lock.

Run verification is mocked only in small discovery fixtures. Fresh official
verification of real runs is a separate execution gate, not inferred here.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.experiment_data import sha256
from src.v5_missing_data import evaluate_policy_models_v2 as v2


class EvaluatorV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v5-evaluator-v2-synthetic-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root/'full_runs'
        self.runs.mkdir()
        self.context = SimpleNamespace(receipt_sha256='synthetic-cohort-receipt',
            contract={'cohorts': {'common/all_policies/test': {'identity': {
                'count': 1, 'ordered_sha256': 'synthetic-test-identity-not-a-real-cohort'}}}})
        (self.root/'protected.py').write_text('# unchanged synthetic source\n')
        (self.root/'evidence.json').write_text('{}\n')
        self.plan = {'cohort_receipt_sha256': self.context.receipt_sha256,
                     'sources': {'protected.py': sha256(self.root/'protected.py')},
                     'evidence': {'evidence.json': sha256(self.root/'evidence.json')}}
        adapter = self.root/v2.ADAPTER_SOURCE
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(Path(v2.__file__).read_bytes())
        self.guards = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target in ('src.v5_missing_data.evaluate_policy_models.SplitData',
                       'src.v5_missing_data.evaluate_policy_models.predict_and_score',
                       'src.v5_missing_data.evaluate_policy_models.evaluate_locked',
                       'src.v5_missing_data.evaluate_policy_models.metric_pair',
                       'src.v5_missing_data.run_policy_experiments.runtime',
                       'src.v5_missing_data.run_policy_experiments.run_one',
                       'src.v5_missing_data.run_policy_experiments.train_local',
                       'src.model.SequenceRegressor.forward', 'torch.optim.SGD.step'):
            guard = self.stack.enter_context(mock.patch(target, side_effect=AssertionError('Forbidden inference/fit/scoring')))
            self.guards.append(guard)

    def tearDown(self):
        for guard in self.guards:
            self.assertEqual(guard.call_count, 0, 'A forbidden operation was attempted')

    def write_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True)+'\n')

    def populate(self, layout='legacy'):
        found = {}
        for method, policy, seed in v2.EXPECTED:
            if layout == 'nested':
                path = self.runs/method/f'{policy}_seed{seed}'
            elif layout == 'arbitrary':
                path = self.runs/f'run-{len(found):02d}'
            else:
                label = 'none' if policy == 'no_fill' else policy
                prefix = 'centralized' if method == 'centralized_sgd' and (policy, seed) in (
                    ('ff10', 42), ('ff10', 43), ('ff20', 42)) else method
                path = self.runs/f'{prefix}_{label}_seed{seed}_v1'
            (path/'selected').mkdir(parents=True)
            self.write_json(path/'manifest.json', {'version': 1, 'method': method,
                'policy': policy, 'seed': seed, 'kind': 'main', 'cap': 200})
            self.write_json(path/'configuration.json', self.plan)
            (path/'selected/model.pt').write_bytes(b'synthetic hash fixture; not a model checkpoint')
            validation = {'rmse_w': 2.0, 'split': 'val', 'synthetic_metadata': True}
            self.write_json(path/'selected/receipt.json', {'method': method, 'policy': policy,
                'seed': seed, 'selected_round': 10, 'validation': validation,
                'manifest_sha256': sha256(path/'manifest.json'),
                'cohort_receipt_sha256': self.context.receipt_sha256,
                'model_sha256': sha256(path/'selected/model.pt')})
            self.write_json(path/'completed.json', {'status': 'complete', 'completed_round': 200,
                'selected_round': 10, 'selected_validation': validation})
            found[(method, policy, seed)] = path
        return found

    def checked(self, plan, context, run):
        self.assertEqual(plan, self.plan)
        self.assertIs(context, self.context)
        self.assertTrue((run/'manifest.json').is_file())
        return {'status': 'verified_complete', 'completed_round': 200,
                'selected_round': 10, 'model_forward_or_scoring': False}

    def preflight(self):
        with mock.patch.object(v2.runner, 'verify_run', side_effect=self.checked) as verifier, redirect_stderr(io.StringIO()):
            records = v2.preflight(self.plan, self.context, [self.runs], root=self.root)
        self.assertEqual(verifier.call_count, 24)
        return records

    def test_legacy_prefix_and_none_names_use_exact_manifest_identities(self):
        expected = self.populate()
        self.assertEqual(v2.discover_runs([self.runs]), expected)
        self.assertIn('centralized_ff10_seed42_v1', str(expected[('centralized_sgd', 'ff10', 42)]))
        self.assertEqual(len(self.preflight()), 24)

    def test_nested_and_arbitrary_directory_names_are_supported(self):
        self.populate('nested')
        self.assertEqual(tuple(v2.discover_runs([self.runs])), v2.EXPECTED)
        # Renaming is confined to synthetic temporary fixtures.
        for i, path in enumerate(sorted(self.runs.glob('*/*/manifest.json'))):
            path.parent.rename(self.runs/f'arbitrary-{i}')
        self.assertEqual(len(v2.discover_runs([self.runs])), 24)

    def test_separate_primary_sensitivity_roots_and_identical_roots(self):
        self.populate('arbitrary')
        primary, sensitivity = self.root/'primary', self.root/'sensitivity'
        primary.mkdir(); sensitivity.mkdir()
        for key, path in v2.discover_runs([self.runs]).items():
            path.rename((primary if key[1] in ('no_fill', 'ff20') else sensitivity)/path.name)
        self.assertEqual(len(v2.discover_runs([primary, sensitivity])), 24)
        self.assertEqual(len(v2.discover_runs([primary, primary, sensitivity])), 24)

    def test_duplicate_identity_is_rejected(self):
        runs = self.populate()
        duplicate = self.runs/'duplicate'
        duplicate.mkdir()
        (duplicate/'manifest.json').write_bytes((next(iter(runs.values()))/'manifest.json').read_bytes())
        with self.assertRaisesRegex(ValueError, 'Duplicate'): v2.discover_runs([self.runs])

    def test_missing_identity_is_rejected(self):
        runs = self.populate()
        path = next(iter(runs.values()))
        path.rename(self.root/'outside-declared-roots')
        with self.assertRaisesRegex(ValueError, 'Missing expected'): v2.discover_runs([self.runs])

    def test_unexpected_identity_is_rejected(self):
        path = next(iter(self.populate().values()))
        m = json.loads((path/'manifest.json').read_text());m['policy'] = 'none'
        self.write_json(path/'manifest.json', m)
        with self.assertRaisesRegex(ValueError, 'Unexpected'): v2.discover_runs([self.runs])

    def test_smoke_invalid_cap_and_malformed_manifests_are_rejected(self):
        path = next(iter(self.populate().values()))
        original = json.loads((path/'manifest.json').read_text())
        for change in ({'kind': 'smoke'}, {'cap': 3}, {'version': 9}, {'seed': True}):
            self.write_json(path/'manifest.json', {**original, **change})
            with self.assertRaises(ValueError): v2.discover_runs([self.runs])
        self.write_json(path/'manifest.json', [])
        with self.assertRaisesRegex(ValueError, 'Invalid run manifest'): v2.discover_runs([self.runs])

    def test_orphaned_run_evidence_is_rejected(self):
        path = next(iter(self.populate().values()))
        (path/'manifest.json').rename(path/'misplaced.json')
        with self.assertRaisesRegex(ValueError, 'without manifest'): v2.discover_runs([self.runs])

    def test_symlinks_and_overlapping_roots_are_rejected(self):
        self.populate()
        alias = self.root/'alias';alias.symlink_to(self.runs, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'non-symlink'): v2.discover_runs([alias])
        with self.assertRaisesRegex(ValueError, 'Overlapping'): v2.discover_runs([self.root, self.runs])
        (self.runs/'alias').symlink_to(self.root/'protected.py')
        with self.assertRaisesRegex(ValueError, 'Symlink'): v2.discover_runs([self.runs])

    def test_original_verification_failure_propagates_without_writes(self):
        self.populate()
        with mock.patch.object(v2.runner, 'verify_run', side_effect=ValueError('Round integrity mismatch')) as verifier:
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                v2.preflight(self.plan, self.context, [self.runs], root=self.root)
        self.assertEqual(verifier.call_count, 1)
        self.assertFalse(list(self.root.rglob('selection_lock.json')))

    def test_partial_verification_is_never_accepted(self):
        self.populate()
        result = {'status': 'verified_partial', 'completed_round': 1, 'selected_round': 0,
                  'model_forward_or_scoring': False}
        with mock.patch.object(v2.runner, 'verify_run', return_value=result):
            with self.assertRaisesRegex(ValueError, 'verified complete'):
                v2.preflight(self.plan, self.context, [self.runs], root=self.root)

    def test_changed_original_source_or_evidence_stops_before_verification(self):
        self.populate()
        for name in ('protected.py', 'evidence.json'):
            path = self.root/name;before = path.read_bytes();path.write_bytes(before+b' ')
            with mock.patch.object(v2.runner, 'verify_run') as verifier:
                with self.assertRaisesRegex(ValueError, 'changed'):
                    v2.preflight(self.plan, self.context, [self.runs], root=self.root)
                verifier.assert_not_called()
            path.write_bytes(before)

    def test_changed_export_hash_is_rejected(self):
        path = next(iter(self.populate().values()))
        (path/'selected/model.pt').write_bytes(b'changed synthetic fixture')
        with mock.patch.object(v2.runner, 'verify_run', side_effect=self.checked):
            with self.assertRaisesRegex(ValueError, 'binding changed'):
                v2.preflight(self.plan, self.context, [self.runs], root=self.root)

    def test_selected_receipt_identity_and_validation_must_match(self):
        path = next(iter(self.populate().values()))
        receipt = json.loads((path/'selected/receipt.json').read_text())
        for change in ({'seed': 99}, {'validation': {'synthetic_metadata': 'changed'}}):
            self.write_json(path/'selected/receipt.json', {**receipt, **change})
            with mock.patch.object(v2.runner, 'verify_run', side_effect=self.checked):
                with self.assertRaises(ValueError): v2.preflight(self.plan, self.context, [self.runs], root=self.root)

    def test_preflight_never_opens_test_examples_and_preserves_all_fixture_files(self):
        self.populate()
        before = {str(p): (sha256(p), p.stat().st_mtime_ns, p.stat().st_mode) for p in self.root.rglob('*') if p.is_file()}
        with mock.patch('numpy.load', side_effect=AssertionError('No example mapping in discovery fixture')) as load:
            records = self.preflight()
        load.assert_not_called()
        after = {str(p): (sha256(p), p.stat().st_mtime_ns, p.stat().st_mode) for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(records), 24)
        self.assertFalse(list(self.root.rglob('selection_lock.json')))

    def test_lock_metadata_binds_adapter_and_config_without_writing_a_lock(self):
        self.populate();records = self.preflight()
        document = v2.lock_document(self.plan, self.context, records, [self.runs], root=self.root)
        self.assertEqual(document['adapter']['sources'][v2.ADAPTER_SOURCE], sha256(Path(v2.__file__)))
        self.assertEqual(document['plan_digest'], v2.tio.digest(self.plan))
        for item in document['fits']:
            self.assertEqual(item['configuration_sha256'], sha256(Path(item['run'])/'configuration.json'))
        self.assertFalse(list(self.root.rglob('selection_lock.json')))
        with self.assertRaisesRegex(ValueError, '24 unique'):
            v2.lock_document(self.plan, self.context, records[:-1]+records[:1], [self.runs], root=self.root)

    def test_adapter_source_tampering_is_rejected(self):
        path = self.root/v2.ADAPTER_SOURCE
        path.write_bytes(path.read_bytes()+b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Execution source changed'): v2.adapter_provenance(self.root)

    def test_existing_lock_destination_is_rejected_before_preflight(self):
        directory = self.root/'output/experiments/v5_missing_data/model_comparison/v1/occupied'
        directory.mkdir(parents=True);(directory/'historical.txt').write_text('preserve')
        with mock.patch.object(v2, 'preflight') as preflight:
            with self.assertRaisesRegex(ValueError, 'destination exists'):
                v2.make_lock(self.plan, self.context, [self.runs], directory, root=self.root)
            preflight.assert_not_called()
        self.assertEqual((directory/'historical.txt').read_text(), 'preserve')

    def test_lock_destination_inside_run_root_is_rejected(self):
        root = self.root/'output/experiments/v5_missing_data/model_comparison/v1/full_runs'
        root.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'separate from run roots'):
            v2.make_lock(self.plan, self.context, [root], root/'new_lock', root=self.root)

    def test_v2_lock_verification_retains_original_hash_checks(self):
        with mock.patch.object(v2.original, 'load_lock', side_effect=ValueError('Locked fit/export changed')) as loader:
            with self.assertRaisesRegex(ValueError, 'Locked fit/export changed'):
                v2.verify_lock(self.plan, self.context, self.root/'unused', root=self.root)
        loader.assert_called_once()

    def test_lock_configuration_byte_change_is_rejected(self):
        self.populate();records = self.preflight()
        document = v2.lock_document(self.plan, self.context, records, [self.runs], root=self.root)
        path = Path(records[0]['run'])/'configuration.json';path.write_bytes(path.read_bytes()+b' ')
        with mock.patch.object(v2.original, 'load_lock', return_value=(document, object())):
            with self.assertRaisesRegex(ValueError, 'Locked configuration changed'):
                v2.verify_lock(self.plan, self.context, self.root/'unused', root=self.root)

    def test_cli_preflight_calls_original_plan_checks_and_creates_no_output(self):
        self.populate()
        with mock.patch.object(v2.runner, 'load_plan', return_value=self.plan) as loader, \
             mock.patch.object(v2, 'CohortContext', return_value=self.context), \
             mock.patch.object(v2.runner, 'verify_run', side_effect=self.checked), \
             mock.patch.object(v2, 'preflight', wraps=lambda p,c,roots: v2_preflight(p,c,roots,root=self.root)), \
             redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
            result = v2.main(['--plan', str(self.root/'plan.json'), '--data-dir', str(self.root/'cohorts'),
                '--stage', 'preflight', '--runs-dir', str(self.runs)])
        self.assertEqual(result, 0);loader.assert_called_once()
        result = json.loads(out.getvalue());self.assertEqual(result['models'], 24)
        self.assertFalse(result['selection_lock_created']);self.assertFalse(result['test_examples_opened'])
        self.assertFalse(result['model_forward_or_scoring'])

    def test_cli_plan_integrity_failure_stops_before_cohort_access(self):
        with mock.patch.object(v2.runner, 'load_plan', side_effect=ValueError('Execution source changed')) as loader, \
             mock.patch.object(v2, 'CohortContext') as cohort, redirect_stderr(io.StringIO()):
            result = v2.main(['--plan', 'plan.json', '--data-dir', 'cohorts', '--stage', 'preflight',
                              '--runs-dir', str(self.runs)])
        self.assertEqual(result, 1);loader.assert_called_once();cohort.assert_not_called()

    def test_cli_lock_uses_lock_dir_without_creating_a_lock(self):
        directory = self.root/'future-lock'
        with mock.patch.object(v2.runner, 'load_plan', return_value=self.plan), \
             mock.patch.object(v2, 'CohortContext', return_value=self.context), \
             mock.patch.object(v2, 'make_lock', return_value={'status': 'synthetic-dispatch-only'}) as lock, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = v2.main(['--plan', 'plan.json', '--data-dir', 'cohorts', '--stage', 'lock',
                              '--runs-dir', str(self.runs), '--lock-dir', str(directory)])
        self.assertEqual(result, 0)
        lock.assert_called_once_with(self.plan, self.context, [self.runs], directory)
        self.assertFalse(directory.exists())

    def test_cli_lock_rejects_ambiguous_output_dir_before_any_action(self):
        with mock.patch.object(v2.runner, 'load_plan') as loader, \
             mock.patch.object(v2, 'make_lock') as lock, redirect_stderr(io.StringIO()) as out:
            result = v2.main(['--plan', 'plan.json', '--data-dir', 'cohorts', '--stage', 'lock',
                '--runs-dir', str(self.runs), '--lock-dir', str(self.root/'lock'),
                '--output-dir', str(self.root/'evaluation')])
        self.assertEqual(result, 1);loader.assert_not_called();lock.assert_not_called()
        self.assertIn('lock creation uses --lock-dir', out.getvalue())
        self.assertFalse((self.root/'lock').exists());self.assertFalse((self.root/'evaluation').exists())

    def test_conflicting_or_missing_cli_roots_are_rejected(self):
        for args in (SimpleNamespace(runs_dir='x', primary_dir='y', sensitivity_dir=None),
                     SimpleNamespace(runs_dir=None, primary_dir='y', sensitivity_dir=None)):
            with self.assertRaises(ValueError): v2.input_roots(args)


# Keep the actual function available when a CLI-dispatch test wraps its public name.
v2_preflight = v2.preflight


if __name__ == '__main__':
    unittest.main()
