"""Additive v5 evaluator: manifest discovery and unchanged v1 verification.

Stages are explicit: preflight (read only), lock, verify (lock only), evaluate.
Lock creation uses --lock-dir; --output-dir is accepted only for evaluate.
Neither import, preflight, lock nor verify constructs test SplitData or scores
predictions. Cohort verification hashes file bytes, including test artifacts,
but does not map test examples. Only a separately invoked evaluate stage can
call the original evaluator. No training entry point is provided.

The original plan/source hashes remain authoritative for training. A future
lock additionally binds this adapter's source hash, discovered roots, and each
recorded configuration hash. Existing v1 locks are not silently upgraded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.experiment_data import immutable_json, read_json, sha256
from src.v5_missing_data import evaluate_policy_models as original
from src.v5_missing_data import run_policy_experiments as runner
from src.v5_missing_data import training_io as tio
from src.v5_missing_data.training_data import CohortContext, POLICIES

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_SOURCE = 'src/v5_missing_data/evaluate_policy_models_v2.py'
ADAPTER_SHA256 = sha256(Path(__file__))
EXPECTED = tuple((m, p, s) for m in runner.ALGORITHMS for p in POLICIES for s in runner.SEEDS)


def execution_checks(plan, root=ROOT):
    """Retain actual-byte checks even when preflight is called as a function."""
    tio.sources(root, plan['sources'])
    for name, checksum in plan['evidence'].items():
        if sha256(root / name) != checksum:
            raise ValueError('Protocol or cohort evidence changed: ' + name)


def discover_runs(roots):
    """Find exactly 24 main manifests. Directory names never define identity."""
    resolved = []
    for value in roots:
        path = Path(value).absolute()
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Run root must be an existing non-symlink directory: ' + str(path))
        path = path.resolve()
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise ValueError('At least one completed run root is required')
    for i, path in enumerate(resolved):
        for other in resolved[i+1:]:
            if path.is_relative_to(other) or other.is_relative_to(path):
                raise ValueError('Overlapping run roots are ambiguous')
    found = {}
    for root in resolved:
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink inside run root is not allowed: ' + str(path))
            if not path.is_file():
                continue
            if path.name in ('configuration.json', 'completed.json') and not (path.parent/'manifest.json').is_file():
                raise ValueError('Run evidence without manifest: ' + str(path.parent))
            if path.name != 'manifest.json':
                continue
            manifest = read_json(path)
            if not isinstance(manifest, dict):
                raise ValueError('Invalid run manifest: ' + str(path))
            key = (manifest.get('method'), manifest.get('policy'), manifest.get('seed'))
            if (not isinstance(key[0], str) or not isinstance(key[1], str)
                    or type(key[2]) is not int or key not in EXPECTED):
                raise ValueError('Unexpected run identity: ' + str(path))
            if manifest.get('version') != 1 or manifest.get('kind') != 'main' or manifest.get('cap') != 200:
                raise ValueError('Invalid main run/version/cap: ' + str(path))
            if key in found:
                raise ValueError('Duplicate run identity: ' + repr(key))
            found[key] = path.parent
    missing = [key for key in EXPECTED if key not in found]
    if missing or len(found) != 24:
        raise ValueError('Missing expected run identities: ' + repr(missing))
    return {key: found[key] for key in EXPECTED}


def preflight(plan, context, roots, *, root=ROOT):
    """Call the original official verifier for every discovered fit; no writes."""
    execution_checks(plan, root)
    runs = discover_runs(roots)
    records = []
    for key, run in runs.items():
        before = sha256(run/'manifest.json')
        checked = runner.verify_run(plan, context, run)
        manifest = read_json(run/'manifest.json')
        complete = read_json(run/'completed.json')
        receipt = read_json(run/'selected/receipt.json')
        if (checked.get('status') != 'verified_complete' or checked.get('completed_round') != 200
                or checked.get('model_forward_or_scoring') is not False
                or complete.get('status') != 'complete' or complete.get('completed_round') != 200
                or checked.get('selected_round') != receipt.get('selected_round')
                or complete.get('selected_round') != receipt.get('selected_round')
                or (manifest.get('method'), manifest.get('policy'), manifest.get('seed')) != key
                or (receipt.get('method'), receipt.get('policy'), receipt.get('seed')) != key
                or manifest.get('kind') != 'main' or manifest.get('cap') != 200):
            raise ValueError('Final test requires every declared verified complete main fit: ' + str(run))
        if (before != sha256(run/'manifest.json')
                or receipt.get('manifest_sha256') != before
                or receipt.get('cohort_receipt_sha256') != context.receipt_sha256
                or receipt.get('model_sha256') != sha256(run/'selected/model.pt')
                or receipt.get('validation') != complete.get('selected_validation')
                or read_json(run/'configuration.json') != plan):
            raise ValueError('Selected export/configuration binding changed: ' + str(run))
        records.append({'method': key[0], 'policy': key[1], 'seed': key[2],
            'selected_round': receipt['selected_round'], 'validation': receipt['validation'],
            'run': str(run), 'model_sha256': sha256(run/'selected/model.pt'),
            'selected_receipt_sha256': sha256(run/'selected/receipt.json'),
            'completed_sha256': sha256(run/'completed.json'), 'manifest_sha256': before,
            'configuration_sha256': sha256(run/'configuration.json')})
        print('Verified %d/24: %s %s seed%d' % (len(records), *key), file=sys.stderr, flush=True)
    execution_checks(plan, root)
    return records


def adapter_provenance(root=ROOT):
    tio.sources(root, {ADAPTER_SOURCE: ADAPTER_SHA256})
    return {'version': 2, 'sources': {ADAPTER_SOURCE: ADAPTER_SHA256},
            'discovery': 'exact manifest identity; 24 unique complete main fits'}


def lock_document(plan, context, records, roots, *, root=ROOT):
    """Construct metadata in memory; this function opens no test examples."""
    keys = [(r['method'], r['policy'], r['seed']) for r in records]
    if len(keys) != 24 or set(keys) != set(EXPECTED):
        raise ValueError('Lock requires exactly 24 unique expected identities')
    return {'version': 1, 'plan_digest': tio.digest(plan),
        'cohort_receipt_sha256': context.receipt_sha256,
        'common_test_identity': context.contract['cohorts']['common/all_policies/test']['identity'],
        'fits': records, 'adapter': adapter_provenance(root),
        'run_roots': sorted({str(Path(p).resolve()) for p in roots}),
        'selection': 'Validation only; test period previously observed; no best-seed selection'}


def make_lock(plan, context, roots, directory, *, root=ROOT):
    directory = runner.output_path(root, directory)
    if directory.exists() or directory.is_symlink():
        raise ValueError('Lock destination exists; preserve partial/completed artifacts')
    for run_root in roots:
        run_root = Path(run_root).resolve()
        if directory.resolve().is_relative_to(run_root) or run_root.is_relative_to(directory.resolve()):
            raise ValueError('Lock directory must be separate from run roots')
    records = preflight(plan, context, roots, root=root)
    document = lock_document(plan, context, records, roots, root=root)
    directory.mkdir(parents=True, exist_ok=False)
    immutable_json(directory/'selection_lock.json', document)
    immutable_json(directory/'completed.json', {'status': 'locked',
        'selection_lock_sha256': sha256(directory/'selection_lock.json')})
    return {'status': 'locked', 'models': len(records), 'lock_directory': str(directory),
        'test_examples_opened': False, 'model_forward_or_scoring': False}


def verify_lock(plan, context, directory, *, root=ROOT):
    execution_checks(plan, root)
    lock, permission = original.load_lock(plan, context, Path(directory))
    if lock.get('adapter') != adapter_provenance(root):
        raise ValueError('Evaluator v2 source/provenance changed or missing')
    for item in lock['fits']:
        run = Path(item['run'])
        if (sha256(run/'configuration.json') != item.get('configuration_sha256')
                or read_json(run/'configuration.json') != plan):
            raise ValueError('Locked configuration changed: ' + str(run))
    return lock, permission


def input_roots(args):
    if args.runs_dir:
        if args.primary_dir or args.sensitivity_dir:
            raise ValueError('Use --runs-dir OR both matrix roots, not both forms')
        return [Path(args.runs_dir).absolute()]
    if not args.primary_dir or not args.sensitivity_dir:
        raise ValueError('Provide --runs-dir or both --primary-dir and --sensitivity-dir')
    return [Path(args.primary_dir).absolute(), Path(args.sensitivity_dir).absolute()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--stage', choices=('preflight', 'lock', 'verify', 'evaluate'), required=True)
    parser.add_argument('--runs-dir')
    parser.add_argument('--primary-dir')
    parser.add_argument('--sensitivity-dir')
    parser.add_argument('--lock-dir', help='Lock-stage destination; required existing lock for verify/evaluate')
    parser.add_argument('--output-dir', help='New test-evaluation destination; only valid for stage evaluate')
    args = parser.parse_args(argv)
    try:
        if args.stage != 'evaluate' and args.output_dir:
            raise ValueError('--output-dir is only for evaluate; lock creation uses --lock-dir')
        if args.stage in ('lock', 'verify', 'evaluate') and not args.lock_dir:
            raise ValueError('--lock-dir is required for this stage')
        if args.stage == 'preflight' and args.lock_dir:
            raise ValueError('Preflight writes nothing; omit --lock-dir')
        if args.stage in ('verify', 'evaluate') and (args.runs_dir or args.primary_dir or args.sensitivity_dir):
            raise ValueError('Verify/evaluate use recorded locked paths; omit run roots')
        if args.stage == 'evaluate' and not args.output_dir:
            raise ValueError('Evaluate requires a new --output-dir')
        roots = input_roots(args) if args.stage in ('preflight', 'lock') else None
        # Do not alter/bypass load_plan: it verifies all 34 original sources,
        # the exact protocol fields and six evidence hashes before any action.
        plan = runner.load_plan(Path(args.plan).resolve())
        context = CohortContext(args.data_dir, plan['cohort_receipt_sha256'])
        if args.stage == 'preflight':
            records = preflight(plan, context, roots)
            result = {'status': 'verified_complete', 'models': len(records), 'fits': records,
                'selection_lock_created': False, 'test_examples_opened': False,
                'model_forward_or_scoring': False}
        elif args.stage == 'lock':
            result = make_lock(plan, context, roots, Path(args.lock_dir).absolute())
        else:
            lock_dir = runner.output_path(ROOT, args.lock_dir)
            verify_lock(plan, context, lock_dir)
            if args.stage == 'verify':
                result = {'status': 'verified_complete', 'models': 24,
                    'test_examples_opened': False, 'model_forward_or_scoring': False}
            else:
                output = runner.output_path(ROOT, args.output_dir)
                result = original.evaluate_locked(plan, context, lock_dir, output)
        import json
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print('V5 EVALUATOR V2 ERROR: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
