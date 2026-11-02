"""Final test-only inference for explicitly locked, completed v2 FedAvg models.

Use --verify-only to inspect provenance without loading examples or predicting.
Evaluation exclusively creates its output. There is no training, tuning,
selection by test score, resume or overwrite option.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics

import numpy as np
import torch

from .experiment_data import FEATURES, FrozenData, immutable_json, read_json, sha256
from .experiment_eval import evaluate as common_evaluate
from .experiment_models import neural_predict
from .experiment_resources import Resources
from .federated_io import (digest_json, environment, load_commits, load_selected,
                           source_inventory)
from .federated_rounds import DEFAULTS, configure_runtime, initialize_model, sample_clients, validation_due
from .federated_train import GlobalModelState
from .io_utils import sync_directory

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)


def execution_sources(root=ROOT):
    names = [*source_inventory(), 'src/experiment_models.py', 'src/federated_evaluate.py']
    return {name: sha256(Path(root) / name) for name in names}


def _inside(root, value):
    if not isinstance(value, str) or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Recorded paths must be relative paths inside the project/run')
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path == Path(root).resolve():
        raise ValueError('Recorded path escapes its project/run')
    return path


def _hash(path, expected):
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ValueError(f'Hash mismatch: {path}')


def verify_header(directory, cohort):
    """Read contract/metadata only; never open example arrays in preflight."""
    directory = Path(directory)
    _hash(directory / 'contract.json', cohort['contract_sha256'])
    contract = read_json(directory / 'contract.json')
    _hash(directory / 'preprocessing.json', contract['metadata_sha256'])
    meta = read_json(directory / 'preprocessing.json')
    count, node_count = cohort['expected_windows'], cohort['expected_nodes']
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (count, node_count)):
        raise ValueError('Expected cohort sizes must be positive integers')
    if (contract['feature_names'] != FEATURES or meta['feature_names'] != FEATURES
            or contract['weather_mask_indices'] != [7, 8]
            or contract['history_steps'] != 6 or meta['history_steps'] != 6
            or contract['horizon_steps'] != 6 or meta['horizon_steps'] != 6):
        raise ValueError('Feature/history/horizon/weather contract differs from the benchmark')
    if meta['settings']['resample_minutes'] != 10 or meta['settings']['horizon_minutes'] != 60:
        raise ValueError('Unexpected sampling interval or forecast horizon')
    if contract['chronological_cuts'] != meta['split_boundaries_utc']:
        raise ValueError('Chronological cuts differ from frozen preprocessing')
    if len(meta['nodes']) != node_count or len(set(meta['nodes'])) != node_count:
        raise ValueError('Node cohort differs from declaration')
    for name, shape in (('X_test', [count, 6, 14]), ('y_test', [count, 1]),
                        ('node_test', [count]), ('origin_test', [count]), ('node_names', [node_count])):
        if contract['arrays'][name + '.npy']['shape'] != shape:
            raise ValueError(f'Test cohort shape/count mismatch: {name}')
    for key, width in (('x_mean', 14), ('x_std', 14), ('y_mean', 1), ('y_std', 1)):
        values = np.asarray(meta[key], dtype=np.float64)
        if (meta[key] != contract['scalers'][key] or values.shape != (width,)
                or not np.isfinite(values).all() or (key.endswith('std') and (values <= 0).any())):
            raise ValueError(f'Invalid frozen scaler: {key}')
    if meta['x_mean'][13] != 0 or meta['x_std'][13] != 1:
        raise ValueError('Frozen categorical node index was standardized')
    return contract, meta


def identity_record(nodes, origins, expected_count):
    nodes, origins = np.asarray(nodes), np.asarray(origins)
    if (nodes.shape != (expected_count,) or nodes.dtype.kind != 'U'
            or origins.shape != (expected_count,) or origins.dtype != np.dtype('datetime64[ns]')
            or np.isnat(origins).any() or (nodes == '').any()):
        raise ValueError('Missing, malformed or invalid test-window identities')
    keys = np.rec.fromarrays([nodes, origins.view(np.int64)], names=['node', 'origin_ns'])
    if len(np.unique(keys)) != expected_count:
        raise ValueError('Duplicate test-window identities')
    digest = hashlib.sha256(b'M100 ordered (node UTF-8, origin UTC ns) v1\0')
    for node, origin in zip(nodes, origins.view(np.int64)):
        encoded = str(node).encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
        digest.update(int(origin).to_bytes(8, 'little', signed=True))
    return {'count': expected_count, 'unique_count': len(keys),
            'ordered_sha256': digest.hexdigest(), 'origin_timezone': 'UTC',
            'identity': '(node string, forecast origin datetime64[ns])'}


class FrozenTestCohort:
    """Test-only facade over the existing verified FrozenData implementation.

FrozenData.verify checks all splits read-only for integrity. Only the test split
is exposed to inference/reporting; no scalers are fitted or data transformed here.
"""
    def __init__(self, directory, declaration):
        contract, meta = verify_header(directory, declaration)
        data = FrozenData(directory)
        data.verify()
        self.__data = data
        self.__arrays = data.split('test')
        x, y, origins, nodes = self.__arrays
        for key, array in zip(('X_test', 'y_test', 'origin_test', 'node_test'), self.__arrays):
            expected = contract['arrays'][key + '.npy']
            if list(array.shape) != expected['shape'] or str(array.dtype) != expected['dtype']:
                raise ValueError(f'Frozen test array schema mismatch: {key}')
        if x.dtype != np.float32 or y.dtype != np.float32:
            raise ValueError('Benchmark inputs and targets must remain float32')
        self.names = data.names
        self.coefficients = data.coefficients
        self.node_names = tuple(str(name) for name in data.names)
        self.baseline_coefficients = data.coefficients
        self.meta, self.contract, self.contract_sha256 = meta, contract, data.contract_sha256
        if set(nodes.tolist()) != set(self.node_names) or len(self.names) != declaration['expected_nodes']:
            raise ValueError('Missing or unexpected nodes in test cohort; filtering is forbidden')
        coefficients = np.array([meta['x_std'][0] / meta['y_std'][0],
                                 (meta['x_mean'][0] - meta['y_mean'][0]) / meta['y_std'][0]])
        if self.coefficients.shape != (2,) or not np.allclose(self.coefficients, coefficients, rtol=0, atol=1e-12):
            raise ValueError('Frozen persistence coefficients differ from scalers')
        self.identity = identity_record(nodes, origins, declaration['expected_windows'])
        self.__nodes = nodes.copy()
        self.__origins = origins.copy()

    def split(self, split):
        if split != 'test':
            raise PermissionError('The final evaluator exposes only the test split')
        return self.__arrays

    def check_identity(self, nodes, origins):
        record = identity_record(nodes, origins, self.identity['count'])
        if (record != self.identity or not np.array_equal(nodes, self.__nodes)
                or not np.array_equal(origins, self.__origins)):
            raise ValueError('Prediction test-window identities/order differ from the frozen test cohort')
        return record

    def verify_unchanged(self):
        _hash(self.__data.path / 'contract.json', self.contract_sha256)
        self.__data.verify()
        self.check_identity(self.__arrays[3], self.__arrays[2])


@dataclass(frozen=True)
class VerifiedSelection:
    entry: dict
    manifest: dict
    exported: dict
    record: dict


def verify_selection(root, entry, cohort, sources):
    root = Path(root)
    run = _inside(root, entry['run_dir'])
    for suffix, key in (('manifest.json', 'manifest_sha256'), ('selected/receipt.json', 'selected_receipt_sha256'),
                        ('selected/model.pt', 'model_sha256')):
        _hash(run / suffix, entry[key])
    _hash(_inside(root, entry['config_file']), entry['config_file_sha256'])
    event_path = _inside(run, entry['completion_event'])
    _hash(event_path, entry['completion_event_sha256'])
    manifest, event = read_json(run / 'manifest.json'), read_json(event_path)
    config = manifest['config']
    if (config != read_json(_inside(root, entry['config_file'])) or config['seed'] != entry['seed']
            or config['kind'] != 'main' or config['rounds'] != 200
            or config['model'] != 'gru' or config['weather'] != 'masked'
            or config['frozen_contract_sha256'] != cohort['contract_sha256']
            or manifest['frozen_contract_sha256'] != cohort['contract_sha256']):
        raise ValueError('Selected run configuration, seed, protocol or frozen contract differs')
    if manifest['config_digest'] != digest_json(config):
        raise ValueError('Training configuration digest mismatch')
    if set(manifest['source_hashes']) != set(source_inventory()):
        raise ValueError('Training execution-source inventory differs')
    for name, expected in manifest['source_hashes'].items():
        if sources.get(name) != expected:
            raise ValueError(f'Training source differs from evaluation source: {name}')
        _hash(_inside(run / 'source', name), expected)
    current_env = environment('cpu')
    if (manifest['environment']['dependencies'] != current_env['dependencies']
            or manifest['environment']['python'] != platform.python_version()):
        raise ValueError('Dependencies/Python differ from the completed training run')
    payload, receipts = load_commits(run)
    exported = load_selected(run)
    if payload is None or payload['round'] != 200 or len(receipts) != 201:
        raise ValueError('A complete 200-round checkpoint chain is required')
    names = tuple(client['id'] for client in manifest['clients'])
    counts = {client['id']: client['count'] for client in manifest['clients']}
    if (len(names) != cohort['expected_nodes'] or len(counts) != len(names)
            or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts.values())
            or sum(counts.values()) != manifest['training_windows']):
        raise ValueError('Invalid frozen client membership/counts')
    examples = steps = 0
    for i, receipt in enumerate(receipts):
        expected = list(sample_clients(names, config['clients_per_round'], config['seed'], i)) if i else []
        if (receipt['selected_clients'] != expected
                or [c['client_id'] for c in receipt['clients']] != expected
                or (receipt['validation'] is not None) != validation_due(i, config)):
            raise ValueError('Recorded participation/validation schedule differs')
        for client in receipt['clients']:
            count = counts[client['client_id']]
            if (client['training_windows'] != count or client['examples_processed'] != count
                    or client['optimizer_steps'] != math.ceil(count / config['batch_size'])
                    or client['initial_fingerprint'] != receipts[i-1]['global_fingerprint']):
                raise ValueError('Recorded local training counters/initialization differ')
            examples += count; steps += client['optimizer_steps']
        if receipt['validation'] is not None:
            metrics = receipt['validation']
            if (metrics['split'] != 'val' or metrics['unit'] != 'W'
                    or metrics['samples'] != manifest['validation_windows']
                    or metrics['evaluated_nodes'] != cohort['expected_nodes']
                    or not np.isfinite(metrics['rmse_w']) or metrics['rmse_w'] < 0):
                raise ValueError('Invalid recorded validation cohort/selection score')
    best = min((r for r in receipts if r['validation'] is not None),
               key=lambda r: (r['validation']['rmse_w'], r['round']))
    if (best['round'] != entry['selected_round'] or exported['selected_round'] != best['round']
            or payload['best_round'] != best['round'] or exported['validation_rmse_w'] != best['validation']['rmse_w']
            or payload['best_rmse_w'] != best['validation']['rmse_w']
            or payload['examples_processed'] != examples or payload['optimizer_steps'] != steps):
        raise ValueError('Frozen selection is not the earliest validation minimum, or counters differ')
    selected_checkpoint = torch.load(run / 'rounds' / f"{best['round']:04d}" / 'state.pt',
                                     map_location='cpu', weights_only=True)
    for state in (payload['best_state'], selected_checkpoint['state']):
        if set(state) != set(exported['state']) or any(not torch.equal(state[k], v) for k, v in exported['state'].items()):
            raise ValueError('Selected export differs from its recorded validation checkpoint')
    if any(not torch.isfinite(value).all().item() for value in exported['state'].values()):
        raise ValueError('Selected checkpoint has nonfinite parameters')
    if (event['status'] != 'complete' or event['completed_round'] != 200 or event['selected_round'] != best['round']
            or event['examples_processed'] != examples or event['optimizer_steps'] != steps
            or event['selected_validation_rmse_w'] != best['validation']['rmse_w']
            or event['export_reload_check']['exact_predictions_equal'] is not True
            or event['export_reload_check']['finite_parameters'] is not True
            or event['export_reload_check']['validation'] != best['validation']):
        raise ValueError('Completion event does not corroborate the selected checkpoint')
    record = {**entry, 'selected_model': entry['run_dir'] + '/selected/model.pt',
              'config': config, 'training_source_hashes': manifest['source_hashes'],
              'selected_validation_rmse_w': best['validation']['rmse_w'], 'verified_round_checkpoints': len(receipts),
              'final_round_receipt_sha256': sha256(run / 'rounds/0200/receipt.json')}
    return VerifiedSelection(entry, manifest, exported, record)


def preflight(root, plan, directory):
    if (plan['version'] != 1 or plan['kind'] != 'frozen_fedavg_final_test'
            or [run['seed'] for run in plan['runs']] != list(SEEDS)):
        raise ValueError('The final test plan must declare seeds 42, 43 and 44 exactly once, in order')
    sources = execution_sources(root)
    if sources != plan['execution_sources']:
        raise ValueError('Evaluation execution-source hashes differ from the reviewed plan')
    _, meta = verify_header(directory, plan['cohort'])
    selections = [verify_selection(root, entry, plan['cohort'], sources) for entry in plan['runs']]
    for selected in selections:
        if ([client['id'] for client in selected.manifest['clients']] != meta['nodes']
                or [client['node_index'] for client in selected.manifest['clients']] != list(range(len(meta['nodes'])))):
            raise ValueError('Training node-index mapping differs from frozen preprocessing')
    config = {k: v for k, v in selections[0].manifest['config'].items() if k != 'seed'}
    for selected in selections[1:]:
        if ({k: v for k, v in selected.manifest['config'].items() if k != 'seed'} != config
                or selected.manifest['clients'] != selections[0].manifest['clients']):
            raise ValueError('Models do not share identical protocol/partitions except seed')
    return selections


def load_frozen_model(cohort, selection, device):
    if tuple(client['id'] for client in selection.manifest['clients']) != cohort.node_names:
        raise ValueError('Selected model node-index mapping differs from the test cohort')
    model = initialize_model(cohort, selection.manifest['config'])
    expected = GlobalModelState(model, round_id=selection.entry['selected_round'])
    checked = expected._validate(selection.exported['state'], label='frozen selected export')
    model.load_state_dict(checked, strict=True)
    model.eval()
    return model.to(device)


@dataclass(frozen=True)
class TestPredictions:
    scaled: np.ndarray
    nodes: np.ndarray
    origins: np.ndarray
    split: str = 'test'


def infer_test(cohort, selection, device):
    model = load_frozen_model(cohort, selection, device)
    before = GlobalModelState(model, round_id=selection.entry['selected_round']).fingerprint
    x, _, origins, nodes = cohort.split('test')
    config = {**selection.manifest['config'], 'device': str(device)}
    predicted = neural_predict(model, x, config)
    if GlobalModelState(model, round_id=selection.entry['selected_round']).fingerprint != before:
        raise ValueError('Inference changed frozen model parameters or buffers')
    result = TestPredictions(predicted, np.array(nodes, copy=True), np.array(origins, copy=True))
    validate_predictions(cohort, result)
    return result


def validate_predictions(cohort, predictions):
    if predictions.split != 'test':
        raise ValueError('Only final test predictions are accepted; validation is forbidden')
    identity = cohort.check_identity(predictions.nodes, predictions.origins)
    values = np.asarray(predictions.scaled)
    if values.shape != (identity['count'],) or not np.isfinite(values).all():
        raise ValueError('Missing, mis-sized or nonfinite test predictions')
    return identity


def improvement_percent(model_score, persistence_score):
    if not np.isfinite([model_score, persistence_score]).all() or min(model_score, persistence_score) < 0:
        raise ValueError('Comparison metrics must be finite nonnegative errors')
    if persistence_score == 0:
        return 0. if model_score == 0 else None
    return 100. * (1. - model_score / persistence_score)


def comparison_metrics(metrics):
    return {f'{key}_improvement_percent': improvement_percent(metrics[f'{key}_w'], metrics[f'persistence_{key}_w'])
            for key in ('rmse', 'mae')}


def evaluate_predictions(cohort, predictions, output):
    identity = validate_predictions(cohort, predictions)
    # This is the authoritative v1 watts transformation, persistence and metric code.
    metrics = common_evaluate(cohort, 'test', predictions.scaled, predictions.nodes, predictions.origins, output)
    per_node = read_json(Path(output) / 'per_node.json')
    expected_counts = Counter(cohort.split('test')[3].tolist())
    if set(per_node) != set(cohort.node_names) or any(row['samples'] != expected_counts[node] for node, row in per_node.items()):
        raise ValueError('Common evaluator omitted or changed a test-node cohort')
    augmented = {node: {**row, **comparison_metrics(row)} for node, row in per_node.items()}
    comparison = {**metrics, **comparison_metrics(metrics), 'window_identity': identity,
                  'nodes_beating_persistence_rmse': sum(r['rmse_w'] < r['persistence_rmse_w'] for r in per_node.values()),
                  'nodes_beating_persistence_mae': sum(r['mae_w'] < r['persistence_mae_w'] for r in per_node.values())}
    immutable_json(Path(output) / 'per_node_comparison.json', augmented)
    immutable_json(Path(output) / 'comparison.json', comparison)
    return comparison


def summarize(rows):
    if [row['seed'] for row in rows] != list(SEEDS):
        raise ValueError('Results must contain all three predeclared seeds without selection/ranking')
    first = rows[0]
    if any(row['window_identity'] != first['window_identity'] or row['samples'] != first['samples']
           or row['persistence_rmse_w'] != first['persistence_rmse_w']
           or row['persistence_mae_w'] != first['persistence_mae_w'] for row in rows):
        raise ValueError('Models/persistence used different test-window identities, targets or cohorts')
    result = {'split': 'test', 'unit': 'W', 'seeds': list(SEEDS), 'fits': rows,
              'window_identity': first['window_identity'], 'samples_per_seed': first['samples'],
              'std_definition': 'sample standard deviation across three seed metrics; ddof=1; not a confidence interval',
              'aggregation': 'mean of three individual seed metrics; not an ensemble prediction',
              'percentage_sign': 'positive means improvement; negative means deterioration',
              'zero_baseline_policy': 'both zero: 0 percent; positive model error with zero baseline: null (undefined)',
              'test_results_are_final': True, 'test_based_model_selection': False}
    for key in ('rmse', 'mae'):
        scores = [row[f'{key}_w'] for row in rows]
        result[f'mean_{key}_w'] = statistics.mean(scores)
        result[f'std_{key}_w'] = statistics.stdev(scores)
        result[f'persistence_{key}_w'] = first[f'persistence_{key}_w']
        result[f'mean_{key}_improvement_percent'] = improvement_percent(statistics.mean(scores), first[f'persistence_{key}_w'])
    return result


def evaluate_locked(root, plan, directory, output, *, device='cuda:0', plan_sha256=None):
    """The only inference entry point. Call only after implementation review."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('Evaluation output exists; completed or partial results are never overwritten')
    selections = preflight(root, plan, directory)
    runtime_config = {**selections[0].manifest['config'], 'device': device}
    target_device = configure_runtime(runtime_config)
    output.mkdir(parents=True, exist_ok=False)
    # Bind ALL models before loading any test examples or producing test scores.
    lock = {'version': 1, 'plan_sha256': plan_sha256, 'plan_digest': digest_json(plan),
            'cohort': plan['cohort'], 'models': [selected.record for selected in selections],
            'selection_criterion': 'earliest minimum recorded pooled validation RMSE; frozen before test access',
            'test_based_model_selection': False}
    immutable_json(output / 'selection_lock.json', lock)
    immutable_json(output / 'plan.json', plan)
    settings = {'split': 'test', 'weather': 'masked', 'dtype': 'float32', 'device': str(target_device),
                'eval_batch_size': runtime_config['eval_batch_size'], 'num_threads': runtime_config['num_threads'],
                'scalers': 'unchanged frozen preprocessing; no fitting',
                'persistence': 'last completed ten-minute binned power (last input step, channel 0), held constant over the next-hour target',
                'environment': environment(target_device), 'execution_sources': plan['execution_sources'],
                'plan_sha256': plan_sha256, 'selection_lock_sha256': sha256(output / 'selection_lock.json'),
                'project_directory': str(root.resolve()),
                'data_directory': str(Path(directory).resolve()), 'data_status': 'existing development benchmark'}
    immutable_json(output / 'manifest.json', settings)
    for name, expected in plan['execution_sources'].items():
        source = _inside(root, name)
        _hash(source, expected)
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(source.read_bytes()); handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o444)
        _hash(destination, expected)
        sync_directory(destination.parent)
    sync_directory(output)
    try:
        with Resources() as resources:
            cohort = FrozenTestCohort(directory, plan['cohort'])
            immutable_json(output / 'test_window_identity.json', cohort.identity)
            rows = []
            for selection in selections:
                prediction = infer_test(cohort, selection, target_device)
                metrics = evaluate_predictions(cohort, prediction, output / f"seed{selection.entry['seed']}")
                rows.append({'seed': selection.entry['seed'], 'selected_model': selection.record['selected_model'],
                             'model_sha256': selection.entry['model_sha256'], 'selected_round': selection.entry['selected_round'], **metrics})
            result = summarize(rows)
            # Stop rather than finalize if protected sources, exports or frozen data changed during inference.
            preflight(root, plan, directory)
            cohort.verify_unchanged()
            immutable_json(output / 'results.json', result)
        immutable_json(output / 'resources.json', resources.result)
        artifacts = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*')
                     if p.is_file() and '.matplotlib' not in p.parts}
        immutable_json(output / 'completed.json', {'version': 1, 'status': 'complete', 'split': 'test',
                       'selection_lock_sha256': sha256(output / 'selection_lock.json'), 'artifacts': artifacts})
        sync_directory(output)
        return result
    except BaseException as error:
        immutable_json(output / 'failed.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                       'policy': 'Preserve partial artifacts. Never tune from observed test scores or overwrite this output.'})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--device', default='cuda:0', help='Explicit inference device; CUDA never silently falls back')
    parser.add_argument('--verify-only', action='store_true', help='No examples loaded, predictions, metrics or output writes')
    args = parser.parse_args(argv)
    try:
        plan_path = Path(args.plan).resolve()
        if not plan_path.is_relative_to(ROOT / 'configs/experiments/v2_fedavg'):
            raise ValueError('Plan must be a new v2 FedAvg configuration')
        plan = read_json(plan_path)
        if plan['cohort'] != {'contract_sha256': DEFAULTS['frozen_contract_sha256'],
                              'expected_windows': 75566, 'expected_nodes': 979}:
            raise ValueError('CLI requires the original M100 75,566-window / 979-node test benchmark')
        if args.verify_only:
            selections = preflight(ROOT, plan, args.data_dir)
            result = {'status': 'verified_only', 'test_evaluation_run': False,
                      'cohort': plan['cohort'], 'models': [selection.record for selection in selections]}
        else:
            if args.output_dir is None:
                raise ValueError('--output-dir is required for actual evaluation')
            output = Path(args.output_dir).resolve()
            allowed = ROOT / 'output/experiments/v2_fedavg/test_evaluation'
            if not output.is_relative_to(allowed) or output == allowed:
                raise ValueError('Output must be a new directory under output/experiments/v2_fedavg/test_evaluation/')
            result = evaluate_locked(ROOT, plan, args.data_dir, output, device=args.device, plan_sha256=sha256(plan_path))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'FEDAVG EVALUATION ERROR: {error}\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
