#!/usr/bin/env python3
"""Frozen M100 comparisons: fit/validation and test reporting are separate commands."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

# Required before any CUDA context for deterministic recurrent training.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from src.experiment_data import FrozenData, freeze, immutable_json, read_json, sha256
from src.experiment_eval import evaluate
from src.experiment_models import fit_neural, fit_xgboost, predict_selected, training_indices
from src.experiment_resources import Resources

ROOT = Path(__file__).resolve().parent
COMMON = {'version', 'kind', 'model', 'weather', 'seed', 'device', 'num_threads', 'learning_rate', 'patience'}
NEURAL = {'epochs', 'batch_size', 'eval_batch_size', 'hidden_size', 'num_layers', 'node_embedding_size'}
BOOSTED = {'trees', 'max_depth', 'max_bin', 'subsample', 'colsample_bytree'}
LOG = logging.getLogger('m100.experiments')


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def validate_config(config):
    if config.get('model') not in ('lstm', 'gru', 'xgboost'):
        raise ValueError('Unsupported model')
    fields = COMMON | (BOOSTED if config['model'] == 'xgboost' else NEURAL)
    if config.get('kind') == 'pilot':
        fields |= {'train_limit'}
    if set(config) != fields:
        raise ValueError(f'Configuration fields mismatch: {set(config) ^ fields}')
    if config['version'] != 1 or config['kind'] not in ('comparison', 'pilot') or config['weather'] not in ('enabled', 'masked'):
        raise ValueError('Invalid version, experiment kind or weather mode')
    if config['seed'] not in (42, 43, 44) or config['device'] not in ('cpu', 'cuda:0', 'cuda:1'):
        raise ValueError('Invalid declared seed or device')
    integers = ['num_threads', 'patience'] + (['trees', 'max_depth', 'max_bin'] if config['model'] == 'xgboost' else
                ['epochs', 'batch_size', 'eval_batch_size', 'hidden_size', 'num_layers', 'node_embedding_size'])
    for field in integers:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if not 0 < config['learning_rate'] <= 1:
        raise ValueError('Invalid learning rate')
    if config['model'] == 'xgboost' and not all(0 < config[key] <= 1 for key in ('subsample', 'colsample_bytree')):
        raise ValueError('Invalid sampling fraction')
    if config['kind'] == 'pilot' and config['train_limit'] is not None and (type(config['train_limit']) is not int or config['train_limit'] <= 0):
        raise ValueError('Invalid pilot training limit')


def source_hashes():
    paths = [ROOT / 'experiments.py', ROOT / 'src/model.py', ROOT / 'src/io_utils.py', ROOT / 'src/forecast_plot.py']
    paths += sorted((ROOT / 'src').glob('experiment_*.py'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def dependencies():
    return {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}


def check_provenance(output):
    manifest = read_json(output / 'manifest.json')
    selection = read_json(output / 'selection.json')
    receipt = read_json(output / 'fit_completed.json')
    if receipt['manifest_sha256'] != sha256(output / 'manifest.json'):
        raise ValueError('Experiment manifest changed')
    if manifest['source_hashes'] != source_hashes():
        raise ValueError('Execution source changed; restore the recorded version for reporting')
    if manifest['dependencies'] != dependencies() or manifest['python'] != platform.python_version():
        raise ValueError('Dependency environment changed; restore recorded versions for reporting')
    for name, checksum in receipt['artifacts'].items():
        if sha256(output / name) != checksum:
            raise ValueError(f'Selected fit artifact changed: {name}')
    data = FrozenData(manifest['frozen_data'])
    if manifest['frozen_contract_sha256'] != data.contract_sha256:
        raise ValueError('Frozen data contract changed')
    data.verify()
    return manifest, selection, data


def fit(config_path, data_path, output):
    config = read_json(config_path)
    validate_config(config)
    data = FrozenData(data_path)
    data.verify()
    # Fail before touching any existing directory, including partially completed fits.
    output.mkdir(parents=True, exist_ok=False)
    import torch
    indices = training_indices(data, config)
    manifest = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                'config_file_sha256': sha256(config_path), 'frozen_data': str(data.path),
                'frozen_contract_sha256': data.contract_sha256, 'source_hashes': source_hashes(),
                'dependencies': dependencies(), 'python': platform.python_version(), 'platform': platform.platform(),
                'device': config['device'], 'input_dtype': 'float32', 'prediction_selection': 'validation RMSE in W only',
                'test_prediction_and_scoring': 'prohibited for pilot; separate report command for comparisons',
                'integrity_checks': 'All split hashes and alignment checked; test labels never used for selection',
                'training_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest(),
                'training_windows': len(indices), 'weather_mask': [7, 8] if config['weather'] == 'masked' else [],
                'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(config['device']) if config['device'].startswith('cuda') and torch.cuda.is_available() else None,
                'seed_policy': 'seeded initialization and shuffle; paired masks preserve seed and window order',
                'determinism': 'PyTorch deterministic algorithms; XGBoost GPU results may vary across versions/hardware'}
    immutable_json(output / 'manifest.json', manifest)
    for relative in manifest['source_hashes']:
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        target.chmod(0o444)
    del indices
    handler = logging.FileHandler(output / 'fit.log')
    logging.getLogger().addHandler(handler)
    try:
        with Resources() as resources:
            if config['model'] == 'xgboost':
                selection, original_predictions = fit_xgboost(data, config, output)
            else:
                selection, original_predictions = fit_neural(data, config, output)
            # Exercise the exported adapter with validation, never with held-out test labels.
            reloaded_predictions = predict_selected(data, config, output, selection, 'val')
            max_difference_w = float(np.max(np.abs(original_predictions - reloaded_predictions)) * data.meta['y_std'][0] * 1000)
            if max_difference_w > 0.1:
                raise ValueError(f'Export reload disagreement: {max_difference_w} W')
            _, y, origins, nodes = data.split('val')
            metrics = evaluate(data, 'val', reloaded_predictions, nodes, origins, output / 'validation')
            if abs(metrics['rmse_w'] - selection['best_validation_rmse_w']) > 0.01:
                raise ValueError('Common evaluator disagrees with validation selection')
            selection['reload_max_difference_w'] = max_difference_w
            selection['common_evaluator_verified_on'] = 'validation only'
        immutable_json(output / 'selection.json', selection)
        immutable_json(output / 'resources.json', resources.result)
        artifacts = ['selection.json', 'resources.json', 'history.json', selection['model_file']]
        if config['model'] == 'xgboost':
            artifacts += ['lag_mapping.json', 'xgboost_config.json']
        artifacts += ['source/' + name for name in manifest['source_hashes']]
        artifacts += ['validation/' + name for name in ('metrics.json', 'per_node.json', 'coverage.json', 'predictions.npz', 'forecast.png', 'evaluation.json')]
        immutable_json(output / 'fit_completed.json', {'manifest_sha256': sha256(output / 'manifest.json'),
                       'test_evaluated': False, 'artifacts': {name: sha256(output / name) for name in artifacts}})
        LOG.info('Completed %s: validation RMSE %.4f W, %.2fs; test not evaluated', output.name, metrics['rmse_w'], resources.result['wall_seconds'])
        print(json.dumps({'output': str(output), 'selection': selection, 'resources': resources.result}, indent=2))
    except Exception as error:
        immutable_json(output / 'failed.json', {'error': str(error), 'test_evaluated': False})
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def report(output):
    manifest, selection, data = check_provenance(output)
    if manifest['config']['kind'] == 'pilot':
        raise ValueError('Pilot runs cannot evaluate the held-out test split')
    # Exclusive claim prevents duplicate reports and retains failed reports for diagnosis.
    claim = output / 'test_report_claim.json'
    immutable_json(claim, {'selected_fit_sha256': sha256(output / 'fit_completed.json'),
                          'created_utc': datetime.now(timezone.utc).isoformat()})
    prediction = predict_selected(data, manifest['config'], output, selection, 'test')
    _, _, origins, nodes = data.split('test')
    metrics = evaluate(data, 'test', prediction, nodes, origins, output / 'test')
    print(json.dumps(metrics, indent=2))
    return metrics


def plan_entries(plan_path, root):
    plan = read_json(plan_path)
    if set(plan) != {'version', 'fits'} or plan['version'] != 1 or len(plan['fits']) != 18:
        raise ValueError('Expected frozen 18-fit matrix')
    entries = []
    factors = set()
    for item in plan['fits']:
        if set(item) != {'id', 'config'} or Path(item['id']).name != item['id']:
            raise ValueError('Invalid matrix entry')
        path = resolve(item['config'])
        config = read_json(path)
        validate_config(config)
        if config['kind'] != 'comparison':
            raise ValueError('Pilots cannot enter the comparison matrix')
        factors.add((config['model'], config['weather'], config['seed']))
        entries.append((path, config, root / item['id']))
    expected = {(model, weather, seed) for model in ('lstm', 'gru', 'xgboost')
                for weather in ('enabled', 'masked') for seed in (42, 43, 44)}
    if factors != expected or len({entry[2] for entry in entries}) != 18:
        raise ValueError('Matrix must have exactly one fit per factor combination')
    return entries


def matrix(plan_path, data_path, output, stage):
    entries = plan_entries(plan_path, output)
    FrozenData(data_path).verify()
    if output.exists() and any(output.iterdir()) and not (output / 'matrix_manifest.json').is_file():
        raise ValueError('Matrix requires a new directory or its own existing immutable declaration')
    output.mkdir(parents=True, exist_ok=True)
    declaration = output / 'matrix_manifest.json'
    expected = {'plan': read_json(plan_path), 'plan_sha256': sha256(plan_path),
                'configs': {str(path.relative_to(ROOT)): sha256(path) for path, _, _ in entries},
                'frozen_contract_sha256': FrozenData(data_path).contract_sha256, 'source_hashes': source_hashes()}
    if declaration.exists():
        if read_json(declaration) != expected:
            raise ValueError('Matrix declaration changed')
    else:
        if stage != 'fit':
            raise ValueError('Fit the matrix before reporting test results')
        immutable_json(declaration, expected)
    if stage == 'fit':
        for path, config, run in entries:
            if run.exists():
                manifest, _, data = check_provenance(run)
                if manifest['config'] != config or data.contract_sha256 != expected['frozen_contract_sha256']:
                    raise ValueError('Completed run does not match this matrix')
                LOG.info('Skipping verified completed fit %s', run.name)
                continue
            # Fresh process gives each fit an independent resource high-water mark.
            subprocess.run([sys.executable, str(ROOT / 'experiments.py'), 'fit', '--config', str(path),
                            '--data-dir', str(data_path), '--output-dir', str(run)], check=True)
    else:
        # Lock all 18 validation selections before evaluating any held-out labels.
        for _, config, run in entries:
            manifest, _, data = check_provenance(run)
            if manifest['config'] != config or data.contract_sha256 != expected['frozen_contract_sha256']:
                raise ValueError('Matrix member has incompatible provenance')
        ledger = output / 'test_selection_lock.json'
        lock = {'matrix_manifest_sha256': sha256(declaration),
                'selected_fits': {run.name: sha256(run / 'fit_completed.json') for _, _, run in entries}}
        if ledger.exists():
            if read_json(ledger) != lock:
                raise ValueError('Test selection lock changed')
        else:
            immutable_json(ledger, lock)
        rows = []
        for _, config, run in entries:
            if (run / 'test/evaluation.json').exists():
                receipt = read_json(run / 'test/evaluation.json')
                for name, checksum in receipt['artifacts'].items():
                    if sha256(run / 'test' / name) != checksum:
                        raise ValueError('Previously reported test artifacts changed')
                metrics = read_json(run / 'test/metrics.json')
            else:
                report(run)
                metrics = read_json(run / 'test/metrics.json')
            rows.append({'id': run.name, 'model': config['model'], 'weather': config['weather'], 'seed': config['seed'],
                         **metrics, 'selection': read_json(run / 'selection.json'), 'resources': read_json(run / 'resources.json')})
        immutable_json(output / 'comparison.json', {'fits': rows,
            'paired_weather_rmse_difference_w': [{'model': model, 'seed': seed, 'masked_minus_enabled_w':
                next(r['rmse_w'] for r in rows if (r['model'], r['weather'], r['seed']) == (model, 'masked', seed)) -
                next(r['rmse_w'] for r in rows if (r['model'], r['weather'], r['seed']) == (model, 'enabled', seed))}
                for model in ('lstm', 'gru', 'xgboost') for seed in (42, 43, 44)],
            'between_seed': [{'model': model, 'weather': weather,
                'mean_rmse_w': float(np.mean([r['rmse_w'] for r in rows if r['model'] == model and r['weather'] == weather])),
                'std_rmse_w': float(np.std([r['rmse_w'] for r in rows if r['model'] == model and r['weather'] == weather], ddof=1))}
                for model in ('lstm', 'gru', 'xgboost') for weather in ('enabled', 'masked')]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze')
    p.add_argument('--source-run', default='output/m100_all_nodes')
    p.add_argument('--output-dir', default='output/experiments/v1/frozen')
    p = commands.add_parser('fit')
    p.add_argument('--config', required=True)
    p.add_argument('--data-dir', default='output/experiments/v1/frozen')
    p.add_argument('--output-dir', required=True)
    p = commands.add_parser('report')
    p.add_argument('--output-dir', required=True)
    p = commands.add_parser('matrix')
    p.add_argument('--plan', default='configs/experiments/v1/matrix.json')
    p.add_argument('--data-dir', default='output/experiments/v1/frozen')
    p.add_argument('--output-dir', default='output/experiments/v1/comparison')
    p.add_argument('--stage', choices=('fit', 'report'), required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        if args.command == 'freeze':
            freeze(resolve(args.source_run), resolve(args.output_dir))
            LOG.info('Frozen reviewed data at %s', resolve(args.output_dir))
        elif args.command == 'fit':
            fit(resolve(args.config), resolve(args.data_dir), resolve(args.output_dir))
        elif args.command == 'report':
            report(resolve(args.output_dir))
        else:
            matrix(resolve(args.plan), resolve(args.data_dir), resolve(args.output_dir), args.stage)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f'EXPERIMENT ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
