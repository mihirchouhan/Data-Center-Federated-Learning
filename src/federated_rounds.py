"""Deterministic single-process FedAvg GRU orchestration; training/validation only.

No test capability is accepted. A failed round is replayed in full from its
previous immutable global state. Sampling/shuffle streams are indexed by round,
so interruption cannot consume or change future participation.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from .experiment_data import sha256, weather_view
from .experiment_eval import metric_pair
from .experiment_resources import Resources
from .federated_data import (ClientTrainDataset, TrainingClients, CoordinatorValidationData,
                             load_training_clients, load_coordinator_validation)
from .federated_io import (digest_json, make_manifest, open_run, commit_round, load_commits,
                           write_event, export_selected, load_selected)
from .federated_train import GlobalModelState, aggregate_fedavg
from .io_utils import RunLock
from .model import SequenceRegressor


DEFAULTS = {'version': 2, 'kind': 'main', 'model': 'gru', 'weather': 'masked', 'seed': 42,
            'device': 'cuda:0', 'num_threads': 1, 'rounds': 200, 'clients_per_round': 98,
            'local_epochs': 1, 'batch_size': 512, 'drop_last': False, 'optimizer': 'sgd',
            'learning_rate': .01, 'momentum': 0., 'weight_decay': 0., 'validation_every': 10,
            'eval_batch_size': 1024, 'hidden_size': 64, 'num_layers': 2, 'node_embedding_size': 8,
            'minimum_windows': 1024, 'minimum_days': 7,
            'frozen_contract_sha256': '24b2224ea9a7e1f255faf6357ffe7a3536fcbcfd90ab8b73179798349736b722'}


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError(f'Configuration fields differ: {sorted(set(config) ^ set(DEFAULTS))}')
    for key in ('seed', 'num_threads', 'rounds', 'clients_per_round', 'local_epochs', 'batch_size',
                'validation_every', 'eval_batch_size', 'hidden_size', 'num_layers', 'node_embedding_size',
                'minimum_windows', 'minimum_days'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'seed' else 1):
            raise ValueError(f'{key}: invalid integer')
    if config['seed'] >= 2**32 or config['rounds'] > 200 or config['kind'] not in ('main', 'smoke'):
        raise ValueError('Invalid seed, kind or maximum rounds')
    for key in ('version', 'model', 'weather', 'local_epochs', 'batch_size', 'drop_last', 'optimizer',
                'learning_rate', 'momentum', 'weight_decay', 'hidden_size', 'num_layers', 'node_embedding_size'):
        if config[key] != DEFAULTS[key] or (key == 'drop_last' and config[key] is not False):
            raise ValueError(f'{key}: differs from the approved fixed protocol')
    if config['kind'] == 'main':
        for key in ('rounds', 'clients_per_round', 'validation_every', 'minimum_windows', 'minimum_days',
                    'frozen_contract_sha256'):
            if config[key] != DEFAULTS[key]:
                raise ValueError(f'{key}: main protocol differs from approval')
    elif config['rounds'] > 3 or config['clients_per_round'] > 3:
        raise ValueError('Smoke pilot is limited to three rounds and three clients per round')
    anchor = config['frozen_contract_sha256']
    if not isinstance(anchor, str) or len(anchor) != 64 or any(c not in '0123456789abcdef' for c in anchor):
        raise ValueError('An explicit frozen SHA-256 anchor is required')
    device = config['device']
    if not isinstance(device, str) or (device != 'cpu' and not device.startswith('cuda:')):
        raise ValueError('Device must be cpu or an explicit cuda index')
    torch.device(device)
    return dict(config)


def stream_seed(seed, domain, round_id=0, node_index=0):
    return int(np.random.SeedSequence([seed, domain, round_id, node_index]).generate_state(1)[0])


def sample_clients(client_ids, count, seed, round_id):
    if round_id < 1 or count < 1 or count > len(client_ids) or len(set(client_ids)) != len(client_ids):
        raise ValueError('Invalid client population, sample size or round')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1, round_id, 0]))
    positions = rng.choice(len(client_ids), size=count, replace=False)
    return tuple(client_ids[int(i)] for i in positions)


def validation_due(round_id, config):
    return round_id == 0 or round_id == config['rounds'] or round_id % config['validation_every'] == 0


def configure_runtime(config):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('Deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG')
    torch.set_num_threads(config['num_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(config['device'])
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; CPU fallback is forbidden')
        torch.cuda.set_device(device)
    return device


def initialize_model(clients, config):
    # All initialization is on CPU. fork_rng avoids consuming the caller's stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream_seed(config['seed'], 0))
        return SequenceRegressor(14, 1, hidden_size=config['hidden_size'], num_layers=config['num_layers'],
                                 model='gru', node_count=len(clients.node_names),
                                 node_embedding_size=config['node_embedding_size'],
                                 baseline_coefficients=clients.baseline_coefficients)


def train_client(dataset, snapshot, template, config, device, *, round_id):
    """Worker accepts ONLY copied client-local training data and a round snapshot."""
    if not isinstance(dataset, ClientTrainDataset) or dataset.weather != 'masked':
        raise ValueError('Worker requires the restricted weather-masked training capability')
    if not isinstance(snapshot, GlobalModelState) or snapshot.round_id != round_id - 1:
        raise ValueError('Client initialization uses the wrong global round')
    started = time.perf_counter()
    model = snapshot.initialize_client(template).to(device)
    # No optimizer or optimizer state survives across client/round sessions.
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0., weight_decay=0.)
    if optimizer.state:
        raise ValueError('Local optimizer did not start empty')
    shuffle_seed = stream_seed(config['seed'], 2, round_id, dataset.node_index)
    positions = np.random.default_rng(shuffle_seed).permutation(len(dataset))
    examples = steps = 0
    loss_sum = 0.
    for first in range(0, len(positions), config['batch_size']):
        batch = dataset.batch(positions[first:first + config['batch_size']])
        inputs = torch.from_numpy(batch.inputs).to(device)
        targets = torch.from_numpy(batch.targets).to(device)
        optimizer.zero_grad(set_to_none=True)
        predicted = model(inputs)
        loss = torch.nn.functional.mse_loss(predicted, targets)
        if not torch.isfinite(predicted).all().item() or not torch.isfinite(loss).item():
            raise ValueError(f'{dataset.client_id}: nonfinite local prediction/loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError(f'{dataset.client_id}: nonfinite local gradient')
        optimizer.step()
        examples += len(batch.targets); steps += 1
        loss_sum += loss.item() * len(batch.targets)
    if examples != len(dataset) or steps != math.ceil(len(dataset) / config['batch_size']):
        raise ValueError('Local epoch did not process every training window exactly once')
    update = snapshot.capture_update(dataset.client_id, model, len(dataset))
    return update, {'client_id': dataset.client_id, 'training_windows': len(dataset),
                    'examples_processed': examples, 'optimizer_steps': steps, 'shuffle_seed': shuffle_seed,
                    'initial_fingerprint': snapshot.fingerprint, 'optimizer_initial_state_entries': 0,
                    'mean_training_mse_scaled': loss_sum / examples,
                    'seconds': time.perf_counter() - started}


def predict_validation(model, validation, config, device):
    if not isinstance(validation, CoordinatorValidationData):
        raise ValueError('Coordinator validation capability required')
    x, _, _, _ = validation.split('val')
    chunks = []
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(x), config['eval_batch_size']):
            batch = weather_view(np.array(x[first:first + config['eval_batch_size']], copy=True), 'masked')
            prediction = model(torch.from_numpy(batch).to(device)).detach().cpu().numpy().reshape(-1)
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite validation prediction')
            chunks.append(prediction)
    return np.concatenate(chunks)


def validation_metrics(prediction, validation):
    """Same W conversion and metric_pair as the v1 common evaluator; no reports."""
    x, y, _, nodes = validation.split('val')
    prediction = np.asarray(prediction)
    if prediction.shape != (len(y),) or not np.isfinite(prediction).all():
        raise ValueError('Validation predictions must cover the entire cohort in frozen order')
    mean, std = validation.meta['y_mean'][0], validation.meta['y_std'][0]
    actual = (np.asarray(y[:, 0], dtype=np.float64) * std + mean) * 1000
    predicted = (prediction.astype(np.float64) * std + mean) * 1000
    persistence = (np.asarray(x[:, -1, 0], dtype=np.float64) * validation.meta['x_std'][0]
                   + validation.meta['x_mean'][0]) * 1000
    result = metric_pair(predicted - actual, persistence - actual)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite validation metric')
    return {**result, 'unit': 'W', 'split': 'val', 'evaluated_nodes': len(np.unique(nodes)),
            'predictions_sha256': hashlib.sha256(prediction.tobytes()).hexdigest(), 'finite_predictions': True}


def _validate_payload(payload, receipts, model, config, clients):
    last = receipts[-1]
    if payload['round'] > config['rounds'] or payload['round'] != last['round']:
        raise ValueError('Invalid resumed round cursor')
    base = GlobalModelState(model, round_id=payload['round'])
    checked = base._validate(payload['state'], label='resumed state')
    base._validate(payload['best_state'], label='resumed selected state')
    model.load_state_dict(checked, strict=True)
    fingerprint = GlobalModelState(model, round_id=payload['round']).fingerprint
    if fingerprint != last['global_fingerprint']:
        raise ValueError('Resumed global fingerprint mismatch')
    selected = [r for r in receipts if r['validation'] is not None]
    best = min(selected, key=lambda r: (r['validation']['rmse_w'], r['round']))
    if payload['best_round'] != best['round'] or payload['best_rmse_w'] != best['validation']['rmse_w']:
        raise ValueError('Resumed validation selection mismatch')
    model.load_state_dict(payload['best_state'], strict=True)
    if GlobalModelState(model, round_id=best['round']).fingerprint != best['global_fingerprint']:
        raise ValueError('Resumed selected state differs from its validation checkpoint')
    model.load_state_dict(checked, strict=True)
    steps = examples = 0
    for record in receipts:
        i = record['round']
        expected = list(sample_clients(clients.client_ids, config['clients_per_round'], config['seed'], i)) if i else []
        if record['selected_clients'] != expected or (record['validation'] is not None) != validation_due(i, config):
            raise ValueError('Resumed sampling/validation schedule mismatch')
        if [entry['client_id'] for entry in record['clients']] != expected:
            raise ValueError('Resumed client records mismatch')
        for entry in record['clients']:
            count = clients.training_counts[entry['client_id']]
            if entry['training_windows'] != count or entry['examples_processed'] != count or entry['optimizer_steps'] != math.ceil(count / config['batch_size']):
                raise ValueError('Resumed training counts mismatch')
            examples += count; steps += entry['optimizer_steps']
    if payload['examples_processed'] != examples or payload['optimizer_steps'] != steps:
        raise ValueError('Resumed cumulative training counters mismatch')


def run_rounds(clients, validation, config, output, *, resume=False, stop_after_round=None):
    config = validate_config(config)
    if not isinstance(clients, TrainingClients) or clients.weather != 'masked' or not isinstance(validation, CoordinatorValidationData):
        raise ValueError('Restricted training registry and separate coordinator validation are required')
    if clients.contract_sha256 != config['frozen_contract_sha256']:
        raise ValueError('Training contract differs from configuration anchor')
    if config['clients_per_round'] > len(clients.client_ids):
        raise ValueError('Sampling exceeds the client population')
    if config['kind'] == 'main' and (len(clients.client_ids) != 979 or clients.num_windows != 4506133):
        raise ValueError('Main run must preserve all 979 clients and 4,506,133 training windows')
    if stop_after_round is not None and (isinstance(stop_after_round, bool) or not isinstance(stop_after_round, int)
                                        or not 0 <= stop_after_round <= config['rounds']):
        raise ValueError('Invalid stop-after-round cursor')
    device = configure_runtime(config)
    manifest = make_manifest(config, clients, validation, device)
    output = open_run(output, manifest, resume=resume)
    with RunLock(output / '.run.lock'):
        model = initialize_model(clients, config)
        payload, receipts = load_commits(output)
        if payload is not None:
            _validate_payload(payload, receipts, model, config, clients)
        elif receipts:
            raise ValueError('Missing committed model state')
        started_round = payload['round'] if payload is not None else -1
        attempted_round = None
        resources = Resources()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        try:
            with resources:
                model.to(device)
                if payload is None:
                    started = time.perf_counter()
                    initial = GlobalModelState(model, round_id=0)
                    metrics = validation_metrics(predict_validation(model, validation, config, device), validation)
                    payload = {'round': 0, 'state': initial.state_dict(), 'best_state': initial.state_dict(),
                               'best_round': 0, 'best_rmse_w': metrics['rmse_w'], 'optimizer_steps': 0,
                               'examples_processed': 0, 'manifest_digest': digest_json(manifest)}
                    record = {'round': 0, 'candidate': 'initial persistence', 'selected_clients': [], 'clients': [],
                              'validation': metrics, 'seconds': time.perf_counter() - started,
                              'global_fingerprint': initial.fingerprint, 'previous_receipt_sha256': None}
                    receipts.append(commit_round(output, payload, record))
                    print(f"round=0 validation_rmse_w={metrics['rmse_w']:.8f} candidate=persistence", flush=True)
                target = config['rounds'] if stop_after_round is None else stop_after_round
                if target < payload['round']:
                    raise ValueError('Stop cursor precedes the committed round')
                for round_id in range(payload['round'] + 1, target + 1):
                    attempted_round = round_id
                    started = time.perf_counter()
                    snapshot = GlobalModelState(model, round_id=round_id - 1)
                    selected = sample_clients(clients.client_ids, config['clients_per_round'], config['seed'], round_id)
                    updates, records = [], []
                    for client_id in selected:
                        update, record = train_client(clients.client(client_id), snapshot, model, config, device, round_id=round_id)
                        count = clients.training_counts[client_id]
                        if (update.client_id != client_id or record['client_id'] != client_id
                                or update.num_examples != count or record['training_windows'] != count
                                or record['examples_processed'] != count or record['optimizer_steps'] != math.ceil(count / config['batch_size'])
                                or record['initial_fingerprint'] != snapshot.fingerprint):
                            raise ValueError('Client participation, counts or initialization mismatch; round aborted')
                        updates.append(update); records.append(record)
                    # Every selected update is mandatory; no retries with changed participants.
                    averaged = aggregate_fedavg(snapshot, updates, expected_counts=clients.training_counts)
                    model.load_state_dict(averaged, strict=True)
                    new_global = GlobalModelState(model, round_id=round_id)
                    metrics = validation_metrics(predict_validation(model, validation, config, device), validation) if validation_due(round_id, config) else None
                    better = metrics is not None and metrics['rmse_w'] < payload['best_rmse_w']
                    payload = {'round': round_id, 'state': new_global.state_dict(),
                               'best_state': new_global.state_dict() if better else payload['best_state'],
                               'best_round': round_id if better else payload['best_round'],
                               'best_rmse_w': metrics['rmse_w'] if better else payload['best_rmse_w'],
                               'optimizer_steps': payload['optimizer_steps'] + sum(r['optimizer_steps'] for r in records),
                               'examples_processed': payload['examples_processed'] + sum(r['examples_processed'] for r in records),
                               'manifest_digest': digest_json(manifest)}
                    record = {'round': round_id, 'selected_clients': list(selected), 'clients': records,
                              'validation': metrics, 'seconds': time.perf_counter() - started,
                              'global_fingerprint': new_global.fingerprint,
                              'previous_receipt_sha256': sha256(output / 'rounds' / f'{round_id-1:04d}' / 'receipt.json')}
                    receipts.append(commit_round(output, payload, record))
                    print(f"round={round_id} clients={list(selected)} examples={sum(r['examples_processed'] for r in records)} steps={sum(r['optimizer_steps'] for r in records)} validation={metrics} selected_round={payload['best_round']}", flush=True)
                    attempted_round = None
                export_check = None
                if payload['round'] == config['rounds']:
                    export_selected(output, payload)
                    exported = load_selected(output)
                    # Compare independently reloaded selected models on the complete validation cohort.
                    selected_model = initialize_model(clients, config).to(device)
                    GlobalModelState(model)._validate(exported['state'], label='selected export')
                    selected_model.load_state_dict(exported['state'], strict=True)
                    prediction = predict_validation(selected_model, validation, config, device)
                    reloaded = initialize_model(clients, config).to(device)
                    reloaded.load_state_dict(payload['best_state'], strict=True)
                    reference = predict_validation(reloaded, validation, config, device)
                    if not np.array_equal(prediction, reference):
                        raise ValueError('Export/reload validation predictions differ')
                    export_metrics = validation_metrics(prediction, validation)
                    if export_metrics['rmse_w'] != payload['best_rmse_w']:
                        raise ValueError('Reloaded selected validation score differs')
                    export_check = {'exact_predictions_equal': True, 'finite_parameters': True,
                                    'validation': export_metrics}
            summary = {'status': 'complete' if payload['round'] == config['rounds'] else 'stopped',
                       'started_from_round': started_round, 'completed_round': payload['round'],
                       'selected_round': payload['best_round'], 'selected_validation_rmse_w': payload['best_rmse_w'],
                       'examples_processed': payload['examples_processed'], 'optimizer_steps': payload['optimizer_steps'],
                       'resources': resources.result, 'export_reload_check': export_check,
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                       'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None}
            event = write_event(output, summary)
            return {**summary, 'event_file': str(event)}
        except BaseException as error:
            # Disk commits are authoritative even if a signal arrived just after rename.
            write_event(output, {'status': 'aborted', 'attempted_round': attempted_round,
                                 'error': f'{type(error).__name__}: {error}',
                                 'resources': getattr(resources, 'result', None)})
            raise


def run_from_paths(config, data_dir, output_dir, *, resume=False, stop_after_round=None):
    config = validate_config(config)
    clients = load_training_clients(data_dir, expected_contract_sha256=config['frozen_contract_sha256'],
                                    weather='masked', minimum_windows=config['minimum_windows'],
                                    minimum_days=config['minimum_days'])
    validation = load_coordinator_validation(data_dir, expected_contract_sha256=config['frozen_contract_sha256'])
    return run_rounds(clients, validation, config, output_dir, resume=resume, stop_after_round=stop_after_round)
