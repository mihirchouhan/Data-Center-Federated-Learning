"""Exclusive, hash-bound round commits for the additive v2 FedAvg experiment.

A round directory becomes visible only after its checkpoint and receipt are
fsynced. Hidden interrupted writes are retained for diagnosis and never resumed.
This is crash consistency and provenance checking, not adversarial security.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import torch

from .experiment_data import immutable_json, read_json, sha256
from .io_utils import sync_directory


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    names = ['federated.py', 'src/federated_io.py', 'src/federated_rounds.py',
             'src/federated_data.py', 'src/federated_train.py', 'src/model.py',
             'src/experiment_data.py', 'src/experiment_eval.py', 'src/experiment_resources.py',
             'src/forecast_plot.py', 'src/io_utils.py', 'src/__init__.py']
    return {name: sha256(root / name) for name in names}


def environment(device):
    cuda = str(device).startswith('cuda')
    return {'python': platform.python_version(), 'dependencies': {
        d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if cuda else platform.machine(),
        'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if cuda else None,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'num_threads': torch.get_num_threads(), 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'allow_tf32': False}


def make_manifest(config, clients, validation, device):
    partitions = [{'id': p.client_id, 'node_index': p.node_index, 'count': p.num_examples,
                   'days': p.distinct_training_days,
                   'row_ids_sha256': hashlib.sha256(p.row_indices.tobytes()).hexdigest()}
                  for p in clients.partitions]
    if clients.contract_sha256 != validation.contract_sha256 or clients.node_names != validation.names:
        raise ValueError('Training and validation do not share the same frozen contract and node mapping')
    return {'version': 2, 'algorithm': 'FedAvg-GRU', 'config': config, 'config_digest': digest_json(config),
            'frozen_contract_sha256': clients.contract_sha256,
            'training_windows': clients.num_windows, 'clients': partitions,
            'validation_windows': len(validation.split('val')[1]),
            'source_hashes': source_inventory(), 'environment': environment(device),
            'rng_policy': 'NumPy SeedSequence([seed, domain, round, node_index]); domains 0=initialization, 1=sampling, 2=shuffle',
            'selection': 'earliest strict minimum pooled validation RMSE in W, including round-zero persistence',
            'embedding': 'standard sample-weighted FedAvg over the complete shared embedding table',
            'privacy': 'single-process simulation; no secure aggregation, differential privacy or OS isolation'}


def open_run(output, manifest, *, resume):
    output = Path(output)
    if resume:
        if not output.is_dir() or read_json(output / 'manifest.json') != manifest:
            raise ValueError('Resume manifest differs: config, source, data, environment or partitions changed')
        for name, expected in manifest['source_hashes'].items():
            if sha256(output / 'source' / name) != expected:
                raise ValueError(f'Recorded execution source snapshot differs: {name}')
        if read_json(output / 'configuration.json') != manifest['config']:
            raise ValueError('Recorded configuration snapshot differs')
    else:
        output.mkdir(parents=True, exist_ok=False)
        immutable_json(output / 'manifest.json', manifest)
        immutable_json(output / 'configuration.json', manifest['config'])
        root = Path(__file__).resolve().parents[1]
        for name, expected in manifest['source_hashes'].items():
            incoming = (root / name).read_bytes()
            if hashlib.sha256(incoming).hexdigest() != expected:
                raise ValueError(f'Execution source changed during initialization: {name}')
            destination = output / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(incoming); handle.flush(); os.fsync(handle.fileno())
            destination.chmod(0o444)
            sync_directory(destination.parent)
        (output / 'rounds').mkdir()
        sync_directory(output)
    return output


def commit_round(output, payload, record):
    """Atomic directory commit. Never overwrite any completed round."""
    output = Path(output)
    parent = output / 'rounds'
    round_id = record['round']
    destination = parent / f'{round_id:04d}'
    if destination.exists():
        raise FileExistsError(f'Round already committed: {destination}')
    temporary = Path(tempfile.mkdtemp(prefix=f'.pending-{round_id:04d}-', dir=parent))
    # Retain a failed temporary directory. Only four-digit directories are commits.
    checkpoint = temporary / 'state.pt'
    with checkpoint.open('xb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    checkpoint.chmod(0o444)
    receipt = {**record, 'checkpoint_sha256': sha256(checkpoint),
               'manifest_sha256': sha256(output / 'manifest.json')}
    immutable_json(temporary / 'receipt.json', receipt)
    sync_directory(temporary)
    os.rename(temporary, destination)
    sync_directory(parent)
    return receipt


def load_commits(output):
    """Verify the entire committed chain; only then load the final safe checkpoint."""
    output = Path(output)
    parent = output / 'rounds'
    receipts = []
    previous = None
    for directory in sorted(parent.iterdir()):
        if directory.name.startswith('.pending-'):
            continue
        if not directory.is_dir() or len(directory.name) != 4 or not directory.name.isdecimal():
            raise ValueError(f'Unrecognized round entry: {directory}')
        receipt = read_json(directory / 'receipt.json')
        index = len(receipts)
        if directory.name != f'{index:04d}' or receipt['round'] != index:
            raise ValueError('Round sequence is not contiguous from zero')
        if receipt['manifest_sha256'] != sha256(output / 'manifest.json'):
            raise ValueError('Checkpoint manifest hash mismatch')
        if receipt['previous_receipt_sha256'] != previous:
            raise ValueError('Round receipt chain mismatch')
        if receipt['checkpoint_sha256'] != sha256(directory / 'state.pt'):
            raise ValueError(f'Checkpoint hash mismatch at round {index}')
        receipts.append(receipt)
        previous = sha256(directory / 'receipt.json')
    if not receipts:
        return None, []
    payload = torch.load(parent / f'{len(receipts)-1:04d}' / 'state.pt', map_location='cpu', weights_only=True)
    if payload['round'] != receipts[-1]['round'] or payload['manifest_digest'] != digest_json(read_json(output / 'manifest.json')):
        raise ValueError('Checkpoint payload round/manifest mismatch')
    return payload, receipts


def write_event(output, value):
    """One exclusive event per invocation; errors never advance committed progress."""
    directory = Path(output) / 'events'
    directory.mkdir(exist_ok=True)
    import uuid
    path = directory / (uuid.uuid4().hex + '.json')
    immutable_json(path, value)
    sync_directory(directory)
    return path


def export_selected(output, payload):
    """Atomic, exclusive export directory with an independently verified receipt."""
    output = Path(output)
    final = output / 'selected'
    if final.exists():
        load_selected(output)
        return final
    temporary = Path(tempfile.mkdtemp(prefix='.pending-selected-', dir=output))
    path = temporary / 'model.pt'
    exported = {'state': payload['best_state'], 'selected_round': payload['best_round'],
                'validation_rmse_w': payload['best_rmse_w'], 'manifest_digest': payload['manifest_digest']}
    with path.open('xb') as handle:
        torch.save(exported, handle); handle.flush(); os.fsync(handle.fileno())
    path.chmod(0o444)
    immutable_json(temporary / 'receipt.json', {'model_sha256': sha256(path),
                   'selected_round': payload['best_round'], 'validation_rmse_w': payload['best_rmse_w'],
                   'manifest_sha256': sha256(output / 'manifest.json'),
                   'final_round_receipt_sha256': sha256(output / 'rounds' / f"{payload['round']:04d}" / 'receipt.json')})
    sync_directory(temporary)
    os.rename(temporary, final)
    sync_directory(output)
    return final


def load_selected(output):
    output = Path(output)
    receipt = read_json(output / 'selected/receipt.json')
    path = output / 'selected/model.pt'
    if receipt['model_sha256'] != sha256(path) or receipt['manifest_sha256'] != sha256(output / 'manifest.json'):
        raise ValueError('Selected export integrity mismatch')
    manifest = read_json(output / 'manifest.json')
    final_receipt = output / 'rounds' / f"{manifest['config']['rounds']:04d}" / 'receipt.json'
    if sha256(final_receipt) != receipt['final_round_receipt_sha256']:
        raise ValueError('Selected export final-round mismatch')
    exported = torch.load(path, map_location='cpu', weights_only=True)
    if (exported['manifest_digest'] != digest_json(manifest)
            or exported['selected_round'] != receipt['selected_round']
            or exported['validation_rmse_w'] != receipt['validation_rmse_w']):
        raise ValueError('Selected export metadata mismatch')
    return exported
