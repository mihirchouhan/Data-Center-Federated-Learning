"""Exclusive v3 control artifacts; old experiment inventories are never edited.

Block commits and final exports use fsync followed by an atomic directory rename.
Interrupted .pending-* directories are retained. Hashes establish consistency,
not protection against a person able to rewrite every file and receipt.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import torch

from .experiment_data import immutable_json, read_json, sha256
from .federated_io import digest_json, environment, source_inventory
from .io_utils import sync_directory

ROOT = Path(__file__).resolve().parents[1]


def execution_sources():
    names = [*source_inventory(), 'src/centralized_sgd.py', 'src/centralized_io.py']
    return {name: sha256(ROOT / name) for name in names}


def software():
    # environment(cpu) records versions without initializing CUDA.
    value = environment('cpu')
    return {key: value[key] for key in ('python', 'dependencies', 'torch_cuda')}


def capture_inventory(root):
    """Read-only full byte inventory, excluding the environment and access times."""
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '.venv' not in p.relative_to(root).parts}


def check_inventory(root, baseline):
    failures = []
    for name, expected in baseline.items():
        path = Path(root) / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(name)
    return {'checked': len(baseline), 'passed': len(baseline) - len(failures),
            'failed': len(failures), 'mismatches': failures}


def open_run(output, manifest, *, resume):
    output = Path(output)
    if resume:
        recorded = read_json(output / 'manifest.json')
        if recorded != manifest or read_json(output / 'configuration.json') != manifest['config']:
            raise ValueError('Resume manifest/configuration differs')
        for name, expected in manifest['source_hashes'].items():
            if sha256(output / 'source' / name) != expected:
                raise ValueError(f'Source snapshot changed: {name}')
        if read_json(output / 'reference_lock.json') != manifest['reference']:
            raise ValueError('Reference lock changed')
        return output
    output.mkdir(parents=True, exist_ok=False)
    immutable_json(output / 'manifest.json', manifest)
    immutable_json(output / 'configuration.json', manifest['config'])
    immutable_json(output / 'reference_lock.json', manifest['reference'])
    for name, expected in manifest['source_hashes'].items():
        data = (ROOT / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Source changed while snapshotting: {name}')
        path = output / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        path.chmod(0o444)
        sync_directory(path.parent)
    (output / 'blocks').mkdir()
    sync_directory(output)
    return output


def save_tensor_file(path, value):
    with Path(path).open('xb') as handle:
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    Path(path).chmod(0o444)


def commit_block(output, payload, record):
    output = Path(output)
    parent = output / 'blocks'
    destination = parent / f"{record['block']:04d}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Block already committed: {destination}')
    pending = Path(tempfile.mkdtemp(prefix=f".pending-{record['block']:04d}-", dir=parent))
    save_tensor_file(pending / 'state.pt', payload)
    previous = output / 'blocks' / f"{record['block'] - 1:04d}" / 'receipt.json'
    receipt = {**record, 'manifest_sha256': sha256(output / 'manifest.json'),
               'checkpoint_sha256': sha256(pending / 'state.pt'),
               'previous_receipt_sha256': sha256(previous) if record['block'] else None}
    immutable_json(pending / 'receipt.json', receipt)
    sync_directory(pending)
    os.rename(pending, destination)
    sync_directory(parent)
    return receipt


def load_blocks(output):
    output = Path(output)
    manifest = read_json(output / 'manifest.json')
    expected_manifest = sha256(output / 'manifest.json')
    previous = None
    receipts = []
    for directory in sorted((output / 'blocks').iterdir()):
        if directory.name.startswith('.pending-'):
            continue
        if (not directory.is_dir() or directory.is_symlink()
                or directory.name != f'{len(receipts):04d}'):
            raise ValueError('Block chain must be contiguous from zero')
        receipt = read_json(directory / 'receipt.json')
        if (receipt['block'] != len(receipts)
                or receipt['manifest_sha256'] != expected_manifest
                or receipt['previous_receipt_sha256'] != previous
                or receipt['checkpoint_sha256'] != sha256(directory / 'state.pt')):
            raise ValueError(f'Block integrity mismatch: {directory.name}')
        receipts.append(receipt)
        previous = sha256(directory / 'receipt.json')
    if not receipts:
        return None, receipts
    value = torch.load(output / 'blocks' / f'{len(receipts)-1:04d}' / 'state.pt',
                       map_location='cpu', weights_only=True)
    if value['block'] != len(receipts)-1 or value['manifest_digest'] != digest_json(manifest):
        raise ValueError('Checkpoint cursor/manifest mismatch')
    return value, receipts


def verify_final(output):
    """Hash/metadata verification only: never predicts or evaluates any split."""
    output = Path(output)
    receipt = read_json(output / 'final/receipt.json')
    manifest = read_json(output / 'manifest.json')
    last = output / 'blocks' / f"{manifest['config']['rounds']:04d}" / 'receipt.json'
    if (receipt['manifest_sha256'] != sha256(output / 'manifest.json')
            or receipt['last_block_receipt_sha256'] != sha256(last)):
        raise ValueError('Final export manifest/checkpoint binding differs')
    for name, expected in receipt['artifacts'].items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or sha256(output / 'final' / p) != expected:
            raise ValueError(f'Final artifact changed: {name}')
    exported = torch.load(output / 'final/model.pt', map_location='cpu', weights_only=True)
    if (exported['selected_block'] != receipt['selected_block']
            or exported['validation_rmse_w'] != receipt['validation_rmse_w']
            or exported['manifest_digest'] != digest_json(manifest)):
        raise ValueError('Final export metadata differs')
    if (output / 'completed.json').exists():
        complete = read_json(output / 'completed.json')
        if (complete['status'] != 'complete'
                or complete['final_receipt_sha256'] != sha256(output / 'final/receipt.json')
                or complete['manifest_sha256'] != sha256(output / 'manifest.json')):
            raise ValueError('Completion receipt differs')
    return exported, receipt
