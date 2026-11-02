"""Immutable v5 round commits and exports. No training/inference operations."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from src.experiment_data import immutable_json, read_json, sha256
from src.history_data_v4 import inside
from src.io_utils import sync_directory


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def software():
    return {'python': platform.python_version(), 'platform': platform.platform(),
        'torch': str(torch.__version__), 'numpy': np.__version__, 'cuda_build': torch.version.cuda,
        'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


def sources(root, expected):
    for name, checksum in expected.items():
        if sha256(inside(root, name)) != checksum:
            raise ValueError('Execution source changed: ' + name)


def open_run(root, output, manifest, *, resume=False):
    output = Path(output)
    if resume:
        if not output.is_dir() or read_json(output / 'manifest.json') != manifest:
            raise ValueError('Resume manifest differs: data/source/configuration/software/device changed')
        if read_json(output / 'configuration.json') != manifest['plan']:
            raise ValueError('Recorded matrix configuration differs')
        for name, checksum in manifest['sources'].items():
            if sha256(inside(output / 'source', name)) != checksum:
                raise ValueError('Recorded source snapshot changed: ' + name)
    else:
        output.mkdir(parents=True, exist_ok=False)
        immutable_json(output / 'manifest.json', manifest)
        immutable_json(output / 'configuration.json', manifest['plan'])
        for name, checksum in manifest['sources'].items():
            source = inside(root, name)
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError('Source changed during snapshot')
            target = output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            target.chmod(0o444)
        (output / 'rounds').mkdir()
        sync_directory(output)
    return output


@contextmanager
def locked(output):
    # flock on a read-only manifest descriptor needs no mutable lock artifact.
    with (Path(output) / 'manifest.json').open('rb') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _save(path, value):
    with path.open('xb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def commit(output, payload, record, plans):
    """Atomically publish a full round; leave interrupted .pending directories."""
    parent = Path(output) / 'rounds'
    r = record['round']
    destination = parent / f'{r:04d}'
    if destination.exists():
        raise FileExistsError('Committed rounds are never overwritten')
    if r and not (parent / f'{r-1:04d}/receipt.json').is_file():
        raise ValueError('Cannot skip a communication round')
    temporary = Path(tempfile.mkdtemp(prefix=f'.pending-{r:04d}-', dir=parent))
    _save(temporary / 'state.pt', payload)
    offsets = np.r_[0, np.cumsum([len(p['rows']) for p in plans])].astype(np.int64)
    rows = np.concatenate([p['rows'] for p in plans]) if plans else np.array([], dtype=np.int64)
    with (temporary / 'sampled_rows.npz').open('xb') as handle:
        np.savez_compressed(handle, node_indices=np.array([p['node_index'] for p in plans], dtype=np.int64),
                            offsets=offsets, rows=rows)
        handle.flush()
        os.fsync(handle.fileno())
    (temporary / 'sampled_rows.npz').chmod(0o444)
    receipt = {**record, 'manifest_sha256': sha256(Path(output) / 'manifest.json'),
        'previous_receipt_sha256': sha256(parent / f'{r-1:04d}/receipt.json') if r else None,
        'state_sha256': sha256(temporary / 'state.pt'), 'sampled_rows_sha256': sha256(temporary / 'sampled_rows.npz')}
    immutable_json(temporary / 'receipt.json', receipt)
    sync_directory(temporary)
    os.rename(temporary, destination)
    sync_directory(parent)
    return receipt


def load_commits(output):
    output = Path(output)
    parent = output / 'rounds'
    paths = sorted(p for p in parent.iterdir() if p.is_dir() and len(p.name) == 4 and p.name.isdigit())
    records, previous = [], None
    for i, path in enumerate(paths):
        if int(path.name) != i:
            raise ValueError('Noncontiguous round ledger')
        record = read_json(path / 'receipt.json')
        if (record['round'] != i or record['previous_receipt_sha256'] != previous
                or record['manifest_sha256'] != sha256(output / 'manifest.json')
                or record['state_sha256'] != sha256(path / 'state.pt')
                or record['sampled_rows_sha256'] != sha256(path / 'sampled_rows.npz')):
            raise ValueError('Round receipt/checkpoint/plan integrity mismatch')
        previous = sha256(path / 'receipt.json')
        records.append(record)
    payload = torch.load(paths[-1] / 'state.pt', map_location='cpu', weights_only=True) if paths else None
    return payload, records


def check_trace(output, round_id, plans):
    with np.load(Path(output) / f'rounds/{round_id:04d}/sampled_rows.npz', allow_pickle=False) as saved:
        offsets = np.r_[0, np.cumsum([len(p['rows']) for p in plans])]
        rows = np.concatenate([p['rows'] for p in plans]) if plans else np.array([], dtype=np.int64)
        if (not np.array_equal(saved['node_indices'], [p['node_index'] for p in plans])
                or not np.array_equal(saved['offsets'], offsets) or not np.array_equal(saved['rows'], rows)):
            raise ValueError('Recorded rows differ from deterministic sampled-pass replay')


def export(output, payload, record):
    output = Path(output)
    target = output / 'selected'
    if target.exists():
        receipt = read_json(target / 'receipt.json')
        saved = torch.load(target / 'model.pt', map_location='cpu', weights_only=True)['model_state']
        if (receipt['model_sha256'] != sha256(target / 'model.pt') or receipt['selected_round'] != payload['best_round']
                or receipt['manifest_sha256'] != sha256(output / 'manifest.json')
                or any(receipt.get(k) != v for k, v in record.items())
                or set(saved) != set(payload['best_state'])
                or any(not torch.equal(saved[k], payload['best_state'][k]) for k in saved)):
            raise ValueError('Existing selected export differs')
        return receipt
    temporary = Path(tempfile.mkdtemp(prefix='.pending-selected-', dir=output))
    _save(temporary / 'model.pt', {'model_state': payload['best_state']})
    reloaded = torch.load(temporary / 'model.pt', map_location='cpu', weights_only=True)['model_state']
    if set(reloaded) != set(payload['best_state']) or any(not torch.equal(reloaded[k], payload['best_state'][k]) for k in reloaded):
        raise ValueError('Selected export/reload is not tensor-exact')
    receipt = {**record, 'selected_round': payload['best_round'], 'model_sha256': sha256(temporary / 'model.pt'),
               'completed_round': payload['round'], 'export_reload_tensor_exact': True,
               'manifest_sha256': sha256(output / 'manifest.json')}
    immutable_json(temporary / 'receipt.json', receipt)
    sync_directory(temporary)
    os.rename(temporary, target)
    sync_directory(output)
    return receipt
