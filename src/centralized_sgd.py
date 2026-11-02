"""V3 centralized SGD with recorded FedAvg data/batch schedule replay.

Only the model trajectory changes: one model takes all sequential SGD steps.
There is no local model reset, averaging, test capability or Adam optimizer.
CLI runs are restricted to the original frozen M100 cohort and new v3 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import centralized_io as cio
from .experiment_data import immutable_json, read_json, sha256
from .experiment_eval import evaluate
from .experiment_resources import Resources
from .federated_data import (TrainingClients, CoordinatorValidationData,
                             load_training_clients, load_coordinator_validation)
from .federated_io import digest_json, environment, load_commits, load_selected, source_inventory, write_event
from .federated_rounds import (DEFAULTS as FED_DEFAULTS, configure_runtime, initialize_model,
                               predict_validation, sample_clients, stream_seed,
                               validate_config as validate_fed_config, validation_metrics)
from .federated_train import GlobalModelState
from .io_utils import RunLock, sync_directory

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {**FED_DEFAULTS, 'version': 3,
            'reference_run': 'output/experiments/v2_fedavg/main/gru_masked_seed42b',
            'reference_manifest_sha256': 'cda2eb5c5cade7f43792c84cf5491d2a703b1f36a16dec5d30c34e8f671bc221'}


def validate_config(config):
    if set(config) != set(DEFAULTS):
        raise ValueError('V3 configuration fields differ')
    for key in ('seed', 'num_threads', 'rounds', 'clients_per_round', 'local_epochs', 'batch_size',
                'validation_every', 'eval_batch_size', 'hidden_size', 'num_layers',
                'node_embedding_size', 'minimum_windows', 'minimum_days'):
        v = config[key]
        if type(v) is not int or v < (0 if key == 'seed' else 1):
            raise ValueError(f'{key}: invalid integer')
    if config['seed'] not in (42, 43, 44) or config['kind'] not in ('main', 'smoke'):
        raise ValueError('Expected approved seed and main/smoke kind')
    for key in ('version', 'model', 'weather', 'local_epochs', 'batch_size', 'optimizer',
                'learning_rate', 'momentum', 'weight_decay', 'hidden_size', 'num_layers',
                'node_embedding_size', 'num_threads', 'validation_every', 'eval_batch_size'):
        if type(config[key]) is not type(DEFAULTS[key]) or config[key] != DEFAULTS[key]:
            raise ValueError(f'{key}: differs from matched protocol')
    if config['drop_last'] is not False:
        raise ValueError('drop_last must be false')
    if config['kind'] == 'main':
        for key in ('rounds', 'clients_per_round', 'minimum_windows', 'minimum_days', 'frozen_contract_sha256'):
            if config[key] != DEFAULTS[key]:
                raise ValueError(f'{key}: main protocol differs')
    elif config['rounds'] > 3:
        raise ValueError('Smoke is limited to three schedule blocks')
    for key in ('frozen_contract_sha256', 'reference_manifest_sha256'):
        v = config[key]
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError(f'{key}: explicit SHA-256 required')
    if not isinstance(config['reference_run'], str):
        raise ValueError('Reference path must be a string')
    if not isinstance(config['device'], str) or (config['device'] != 'cpu' and not config['device'].startswith('cuda:')):
        raise ValueError('Explicit CPU/CUDA device required')
    torch.device(config['device'])
    return dict(config)


def due(block, config):
    return block == 0 or block == config['rounds'] or block % config['validation_every'] == 0


def choose(score, previous, block, selected):
    if not math.isfinite(score):
        raise ValueError('Nonfinite validation score')
    return (score, block) if score < previous else (previous, selected)


def partitions_record(clients):
    return [{'id': p.client_id, 'node_index': p.node_index, 'count': p.num_examples,
             'days': p.distinct_training_days, 'row_ids_sha256': hashlib.sha256(p.row_indices.tobytes()).hexdigest()}
            for p in clients.partitions]


def verify_reference(path, clients, config, initial):
    """Verify recorded sources, initialization, partitions, schedule and receipts.

    Historical per-batch IDs were not saved. They are reconstructed using the
    recorded shuffle seed, immutable row order and original NumPy procedure.
    This is not an independent historical per-batch trace.
    """
    path = Path(path)
    if sha256(path / 'manifest.json') != config['reference_manifest_sha256']:
        raise ValueError('Reference manifest hash differs')
    manifest = read_json(path / 'manifest.json')
    ref = validate_fed_config(manifest['config'])
    if manifest['algorithm'] != 'FedAvg-GRU' or manifest['config_digest'] != digest_json(ref):
        raise ValueError('Reference algorithm/configuration differs')
    for key in FED_DEFAULTS:
        if key not in ('version', 'kind', 'rounds') and ref[key] != config[key]:
            raise ValueError(f'Reference protocol differs: {key}')
    if ref['rounds'] < config['rounds'] or manifest['clients'] != partitions_record(clients):
        raise ValueError('Reference duration or frozen training partitions differ')
    if manifest['training_windows'] != clients.num_windows or manifest['frozen_contract_sha256'] != clients.contract_sha256:
        raise ValueError('Reference frozen training cohort differs')
    if read_json(path / 'configuration.json') != ref or manifest['source_hashes'] != source_inventory():
        raise ValueError('Reference configuration/execution sources differ')
    for name, expected in manifest['source_hashes'].items():
        if sha256(path / 'source' / name) != expected:
            raise ValueError(f'Reference source snapshot differs: {name}')
    for key, current in cio.software().items():
        if manifest['environment'][key] != current:
            raise ValueError(f'Reference software differs: {key}')
    final, receipts = load_commits(path)  # CPU load, hash-only validation of older checkpoints.
    if final is None or len(receipts) != ref['rounds'] + 1 or final['round'] != ref['rounds']:
        raise ValueError('Reference is not a complete contiguous run')
    zero = torch.load(path / 'rounds/0000/state.pt', map_location='cpu', weights_only=True)
    anchor = GlobalModelState(initial, round_id=0)
    anchor._validate(zero['state'], label='reference initialization')
    if (receipts[0]['global_fingerprint'] != anchor.fingerprint
            or any(not torch.equal(v, zero['state'][k]) for k, v in initial.state_dict().items())):
        raise ValueError('Fresh initialization differs from recorded round zero')
    examples = steps = 0
    for index, record in enumerate(receipts):
        expected = list(sample_clients(clients.client_ids, config['clients_per_round'], config['seed'], index)) if index else []
        if (record['round'] != index or record['selected_clients'] != expected
                or [e['client_id'] for e in record['clients']] != expected
                or (record['validation'] is not None) != (index == 0 or index == ref['rounds'] or index % ref['validation_every'] == 0)):
            raise ValueError('Reference participation/validation schedule differs')
        if record['validation'] is not None:
            metrics=record['validation']
            if (metrics['split']!='val' or metrics['unit']!='W'
                    or metrics['samples']!=manifest['validation_windows']
                    or metrics['evaluated_nodes']!=len(clients.node_names)
                    or metrics['finite_predictions'] is not True
                    or not all(math.isfinite(metrics[k]) and metrics[k]>=0 for k in ('rmse_w','mae_w','persistence_rmse_w','persistence_mae_w'))):
                raise ValueError('Reference validation cohort/metrics differ')
        for entry in record['clients']:
            p = clients.partitions[clients.client_ids.index(entry['client_id'])]
            count = p.num_examples
            if (entry['training_windows'] != count or entry['examples_processed'] != count
                    or entry['optimizer_steps'] != math.ceil(count / config['batch_size'])
                    or entry['shuffle_seed'] != stream_seed(config['seed'], 2, index, p.node_index)
                    or entry['optimizer_initial_state_entries'] != 0
                    or entry['initial_fingerprint'] != receipts[index-1]['global_fingerprint']):
                raise ValueError('Reference training counts/shuffle/initialization differ')
            examples += count; steps += entry['optimizer_steps']
    if final['examples_processed'] != examples or final['optimizer_steps'] != steps:
        raise ValueError('Reference cumulative training budget differs')
    best = min((r for r in receipts if r['validation'] is not None), key=lambda r: (r['validation']['rmse_w'], r['round']))
    selected = load_selected(path)
    if selected['selected_round'] != best['round'] or selected['validation_rmse_w'] != best['validation']['rmse_w']:
        raise ValueError('Reference validation selection differs')
    anchor._validate(final['state'],label='reference final state')
    anchor._validate(selected['state'],label='reference selected state')
    if (final['best_round']!=best['round'] or final['best_rmse_w']!=best['validation']['rmse_w']
            or any(not torch.equal(v,selected['state'][k]) for k,v in final['best_state'].items())):
        raise ValueError('Reference selected weights differ from final checkpoint')
    completed = [p for p in (path / 'events').glob('*.json') if read_json(p).get('status') == 'complete']
    if not completed:
        raise ValueError('Reference has no completion event')
    for event_path in completed:
        event=read_json(event_path)
        if (event['completed_round']!=ref['rounds'] or event['selected_round']!=best['round']
                or event['examples_processed']!=examples or event['optimizer_steps']!=steps
                or event['selected_validation_rmse_w']!=best['validation']['rmse_w']
                or event['export_reload_check']['exact_predictions_equal'] is not True
                or event['export_reload_check']['finite_parameters'] is not True
                or event['export_reload_check']['validation']!=best['validation']):
            raise ValueError('Reference completion event differs')
    lock = {'manifest_sha256': sha256(path / 'manifest.json'),
            'round_receipt_sha256': [sha256(path / 'rounds' / f'{i:04d}' / 'receipt.json') for i in range(len(receipts))],
            'selected_receipt_sha256': sha256(path / 'selected/receipt.json'),
            'completion_events': {p.name: sha256(p) for p in sorted(completed)},
            'initial_fingerprint': anchor.fingerprint,
            'schedule_policy': 'reconstruct original NumPy permutation; per-batch historical trace unavailable'}
    return receipts, lock


def plan_block(clients, config, reference, block):
    result = []
    for entry in reference[block]['clients']:
        p = clients.partitions[clients.client_ids.index(entry['client_id'])]
        positions = np.random.default_rng(entry['shuffle_seed']).permutation(p.num_examples)
        row_ids = p.row_indices[positions]
        sizes = [min(config['batch_size'], p.num_examples-i) for i in range(0, p.num_examples, config['batch_size'])]
        result.append({'client_id': p.client_id, 'shuffle_seed': entry['shuffle_seed'],
                       'training_windows': p.num_examples, 'examples_processed': p.num_examples,
                       'optimizer_steps': len(sizes), 'batch_sizes': sizes,
                       'row_order_sha256': hashlib.sha256(row_ids.tobytes()).hexdigest()})
    return result


def train_block(model, clients, plan, config, device, anchor, *, block):
    if not isinstance(clients, TrainingClients) or clients.weather != 'masked':
        raise ValueError('Training-only weather-masked registry required')
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0., weight_decay=0.)
    if optimizer.state:
        raise ValueError('SGD optimizer must be stateless')
    records = []
    for item in plan:
        started = time.perf_counter()
        dataset = clients.client(item['client_id'])
        order = np.random.default_rng(item['shuffle_seed']).permutation(len(dataset))
        observed = steps = 0
        loss_sum = 0.
        rows = hashlib.sha256()
        sizes = []
        initial = GlobalModelState(model, round_id=block-1).fingerprint
        for first in range(0, len(order), config['batch_size']):
            batch = dataset.batch(order[first:first+config['batch_size']])
            rows.update(batch.row_indices.tobytes()); sizes.append(len(batch.targets))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(batch.inputs).to(device))
            target = torch.from_numpy(batch.targets).to(device)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(prediction).all().item() or not torch.isfinite(loss).item():
                raise ValueError('Nonfinite centralized prediction/loss; block aborted')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
                raise ValueError('Nonfinite centralized gradient; block aborted')
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in model.parameters()):
                raise ValueError('Nonfinite centralized parameter; block aborted')
            observed += len(batch.targets); steps += 1
            loss_sum += loss.item()*len(batch.targets)
        anchor._validate(model.state_dict(), label='centralized update')
        actual = {**item, 'row_order_sha256': rows.hexdigest(), 'batch_sizes': sizes,
                  'examples_processed': observed, 'optimizer_steps': steps}
        if actual != item or optimizer.state:
            raise ValueError('Actual batches/counts differ from reference plan')
        records.append({**actual, 'initial_fingerprint': initial, 'optimizer_state_entries': 0,
                        'mean_training_mse_scaled': loss_sum/observed, 'seconds': time.perf_counter()-started})
    return records


def make_manifest(config, clients, validation, device, reference, *, stored_environment=None):
    if (not isinstance(clients, TrainingClients) or not isinstance(validation, CoordinatorValidationData)
            or clients.node_names != validation.names or clients.contract_sha256 != validation.contract_sha256):
        raise ValueError('Separate training/validation capabilities on identical frozen contract required')
    return {'version': 3, 'algorithm': 'Centralized-SGD-GRU schedule-replay', 'config': config,
            'config_digest': digest_json(config), 'training_windows': clients.num_windows,
            'validation_windows': len(validation.split('val')[1]), 'clients': partitions_record(clients),
            'frozen_contract_sha256': clients.contract_sha256, 'reference': reference,
            'source_hashes': cio.execution_sources(),
            'environment': stored_environment if stored_environment is not None else environment(device),
            'selection': 'earliest strict minimum pooled validation RMSE W; initial persistence included',
            'training': 'one continuing model; no resets or aggregation; identical node-grouped batch schedule',
            'rng': 'original SeedSequence domains; no stochastic layer or mutable shuffle cursor',
            'privacy': 'centralized simulation; no differential privacy or secure aggregation',
            'test_access': 'forbidden'}


def validate_payload(payload, receipts, model, clients, config, reference, anchor):
    if payload['block'] > config['rounds']:
        raise ValueError('Checkpoint exceeds configured budget')
    anchor._validate(payload['state'], label='resumed state')
    anchor._validate(payload['best_state'], label='resumed selection')
    examples = steps = 0
    for record in receipts:
        block = record['block']
        expected = plan_block(clients, config, reference, block) if block else []
        if (record['selected_clients'] != [p['client_id'] for p in expected]
                or (record['validation'] is not None) != due(block, config)
                or len(record['clients']) != len(expected)):
            raise ValueError('Checkpoint participation/validation schedule differs')
        for entry, item in zip(record['clients'], expected):
            if any(entry.get(k) != v for k, v in item.items()) or entry['optimizer_state_entries'] != 0:
                raise ValueError('Checkpoint batch schedule/counters differ')
            examples += entry['examples_processed']; steps += entry['optimizer_steps']
        if record['validation'] is not None:
            metrics = record['validation']
            if (metrics['split'] != 'val' or metrics['unit'] != 'W'
                    or metrics['samples'] != reference[0]['validation']['samples']
                    or metrics['evaluated_nodes'] != len(clients.node_names)
                    or metrics['finite_predictions'] is not True
                    or not all(math.isfinite(metrics[k]) and metrics[k]>=0 for k in ('rmse_w','mae_w','persistence_rmse_w','persistence_mae_w'))):
                raise ValueError('Invalid validation selection record')
    if payload['examples_processed'] != examples or payload['optimizer_steps'] != steps:
        raise ValueError('Checkpoint cumulative accounting differs')
    best = min((r for r in receipts if r['validation'] is not None), key=lambda r: (r['validation']['rmse_w'],r['block']))
    if payload['best_block'] != best['block'] or payload['best_rmse_w'] != best['validation']['rmse_w']:
        raise ValueError('Checkpoint selected candidate differs')
    for key, block, expected in [('state', payload['block'], receipts[-1]['global_fingerprint']),
                                  ('best_state', best['block'], best['global_fingerprint'])]:
        model.load_state_dict(payload[key], strict=True)
        if GlobalModelState(model, round_id=block).fingerprint != expected:
            raise ValueError('Checkpoint model fingerprint differs')
    model.load_state_dict(payload['state'], strict=True)


def verify_completion(output, payload):
    exported,receipt = cio.verify_final(output)
    if (exported['selected_block'] != payload['best_block']
            or exported['validation_rmse_w'] != payload['best_rmse_w']
            or any(not torch.equal(v,exported['state'][k]) for k,v in payload['best_state'].items())):
        raise ValueError('Final model differs from selected checkpoint')
    if (Path(output)/'completed.json').exists():
        complete = read_json(Path(output)/'completed.json')
        for a,b in [('completed_block','block'),('selected_block','best_block'),
                    ('validation_rmse_w','best_rmse_w'),('examples_processed','examples_processed'),
                    ('optimizer_steps','optimizer_steps')]:
            if complete[a] != payload[b]:raise ValueError('Completion accounting/selection differs')
        if complete['export_reload_exact'] is not True:raise ValueError('Missing export reload verification')
    return receipt


def finalize(output, payload, model, clients, validation, config, device):
    output = Path(output)
    if (output / 'final').exists():
        return verify_completion(output,payload)
    selected = initialize_model(clients, config).to(device)
    selected.load_state_dict(payload['best_state'], strict=True)
    prediction = predict_validation(selected, validation, config, device)
    pending = Path(tempfile.mkdtemp(prefix='.pending-final-', dir=output))
    exported = {'state': payload['best_state'], 'selected_block': payload['best_block'],
                'validation_rmse_w': payload['best_rmse_w'], 'manifest_digest': payload['manifest_digest']}
    cio.save_tensor_file(pending / 'model.pt', exported)
    reloaded = initialize_model(clients, config).to(device)
    reloaded.load_state_dict(torch.load(pending/'model.pt',map_location='cpu',weights_only=True)['state'], strict=True)
    repeated = predict_validation(reloaded, validation, config, device)
    if not np.array_equal(prediction, repeated):
        raise ValueError('Export/reload validation predictions differ')
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    _, _, origins, nodes = validation.split('val')
    metrics = evaluate(validation, 'val', repeated, nodes, origins, pending / 'validation')
    if metrics['rmse_w'] != payload['best_rmse_w']:
        raise ValueError('Common evaluator and selection differ')
    receipt = {'selected_block': payload['best_block'], 'validation_rmse_w': metrics['rmse_w'],
               'manifest_sha256': sha256(output/'manifest.json'),
               'last_block_receipt_sha256': sha256(output/'blocks'/f"{config['rounds']:04d}"/'receipt.json'),
               'export_reload_exact': True, 'finite_predictions': True,
               'predictions_sha256': hashlib.sha256(repeated.tobytes()).hexdigest(),
               'artifacts': {str(p.relative_to(pending)): sha256(p) for p in sorted(pending.rglob('*')) if p.is_file()}}
    immutable_json(pending/'receipt.json',receipt)
    sync_directory(pending); os.rename(pending,output/'final'); sync_directory(output)
    cio.verify_final(output)
    return receipt


def run_control(clients, validation, config, output, reference_path, *, resume=False,
                stop_after_block=None, verify_only=False):
    config = validate_config(config)
    output = Path(output)
    if not resume and not verify_only and (output.exists() or output.is_symlink()):
        raise FileExistsError(f'Output exists; preserve it: {output}')
    if stop_after_block is not None and (type(stop_after_block) is not int or not 0 <= stop_after_block <= config['rounds']):
        raise ValueError('Invalid stop-after-block cursor')
    if not isinstance(clients, TrainingClients) or clients.weather != 'masked' or not isinstance(validation, CoordinatorValidationData):
        raise ValueError('Restricted train and coordinator validation capabilities required')
    if clients.contract_sha256 != config['frozen_contract_sha256']:
        raise ValueError('Frozen contract differs')
    initial = initialize_model(clients, config)  # fresh; never load a trained initialization
    anchor = GlobalModelState(initial, round_id=0)
    reference, lock = verify_reference(reference_path,clients,config,initial)
    if len(validation.split('val')[1]) != read_json(Path(reference_path)/'manifest.json')['validation_windows']:
        raise ValueError('Reference validation cohort differs')
    if verify_only:
        stored = read_json(output/'manifest.json')['environment']
        for k,v in cio.software().items():
            if stored[k] != v:raise ValueError('Recorded software environment differs')
        manifest = make_manifest(config,clients,validation,'cpu',lock,stored_environment=stored)
        cio.open_run(output,manifest,resume=True)
        payload, receipts = cio.load_blocks(output)
        if payload is None:raise ValueError('No committed blocks')
        validate_payload(payload,receipts,initial,clients,config,reference,anchor)
        if (output/'final').exists():
            verify_completion(output,payload)
        complete = (output/'completed.json').exists()
        if complete and not (output/'final').is_dir():raise ValueError('Missing final export')
        if complete and payload['block'] != config['rounds']:raise ValueError('Premature completion receipt')
        return {'status':'verified_complete' if complete else 'verified_partial','completed_block':payload['block'],
                'selected_block':payload['best_block'],'examples_processed':payload['examples_processed'],
                'optimizer_steps':payload['optimizer_steps'],'model_inference_executed':False}
    device = configure_runtime(config)
    manifest = make_manifest(config,clients,validation,device,lock)
    cio.open_run(output,manifest,resume=resume)
    with RunLock(output/'.run.lock'):
        payload,receipts = cio.load_blocks(output)
        if payload is not None:validate_payload(payload,receipts,initial,clients,config,reference,anchor)
        if (output/'completed.json').exists():
            verify_completion(output,payload)
            return {'status':'already_complete','completed_block':payload['block'],'output':str(output)}
        model = initial.to(device)
        attempted = None
        resource = Resources()
        if device.type == 'cuda':torch.cuda.reset_peak_memory_stats(device)
        try:
            with resource:
                if payload is None:
                    score = validation_metrics(predict_validation(model,validation,config,device),validation)
                    payload = {'block':0,'state':anchor.state_dict(),'best_state':anchor.state_dict(),
                               'best_block':0,'best_rmse_w':score['rmse_w'],'examples_processed':0,'optimizer_steps':0,
                               'manifest_digest':digest_json(manifest)}
                    record = {'block':0,'selected_clients':[],'clients':[],'candidate':'initial persistence',
                              'global_fingerprint':anchor.fingerprint,'validation':score,'seconds':0.}
                    receipts.append(cio.commit_block(output,payload,record))
                target = config['rounds'] if stop_after_block is None else stop_after_block
                if target < payload['block']:raise ValueError('Stop cursor precedes committed progress')
                for block in range(payload['block']+1,target+1):
                    attempted = block; started = time.perf_counter()
                    plan = plan_block(clients,config,reference,block)
                    records = train_block(model,clients,plan,config,device,anchor,block=block)
                    snapshot = GlobalModelState(model,round_id=block)
                    metrics = validation_metrics(predict_validation(model,validation,config,device),validation) if due(block,config) else None
                    best_score,best_block = choose(metrics['rmse_w'],payload['best_rmse_w'],block,payload['best_block']) if metrics else (payload['best_rmse_w'],payload['best_block'])
                    state = snapshot.state_dict()
                    payload = {'block':block,'state':state,'best_state':state if best_block==block else payload['best_state'],
                               'best_block':best_block,'best_rmse_w':best_score,
                               'examples_processed':payload['examples_processed']+sum(p['examples_processed'] for p in plan),
                               'optimizer_steps':payload['optimizer_steps']+sum(p['optimizer_steps'] for p in plan),
                               'manifest_digest':digest_json(manifest)}
                    record = {'block':block,'selected_clients':[p['client_id'] for p in plan],'clients':records,
                              'validation':metrics,'global_fingerprint':snapshot.fingerprint,'seconds':time.perf_counter()-started}
                    receipts.append(cio.commit_block(output,payload,record))
                    print(f"block={block} examples={payload['examples_processed']} steps={payload['optimizer_steps']} validation={metrics} selected={best_block}",flush=True)
                final = finalize(output,payload,model,clients,validation,config,device) if payload['block']==config['rounds'] else None
            summary = {'status':'complete' if final else 'stopped','completed_block':payload['block'],
                       'selected_block':payload['best_block'],'validation_rmse_w':payload['best_rmse_w'],
                       'examples_processed':payload['examples_processed'],'optimizer_steps':payload['optimizer_steps'],
                       'resources':resource.result,'export_reload_exact':bool(final and final['export_reload_exact']),
                       'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                       'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
            write_event(output,summary)
            if final:
                immutable_json(output/'completed.json',{**summary,'manifest_sha256':sha256(output/'manifest.json'),
                               'final_receipt_sha256':sha256(output/'final/receipt.json')})
            return summary
        except BaseException as error:
            write_event(output,{'status':'aborted','attempted_block':attempted,'error':f'{type(error).__name__}: {error}'})
            raise


def run_from_paths(config, data_dir, output, *, resume=False, stop_after_block=None, verify_only=False):
    config = validate_config(config)
    # CLI/API disk entry is M100-only. Synthetic tests call run_control directly.
    for key in ('clients_per_round','minimum_windows','minimum_days','frozen_contract_sha256'):
        if config[key] != DEFAULTS[key]:raise ValueError('Disk runner requires original M100 protocol')
    reference = Path(config['reference_run'])
    if reference.is_absolute() or '..' in reference.parts:
        raise ValueError('Reference must be project-relative')
    reference = (ROOT/reference).resolve()
    if not reference.is_relative_to(ROOT/'output/experiments/v2_fedavg/main'):
        raise ValueError('Reference must be an authoritative v2 main run')
    clients = load_training_clients(data_dir,expected_contract_sha256=config['frozen_contract_sha256'],weather='masked',
                                    minimum_windows=config['minimum_windows'],minimum_days=config['minimum_days'])
    validation = load_coordinator_validation(data_dir,expected_contract_sha256=config['frozen_contract_sha256'])
    if clients.num_windows!=4506133 or len(clients.client_ids)!=979 or len(validation.split('val')[1])!=199670:
        raise ValueError('Original complete M100 training/validation cohort required')
    return run_control(clients,validation,config,output,reference,resume=resume,
                       stop_after_block=stop_after_block,verify_only=verify_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output-dir',required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume',action='store_true')
    mode.add_argument('--verify-only',action='store_true',help='Read-only CPU checkpoint verification; no inference')
    parser.add_argument('--stop-after-block',type=int)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve(); output = Path(args.output_dir).resolve()
    if not config_path.is_relative_to(ROOT/'configs/experiments/v3_centralized_sgd'):
        parser.error('Config must be under configs/experiments/v3_centralized_sgd/')
    allowed = ROOT/'output/experiments/v3_centralized_sgd'
    if not output.is_relative_to(allowed) or output==allowed:
        parser.error('Output must be a distinct run under output/experiments/v3_centralized_sgd/')
    if args.verify_only and args.stop_after_block is not None:
        parser.error('--verify-only cannot take --stop-after-block')
    try:
        result = run_from_paths(read_json(config_path),args.data_dir,output,resume=args.resume,
                                stop_after_block=args.stop_after_block,verify_only=args.verify_only)
    except (OSError,ValueError,RuntimeError,KeyError) as error:
        parser.exit(1,f'CENTRALIZED SGD ERROR: {error}\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
