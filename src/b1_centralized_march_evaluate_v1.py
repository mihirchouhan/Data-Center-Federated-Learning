"""Additive locked centralized B1 March evaluation. Default: metadata only.

March preflight and scoring both require separate, expiring operator statements.
No fit, scaler-fit, model-selection or FedAvg inference interface exists here.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import io as string_io
import json
import os
import time

import numpy as np
from src import b1_march_integrity_v1 as base
from src import b1_preparation_io_v1 as storage

ROOT = base.ROOT
SEEDS = (42, 43, 44)
BLOCKS = {42: 200, 43: 190, 44: 190}
LOCK_DIR = 'output/march_window_audit/b1_centralized_model_selection_lock_v1'
LOCK_SHA = '7d03fb516a2ed23bf2ae736a5e5a6cfc0ba8b2c0e83bb88e7019bea39e97a7b5'
INVENTORY_SHA = '625976b16141c4166bd861fd3b98b36c3814800bc0a3382b04b311fe95b3b801'
OLD_PLAN = 'configs/experiments/b1_no_ff/march_test_v1.json'
OLD_PLAN_SHA = 'aa2d93a1a19295127098287eb40b47af9239249a69f04b4619728cbd8fc96345'
SOURCE = 'src/b1_centralized_march_evaluate_v1.py'
PATHS = {
    'preflight_output': 'output/experiments/b1_no_ff/centralized_march_preflight_v1',
    'receipt_output': 'output/experiments/b1_no_ff/centralized_march_authorization_v1',
    'evaluation_output': 'output/experiments/b1_no_ff/centralized_march_evaluation_v1',
}
OP_PREFLIGHT = 'B1_centralized_March_preflight_only'
OP_EVALUATE = 'B1_centralized_March_test_evaluation_only'
OP_VERIFY = 'B1_centralized_March_saved_verification_only'


def protocol():
    p = dict(base.PROTOCOL)
    del p['selected_round']
    p.update(centralized_selected_blocks={str(s): BLOCKS[s] for s in SEEDS},
             reference_FedAvg_selected_round=200,
             comparison='Paired centralized-minus-FedAvg scores; existing FedAvg predictions only')
    return p


def load_plan(path):
    p = base.read_json(path)
    expected = {'schema', 'selection_lock', 'original_test_plan', 'protocol',
                'sources', 'limits', 'historical_preflight', 'semantic_reproduction', *PATHS}
    if set(p) != expected or p['schema'] != 'b1.centralized.march.test.plan.v1':
        raise ValueError('Central evaluation plan schema differs')
    if p['selection_lock'] != {'path': LOCK_DIR+'/selection_lock.json', 'sha256': LOCK_SHA}:
        raise ValueError('Wrong central selection lock')
    if p['original_test_plan'] != {'path': OLD_PLAN, 'sha256': OLD_PLAN_SHA}:
        raise ValueError('Original March plan differs')
    if any(p[k] != v for k, v in PATHS.items()) or p['protocol'] != protocol():
        raise ValueError('Evaluation outputs or scientific protocol differ')
    if (p['historical_preflight'], p['semantic_reproduction']) != ('FAILED', 'UNKNOWN'):
        raise ValueError('Historical status differs')
    old = base.load_plan(ROOT/OLD_PLAN)  # Source/config metadata, no March arrays.
    if p['limits'] != old['limits']:
        raise ValueError('Existing resource limits must be preserved')
    raw = base.read_json(ROOT/LOCK_DIR/'selection_lock.json')
    expected_sources = {**old['sources'], **raw['sources']}
    if set(p['sources']) != set(expected_sources) | {SOURCE}:
        raise ValueError('Execution source graph differs')
    for n, b in expected_sources.items():
        if p['sources'][n] != b:
            raise ValueError('Historical source binding changed')
    if storage.binding(ROOT/OLD_PLAN)['sha256'] != OLD_PLAN_SHA:
        raise ValueError('Original plan hash differs')
    for n, b in p['sources'].items():
        base.check(base.child(ROOT, n), b)
    return p


def validate_selected(lock):
    records = lock.get('selected_models', [])
    if lock.get('seedset') != list(SEEDS) or [r.get('seed') for r in records] != list(SEEDS):
        raise ValueError('Missing, duplicate or unordered selected seeds')
    if any(r.get('selected_block') != BLOCKS[r['seed']] or r.get('completed_block') != 200 for r in records):
        raise ValueError('Selected blocks differ')
    if lock.get('test_evaluation_authorized') is not False:
        raise ValueError('A selection lock cannot authorize test evaluation')
    return records


def locked_context(plan):
    """Pinned central package and original FedAvg context, with no March I/O."""
    directory = ROOT/LOCK_DIR
    if storage.binding(directory/'selection_lock.json')['sha256'] != LOCK_SHA:
        raise ValueError('Central selection lock hash differs')
    if storage.binding(directory/'generated_files.json')['sha256'] != INVENTORY_SHA:
        raise ValueError('Central selection inventory hash differs')
    inventory = base.read_json(directory/'generated_files.json')['files']
    for n, b in inventory.items():
        base.check(base.child(directory, n), b)
    spec = importlib.util.spec_from_file_location('_central_lock_verifier', directory/'verify_lock.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    if verifier.verify(directory)['status'] != 'PASS':
        raise ValueError('Central selection verification failed')
    lock = base.read_json(directory/'selection_lock.json')
    validate_selected(lock)
    old = base.load_plan(ROOT/OLD_PLAN)
    fed, scaler, _ = base.locked_context(old)
    for key in ('dataset_manifest', 'scaler_lock', 'scaler_statistics'):
        if lock[key] != fed[key]:
            raise ValueError('Central and FedAvg training/scaler bindings differ')
    for n, b in lock['sources'].items():
        if plan['sources'].get(n) != b:
            raise ValueError('Central training source graph differs')
    return lock, old, fed, scaler


def unused(path):
    path = storage.safe_path(path)
    if os.path.lexists(path) or os.path.lexists(path.parent/('.'+path.name+'.pending')):
        raise FileExistsError('Occupied final or incomplete output; preserve it')
    if not path.parent.is_dir() or path.parent.stat().st_uid != os.getuid():
        raise ValueError('Existing owned output parent required')
    return path


def device_record(device):
    from src.b1_march_evaluate_v1 import device_contract
    return device_contract(device)  # Environment validation, no CUDA initialization.


def operator_bindings(plan, device, preflight=False, verification=False):
    bindings = {'plan_sha256': base.digest(plan), 'selection_lock_sha256': LOCK_SHA,
                'device': device, 'output': str(ROOT/plan['evaluation_output'])}
    if preflight:
        bindings['preflight_output'] = str(ROOT/plan['preflight_output'])
    else:
        bindings['preflight_evidence'] = storage.binding(ROOT/plan['preflight_output']/'evidence.json')
    if verification:
        bindings['evaluation_receipt'] = storage.binding(ROOT/plan['receipt_output']/'receipt.json')
    return bindings


def require_operator(plan, path, device, preflight=False, verification=False):
    # This check MUST precede construction of MarchData or reading predictions.
    device_record(device)
    return base.statement(path, OP_VERIFY if verification else OP_PREFLIGHT if preflight else OP_EVALUATE,
                          operator_bindings(plan, device, preflight, verification))


def check_receipt(receipt, expected):
    if storage.canonical(receipt) != storage.canonical(expected):
        raise PermissionError('Receipt does not bind exact central models, March data and output')
    if receipt['operator_assertion'].get('explicit_operator_authorization') is not True:
        raise PermissionError('Explicit evaluation authorization absent')


def aligned_reference(reference, other):
    """Exact structured rows: ordinals, UTC origins, target values and row order."""
    if np.ma.isMaskedArray(reference) or np.ma.isMaskedArray(other):
        raise ValueError('Masked reference')
    a, b = np.asarray(reference), np.asarray(other)
    if a.dtype != b.dtype or a.ndim != 1 or not len(a) or a.shape != b.shape:
        raise ValueError('Reference shape/schema differs')
    required = ('node_index', 'origin_ns', 'power_w', 'temperature_c')
    if a.dtype.names != required:
        raise ValueError('Reference fields differ')
    if a.dtype['node_index'].kind not in 'iu' or a.dtype['origin_ns'] != np.dtype('int64') or any(a.dtype[k] != np.dtype('float64') for k in ('power_w', 'temperature_c')):
        raise ValueError('Reference field dtypes differ')
    if not np.array_equal(a, b):
        raise ValueError('Window, node, target or ordered identity mismatch')
    if not np.isfinite(a['power_w']).all() or not np.isfinite(a['temperature_c']).all():
        raise ValueError('Missing/nonfinite observed targets')
    if (a['node_index'] < 0).any() or (a['node_index'] >= 979).any():
        raise ValueError('Invalid original ordinal')
    ids = list(zip(a['node_index'].tolist(), a['origin_ns'].tolist()))
    if len(set(ids)) != len(a) or ids != sorted(ids):
        raise ValueError('Duplicate/unordered window identities')


def valid_predictions(predicted, n):
    if np.ma.isMaskedArray(predicted):
        raise ValueError('Masked predictions')
    p = np.asarray(predicted)
    if p.shape != (n, 2) or p.dtype != np.dtype('float64') or not np.isfinite(p).all():
        raise ValueError('Incomplete, invalid or nonfinite predictions')
    return p


def check_inventory(directory, completed):
    entries = completed['artifacts']
    if {p.name for p in directory.iterdir()} != set(entries) | {'completed.json'}:
        raise ValueError('Missing/extra completed artifacts')
    for n, b in entries.items():
        if Path(n).name != n:
            raise ValueError('Unexpected nested artifact path')
        base.check(base.child(directory, n), b)


def fedavg_context(old, fed_lock, data):
    """Validate prior inference evidence/alignment; no new metrics or forward."""
    from src import b1_march_evaluate_v1 as common
    from src import b1_march_data_v1 as march
    directory = ROOT/old['evaluation_output']
    complete = base.read_json(directory/'completed.json')
    check_inventory(directory, complete)
    receipt = base.read_json(directory/'authorization_receipt.json')
    if complete['status'] != 'complete' or complete['plan_sha256'] != base.digest(old) or complete['receipt_sha256'] != base.digest(receipt):
        raise ValueError('Prior FedAvg evaluation completion differs')
    expected = {
        'schema': 'b1.march.evaluation.authorization.v1', 'decision': 'AUTHORIZED_TEST_INFERENCE_ONLY',
        'plan_sha256': base.digest(old), 'selection_lock_sha256': base.LOCK_SHA,
        'training_configuration': fed_lock['configuration'], 'training_dataset_manifest': fed_lock['dataset_manifest'],
        'scaler_lock': fed_lock['scaler_lock'], 'scaler_statistics': fed_lock['scaler_statistics'],
        'selected_models': fed_lock['selected_models'], 'sources': old['sources'], 'protocol': old['protocol'],
        'test_dataset_manifest': data.manifest_binding, 'test_reference': data.manifest['reference_file'],
        'test_results': data.manifest['results'], 'output': str(directory),
        'device_environment': receipt.get('device_environment'), 'operator_assertion': receipt.get('operator_assertion'),
        'historical_preflight': 'FAILED', 'semantic_reproduction': 'UNKNOWN'}
    common.check_receipt(receipt, expected)
    # Historical operator expiry is not refreshed: this verifies existing evidence.
    env = receipt['device_environment']
    if env.get('device') != 'cuda:0' or env.get('CUDA_VISIBLE_DEVICES') != base.GPU_UUID or env.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Prior FedAvg reviewed GPU evidence differs')
    op = receipt['operator_assertion']
    expected_bindings = {'plan_sha256': base.digest(old), 'selection_lock_sha256': base.LOCK_SHA,
                         'dataset_manifest': data.manifest_binding, 'output': str(directory), 'device': 'cuda:0'}
    if op.get('operation') != 'B1_March_test_evaluation_only' or op.get('bindings') != expected_bindings:
        raise ValueError('Prior FedAvg operator evidence differs')
    reference = march.read_reference(directory/'reference.npy')
    aligned_reference(data.reference, reference)
    execution = base.read_json(directory/'execution.json')
    if [r['seed'] for r in execution['seeds']] != list(SEEDS):
        raise ValueError('Prior FedAvg seed execution differs')
    predicted = {}
    for rec, selected in zip(execution['seeds'], fed_lock['selected_models']):
        if rec['round'] != 200 or rec['model'] != selected['selected_model'] or rec['state_unchanged'] is not True:
            raise ValueError('Prior FedAvg locked model identity differs')
        predicted[rec['seed']] = valid_predictions(common._read_predictions(directory/f"predictions_seed{rec['seed']}.npy", len(reference)), len(reference))
    return directory, complete, predicted


def frozen_context(plan):
    """Only called after operation-specific operator authorization."""
    lock, old, fed, scaler = locked_context(plan)
    from src.b1_march_data_v1 import MarchData
    # IMPORTANT: the immutable prepared manifest embeds the ORIGINAL plan.
    data = MarchData(ROOT/old['dataset_output']/'dataset', old, scaler)
    directory, complete, predicted = fedavg_context(old, fed, data)
    evidence = {
        'schema': 'b1.centralized.march.frozen.context.v1', 'plan_sha256': base.digest(plan),
        'selection_lock_sha256': LOCK_SHA, 'matrix_completion': lock['matrix_completion'],
        'central_models': lock['selected_models'], 'training_configuration': lock['configuration'],
        'training_dataset_manifest': lock['dataset_manifest'], 'scaler_lock': lock['scaler_lock'],
        'scaler_statistics': lock['scaler_statistics'], 'test_dataset_manifest': data.manifest_binding,
        'test_reference': data.manifest['reference_file'], 'test_results': data.manifest['results'],
        'reference_FedAvg_lock_sha256': base.LOCK_SHA,
        'reference_FedAvg_completion': storage.binding(directory/'completed.json'),
        'reference_FedAvg_artifacts': complete['artifacts'], 'protocol': plan['protocol'],
        'sources': plan['sources'], 'evaluation_output': str(ROOT/plan['evaluation_output']),
    }
    return lock, data, directory, predicted, evidence


def resource_budget(plan, output, quota_path):
    return storage.Budget(output, storage.Limits(**plan['limits']), base.read_json(quota_path))


def publish_small(plan, key, files, budget):
    destination = unused(ROOT/plan[key]); stage = base.stage_directory(destination)
    try:
        for name, content in files.items():
            raw = storage.canonical(content); budget.charge(len(raw)); storage.exclusive(stage/name, raw)
        base.publish(stage, destination)
    except BaseException:
        # Keep all partial evidence; never remove/reuse an occupied stage.
        raise
    return destination


def preflight(plan, operator_path, device, quota_path):
    operator = require_operator(plan, operator_path, device, preflight=True)
    output = unused(ROOT/plan['preflight_output']); unused(ROOT/plan['receipt_output']); unused(ROOT/plan['evaluation_output'])
    budget = resource_budget(plan, output, quota_path)
    _, data, _, _, evidence = frozen_context(plan)
    budget.check(); assert_context_unchanged(plan, evidence, data)
    path = publish_small(plan, 'preflight_output', {'evidence.json': evidence, 'operator_statement.json': operator}, budget)
    return {'status': 'frozen_context_verified', 'evidence': storage.binding(path/'evidence.json'), 'inference': False, 'metrics_calculated': False}


def read_preflight(plan):
    directory = ROOT/plan['preflight_output']
    if {p.name for p in directory.iterdir()} != {'owner.json', 'evidence.json', 'operator_statement.json'}:
        raise ValueError('Preflight artifacts differ')
    evidence = base.read_json(directory/'evidence.json')
    if evidence['plan_sha256'] != base.digest(plan) or evidence['selection_lock_sha256'] != LOCK_SHA:
        raise ValueError('Preflight source/plan binding differs')
    return evidence


def receipt_body(plan, evidence, operator, device):
    return {'schema': 'b1.centralized.march.authorization.v1', 'decision': 'AUTHORIZED_CENTRAL_TEST_INFERENCE_ONLY',
            'preflight_evidence': storage.binding(ROOT/plan['preflight_output']/'evidence.json'),
            'context': evidence, 'device_environment': device_record(device), 'operator_assertion': operator,
            'historical_preflight': 'FAILED', 'semantic_reproduction': 'UNKNOWN'}


def authorize(plan, operator_path, device, quota_path):
    operator = require_operator(plan, operator_path, device)
    output = unused(ROOT/plan['receipt_output']); unused(ROOT/plan['evaluation_output'])
    budget = resource_budget(plan, output, quota_path)
    _, data, _, _, evidence = frozen_context(plan)
    if evidence != read_preflight(plan):
        raise ValueError('Frozen preflight bindings changed')
    receipt = receipt_body(plan, evidence, operator, device)
    assert_context_unchanged(plan, evidence, data)
    path = publish_small(plan, 'receipt_output', {'receipt.json': receipt}, budget)
    return {'status': 'authorized_receipt_published', 'receipt': storage.binding(path/'receipt.json'), 'inference': False}


def load_selected(record, data, device):
    """Only evaluate calls this, after exact receipt equality. No FedAvg loader."""
    from src.b1_device_adapter_v1 import B1DeviceAdapter
    from src.b1_real_fedavg_runner_v1 import state_contract
    from src.b1_fedavg_orchestration_v1 import RoundSnapshot
    import torch
    path = ROOT/record['selected_model']['path']; base.check(path, record['selected_model'])
    with storage.safe_open(path) as f:
        saved = torch.load(f, map_location='cpu', weights_only=True)
    manifest = base.read_json(ROOT/record['run_manifest']['path'])
    validate_export(record, saved, manifest)
    roles = state_contract(saved['state'], data.scaler.target_scaling)
    snapshot = RoundSnapshot(saved['state'], roles, round_id=record['selected_block'])
    if snapshot.fingerprint != record['selected_state_fingerprint']:
        raise ValueError('Central selected state differs')
    base.check(path, record['selected_model'])
    model = B1DeviceAdapter(target_scaling=data.scaler.target_scaling, root_seed=record['seed'], device=device)
    model.restore_round(snapshot); model.eval()
    return model


def validate_export(record, saved, manifest):
    if record.get('seed') not in BLOCKS or record.get('selected_block') != BLOCKS[record['seed']]:
        raise ValueError('Selected export seed/block differs')
    if manifest.get('seed') != record['seed'] or manifest.get('schema') != 'b1.centralized.training.manifest.v1':
        raise ValueError('Central run manifest identity differs')
    if saved.get('schema') != 'b1.centralized.selected.model.v1' or saved.get('manifest_sha256') != base.digest(manifest):
        raise ValueError('Central export envelope differs')


def assert_context_unchanged(plan, evidence, data):
    data.assert_unchanged()
    if storage.binding(ROOT/LOCK_DIR/'selection_lock.json')['sha256'] != LOCK_SHA:
        raise ValueError('Central lock changed during operation')
    for n, b in plan['sources'].items():
        base.check(ROOT/n, b)
    for record in evidence['central_models']:
        for key in ('selected_model', 'selected_round_checkpoint', 'selected_round_receipt', 'selected_receipt', 'completion', 'run_manifest'):
            entry = record[key]; base.check(ROOT/entry['path'], entry)
    for key in ('training_configuration', 'training_dataset_manifest', 'scaler_lock', 'scaler_statistics', 'matrix_completion'):
        entry = evidence[key]; base.check(ROOT/entry['path'], entry)
    directory = ROOT/'output/experiments/b1_no_ff/march_test_evaluation_v2'
    base.check(directory/'completed.json', evidence['reference_FedAvg_completion'])
    for n, b in evidence['reference_FedAvg_artifacts'].items():
        base.check(base.child(directory, n), b)


def csv_bytes(rows):
    stream = string_io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def comparison_tables(central, fed):
    """Called only in separately authorized evaluation/verification, never tests."""
    from src.b1_march_evaluate_v1 import aggregate
    byseed, bynode = [], []
    if set(central) != set(fed) or set(central) != set(SEEDS):
        raise ValueError('Comparison seedsets differ')
    for seed in SEEDS:
        a, b = central[seed], fed[seed]
        if (a['samples'], a['identity_sha256']) != (b['samples'], b['identity_sha256']) or a['per_node'].keys() != b['per_node'].keys():
            raise ValueError('Comparison cohort differs')
        for group in ('pooled', 'macro_node'):
            for j, target in enumerate(('power_W', 'ambient_degC')):
                for metric in ('mae', 'rmse'):
                    x, y = a[group][metric][j], b[group][metric][j]
                    byseed.append({'seed': seed, 'aggregation': group, 'output': target, 'metric': metric,
                                   'centralized_sgd': x, 'FedAvg': y, 'centralized_minus_FedAvg': x-y})
        for node, x in a['per_node'].items():
            y = b['per_node'][node]
            if x['samples'] != y['samples']:
                raise ValueError('Per-node sample counts differ')
            for j, target in enumerate(('power_W', 'ambient_degC')):
                bynode.append({'seed': seed, 'node': node, 'samples': x['samples'], 'output': target,
                               'centralized_minus_FedAvg_MAE': x['mae'][j]-y['mae'][j],
                               'centralized_minus_FedAvg_RMSE': x['rmse'][j]-y['rmse'][j]})
    a, b = aggregate(central), aggregate(fed); summaries = []
    for group in ('pooled', 'macro_node'):
        for j, target in enumerate(('power_W', 'ambient_degC')):
            for metric in ('mae', 'rmse'):
                delta = [central[s][group][metric][j]-fed[s][group][metric][j] for s in SEEDS]
                summaries.append({'aggregation': group, 'output': target, 'metric': metric,
                                  'centralized_mean': a[group][metric]['mean'][j], 'centralized_sample_sd': a[group][metric]['sample_sd'][j],
                                  'FedAvg_mean': b[group][metric]['mean'][j], 'FedAvg_sample_sd': b[group][metric]['sample_sd'][j],
                                  'paired_difference_mean': float(np.mean(delta)), 'paired_difference_sample_sd': float(np.std(delta, ddof=1))})
    return {'comparison_seed_metrics.csv': csv_bytes(byseed), 'comparison_node_metrics.csv': csv_bytes(bynode),
            'comparison_aggregate_metrics.csv': csv_bytes(summaries)}


def saved_scores(output, data, fed_directory, fed_predictions):
    from src import b1_march_evaluate_v1 as common
    from src.b1_march_data_v1 import bridge
    common.verify_saved_predictions(output, data)  # Independent math.fsum, all tables.
    central = {int(k): v for k, v in base.read_json(output/'metrics.json').items()}
    fed = {int(k): v for k, v in base.read_json(fed_directory/'metrics.json').items()}
    if set(fed) != set(SEEDS):
        raise ValueError('Missing saved FedAvg metrics')
    common.verify_saved_predictions(fed_directory, data)  # Saved arrays only; no FedAvg forward.
    for name, raw in comparison_tables(central, fed).items():
        if (output/name).read_bytes() != raw:
            raise ValueError('Comparison table differs')
    for seed in SEEDS:
        predicted = common._read_predictions(output/f'predictions_seed{seed}.npy', len(data.reference))
        expected = predicted-fed_predictions[seed]
        got = common._read_predictions(output/f'prediction_delta_seed{seed}.npy', len(data.reference))
        valid_predictions(expected, len(data.reference)); valid_predictions(got, len(data.reference))
        if not np.array_equal(expected, got):
            raise ValueError('Prediction-level differences are misaligned')
    return {'status': 'verified_complete', 'seeds': list(SEEDS), 'windows_per_seed': len(data.reference),
            'nodes': data.manifest['results']['nodes'], 'identical_observed_targets_and_windows': True,
            'independent_saved_prediction_metrics': True, 'FedAvg_inference': False, 'model_forward': False}


def verify_output(output, plan, receipt, data, fed_directory, fed_predictions):
    completed = base.read_json(output/'completed.json')
    if completed['status'] != 'complete' or completed['plan_sha256'] != base.digest(plan) or completed['receipt_sha256'] != base.digest(receipt):
        raise ValueError('Central evaluation completion differs')
    check_inventory(output, completed)
    if base.read_json(output/'authorization_receipt.json') != receipt:
        raise ValueError('Published authorization receipt differs')
    execution = base.read_json(output/'execution.json')
    if [(r['seed'], r['selected_block']) for r in execution['seeds']] != list(BLOCKS.items()):
        raise ValueError('Executed seed/block identities differ')
    records = receipt['context']['central_models']
    for got, record in zip(execution['seeds'], records):
        if got['model'] != record['selected_model'] or got['state_unchanged'] is not True:
            raise ValueError('Executed model binding differs')
    result = saved_scores(output, data, fed_directory, fed_predictions)
    if base.read_json(output/'verification.json') != result:
        raise ValueError('Independent verification evidence differs')
    assert_context_unchanged(plan, receipt['context'], data)
    return result


def evaluate(plan, operator_path, device, quota_path, receipt_path):
    operator = require_operator(plan, operator_path, device)
    output = unused(ROOT/plan['evaluation_output']); budget = resource_budget(plan, output, quota_path)
    lock, data, fed_directory, fed_predictions, evidence = frozen_context(plan)
    if evidence != read_preflight(plan):
        raise ValueError('Preflight bindings differ')
    if storage.safe_path(receipt_path) != ROOT/plan['receipt_output']/'receipt.json':
        raise PermissionError('Exact isolated receipt path required')
    receipt = base.read_json(receipt_path)
    check_receipt(receipt, receipt_body(plan, evidence, operator, device))
    stage = base.stage_directory(output); started, finished = [], []
    # Conservative reservation for predictions, differences, reference, tables and evidence.
    budget.charge(len(data.reference)*16*8+64*1024**2)
    allowed = [data.path/'reference.npy', *(data.path/r['path'] for r in data.manifest['nodes']),
               *(fed_directory/n for n in evidence['reference_FedAvg_artifacts'] if n.endswith(('.npy', '.npz')))]
    models = [ROOT/r[key]['path'] for r in lock['selected_models'] for key in ('selected_model', 'selected_round_checkpoint')]
    base.guard_test_access(stage, allowed, models)
    try:
        import torch
        from src.b1_real_fedavg_runner_v1 import runtime
        from src import b1_march_evaluate_v1 as common
        intended, environment = runtime(device)
        if device == 'cuda:0' and str(torch.cuda.get_device_properties(intended).uuid).removeprefix('GPU-') != base.GPU_UUID.removeprefix('GPU-'):
            raise ValueError('Physical GPU differs')
        storage.exclusive(stage/'authorization_receipt.json', storage.canonical(receipt))
        with (stage/'reference.npy').open('xb') as f:
            np.save(f, data.reference, allow_pickle=False)
        scores, execution = {}, []
        for record in lock['selected_models']:
            seed, block = record['seed'], record['selected_block']; started.append(seed); begin = time.monotonic()
            budget.check(); model = load_selected(record, data, device); fingerprint = model.snapshot(block).fingerprint
            predicted = np.lib.format.open_memmap(stage/f'predictions_seed{seed}.npy', mode='w+', dtype='f8', shape=(len(data.reference), 2))
            cursor = 0
            if intended.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(intended)
            with torch.inference_mode():
                for x, y, nodes, origins in data.batches(plan['protocol']['evaluation_batch_size']):
                    budget.check(); stop = cursor+len(x); ref = data.reference[cursor:stop]
                    if not np.array_equal(nodes, np.asarray(common.bridge_names())[ref['node_index']]) or not np.array_equal(origins.view('i8'), ref['origin_ns']) or not np.array_equal(y, common._actual(ref)):
                        raise ValueError('Pre-forward identities/targets differ')
                    result = model(torch.tensor(np.array(x, copy=True), dtype=torch.float32, device=intended)).cpu().numpy()
                    valid_predictions(result, len(x)); predicted[cursor:stop] = result; cursor = stop
            if cursor != len(data.reference) or model.snapshot(block).fingerprint != fingerprint:
                raise ValueError('Incomplete inference or changed model')
            predicted.flush()
            with (stage/f'prediction_delta_seed{seed}.npy').open('xb') as f:
                np.save(f, np.asarray(predicted)-fed_predictions[seed], allow_pickle=False)
            nodes = np.asarray(common.bridge_names())[data.reference['node_index']]
            origins = data.reference['origin_ns'].astype('datetime64[ns]')
            scores[seed] = common.metrics(common._actual(data.reference), predicted, nodes, origins)
            execution.append({'seed': seed, 'selected_block': block, 'model': record['selected_model'], 'state_unchanged': True,
                              'seconds': time.monotonic()-begin, 'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(intended) if intended.type == 'cuda' else None,
                              'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(intended) if intended.type == 'cuda' else None})
            finished.append(seed); del predicted, model
        storage.exclusive(stage/'metrics.json', storage.canonical({str(s): v for s, v in scores.items()}))
        storage.exclusive(stage/'aggregate_metrics.json', storage.canonical(common.aggregate(scores)))
        common.write_tables(stage, scores)
        fed_scores = {int(k): v for k, v in base.read_json(fed_directory/'metrics.json').items()}
        for name, raw in comparison_tables(scores, fed_scores).items():
            storage.exclusive(stage/name, raw)
        storage.exclusive(stage/'execution.json', storage.canonical({'environment': environment, 'seeds': execution, 'FedAvg_inference': False}))
        report = '# Centralized B1 March comparison\n\nPreviously inspected retrospective March evaluation data; not untouched confirmatory data. Central locked blocks200/190/190 versus existing FedAvg Round200 predictions on identical observed windows. No training, scaler fitting or reselection.\n\nSee seed_metrics.csv, node_metrics.csv and aggregate_metrics.csv for power W/ambient degC pooled and equal-node macro MAE/RMSE. comparison_*.csv gives paired central-minus-FedAvg differences; negative error differences favor central. Prediction vectors and prediction deltas share reference.npy row order. Across-seed SD uses ddof=1 and is not a confidence interval. No energy-saving or formal privacy claim. Historical preflight FAILED; semantic reproduction UNKNOWN.\n'
        storage.exclusive(stage/'REPORT.md', report.encode())
        checked = saved_scores(stage, data, fed_directory, fed_predictions)
        storage.exclusive(stage/'verification.json', storage.canonical(checked))
        assert_context_unchanged(plan, evidence, data)
        # Rehash all prior FedAvg artifacts after comparison; do not republish them.
        check_inventory(fed_directory, base.read_json(fed_directory/'completed.json'))
        completed = {'status': 'complete', 'plan_sha256': base.digest(plan), 'receipt_sha256': base.digest(receipt), 'artifacts': storage.inventory(stage)}
        storage.exclusive(stage/'completed.json', storage.canonical(completed))
        verify_output(stage, plan, receipt, data, fed_directory, fed_predictions)
        base.publish(stage, output)
        return {**checked, 'inference_completed': True, 'output': str(output)}
    except BaseException as error:
        try:
            storage.exclusive(stage/'failure.json', storage.canonical({'error': str(error), 'models_started': started, 'models_completed': finished, 'automatic_retry': False}))
        except OSError:
            pass
        raise


def verify(plan, operator_path, device, receipt_path):
    require_operator(plan, operator_path, device, verification=True)
    _, data, fed_directory, fed_predictions, evidence = frozen_context(plan)
    if evidence != read_preflight(plan):
        raise ValueError('Frozen preflight differs')
    if storage.safe_path(receipt_path) != ROOT/plan['receipt_output']/'receipt.json':
        raise PermissionError('Exact isolated receipt required')
    receipt = base.read_json(receipt_path)
    # Verify the original immutable authorization; a fresh verification-only
    # statement gates this new read/scoring operation without rewriting receipts.
    check_receipt(receipt, receipt_body(plan, evidence, receipt['operator_assertion'], device))
    return verify_output(ROOT/plan['evaluation_output'], plan, receipt, data, fed_directory, fed_predictions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--stage', choices=('describe', 'inspect', 'preflight', 'authorize', 'evaluate', 'verify'), default='describe')
    p.add_argument('--device', choices=('cpu', 'cuda:0'))
    p.add_argument('--operator-authorization'); p.add_argument('--quota-evidence'); p.add_argument('--receipt')
    a = p.parse_args(argv); plan = load_plan(a.plan)
    if a.stage in ('describe', 'inspect'):
        if a.stage == 'inspect':
            locked_context(plan)
        print(json.dumps({'status': 'implementation_only' if a.stage == 'describe' else 'selection_sources_verified',
                          'March_access': False, 'model_initialization': False, 'metrics_calculated': False,
                          'authorization_required': True, 'selected_blocks': BLOCKS, 'outputs': PATHS})); return
    if not a.device or not a.operator_authorization:
        raise PermissionError('Explicit device and exact-scope operator statement required BEFORE March access')
    if a.stage in ('preflight', 'authorize', 'evaluate') and not a.quota_evidence:
        raise PermissionError('Fresh quota evidence required')
    if a.stage in ('evaluate', 'verify') and not a.receipt:
        raise PermissionError('Separate immutable central evaluation receipt required')
    if a.stage == 'preflight':
        result = preflight(plan, a.operator_authorization, a.device, a.quota_evidence)
    elif a.stage == 'authorize':
        result = authorize(plan, a.operator_authorization, a.device, a.quota_evidence)
    elif a.stage == 'evaluate':
        result = evaluate(plan, a.operator_authorization, a.device, a.quota_evidence, a.receipt)
    else:
        result = verify(plan, a.operator_authorization, a.device, a.receipt)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
