"""Explicit v5 plan/smoke/fit/verify CLI. Importing never executes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from src.experiment_data import immutable_json, read_json, sha256
from src.federated_train import GlobalModelState, aggregate_fedavg
from src.model import SequenceRegressor
from src.v5_missing_data import training_io as tio
from src.v5_missing_data.sampling import round_plan, metadata, stream_seed, due, select, verify_budget_totals
from src.v5_missing_data.training_data import CohortContext, TrainingRegistry, SplitData, POLICIES

ROOT = Path(__file__).resolve().parents[2]
ALGORITHMS = ('fedavg', 'centralized_sgd')
SEEDS = (42, 43, 44)
FIXED = {'history_steps': 36, 'horizon_steps': 6, 'hidden_size': 64, 'num_layers': 2,
    'node_embedding_size': 8, 'learning_rate': .01, 'momentum': 0., 'weight_decay': 0.,
    'batch_size': 512, 'drop_last': False, 'clients_per_round': 98, 'rounds': 200,
    'validation_every': 10, 'eval_batch_size': 1024, 'num_threads': 1, 'smoke_rounds': 3}


def load_plan(path, root=ROOT):
    plan = read_json(path)
    fields = {'version','settings','algorithms','policies','seeds','device','subset_algorithm','sources','evidence','cohort_receipt_sha256'}
    if (set(plan) != fields or plan['version'] != 1 or plan['settings'] != FIXED or plan['algorithms'] != list(ALGORITHMS)
            or plan['policies'] != list(POLICIES) or plan['seeds'] != list(SEEDS)
            or plan['device'] != 'cuda:0'
            or plan['subset_algorithm'] != 'choice_without_replacement_sorted_then_domain2_permutation_v1'):
        raise ValueError('Matrix/protocol differs from the locked implementation')
    tio.sources(root, plan['sources'])
    for name, checksum in plan['evidence'].items():
        if sha256(root / name) != checksum:
            raise ValueError('Protocol or cohort evidence changed: ' + name)
    return plan


def initialize(context, seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stream_seed(seed, 0))
        return SequenceRegressor(14, 1, hidden_size=64, num_layers=2, model='gru',
            node_count=len(context.names), node_embedding_size=8, baseline_coefficients=context.coefficients)


def runtime(plan):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG'] not in (':4096:8', ':16:8'):
        raise ValueError('Unsupported deterministic cuBLAS setting')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback')
    device = torch.device(plan['device'])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    return device


def train_local(model, data, anchor, device):
    """Future authorized fit path only. Optimizer state starts empty each session."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0., weight_decay=0.)
    if optimizer.state:
        raise ValueError('Local SGD state did not start empty')
    examples = steps = 0
    for inputs, targets in data.batches(512):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(torch.from_numpy(inputs).to(device))
        target = torch.from_numpy(targets).to(device)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not torch.isfinite(prediction).all().item() or not torch.isfinite(loss).item():
            raise ValueError('Nonfinite prediction/loss; abort entire round')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in model.parameters()):
            raise ValueError('Nonfinite gradient; abort entire round')
        optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError('Nonfinite update; abort entire round')
        examples += len(targets)
        steps += 1
    anchor._validate(model.state_dict(), label='sampled-pass update')
    if examples != len(data.targets) or steps != (examples+511)//512 or optimizer.state:
        raise ValueError('Sampled-pass optimizer/example accounting mismatch')
    return examples, steps


def validate_payload(output, payload, records, registry, initial, seed, cap):
    if payload is None:
        return
    anchor = GlobalModelState(initial, round_id=0)
    if not records or records[0]['global_fingerprint'] != anchor.fingerprint:
        raise ValueError('Recorded initial parameters differ from fresh seeded initialization')
    anchor._validate(payload['state'], label='resumed global')
    anchor._validate(payload['best_state'], label='resumed selected')
    examples = steps = 0
    for record in records:
        r = record['round']
        plans = round_plan(registry, seed, r) if r else []
        if record['clients'] != metadata(plans) or (record['validation'] is not None) != due(r, cap):
            raise ValueError('Saved sampling/validation schedule differs')
        tio.check_trace(output, r, plans)
        examples += sum(p['m_k'] for p in plans)
        steps += sum(p['optimizer_steps'] for p in plans)
        if record['examples_processed'] != examples or record['optimizer_steps'] != steps:
            raise ValueError('Saved cumulative budgets differ')
    if payload['round'] != len(records)-1 or payload['round'] > cap or payload['examples_processed'] != examples or payload['optimizer_steps'] != steps:
        raise ValueError('Checkpoint cursor/counts differ')
    candidates = [r for r in records if r['validation'] is not None]
    if any(not np.isfinite(r['validation']['rmse_w']) for r in candidates):
        raise ValueError('Nonfinite saved validation selection score')
    best = min(candidates, key=lambda r: (r['validation']['rmse_w'], r['round']))
    if payload['best_round'] != best['round'] or payload['best_score'] != best['validation']['rmse_w']:
        raise ValueError('Selected checkpoint is not earliest validation minimum')
    initial.load_state_dict(payload['state'], strict=True)
    if GlobalModelState(initial, round_id=payload['round']).fingerprint != records[-1]['global_fingerprint']:
        raise ValueError('Last global fingerprint differs')
    initial.load_state_dict(payload['best_state'], strict=True)
    if GlobalModelState(initial, round_id=payload['best_round']).fingerprint != best['global_fingerprint']:
        raise ValueError('Selected fingerprint differs')
    initial.load_state_dict(payload['state'], strict=True)


def output_path(root, value):
    output = Path(value).absolute()
    allowed = root / 'output/experiments/v5_missing_data/model_comparison/v1'
    cohorts = allowed / 'cohorts'
    if (not output.resolve().is_relative_to(allowed.resolve()) or output.resolve() == allowed.resolve()
            or output.resolve().is_relative_to(cohorts.resolve()) or output.is_symlink()):
        raise ValueError('Run output must be a new v5 comparison directory outside frozen cohorts')
    return output


def manifest(plan, context, method, policy, seed, kind, device_name):
    cap = 3 if kind == 'smoke' else 200
    return {'version': 1, 'method': method, 'policy': policy, 'seed': seed, 'kind': kind, 'cap': cap,
        'plan': plan, 'plan_digest': tio.digest(plan),
        'settings': plan['settings'], 'subset_algorithm': plan['subset_algorithm'],
        'sources': plan['sources'], 'software': tio.software(), 'device': plan['device'], 'device_name': device_name,
        'cohort_receipt_sha256': context.receipt_sha256, 'cohort_directory': str(context.path),
        'common_validation_identity': context.contract['cohorts']['common/all_policies/val']['identity'],
        'common_test_identity': context.contract['cohorts']['common/all_policies/test']['identity'],
        'budgets': context.budgets['budget_totals'][str(seed)],
        'selection': 'Earliest strict minimum common validation RMSE W at 0,10,...,200; smoke also at 3',
        'privacy': 'Simulation only; no differential privacy, secure aggregation or OS isolation'}


def run_one(root, plan, context, method, policy, seed, kind, output, *, resume=False, stop_after_round=None):
    """Called only by an explicit future --stage fit/smoke invocation."""
    if method not in ALGORITHMS or policy not in POLICIES or seed not in SEEDS:
        raise ValueError('Unknown declared fit')
    output = output_path(root, output)
    if not resume and (output.exists() or output.is_symlink()):
        raise ValueError('Output exists; never overwrite a fit')
    if resume and (output / 'completed.json').is_file():
        saved = read_json(output / 'manifest.json')
        if (saved['method'],saved['policy'],saved['seed'],saved['kind']) != (method,policy,seed,kind):
            raise ValueError('Requested resume identifies a different completed fit')
        return verify_run(plan, context, output)
    cap = 3 if kind == 'smoke' else 200
    if stop_after_round is not None and not 0 <= stop_after_round <= cap:
        raise ValueError('Invalid interruption cursor')
    registry, validation = TrainingRegistry(context, policy), SplitData(context, 'val')
    device = runtime(plan)
    run_manifest = manifest(plan, context, method, policy, seed, kind, torch.cuda.get_device_name(device))
    tio.open_run(root, output, run_manifest, resume=resume)
    from src.v5_missing_data.evaluate_policy_models import predict_and_score
    with tio.locked(output):
        initial = initialize(context, seed)
        payload, records = tio.load_commits(output)
        validate_payload(output, payload, records, registry, initial, seed, cap)
        model = initial.to(device)
        started = time.monotonic()
        if payload is None:
            score, _ = predict_and_score(model, validation, device)
            state = GlobalModelState(model, round_id=0)
            payload = {'round': 0, 'state': state.state_dict(), 'best_state': state.state_dict(),
                'best_round': 0, 'best_score': score['rmse_w'], 'examples_processed': 0, 'optimizer_steps': 0}
            record = {'round': 0, 'clients': [], 'validation': score, 'examples_processed': 0,
                'optimizer_steps': 0, 'global_fingerprint': state.fingerprint, 'seconds': 0.}
            records.append(tio.commit(output, payload, record, []))
        else:
            model.load_state_dict(payload['state'], strict=True)
        for r in range(payload['round']+1, cap+1):
            if stop_after_round is not None and r > stop_after_round:
                break
            before = time.monotonic()
            plans = round_plan(registry, seed, r)
            snapshot = GlobalModelState(model, round_id=r-1)
            updates, client_seconds = [], []
            for item in plans:
                client_started = time.monotonic()
                local_data = registry.local(item['node_index'], item['rows'])
                if (hashlib.sha256(local_data.row_positions.astype('<i8').tobytes()).hexdigest() != item['row_positions_sha256']
                        or hashlib.sha256(local_data.identities.tobytes()).hexdigest() != item['window_ids_sha256']):
                    raise ValueError('Worker receives different sampled rows/identities')
                if method == 'fedavg':
                    local_model = snapshot.initialize_client(model).to(device)
                else:
                    local_model = model
                examples, steps = train_local(local_model, local_data, snapshot, device)
                if examples != item['m_k'] or steps != item['optimizer_steps']:
                    raise ValueError('Worker accounting differs from immutable plan')
                if method == 'fedavg':
                    updates.append(snapshot.capture_update(item['client_id'], local_model, item['m_k']))
                client_seconds.append({'client_id':item['client_id'], 'seconds':time.monotonic()-client_started})
            if method == 'fedavg':
                state = aggregate_fedavg(snapshot, updates, expected_counts={p['client_id']: p['m_k'] for p in plans})
                model.load_state_dict(state, strict=True)
            state = GlobalModelState(model, round_id=r)
            validation_started = time.monotonic()
            validation_record = predict_and_score(model, validation, device)[0] if due(r, cap) else None
            validation_seconds = time.monotonic()-validation_started if validation_record is not None else 0.
            if validation_record is not None:
                for key in ('targets_sha256','persistence_sha256','window_identity'):
                    if validation_record[key] != records[0]['validation'][key]:
                        raise ValueError('Validation windows/targets/persistence changed')
            best_score, best_round = (select(validation_record['rmse_w'], payload['best_score'], r, payload['best_round'])
                                      if validation_record is not None else (payload['best_score'], payload['best_round']))
            payload = {'round': r, 'state': state.state_dict(), 'best_round': best_round, 'best_score': best_score,
                'best_state': state.state_dict() if best_round == r else payload['best_state'],
                'examples_processed': payload['examples_processed']+sum(p['m_k'] for p in plans),
                'optimizer_steps': payload['optimizer_steps']+sum(p['optimizer_steps'] for p in plans)}
            record = {'round': r, 'clients': metadata(plans), 'validation': validation_record,
                'examples_processed': payload['examples_processed'], 'optimizer_steps': payload['optimizer_steps'],
                'global_fingerprint': state.fingerprint, 'seconds': time.monotonic()-before,
                'client_seconds':client_seconds,'validation_seconds':validation_seconds}
            records.append(tio.commit(output, payload, record, plans))
            print(f"round={r} examples={payload['examples_processed']} steps={payload['optimizer_steps']} selected={best_round}", flush=True)
        if payload['round'] < cap:
            return {'status': 'stopped', 'completed_round': payload['round'], 'examples_processed': payload['examples_processed'],
                    'optimizer_steps': payload['optimizer_steps'], 'selected_round': payload['best_round']}
        model.load_state_dict(payload['best_state'], strict=True)
        reloaded_score, reloaded_arrays = predict_and_score(model, validation, device)
        chosen = next(record for record in records if record['round'] == payload['best_round'])['validation']
        if reloaded_score != chosen:
            raise ValueError('Selected validation reload predictions/metrics differ')
        export = tio.export(output, payload, {'validation': chosen,
            'per_node_validation':reloaded_arrays['per_node'],
            'selected_round_receipt_sha256':sha256(output/f"rounds/{payload['best_round']:04d}/receipt.json"),
            'cohort_receipt_sha256': context.receipt_sha256, 'method': method, 'policy': policy, 'seed': seed})
        if not (output / 'completed.json').exists():
            immutable_json(output / 'completed.json', {'status': 'complete', 'completed_round': cap,
                'selected_round': payload['best_round'], 'selected_receipt_sha256': sha256(output / 'selected/receipt.json'),
                'manifest_sha256': sha256(output / 'manifest.json'), 'examples_processed': payload['examples_processed'],
                'last_round_receipt_sha256':sha256(output/f'rounds/{cap:04d}/receipt.json'),
                'optimizer_steps': payload['optimizer_steps'], 'selected_validation': chosen,
                'validation_export_reload_exact':True,
                'resources': {'invocation_wall_seconds': time.monotonic()-started,
                    'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                    'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)},
                'test_inference_executed': False})
        return {'status': 'complete', 'completed_round': cap, 'selected_round': payload['best_round'],
                'examples_processed': payload['examples_processed'], 'optimizer_steps': payload['optimizer_steps']}


def verify_run(plan, context, output):
    m = read_json(output / 'manifest.json')
    if (m['plan_digest'] != tio.digest(plan) or m['sources'] != plan['sources'] or m['settings'] != plan['settings'] or m['software'] != tio.software()
            or m['cohort_receipt_sha256'] != context.receipt_sha256):
        raise ValueError('Recorded source/configuration/software/cohort differs')
    if (m['plan'] != plan or m['method'] not in ALGORITHMS or m['policy'] not in POLICIES or m['seed'] not in SEEDS
            or m['kind'] not in ('main','smoke') or m['cap'] != (3 if m['kind']=='smoke' else 200)):
        raise ValueError('Recorded method/policy/seed/cap/configuration differs')
    tio.open_run(ROOT, output, m, resume=True)
    registry = TrainingRegistry(context, m['policy'])
    payload, records = tio.load_commits(output)
    if payload is None:
        raise ValueError('No complete round checkpoint')
    validate_payload(output, payload, records, registry, initialize(context, m['seed']), m['seed'], m['cap'])
    if (output / 'completed.json').is_file():
        c = read_json(output / 'completed.json')
        receipt = read_json(output / 'selected/receipt.json')
        if (c['completed_round'] != m['cap'] or payload['round'] != m['cap']
                or c['manifest_sha256'] != sha256(output / 'manifest.json')
                or c['selected_receipt_sha256'] != sha256(output / 'selected/receipt.json')
                or receipt['model_sha256'] != sha256(output / 'selected/model.pt')
                or c['last_round_receipt_sha256'] != sha256(output/f"rounds/{m['cap']:04d}/receipt.json")
                or receipt['selected_round_receipt_sha256'] != sha256(output/f"rounds/{payload['best_round']:04d}/receipt.json")
                or receipt['selected_round'] != payload['best_round']):
            raise ValueError('Completed/selected export integrity differs')
        saved = torch.load(output / 'selected/model.pt', map_location='cpu', weights_only=True)['model_state']
        if set(saved) != set(payload['best_state']) or any(not torch.equal(saved[k], payload['best_state'][k]) for k in saved):
            raise ValueError('Selected export state differs from validation-selected state')
    return {'status': 'verified_complete' if (output/'completed.json').exists() else 'verified_partial',
            'completed_round': payload['round'], 'selected_round': payload['best_round'], 'model_forward_or_scoring': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stage', choices=('plan', 'smoke', 'fit', 'verify'), required=True)
    parser.add_argument('--method', choices=ALGORITHMS)
    parser.add_argument('--policy', choices=POLICIES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--matrix', choices=('primary', 'sensitivity', 'all'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-round', type=int)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(Path(args.plan).resolve())
        context = CohortContext(args.data_dir, plan['cohort_receipt_sha256'])
        if context.budgets['protocol'] != {'batch_size':512,'clients_per_round':98,'rounds':200,'seeds':[42,43,44]}:
            raise ValueError('Prepared sampling-budget protocol differs')
        replayed_budgets = verify_budget_totals(context)
        output = output_path(ROOT, args.output_dir)
        if args.stage == 'verify':
            result = verify_run(plan, context, output)
        elif args.stage == 'plan':
            if output.exists():
                raise ValueError('Proposed output already exists')
            result = {'status': 'preflight_passed', 'cohort_receipt_sha256': context.receipt_sha256,
                'matrix_fits': 24, 'budgets': replayed_budgets,
                'shared_validation': context.contract['cohorts']['common/all_policies/val']['identity'],
                'shared_test': context.contract['cohorts']['common/all_policies/test']['identity'],
                'model_forward_optimizer_or_scoring': False}
        else:
            if args.matrix:
                if (args.stage != 'fit' or args.resume or args.stop_after_round is not None
                        or any(v is not None for v in (args.method,args.policy,args.seed))):
                    raise ValueError('Matrix launch cannot resume/stop; use explicit single-fit commands')
                if output.exists():
                    raise ValueError('Matrix root exists; preserve partial runs and resume individual fits')
                policies = ('no_fill', 'ff20') if args.matrix == 'primary' else ('ff10', 'ff30') if args.matrix == 'sensitivity' else POLICIES
                fits = [(method, policy, seed, output / method / f'{policy}_seed{seed}')
                        for method in ALGORITHMS for policy in policies for seed in SEEDS]
                if any(path.exists() for _, _, _, path in fits):
                    raise ValueError('A proposed matrix fit output already exists')
                result = [run_one(ROOT, plan, context, method, policy, seed, 'main', path)
                          for method, policy, seed, path in fits]
            else:
                if args.method is None or args.policy is None or args.seed is None:
                    raise ValueError('Single fit/smoke requires method, policy and seed')
                result = run_one(ROOT, plan, context, args.method, args.policy, args.seed,
                    'smoke' if args.stage == 'smoke' else 'main', output, resume=args.resume,
                    stop_after_round=args.stop_after_round)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('V5 EXPERIMENT ERROR: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
