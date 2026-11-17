"""Authorized CPU-only local SGD on bounded invented fixtures.

No real-data loader, checkpoint API, evaluation, disk output or fit CLI. Models
must use this module's invented node names. At most four clients per test round.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from numbers import Integral

import numpy as np
import torch

from src.b1_model_adapter_v1 import B1TwoOutputAdapter, mapping_digest
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_fedavg_orchestration_v1 import (
    RoundSnapshot, ClientStateUpdate, LocalSessionPlan, batch_sizes,
    local_row_order, aggregate_fedavg)


SYNTHETIC_NAMES = tuple('synthetic-local-node-%04d' % i for i in range(979))


def _integer(value, label, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(label + ': integer outside synthetic bounds')
    return int(value)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class SyntheticClient:
    """In-memory invented fixture only; not a certificate of real-data provenance."""
    node_index: int
    inputs: torch.Tensor
    future: object
    observed: object

    @property
    def client_id(self):
        return SYNTHETIC_NAMES[_integer(self.node_index, 'node index', 0, 978)]


def make_synthetic_client(node_index, sample_count, *, fixture_seed=42):
    node_index = _integer(node_index, 'node index', 0, 978)
    count = _integer(sample_count, 'sample count', 1, 512)
    seed = _integer(fixture_seed, 'fixture seed', 0, 2**32 - 1)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 3, node_index])))
    x = rng.normal(0, .2, (count, 36, 14)).astype(np.float32)
    x[:, :, 13] = node_index
    future = np.empty((count, 6, 2), dtype=np.float64)
    future[:, :, 0] = .41 + rng.uniform(0, .02, (count, 6))
    future[:, :, 1] = 24. + rng.uniform(0, 2., (count, 6))
    return SyntheticClient(node_index, torch.tensor(x, dtype=torch.float32, device='cpu'),
                           future, np.ones(future.shape, dtype=bool))


def make_synthetic_model(*, target_scaling, seed=42):
    if not isinstance(target_scaling, TargetScaling):
        raise ValueError('Explicit supplied synthetic TargetScaling required')
    return B1TwoOutputAdapter(node_names=SYNTHETIC_NAMES,
        expected_mapping_digest=mapping_digest(SYNTHETIC_NAMES),
        target_scaling=target_scaling, seed=seed)


def _validate_model(model):
    if type(model) is not B1TwoOutputAdapter or model.node_names != SYNTHETIC_NAMES:
        raise ValueError('Only the dedicated invented-node model is accepted')
    model._validate_state()
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Synthetic candidate parameters must all remain trainable')


def _validate_client(model, client):
    if not isinstance(client, SyntheticClient):
        raise ValueError('SyntheticClient fixture required; no real-data capability')
    node = _integer(client.node_index, 'node index', 0, 978)
    # Preserve original containers until hardened target validation has rejected
    # masked/boolean/incomplete/nonfinite labels. No mask-erasing conversion first.
    physical = model.observed_targets(client.future, client.observed)
    scaled = model.target_scaling.standardize(physical)
    with np.errstate(over='raise', invalid='raise'):
        target32 = np.asarray(scaled, dtype=np.float32)
    if not np.isfinite(target32).all():
        raise ValueError('Target scaling is not representable in float32')
    x = client.inputs
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.dtype != torch.float32 or x.requires_grad:
        raise ValueError('Detached CPU float32 fixture inputs required')
    if x.ndim != 3 or tuple(x.shape[1:]) != (36, 14) or not 1 <= len(x) <= 512:
        raise ValueError('Synthetic input shape/count differs from [1..512,36,14]')
    if not torch.isfinite(x).all().item() or not torch.all(x[:, :, 13] == node).item():
        raise ValueError('Nonfinite inputs or another/fractional node identity')
    if len(physical) != len(x) or model.node_index(client.client_id) != node:
        raise ValueError('Synthetic target count or mapping differs')
    return len(x)


def _snapshot(model, round_id):
    return RoundSnapshot(model.state_dict(), tuple(n for n,p in model.named_parameters() if p.requires_grad),
                         round_id=round_id)


def _new_optimizer(parameters):
    return torch.optim.SGD(parameters, lr=.01, momentum=0., weight_decay=0.)


def _validate_optimizer(optimizer, local):
    if type(optimizer) is not torch.optim.SGD or optimizer.state or len(optimizer.param_groups) != 1:
        raise ValueError('Fresh empty-state SGD required per client/round')
    group = optimizer.param_groups[0]
    if tuple(map(id,group['params'])) != tuple(map(id,local.parameters())):
        raise ValueError('Optimizer parameter ownership differs: sharing/reuse rejected')
    if (group['lr'], group['momentum'], group['weight_decay'], group['dampening'],
            group['nesterov'], group['maximize'], group['differentiable']) != (.01, 0., 0., 0., False, False, False):
        raise ValueError('Optimizer settings differ from candidate SGD')


@dataclass(frozen=True)
class LocalSGDResult:
    update: ClientStateUpdate
    receipt: dict


def train_synthetic_client(global_model, client, *, seed=42, round_id=1):
    """One actual local SGD epoch; the global model and fixture remain untouched.

    Caller authorization is a workflow boundary, not granted by this function.
    This version is exclusively for bounded, invented CPU fixtures.
    """
    _validate_model(global_model)
    seed = _integer(seed, 'seed', 42, 44)
    round_id = _integer(round_id, 'round', 1, 200)
    count = _validate_client(global_model, client)  # before optimizer construction
    base = _snapshot(global_model, round_id - 1)
    shuffle_seed = int(np.random.SeedSequence([seed, 2, round_id, client.node_index]).generate_state(1)[0])
    session = LocalSessionPlan(client.client_id, client.node_index, count, batch_sizes(count),
        shuffle_seed, _hash([seed,round_id,client.node_index,base.fingerprint]), round_id, base.fingerprint)
    # Copy a live synthetic object, never serialize/load state or relax the
    # existing adapter's disabled load_state_dict API.
    local = copy.deepcopy(global_model)
    local.zero_grad(set_to_none=True)
    local.train()
    if any(a is b or a.data_ptr() == b.data_ptr() for a,b in zip(local.parameters(),global_model.parameters())):
        raise ValueError('Local model must own independent parameter storage')
    if _snapshot(local, round_id - 1).fingerprint != base.fingerprint:
        raise ValueError('Local initialization differs from immutable round start')
    optimizer = _new_optimizer(local.parameters())
    _validate_optimizer(optimizer, local)
    order = local_row_order(session)
    # Containers have already passed the original mask/dtype checks above.
    future_values = np.asarray(client.future)
    observed_values = np.asarray(client.observed)
    batches, losses, gradients, step_changes = [], [], [], []
    seen = []
    for first in range(0, count, 128):
        rows = np.array(order[first:first+128], copy=True)
        index = torch.tensor(rows, dtype=torch.int64, device='cpu')
        x = client.inputs.index_select(0, index)
        future, observed = future_values[rows], observed_values[rows]
        optimizer.zero_grad(set_to_none=True)
        loss = local.loss(x, future, observed)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise ValueError('Nonfinite/non-scalar synthetic training loss')
        loss.backward()
        for name,parameter in local.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                raise ValueError('Missing/nonfinite local gradient: '+name)
        embedding_gradient = local.core.node_embedding.weight.grad
        other_rows = torch.arange(979, device='cpu') != client.node_index
        if torch.count_nonzero(embedding_gradient[other_rows]).item():
            raise ValueError('Unexpected cross-client embedding gradient')
        gradient_norm = sum(p.grad.to(torch.float64).square().sum().item() for p in local.parameters())**.5
        if not np.isfinite(gradient_norm):
            raise ValueError('Nonfinite local gradient norm')
        before_step = tuple(p.detach().clone() for p in local.parameters())
        optimizer.step()
        local._validate_state()
        if optimizer.state:
            raise ValueError('Stateless candidate SGD unexpectedly retained state')
        step_changes.append(any(not torch.equal(p,b) for p,b in zip(local.parameters(),before_step)))
        gradients.append({'all_finite':True,'l2':gradient_norm,
                          'owner_embedding_l2':embedding_gradient[client.node_index].to(torch.float64).norm().item(),
                          'other_embedding_nonzero':0})
        losses.append(loss.detach().item());batches.append(len(rows));seen.extend(rows.tolist())
    if tuple(batches) != session.batch_sizes or len(seen) != count or sorted(seen) != list(range(count)):
        raise ValueError('Local epoch example/step/row accounting differs')
    state = base.validate(local.state_dict())
    initial = base.state_dict()
    owner_other = torch.arange(979, device='cpu') != client.node_index
    if not torch.equal(state['core.node_embedding.weight'][owner_other], initial['core.node_embedding.weight'][owner_other]):
        raise ValueError('Non-owner embedding rows changed')
    changed = tuple(n for n in base.trainable_keys if not torch.equal(state[n],initial[n]))
    delta_l2 = sum((state[n].to(torch.float64)-initial[n]).square().sum().item() for n in base.trainable_keys)**.5
    if _snapshot(global_model, round_id - 1).fingerprint != base.fingerprint:
        raise ValueError('Global model mutated during local training')
    receipt = {'schema':'b1.synthetic.local.sgd.v1','mode':'invented_CPU_actual_SGD',
        'client_id':client.client_id,'node_index':client.node_index,'seed':seed,'round':round_id,
        'session_id':session.session_id,'base_fingerprint':base.fingerprint,
        'mapping_digest':global_model.mapping_digest,'target_scaling_digest':global_model.target_scaling.digest,
        'optimizer':dict(session.optimizer_spec,execution_implemented=True),
        'optimizer_initial_state_entries':0,'optimizer_final_state_entries':len(optimizer.state),
        'local_epochs':1,'batch_size':128,'drop_last':False,'actual_batch_sizes':tuple(batches),
        'training_windows':count,'examples_processed':len(seen),'optimizer_steps':len(batches),
        'shuffle_seed':shuffle_seed,'row_order_sha256':hashlib.sha256(order.tobytes()).hexdigest(),
        'finite_batch_losses':tuple(losses),'gradient_checks':tuple(gradients),
        'parameter_changed_per_step':tuple(step_changes),'changed_parameter_names':changed,
        'parameter_delta_l2':delta_l2,'global_model_unchanged':True,
        'real_training_authorized':False,'real_data_used':False,'checkpoint_loaded':False}
    return LocalSGDResult(ClientStateUpdate(client.client_id,len(seen),base.fingerprint,state),receipt)


def train_synthetic_round(global_model, clients, *, seed=42, round_id=1):
    """Train 1..4 invented clients from one unchanged live model, then aggregate.

    Returns tensors and receipts only. Does not assign the aggregate into a
    model, create checkpoints, select/evaluate models or run outer rounds.
    """
    _validate_model(global_model)
    seed = _integer(seed, 'seed', 42, 44)
    round_id = _integer(round_id, 'round', 1, 200)
    clients = tuple(clients)
    if (not 1 <= len(clients) <= 4 or any(not isinstance(c,SyntheticClient) for c in clients) or
            len({c.client_id for c in clients}) != len(clients)):
        raise ValueError('Require 1..4 unique invented clients, never the real cohort')
    counts = {c.client_id:_validate_client(global_model,c) for c in clients}
    base = _snapshot(global_model,round_id-1)
    local = tuple(train_synthetic_client(global_model,c,seed=seed,round_id=round_id) for c in clients)
    result = aggregate_fedavg(base,[r.update for r in local],expected_counts=counts,client_order=tuple(counts))
    if _snapshot(global_model,round_id-1).fingerprint != base.fingerprint:
        raise ValueError('Global model changed before aggregation commit')
    return {'state':result,'local_results':local,
            'receipt':{'schema':'b1.synthetic.local.fedavg.round.v1','seed':seed,'round':round_id,
                       'clients':tuple(counts),'base_fingerprint':base.fingerprint,
                       'examples_processed':sum(r.receipt['examples_processed'] for r in local),
                       'optimizer_steps':sum(r.receipt['optimizer_steps'] for r in local),
                       'weights':{n:v/sum(counts.values()) for n,v in counts.items()},
                       'aggregation':'standard_full_table_sample_weighted_CPU_float64',
                       'aggregate_assigned_to_model':False,'real_training':False}}
