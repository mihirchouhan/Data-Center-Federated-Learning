"""CPU tensor-state FedAvg and scripted rounds; no model or optimizer execution.

Local SGD is described, never executed. Scripted states are invented numerical
fixtures, not trained parameters. No datasets, checkpoints or scorers are used.
"""
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np
import torch

from src.b1_model_adapter_v1 import mapping_digest


def _int(value, label, lower=1, upper=2**53):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(label + ': invalid integer')
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   allow_nan=False).encode()).hexdigest()


def _bytes(value):
    return value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()


def _tensor(value, label):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.layout != torch.strided:
        raise ValueError(label + ': dense CPU tensor required')
    if value.is_complex() or value.is_quantized or value.requires_grad:
        raise ValueError(label + ': detached real unquantized state required')
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                           torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool):
        raise ValueError(label + ': unsupported state dtype')
    if not torch.isfinite(value).all().item():
        raise ValueError(label + ': nonfinite state')
    return value.detach().clone()


class RoundSnapshot:
    """Detached coordinator state; all public state access returns fresh copies.

    Trainable roles must be supplied explicitly from a future approved bridge.
    Nonfloating entries and all nontrainable entries are fixed byte-for-byte.
    The B1 target_mean/target_std buffers cannot be trainable.
    """
    __slots__ = ('__state', '__round_id', '__trainable_keys', '__fixed_keys', '__fingerprint')

    def __init__(self, state, trainable_keys, *, round_id=0):
        self.__round_id = _int(round_id, 'round_id', 0, 200)
        if not isinstance(state, Mapping) or not state:
            raise ValueError('Nonempty state mapping required')
        self.__state = OrderedDict()
        for name, value in state.items():
            if not isinstance(name, str) or not name:
                raise ValueError('State names must be nonempty strings')
            self.__state[name] = _tensor(value, name)
        roles = tuple(trainable_keys)
        if not roles or len(roles) != len(set(roles)) or not set(roles) <= set(state):
            raise ValueError('Explicit unique trainable keys required')
        if {'target_mean', 'target_std'} & set(roles):
            raise ValueError('B1 target scaling buffers must remain fixed')
        self.__trainable_keys = tuple(k for k in self.__state if k in roles)
        self.__fixed_keys = tuple(k for k in self.__state if k not in roles)
        if any(not self.__state[k].dtype.is_floating_point for k in roles):
            raise ValueError('Nonfloating state cannot be trainable or averaged')
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps({'schema': 'b1.cpu.round.state.v1', 'round': self.round_id}).encode())
        for name, value in self.__state.items():
            fingerprint.update(json.dumps([name, str(value.dtype), list(value.shape), name in roles]).encode())
            fingerprint.update(_bytes(value))
        self.__fingerprint = fingerprint.hexdigest()

    @property
    def round_id(self):
        return self.__round_id

    @property
    def trainable_keys(self):
        return self.__trainable_keys

    @property
    def fixed_keys(self):
        return self.__fixed_keys

    @property
    def fingerprint(self):
        return self.__fingerprint

    def state_dict(self):
        return OrderedDict((k, v.clone()) for k, v in self.__state.items())

    def validate(self, state):
        if not isinstance(state, Mapping) or set(state) != set(self.__state):
            raise ValueError('State keys differ from round snapshot')
        checked = OrderedDict()
        for name, original in self.__state.items():
            value = _tensor(state[name], name)
            if value.shape != original.shape or value.dtype != original.dtype:
                raise ValueError(name + ': state shape/dtype changed')
            if name in self.fixed_keys and _bytes(value) != _bytes(original):
                raise ValueError(name + ': fixed state changed')
            checked[name] = value
        return checked


@dataclass(frozen=True)
class ClientStateUpdate:
    client_id: str
    num_examples: int
    base_fingerprint: str
    state: Mapping


def aggregate_fedavg(snapshot, updates, *, expected_counts, client_order):
    """Full-table sample-weighted average in explicit order using CPU float64.

    Require every expected participant, never renormalize a surviving subset.
    Fixed floats, integers and booleans retain their exact starting bytes.
    """
    if not isinstance(snapshot, RoundSnapshot) or not isinstance(expected_counts, Mapping) or not expected_counts:
        raise ValueError('Snapshot and expected counts required')
    counts = {}
    for name, count in expected_counts.items():
        if not isinstance(name, str) or not name:
            raise ValueError('Nonempty client names required')
        counts[name] = _int(count, 'training count')
    order = tuple(client_order)
    if len(order) != len(set(order)) or set(order) != set(counts):
        raise ValueError('Order must contain every expected client exactly once')
    _int(sum(counts.values()), 'total training count')
    checked = {}
    for update in updates:
        if not isinstance(update, ClientStateUpdate) or update.client_id not in counts:
            raise ValueError('Unexpected or invalid client update')
        if update.client_id in checked:
            raise ValueError('Duplicate client update')
        if _int(update.num_examples, 'reported count') != counts[update.client_id]:
            raise ValueError('Count differs from coordinator membership')
        if update.base_fingerprint != snapshot.fingerprint:
            raise ValueError('Stale or incompatible round update')
        checked[update.client_id] = snapshot.validate(update.state)
    if set(checked) != set(counts):
        raise ValueError('Missing client updates; round aborted')
    if len(order) == 1:
        return OrderedDict((k, v.clone()) for k, v in checked[order[0]].items())
    total = sum(counts.values())
    result = snapshot.state_dict()
    for name in snapshot.trainable_keys:
        original = result[name]
        accumulator = torch.zeros_like(original, dtype=torch.float64, device='cpu')
        unchanged = torch.ones_like(original, dtype=torch.bool, device='cpu')
        for client in order:
            value = checked[client][name]
            accumulator.add_(value.to(torch.float64), alpha=counts[client] / total)
            unchanged &= value == original
        averaged = accumulator.to(original.dtype)
        if not torch.isfinite(averaged).all().item():
            raise ValueError('Nonfinite aggregate; round aborted')
        result[name] = torch.where(unchanged, original, averaged)
    return snapshot.validate(result)


@dataclass(frozen=True)
class CandidateConfig:
    clients: int = 979
    clients_per_round: int = 98
    rounds: int = 200
    seeds: tuple = (42, 43, 44)
    batch_size: int = 128
    local_epochs: int = 1
    drop_last: bool = False

    def __post_init__(self):
        expected = (979, 98, 200, 128, 1)
        values = (self.clients, self.clients_per_round, self.rounds, self.batch_size, self.local_epochs)
        if (any(type(v) is not int for v in values) or values != expected or
                type(self.seeds) is not tuple or any(type(v) is not int for v in self.seeds) or
                self.seeds != (42, 43, 44) or self.drop_last is not False):
            raise ValueError('Only the recorded candidate protocol is supported')


class ClientRegistry:
    """Supplied training counts only; no dataset/validation/test capability.

    Counts do not prove eligibility, observed targets or real split provenance.
    No real membership is inferred here; tests use invented names and counts.
    """
    def __init__(self, node_names, training_counts, *, expected_mapping_digest):
        names = tuple(node_names)
        if mapping_digest(names) != expected_mapping_digest:
            raise ValueError('Original ordered mapping digest differs')
        if not isinstance(training_counts, Mapping) or set(training_counts) != set(names):
            raise ValueError('Exactly 979 supplied training counts required')
        self.node_names = names
        self.mapping_digest = expected_mapping_digest
        self.__counts = tuple(_int(training_counts[n], 'training count') for n in names)
        self.fingerprint = _digest({'schema': 'b1.training.counts.v1', 'names': names, 'counts': self.__counts})

    def count(self, node_index):
        return self.__counts[_int(node_index, 'node index', 0, 978)]


def sample_clients(registry, seed, round_id):
    if not isinstance(registry, ClientRegistry):
        raise ValueError('Training-count registry required')
    seed = _int(seed, 'seed', 0, 2**32 - 1)
    if seed not in (42, 43, 44):
        raise ValueError('Seed must be 42,43,44')
    round_id = _int(round_id, 'round', 1, 200)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 1, round_id, 0])))
    return tuple(int(i) for i in rng.choice(979, size=98, replace=False))


def batch_sizes(count, batch_size=128):
    count = _int(count, 'training count')
    batch_size = _int(batch_size, 'batch size')
    full, remainder = divmod(count, batch_size)
    return (batch_size,) * full + ((remainder,) if remainder else ())


@dataclass(frozen=True)
class LocalSessionPlan:
    client_id: str
    node_index: int
    training_windows: int
    batch_sizes: tuple
    shuffle_seed: int
    session_id: str
    round_id: int
    base_fingerprint: str

    @property
    def planned_optimizer_steps(self):
        return len(self.batch_sizes)

    @property
    def optimizer_spec(self):
        # A new descriptor on every access; no optimizer is constructed or shared.
        return {'algorithm': 'SGD', 'learning_rate': .01, 'momentum': 0.,
                'weight_decay': 0., 'fresh_each_client_round': True,
                'initial_state_entries': 0, 'execution_implemented': False}


@dataclass(frozen=True)
class RoundPlan:
    seed: int
    round_id: int
    mapping_digest: str
    registry_fingerprint: str
    base_fingerprint: str
    sessions: tuple

    @property
    def fingerprint(self):
        return _digest({'seed': self.seed, 'round': self.round_id, 'registry': self.registry_fingerprint,
                        'mapping': self.mapping_digest, 'base': self.base_fingerprint,
                        'sessions': [(s.session_id, s.training_windows, s.batch_sizes, s.shuffle_seed) for s in self.sessions]})


def plan_round(snapshot, registry, seed, round_id, config=CandidateConfig()):
    if not isinstance(snapshot, RoundSnapshot) or not isinstance(config, CandidateConfig):
        raise ValueError('Candidate config and round snapshot required')
    if snapshot.round_id != round_id - 1:
        raise ValueError('Round must follow the snapshot exactly')
    positions = sample_clients(registry, seed, round_id)
    sessions = []
    for i in positions:
        count = registry.count(i)
        shuffle = int(np.random.SeedSequence([seed, 2, round_id, i]).generate_state(1)[0])
        identifier = _digest([seed, round_id, i, registry.fingerprint, snapshot.fingerprint])
        sessions.append(LocalSessionPlan(registry.node_names[i], i, count,
                                        batch_sizes(count, config.batch_size), shuffle,
                                        identifier, round_id, snapshot.fingerprint))
    return RoundPlan(seed, round_id, registry.mapping_digest, registry.fingerprint,
                     snapshot.fingerprint, tuple(sessions))


def local_row_order(session):
    """Deterministic full local permutation; no repeats, padding or dropped rows."""
    if not isinstance(session, LocalSessionPlan):
        raise ValueError('Local session plan required')
    rng = np.random.Generator(np.random.PCG64(session.shuffle_seed))
    order = rng.permutation(session.training_windows)
    return np.frombuffer(order.tobytes(), dtype=order.dtype)


@dataclass(frozen=True)
class ScriptedUpdate:
    update: ClientStateUpdate
    session_id: str
    planned_examples: int
    planned_steps: int
    executed_optimizer_steps: int = 0


def scripted_update(session, state):
    """Tag an invented state; does NOT certify real examples or SGD execution."""
    return ScriptedUpdate(ClientStateUpdate(session.client_id, session.training_windows,
                                           session.base_fingerprint, state),
                          session.session_id, session.training_windows, session.planned_optimizer_steps)


def commit_scripted_round(snapshot, registry, plan, scripted_updates):
    if not isinstance(plan, RoundPlan):
        raise ValueError('Round plan required')
    expected = plan_round(snapshot, registry, plan.seed, plan.round_id)
    if plan != expected:
        raise ValueError('Round plan changed')
    items = tuple(scripted_updates)
    by_id = {}
    for item in items:
        if not isinstance(item, ScriptedUpdate) or not isinstance(item.update, ClientStateUpdate):
            raise ValueError('Only scripted client updates are supported')
        if item.update.client_id in by_id:
            raise ValueError('Duplicate session update')
        by_id[item.update.client_id] = item
    if set(by_id) != {s.client_id for s in plan.sessions}:
        raise ValueError('Missing or unexpected participants; no replacement allowed')
    for session in plan.sessions:
        item = by_id[session.client_id]
        if (item.session_id != session.session_id or
                _int(item.planned_examples, 'planned examples') != session.training_windows or
                _int(item.planned_steps, 'planned steps') != session.planned_optimizer_steps or
                type(item.executed_optimizer_steps) is not int or item.executed_optimizer_steps != 0):
            raise ValueError('Session/count/step mismatch; synthetic mode requires zero optimizer steps')
    counts = {s.client_id: s.training_windows for s in plan.sessions}
    order = tuple(counts)
    state = aggregate_fedavg(snapshot, [by_id[n].update for n in order],
                             expected_counts=counts, client_order=order)
    next_snapshot = RoundSnapshot(state, snapshot.trainable_keys, round_id=plan.round_id)
    total = sum(counts.values())
    receipt = {'schema': 'b1.scripted.round.v1', 'seed': plan.seed, 'round': plan.round_id,
               'mode': 'synthetic_scripted_states_not_local_training',
               'plan_fingerprint': plan.fingerprint, 'base_fingerprint': snapshot.fingerprint,
               'result_fingerprint': next_snapshot.fingerprint, 'mapping_digest': plan.mapping_digest,
               'registry_fingerprint': registry.fingerprint,
               'planned_examples': total,
               'planned_optimizer_steps': sum(s.planned_optimizer_steps for s in plan.sessions),
               'executed_optimizer_steps': 0, 'actual_examples_trained': 0,
               'optimizer_instances_created': 0,
               'clients': [dict(client_id=s.client_id, node_index=s.node_index, session_id=s.session_id,
                                training_windows=s.training_windows, batch_sizes=s.batch_sizes,
                                planned_steps=s.planned_optimizer_steps, shuffle_seed=s.shuffle_seed,
                                weight=s.training_windows / total, optimizer=s.optimizer_spec)
                           for s in plan.sessions]}
    return next_snapshot, receipt


def run_scripted_rounds(initial, registry, *, seed, round_count, update_factory):
    """Sequential in-memory scripted rounds only; no checkpoint/resume or models.

    The callable gets one independent state-dict copy per session. It must return
    an invented tensor-state fixture. This API is not an OS/security boundary.
    The runner has no local trainer, optimizer, data loader or validation hook.
    """
    if not isinstance(initial, RoundSnapshot) or initial.round_id != 0:
        raise ValueError('Fresh round-zero snapshot required; checkpoint resume is unsupported')
    round_count = _int(round_count, 'round_count', 1, 200)
    if not callable(update_factory):
        raise ValueError('Synthetic update factory required')
    current, receipts = initial, []
    for round_id in range(1, round_count + 1):
        plan = plan_round(current, registry, seed, round_id)
        updates = [scripted_update(session, update_factory(session, current.state_dict()))
                   for session in plan.sessions]
        current, record = commit_scripted_round(current, registry, plan, updates)
        receipts.append(record)
    return current, tuple(receipts)


def participation_summary(registry, seed, rounds=200):
    """Plan accounting, not an executed-training participation receipt."""
    rounds = _int(rounds, 'rounds', 1, 200)
    counts = [0] * 979
    for r in range(1, rounds + 1):
        for i in sample_clients(registry, seed, r):
            counts[i] += 1
    return {'seed': seed, 'rounds': rounds, 'planned_client_sessions': rounds * 98,
            'expected_participations_per_client': rounds * 98 / 979,
            'participations_by_node_index': tuple(counts),
            'planned_examples': sum(counts[i] * registry.count(i) for i in range(979)),
            'planned_optimizer_steps': sum(counts[i] * len(batch_sizes(registry.count(i))) for i in range(979)),
            'actual_examples_trained': 0, 'executed_optimizer_steps': 0}
