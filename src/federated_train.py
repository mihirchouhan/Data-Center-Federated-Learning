"""Immutable GRU round snapshots and standard FedAvg; no optimizer or round loop.

Workers must use GlobalModelState.initialize_client() and return capture_update().
Fingerprints reject stale-round updates; they are provenance checks, not proofs
against malicious workers. baseline_coefficients and every frozen parameter/buffer
are fixed. Shared embedding rows use the same sample weights as all parameters.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import torch
from torch import nn


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 or value > 2**53:
        raise ValueError(f'{name} must be a positive integer no larger than 2**53')
    return int(value)


def _tensor_bytes(tensor):
    return tensor.reshape(-1).contiguous().view(torch.uint8)


def _copy_state(state):
    return OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in state.items())


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    num_examples: int
    base_fingerprint: str
    state: Mapping[str, torch.Tensor]


class GlobalModelState:
    """Coordinator-owned detached snapshot; all public tensor access returns copies."""
    def __init__(self, model: nn.Module, *, round_id=0):
        if isinstance(round_id, bool) or not isinstance(round_id, Integral) or round_id < 0:
            raise ValueError('round_id must be a nonnegative integer')
        self.__round_id = int(round_id)
        if not isinstance(model, nn.Module) or not any(isinstance(module, nn.GRU) for module in model.modules()):
            raise ValueError('Expected a GRU model')
        raw_state = model.state_dict()
        if 'baseline_coefficients' not in raw_state:
            raise ValueError('The GRU must include frozen persistence baseline_coefficients')
        parameters = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if 'baseline_coefficients' in parameters:
            raise ValueError('baseline_coefficients must be fixed, not trainable')
        if not parameters or not parameters.issubset(raw_state):
            raise ValueError('Invalid trainable parameter mapping')
        self.__keys = tuple(raw_state)
        self.__trainable_keys = tuple(name for name in self.__keys if name in parameters)
        self.__fixed_keys = tuple(name for name in self.__keys if name not in parameters)
        self.__model_type = type(model)
        for name, tensor in raw_state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or tensor.is_quantized:
                raise ValueError(f'{name}: state must be a dense unquantized tensor')
            if tensor.is_complex() or not torch.isfinite(tensor).all().item():
                raise ValueError(f'{name}: state must be finite and real')
            if name in parameters and not tensor.dtype.is_floating_point:
                raise ValueError(f'{name}: trainable parameters must be floating point')
        coefficients = raw_state['baseline_coefficients']
        if coefficients.shape != (2,) or not coefficients.dtype.is_floating_point:
            raise ValueError('baseline_coefficients must contain two floating-point constants')
        self.__state = _copy_state(raw_state)
        fingerprint = hashlib.sha256()
        fingerprint.update(json.dumps({'round_id': self.__round_id,
                                       'model_type': type(model).__module__ + '.' + type(model).__qualname__}).encode())
        for name, tensor in self.__state.items():
            fingerprint.update(json.dumps([name, str(tensor.dtype), list(tensor.shape), name in parameters]).encode())
            fingerprint.update(_tensor_bytes(tensor).numpy().tobytes())
        self.__fingerprint = fingerprint.hexdigest()

    @property
    def round_id(self):
        return self.__round_id

    @property
    def fingerprint(self):
        return self.__fingerprint

    @property
    def trainable_keys(self):
        return self.__trainable_keys

    @property
    def fixed_keys(self):
        return self.__fixed_keys

    def state_dict(self):
        return _copy_state(self.__state)

    def _validate(self, state, *, label, compare_fixed=True):
        if not isinstance(state, Mapping) or set(state) != set(self.__keys):
            raise ValueError(f'{label}: parameter/buffer names differ from global state')
        checked = OrderedDict()
        for name in self.__keys:
            value = state[name]
            expected = self.__state[name]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.is_quantized:
                raise ValueError(f'{label}: {name} must be a dense tensor')
            if value.shape != expected.shape:
                raise ValueError(f'{label}: {name} shape differs from global state')
            if value.dtype != expected.dtype:
                raise ValueError(f'{label}: {name} dtype differs from global state')
            if not torch.isfinite(value).all().item():
                raise ValueError(f'{label}: {name} contains nonfinite values')
            copied = value.detach().cpu().clone()
            if compare_fixed and name in self.__fixed_keys and not torch.equal(_tensor_bytes(copied), _tensor_bytes(expected)):
                raise ValueError(f'{label}: fixed state changed: {name}')
            checked[name] = copied
        return checked

    def initialize_client(self, template: nn.Module):
        """Return an independent worker model, overwritten with the same round state."""
        if type(template) is not self.__model_type:
            raise ValueError('Client template type differs from the global model')
        trainable = {name for name, parameter in template.named_parameters() if parameter.requires_grad}
        if trainable != set(self.__trainable_keys):
            raise ValueError('Client template trainable/fixed roles differ from global model')
        self._validate(template.state_dict(), label='template', compare_fixed=False)
        client = copy.deepcopy(template)
        client.load_state_dict(self.state_dict(), strict=True)
        for parameter in client.parameters():
            parameter.grad = None
        client.train()
        client._fedavg_round_fingerprint = self.fingerprint
        return client

    def capture_update(self, client_id, client_model, num_examples):
        """Detach a legitimate client's state, tagging the round it started from."""
        if not isinstance(client_id, str) or not client_id:
            raise ValueError('Client ID must be a nonempty string')
        count = _count(num_examples, 'num_examples')
        if getattr(client_model, '_fedavg_round_fingerprint', None) != self.fingerprint:
            raise ValueError('Client must be initialized from this immutable global round')
        if type(client_model) is not self.__model_type:
            raise ValueError('Updated model type differs from global model')
        roles = {name for name, parameter in client_model.named_parameters() if parameter.requires_grad}
        if roles != set(self.__trainable_keys):
            raise ValueError('Updated model trainable/fixed roles changed')
        state = self._validate(client_model.state_dict(), label=client_id)
        return ClientUpdate(client_id, count, self.fingerprint, state)


def aggregate_fedavg(global_state: GlobalModelState, updates: Sequence[ClientUpdate], *, expected_counts: Mapping[str, int]):
    """Return a fresh CPU state dict, averaging participating clients by sample count.

    expected_counts comes from the coordinator's frozen training partition, never
    validation sizes or worker-reported epoch counts. No clients are dropped on
    failure. IDs are sorted before float64 accumulation, making the result stable
    for repeated calls and worker completion order. This includes the full shared
    embedding table: only a client's changed rows contribute nonzero deltas.
    """
    if not isinstance(global_state, GlobalModelState):
        raise ValueError('A detached GlobalModelState is required')
    if not isinstance(expected_counts, Mapping) or not expected_counts:
        raise ValueError('Expected frozen client training counts are required')
    counts = {}
    for client_id, value in expected_counts.items():
        if not isinstance(client_id, str) or not client_id:
            raise ValueError('Expected client IDs must be nonempty strings')
        counts[client_id] = _count(value, f'expected_counts[{client_id!r}]')
    updates = list(updates)
    if not updates:
        raise ValueError('At least one client update is required')
    checked = {}
    for update in updates:
        if not isinstance(update, ClientUpdate) or not isinstance(update.client_id, str) or not update.client_id:
            raise ValueError('Every update must have a nonempty client ID')
        client_id = update.client_id
        if client_id in checked:
            raise ValueError(f'Duplicate client update: {client_id}')
        count = _count(update.num_examples, f'{client_id}: num_examples')
        if client_id not in counts or counts[client_id] != count:
            raise ValueError(f'{client_id}: training count differs from frozen membership')
        if update.base_fingerprint != global_state.fingerprint:
            raise ValueError(f'{client_id}: update belongs to a different global round')
        checked[client_id] = (count, global_state._validate(update.state, label=client_id))
    ordered = [checked[client_id] for client_id in sorted(checked)]
    if len(ordered) == 1:
        # Exact identity, including very different floating-point magnitudes.
        return _copy_state(ordered[0][1])
    total = sum(count for count, _ in ordered)
    base = global_state.state_dict()
    result = OrderedDict()
    for name, original in base.items():
        if name in global_state.fixed_keys:
            result[name] = original.clone()
            continue
        averaged = torch.zeros_like(original, dtype=torch.float64, device='cpu')
        unchanged = torch.ones_like(original, dtype=torch.bool, device='cpu')
        for count, state in ordered:
            tensor = state[name]
            averaged.add_(tensor.to(torch.float64), alpha=count / total)
            unchanged &= tensor == original
        value = averaged.to(original.dtype)
        if not torch.isfinite(value).all().item():
            raise ValueError(f'Nonfinite aggregate for {name}')
        # Unchanged embedding rows/parameters retain original bits, including -0.
        result[name] = torch.where(unchanged, original, value)
    return result
