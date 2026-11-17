"""Isolated CPU B1 GRU adapter; no loader, fitter, optimizer or checkpoint API."""
import hashlib
import json

import numpy as np
import torch
from torch import nn

from src.model import SequenceRegressor
from src import b1_target_validation_v1 as target_validation
from src.b1_target_scaling_v1 import TargetScaling, TARGET_NAMES, TARGET_UNITS


def mapping_digest(names):
    """Canonical ordered-name fingerprint, distinct from a raw JSON-file hash."""
    if isinstance(names, (str, bytes)):
        raise ValueError('Explicit ordered node-name sequence required')
    names = tuple(names)
    if len(names) != 979 or len(set(names)) != 979 or any(
            not isinstance(n, str) or not n for n in names):
        raise ValueError('Exactly 979 unique literal node strings required')
    raw = json.dumps({'schema': 'b1.ordered.node.mapping.v1', 'names': names},
                     sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _tensor(value, label, shape_tail):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32:
        raise ValueError(label + ' must be a CPU float32 tensor')
    if value.ndim != len(shape_tail) + 1 or not len(value) or tuple(value.shape[1:]) != shape_tail:
        raise ValueError(label + ' shape differs from the B1 contract')
    if not torch.isfinite(value).all().item():
        raise ValueError(label + ' must be finite')
    return value


def standardized_mse(predicted, target):
    """Equal-output mean MSE, rejecting broadcasting or missing outputs."""
    _tensor(predicted, 'predicted standardized targets', (2,))
    _tensor(target, 'standardized targets', (2,))
    if predicted.shape != target.shape:
        raise ValueError('Prediction and target shapes must match exactly')
    loss = ((predicted - target) ** 2).mean()
    if not torch.isfinite(loss).item():
        raise ValueError('Nonfinite standardized loss')
    return loss


class B1TwoOutputAdapter(nn.Module):
    """Public forward: [B,2] W/degC. Core head/loss: standardized [B,2].

    Inputs must already have the approved numerical preprocessing; this adapter
    neither fits nor supplies input scalers, constructs windows, or opens data.
    Only CPU operation is supported by this isolated version. A caller-supplied
    exact node mapping is mandatory and never sorted or renumbered.
    """
    def __init__(self, *, node_names, expected_mapping_digest, target_scaling, seed=42):
        super().__init__()
        names = tuple(node_names) if not isinstance(node_names, (str, bytes)) else node_names
        digest = mapping_digest(names)
        if digest != expected_mapping_digest:
            raise ValueError('Ordered node mapping fingerprint mismatch')
        if not isinstance(target_scaling, TargetScaling):
            raise ValueError('Explicit two-output TargetScaling required')
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Integer seed in [0,2**32) required')
        self.node_names = names
        self.mapping_digest = digest
        self.target_scaling = target_scaling
        self.seed = seed
        # Installed fork_rng probes the current accelerator even with devices=[].
        # Explicit CPU save/restore avoids that query; construction is serial.
        cpu_rng = torch.random.get_rng_state()
        try:
            with torch.device('cpu'):
                torch.random.default_generator.manual_seed(seed)
                self.core = SequenceRegressor(14, 2, hidden_size=64, num_layers=2,
                    model='gru', node_count=979, node_embedding_size=8,
                    baseline_coefficients=None)
                self.register_buffer('target_mean', torch.tensor(target_scaling.mean, dtype=torch.float64))
                self.register_buffer('target_std', torch.tensor(target_scaling.std, dtype=torch.float64))
        finally:
            torch.random.set_rng_state(cpu_rng)
        self.eval()  # Dropout is zero; eval does not disable parameter gradients.

    def node_index(self, name):
        try:
            return self.node_names.index(name)
        except ValueError as error:
            raise KeyError('Unknown node; the mapping cannot be expanded') from error

    def _validate_state(self):
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('Model parameters must remain float32')
        for tensor in (*self.parameters(), *self.buffers()):
            if tensor.device.type != 'cpu' or not torch.isfinite(tensor).all().item():
                raise ValueError('Only finite CPU adapter state is supported')
        if self.core.residual:
            raise ValueError('B1 adapter must remain nonresidual')
        if self.target_mean.dtype != torch.float64 or self.target_std.dtype != torch.float64:
            raise ValueError('Supplied target constants must remain float64')
        if not torch.equal(self.target_mean, torch.tensor(self.target_scaling.mean, dtype=torch.float64)) or not torch.equal(
                self.target_std, torch.tensor(self.target_scaling.std, dtype=torch.float64)):
            raise ValueError('Target scaling buffers changed')

    def forward_standardized(self, inputs):
        _tensor(inputs, 'inputs', (36, 14))
        ids = inputs[:, :, 13]
        if (ids < 0).any().item() or (ids >= 979).any().item() or not torch.equal(ids, ids.round()) or not torch.equal(
                ids, ids[:, :1].expand_as(ids)):
            raise ValueError('Invalid, fractional, out-of-range or mixed node index')
        self._validate_state()
        result = self.core(inputs)
        _tensor(result, 'model standardized outputs', (2,))
        if len(result) != len(inputs):
            raise ValueError('Model output row count changed')
        return result

    def forward(self, inputs):
        """Physical [W, recorded IPMI ambient degC], no clipping or residual."""
        z = self.forward_standardized(inputs)
        physical = z.to(torch.float64) * self.target_std + self.target_mean
        if not torch.isfinite(physical).all().item():
            raise ValueError('Nonfinite physical output after inverse scaling')
        return physical

    def observed_targets(self, future, observed):
        """Validate original six-bin targets before any conversion can hide masks."""
        return target_validation.target_means(future, observed)

    def loss(self, inputs, future, observed):
        # Invalid labels fail before any model forward, not after a masked loss.
        physical = self.observed_targets(future, observed)
        scaled = self.target_scaling.standardize(physical)
        _tensor(inputs, 'inputs', (36, 14))
        if len(scaled) != len(inputs):
            raise ValueError('Target and input batch counts differ')
        target = torch.tensor(np.array(scaled, copy=True), dtype=torch.float32, device='cpu')
        return standardized_mse(self.forward_standardized(inputs), target)

    def metadata(self):
        return {'schema': 'b1.cpu.two.output.adapter.v1',
                'node_mapping_digest': self.mapping_digest,
                'node_mapping_digest_definition': 'canonical ordered names, not raw file bytes',
                'target_names': list(TARGET_NAMES), 'target_units': list(TARGET_UNITS),
                'physical_output_shape': ['B', 2], 'core_output': 'standardized',
                'history_steps': 36, 'features': 14, 'embedding': [979, 8],
                'gru_hidden_size': 64, 'gru_layers': 2, 'head': [64, 2],
                'dropout': 0, 'residual': False, 'seed': self.seed,
                'target_scaling': self.target_scaling.record(),
                'target_scaling_digest': self.target_scaling.digest,
                'input_scaling_or_window_preparation_implemented': False,
                'checkpoint_loading_supported': False, 'training_authorized': False}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        raise PermissionError('Checkpoint/state loading is not supported by B1 adapter v1; no historical warm start')
