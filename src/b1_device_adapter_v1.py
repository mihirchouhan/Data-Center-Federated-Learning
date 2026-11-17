"""Additive device-aware B1 model. Historical CPU adapter remains unchanged."""
from collections import OrderedDict
import numpy as np
import torch
from src.b1_model_adapter_v1 import B1TwoOutputAdapter
from src.b1_real_data_bridge_v1 import ORIGINAL_NAMES, MAPPING_DIGEST, initialization_record
from src.b1_fedavg_orchestration_v1 import RoundSnapshot


def explicit_device(value):
    d = torch.device(value)
    if d.type not in ('cpu', 'cuda') or (d.type == 'cuda' and d.index is None):
        raise ValueError('Explicit cpu or cuda:N required; no automatic fallback')
    if d.type == 'cpu' and d.index is not None:
        raise ValueError('Use plain cpu')
    if d.type == 'cuda' and (not torch.cuda.is_available() or d.index >= torch.cuda.device_count()):
        raise ValueError('Requested CUDA device unavailable; no CPU fallback')
    return d


def tensor(value, tail, device, label):
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != torch.float32:
        raise ValueError(label + ': wrong exact device or float32 dtype')
    if value.ndim != len(tail)+1 or not len(value) or tuple(value.shape[1:]) != tail:
        raise ValueError(label + ': incompatible shape')
    if not torch.isfinite(value).all().item():
        raise ValueError(label + ': nonfinite values')
    return value


class B1DeviceAdapter(B1TwoOutputAdapter):
    """Same core architecture/state names; B1 root seed -> domain-0 init seed.

    Initialize on CPU once, then explicitly move to the declared device.
    Resume accepts only validated B1 round snapshots, never a loose checkpoint.
    """
    def __init__(self, *, target_scaling, root_seed=42, device='cpu', node_names=ORIGINAL_NAMES):
        if tuple(node_names) != ORIGINAL_NAMES:
            raise ValueError('Original 979 node ordinals required')
        d = explicit_device(device)
        init = initialization_record(root_seed)
        super().__init__(node_names=ORIGINAL_NAMES, expected_mapping_digest=MAPPING_DIGEST,
                         target_scaling=target_scaling, seed=init['initialization_seed'])
        self.root_seed = root_seed
        self.initialization = init
        self.intended_device = d
        self.to(d)
        self._validate_state()

    def _validate_state(self):
        d = self.intended_device
        if self.core.residual or any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('Nonresidual float32 B1 core required')
        for t in (*self.parameters(), *self.buffers()):
            if t.device != d or not torch.isfinite(t).all().item():
                raise ValueError('Nonfinite or wrong-device model state')
        for t, values in ((self.target_mean, self.target_scaling.mean), (self.target_std, self.target_scaling.std)):
            if t.dtype != torch.float64 or not torch.equal(t, torch.tensor(values, dtype=torch.float64, device=d)):
                raise ValueError('Fixed physical target-scaling buffers changed')

    def forward_standardized(self, inputs):
        tensor(inputs, (36,14), self.intended_device, 'inputs')
        ids = inputs[:,:,13]
        if (ids < 0).any().item() or (ids >= 979).any().item() or not torch.equal(ids, ids.round()) or not torch.equal(ids, ids[:,:1].expand_as(ids)):
            raise ValueError('Invalid or mixed node ordinal')
        self._validate_state()
        out = self.core(inputs)
        tensor(out, (2,), self.intended_device, 'standardized prediction')
        if len(out) != len(inputs): raise ValueError('Changed batch row count')
        return out

    def loss(self, inputs, future, observed):
        # Hardened target validation BEFORE any mask-erasing conversion or forward.
        physical = self.observed_targets(future, observed)
        scaled = self.target_scaling.standardize(physical)
        tensor(inputs, (36,14), self.intended_device, 'inputs')
        if len(scaled) != len(inputs): raise ValueError('Input/target row count differs')
        target = torch.tensor(np.array(scaled, copy=True), dtype=torch.float32, device=self.intended_device)
        tensor(target, (2,), self.intended_device, 'standardized target')
        loss = ((self.forward_standardized(inputs)-target)**2).mean()
        if not torch.isfinite(loss).item(): raise ValueError('Nonfinite standardized MSE')
        return loss

    def snapshot(self, round_id):
        self._validate_state()
        state = OrderedDict((k,v.detach().cpu().clone()) for k,v in self.state_dict().items())
        return RoundSnapshot(state, tuple(k for k,p in self.named_parameters() if p.requires_grad), round_id=round_id)

    def restore_round(self, snapshot):
        if not isinstance(snapshot, RoundSnapshot): raise TypeError('Validated B1 snapshot required')
        anchor = self.snapshot(snapshot.round_id)
        checked = anchor.validate(snapshot.state_dict())
        if snapshot.trainable_keys != anchor.trainable_keys: raise ValueError('Trainable roles differ')
        with torch.no_grad():
            for k,t in self.state_dict(keep_vars=True).items(): t.copy_(checked[k].to(self.intended_device))
        self.zero_grad(set_to_none=True)
        if self.snapshot(snapshot.round_id).fingerprint != snapshot.fingerprint:
            raise ValueError('State transfer failed exact fingerprint verification')

    def metadata(self):
        r = super().metadata()
        r.update(schema='b1.device.two.output.adapter.v1', device=str(self.intended_device),
                 root_seed=self.root_seed, initialization=self.initialization,
                 checkpoint_loading_supported='verified_round_snapshot_only', training_authorized=False)
        return r
