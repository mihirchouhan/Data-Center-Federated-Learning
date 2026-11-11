"""Future local-training/FedAvg contracts. No training or aggregation algorithm."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, Mapping
import hashlib
import json
import numpy as np
from .schema import frozen_array
from .validation import validate_batch

def positive_count(n):
    if type(n) is not int or n <= 0:
        raise ValueError("Training sample count must be a positive integer")
    return n

def eligible_clients(training,spec):
    validate_batch(training,spec)
    if training.split != "train":
        raise PermissionError("Eligibility accepts training windows only")
    result={}
    for i,name in enumerate(training.mapping):
        selected=training.nodes==i
        count=int(selected.sum())
        days=len(np.unique(training.origins[selected].astype("datetime64[D]")))
        if count >= spec.minimum_windows and days >= spec.minimum_days:
            result[name]=count
    return result

@dataclass(frozen=True)
class ClientData:
    node: str
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    sample_count: int

def client_views(training,spec,scaler):
    eligible=eligible_clients(training,spec)
    x,y=scaler.transform(training,spec)
    views={}
    for i,name in enumerate(training.mapping):
        if name in eligible:
            selected=training.nodes==i
            views[name]=ClientData(name,frozen_array(x[selected]),frozen_array(y[selected]),
                frozen_array(training.origins[selected]),positive_count(eligible[name]))
    return views

@dataclass(frozen=True)
class ModelState:
    round_id: int
    tensors: Mapping[str,np.ndarray]
    fixed_keys: tuple = ()

    def __post_init__(self):
        if type(self.round_id) is not int or self.round_id < 0 or not self.tensors:
            raise ValueError("Nonnegative round and nonempty state required")
        state={}
        for k,v in sorted(self.tensors.items()):
            if not isinstance(k,str) or not k:
                raise ValueError("Named state required")
            a=frozen_array(v)
            if a.dtype not in (np.dtype("float32"),np.dtype("float64")) or not np.isfinite(a).all():
                raise ValueError("Finite float32/64 model state required")
            state[k]=a
        if len(set(self.fixed_keys)) != len(self.fixed_keys) or not set(self.fixed_keys).issubset(state):
            raise ValueError("Invalid fixed-state keys")
        object.__setattr__(self,"tensors",MappingProxyType(state))

    @property
    def fingerprint(self):
        h=hashlib.sha256(json.dumps([self.round_id,list(self.fixed_keys)],separators=(",",":")).encode())
        for k,a in self.tensors.items():
            h.update(json.dumps([k,a.dtype.str,list(a.shape)],separators=(",",":")).encode())
            h.update(a.tobytes())
        return h.hexdigest()

@dataclass(frozen=True)
class ClientUpdate:
    node: str
    training_samples: int
    base_fingerprint: str
    state: ModelState

def validate_update_contract(global_state,updates,expected_counts):
    """Validate a request and return declared weights; NEVER average parameters."""
    if not isinstance(global_state,ModelState) or not expected_counts:
        raise ValueError("Global state and frozen participant counts required")
    for node,n in expected_counts.items():
        if not isinstance(node,str) or not node:
            raise ValueError("Named client required")
        positive_count(n)
    updates=tuple(updates)
    if len(updates)!=len(expected_counts):
        raise ValueError("Missing participant update")
    seen=set()
    for u in updates:
        if not isinstance(u,ClientUpdate) or u.node in seen or u.node not in expected_counts:
            raise ValueError("Duplicate or unexpected client")
        seen.add(u.node)
        if positive_count(u.training_samples)!=expected_counts[u.node]:
            raise ValueError("Update count differs from frozen valid training count")
        if u.base_fingerprint!=global_state.fingerprint:
            raise ValueError("Clients must share identical global start")
        if not isinstance(u.state,ModelState) or u.state.round_id!=global_state.round_id:
            raise ValueError("Wrong round state")
        if u.state.fixed_keys!=global_state.fixed_keys or set(u.state.tensors)!=set(global_state.tensors):
            raise ValueError("Parameter names/fixed keys differ")
        for k,base in global_state.tensors.items():
            a=u.state.tensors[k]
            if a.shape!=base.shape or a.dtype!=base.dtype:
                raise ValueError("Incompatible parameter shape/dtype")
            if k in global_state.fixed_keys and not np.array_equal(a,base):
                raise ValueError("Fixed state changed")
    if seen!=set(expected_counts):
        raise ValueError("Participants differ")
    total=sum(expected_counts.values())
    return {k:expected_counts[k]/total for k in sorted(expected_counts)}

class ClientTrainer(Protocol):
    def train(self,data:ClientData,global_state:ModelState) -> ClientUpdate:
        """Future only: no implementation or optimizer in Phase 1.3."""
        ...

class WeightedAggregator(Protocol):
    def aggregate(self,global_state:ModelState,updates:tuple[ClientUpdate,...],
                  expected_counts:Mapping[str,int]) -> ModelState:
        """Future only: validate first, then new immutable round state."""
        ...
