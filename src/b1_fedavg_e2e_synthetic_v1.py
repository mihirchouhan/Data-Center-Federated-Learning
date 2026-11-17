"""Bounded CPU synthetic FedAvg with actual local SGD and live-state transfer.

No real-data interface, checkpoint loader, evaluation, disk output or fit CLI.
The existing adapter's load_state_dict prohibition remains intact.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from numbers import Integral

import numpy as np
import torch

from src import b1_local_sgd_synthetic_v1 as local_sgd
from src.b1_fedavg_orchestration_v1 import RoundSnapshot, batch_sizes
from src.b1_target_scaling_v1 import TargetScaling


def _integer(value, label, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(label + ': invalid synthetic integer')
    return int(value)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SyntheticE2EConfig:
    rounds: int = 3
    clients_per_round: int = 2
    seed: int = 42

    def __post_init__(self):
        for name,lower,upper in (('rounds',1,5),('clients_per_round',1,4),('seed',42,44)):
            object.__setattr__(self,name,_integer(getattr(self,name),name,lower,upper))


def _cohort(clients):
    clients=tuple(clients)
    if not 1 <= len(clients) <= 4 or any(not isinstance(c,local_sgd.SyntheticClient) for c in clients):
        raise ValueError('Only a pool of 1..4 invented clients is supported')
    if len({c.client_id for c in clients}) != len(clients):
        raise ValueError('Duplicate synthetic client identity')
    # Pool ordering is stable without renumbering any embedding/node index.
    return tuple(sorted(clients,key=lambda c:c.node_index))


def sample_synthetic_clients(clients, *, seed, round_id, count):
    pool=_cohort(clients)
    seed=_integer(seed,'seed',42,44)
    round_id=_integer(round_id,'round_id',1,5)
    count=_integer(count,'sample count',1,len(pool))
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1,round_id,0])))
    return tuple(pool[int(i)] for i in rng.choice(len(pool),size=count,replace=False))


def transfer_synthetic_aggregate(global_model, base_snapshot, aggregate):
    """Copy a validated live synthetic aggregate into a NEW global model.

    Does not call any state/checkpoint loading API or mutate the source model.
    All validation precedes writes; a failed copy can affect only the new clone.
    """
    local_sgd._validate_model(global_model)
    if not isinstance(base_snapshot,RoundSnapshot) or base_snapshot.round_id >= 5:
        raise ValueError('Bounded synthetic starting snapshot required')
    current=local_sgd._snapshot(global_model,base_snapshot.round_id)
    if current.fingerprint != base_snapshot.fingerprint:
        raise ValueError('Aggregate source snapshot differs from live global model')
    checked=base_snapshot.validate(aggregate)
    next_model=copy.deepcopy(global_model)
    destinations=next_model.state_dict(keep_vars=True)
    if tuple(destinations) != tuple(checked):
        raise ValueError('Destination state keys/order changed')
    with torch.no_grad():
        for name,destination in destinations.items():
            destination.copy_(checked[name])
    next_model.zero_grad(set_to_none=True)
    next_model.eval()
    next_model._validate_state()
    expected=RoundSnapshot(checked,base_snapshot.trainable_keys,round_id=base_snapshot.round_id+1)
    actual=local_sgd._snapshot(next_model,base_snapshot.round_id+1)
    if actual.fingerprint != expected.fingerprint:
        raise ValueError('Aggregate-to-global copy is not exact')
    if local_sgd._snapshot(global_model,base_snapshot.round_id).fingerprint != base_snapshot.fingerprint:
        raise ValueError('Source global state mutated during transfer')
    return next_model,actual


@dataclass(frozen=True)
class ExecutedSyntheticRound:
    before: RoundSnapshot
    after: RoundSnapshot
    local_results: tuple
    receipt: dict


@dataclass(frozen=True)
class SyntheticE2EResult:
    global_model: object
    rounds: tuple
    summary: dict


def _validate_round_result(result, selected, base):
    if not isinstance(result,dict) or not isinstance(result.get('local_results'),tuple):
        raise ValueError('Invalid local-round result')
    updates=result['local_results']
    if len(updates) != len(selected):
        raise ValueError('Missing/extra local clients; round aborted')
    examples=steps=0
    for client,returned in zip(selected,updates):
        if not isinstance(returned,local_sgd.LocalSGDResult):
            raise ValueError('Invalid local SGD result')
        update,record=returned.update,returned.receipt
        count=len(client.inputs)
        batches=batch_sizes(count)
        if (update.client_id != client.client_id or update.num_examples != count or
                update.base_fingerprint != base.fingerprint or
                record['base_fingerprint'] != base.fingerprint or
                record['client_id'] != client.client_id or record['training_windows'] != count or
                record['examples_processed'] != count or record['optimizer_steps'] != len(batches) or
                tuple(record['actual_batch_sizes']) != batches or
                record['optimizer_initial_state_entries'] != 0 or record['optimizer_final_state_entries'] != 0):
            raise ValueError('Local initialization/client/count/step receipt differs')
        base.validate(update.state)
        examples+=count;steps+=len(batches)
    if (result['receipt']['base_fingerprint'] != base.fingerprint or
            result['receipt']['examples_processed'] != examples or
            result['receipt']['optimizer_steps'] != steps):
        raise ValueError('Round counters do not reconcile')
    return examples,steps


def run_synthetic_fedavg(clients, *, target_scaling, config=SyntheticE2EConfig()):
    """Fresh initialization -> sample -> actual local SGD -> average -> next model.

    All states and receipts are memory-only. No outer validation/test hook,
    checkpoint/resume, real client registry or 98-of-979 training exists here.
    """
    if not isinstance(config,SyntheticE2EConfig) or not isinstance(target_scaling,TargetScaling):
        raise ValueError('Explicit synthetic config and supplied target constants required')
    pool=_cohort(clients)
    if config.clients_per_round > len(pool):
        raise ValueError('Sample size exceeds invented client pool')
    current=local_sgd.make_synthetic_model(target_scaling=target_scaling,seed=config.seed)
    # Even an invalid unselected client stops before the first local optimizer.
    for client in pool:local_sgd._validate_client(current,client)
    initial=local_sgd._snapshot(current,0)
    previous=initial.fingerprint
    history=[];total_examples=total_steps=0
    participation={c.client_id:0 for c in pool}
    for round_id in range(1,config.rounds+1):
        before=local_sgd._snapshot(current,round_id-1)
        if history and before.fingerprint != history[-1].after.fingerprint:
            raise ValueError('Next round did not start from the prior aggregate')
        selected=sample_synthetic_clients(pool,seed=config.seed,round_id=round_id,count=config.clients_per_round)
        local=local_sgd.train_synthetic_round(current,selected,seed=config.seed,round_id=round_id)
        examples,steps=_validate_round_result(local,selected,before)
        next_model,after=transfer_synthetic_aggregate(current,before,local['state'])
        receipt={'schema':'b1.synthetic.e2e.round.v1','seed':config.seed,'round':round_id,
                 'selected_clients':tuple(c.client_id for c in selected),
                 'selected_node_indices':tuple(c.node_index for c in selected),
                 'examples_processed':examples,'optimizer_steps':steps,
                 'weights':dict(local['receipt']['weights']),
                 'base_fingerprint':before.fingerprint,'result_fingerprint':after.fingerprint,
                 'previous_receipt_digest':previous,'mapping_digest':current.mapping_digest,
                 'target_scaling_digest':current.target_scaling.digest,
                 'global_transfer_exact':True,'optimizer_state_carried':False,
                 'local_receipts':tuple(r.receipt for r in local['local_results'])}
        digest=_digest(receipt)
        receipt['receipt_digest']=digest
        history.append(ExecutedSyntheticRound(before,after,local['local_results'],receipt))
        current=next_model;previous=digest
        total_examples+=examples;total_steps+=steps
        for client in selected:participation[client.client_id]+=1
    return SyntheticE2EResult(current,tuple(history),
        {'schema':'b1.synthetic.e2e.completed.v1','seed':config.seed,'completed_rounds':len(history),
         'client_pool':tuple(participation),'clients_per_round':config.clients_per_round,
         'client_sessions':sum(participation.values()),'participation':participation,
         'examples_processed':total_examples,'optimizer_steps':total_steps,
         'initial_fingerprint':initial.fingerprint,'final_fingerprint':history[-1].after.fingerprint,
         'last_receipt_digest':previous,'batch_size':128,'local_epochs':1,'drop_last':False,
         'sampling_domain':1,'local_shuffle_domain':2,
         'aggregation':'standard_full_table_sample_weighted_CPU_float64',
         'live_state_transfer':'validated_copy_into_fresh_CPU_clone',
         'checkpoint_loading':False,'real_data_used':False,'scoring':False,
         'training_authorized':False})
