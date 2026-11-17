"""Bounded actual synthetic CPU FedAvg; no real arrays/checkpoints/scoring."""
from dataclasses import replace
import hashlib
import json
import unittest
from unittest import mock

import numpy as np
import torch

from src import b1_fedavg_e2e_synthetic_v1 as e2e
from src import b1_local_sgd_synthetic_v1 as local
from src.b1_fedavg_orchestration_v1 import RoundSnapshot
from src.b1_target_scaling_v1 import TargetScaling


def scale():return TargetScaling((300.,20.),(100.,5.))


def clients():
    return (local.make_synthetic_client(0,17),local.make_synthetic_client(7,128),
            local.make_synthetic_client(978,129))


def same(test,a,b):
    test.assertEqual(tuple(a),tuple(b))
    for n in a:
        test.assertEqual(a[n].dtype,b[n].dtype);test.assertEqual(a[n].shape,b[n].shape)
        test.assertTrue(torch.equal(a[n],b[n]),n)


def model():return local.make_synthetic_model(target_scaling=scale())


class B1SyntheticE2ETests(unittest.TestCase):
    def test_sampling_deterministic_unique_and_input_order_independent(self):
        pool=clients();sequence=[]
        for r in range(1,6):
            chosen=e2e.sample_synthetic_clients(pool,seed=42,round_id=r,count=2)
            ids=tuple(c.client_id for c in chosen);sequence.append(ids)
            self.assertEqual(len(set(ids)),2)
            self.assertEqual(ids,tuple(c.client_id for c in e2e.sample_synthetic_clients(tuple(reversed(pool)),seed=42,round_id=r,count=2)))
            self.assertTrue(set(ids)<={c.client_id for c in pool})
        other=[tuple(c.client_id for c in e2e.sample_synthetic_clients(pool,seed=43,round_id=r,count=2)) for r in range(1,6)]
        self.assertNotEqual(sequence,other)

    def test_invalid_synthetic_config_and_pool_bounds(self):
        for kw in ({'rounds':0},{'rounds':6},{'clients_per_round':5},{'clients_per_round':True},{'seed':41},{'seed':42.}):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):e2e.SyntheticE2EConfig(**kw)
        pool=clients()
        for bad in ((),(pool[0],pool[0]),(object(),),tuple(local.make_synthetic_client(i,1) for i in range(5))):
            with self.assertRaises(ValueError):e2e.run_synthetic_fedavg(bad,target_scaling=scale())
        with self.assertRaises(ValueError):e2e.run_synthetic_fedavg(pool[:1],target_scaling=scale())
        with self.assertRaises(ValueError):e2e.run_synthetic_fedavg(pool,target_scaling=None)

    def test_live_transfer_exact_independent_and_source_unchanged(self):
        m=model();base=local._snapshot(m,0);initial=base.state_dict();aggregate=base.state_dict()
        aggregate['core.head.bias']=aggregate['core.head.bias']+torch.tensor([.25,-.5])
        next_model,after=e2e.transfer_synthetic_aggregate(m,base,aggregate)
        same(self,aggregate,next_model.state_dict());same(self,aggregate,after.state_dict())
        same(self,initial,m.state_dict())
        self.assertIsNot(m,next_model);self.assertEqual(after.round_id,1)
        for before,after_parameter in zip(m.parameters(),next_model.parameters()):
            self.assertNotEqual(before.data_ptr(),after_parameter.data_ptr());self.assertIsNone(after_parameter.grad)
        aggregate['core.head.bias'].zero_()
        self.assertFalse(torch.equal(aggregate['core.head.bias'],next_model.state_dict()['core.head.bias']))
        self.assertFalse(next_model.training)

    def test_transfer_rejects_bad_state_before_mutating_source(self):
        m=model();base=local._snapshot(m,0);initial=base.state_dict()
        for kind in ('nan','shape','dtype','fixed','missing','extra','grad'):
            state=base.state_dict()
            if kind=='nan':state['core.head.bias'][0]=np.nan
            elif kind=='shape':state['core.head.bias']=torch.ones(3)
            elif kind=='dtype':state['core.head.bias']=state['core.head.bias'].double()
            elif kind=='fixed':state['target_std'][0]+=1
            elif kind=='missing':del state['core.head.bias']
            elif kind=='extra':state['extra']=torch.ones(1)
            elif kind=='grad':state['core.head.bias'].requires_grad_(True)
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):e2e.transfer_synthetic_aggregate(m,base,state)
            same(self,initial,m.state_dict())
        other=local.make_synthetic_model(target_scaling=scale(),seed=43)
        with self.assertRaises(ValueError):e2e.transfer_synthetic_aggregate(m,local._snapshot(other,0),initial)

    def test_actual_three_round_chain_all_seeds_and_finite_global_parameters(self):
        for seed in (42,43,44):
            with self.subTest(seed=seed):
                result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale(),config=e2e.SyntheticE2EConfig(seed=seed))
                self.assertEqual(len(result.rounds),3);self.assertEqual(result.summary['client_sessions'],6)
                self.assertEqual(result.summary['completed_rounds'],3)
                for i,record in enumerate(result.rounds):
                    self.assertEqual(record.before.round_id,i);self.assertEqual(record.after.round_id,i+1)
                    for update in record.local_results:
                        self.assertEqual(update.receipt['base_fingerprint'],record.before.fingerprint)
                        self.assertGreater(update.receipt['parameter_delta_l2'],0)
                    if i:
                        same(self,result.rounds[i-1].after.state_dict(),record.before.state_dict())
                        self.assertEqual(result.rounds[i-1].after.fingerprint,record.before.fingerprint)
                        self.assertNotEqual(record.before.fingerprint,result.rounds[0].before.fingerprint)
                    self.assertTrue(record.receipt['global_transfer_exact'])
                    self.assertFalse(record.receipt['optimizer_state_carried'])
                same(self,result.rounds[-1].after.state_dict(),result.global_model.state_dict())
                result.global_model._validate_state()
                self.assertTrue(all(torch.isfinite(p).all().item() for p in result.global_model.parameters()))
                self.assertTrue(all(p.grad is None for p in result.global_model.parameters()))

    def test_independent_weighted_aggregate_math_for_every_round(self):
        result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        for record in result.rounds:
            counts=[u.update.num_examples for u in record.local_results];total=sum(counts)
            initial=record.before.state_dict();actual=record.after.state_dict()
            for name,value in actual.items():
                if name in record.before.fixed_keys:
                    self.assertTrue(torch.equal(value,initial[name]));continue
                expected=sum(u.update.state[name].double()*n for u,n in zip(record.local_results,counts))/total
                torch.testing.assert_close(value,expected.to(value.dtype),rtol=1e-6,atol=1e-7)

    def test_embedding_dilution_and_unselected_rows_preserved_across_rounds(self):
        result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        for record in result.rounds:
            name='core.node_embedding.weight';before=record.before.state_dict()[name];after=record.after.state_dict()[name]
            selected=record.receipt['selected_node_indices'];other=torch.ones(979,dtype=torch.bool)
            other[list(selected)]=False
            self.assertTrue(torch.equal(before[other],after[other]))
            for local_result in record.local_results:
                owner=local_result.receipt['node_index'];weight=record.receipt['weights'][local_result.update.client_id]
                delta=local_result.update.state[name][owner].double()-before[owner].double()
                actual=after[owner].double()-before[owner].double()
                self.assertGreater(delta.norm().item(),0)
                torch.testing.assert_close(actual,delta*weight,rtol=.02,atol=3e-7)

    def test_fixed_seed_exact_multiround_determinism_and_prefix_invariance(self):
        pool=clients()
        first=e2e.run_synthetic_fedavg(pool,target_scaling=scale())
        second=e2e.run_synthetic_fedavg(tuple(reversed(pool)),target_scaling=scale())
        same(self,first.global_model.state_dict(),second.global_model.state_dict())
        self.assertEqual(first.summary,second.summary)
        self.assertEqual([r.receipt for r in first.rounds],[r.receipt for r in second.rounds])
        prefix=e2e.run_synthetic_fedavg(pool,target_scaling=scale(),config=e2e.SyntheticE2EConfig(rounds=2))
        for a,b in zip(first.rounds,prefix.rounds):
            self.assertEqual(a.receipt,b.receipt);same(self,a.after.state_dict(),b.after.state_dict())

    def test_local_models_do_not_change_shared_round_start(self):
        observations=[];original=local.train_synthetic_client
        def observed(global_model,client,**kwargs):
            start={n:p.clone() for n,p in global_model.state_dict().items()}
            result=original(global_model,client,**kwargs)
            same(self,start,global_model.state_dict())
            observations.append((kwargs['round_id'],result.receipt['base_fingerprint']))
            return result
        with mock.patch.object(local,'train_synthetic_client',side_effect=observed):
            result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        for r in result.rounds:
            same_round=[fp for i,fp in observations if i==r.receipt['round']]
            self.assertEqual(same_round,[r.before.fingerprint]*2)

    def test_fresh_optimizer_and_parameter_storage_for_each_round_client(self):
        optimizers=[];original=local._new_optimizer
        def factory(parameters):
            opt=original(parameters);optimizers.append(opt);return opt
        with mock.patch.object(local,'_new_optimizer',side_effect=factory):
            result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        self.assertEqual(len(optimizers),6);self.assertEqual(len({id(o) for o in optimizers}),6)
        self.assertEqual(len({id(o.state) for o in optimizers}),6)
        owners=[{p.data_ptr() for p in o.param_groups[0]['params']} for o in optimizers]
        for i,a in enumerate(owners):
            for b in owners[i+1:]:self.assertTrue(a.isdisjoint(b))
        global_storage={p.data_ptr() for p in result.global_model.parameters()}
        self.assertTrue(all(global_storage.isdisjoint(a) for a in owners))
        self.assertTrue(all(not o.state for o in optimizers))

    def test_reused_optimizer_cannot_leak_into_next_round(self):
        cached=[];original=local._new_optimizer
        def reuse(parameters):
            if not cached:cached.append(original(parameters))
            return cached[0]
        with mock.patch.object(local,'_new_optimizer',side_effect=reuse):
            with self.assertRaisesRegex(ValueError,'ownership'):
                e2e.run_synthetic_fedavg(clients()[:1],target_scaling=scale(),
                    config=e2e.SyntheticE2EConfig(clients_per_round=1,rounds=2))

    def test_invalid_any_client_targets_stop_before_all_local_optimization(self):
        pool=clients();last=pool[-1]
        invalid=[replace(last,future=np.ma.array(last.future,mask=False)),
                 replace(last,future=last.future.astype(bool))]
        f=last.future.copy();f[-1,-1,1]=np.inf;invalid.append(replace(last,future=f))
        o=last.observed.copy();o[-1,-1,0]=False;invalid.append(replace(last,observed=o))
        for client in invalid:
            with mock.patch.object(local,'_new_optimizer',side_effect=AssertionError('optimizer reached')):
                with self.assertRaises(ValueError):e2e.run_synthetic_fedavg((*pool[:-1],client),target_scaling=scale())

    def test_round_counts_and_participation_reconcile_with_selected_client_sizes(self):
        pool=clients();count={c.client_id:len(c.inputs) for c in pool}
        result=e2e.run_synthetic_fedavg(pool,target_scaling=scale())
        examples=steps=0;participation={n:0 for n in count}
        for record in result.rounds:
            selected=record.receipt['selected_clients']
            expected_examples=sum(count[n] for n in selected)
            expected_steps=sum((count[n]+127)//128 for n in selected)
            self.assertEqual(record.receipt['examples_processed'],expected_examples)
            self.assertEqual(record.receipt['optimizer_steps'],expected_steps)
            self.assertEqual(len(set(selected)),2)
            for n in selected:participation[n]+=1
            examples+=expected_examples;steps+=expected_steps
        self.assertEqual(result.summary['examples_processed'],examples)
        self.assertEqual(result.summary['optimizer_steps'],steps)
        self.assertEqual(result.summary['participation'],participation)
        self.assertEqual(sum(participation.values()),6)

    def test_corrupt_aggregate_aborts_without_changing_source_global(self):
        source=[];original_model=local.make_synthetic_model;original_round=local.train_synthetic_round
        def make(**kwargs):
            m=original_model(**kwargs);source.append((m,{n:p.clone() for n,p in m.state_dict().items()}));return m
        def corrupt(*args,**kwargs):
            out=original_round(*args,**kwargs);out['state']['core.head.bias'][0]=np.nan;return out
        with mock.patch.object(local,'make_synthetic_model',side_effect=make),mock.patch.object(local,'train_synthetic_round',side_effect=corrupt):
            with self.assertRaises(ValueError):e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        same(self,source[0][0].state_dict(),source[0][1])

    def test_missing_client_and_false_step_receipts_abort_before_transfer(self):
        original=local.train_synthetic_round
        for fault in ('missing','steps','base'):
            def corrupt(*args,**kwargs):
                result=original(*args,**kwargs)
                if fault=='missing':result['local_results']=result['local_results'][:-1]
                elif fault=='steps':result['receipt']['optimizer_steps']+=1
                else:result['receipt']['base_fingerprint']='wrong'
                return result
            with mock.patch.object(local,'train_synthetic_round',side_effect=corrupt),mock.patch.object(e2e,'transfer_synthetic_aggregate',side_effect=AssertionError('transfer reached')):
                with self.assertRaises(ValueError):e2e.run_synthetic_fedavg(clients(),target_scaling=scale())

    def test_receipt_chain_and_state_fingerprints(self):
        result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale())
        previous=result.summary['initial_fingerprint']
        for round_result in result.rounds:
            r=round_result.receipt;body={k:v for k,v in r.items() if k!='receipt_digest'}
            self.assertEqual(r['previous_receipt_digest'],previous)
            calculated=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()
            self.assertEqual(r['receipt_digest'],calculated);previous=calculated
        self.assertEqual(previous,result.summary['last_receipt_digest'])

    def test_fixture_and_caller_random_streams_are_not_mutated(self):
        pool=clients();before=[(c.inputs.clone(),c.future.copy(),c.observed.copy()) for c in pool]
        numpy_before=np.random.get_state();torch_before=torch.random.get_rng_state().clone()
        e2e.run_synthetic_fedavg(pool,target_scaling=scale())
        self.assertTrue(torch.equal(torch_before,torch.random.get_rng_state()))
        numpy_after=np.random.get_state();self.assertTrue(np.array_equal(numpy_before[1],numpy_after[1]))
        for client,(x,f,o) in zip(pool,before):
            self.assertTrue(torch.equal(client.inputs,x));self.assertTrue(np.array_equal(client.future,f));self.assertTrue(np.array_equal(client.observed,o))

    def test_single_client_five_rounds_and_only_one_seeded_initialization(self):
        original=local.make_synthetic_model
        with mock.patch.object(local,'make_synthetic_model',wraps=original) as make:
            result=e2e.run_synthetic_fedavg(clients()[:1],target_scaling=scale(),
                config=e2e.SyntheticE2EConfig(rounds=5,clients_per_round=1))
        self.assertEqual(make.call_count,1);self.assertEqual(result.summary['optimizer_steps'],5)
        for record in result.rounds:
            same(self,record.after.state_dict(),record.local_results[0].update.state)

    def test_checkpoint_and_state_loading_apis_remain_disabled(self):
        with mock.patch.object(local.B1TwoOutputAdapter,'load_state_dict',side_effect=AssertionError('load forbidden')):
            result=e2e.run_synthetic_fedavg(clients(),target_scaling=scale(),config=e2e.SyntheticE2EConfig(rounds=2))
        self.assertFalse(result.summary['checkpoint_loading'])
        self.assertFalse(result.summary['real_data_used']);self.assertFalse(result.summary['scoring'])
        self.assertFalse(result.summary['training_authorized'])


if __name__=='__main__':unittest.main()
