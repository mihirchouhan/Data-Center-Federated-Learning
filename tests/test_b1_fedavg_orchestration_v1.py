"""Invented CPU state fixtures only: no forward, backward, optimizers or data."""
from collections import OrderedDict
from dataclasses import replace
import unittest

import numpy as np
import torch

from src.b1_model_adapter_v1 import B1TwoOutputAdapter, mapping_digest
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_fedavg_orchestration_v1 import (
    CandidateConfig, ClientRegistry, ClientStateUpdate, RoundSnapshot,
    aggregate_fedavg, batch_sizes, sample_clients, plan_round, local_row_order,
    scripted_update, commit_scripted_round, run_scripted_rounds, participation_summary)


NAMES = tuple('invented-node-%04d' % i for i in range(979))


def registry():
    return ClientRegistry(NAMES, {n:148+i%290 for i,n in enumerate(NAMES)},
                          expected_mapping_digest=mapping_digest(NAMES))


def snapshot(dtype=torch.float64):
    state = OrderedDict(weight=torch.zeros(2,dtype=dtype),
                        embedding=torch.zeros((979,8),dtype=dtype),
                        target_mean=torch.tensor([300.,20.],dtype=torch.float64),
                        target_std=torch.tensor([100.,5.],dtype=torch.float64),
                        counter=torch.tensor(2**60+1,dtype=torch.int64),
                        valid=torch.tensor(True))
    return RoundSnapshot(state, ('weight','embedding'))


def update(base, name, count, value):
    state=base.state_dict();state['weight']=torch.full_like(state['weight'],value)
    return ClientStateUpdate(name,count,base.fingerprint,state)


def equal(test,a,b):
    test.assertEqual(tuple(a),tuple(b))
    for n in a:
        test.assertEqual(a[n].dtype,b[n].dtype)
        test.assertEqual(a[n].shape,b[n].shape)
        test.assertTrue(torch.equal(a[n],b[n]),n)


class B1OrchestrationTests(unittest.TestCase):
    def test_unequal_weights_hand_calculation(self):
        s=snapshot();out=aggregate_fedavg(s,[update(s,'a',1,2),update(s,'b',3,6)],
                                        expected_counts={'a':1,'b':3},client_order=('a','b'))
        self.assertTrue(torch.equal(out['weight'],torch.tensor([5.,5.],dtype=torch.float64)))

    def test_single_client_returns_exact_copy(self):
        s=snapshot(torch.float32);u=update(s,'a',9,1e20)
        out=aggregate_fedavg(s,[u],expected_counts={'a':9},client_order=('a',))
        equal(self,out,u.state);out['weight'].zero_()
        self.assertFalse(torch.equal(out['weight'],u.state['weight']))

    def test_cpu_float64_accumulation_before_cast(self):
        s=snapshot(torch.float32)
        out=aggregate_fedavg(s,[update(s,'a',1,1e8),update(s,'b',1,1),update(s,'c',1,-1e8)],
                            expected_counts={'a':1,'b':1,'c':1},client_order=('a','b','c'))
        self.assertAlmostEqual(out['weight'][0].item(),1/3,places=6)
        self.assertEqual(out['weight'].dtype,torch.float32)

    def test_full_embedding_delta_dilution(self):
        s=snapshot();a=update(s,'owner0',1,0);b=update(s,'owner1',3,0)
        a.state['embedding'][0]=8.;b.state['embedding'][1]=4.
        out=aggregate_fedavg(s,[b,a],expected_counts={'owner0':1,'owner1':3},client_order=('owner0','owner1'))
        self.assertTrue(torch.equal(out['embedding'][0],torch.full((8,),2.,dtype=torch.float64)))
        self.assertTrue(torch.equal(out['embedding'][1],torch.full((8,),3.,dtype=torch.float64)))
        self.assertEqual(torch.count_nonzero(out['embedding'][2:]).item(),0)

    def test_fixed_floating_integer_and_boolean_state_exact(self):
        s=snapshot();out=aggregate_fedavg(s,[update(s,'a',1,1),update(s,'b',2,3)],
                                         expected_counts={'a':1,'b':2},client_order=('a','b'))
        for n in s.fixed_keys:self.assertTrue(torch.equal(out[n],s.state_dict()[n]))
        self.assertEqual(out['counter'].item(),2**60+1)
        self.assertEqual(out['valid'].dtype,torch.bool)

    def test_changed_fixed_entries_rejected(self):
        s=snapshot()
        for name in s.fixed_keys:
            u=update(s,'a',1,1)
            u.state[name]=torch.zeros_like(u.state[name])
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    aggregate_fedavg(s,[u],expected_counts={'a':1},client_order=('a',))

    def test_nonfloating_trainable_and_scaler_roles_rejected(self):
        for name in ('counter','valid','target_mean','target_std'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):RoundSnapshot(snapshot().state_dict(),(name,))

    def test_no_global_or_update_mutation_and_completion_order_independence(self):
        s=snapshot();before=s.state_dict();a=update(s,'a',1,2);b=update(s,'b',3,6)
        a_before={n:t.clone() for n,t in a.state.items()}
        first=aggregate_fedavg(s,[a,b],expected_counts={'a':1,'b':3},client_order=('a','b'))
        second=aggregate_fedavg(s,[b,a],expected_counts={'a':1,'b':3},client_order=('a','b'))
        equal(self,first,second);equal(self,before,s.state_dict());equal(self,a_before,a.state)

    def test_snapshot_has_no_tensor_aliases_or_public_metadata_setters(self):
        s=snapshot();state=s.state_dict();state['weight'].add_(50)
        self.assertEqual(s.state_dict()['weight'][0].item(),0)
        for key in ('round_id','fingerprint','trainable_keys','fixed_keys'):
            with self.subTest(key=key):
                with self.assertRaises(AttributeError):setattr(s,key,None)

    def test_shapes_dtypes_keys_nonfinite_and_grad_state_rejected(self):
        s=snapshot()
        bads=[torch.ones(3,dtype=torch.float64),torch.ones(2,dtype=torch.float32),
              torch.full((2,),np.nan,dtype=torch.float64),torch.full((2,),np.inf,dtype=torch.float64),
              torch.ones(2,dtype=torch.float64,requires_grad=True),torch.ones(2,dtype=torch.complex64)]
        for value in bads:
            u=update(s,'a',1,1);u.state['weight']=value
            with self.subTest(dtype=value.dtype,shape=value.shape):
                with self.assertRaises(ValueError):aggregate_fedavg(s,[u],expected_counts={'a':1},client_order=('a',))
        for operation in ('missing','extra'):
            u=update(s,'a',1,1)
            if operation=='missing':del u.state['weight']
            else:u.state['extra']=torch.zeros(1)
            with self.assertRaises(ValueError):aggregate_fedavg(s,[u],expected_counts={'a':1},client_order=('a',))

    def test_invalid_counts_missing_duplicate_unexpected_and_stale_rejected(self):
        s=snapshot();a=update(s,'a',1,1);b=update(s,'b',2,2)
        for values in ([a],[a,a],[a,replace(b,client_id='other')],[a,replace(b,base_fingerprint='old')]):
            with self.assertRaises(ValueError):aggregate_fedavg(s,values,expected_counts={'a':1,'b':2},client_order=('a','b'))
        for count in (0,-1,True,np.bool_(True),1.5,2**53+1):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):aggregate_fedavg(s,[replace(a,num_examples=count)],expected_counts={'a':count},client_order=('a',))
        with self.assertRaises(ValueError):aggregate_fedavg(s,[a],expected_counts={'a':2},client_order=('a',))
        with self.assertRaises(ValueError):aggregate_fedavg(s,[a,b],expected_counts={'a':1,'b':2},client_order=('a','a'))

    def test_supported_float_dtypes_and_fixed_negative_zero(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            s=snapshot(dtype);out=aggregate_fedavg(s,[update(s,'a',1,2),update(s,'b',1,4)],
                                                 expected_counts={'a':1,'b':1},client_order=('a','b'))
            self.assertTrue(torch.equal(out['weight'],torch.full((2,),3.,dtype=dtype)))
        state=snapshot().state_dict();state['target_mean'][0]=-0.
        s=RoundSnapshot(state,('weight','embedding'));u=update(s,'a',1,1)
        u.state['target_mean'][0]=0.
        with self.assertRaises(ValueError):aggregate_fedavg(s,[u],expected_counts={'a':1},client_order=('a',))

    def test_real_adapter_state_schema_from_fresh_synthetic_initialization(self):
        m=B1TwoOutputAdapter(node_names=NAMES,expected_mapping_digest=mapping_digest(NAMES),
                             target_scaling=TargetScaling((300,20),(100,5)),seed=42)
        before={n:p.detach().clone() for n,p in m.named_parameters()}
        raw=m.state_dict();s=RoundSnapshot(raw,tuple(n for n,p in m.named_parameters() if p.requires_grad))
        u=ClientStateUpdate('invented-a',1,s.fingerprint,s.state_dict())
        out=aggregate_fedavg(s,[u],expected_counts={'invented-a':1},client_order=('invented-a',))
        equal(self,raw,out)
        self.assertEqual(out['core.node_embedding.weight'].shape,(979,8))
        self.assertEqual(out['core.head.weight'].shape,(2,64))
        self.assertEqual(s.fixed_keys,('target_mean','target_std'))
        for n,p in m.named_parameters():self.assertTrue(torch.equal(p,before[n]))
        # No load_state_dict, forward, loss, backward or optimizer call.

    def test_candidate_protocol_constants_and_rejected_changes(self):
        self.assertEqual(CandidateConfig().batch_size,128)
        for kw in ({'batch_size':512},{'local_epochs':2},{'rounds':201},{'clients':978},
                   {'drop_last':True},{'seeds':(42.,43.,44.)}):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):CandidateConfig(**kw)

    def test_registry_exact_mapping_training_counts_only(self):
        r=registry();self.assertEqual(r.node_names,NAMES);self.assertEqual(r.count(0),148)
        counts={n:148 for n in NAMES};del counts[NAMES[0]]
        with self.assertRaises(ValueError):ClientRegistry(NAMES,counts,expected_mapping_digest=mapping_digest(NAMES))
        with self.assertRaises(ValueError):ClientRegistry(tuple(reversed(NAMES)),{n:1 for n in NAMES},expected_mapping_digest=mapping_digest(NAMES))
        self.assertFalse(hasattr(r,'validation'));self.assertFalse(hasattr(r,'test'))

    def test_sampling_determinism_population_unique_and_round_domains(self):
        r=registry();initial=np.random.get_state();a=sample_clients(r,42,1)
        self.assertEqual(len(a),98);self.assertEqual(len(set(a)),98)
        self.assertTrue(all(0<=i<979 for i in a));self.assertEqual(a,sample_clients(r,42,1))
        self.assertNotEqual(a,sample_clients(r,43,1));self.assertNotEqual(a,sample_clients(r,42,2))
        after=np.random.get_state();self.assertTrue(np.array_equal(initial[1],after[1]))
        for seed,round_id in ((True,1),(42,0),(42,201),(45,1),(42,1.5)):
            with self.assertRaises(ValueError):sample_clients(r,seed,round_id)

    def test_partial_batch_and_local_step_accounting(self):
        for n,expected in ((1,(1,)),(128,(128,)),(148,(128,20)),(256,(128,128)),(437,(128,128,128,53))):
            self.assertEqual(batch_sizes(n),expected);self.assertEqual(sum(expected),n)
            self.assertEqual(len(expected),(n+127)//128)

    def test_permutation_once_and_separate_rng_stream(self):
        r=registry();s=snapshot();p=plan_round(s,r,42,1)
        for session in p.sessions:
            order=local_row_order(session)
            self.assertTrue(np.array_equal(np.sort(order),np.arange(session.training_windows)))
            self.assertTrue(np.array_equal(order,local_row_order(session)))
            self.assertFalse(order.flags.writeable)
            self.assertEqual(sum(session.batch_sizes),session.training_windows)
        self.assertNotEqual(p.sessions[0].shuffle_seed,p.sessions[1].shuffle_seed)

    def test_fresh_optimizer_plans_no_optimizer_execution(self):
        r=registry();s=snapshot();p=plan_round(s,r,42,1)
        self.assertEqual(len({x.session_id for x in p.sessions}),98)
        descriptor=p.sessions[0].optimizer_spec;descriptor['momentum']=10
        self.assertEqual(p.sessions[0].optimizer_spec['momentum'],0)
        for session in p.sessions:
            self.assertEqual(session.optimizer_spec['initial_state_entries'],0)
            self.assertTrue(session.optimizer_spec['fresh_each_client_round'])
            self.assertFalse(session.optimizer_spec['execution_implemented'])

    def test_round_rejects_missing_invalid_steps_stale_and_changed_plan(self):
        s=snapshot();r=registry();p=plan_round(s,r,42,1)
        updates=[scripted_update(x,s.state_dict()) for x in p.sessions]
        with self.assertRaises(ValueError):commit_scripted_round(s,r,p,updates[:-1])
        for changes in ({'planned_steps':1},{'planned_examples':1},{'executed_optimizer_steps':1},{'session_id':'old'}):
            altered=[replace(updates[0],**changes),*updates[1:]]
            with self.assertRaises(ValueError):commit_scripted_round(s,r,p,altered)
        with self.assertRaises(ValueError):commit_scripted_round(s,r,replace(p,sessions=tuple(reversed(p.sessions))),updates)
        stale=replace(updates[0],update=replace(updates[0].update,base_fingerprint='old'))
        with self.assertRaises(ValueError):commit_scripted_round(s,r,p,[stale,*updates[1:]])
        equal(self,s.state_dict(),snapshot().state_dict())

    def test_multi_round_scripted_execution_all_three_seeds(self):
        r=registry()
        for seed in (42,43,44):
            with self.subTest(seed=seed):
                start=snapshot();before=start.state_dict();starts={}
                def factory(session,state):
                    starts.setdefault(session.round_id,[]).append(state['weight'].clone())
                    state['weight']=state['weight']+(session.node_index%7+1)
                    state['embedding'][session.node_index]+=8.
                    return state
                end,receipts=run_scripted_rounds(start,r,seed=seed,round_count=3,update_factory=factory)
                expected=0.
                for record in receipts:
                    round_id=record['round'];participants=sample_clients(r,seed,round_id)
                    for initial in starts[round_id]:self.assertAlmostEqual(initial[0].item(),expected,places=10)
                    total=sum(r.count(i) for i in participants)
                    expected+=sum(r.count(i)*(i%7+1) for i in participants)/total
                    self.assertEqual(record['planned_examples'],total)
                    self.assertEqual(record['planned_optimizer_steps'],sum((r.count(i)+127)//128 for i in participants))
                    self.assertAlmostEqual(sum(v['weight'] for v in record['clients']),1.)
                    self.assertEqual(record['executed_optimizer_steps'],0)
                    self.assertEqual(record['actual_examples_trained'],0)
                    self.assertEqual(record['optimizer_instances_created'],0)
                self.assertAlmostEqual(end.state_dict()['weight'][0].item(),expected,places=10)
                self.assertEqual(end.round_id,3);equal(self,before,start.state_dict())
                self.assertEqual(receipts[0]['result_fingerprint'],receipts[1]['base_fingerprint'])

    def test_repeated_scripted_run_is_exact_and_failures_do_not_mutate_input(self):
        r=registry();s=snapshot()
        def no_change(session,state):return state
        a,ra=run_scripted_rounds(s,r,seed=42,round_count=2,update_factory=no_change)
        b,rb=run_scripted_rounds(s,r,seed=42,round_count=2,update_factory=no_change)
        equal(self,a.state_dict(),b.state_dict());self.assertEqual(ra,rb)
        def bad(session,state):state['weight'][0]=np.inf;return state
        with self.assertRaises(ValueError):run_scripted_rounds(s,r,seed=42,round_count=1,update_factory=bad)
        self.assertEqual(s.state_dict()['weight'][0].item(),0.)
        with self.assertRaises(ValueError):run_scripted_rounds(a,r,seed=42,round_count=1,update_factory=no_change)

    def test_participation_accounting_over_200_rounds(self):
        r=registry()
        for seed in (42,43,44):
            summary=participation_summary(r,seed)
            counts=summary['participations_by_node_index']
            self.assertEqual(sum(counts),19600)
            self.assertEqual(summary,participation_summary(r,seed))
            self.assertEqual(counts[0],sum(0 in sample_clients(r,seed,k) for k in range(1,201)))
            self.assertAlmostEqual(summary['expected_participations_per_client'],200*98/979)
            self.assertEqual(summary['planned_optimizer_steps'],sum(counts[i]*((r.count(i)+127)//128) for i in range(979)))
            self.assertEqual(summary['actual_examples_trained'],0)


if __name__=='__main__':unittest.main()
