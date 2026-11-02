"""Aggregation tests use manually changed tensors; no optimizer steps or training."""
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.model import SequenceRegressor
from src.federated_train import ClientUpdate, GlobalModelState, aggregate_fedavg


def model(dtype=torch.float32):
    torch.manual_seed(42)
    result=SequenceRegressor(14,1,hidden_size=4,num_layers=2,model='gru',node_count=3,
                             node_embedding_size=2,baseline_coefficients=[2.,-.5])
    return result.to(dtype=dtype)


def update(round_state,template,client_id,count,delta):
    client=round_state.initialize_client(template)
    with torch.no_grad():
        for parameter in client.parameters():
            if parameter.requires_grad:parameter.add_(delta)
    return round_state.capture_update(client_id,client,count)


def assert_state_equal(test,a,b):
    test.assertEqual(tuple(a),tuple(b))
    for name in a:
        test.assertEqual(a[name].dtype,b[name].dtype)
        test.assertTrue(torch.equal(a[name],b[name]),name)


class FedAvgTests(unittest.TestCase):
    def test_unequal_counts_match_weighted_average_for_every_trainable_tensor(self):
        template=model(torch.float64);state=GlobalModelState(template)
        a=update(state,template,'a',1,2.);b=update(state,template,'b',3,6.)
        averaged=aggregate_fedavg(state,[a,b],expected_counts={'a':1,'b':3})
        base=state.state_dict()
        for name in state.trainable_keys:
            torch.testing.assert_close(averaged[name],base[name]+5,rtol=1e-14,atol=1e-14)
        assert_state_equal(self,{name:averaged[name] for name in state.fixed_keys},
                           {name:base[name] for name in state.fixed_keys})

    def test_count_weights_are_not_equal_client_weights(self):
        template=model();state=GlobalModelState(template)
        a=update(state,template,'a',2,1.);b=update(state,template,'b',6,3.)
        result=aggregate_fedavg(state,[a,b],expected_counts={'a':2,'b':6,'unselected':400})
        torch.testing.assert_close(result['head.bias'],torch.tensor([2.5]))

    def test_single_client_is_exact_identity_without_aliases(self):
        for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
            with self.subTest(dtype=dtype):
                template=model(dtype);state=GlobalModelState(template)
                a=update(state,template,'a',np.int64(3),.125)
                result=aggregate_fedavg(state,[a],expected_counts={'a':3})
                assert_state_equal(self,result,a.state)
                for name in result:self.assertNotEqual(result[name].data_ptr(),a.state[name].data_ptr())
        template=model(torch.float64)
        with torch.no_grad():template.head.bias.fill_(1e100)
        state=GlobalModelState(template);worker=state.initialize_client(template)
        with torch.no_grad():worker.head.bias.fill_(1.)
        result=aggregate_fedavg(state,[state.capture_update('a',worker,1)],expected_counts={'a':1})
        self.assertEqual(result['head.bias'].item(),1.)

    def test_global_model_snapshot_updates_and_result_are_independent(self):
        template=model();initial={name:value.clone() for name,value in template.state_dict().items()}
        state=GlobalModelState(template);fingerprint=state.fingerprint
        a=update(state,template,'a',2,1.);b=update(state,template,'b',3,2.)
        saved_a={name:value.clone() for name,value in a.state.items()}
        result=aggregate_fedavg(state,[a,b],expected_counts={'a':2,'b':3})
        assert_state_equal(self,template.state_dict(),initial)
        assert_state_equal(self,state.state_dict(),initial)
        assert_state_equal(self,a.state,saved_a)
        for tensor in result.values():tensor.add_(99)
        exposed=state.state_dict()
        for tensor in exposed.values():tensor.add_(100)
        assert_state_equal(self,state.state_dict(),initial)
        self.assertEqual(state.fingerprint,fingerprint)

    def test_every_client_starts_from_same_snapshot_even_after_template_mutation(self):
        template=model();state=GlobalModelState(template);expected=state.state_dict()
        with torch.no_grad():
            for tensor in template.state_dict().values():tensor.add_(100)
        first=state.initialize_client(template)
        with torch.no_grad():first.head.weight.add_(9)
        second=state.initialize_client(template)
        assert_state_equal(self,second.state_dict(),expected)
        assert_state_equal(self,state.state_dict(),expected)
        self.assertTrue(second.training)
        self.assertTrue(all(parameter.grad is None for parameter in second.parameters()))
        self.assertNotEqual(first.head.weight.data_ptr(),second.head.weight.data_ptr())
        with self.assertRaisesRegex(ValueError,'must be initialized'):
            state.capture_update('uninitialized',model(),5)

    def test_shared_embedding_rows_receive_standard_weighted_deltas(self):
        template=model();state=GlobalModelState(template)
        first=state.initialize_client(template);second=state.initialize_client(template)
        with torch.no_grad():
            first.node_embedding.weight[0].add_(4)
            second.node_embedding.weight[1].add_(8)
        a=state.capture_update('a',first,1);b=state.capture_update('b',second,3)
        result=aggregate_fedavg(state,[a,b],expected_counts={'a':1,'b':3})
        base=state.state_dict()['node_embedding.weight'];embedding=result['node_embedding.weight']
        torch.testing.assert_close(embedding[0],base[0]+1)
        torch.testing.assert_close(embedding[1],base[1]+6)
        self.assertTrue(torch.equal(embedding[2],base[2]))
        self.assertFalse(torch.equal(embedding[0],first.node_embedding.weight[0]))

    def test_fixed_parameters_and_integer_buffers_remain_exactly_fixed(self):
        template=model();template.head.bias.requires_grad_(False)
        template.register_buffer('completed_rounds',torch.tensor(0,dtype=torch.int64))
        state=GlobalModelState(template)
        self.assertIn('head.bias',state.fixed_keys)
        self.assertIn('baseline_coefficients',state.fixed_keys)
        a=update(state,template,'a',2,.5);b=update(state,template,'b',3,1.)
        result=aggregate_fedavg(state,[a,b],expected_counts={'a':2,'b':3})
        for name in state.fixed_keys:self.assertTrue(torch.equal(result[name],state.state_dict()[name]))
        for name in state.fixed_keys:
            broken=OrderedDict((k,v.clone()) for k,v in a.state.items());broken[name].add_(1)
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'fixed state changed'):
                aggregate_fedavg(state,[replace(a,state=broken),b],expected_counts={'a':2,'b':3})
        worker=state.initialize_client(template)
        with torch.no_grad():worker.baseline_coefficients[0]+=1
        with self.assertRaisesRegex(ValueError,'fixed state changed'):state.capture_update('a',worker,2)

    def test_fixed_signed_zero_is_verified_bit_for_bit(self):
        template=model()
        with torch.no_grad():template.baseline_coefficients[1]=-0.
        state=GlobalModelState(template);a=update(state,template,'a',1,1)
        broken=OrderedDict((name,value.clone()) for name,value in a.state.items())
        broken['baseline_coefficients'][1]=0.
        with self.assertRaisesRegex(ValueError,'fixed state changed'):
            aggregate_fedavg(state,[replace(a,state=broken)],expected_counts={'a':1})

    def test_deterministic_aggregation_and_completion_order_independence(self):
        template=model();state=GlobalModelState(template)
        updates=[update(state,template,'z',7,.03),update(state,template,'a',3,-.02),update(state,template,'q',5,.12)]
        counts={'z':7,'a':3,'q':5}
        first=aggregate_fedavg(state,updates,expected_counts=counts)
        for candidates in (updates,list(reversed(updates)),[updates[1],updates[0],updates[2]]):
            assert_state_equal(self,first,aggregate_fedavg(state,candidates,expected_counts=counts))

    def test_missing_extra_names_invalid_shapes_dtypes_and_values_rejected(self):
        template=model();state=GlobalModelState(template);a=update(state,template,'a',1,.5)
        variants={}
        missing=dict(a.state);missing.pop('head.weight');variants['missing']=missing
        extra=dict(a.state);extra['foreign']=torch.tensor(0.);variants['extra']=extra
        shape=dict(a.state);shape['head.weight']=torch.zeros(2,4);variants['shape']=shape
        dtype=dict(a.state);dtype['head.weight']=a.state['head.weight'].double();variants['dtype']=dtype
        not_tensor=dict(a.state);not_tensor['head.weight']=np.zeros((1,4));variants['not_tensor']=not_tensor
        for label,value in (('nan',float('nan')),('inf',float('inf')),('negative_inf',-float('inf'))):
            wrong=dict(a.state);wrong['head.weight']=torch.full_like(a.state['head.weight'],value);variants[label]=wrong
        for label,values in variants.items():
            with self.subTest(label=label),self.assertRaises(ValueError):
                aggregate_fedavg(state,[replace(a,state=values)],expected_counts={'a':1})

    def test_invalid_counts_empty_duplicate_unknown_or_stale_clients_rejected(self):
        template=model();state=GlobalModelState(template);a=update(state,template,'a',1,.5)
        for bad in (0,-1,True,np.bool_(True),1.5,float('nan'),float('inf'),'1',None,2**54):
            with self.subTest(count=bad),self.assertRaises(ValueError):
                aggregate_fedavg(state,[replace(a,num_examples=bad)],expected_counts={'a':1})
            with self.subTest(expected=bad),self.assertRaises(ValueError):
                aggregate_fedavg(state,[a],expected_counts={'a':bad})
        for values,counts in (([],{'a':1}),([a,a],{'a':1}),([a],{'b':1}),([a],{'a':2}),([a],{})):
            with self.subTest(values=len(values),counts=counts),self.assertRaises(ValueError):
                aggregate_fedavg(state,values,expected_counts=counts)
        for bad_id in ('',1,None):
            with self.subTest(client=bad_id),self.assertRaises(ValueError):
                aggregate_fedavg(state,[replace(a,client_id=bad_id)],expected_counts={'a':1})
        with self.assertRaisesRegex(ValueError,'different global round'):
            aggregate_fedavg(state,[replace(a,base_fingerprint='wrong')],expected_counts={'a':1})
        different=model()
        with torch.no_grad():different.head.weight.add_(1)
        next_round=GlobalModelState(different)
        with self.assertRaisesRegex(ValueError,'different global round'):
            aggregate_fedavg(next_round,[a],expected_counts={'a':1})

    def test_round_ids_prevent_replay_even_when_model_weights_are_identical(self):
        template=model()
        first=GlobalModelState(template,round_id=7)
        second=GlobalModelState(template,round_id=8)
        assert_state_equal(self,first.state_dict(),second.state_dict())
        self.assertNotEqual(first.fingerprint,second.fingerprint)
        self.assertEqual(first.round_id,7)
        captured=update(first,template,'a',1,.01)
        with self.assertRaisesRegex(ValueError,'different global round'):
            aggregate_fedavg(second,[captured],expected_counts={'a':1})
        for invalid in (-1,True,1.5,'1'):
            with self.subTest(round_id=invalid),self.assertRaises(ValueError):GlobalModelState(template,round_id=invalid)

    def test_snapshot_rejects_non_gru_missing_baseline_or_nonfinite_global_state(self):
        bad_models=[nn.Linear(3,1),nn.GRU(3,4)]
        nonfinite=model()
        with torch.no_grad():nonfinite.head.weight.fill_(float('nan'))
        bad_models.append(nonfinite)
        trainable_baseline=model()
        del trainable_baseline.baseline_coefficients
        trainable_baseline.baseline_coefficients=nn.Parameter(torch.tensor([2.,-.5]))
        bad_models.append(trainable_baseline)
        for candidate in bad_models:
            with self.subTest(model=type(candidate)),self.assertRaises(ValueError):GlobalModelState(candidate)

    def test_client_template_schema_and_trainable_roles_cannot_change(self):
        template=model();state=GlobalModelState(template)
        incompatible=SequenceRegressor(14,1,hidden_size=5,model='gru',node_count=3,baseline_coefficients=[2.,-.5])
        with self.assertRaises(ValueError):state.initialize_client(incompatible)
        wrong_roles=model();wrong_roles.head.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError,'roles differ'):state.initialize_client(wrong_roles)
        worker=state.initialize_client(template);worker.head.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError,'roles changed'):state.capture_update('a',worker,1)
        with self.assertRaises(ValueError):aggregate_fedavg(None,[],expected_counts={'a':1})

    def test_capture_update_detaches_from_later_worker_changes_and_fingerprints_are_repeatable(self):
        template=model();state=GlobalModelState(template)
        self.assertEqual(state.fingerprint,GlobalModelState(model()).fingerprint)
        worker=state.initialize_client(template)
        captured=state.capture_update('a',worker,1)
        with torch.no_grad():worker.head.weight.add_(1)
        self.assertTrue(torch.equal(captured.state['head.weight'],state.state_dict()['head.weight']))
        self.assertFalse(captured.state['head.weight'].requires_grad)


if __name__=='__main__':unittest.main()
