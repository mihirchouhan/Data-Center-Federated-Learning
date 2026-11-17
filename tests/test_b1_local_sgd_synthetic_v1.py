"""Actual CPU SGD on invented fixtures only; no real data/checkpoints/scoring."""
from dataclasses import replace
import unittest
from unittest import mock

import numpy as np
import torch

from src import b1_local_sgd_synthetic_v1 as sgd
from src import b1_target_validation_v1 as target_validation
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_fedavg_orchestration_v1 import RoundSnapshot


def model(seed=42):
    return sgd.make_synthetic_model(target_scaling=TargetScaling((300.,20.),(100.,5.)),seed=seed)


def same(test,a,b):
    test.assertEqual(tuple(a),tuple(b))
    for name in a:
        test.assertEqual(a[name].dtype,b[name].dtype)
        test.assertEqual(a[name].shape,b[name].shape)
        test.assertTrue(torch.equal(a[name],b[name]),name)


def collect_optimizers():
    optimizers=[];original=sgd._new_optimizer
    def factory(parameters):
        opt=original(parameters);optimizers.append(opt);return opt
    return optimizers,factory


class B1SyntheticLocalSGDTests(unittest.TestCase):
    def test_sample_counts_less_equal_greater_and_partial_final_batch(self):
        m=model();initial={n:v.clone() for n,v in m.state_dict().items()}
        for count,batches in ((17,(17,)),(128,(128,)),(129,(128,1)),(257,(128,128,1))):
            with self.subTest(samples=count):
                result=sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,count))
                self.assertEqual(result.receipt['actual_batch_sizes'],batches)
                self.assertEqual(result.receipt['examples_processed'],count)
                self.assertEqual(result.update.num_examples,count)
                self.assertEqual(result.receipt['optimizer_steps'],len(batches))
                self.assertEqual(result.receipt['local_epochs'],1)
                self.assertFalse(result.receipt['drop_last'])
        same(self,initial,m.state_dict())

    def test_actual_parameter_and_embedding_changes_and_finite_gradients(self):
        m=model();initial=m.state_dict();result=sgd.train_synthetic_client(m,sgd.make_synthetic_client(7,129))
        receipt=result.receipt
        self.assertTrue(receipt['parameter_changed_per_step'])
        self.assertTrue(all(receipt['parameter_changed_per_step']))
        self.assertGreater(receipt['parameter_delta_l2'],0)
        self.assertTrue(any(n.startswith('core.recurrent.') for n in receipt['changed_parameter_names']))
        self.assertIn('core.head.weight',receipt['changed_parameter_names'])
        self.assertIn('core.node_embedding.weight',receipt['changed_parameter_names'])
        for check in receipt['gradient_checks']:
            self.assertTrue(check['all_finite']);self.assertGreater(check['l2'],0)
            self.assertGreater(check['owner_embedding_l2'],0);self.assertEqual(check['other_embedding_nonzero'],0)
        before=initial['core.node_embedding.weight'];after=result.update.state['core.node_embedding.weight']
        self.assertFalse(torch.equal(after[7],before[7]))
        other=torch.arange(979)!=7;self.assertTrue(torch.equal(after[other],before[other]))
        for n in ('target_mean','target_std'):self.assertTrue(torch.equal(initial[n],result.update.state[n]))
        self.assertTrue(all(np.isfinite(v) for v in receipt['finite_batch_losses']))
        self.assertTrue(all(not v.requires_grad for v in result.update.state.values()))

    def test_global_parameters_gradients_modes_and_fixture_unchanged(self):
        m=model();c=sgd.make_synthetic_client(0,17)
        before={n:t.clone() for n,t in m.state_dict().items()};x=c.inputs.clone();f=c.future.copy();o=c.observed.copy()
        sgd.train_synthetic_client(m,c)
        same(self,before,m.state_dict());self.assertFalse(m.training)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertTrue(torch.equal(x,c.inputs));self.assertIsNone(c.inputs.grad)
        self.assertTrue(np.array_equal(f,c.future));self.assertTrue(np.array_equal(o,c.observed))

    def test_exact_deterministic_local_updates_under_fixed_seed(self):
        m=model();c=sgd.make_synthetic_client(3,129)
        first=sgd.train_synthetic_client(m,c,seed=43);second=sgd.train_synthetic_client(m,c,seed=43)
        same(self,first.update.state,second.update.state);self.assertEqual(first.receipt,second.receipt)

    def test_shuffle_seed_and_round_domains_differ(self):
        m=model();c=sgd.make_synthetic_client(3,129)
        a=sgd.train_synthetic_client(m,c,seed=42,round_id=1)
        b=sgd.train_synthetic_client(m,c,seed=43,round_id=1)
        c2=sgd.train_synthetic_client(m,c,seed=42,round_id=2)
        self.assertNotEqual(a.receipt['shuffle_seed'],b.receipt['shuffle_seed'])
        self.assertNotEqual(a.receipt['row_order_sha256'],b.receipt['row_order_sha256'])
        self.assertNotEqual(a.receipt['session_id'],c2.receipt['session_id'])
        self.assertTrue(any(not torch.equal(a.update.state[n],b.update.state[n]) for n in a.update.state))

    def test_fresh_optimizer_objects_and_parameter_ownership_each_session(self):
        m=model();opts,factory=collect_optimizers()
        with mock.patch.object(sgd,'_new_optimizer',side_effect=factory):
            a=sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17),round_id=1)
            b=sgd.train_synthetic_client(m,sgd.make_synthetic_client(1,17),round_id=1)
            c=sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17),round_id=2)
        self.assertEqual(len(opts),3);self.assertEqual(len({id(o) for o in opts}),3)
        groups=[{id(p) for p in o.param_groups[0]['params']} for o in opts]
        self.assertTrue(groups[0].isdisjoint(groups[1]));self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(all(not o.state for o in opts))
        self.assertEqual(len({id(o.state) for o in opts}),3)
        for result in (a,b,c):
            self.assertEqual(result.receipt['optimizer_initial_state_entries'],0)
            self.assertEqual(result.receipt['optimizer_final_state_entries'],0)
            self.assertEqual(result.receipt['optimizer']['learning_rate'],.01)
            self.assertEqual(result.receipt['optimizer']['momentum'],0)
            self.assertEqual(result.receipt['optimizer']['weight_decay'],0)

    def test_accidentally_reused_optimizer_rejected_even_if_state_is_empty(self):
        m=model();opts,factory=collect_optimizers()
        with mock.patch.object(sgd,'_new_optimizer',side_effect=factory):
            sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17))
        self.assertFalse(opts[0].state)
        previous=[p.detach().clone() for p in opts[0].param_groups[0]['params']]
        with mock.patch.object(sgd,'_new_optimizer',return_value=opts[0]):
            with self.assertRaisesRegex(ValueError,'ownership'):
                sgd.train_synthetic_client(m,sgd.make_synthetic_client(1,17))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(previous,opts[0].param_groups[0]['params'])))

    def test_nonempty_optimizer_or_changed_settings_rejected(self):
        m=model();original=sgd._new_optimizer
        for change in ('state','lr','momentum','weight_decay'):
            def factory(parameters):
                opt=original(parameters)
                if change=='state':opt.state[opt.param_groups[0]['params'][0]]={'invented':torch.tensor(1.)}
                else:opt.param_groups[0][change]=.5
                return opt
            with self.subTest(change=change),mock.patch.object(sgd,'_new_optimizer',side_effect=factory):
                with self.assertRaises(ValueError):sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17))

    def test_hardened_target_helper_and_supplied_scaler_are_used(self):
        m=model();c=sgd.make_synthetic_client(0,17)
        with mock.patch.object(target_validation,'target_means',wraps=target_validation.target_means) as helper:
            result=sgd.train_synthetic_client(m,c)
        self.assertGreaterEqual(helper.call_count,2)
        self.assertEqual(result.receipt['target_scaling_digest'],m.target_scaling.digest)
        self.assertEqual(m.target_scaling.mean,(300.,20.));self.assertEqual(m.target_scaling.std,(100.,5.))
        self.assertFalse(m.target_scaling.record()['fit_performed_by_this_object'])

    def test_masked_and_boolean_targets_fail_before_optimizer_construction(self):
        m=model();c=sgd.make_synthetic_client(0,129)
        mask=np.zeros(c.future.shape,dtype=bool);mask[-1,0,0]=True
        for future in (np.ma.array(c.future,mask=False),np.ma.array(c.future,mask=mask),c.future.astype(bool)):
            with self.subTest(kind=type(future)),mock.patch.object(sgd,'_new_optimizer',side_effect=AssertionError('optimizer reached')):
                with self.assertRaises(ValueError):sgd.train_synthetic_client(m,replace(c,future=future))

    def test_missing_nonfinite_and_late_invalid_targets_fail_before_any_step(self):
        m=model();c=sgd.make_synthetic_client(0,129)
        bad=[]
        for val in (np.nan,np.inf,-np.inf):
            f=c.future.copy();f[-1,-1,0]=val;bad.append(replace(c,future=f))
        missing=c.observed.copy();missing[-1,-1,1]=False;bad.append(replace(c,observed=missing))
        bad.append(replace(c,observed=np.ma.array(c.observed,mask=False)))
        for client in bad:
            with mock.patch.object(sgd,'_new_optimizer',side_effect=AssertionError('optimizer reached')):
                with self.assertRaises(ValueError):sgd.train_synthetic_client(m,client)

    def test_invalid_input_nodes_shapes_counts_and_values_rejected_before_optimizer(self):
        m=model();c=sgd.make_synthetic_client(0,17);cases=[]
        for val in (1.,.5,np.nan):
            x=c.inputs.clone();x[-1,-1,13]=val;cases.append(replace(c,inputs=x))
        cases.extend([replace(c,inputs=c.inputs.double()),replace(c,inputs=c.inputs[:,:6]),
                      replace(c,inputs=c.inputs[:1]),replace(c,node_index=True)])
        for case in cases:
            with mock.patch.object(sgd,'_new_optimizer',side_effect=AssertionError('optimizer reached')):
                with self.assertRaises(ValueError):sgd.train_synthetic_client(m,case)

    def test_nonfinite_loss_rejected_without_parameter_step(self):
        m=model();opts,initials=[],[];original=sgd._new_optimizer
        def factory(params):
            opt=original(params);opts.append(opt);initials.append([p.detach().clone() for p in opt.param_groups[0]['params']]);return opt
        with mock.patch.object(sgd,'_new_optimizer',side_effect=factory),mock.patch.object(sgd.B1TwoOutputAdapter,'loss',return_value=torch.tensor(np.nan)):
            with self.assertRaises(ValueError):sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17))
        self.assertEqual(len(opts),1)
        self.assertTrue(all(torch.equal(p,b) for p,b in zip(opts[0].param_groups[0]['params'],initials[0])))

    def test_nonfinite_gradient_rejected_before_parameter_step(self):
        m=model();opts,initials=[],[];original=sgd._new_optimizer
        def factory(params):
            opt=original(params);opts.append(opt);parameters=opt.param_groups[0]['params']
            initials.append([p.detach().clone() for p in parameters])
            parameters[0].register_hook(lambda grad:torch.full_like(grad,np.inf));return opt
        with mock.patch.object(sgd,'_new_optimizer',side_effect=factory):
            with self.assertRaisesRegex(ValueError,'gradient'):
                sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,17))
        self.assertTrue(all(torch.equal(p,b) for p,b in zip(opts[0].param_groups[0]['params'],initials[0])))

    def test_multi_client_actual_training_weighted_aggregation_and_dilution(self):
        for seed in (42,43,44):
            with self.subTest(seed=seed):
                m=model(seed);initial=m.state_dict();a=sgd.make_synthetic_client(0,17);b=sgd.make_synthetic_client(978,129)
                result=sgd.train_synthetic_round(m,(a,b),seed=seed)
                receipt=result['receipt'];local=result['local_results'];out=result['state']
                self.assertEqual(receipt['examples_processed'],146);self.assertEqual(receipt['optimizer_steps'],3)
                self.assertAlmostEqual(receipt['weights'][a.client_id],17/146)
                self.assertEqual(local[0].receipt['base_fingerprint'],local[1].receipt['base_fingerprint'])
                for name in out:
                    if name in ('target_mean','target_std'):self.assertTrue(torch.equal(out[name],initial[name]));continue
                    expected=(local[0].update.state[name].double()*17+local[1].update.state[name].double()*129)/146
                    torch.testing.assert_close(out[name],expected.to(out[name].dtype),rtol=1e-6,atol=1e-7)
                for owner,index in ((0,0),(1,978)):
                    before=initial['core.node_embedding.weight'][index].double()
                    delta_local=local[owner].update.state['core.node_embedding.weight'][index].double()-before
                    delta_global=out['core.node_embedding.weight'][index].double()-before
                    self.assertGreater(delta_local.norm().item(),0)
                    self.assertLess(delta_global.norm().item(),delta_local.norm().item())
                    torch.testing.assert_close(delta_global,delta_local*receipt['weights'][local[owner].update.client_id],rtol=.01,atol=2e-7)
                other=torch.arange(979);other=other[(other!=0)&(other!=978)]
                self.assertTrue(torch.equal(out['core.node_embedding.weight'][other],initial['core.node_embedding.weight'][other]))
                same(self,initial,m.state_dict());self.assertFalse(receipt['aggregate_assigned_to_model'])

    def test_full_round_determinism_and_exact_single_client_aggregate(self):
        m=model();clients=(sgd.make_synthetic_client(0,17),sgd.make_synthetic_client(2,17))
        a=sgd.train_synthetic_round(m,clients);b=sgd.train_synthetic_round(m,clients)
        same(self,a['state'],b['state']);self.assertEqual(a['receipt'],b['receipt'])
        one=sgd.train_synthetic_round(m,clients[:1]);same(self,one['state'],one['local_results'][0].update.state)

    def test_entire_round_prevalidates_targets_and_rejects_duplicate_or_large_cohort(self):
        m=model();c=sgd.make_synthetic_client(0,17);bad=replace(sgd.make_synthetic_client(1,17),future=np.ones((17,6,2),dtype=bool))
        for cohort in ((c,bad),(c,c),tuple(sgd.make_synthetic_client(i,1) for i in range(5))):
            with mock.patch.object(sgd,'_new_optimizer',side_effect=AssertionError('optimizer reached')):
                with self.assertRaises(ValueError):sgd.train_synthetic_round(m,cohort)

    def test_synthetic_scope_caps_mapping_and_explicit_scaling(self):
        for count in (0,513,True):
            with self.assertRaises(ValueError):sgd.make_synthetic_client(0,count)
        with self.assertRaises(ValueError):sgd.make_synthetic_model(target_scaling=None)
        m=model();m.node_names=tuple('different-%d'%i for i in range(979))
        with self.assertRaises(ValueError):sgd.train_synthetic_client(m,sgd.make_synthetic_client(0,1))


if __name__=='__main__':unittest.main()
