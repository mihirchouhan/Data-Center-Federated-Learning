"""CPU-only invented tensors; forward/backward tests need explicit permission.

No optimizers, optimizer steps, real arrays, files or checkpoints are used.
"""
import unittest
from unittest import mock
import numpy as np
import torch

from src.b1_model_adapter_v1 import B1TwoOutputAdapter, mapping_digest, standardized_mse
from src.b1_target_scaling_v1 import TargetScaling
from src import b1_target_validation_v1 as hardened


NAMES = tuple('synthetic-node-%04d' % i for i in range(979))


def model(seed=42, names=NAMES):
    return B1TwoOutputAdapter(node_names=names, expected_mapping_digest=mapping_digest(names),
        target_scaling=TargetScaling((300., 20.), (100., 5.)), seed=seed)


def inputs(count=2):
    # Every numerical value is invented, with an ID fixed across each sequence.
    x = torch.linspace(-.5, .5, count*36*14, dtype=torch.float32, device='cpu').reshape(count, 36, 14)
    x[:, :, 13] = torch.tensor([0 if i % 2 == 0 else 978 for i in range(count)], device='cpu')[:, None]
    return x


def future(count=2):
    f = np.broadcast_to([.35, 22.5], (count, 6, 2)).copy()
    return f, np.ones(f.shape, dtype=bool)


class B1ModelAdapterTests(unittest.TestCase):
    def test_candidate_architecture_and_nonresidual_state(self):
        m = model()
        self.assertIsInstance(m.core.recurrent, torch.nn.GRU)
        self.assertEqual(m.core.recurrent.hidden_size, 64)
        self.assertEqual(m.core.recurrent.num_layers, 2)
        self.assertEqual(m.core.recurrent.input_size, 21)
        self.assertEqual(m.core.recurrent.dropout, 0)
        self.assertEqual(tuple(m.core.node_embedding.weight.shape), (979, 8))
        self.assertEqual((m.core.head.in_features, m.core.head.out_features), (64, 2))
        self.assertFalse(m.core.residual)
        self.assertFalse(any('baseline_coefficients' in n for n in m.state_dict()))
        self.assertEqual(sum(p.numel() for p in m.parameters()), 49626)

    def test_forward_shape_dtype_and_finiteness(self):
        m = model()
        for count in (1, 2, 128):
            with self.subTest(batch=count):
                x = inputs(count); before = x.clone()
                z = m.forward_standardized(x); y = m(x)
                self.assertEqual(tuple(z.shape), (count, 2))
                self.assertEqual(tuple(y.shape), (count, 2))
                self.assertEqual(z.dtype, torch.float32)
                self.assertEqual(y.dtype, torch.float64)
                self.assertEqual(y.device.type, 'cpu')
                self.assertTrue(torch.isfinite(y).all().item())
                self.assertTrue(torch.equal(x, before))

    def test_physical_output_order_and_independent_inverse(self):
        m = model()
        with torch.no_grad():
            m.core.head.weight.zero_()
            m.core.head.bias.copy_(torch.tensor([.5, -1.]))
        y = m(inputs(1))
        self.assertTrue(torch.equal(y, torch.tensor([[350., 15.]], dtype=torch.float64)))
        self.assertEqual(m.metadata()['target_units'], ['W', 'degC'])

    def test_standardized_loss_hand_calculation_and_no_broadcast(self):
        p = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        t = torch.tensor([[0., 2.], [1., 4.]])
        loss = standardized_mse(p, t)
        self.assertEqual(loss.item(), 1.25)  # (1+0+4+0)/4
        loss.backward()
        self.assertTrue(torch.equal(p.grad, torch.tensor([[.5, 0.], [1., 0.]])))
        for bad in (torch.ones(2), torch.ones(2, 1), torch.ones(1, 2), torch.ones(4, 2)):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):standardized_mse(p, bad)

    def test_loss_uses_hardened_target_validation(self):
        m = model(); f, observed = future()
        with mock.patch.object(hardened, 'target_means', wraps=hardened.target_means) as called:
            loss = m.loss(inputs(), f, observed)
            self.assertTrue(torch.isfinite(loss).item())
            called.assert_called_once_with(f, observed)

    def test_integrated_loss_uses_each_target_scale(self):
        m = model(); f, observed = future(1)
        with torch.no_grad():
            m.core.head.weight.zero_()
            m.core.head.bias.copy_(torch.tensor([.5, -1.]))
        # Future350W/22.5C scales to(.5,.5); (0^2+(-1.5)^2)/2.
        self.assertEqual(m.loss(inputs(1), f, observed).item(), 1.125)

    def test_invalid_targets_fail_before_model_forward(self):
        m = model(); f, observed = future()
        bad = [np.ma.array(f, mask=False), np.ma.array(f, mask=True), f.astype(bool)]
        f_nan = f.copy(); f_nan[0, 0, 0] = np.nan; bad.append(f_nan)
        f_inf = f.copy(); f_inf[0, 0, 1] = np.inf; bad.append(f_inf)
        with mock.patch.object(m.core, 'forward', side_effect=AssertionError('invalid label reached model')):
            for value in bad:
                with self.subTest(kind=type(value)):
                    with self.assertRaises(ValueError):m.loss(inputs(), value, observed)
            missing = observed.copy(); missing[0, 0, 1] = False
            with self.assertRaises(ValueError):m.loss(inputs(), f, missing)

    def test_target_batch_mismatch_rejected_before_forward(self):
        m = model(); f, observed = future(1)
        with mock.patch.object(m.core, 'forward', side_effect=AssertionError('bad batch reached model')):
            with self.assertRaises(ValueError):m.loss(inputs(2), f, observed)

    def test_mapping_order_fingerprint_and_lookup_preserved(self):
        names = list(NAMES); names[0], names[1] = names[1], names[0]
        m = model(names=names)
        self.assertEqual(m.node_names, tuple(names))
        self.assertEqual(m.node_index(names[0]), 0)
        self.assertEqual(m.node_index(names[1]), 1)
        self.assertEqual(m.mapping_digest, mapping_digest(names))
        with self.assertRaises(KeyError):m.node_index('unknown-client')
        with self.assertRaises(ValueError):
            B1TwoOutputAdapter(node_names=names, expected_mapping_digest=mapping_digest(NAMES),
                               target_scaling=TargetScaling((300, 20), (100, 5)))

    def test_invalid_mapping_or_seed_rejected(self):
        for names in (NAMES[:-1], NAMES+(NAMES[0],), (NAMES[0],)*979, 'arbitrary'):
            with self.subTest(mapping_type=type(names)):
                with self.assertRaises(ValueError):mapping_digest(names)
        for seed in (-1, True, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):model(seed)

    def test_embedding_bounds_fractional_and_mixed_ids(self):
        m = model()
        for value in (-1., 979., .5, float('nan'), float('inf')):
            with self.subTest(value=value):
                x = inputs(); x[0, :, 13] = value
                with self.assertRaises(ValueError):m(x)
        x = inputs(); x[0, 1, 13] = 1
        with self.assertRaises(ValueError):m(x)
        # Both valid endpoint rows 0 and978 are used in a successful forward.
        self.assertEqual(tuple(m(inputs()).shape), (2, 2))

    def test_invalid_input_shapes_dtypes_and_values_rejected(self):
        m = model()
        for x in (torch.ones(1, 6, 14), torch.ones(1, 36, 13), torch.empty(0, 36, 14),
                  inputs().double(), inputs().bool(), np.ones((1, 36, 14))):
            with self.subTest(kind=type(x)):
                with self.assertRaises(ValueError):m(x)
        x = inputs(); x[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):m(x)

    def test_same_seed_cpu_determinism_and_rng_preservation(self):
        original = torch.random.get_rng_state().clone()
        first = model(43); second = model(43)
        self.assertTrue(torch.equal(original, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))
        self.assertTrue(torch.equal(first(inputs()), second(inputs())))
        self.assertTrue(torch.equal(first(inputs()), first(inputs())))
        different = model(44)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first.parameters(), different.parameters())))

    def test_backward_flows_without_parameter_updates(self):
        m = model(); x = inputs(); f, observed = future()
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        loss = m.loss(x, f, observed)
        loss.backward()  # Explicitly authorized synthetic autograd only; no optimizer.
        for name, p in m.named_parameters():
            self.assertTrue(p.requires_grad, name)
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all().item(), name)
            self.assertTrue((p.grad != 0).any().item(), name)
            self.assertTrue(torch.equal(p, before[name]), name)
        grad = m.core.node_embedding.weight.grad
        self.assertTrue((grad[0] != 0).any().item())
        self.assertTrue((grad[978] != 0).any().item())
        self.assertEqual(torch.count_nonzero(grad[1:978]).item(), 0)
        self.assertIsNone(m.target_mean.grad)
        self.assertIsNone(m.target_std.grad)

    def test_nonfinite_loss_output_and_scaling_state_rejected(self):
        with self.assertRaises(ValueError):
            standardized_mse(torch.tensor([[np.inf, 0.]], dtype=torch.float32), torch.zeros(1, 2))
        m = model()
        with mock.patch.object(m.core, 'forward', return_value=torch.full((2, 2), np.nan)):
            with self.assertRaises(ValueError):m(inputs())
        with torch.no_grad():m.target_mean[0] = np.nan
        with self.assertRaises(ValueError):m(inputs())

    def test_checkpoint_loading_api_is_disabled(self):
        m = model()
        # Empty in-memory dummy only. No checkpoint or state is loaded.
        for strict in (True, False):
            with self.assertRaises(PermissionError):m.load_state_dict({}, strict=strict)
        self.assertFalse(m.metadata()['checkpoint_loading_supported'])


if __name__ == '__main__':unittest.main()
