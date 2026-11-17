"""Synthetic independent rational rounding oracles and B1 acceptance regressions."""
from fractions import Fraction as F
import ast
import dataclasses
from pathlib import Path
import sys
import unittest
import numpy as np
from src import b1_scaler_acceptance_v1 as a
from src import b1_bridge_scaling_v1 as s
from src import b1_real_data_bridge_v1 as b
from test_b1_real_data_bridge_v1 import fixture


def pow2(e):
    return F(2**e) if e >= 0 else F(1, 2**(-e))


def rn(value, bits):
    """Integer quotient/remainder oracle; no hardware FP rounding to select bits."""
    value = F(value)
    if not value:
        return F(0)
    sign = -1 if value < 0 else 1
    v = abs(value); p, emin, emax = (24, -126, 127) if bits == 32 else (53, -1022, 1023)
    e = v.numerator.bit_length()-v.denominator.bit_length()
    if v < pow2(e): e -= 1
    step = pow2(max(e-(p-1), emin-(p-1)))
    scaled = v/step
    q, r = divmod(scaled.numerator, scaled.denominator)
    if 2*r > scaled.denominator or (2*r == scaled.denominator and q % 2): q += 1
    rounded = q*step
    if rounded > (2-pow2(1-p))*pow2(emax):
        raise OverflowError('Rational oracle finite format overflow')
    return sign*rounded


def exact_cell(value, bits):
    dt = np.float32 if bits == 32 else np.float64
    v = dt(value); fv = F(float(v))
    with np.errstate(over='ignore', under='ignore'):
        lo = np.nextafter(v, dt(-np.inf)); hi = np.nextafter(v, dt(np.inf))
    gaps = []
    if np.isfinite(lo): gaps.append(fv-F(float(lo)))
    if np.isfinite(hi): gaps.append(F(float(hi))-fv)
    return max(gaps)/2


def rational_witness(x, mu, scale):
    fx, fm, fs = F(float(x)), F(float(mu)), F(float(scale))
    t = rn(fx-fm, 64); z = rn(t/fs, 64); q = rn(z, 32)
    p = rn(q*fs, 64); hat = rn(p+fm, 64); p0 = rn(z*fs, 64); hat0 = rn(p0+fm, 64)
    raw = a.reference(np.array(x), np.array(mu), np.array(scale))
    for k, v in [('t', t), ('z', z), ('q', q), ('p', p), ('hat', hat), ('p0', p0), ('hat0', hat0)]:
        if F(float(raw[k])) != v: raise AssertionError('Independent rounding differs: '+k)
    terms = [t-(fx-fm), fs*(z-t/fs), fs*(q-z), p-q*fs, hat-(p+fm)]
    if sum(terms) != hat-fx: raise AssertionError('Exact telescoping identity differs')
    bound = exact_cell(float(t), 64)+fs*exact_cell(float(z), 64)+fs*exact_cell(float(q), 32)+exact_cell(float(p), 64)+exact_cell(float(hat), 64)
    aff = exact_cell(float(t), 64)+fs*exact_cell(float(z), 64)+exact_cell(float(p0), 64)+exact_cell(float(hat0), 64)
    if abs(hat-fx) > bound or abs(hat0-fx) > aff or abs(q-z) > exact_cell(float(q), 32):
        raise AssertionError('Exact bound fails')
    if F(float(raw['bound'])) < bound or F(float(raw['affine_bound'])) < aff:
        raise AssertionError('Floating bound is not an enclosure')
    return {'rounding_oracle_pass': True, 'exact_inverse_error': {'numerator': (hat-fx).numerator,
        'denominator': (hat-fx).denominator}, 'computed_bound': float(raw['bound']),
        'exact_cell_sum': {'numerator': bound.numerator, 'denominator': bound.denominator}}


def scaler():
    d = fixture(); d['x'][:,:,1] = 25.
    return s.fit_training_scaler(b.accept_windows(**d), implementation_sha256='a'*64)


class AcceptanceTests(unittest.TestCase):
    def test_environment(self): self.assertTrue(a.environment_check()['gradual_underflow'])
    def test_float32_zero_and_subnormal_cell(self):
        for x in (0., -0., 2.**-149, -2.**-149):
            self.assertEqual(F(float(a.half_cell(x, 'f4'))), F(1, 2**150))
    def test_float64_subnormal_cell_rounded_up(self):
        for x in (0., -0., 2.**-1074): self.assertGreaterEqual(F(float(a.half_cell(x, 'f8'))), F(1, 2**1075))
    def test_asymmetric_binade_cells(self):
        for x in (1., -1., 2., .5): self.assertEqual(F(float(a.half_cell(x, 'f4'))), exact_cell(x, 32))
    def test_finite_endpoints(self):
        for dt in ('f4', 'f8'):
            for sign in (-1, 1):
                x = sign*np.finfo(dt).max
                self.assertGreaterEqual(F(float(a.half_cell(x, dt))), exact_cell(x, 32 if dt == 'f4' else 64))
    def test_nonfinite_centers_rejected(self):
        for x in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError): a.half_cell(x, 'f4')
    def test_unsupported_format_rejected(self):
        with self.assertRaises(ValueError): a.half_cell(1., 'f2')
    def test_upward_helpers_exact_enclosures(self):
        for x,y in [(1., 2.**-54), (2.**-1074, .5), (1.23, 4.56), (0., 0.)]:
            self.assertGreaterEqual(F(float(a.up_add(x,y))), F(x)+F(y))
            self.assertGreaterEqual(F(float(a.up_mul(x,y))), F(x)*F(y))
    def test_invalid_bounds_rejected(self):
        for x in (-1., np.inf, np.nan):
            with self.assertRaises(ValueError): a.up_add(x, 1.)
    def test_bound_overflow_rejected(self):
        with self.assertRaises((ValueError,FloatingPointError)): a.up_add(np.finfo('f8').max, np.finfo('f8').max)
    def test_model_input_overflow_rejected(self):
        with self.assertRaises((ValueError,FloatingPointError)): a.reference([1e300], [0.], [1.])
    def test_invalid_scale_rejected(self):
        for v in (0., -1., np.inf):
            with self.assertRaises(ValueError): a.reference([1.], [0.], [v])
    def test_rational_ties_even(self):
        self.assertEqual(rn(1+F(1, 2**24), 32), F(1))
        self.assertEqual(rn(1+F(3, 2**24), 32), 1+F(1, 2**22))
        self.assertEqual(rn(F(1, 2**150), 32), 0)
        self.assertEqual(rn(F(3, 2**150), 32), F(1, 2**148))
    def test_rational_endpoints_and_overflow(self):
        for bits,dt in ((32,'f4'),(64,'f8')):
            self.assertEqual(rn(F(float(np.finfo(dt).max)), bits), F(float(np.finfo(dt).max)))
        with self.assertRaises(OverflowError): rn(F(2)**128, 32)
    def test_exact_witness(self):
        r = rational_witness(0.,89.9267961435083,18.246770753926704)
        self.assertTrue(r['rounding_oracle_pass'])
    def test_known_witness_old_failure_preserved(self):
        d = a.reference(np.array([0.]), [89.9267961435083], [18.246770753926704])
        out = a.verify_affine([0.], [89.9267961435083], [18.246770753926704], d['q'], d['hat'])
        self.assertTrue(out['historical_failures'][0]); self.assertEqual(s.FLOAT32_ATOL, 2e-6)
    def test_synthetic_rational_matrix(self):
        for x,mu,scale in [(0.,0.,1.), (-0.,0.,1.), (1.,2.,.1), (-1.,1.,1e3),
                           (1e-300,1e-301,1e-280), (2**30, .125, 3.), (.125,.125,1.)]:
            rational_witness(x,mu,scale)
    def test_exact_quantization_tie_accepted(self):
        x = np.array([1+2.**-24]); d = a.reference(x, [0.], [1.])
        a.verify_affine(x, [0.], [1.], d['q'], d['hat'])
    def test_wrong_forward_ulp_rejected(self):
        d = a.reference([1.23], [0.], [1.])
        with self.assertRaisesRegex(ValueError,'Forward operation bits'): a.verify_affine([1.23],[0.],[1.],np.nextafter(d['q'],np.float32(np.inf)),d['hat'])
    def test_wrong_inverse_within_bound_rejected(self):
        d = a.reference([0.], [89.9267961435083], [18.246770753926704])
        wrong = np.nextafter(d['hat'], np.inf)
        self.assertLess(float(wrong[0]), float(d['bound'][0]))
        with self.assertRaisesRegex(ValueError,'Inverse operation bits'): a.verify_affine([0.],[89.9267961435083],[18.246770753926704],d['q'],wrong)
    def test_wrong_inverse_outside_bound_rejected(self):
        d = a.reference([1.], [0.], [1.])
        with self.assertRaises(ValueError): a.verify_affine([1.],[0.],[1.],d['q'],d['hat']+.001)
    def test_wrong_dtype_rejected(self):
        d = a.reference([1.], [0.], [1.])
        with self.assertRaises(ValueError): a.verify_affine([1.],[0.],[1.],d['q'].astype('f8'),d['hat'])
    def test_masked_boolean_nonfinite_rejected(self):
        for x in (np.ma.array([1.], mask=False), np.array([True]), np.array([np.nan])):
            with self.assertRaises(ValueError): a.reference(x,[0.],[1.])
    def test_actual_synthetic_inputs_and_targets(self):
        w=b.accept_windows(**fixture()); sc=scaler()
        a.verify_inputs(sc,w.x);a.verify_targets(sc,w.physical_targets)
    def test_candidate_and_constant_channels(self):
        sc=scaler(); self.assertEqual(a.verify_candidate(sc,sc.record()),sc.fingerprint)
        self.assertIn(1,sc.record()['constant_input_channels'])
    def test_changed_candidate_rejected(self):
        sc=scaler(); r=sc.record();r['x_mean'][0]+=.1
        with self.assertRaises(ValueError): a.verify_candidate(sc,r)
    def test_wrong_constant_scale_rejected(self):
        sc=scaler();new=list(sc.x_std);new[1]=2.;other=dataclasses.replace(sc,x_std=tuple(new))
        with self.assertRaisesRegex(ValueError,'Constant-channel'):a.verify_candidate(other,other.record())
    def test_bad_node_ordinal_rejected(self):
        sc=scaler(); x=b.accept_windows(**fixture()).x.copy();x[:,:,13]=979
        with self.assertRaises(ValueError):a.verify_inputs(sc,x)
    def test_frozen_candidate_and_validation_not_fitted(self):
        sc=scaler();before=sc.fingerprint;val=b.accept_windows(**fixture('val'))
        a.verify_inputs(sc,val.x);self.assertEqual(sc.fingerprint,before)
        with self.assertRaises(PermissionError):s.fit_training_scaler(val,implementation_sha256='a'*64)
    def test_no_framework_import(self): self.assertNotIn('torch',sys.modules)
    def test_verifier_has_no_file_fit_or_model_operations(self):
        tree=ast.parse(Path(a.__file__).read_text())
        names={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        self.assertTrue(names.isdisjoint({'open','torch','fit_training_scaler'}))

if __name__ == '__main__': unittest.main()
