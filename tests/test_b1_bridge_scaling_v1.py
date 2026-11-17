"""Synthetic training-window statistics and data capabilities; no model calls."""
import dataclasses
import hashlib
from pathlib import Path
import unittest
import numpy as np
from src import b1_real_data_bridge_v1 as b
from src import b1_bridge_scaling_v1 as s
from test_b1_real_data_bridge_v1 import fixture

IMPLEMENTATION_SHA256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()

def fit(w=None,chunk=13):
    return s.fit_training_scaler(w if w is not None else b.accept_windows(**fixture()),
        implementation_sha256=IMPLEMENTATION_SHA256,chunk_size=chunk)


class BridgeScalerTests(unittest.TestCase):
    def test_two_pass_population_statistics_match_independent_numpy(self):
        w=b.accept_windows(**fixture());sc=fit(w)
        np.testing.assert_allclose(sc.x_mean[:13],w.x[:,:,:13].mean(axis=(0,1)),rtol=s.FLOAT64_RTOL,atol=s.FLOAT64_ATOL)
        np.testing.assert_allclose(sc.original_x_std[:13],w.x[:,:,:13].std(axis=(0,1),ddof=0),rtol=s.FLOAT64_RTOL,atol=s.FLOAT64_ATOL)
        np.testing.assert_allclose(sc.y_mean,w.physical_targets.mean(axis=0),rtol=s.FLOAT64_RTOL,atol=s.FLOAT64_ATOL)
        np.testing.assert_allclose(sc.y_std,w.physical_targets.std(axis=0,ddof=0),rtol=s.FLOAT64_RTOL,atol=s.FLOAT64_ATOL)
    def test_hand_calculated_mean_variance_and_target_units(self):
        d=fixture(ordinals=(2,));n=len(d['x']);d['x'][:,:,0]=np.arange(n)[:,None];d['future'][:,:,0]=np.arange(n)[:,None]+1;d['future'][:,:,1]=2*np.arange(n)[:,None]
        sc=fit(b.accept_windows(**d));variance=(n*n-1)/12
        self.assertAlmostEqual(sc.x_mean[0],(n-1)/2);self.assertAlmostEqual(sc.x_std[0],np.sqrt(variance))
        self.assertAlmostEqual(sc.y_mean[0],1000*((n-1)/2+1));self.assertAlmostEqual(sc.y_std[0],1000*np.sqrt(variance))
        self.assertAlmostEqual(sc.y_mean[1],n-1);self.assertAlmostEqual(sc.y_std[1],2*np.sqrt(variance))
    def test_fixed_order_exact_chunk_invariance_and_fingerprints(self):
        w=b.accept_windows(**fixture());ref=fit(w,1)
        for chunk in (2,7,13,28,55,56,1000):
            with self.subTest(chunk=chunk):
                sc=fit(w,chunk)
                self.assertEqual(sc.record(),ref.record());self.assertEqual(sc.fingerprint,ref.fingerprint)
        self.assertEqual(fit(w,7).fingerprint,fit(w,7).fingerprint)
    def test_input_roundtrip_and_no_raw_float32_before_normalization(self):
        w=b.accept_windows(**fixture());sc=fit(w);x=sc.transform_inputs(w.x)
        expected=((w.x-np.asarray(sc.x_mean))/np.asarray(sc.x_std)).astype(np.float32)
        np.testing.assert_array_equal(x,expected);self.assertEqual(x.dtype,np.float32)
        np.testing.assert_allclose(sc.inverse_inputs(x),w.x,rtol=s.FLOAT32_RTOL,atol=s.FLOAT32_ATOL)
        np.testing.assert_array_equal(x[:,:,13],w.x[:,:,13])
    def test_independent_target_inverse_without_another_thousand(self):
        w=b.accept_windows(**fixture());sc=fit(w);z=sc.standardize_targets(w.physical_targets)
        self.assertEqual(z.dtype,np.float32)
        np.testing.assert_allclose(sc.inverse_targets(z),w.physical_targets,rtol=s.FLOAT32_RTOL,atol=s.FLOAT32_ATOL)
        np.testing.assert_allclose(sc.target_scaling.inverse(sc.target_scaling.standardize(w.physical_targets)),w.physical_targets,rtol=s.FLOAT64_RTOL,atol=s.FLOAT64_ATOL)
        self.assertLess(sc.inverse_targets(z)[0,0],1000)
    def test_validation_cannot_enter_fitting_or_admission(self):
        val=b.accept_windows(**fixture('val'))
        with self.assertRaises(PermissionError):fit(val)
        with self.assertRaises(PermissionError):s.fit_training_scaler(object(),implementation_sha256=IMPLEMENTATION_SHA256)
    def test_validation_applies_same_frozen_training_statistics(self):
        train=b.accept_windows(**fixture());d=fixture('val');d['x'][:,:,0]+=10;d['future'][:,:,0]+=10
        val=b.accept_windows(**d);sc=fit(train);before=sc.fingerprint
        tr,va=b.make_datasets(train,val,sc);batch=va.client('3').batch(np.arange(5,dtype=np.int64))
        np.testing.assert_array_equal(batch.x,sc.transform_inputs(val.x[:5]))
        np.testing.assert_array_equal(batch.y,val.physical_targets[:5])
        self.assertEqual(sc.fingerprint,before);self.assertGreater(abs(float(batch.x[:,:,0].mean())),100)
        self.assertEqual(tr.client('3').split,'train');self.assertEqual(va.client('3').split,'val')
    def test_statistics_provenance_contains_exact_training_population(self):
        w=b.accept_windows(**fixture());sc=fit(w);r=sc.record();p=r['provenance']
        self.assertEqual(p['fit_split'],'train');self.assertEqual(p['fit_windows'],56)
        self.assertEqual(p['fit_input_step_occurrences'],56*36);self.assertEqual(p['fit_target_count'],56)
        self.assertEqual(p['fit_identity_sha256'],w.identity_sha256);self.assertEqual(p['fit_target_sha256'],w.target_sha256)
        self.assertEqual(p['fit_content_sha256'],w.content_sha256);self.assertEqual(p['mapping_digest'],b.MAPPING_DIGEST)
        self.assertEqual(p['implementation_sha256'],IMPLEMENTATION_SHA256);self.assertEqual(r['target_units'],['W','degC'])
        self.assertEqual(p['algorithm'],s.ALGORITHM);self.assertEqual(p['ddof'],0)
    def test_constant_input_uses_one_but_constant_target_stops(self):
        d=fixture();d['x'][:,:,1]=25;sc=fit(b.accept_windows(**d))
        self.assertEqual(sc.original_x_std[1],0);self.assertEqual(sc.x_std[1],1)
        self.assertEqual(sc.x_mean[13],0);self.assertEqual(sc.x_std[13],1)
        self.assertIn(1,sc.record()['constant_input_channels'])
        for column in (0,1):
            d=fixture();d['future'][:,:,column]=1
            with self.assertRaisesRegex(ValueError,'constant target'):fit(b.accept_windows(**d))
    def test_ineligible_training_is_not_silently_filtered(self):
        w=b.accept_windows(**fixture(days=1))
        with self.assertRaisesRegex(ValueError,'eligible training'):fit(w)
    def test_wrong_scaler_training_content_is_rejected(self):
        w=b.accept_windows(**fixture());sc=fit(w);d=fixture();d['x'][0,0,1]+=.1
        other=b.accept_windows(**d)
        self.assertEqual(w.identity_sha256,other.identity_sha256)
        with self.assertRaisesRegex(ValueError,'content differs'):sc.require_training(other)
        with self.assertRaises(ValueError):b.make_datasets(other,b.accept_windows(**fixture('val')),sc)
    def test_no_validation_only_client_admission(self):
        w=b.accept_windows(**fixture(ordinals=(2,)));sc=fit(w)
        val=b.accept_windows(**fixture('val',ordinals=(978,)))
        with self.assertRaisesRegex(ValueError,'cannot admit'):b.make_datasets(w,val,sc)
    def test_client_local_capabilities_preserve_ids_and_original_ordinals(self):
        w=b.accept_windows(**fixture());val=b.accept_windows(**fixture('val'));tr,va=b.make_datasets(w,val,fit(w))
        c=tr.client('979');self.assertEqual(c.node_index,978);self.assertEqual(len(c),28)
        batch=c.batch(np.array([5,1,27]))
        np.testing.assert_array_equal(batch.nodes,['979']*3);np.testing.assert_array_equal(batch.positions,[33,29,55])
        np.testing.assert_array_equal(batch.origins,w.origins[[33,29,55]])
        np.testing.assert_array_equal(batch.x[:,:,13],np.full((3,36),978))
        self.assertFalse(hasattr(c,'_windows'));self.assertFalse(hasattr(c,'context'))
        self.assertFalse(hasattr(c,'validation'));self.assertFalse(hasattr(c,'test'))
        self.assertEqual(va.client('979').node_index,978)
    def test_invalid_local_positions_and_unknown_clients_rejected(self):
        w=b.accept_windows(**fixture());tr,va=b.make_datasets(w,b.accept_windows(**fixture('val')),fit(w))
        for name in ('1','unknown'):
            with self.assertRaises(KeyError):tr.client(name)
        c=tr.client('3')
        for p in ([],[-1],[28],[0.5],[True],[[0]], [0,0]):
            with self.subTest(positions=p):
                with self.assertRaises(ValueError):c.batch(p)
    def test_batch_arrays_and_frozen_scaler_are_immutable(self):
        w=b.accept_windows(**fixture());sc=fit(w);tr,_=b.make_datasets(w,b.accept_windows(**fixture('val')),sc)
        batch=tr.client('3').batch(np.array([0,1]))
        for name in batch.__dataclass_fields__:
            with self.assertRaises(ValueError):getattr(batch,name).setflags(write=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):sc.y_mean=(0,0)
        r=sc.record();r['y_mean'][0]=999;self.assertNotEqual(sc.y_mean[0],999)
    def test_raw_future_is_retained_for_existing_adapter_loss_contract(self):
        w=b.accept_windows(**fixture());tr,_=b.make_datasets(w,b.accept_windows(**fixture('val')),fit(w))
        batch=tr.client('3').batch(np.array([0,2]))
        self.assertEqual(batch.future.shape,(2,6,2));self.assertEqual(batch.y.shape,(2,2))
        np.testing.assert_array_equal(b.target_validation.target_means(batch.future,batch.target_observed),batch.y)
    def test_tensor_conversion_only_cpu_and_independent_copies(self):
        w=b.accept_windows(**fixture());tr,_=b.make_datasets(w,b.accept_windows(**fixture('val')),fit(w))
        batch=tr.client('3').batch(np.array([0,1]));t=batch.cpu_tensors()
        self.assertEqual(tuple(t['x'].shape),(2,36,14));self.assertEqual(tuple(t['y'].shape),(2,2))
        self.assertTrue(all(v.device.type=='cpu' and not v.requires_grad for v in t.values()))
        old=batch.x.copy();t['x'].zero_();np.testing.assert_array_equal(batch.x,old)
    def test_invalid_fit_arguments_and_invalid_transform_payloads(self):
        w=b.accept_windows(**fixture())
        for chunk in (0,-1,True,1.5):
            with self.assertRaises(ValueError):fit(w,chunk)
        with self.assertRaises(ValueError):s.fit_training_scaler(w,implementation_sha256='bad')
        sc=fit(w)
        for x in (np.ones((2,6,14)),np.ones((2,36,13)),np.ones((2,36,14),bool),np.ma.array(w.x,mask=False)):
            with self.assertRaises(ValueError):sc.transform_inputs(x)
        for y in (np.ones((2,1)),np.ones((2,2),bool),np.array([[np.nan,1.]])):
            with self.assertRaises(ValueError):sc.standardize_targets(y)
    def test_scaler_constructor_rejects_invalid_fixed_roles(self):
        sc=fit()
        for k,value in [('x_std',(0,)*14),('y_std',(0,1)),('x_mean',sc.x_mean[:13]+(1.,)),('provenance_json',b'{}')]:
            with self.subTest(field=k):
                with self.assertRaises(ValueError):dataclasses.replace(sc,**{k:value})
    def test_no_test_dataset_factory_or_march_loading_api(self):
        w=b.accept_windows(**fixture());sc=fit(w);val=b.accept_windows(**fixture('val'))
        with self.assertRaises(PermissionError):b.make_datasets(val,w,sc)
        self.assertFalse(hasattr(b,'load_test'));self.assertNotIn('test',b.SPLITS)

if __name__=='__main__':unittest.main()
