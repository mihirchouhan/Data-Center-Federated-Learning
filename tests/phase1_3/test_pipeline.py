"""Only fabricated data. No historical imports, optimizers, metrics or FedAvg."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
import torch

from src.phase1_3.config import parse_spec,load_spec,utc_ns
from src.phase1_3.schema import Telemetry,UNITS,FEATURES,WIDTH_NS,frozen_array
from src.phase1_3.validation import sanitize,validate_batch
from src.phase1_3.synthetic_data import generate
from src.phase1_3.windowing import build_windows
from src.phase1_3.scaling import fit_training_scaler,Scaler
from src.phase1_3.split import split_for_origin
from src.phase1_3.model import MultiOutputGRU,synthetic_shape_check
from src.phase1_3.interfaces import (ModelState,ClientUpdate,validate_update_contract,
    eligible_clients,client_views,positive_count)
from src.phase1_3.preservation import compare_records
from src.phase1_3.artifacts import ROOT,write_json,checked_path

SPEC_FILE=ROOT/"configs/phase1_3/experiment_spec_v1.json"
GUARDS=[]
AUDIT_ACTIVE=False

def setUpModule():
    global AUDIT_ACTIVE
    AUDIT_ACTIVE=True
    def audit(event,args):
        if not AUDIT_ACTIVE or event!="open" or not isinstance(args[0],(str,bytes)):
            return
        raw=args[0].decode() if isinstance(args[0],bytes) else args[0]
        p=Path(raw).absolute()
        for forbidden in (ROOT/"data",ROOT/"output/experiments"):
            if p.is_relative_to(forbidden):
                raise AssertionError("Synthetic test attempted historical data access: "+str(p))
    sys.addaudithook(audit)
    for target in ("torch.Tensor.backward","torch.autograd.backward","torch.optim.SGD.step",
                   "torch.optim.Adam.step","torch.Tensor.cuda","torch.nn.Module.cuda"):
        p=mock.patch(target,side_effect=AssertionError("Forbidden training/GPU operation"))
        GUARDS.append((p,p.start()))

def tearDownModule():
    global AUDIT_ACTIVE
    AUDIT_ACTIVE=False
    for p,guard in GUARDS:
        assert guard.call_count==0,"Forbidden operation attempted"
        p.stop()

def setup_data(faults=False,clients=2):
    spec=load_spec(SPEC_FILE,"synthetic")
    records=generate(spec,clients,faults=faults)
    mapping=tuple(r.node for r in records)
    windows,audit=build_windows(records,mapping,spec)
    return spec,records,windows,audit

def edited(record,*,values=None,observed=None,times=None):
    return replace(record,values=record.values.copy() if values is None else values,
        observed=record.observed.copy() if observed is None else observed,
        times=record.times.copy() if times is None else times)

class SchemaTests(unittest.TestCase):
    def test_exact_feature_and_target_schema(self):
        d=json.loads(SPEC_FILE.read_text())
        s=parse_spec(d,"synthetic")
        self.assertEqual(s.history,36);self.assertEqual(s.horizon,6)
        self.assertEqual(len(FEATURES),14)
        self.assertEqual(d["target_units"],["W","degC"])
        self.assertEqual(s.stride,6)

    def test_no_implicit_naive_or_fractional_timestamp(self):
        for v in ("2030-01-01","2030-01-01T00:00:00","2030-01-01T00:00:00.1Z",
                  "2030-01-01T00:00:00+00:00"):
            with self.subTest(v=v),self.assertRaises(ValueError):utc_ns(v)

    def test_malformed_config_is_rejected(self):
        for key,value in (("history_steps",6),("horizon_steps",5),("origin_stride_steps",1),
                          ("feature_units",["W"]*14),("targets",["power"]),("extra","ignored")):
            d=json.loads(SPEC_FILE.read_text());d[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):parse_spec(d,"synthetic")

    def test_bad_dimensions_and_cut_order_are_rejected(self):
        for dimension in (0,-1,True,1.5):
            d=json.loads(SPEC_FILE.read_text());d["model"]["hidden_size"]=dimension
            with self.assertRaises(ValueError):parse_spec(d,"synthetic")
        d=json.loads(SPEC_FILE.read_text())
        d["profiles"]["synthetic"]["val_end_utc"]=d["profiles"]["synthetic"]["train_end_utc"]
        with self.assertRaises(ValueError):parse_spec(d,"synthetic")

    def test_real_profile_does_not_open_real_arrays(self):
        s=load_spec(SPEC_FILE,"experiment")
        with self.assertRaises(PermissionError):generate(s)
        with self.assertRaises(PermissionError):build_windows([],("node",),s)
        with self.assertRaises(PermissionError):synthetic_shape_check(np.zeros((1,36,14)),s,1)

    def test_unit_mismatch_is_rejected(self):
        s,r,_,_=setup_data()
        units=list(UNITS[:-1]);units[0]="W"
        with self.assertRaisesRegex(ValueError,"unit"):
            sanitize(replace(r[0],units=tuple(units)),s)

    def test_duplicate_timestamp_is_rejected(self):
        s,r,_,_=setup_data()
        t=r[0].times.copy();t[10]=t[9]
        with self.assertRaisesRegex(ValueError,"Duplicate"):sanitize(edited(r[0],times=t),s)

    def test_unordered_timestamp_is_rejected(self):
        s,r,_,_=setup_data()
        t=r[0].times.copy();t[[10,11]]=t[[11,10]]
        with self.assertRaisesRegex(ValueError,"unordered"):sanitize(edited(r[0],times=t),s)

    def test_off_grid_and_nat_timestamp_are_rejected(self):
        s,r,_,_=setup_data()
        for val in (np.datetime64("NaT","ns"),r[0].times[10]+np.timedelta64(1,"s")):
            t=r[0].times.copy();t[10]=val
            with self.assertRaises(ValueError):sanitize(edited(r[0],times=t),s)

    def test_wrong_feature_or_observation_shape_is_rejected(self):
        s,r,_,_=setup_data()
        with self.assertRaises(ValueError):sanitize(edited(r[0],values=r[0].values[:,:12]),s)
        with self.assertRaises(ValueError):sanitize(edited(r[0],observed=r[0].observed.astype(int)),s)

    def test_invalid_physical_values_become_missing(self):
        s,r,_,_=setup_data()
        v=r[0].values.copy();v[10,0]=-1;v[11,6]=-300;v[12,1]=101;v[13,5]=2;v[14,6]=np.inf
        clean,audit=sanitize(edited(r[0],values=v),s)
        for row,col in ((10,0),(11,6),(12,1),(13,5),(14,6)):
            self.assertTrue(np.isnan(clean.values[row,col]));self.assertFalse(clean.observed[row,col])
        self.assertEqual(audit["invalid_or_missing_feature_values"],5)

    def test_no_guessed_temperature_upper_cutoff(self):
        s,r,_,_=setup_data();v=r[0].values.copy();v[2,6]=200
        clean,_=sanitize(edited(r[0],values=v),s)
        self.assertEqual(clean.values[2,6],200)

    def test_missing_timestamp_is_reported_without_insertion(self):
        s,r,_,_=setup_data()
        keep=np.ones(len(r[0].times),bool);keep[70:73]=False
        rec=edited(r[0],times=r[0].times[keep],values=r[0].values[keep],observed=r[0].observed[keep])
        clean,audit=sanitize(rec,s)
        self.assertEqual(audit["missing_timestamp_bins"],3)
        self.assertEqual(len(clean.times),len(r[0].times)-3)

    def test_sanitizing_does_not_modify_source(self):
        s,r,_,_=setup_data();before=r[0].values.tobytes()
        clean,_=sanitize(r[0],s)
        self.assertEqual(r[0].values.tobytes(),before)
        with self.assertRaises(ValueError):clean.values.setflags(write=True)

class WindowTests(unittest.TestCase):
    def test_exact_shapes_and_coverage_on_clean_grid(self):
        s,r,w,a=setup_data(clients=1)
        self.assertEqual(w["train"].x.shape,(42,36,14))
        self.assertEqual(w["train"].y.shape,(42,2))
        self.assertEqual(len(w["val"]),18);self.assertEqual(len(w["test"]),18)
        self.assertEqual(w["train"].target_times.shape,(42,6))
        self.assertEqual(a["candidate_windows"],96)

    def test_hand_calculated_physical_means(self):
        s,r,_,_=setup_data(clients=1);v=r[0].values.copy()
        v[:,0]=(1000+10*np.arange(len(v)))/1000
        v[:,6]=20+.5*np.arange(len(v))
        w,_=build_windows([edited(r[0],values=v)],(r[0].node,),s)
        np.testing.assert_allclose(w["train"].y[0],[1385,39.25],rtol=0,atol=1e-12)
        self.assertEqual(w["train"].x[0,-1,0],1.35)
        self.assertEqual(w["train"].x[0,-1,6],37.5)

    def test_history_and_future_are_separated_and_contiguous(self):
        s,_,w,_=setup_data()
        for batch in w.values():
            h,t,o=batch.history_times.view("i8"),batch.target_times.view("i8"),batch.origins.view("i8")
            self.assertTrue((h[:,-1]+s.width==o).all())
            self.assertTrue((t[:,0]==o).all())
            self.assertTrue((np.diff(h,axis=1)==s.width).all())
            self.assertTrue((np.diff(t,axis=1)==s.width).all())

    def test_future_changes_do_not_change_same_origin_input(self):
        s,r,w,_=setup_data(clients=1)
        v=r[0].values.copy();v[36:42,0]+=1;v[36:42,6]+=10
        changed,_=build_windows([edited(r[0],values=v)],(r[0].node,),s)
        np.testing.assert_array_equal(changed["train"].x[0],w["train"].x[0])
        np.testing.assert_allclose(changed["train"].y[0]-w["train"].y[0],[1000,10])

    def test_missing_history_timestamp_is_not_bridged(self):
        s,r,w,_=setup_data(clients=1)
        keep=np.arange(len(r[0].times))!=10
        rec=edited(r[0],times=r[0].times[keep],values=r[0].values[keep],observed=r[0].observed[keep])
        changed,a=build_windows([rec],(rec.node,),s)
        self.assertNotIn(w["train"].origins[0],changed["train"].origins)
        self.assertGreater(a["exclusive_rejections"]["missing_timestamp"],0)

    def test_missing_future_timestamp_rejects_target(self):
        s,r,w,_=setup_data(clients=1)
        keep=np.arange(len(r[0].times))!=40
        rec=edited(r[0],times=r[0].times[keep],values=r[0].values[keep],observed=r[0].observed[keep])
        changed,_=build_windows([rec],(rec.node,),s)
        self.assertNotIn(w["train"].origins[0],changed["train"].origins)

    def test_missing_or_unobserved_target_never_filled(self):
        s,r,w,_=setup_data(clients=1)
        for col,observed in ((0,True),(6,True),(6,False)):
            v=r[0].values.copy();m=r[0].observed.copy()
            if observed:v[38,col]=np.nan
            else:m[38,col]=False
            changed,a=build_windows([edited(r[0],values=v,observed=m)],(r[0].node,),s)
            self.assertNotIn(w["train"].origins[0],changed["train"].origins)
            self.assertGreater(a["exclusive_rejections"]["target_invalid"],0)

    def test_incomplete_or_invalid_history_rejects_window(self):
        s,r,w,_=setup_data(clients=1);v=r[0].values.copy();v[5,0]=-2
        changed,a=build_windows([edited(r[0],values=v)],(r[0].node,),s)
        self.assertNotIn(w["train"].origins[0],changed["train"].origins)
        self.assertGreater(a["exclusive_rejections"]["input_invalid"],0)

    def test_split_edges_are_exact_and_history_cannot_cross(self):
        s,_,w,_=setup_data()
        for split,b in w.items():
            lo,hi=s.bounds(split)
            self.assertTrue((b.history_times.view("i8")[:,0]>=lo).all())
            self.assertTrue((b.target_times.view("i8")[:,-1]+s.width<=hi).all())
            self.assertIsNone(split_for_origin(lo+35*s.width,s))
            self.assertIsNone(split_for_origin(hi-5*s.width,s))
            self.assertEqual(split_for_origin(lo+36*s.width,s),split)
            self.assertEqual(split_for_origin(hi-6*s.width,s),split)

    def test_no_overlapping_future_target_bins_per_node(self):
        _,_,w,_=setup_data()
        for b in w.values():
            for i in range(len(b.mapping)):
                bins=b.target_times[b.nodes==i].view("i8").ravel()
                self.assertEqual(len(bins),len(np.unique(bins)))

    def test_target_intervals_do_not_overlap_across_splits(self):
        _,_,w,_=setup_data()
        sets=[set(b.target_times.view("i8").ravel()) for b in w.values()]
        self.assertFalse(sets[0]&sets[1]);self.assertFalse(sets[1]&sets[2])

    def test_node_series_not_concatenated_or_reindexed(self):
        s,r,w,_=setup_data()
        reversed_w,_=build_windows(list(reversed(r)),tuple(q.node for q in r),s)
        for split,b in w.items():
            np.testing.assert_array_equal(reversed_w[split].x,b.x)
            self.assertTrue(np.array_equal(b.x[:,:,13],np.broadcast_to(b.nodes[:,None],(len(b),36))))
        with self.assertRaises(ValueError):build_windows(r,(r[0].node,),s)
        with self.assertRaises(ValueError):build_windows([r[0],r[0]],(r[0].node,),s)

    def test_masked_weather_missingness_is_ignored_only_in_inputs(self):
        s,r,w,_=setup_data(clients=1);v=r[0].values.copy();v[:,7:9]=np.nan
        changed,_=build_windows([edited(r[0],values=v)],(r[0].node,),s)
        for split,b in changed.items():
            self.assertEqual(len(b),len(w[split]))
            self.assertTrue((b.x[:,:,[7,8]]==0).all())
            np.testing.assert_array_equal(b.y,w[split].y)

    def test_identity_order_and_mutation_are_checked(self):
        s,_,w,_=setup_data()
        b=w["train"]
        with self.assertRaises(ValueError):b.x.setflags(write=True)
        with self.assertRaises(ValueError):validate_batch(replace(b,origins=b.origins[::-1]),s)
        wrong=b.history_times.copy();wrong[0,-1]=b.origins[0]
        with self.assertRaises(ValueError):validate_batch(replace(b,history_times=wrong),s)

    def test_exclusion_accounting_reconciles(self):
        _,_,w,a=setup_data(faults=True)
        self.assertEqual(sum(map(len,w.values()))+sum(a["exclusive_rejections"].values()),a["candidate_windows"])
        self.assertGreater(a["exclusive_rejections"]["missing_timestamp"],0)

    def test_synthetic_generation_and_windows_are_deterministic(self):
        s,r,w,a=setup_data(faults=True);_,r2,w2,a2=setup_data(faults=True)
        self.assertEqual(a,a2)
        for x,y in zip(r,r2):
            self.assertEqual(x.values.tobytes(),y.values.tobytes())
            self.assertEqual(x.times.tobytes(),y.times.tobytes())
        for split in w:
            self.assertEqual(w[split].x.tobytes(),w2[split].x.tobytes())
            self.assertEqual(w[split].y.tobytes(),w2[split].y.tobytes())

class ScalingTests(unittest.TestCase):
    def test_training_only_fit_rejects_validation_and_test(self):
        s,_,w,_=setup_data()
        for split in ("val","test"):
            with self.assertRaises(ValueError):fit_training_scaler(w[split],s)

    def test_relabelled_validation_cannot_be_used_to_fit(self):
        s,_,w,_=setup_data()
        with self.assertRaises(ValueError):fit_training_scaler(replace(w["val"],split="train"),s)

    def test_heldout_values_cannot_change_training_scaler(self):
        s,r,w,_=setup_data()
        original=fit_training_scaler(w["train"],s).to_json()
        altered=[]
        for rec in r:
            v=rec.values.copy();late=rec.times.view("i8")>=s.train_end
            v[late,0]=500;v[late,6]=200
            altered.append(edited(rec,values=v))
        w2,_=build_windows(altered,tuple(v.node for v in r),s)
        self.assertEqual(fit_training_scaler(w2["train"],s).to_json(),original)

    def test_fit_matches_hand_numpy_training_only_statistics(self):
        s,_,w,_=setup_data()
        a=fit_training_scaler(w["train"],s)
        self.assertAlmostEqual(a.x_mean[0],w["train"].x[:,:,0].mean(),places=12)
        self.assertAlmostEqual(a.y_mean[0],w["train"].y[:,0].mean(),places=9)
        self.assertAlmostEqual(a.y_mean[1],w["train"].y[:,1].mean(),places=12)

    def test_inverse_scaling_and_units_are_preserved(self):
        s,_,w,_=setup_data()
        a=fit_training_scaler(w["train"],s)
        for b in w.values():
            _,y=a.transform(b,s)
            np.testing.assert_allclose(a.inverse_targets(y),b.y,rtol=1e-6,atol=1e-5)
        self.assertGreater(a.y_mean[0],100)
        self.assertTrue(10<a.y_mean[1]<50)

    def test_constant_channels_id_and_weather_are_safe(self):
        s,_,w,_=setup_data()
        a=fit_training_scaler(w["train"],s);x,_=a.transform(w["train"],s)
        self.assertEqual(a.x_std[2],1)
        for k in (7,8,13):self.assertEqual((a.x_mean[k],a.x_std[k]),(0,1))
        self.assertTrue((x[:,:,[7,8]]==0).all())
        np.testing.assert_array_equal(x[:,:,13],w["train"].x[:,:,13])

    def test_constant_targets_have_positive_unit_scale(self):
        s,r,_,_=setup_data(clients=1);v=r[0].values.copy();v[:,0]=1.2;v[:,6]=25
        w,_=build_windows([edited(r[0],values=v)],(r[0].node,),s)
        a=fit_training_scaler(w["train"],s)
        np.testing.assert_array_equal(a.y_std,[1,1])
        np.testing.assert_allclose(a.y_mean,[1200,25])

    def test_scaler_serialization_and_state_are_immutable(self):
        s,_,w,_=setup_data();a=fit_training_scaler(w["train"],s)
        restored=Scaler.from_json(a.to_json(),s)
        self.assertEqual(restored.to_json(),a.to_json())
        with self.assertRaises(ValueError):a.x_mean.setflags(write=True)

    def test_invalid_scaler_state_is_rejected(self):
        s,_,w,_=setup_data();a=fit_training_scaler(w["train"],s)
        for key,val in (("y_std",[0,1]),("y_mean",[float("nan"),1]),
                        ("target_units",["kW","degC"]),("spec_digest","0"*64)):
            d=json.loads(a.to_json());d[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):Scaler.from_json(json.dumps(d),s)

class ContractTests(unittest.TestCase):
    def test_eligibility_and_client_views_are_training_only(self):
        s,_,w,_=setup_data();a=fit_training_scaler(w["train"],s)
        eligible=eligible_clients(w["train"],s);views=client_views(w["train"],s,a)
        self.assertEqual(sum(eligible.values()),len(w["train"]))
        for name,v in views.items():
            self.assertEqual(v.sample_count,eligible[name])
            self.assertTrue((v.origins.view("i8")<s.train_end).all())
            self.assertFalse(hasattr(v,"validation"));self.assertFalse(hasattr(v,"test"))
        for split in ("val","test"):
            with self.assertRaises(PermissionError):eligible_clients(w[split],s)

    def test_eligibility_thresholds_depend_on_training_only(self):
        s,_,w,_=setup_data()
        higher=replace(s,minimum_windows=1000)
        self.assertEqual(eligible_clients(w["train"],higher),{})
        higher=replace(s,minimum_days=7)
        self.assertEqual(eligible_clients(w["train"],higher),{})

    def test_count_validation(self):
        for n in (0,-1,True,1.5,np.int64(4)):
            with self.subTest(n=n),self.assertRaises(ValueError):positive_count(n)

    def test_sample_weights_use_frozen_training_counts_without_aggregation(self):
        g=ModelState(0,{"w":np.zeros((2,3),np.float32)})
        u=[ClientUpdate("a",1,g.fingerprint,g),ClientUpdate("b",3,g.fingerprint,g)]
        before=g.fingerprint
        self.assertEqual(validate_update_contract(g,u,{"a":1,"b":3}),{"a":.25,"b":.75})
        self.assertEqual(g.fingerprint,before)

    def test_parameter_shape_dtype_names_and_round_are_validated(self):
        g=ModelState(0,{"w":np.zeros((2,3),np.float32)})
        states=[ModelState(0,{"w":np.zeros((3,2),np.float32)}),
                ModelState(0,{"w":np.zeros((2,3),np.float64)}),
                ModelState(0,{"other":np.zeros((2,3),np.float32)}),
                ModelState(1,{"w":np.zeros((2,3),np.float32)})]
        for state in states:
            with self.assertRaises(ValueError):
                validate_update_contract(g,[ClientUpdate("a",1,g.fingerprint,state)],{"a":1})

    def test_invalid_model_state_values_and_fixed_buffers(self):
        for a in (np.array([np.nan],np.float32),np.array([np.inf],np.float32),np.zeros(1,np.int64)):
            with self.assertRaises(ValueError):ModelState(0,{"w":a})
        g=ModelState(0,{"fixed":np.ones(1,np.float32)},("fixed",))
        altered=ModelState(0,{"fixed":np.zeros(1,np.float32)},("fixed",))
        with self.assertRaises(ValueError):
            validate_update_contract(g,[ClientUpdate("a",1,g.fingerprint,altered)],{"a":1})

    def test_missing_duplicate_stale_or_wrong_count_updates(self):
        g=ModelState(0,{"w":np.zeros(2,np.float32)})
        valid=ClientUpdate("a",1,g.fingerprint,g)
        cases=[([],{"a":1}),([valid,valid],{"a":1,"b":1}),
               ([replace(valid,base_fingerprint="wrong")],{"a":1}),
               ([replace(valid,training_samples=2)],{"a":1})]
        for updates,counts in cases:
            with self.assertRaises(ValueError):validate_update_contract(g,updates,counts)

    def test_model_state_is_copied_and_immutable(self):
        raw=np.zeros(2,np.float32);g=ModelState(0,{"w":raw});before=g.fingerprint;raw[0]=1
        self.assertEqual(before,g.fingerprint)
        with self.assertRaises(ValueError):g.tensors["w"].setflags(write=True)
        with self.assertRaises(TypeError):g.tensors["extra"]=raw

class ArchitectureTests(unittest.TestCase):
    def test_two_output_synthetic_shape_cpu_only(self):
        s,_,w,_=setup_data();a=fit_training_scaler(w["train"],s);x,_=a.transform(w["train"],s)
        report=synthetic_shape_check(x[:3],s,len(w["train"].mapping))
        self.assertEqual(report["output_shape"],[3,2])
        self.assertEqual(report["device"],"cpu")
        self.assertTrue(report["finite"]);self.assertFalse(report["trainable_parameters_enabled"])

    def test_model_dimensions_are_configurable(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(11)
            m=MultiOutputGRU(node_count=2,hidden_size=8,layers=1,embedding_size=3).eval()
            with torch.inference_mode():y=m(torch.zeros((2,36,14),dtype=torch.float32))
        self.assertEqual(m.recurrent.input_size,16)
        self.assertEqual(tuple(y.shape),(2,2))
        self.assertFalse(any(p.requires_grad for p in m.parameters()))

    def test_model_input_validation(self):
        with torch.random.fork_rng(devices=[]):
            m=MultiOutputGRU(node_count=2,hidden_size=4,layers=1).eval()
        samples=[torch.zeros((1,35,14)),torch.zeros((1,36,13)),torch.zeros((1,36,14),dtype=torch.float64)]
        for x in samples:
            with self.assertRaises(ValueError),torch.inference_mode():m(x)
        for bad in (.5,2,-1,float("nan")):
            x=torch.zeros((1,36,14));x[0,0,13]=bad
            with self.assertRaises(ValueError),torch.inference_mode():m(x)

    def test_seeded_outputs_are_deterministic_without_training(self):
        outputs=[]
        for _ in range(2):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(123)
                m=MultiOutputGRU(node_count=1,hidden_size=4,layers=1).eval()
                with torch.inference_mode():outputs.append(m(torch.zeros((2,36,14))))
        self.assertTrue(torch.equal(*outputs))

    def test_shape_check_preserves_cpu_rng(self):
        s,_,w,_=setup_data();a=fit_training_scaler(w["train"],s);x,_=a.transform(w["train"],s)
        before=torch.random.get_rng_state().clone()
        synthetic_shape_check(x[:1],s,2)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

class PreservationTests(unittest.TestCase):
    def test_missing_changed_metadata_content_or_unapproved_addition_fails(self):
        before={"old":{"sha256":"abc","bytes":3,"mtime_ns":1791795740722105880,"mode":436}}
        self.assertTrue(compare_records(before,copy.deepcopy(before),[])["unchanged"])
        for field in before["old"]:
            after=copy.deepcopy(before);after["old"][field]="different"
            result=compare_records(before,after,[])
            self.assertFalse(result["unchanged"]);self.assertEqual(result["mismatches"][0]["field"],field)
        self.assertFalse(compare_records(before,{},[])["unchanged"])
        extra={**before,"new":before["old"]}
        self.assertFalse(compare_records(before,extra,[])["unchanged"])
        self.assertTrue(compare_records(before,extra,["new"])["unchanged"])

    def test_paths_outside_new_namespace_are_rejected(self):
        for p in (ROOT/"data/file",ROOT/"output/experiments/new.json",ROOT/"output/phase1_3/../bad"):
            with self.assertRaises(ValueError):checked_path(p)

    def test_exclusive_writes_never_overwrite(self):
        parent=ROOT/"output/phase1_3/test_artifacts_v1"
        parent.mkdir(exist_ok=False)
        p=parent/"receipt.json"
        write_json(p,{"synthetic_only":True})
        before=p.read_bytes()
        with self.assertRaises(FileExistsError):write_json(p,{"bad":"overwrite"})
        self.assertEqual(before,p.read_bytes())

if __name__=="__main__":
    unittest.main()
