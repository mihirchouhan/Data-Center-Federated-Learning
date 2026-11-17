"""Invented window tensors only; no telemetry, models, scalers on project data."""
import copy
import unittest
from unittest import mock
import numpy as np
from src import b1_real_data_bridge_v1 as b


def fixture(split='train', ordinals=(2,978), days=7):
    """Actual ordinal labels, wholly invented values/times; no files opened."""
    base=np.datetime64('2022-01-10' if split=='train' else '2022-01-28','ns')
    origins=np.array([base+np.timedelta64(day*24+hour,'h') for node in ordinals
                      for day in range(days) for hour in (8,12,16,20)])
    ids=np.repeat(np.array(ordinals,dtype=np.int64),days*4);n=len(ids)
    h=origins[:,None]+np.arange(-36,0)*np.timedelta64(10,'m')
    t=origins[:,None]+np.arange(6)*np.timedelta64(10,'m')
    row=np.arange(n,dtype=np.float64);step=np.arange(36,dtype=np.float64)
    x=np.zeros((n,36,14),dtype=np.float64)
    x[:,:,0]=.4+row[:,None]*.0007+step[None,:]*.00001
    for j in range(1,5):x[:,:,j]=10*j+row[:,None]*.02+step[None,:]*.001
    x[:,:,5]=.5+row[:,None]*.001
    x[:,:,6]=24+row[:,None]*.02+step[None,:]*.002
    x[:,:,7]=12+row[:,None]*.03;x[:,:,8]=60+row[:,None]*.02
    seconds=(h.view('i8')+b.WIDTH_NS)/1e9
    d=2*np.pi*(seconds%86400)/86400;w=2*np.pi*((seconds+259200)%604800)/604800
    x[:,:,9]=np.sin(d);x[:,:,10]=np.cos(d);x[:,:,11]=np.sin(w);x[:,:,12]=np.cos(w);x[:,:,13]=ids[:,None]
    future=np.empty((n,6,2),dtype=np.float64)
    future[:,:,0]=.3+row[:,None]*.0002+np.arange(6)*.001
    future[:,:,1]=20+row[:,None]*.03+np.arange(6)*.1
    first=np.broadcast_to(h[:,:,None],(n,36,9)).copy();last=first+np.timedelta64(300,'s')
    first[:,:,7:]+=np.timedelta64(120,'s');last[:,:,7:]=first[:,:,7:]
    tf=np.broadcast_to(t[:,:,None],(n,6,2)).copy();tl=tf+np.timedelta64(300,'s')
    return dict(split=split,mapping=b.ORIGINAL_NAMES,x=x,observed=np.ones((n,36,13),bool),
        future=future,target_observed=np.ones((n,6,2),bool),nodes=np.asarray(b.ORIGINAL_NAMES)[ids],origins=origins,
        history_times=h,target_times=t,history_nodes=np.broadcast_to(ids[:,None],(n,36)).copy(),
        target_nodes=np.broadcast_to(ids[:,None],(n,6)).copy(),source_earliest=first,source_latest=last,
        target_source_earliest=tf,target_source_latest=tl,history_coverage=np.ones((n,36,7)),target_coverage=np.ones((n,6,2)),
        source_evidence={'source_kind':'synthetic','raw_contract_sha256':'1'*64,'bin_contract_sha256':'2'*64,'producer_source_sha256':'3'*64})

def refresh_calendar(d):
    seconds=(d['history_times'].view('i8')+b.WIDTH_NS)/1e9
    daily=2*np.pi*(seconds%86400)/86400;weekly=2*np.pi*((seconds+259200)%604800)/604800
    d['x'][:,:,9:13]=np.stack((np.sin(daily),np.cos(daily),np.sin(weekly),np.cos(weekly)),axis=2)


class BridgeWindowTests(unittest.TestCase):
    def test_shapes_order_units_and_fixed_ordinal_slots(self):
        d=fixture();w=b.accept_windows(**d)
        self.assertEqual(w.x.shape,(56,36,14));self.assertEqual(w.physical_targets.shape,(56,2))
        np.testing.assert_array_equal(w.physical_targets[:,0],d['future'][:,:,0].mean(axis=1)*1000)
        np.testing.assert_array_equal(w.physical_targets[:,1],d['future'][:,:,1].mean(axis=1))
        np.testing.assert_array_equal(w.x[:,:,13],d['history_nodes'])
        self.assertEqual(tuple(np.unique(w.x[:,:,13])),(2.,978.))
    def test_original_input_containers_not_mutated_and_copy_is_immutable(self):
        d=fixture();old=d['x'].copy();w=b.accept_windows(**d);d['x'][:]=0
        np.testing.assert_array_equal(w.x,old)
        for a in (w.x,w.future,w.physical_targets,w.origins,w.nodes):
            with self.assertRaises(ValueError):a.setflags(write=True)
        with self.assertRaises(AttributeError):w.split='val'
    def test_hardened_target_helper_called_before_windows_constructor(self):
        d=fixture();original=d['future'];d['future']=np.ma.array(original,mask=False)
        with mock.patch.object(b.target_validation,'target_means',wraps=b.target_validation.target_means) as call:
            with mock.patch.object(b,'Windows',side_effect=AssertionError('mask erased')):
                with self.assertRaisesRegex(ValueError,'Masked-array'):b.accept_windows(**d)
        self.assertIs(call.call_args.args[0],d['future'])
    def test_target_masks_boolean_dtype_and_masked_flags_rejected(self):
        for key,value in [('future',np.ma.array(fixture()['future'],mask=True)),
                          ('future',fixture()['future'].astype(bool)),
                          ('target_observed',np.ma.array(fixture()['target_observed'],mask=False))]:
            d=fixture();d[key]=value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_bad_targets_nonfinite_missing_shape_and_ranges(self):
        for value in (np.nan,np.inf,-np.inf,-.1):
            d=fixture();d['future'][0,0,0]=value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['future'][0,0,1]=-274
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['target_observed'][0,0,1]=False
        with self.assertRaises(ValueError):b.accept_windows(**d)
        for key in ('future','target_observed'):
            d=fixture();d[key]=d[key][:,:5]
            with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_masked_inputs_and_invalid_physical_feature_rejected(self):
        d=fixture();d['x']=np.ma.array(d['x'],mask=False)
        with self.assertRaises(ValueError):b.accept_windows(**d)
        for channel,value in [(0,-1),(1,101),(5,1.1),(6,-274),(7,-274),(8,101),(9,2),(0,np.nan)]:
            d=fixture();d['x'][0,0,channel]=value
            with self.subTest(channel=channel,value=value):
                with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_no_weather_missingness_expansion_or_feature_changes(self):
        d=fixture();d['observed'][0,0,7]=False
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['features']=tuple(reversed(b.FEATURES))
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['units']=tuple(reversed(b.UNITS))
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_mapping_reorder_unknown_node_and_fractional_index_rejected(self):
        d=fixture();d['mapping']=tuple(reversed(d['mapping']))
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['nodes']=d['nodes'].astype('<U20');d['nodes'][0]='unknown'
        with self.assertRaises(ValueError):b.accept_windows(**d)
        for value in (-1,.5,979):
            d=fixture();d['x'][0,:,13]=value
            with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_same_node_history_and_future_no_concatenation(self):
        for key in ('history_nodes','target_nodes'):
            d=fixture();d[key][0,0]=0
            with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_utc_and_exact_grid_missing_steps_and_future_input_rejected(self):
        d=fixture();d['timezone']='Asia/Kolkata'
        with self.assertRaises(ValueError):b.accept_windows(**d)
        for key in ('history_times','target_times'):
            d=fixture();d[key][0,0]+=np.timedelta64(10,'m')
            with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['origins'][0]+=np.timedelta64(10,'m')
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['origins']=d['origins'].astype('datetime64[s]')
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_duplicate_unordered_or_nat_identity_rejected(self):
        d=fixture();d['origins'][1]=d['origins'][0]
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['nodes'][0],d['nodes'][-1]=d['nodes'][-1],d['nodes'][0]
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['origins'][0]=np.datetime64('NaT','ns')
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_exact_split_containment_and_cross_boundary_rejection(self):
        d=fixture('val',ordinals=(2,),days=1)
        # Shift the complete fixture so the first history starts exactly at Jan28 00:00.
        for key in ('origins','history_times','target_times','source_earliest','source_latest','target_source_earliest','target_source_latest'):
            d[key]-=np.timedelta64(2,'h')
        refresh_calendar(d);b.accept_windows(**d)
        for key in ('origins','history_times','target_times','source_earliest','source_latest','target_source_earliest','target_source_latest'):
            d[key]-=np.timedelta64(1,'h')
        with self.assertRaisesRegex(ValueError,'split leakage'):b.accept_windows(**d)
        d=fixture('val',ordinals=(2,),days=1)
        delta=np.datetime64('2022-02-03T23','ns')-d['origins'][-1]
        for key in ('origins','history_times','target_times','source_earliest','source_latest','target_source_earliest','target_source_latest'):d[key]+=delta
        refresh_calendar(d);b.accept_windows(**d)
        for key in ('origins','history_times','target_times','source_earliest','source_latest','target_source_earliest','target_source_latest'):d[key]+=np.timedelta64(1,'h')
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_source_split_reset_future_or_stale_evidence_rejected(self):
        d=fixture();d['source_earliest'][0,0,0]=np.datetime64('2021-12-31','ns')
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['source_latest'][0,0,0]=d['history_times'][0,0]+np.timedelta64(600,'s')
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['source_earliest'][0,0,0]=d['history_times'][0,0]-np.timedelta64(301,'s')
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['source_earliest'][0,0,7]=d['history_times'][0,0]-np.timedelta64(1300,'s');d['source_latest'][0,0,7]=d['source_earliest'][0,0,7]
        with self.assertRaisesRegex(ValueError,'weather'):b.accept_windows(**d)
    def test_target_contributors_and_coverage_verified(self):
        for key,value in [('history_coverage',.499),('target_coverage',.499),('target_coverage',1.01)]:
            d=fixture();d[key].flat[0]=value
            with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();d['target_source_latest'][0,0,0]=d['target_times'][0,0]+np.timedelta64(600,'s')
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture('val');d['target_source_earliest'][0,0,0]=np.datetime64('2022-01-27','ns')
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_test_split_rejected_without_touching_any_payload(self):
        class Poison:
            def __array__(self,*a,**k):raise AssertionError('test payload touched')
        d=fixture();d['split']='test';d['future']=Poison()
        with self.assertRaises(PermissionError):b.accept_windows(**d)
        d=fixture('val')
        d['origins'][:]=np.datetime64('2022-03-02T12','ns')
        with self.assertRaises(ValueError):b.accept_windows(**d)
    def test_mislabeled_march_payload_rejected_before_target_access(self):
        class Poison:
            def __array__(self,*a,**k):raise AssertionError('March target payload touched')
        d=fixture('val');d['origins'][:]=np.datetime64('2022-03-02T12','ns');d['future']=Poison()
        with mock.patch.object(b.target_validation,'target_means',side_effect=AssertionError('target helper called')):
            with self.assertRaisesRegex(ValueError,'split leakage'):b.accept_windows(**d)
    def test_calendar_definition_and_input_history_shape_are_enforced(self):
        d=fixture();d['x'][0,0,9]=.5
        with self.assertRaisesRegex(ValueError,'Calendar'):b.accept_windows(**d)
        d=fixture();d['x']=d['x'][:,:6]
        with self.assertRaisesRegex(ValueError,'shape'):b.accept_windows(**d)
    def test_input_source_evidence_schema_is_required_and_immutable(self):
        d=fixture();d['source_evidence']['raw_contract_sha256']='bad'
        with self.assertRaises(ValueError):b.accept_windows(**d)
        d=fixture();w=b.accept_windows(**d);d['source_evidence']['raw_contract_sha256']='0'*64
        self.assertEqual(w.source_evidence['raw_contract_sha256'],'1'*64)
        record=w.source_evidence;record['source_kind']='observed_prepared'
        self.assertEqual(w.source_evidence['source_kind'],'synthetic')
    def test_training_only_eligibility_and_no_silent_client_renumber(self):
        w=b.accept_windows(**fixture());e=b.training_eligibility(w)
        self.assertEqual(e,{'3':{'node_index':2,'windows':28,'positive_dates':7},'979':{'node_index':978,'windows':28,'positive_dates':7}})
        self.assertEqual(b.training_eligibility(b.accept_windows(**fixture(days=1))),{})
        with self.assertRaises(PermissionError):b.training_eligibility(b.accept_windows(**fixture('val')))
    def test_initialization_uses_authorized_domain_zero_without_model(self):
        for root in (42,43,44):
            expected=int(np.random.SeedSequence([root,0,0,0]).generate_state(1)[0]);r=b.initialization_record(root)
            self.assertEqual(r['initialization_seed'],expected);self.assertNotEqual(expected,root)
            self.assertFalse(r['historical_reproducibility_claim'])
        for invalid in (True,41,45,42.):
            with self.assertRaises(ValueError):b.initialization_record(invalid)

if __name__=='__main__':unittest.main()
