"""Invented bins/targets and metric arithmetic only; no model initialization."""
import copy,unittest
import numpy as np
from src import b1_march_persistence_v1 as m
from src import b1_march_data_v1 as d
from src import b1_march_evaluate_v1 as e


def fixture():
    n=len(d.EDGES)-1
    x=np.zeros((n,13));x[:,0]=1.;x[:,1:5]=[10,5,80,5];x[:,5]=.5;x[:,6:9]=[20,10,50]
    seconds=d.EDGES[1:]/1e9;daily=2*np.pi*(seconds%86400)/86400;weekly=2*np.pi*((seconds+259200)%604800)/604800
    x[:,9:13]=np.column_stack((np.sin(daily),np.cos(daily),np.sin(weekly),np.cos(weekly)))
    t=np.tile(d.EDGES[:-1,None],(1,9)).astype('datetime64[ns]')
    b={'features':x,'earliest':t,'latest':t.copy(),'coverage':np.ones((n,7))}
    p=d.packet('4',b,np.array([36,42]));y=d.validate_packet(p)
    ref=np.empty(2,dtype=d.audit.WINDOW_DTYPE);ref['node_index']=3;ref['origin_ns']=p['origins'].view('i8');ref['power_w']=y[:,0];ref['temperature_c']=y[:,1]
    return p,ref


class PersistenceTests(unittest.TestCase):
    def setUp(self):self.p,self.ref=fixture()
    def reject(self,field,fn):
        p=copy.deepcopy(self.p);fn(p[field])
        with self.assertRaises(ValueError):m.packet_persistence(p)
    def test_units_order(self):
        predicted,actual=m.packet_persistence(self.p);np.testing.assert_array_equal(predicted,[[1000,20],[1000,20]]);np.testing.assert_array_equal(predicted,actual)
    def test_different_future_does_not_change_prediction(self):
        p=copy.deepcopy(self.p);p['future'][:,:,0]=2.;p['future'][:,:,1]=25
        predicted,y=m.packet_persistence(p);np.testing.assert_array_equal(predicted,[[1000,20],[1000,20]]);np.testing.assert_array_equal(y,[[2000,25],[2000,25]])
    def test_last_bin_not_history_mean(self):
        p=copy.deepcopy(self.p);p['x'][:,-1,0]=1.5;p['x'][:,-1,6]=22
        pred,_=m.packet_persistence(p);np.testing.assert_array_equal(pred,[[1500,22],[1500,22]])
    def test_no_input_mutation(self):
        old=self.p['x'].copy();m.packet_persistence(self.p);np.testing.assert_array_equal(old,self.p['x'])
    def test_masked_target_unmasked_mask_rejected(self):
        p=copy.deepcopy(self.p);p['future']=np.ma.array(p['future'],mask=False)
        with self.assertRaises(ValueError):m.packet_persistence(p)
    def test_masked_target_partial_rejected(self):
        p=copy.deepcopy(self.p);p['future']=np.ma.array(p['future'],mask=False);p['future'].mask[0,0,0]=True
        with self.assertRaises(ValueError):m.packet_persistence(p)
    def test_boolean_targets(self):
        p=copy.deepcopy(self.p);p['future']=np.ones((2,6,2),dtype=bool)
        with self.assertRaises(ValueError):m.packet_persistence(p)
    def test_missing_observed_target(self):self.reject('target_observed',lambda a:a.__setitem__((0,0,1),False))
    def test_nan_future(self):self.reject('future',lambda a:a.__setitem__((0,0,1),np.nan))
    def test_inf_future(self):self.reject('future',lambda a:a.__setitem__((0,0,0),np.inf))
    def test_negative_future_power(self):self.reject('future',lambda a:a.__setitem__((0,0,0),-1.))
    def test_missing_last_input(self):self.reject('x',lambda a:a.__setitem__((0,-1,0),np.nan))
    def test_input_mask(self):self.reject('observed',lambda a:a.__setitem__((0,-1,6),False))
    def test_last_timestamp(self):self.reject('history_times',lambda a:a.__setitem__((0,-1),a[0,-1]+np.timedelta64(1,'m')))
    def test_future_source_leakage(self):self.reject('source_latest',lambda a:a.__setitem__((0,-1,0),self.p['origins'][0]))
    def test_low_coverage(self):self.reject('history_coverage',lambda a:a.__setitem__((0,-1,0),.49))
    def test_cross_node_history(self):self.reject('history_nodes',lambda a:a.__setitem__((0,-1),4))
    def test_exact_alignment(self):m.alignment(self.ref,self.p,d.validate_packet(self.p))
    def test_swapped_origin_alignment(self):
        p=copy.deepcopy(self.p);p['origins']=p['origins'][::-1]
        with self.assertRaises(ValueError):m.alignment(self.ref,p,d.validate_packet(self.p))
    def test_wrong_node_alignment(self):
        p=copy.deepcopy(self.p);p['nodes'][0]='5'
        with self.assertRaises(ValueError):m.alignment(self.ref,p,d.validate_packet(self.p))
    def test_wrong_target_alignment(self):
        ref=self.ref.copy();ref['temperature_c'][0]+=1
        with self.assertRaises(ValueError):m.alignment(ref,self.p,d.validate_packet(self.p))
    def test_missing_reference_target(self):
        ref=self.ref.copy();ref['power_w'][0]=np.nan
        with self.assertRaises(ValueError):m.alignment(ref,self.p,d.validate_packet(self.p))
    def test_duplicate_identity(self):
        ref=self.ref.copy();ref[1]=ref[0]
        with self.assertRaises(ValueError):m.alignment(ref,self.p,d.validate_packet(self.p))
    def test_invalid_ordinal(self):
        ref=self.ref.copy();ref['node_index'][0]=-1
        with self.assertRaises(ValueError):m.alignment(ref,self.p,d.validate_packet(self.p))
    def test_seedless_prediction_deterministic(self):
        np.random.seed(1);a,_=m.packet_persistence(self.p);np.random.seed(999);b,_=m.packet_persistence(self.p);np.testing.assert_array_equal(a,b)


class MetricsTests(unittest.TestCase):
    def data(self):
        actual=np.array([[0.,20.],[0.,20.],[0.,20.],[0.,20.]])
        pred=actual+np.array([[1.,2.],[1.,2.],[1.,2.],[9.,6.]])
        nodes=np.array(['1','1','1','2']);times=np.arange(4).astype('datetime64[h]').astype('datetime64[ns]')
        return actual,pred,nodes,times
    def test_pooled_macro_independent(self):
        a,p,n,t=self.data();s=e.metrics(a,p,n,t);e.compare_metrics(s,e.independent_metrics(a,p,n,t))
        self.assertEqual(s['pooled']['mae'],[3.,3.]);self.assertEqual(s['macro_node']['mae'],[5.,4.]);self.assertAlmostEqual(s['pooled']['rmse'][0],np.sqrt(21));self.assertEqual(s['macro_node']['rmse'],[5.,4.])
    def test_improvement(self):self.assertEqual(m.change(10.,8.)['improvement_percent'],20.)
    def test_degradation(self):self.assertEqual(m.change(10.,12.)['direction'],'degraded')
    def test_tie(self):self.assertEqual(m.change(3.,3.)['direction'],'equal')
    def test_zero_baseline(self):self.assertIsNone(m.change(0.,1.)['improvement_percent']);self.assertEqual(m.change(0.,0.)['direction'],'equal')
    def test_negative_error_rejected(self):
        with self.assertRaises(ValueError):m.change(-1.,0.)
    def test_nonfinite_metric(self):
        with self.assertRaises(ValueError):m.change(np.inf,0.)
    def test_distribution(self):self.assertEqual(m.distribution([-2.,0.,2.])['median'],0.)
    def test_missing_seed_rejected(self):
        a,p,n,t=self.data();s=e.metrics(a,p,n,t)
        with self.assertRaises(ValueError):m.tables(s,{42:s,43:s},('1','2'))
    def test_mean_scores_not_ensemble_and_node_consistency(self):
        a,p,n,t=self.data();base=e.metrics(a,p,n,t);scores={s:e.metrics(a,a+(p-a)*f,n,t) for s,f in [(42,.5),(43,1.),(44,2.)]}
        tab=m.tables(base,scores,('1','2'));self.assertEqual(len(tab['node_comparisons.csv']),32)
        row=next(r for r in tab['comparison_metrics.csv'] if (r['model'],r['aggregation'],r['output'],r['metric'])==('seed_mean','pooled','power_W','mae'))
        self.assertAlmostEqual(row['B1'],3.5);self.assertAlmostEqual(row['B1_sample_sd'],np.std([1.5,3.,6.],ddof=1))
        self.assertTrue(all(r['mixed_or_partial_ties']==2 for r in tab['seed_consistency.csv']))
    def test_unequal_nodes_rejected(self):
        a,p,n,t=self.data();base=e.metrics(a,p,n,t);bad=copy.deepcopy(base);bad['per_node']['1']['samples']=2
        with self.assertRaises(ValueError):m.tables(base,{42:base,43:base,44:bad},('1','2'))
    def test_duplicate_metric_identity(self):
        a,p,n,t=self.data();t[1]=t[0]
        with self.assertRaises(ValueError):e.metrics(a,p,n,t)

if __name__=='__main__':unittest.main()
