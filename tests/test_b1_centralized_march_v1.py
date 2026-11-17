"""Invented metadata/arrays only. No metric functions or models are executed."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from src import b1_centralized_march_evaluate_v1 as adapter


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='b1-central-march-test-')
        self.root = Path(self.temp.name)
        self.reference = np.array([(0, 1, 100., 20.), (0, 2, 101., 21.), (1, 1, 102., 22.)],
            dtype=[('node_index', '<i8'), ('origin_ns', '<i8'), ('power_w', '<f8'), ('temperature_c', '<f8')])

    def tearDown(self):
        self.temp.cleanup()

    def selection(self):
        return {'seedset': [42,43,44], 'test_evaluation_authorized': False,
                'selected_models': [{'seed': s, 'completed_block': 200, 'selected_block': b} for s,b in adapter.BLOCKS.items()]}

    def test_three_locked_seeds(self):
        self.assertEqual(len(adapter.validate_selected(self.selection())), 3)

    def test_missing_seed(self):
        x=self.selection();x['selected_models'].pop()
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_duplicate_seed(self):
        x=self.selection();x['selected_models'][1]['seed']=42
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_seed_order(self):
        x=self.selection();x['selected_models'].reverse()
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_wrong_selected_block(self):
        x=self.selection();x['selected_models'][1]['selected_block']=200
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_incomplete_training(self):
        x=self.selection();x['selected_models'][1]['completed_block']=190
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_selection_does_not_authorize(self):
        x=self.selection();x['test_evaluation_authorized']=True
        with self.assertRaises(ValueError):adapter.validate_selected(x)

    def test_aligned_rows(self):
        adapter.aligned_reference(self.reference,self.reference.copy())

    def test_missing_window(self):
        with self.assertRaises(ValueError):adapter.aligned_reference(self.reference,self.reference[:-1])

    def test_changed_node(self):
        x=self.reference.copy();x['node_index'][-1]=2
        with self.assertRaises(ValueError):adapter.aligned_reference(self.reference,x)

    def test_changed_origin(self):
        x=self.reference.copy();x['origin_ns'][-1]+=1
        with self.assertRaises(ValueError):adapter.aligned_reference(self.reference,x)

    def test_changed_target(self):
        x=self.reference.copy();x['temperature_c'][0]+=1
        with self.assertRaises(ValueError):adapter.aligned_reference(self.reference,x)

    def test_reordered_windows(self):
        with self.assertRaises(ValueError):adapter.aligned_reference(self.reference,self.reference[::-1])

    def test_duplicate_window(self):
        x=self.reference.copy();x[1]=x[0]
        with self.assertRaises(ValueError):adapter.aligned_reference(x,x.copy())

    def test_nonfinite_targets(self):
        for value in (np.nan,np.inf):
            x=self.reference.copy();x['power_w'][0]=value
            with self.assertRaises(ValueError):adapter.aligned_reference(x,x.copy())

    def test_masked_targets(self):
        x=np.ma.array(self.reference,mask=False)
        with self.assertRaises(ValueError):adapter.aligned_reference(x,x)

    def test_out_of_bounds_node(self):
        x=self.reference.copy();x['node_index'][-1]=979
        with self.assertRaises(ValueError):adapter.aligned_reference(x,x.copy())

    def test_finite_float64_predictions(self):
        adapter.valid_predictions(np.zeros((3,2),dtype='f8'),3)

    def test_prediction_count(self):
        with self.assertRaises(ValueError):adapter.valid_predictions(np.zeros((2,2)),3)

    def test_prediction_channels(self):
        with self.assertRaises(ValueError):adapter.valid_predictions(np.zeros((3,1)),3)

    def test_prediction_dtype(self):
        for dtype in ('f4','bool','i8'):
            with self.assertRaises(ValueError):adapter.valid_predictions(np.zeros((3,2),dtype=dtype),3)

    def test_prediction_finite(self):
        x=np.zeros((3,2));x[0,0]=np.nan
        with self.assertRaises(ValueError):adapter.valid_predictions(x,3)

    def test_prediction_masks(self):
        with self.assertRaises(ValueError):adapter.valid_predictions(np.ma.array(np.zeros((3,2)),mask=False),3)

    def test_existing_output(self):
        with self.assertRaises(FileExistsError):adapter.unused(self.root)

    def test_pending_output(self):
        (self.root/'.out.pending').mkdir()
        with self.assertRaises(FileExistsError):adapter.unused(self.root/'out')

    def test_symlink_output(self):
        (self.root/'link').symlink_to(self.root/'absent')
        with self.assertRaises(ValueError):adapter.unused(self.root/'link')

    def test_parent_required(self):
        with self.assertRaises(ValueError):adapter.unused(self.root/'absent'/'out')

    def test_unused_output_nonmutating(self):
        self.assertEqual(adapter.unused(self.root/'out'),self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_exact_file_hash(self):
        p=self.root/'receipt.json';p.write_bytes(b'original');b=adapter.storage.binding(p)
        adapter.base.check(p,b);p.write_bytes(b'changed')
        with self.assertRaises(ValueError):adapter.base.check(p,b)

    def test_inventory_extra(self):
        (self.root/'completed.json').write_text('{}');(self.root/'extra').write_text('x')
        with self.assertRaises(ValueError):adapter.check_inventory(self.root,{'artifacts':{}})

    def test_inventory_missing(self):
        (self.root/'completed.json').write_text('{}')
        with self.assertRaises(ValueError):adapter.check_inventory(self.root,{'artifacts':{'missing':{'bytes':1,'sha256':'0'*64}}})

    def test_inventory_bad_hash(self):
        (self.root/'completed.json').write_text('{}');(self.root/'a').write_bytes(b'a')
        with self.assertRaises(ValueError):adapter.check_inventory(self.root,{'artifacts':{'a':{'bytes':1,'sha256':'0'*64}}})

    def test_export_schema(self):
        m={'schema':'b1.centralized.training.manifest.v1','seed':43}
        r={'seed':43,'selected_block':190}
        s={'schema':'b1.centralized.selected.model.v1','manifest_sha256':adapter.base.digest(m)}
        adapter.validate_export(r,s,m)
        s['schema']='b1.production.selected.model.v1'
        with self.assertRaises(ValueError):adapter.validate_export(r,s,m)

    def test_export_manifest_hash(self):
        with self.assertRaises(ValueError):adapter.validate_export({'seed':43,'selected_block':190},{'schema':'b1.centralized.selected.model.v1','manifest_sha256':'0'*64},{'schema':'b1.centralized.training.manifest.v1','seed':43})

    def test_export_wrong_seed(self):
        m={'schema':'b1.centralized.training.manifest.v1','seed':42}
        with self.assertRaises(ValueError):adapter.validate_export({'seed':43,'selected_block':190},{},m)

    def test_exact_receipt(self):
        x={'operator_assertion':{'explicit_operator_authorization':True},'context':{'hash':'a'}}
        adapter.check_receipt(x,copy.deepcopy(x))
        y=copy.deepcopy(x);y['context']['hash']='b'
        with self.assertRaises(PermissionError):adapter.check_receipt(y,x)

    def test_receipt_never_infers_authorization(self):
        x={'operator_assertion':{'explicit_operator_authorization':False}}
        with self.assertRaises(PermissionError):adapter.check_receipt(x,x)

    def write_operator(self,plan,operation=adapter.OP_PREFLIGHT,**updates):
        x={'operation':operation,'explicit_operator_authorization':True,'uid':os.getuid(),
           'bindings':adapter.operator_bindings(plan,'cpu',True),'authorization_text':'Invented fixture authorization, not production approval',
           'expires_at_unix':time.time()+300}
        x.update(updates);p=self.root/'operator.json';p.write_text(json.dumps(x));return p

    def test_operator_scope(self):
        plan={**adapter.PATHS};p=self.write_operator(plan,operation='wrong')
        with self.assertRaises(PermissionError):adapter.require_operator(plan,p,'cpu',True)

    def test_operator_expired(self):
        plan={**adapter.PATHS};p=self.write_operator(plan,expires_at_unix=time.time()-1)
        with self.assertRaises(PermissionError):adapter.require_operator(plan,p,'cpu',True)

    def test_operator_user(self):
        plan={**adapter.PATHS};p=self.write_operator(plan,uid=os.getuid()+1)
        with self.assertRaises(PermissionError):adapter.require_operator(plan,p,'cpu',True)

    def test_operator_false(self):
        plan={**adapter.PATHS};p=self.write_operator(plan,explicit_operator_authorization=False)
        with self.assertRaises(PermissionError):adapter.require_operator(plan,p,'cpu',True)

    def test_valid_synthetic_preflight_operator(self):
        plan={**adapter.PATHS};p=self.write_operator(plan)
        self.assertTrue(adapter.require_operator(plan,p,'cpu',True)['explicit_operator_authorization'])

    def test_missing_auth_blocks_every_data_stage(self):
        for stage in ('preflight','authorize','evaluate','verify'):
            with mock.patch.object(adapter,'load_plan',return_value={}),mock.patch.object(adapter,'frozen_context',side_effect=AssertionError('March forbidden')) as march:
                with self.assertRaises(PermissionError):adapter.main(['--plan','invented','--stage',stage,'--device','cpu'])
                march.assert_not_called()

    def test_preflight_rejects_auth_before_context(self):
        with mock.patch.object(adapter,'require_operator',side_effect=PermissionError('no')),mock.patch.object(adapter,'frozen_context') as context:
            with self.assertRaises(PermissionError):adapter.preflight({},'missing','cpu','missing')
            context.assert_not_called()

    def test_authorize_rejects_auth_before_context(self):
        with mock.patch.object(adapter,'require_operator',side_effect=PermissionError('no')),mock.patch.object(adapter,'frozen_context') as context:
            with self.assertRaises(PermissionError):adapter.authorize({},'missing','cpu','missing')
            context.assert_not_called()

    def test_verify_rejects_auth_before_context(self):
        with mock.patch.object(adapter,'require_operator',side_effect=PermissionError('no')),mock.patch.object(adapter,'frozen_context') as context:
            with self.assertRaises(PermissionError):adapter.verify({},'missing','cpu','missing')
            context.assert_not_called()

    def test_invalid_receipt_prevents_model_loading(self):
        plan={**adapter.PATHS};evidence={'invented':True}
        with mock.patch.object(adapter,'require_operator',return_value={}),mock.patch.object(adapter,'unused'),mock.patch.object(adapter,'resource_budget'),mock.patch.object(adapter,'frozen_context',return_value=({},object(),Path('/invented'),{},evidence)),mock.patch.object(adapter,'read_preflight',return_value=evidence),mock.patch.object(adapter.base,'read_json',return_value={'invalid':True}),mock.patch.object(adapter,'receipt_body',return_value={'expected':True}),mock.patch.object(adapter,'load_selected',side_effect=AssertionError('Model forbidden')) as model:
            with self.assertRaises(PermissionError):adapter.evaluate(plan,'invented','cpu','invented',adapter.ROOT/plan['receipt_output']/'receipt.json')
            model.assert_not_called()

    def test_describe_and_inspect_do_not_enter_data_or_metrics(self):
        for stage in ('describe','inspect'):
            with mock.patch.object(adapter,'load_plan',return_value={}),mock.patch.object(adapter,'locked_context'),mock.patch.object(adapter,'frozen_context',side_effect=AssertionError('March forbidden')) as context,mock.patch.object(adapter,'comparison_tables',side_effect=AssertionError('Metrics forbidden')) as metrics,mock.patch.object(adapter,'load_selected',side_effect=AssertionError('Model forbidden')) as model:
                adapter.main(['--plan','invented','--stage',stage]);context.assert_not_called();metrics.assert_not_called();model.assert_not_called()

    def test_cuda_environment_gate_does_not_initialize(self):
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(ValueError):adapter.device_record('cuda:0')

    def test_production_plan_source_and_protocol_bindings(self):
        path=adapter.ROOT/'configs/experiments/b1_no_ff/centralized_march_test_v1.json'
        self.assertEqual(adapter.load_plan(path)['protocol']['centralized_selected_blocks'],{'42':200,'43':190,'44':190})

    def rejected_plan(self,mutate):
        path=adapter.ROOT/'configs/experiments/b1_no_ff/centralized_march_test_v1.json'
        plan=json.loads(path.read_text());mutate(plan);p=self.root/'bad-plan.json';p.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):adapter.load_plan(p)

    def test_wrong_selection_hash_in_plan(self):
        self.rejected_plan(lambda p:p['selection_lock'].update(sha256='0'*64))

    def test_wrong_source_hash_in_plan(self):
        self.rejected_plan(lambda p:p['sources'][adapter.SOURCE].update(sha256='0'*64))

    def test_wrong_output_in_plan(self):
        self.rejected_plan(lambda p:p.update(evaluation_output='output/experiments/b1_no_ff/march_test_evaluation_v2'))

    def test_wrong_mapping_in_plan(self):
        self.rejected_plan(lambda p:p['protocol'].update(mapping_digest='0'*64))

    def test_wrong_metric_protocol(self):
        self.rejected_plan(lambda p:p['protocol'].update(macro_node='pooled instead'))

    def test_changed_resource_limits(self):
        self.rejected_plan(lambda p:p['limits'].update(reserve_bytes=1))


if __name__=='__main__':unittest.main()
