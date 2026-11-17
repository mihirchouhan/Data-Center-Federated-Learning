"""Invented CPU state dictionaries only; model construction is mocked."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from src import b1_march_integrity_v1 as k
from src import b1_march_evaluate_v1 as e
from src import b1_preparation_io_v1 as io
from src.b1_target_scaling_v1 import TargetScaling
from src.b1_real_fedavg_runner_v1 import state_contract
from src.b1_fedavg_orchestration_v1 import RoundSnapshot
import src.b1_device_adapter_v1 as adapter


class CheckpointTests(unittest.TestCase):
    def state(self):
        shapes={'core.node_embedding.weight':(979,8),'core.head.weight':(2,64),'core.head.bias':(2,)}
        for layer in (0,1):
            shapes.update({f'core.recurrent.weight_ih_l{layer}':(192,21 if layer==0 else 64),
                f'core.recurrent.weight_hh_l{layer}':(192,64),f'core.recurrent.bias_ih_l{layer}':(192,),f'core.recurrent.bias_hh_l{layer}':(192,)})
        state={n:torch.zeros(s,dtype=torch.float32,device='cpu') for n,s in shapes.items()}
        state['target_mean']=torch.tensor([600.,20.],dtype=torch.float64,device='cpu')
        state['target_std']=torch.tensor([100.,2.],dtype=torch.float64,device='cpu')
        return state
    def exercise(self,kind):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp);state=self.state();scale=TargetScaling((600.,20.),(100.,2.));scaler=type('SyntheticScaler',(),{'target_scaling':scale})()
            roles=state_contract(state,scale);fingerprint=RoundSnapshot(state,roles,round_id=200).fingerprint
            manifest={'synthetic_fixture_only':True};(root/'manifest.json').write_bytes(io.canonical(manifest))
            saved={'schema':'b1.production.selected.model.v1','manifest_sha256':k.digest(manifest),'state':state}
            if kind=='bad_envelope':saved['schema']='wrong'
            if kind=='wrong_manifest':saved['manifest_sha256']='0'*64
            if kind=='wrong_scale':state['target_std']=torch.tensor([101.,2.],dtype=torch.float64)
            if kind=='wrong_shape':state['core.head.weight']=torch.zeros((1,64))
            if kind=='nonfinite':state['core.head.bias'][0]=float('nan')
            torch.save(saved,root/'model.pt')
            record={'selected_model':{'path':'model.pt',**io.binding(root/'model.pt')},'run_manifest':{'path':'manifest.json'},'seed':42,
                'selected_state_fingerprint':'0'*64 if kind=='wrong_fingerprint' else fingerprint}
            model=mock.Mock();model.restore_round.side_effect=lambda snapshot:self.assertEqual(snapshot.fingerprint,fingerprint)
            with mock.patch.object(k,'ROOT',root),mock.patch.object(adapter,'B1DeviceAdapter',return_value=model) as factory:
                if kind=='valid':
                    result=e.load_selected(record,{},scaler,'cpu');self.assertIs(result,model);factory.assert_called_once_with(target_scaling=scale,root_seed=42,device='cpu');model.eval.assert_called_once();model.assert_not_called()
                else:
                    with self.assertRaises(ValueError):e.load_selected(record,{},scaler,'cpu')
                    factory.assert_not_called()
    def test_valid_synthetic_state_transfer_no_forward(self):self.exercise('valid')
    def test_wrong_envelope_rejected(self):self.exercise('bad_envelope')
    def test_wrong_manifest_rejected(self):self.exercise('wrong_manifest')
    def test_wrong_target_scaling_rejected(self):self.exercise('wrong_scale')
    def test_incompatible_head_shape_rejected(self):self.exercise('wrong_shape')
    def test_nonfinite_weights_rejected(self):self.exercise('nonfinite')
    def test_wrong_state_fingerprint_rejected(self):self.exercise('wrong_fingerprint')
    def test_no_GPU_initialization(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main(verbosity=2)
