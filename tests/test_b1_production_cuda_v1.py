"""Optional GPU integration smoke: invented data only, never real artifacts."""
from pathlib import Path
import contextlib,io,tempfile,unittest
import torch
from src import b1_real_fedavg_runner_v1 as runner
from src.b1_production_data_v1 import read_json
from test_b1_production_runner_v1 import InventedData,test_config


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable: GPU runner/resume smoke not executed')
class CUDARunnerSmoke(unittest.TestCase):
    def test_invented_two_round_GPU_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);data=InventedData(count=2);cfg=test_config()
            runner.run_seed(data,root/'clean',cfg,42,'cuda:0')
            partial=runner.run_seed(data,root/'resume',cfg,42,'cuda:0',stop_after_round=1)
            self.assertEqual(partial['status'],'verified_partial')
            result=runner.run_seed(data,root/'resume',cfg,42,'cuda:0',resume=True)
            self.assertEqual(result['status'],'training_complete')
            for index in (0,1,2):
                a=read_json(root/'clean/rounds'/f'{index:04d}'/'receipt.json')
                b=read_json(root/'resume/rounds'/f'{index:04d}'/'receipt.json')
                self.assertEqual(a['state_fingerprint'],b['state_fingerprint'])
                self.assertIsNotNone(b['GPU_peak_allocated_bytes'])
            check=runner.verify_run(data,root/'resume',{'settings':cfg['settings'],'sources':{}},42)
            self.assertEqual(check['status'],'verified_complete')

if __name__=='__main__':unittest.main()
