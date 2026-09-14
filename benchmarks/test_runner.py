"""Checks for measurement windows, pinned sources, and transform conventions."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import run

class BenchmarkTests(unittest.TestCase):
    def test_pinned_sources(self):
        self.assertEqual(run.verify_sources()['nn4mgf_commit'],run.UPSTREAM)
    def test_warmup_once_across_rounds(self):
        t=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        m=run.Meter(t,2)
        list(m.iterations(range(3)))
        list(m.iterations(range(2)))
        self.assertEqual(len(m.times),3)
        self.assertEqual(m.seen,5)
        self.assertIsNone(m.result(100)['peak_allocated_bytes'])
    def test_upstream_sampling_budget(self):
        c=json.loads((Path(__file__).parent/'nn4mgf_public_config.json').read_text())
        run.validate_nn_config(c)
        self.assertEqual(sum(s['epochs'] for s in c['joint_rounds'])*c['train_freq']*c['batch_size'],3276800000)
        c['batch_size']=4
        with self.assertRaises(ValueError):run.validate_nn_config(c)
    def test_transform_sign_conventions(self):
        import torch
        class ExactLogTransform(torch.nn.Module):
            def __init__(self, sign):super().__init__();self.sign=sign
            def forward(self,theta):
                rates=torch.arange(1,21,dtype=torch.float64)
                return -torch.log(1+self.sign*theta/rates).sum(1,keepdim=True)
        h,_=run.evaluate(ExactLogTransform(1),'hcn',20,'cpu',torch)
        n,_=run.evaluate(ExactLogTransform(-1),'nn4mgf',20,'cpu',torch)
        self.assertEqual(h,n)
        for row in h:
            if row['degree']==8:self.assertLess(row['relative_error_percent'],2)

if __name__=='__main__':unittest.main()
