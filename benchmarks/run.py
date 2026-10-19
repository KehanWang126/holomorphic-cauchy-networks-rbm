#!/usr/bin/env python3
"""Isolated HCN / pinned NN4MGF resource benchmarks, with recorded provenance."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
UPSTREAM = '1956666902281f6ce4558ca82d8d15fa038f06fb'

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def verify_sources():
    lock = json.loads((HERE / 'sources.json').read_text())
    for rel, digest in lock['sha256'].items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Pinned source changed: ' + rel)
    return lock

def load_hcn(d):
    """Execute setup cells, then definitions only: never notebook run dispatch."""
    path = ROOT / f'notebooks/d{d}_skew_symmetric_product_form.ipynb'
    cells = json.loads(path.read_text())['cells']
    module = types.ModuleType('hcn_benchmark_source')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    # The pinned notebooks place training dispatch after setup/model/loss cells.
    setup_end = 12 if d == 20 else 10
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        if i > setup_end:
            tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(tree, str(path) + f':cell{i}', 'exec'), ns)
    for key, value in list(ns.items()):
        if isinstance(value, ns['torch'].nn.Module):
            del ns[key]
    import gc
    gc.collect()
    return ns

class Meter:
    def __init__(self, torch, warmup):
        self.torch = torch
        self.warmup = warmup
        self.times = []
        self.seen = 0
    def sync(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
    def iterations(self, values, **kwargs):
        for i in values:
            self.sync()
            if self.seen == self.warmup and self.torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            yield i
            self.sync()
            elapsed = time.perf_counter() - start
            if self.seen >= self.warmup:
                self.times.append(elapsed)
            self.seen += 1
            if (i + 1) % 100 == 0:
                print(f'Completed {i+1} updates', flush=True)
    def result(self, samples_per_update):
        t = self.torch
        total = sum(self.times)
        return dict(measured_updates=len(self.times), measured_update_seconds=total,
                    mean_update_seconds=statistics.mean(self.times),
                    median_update_seconds=statistics.median(self.times),
                    base_inputs_per_second=len(self.times)*samples_per_update/total,
                    peak_allocated_bytes=t.cuda.max_memory_allocated() if t.cuda.is_available() else None,
                    peak_reserved_bytes=t.cuda.max_memory_reserved() if t.cuda.is_available() else None)

def evaluate(model, method, d, device, torch):
    import mpmath as mp
    rows = []
    model.eval()
    def tail_transform(s):
        z = complex(s)
        if z == 0:
            raise ValueError('Fixed Talbot nodes should be nonzero')
        # NN4MGF uses E exp(theta X); HCN uses E exp(-theta X).
        theta = torch.full((1, d), z if method == 'hcn' else -z,
                           dtype=torch.complex128, device=device)
        with torch.no_grad():
            phi = torch.exp(model(theta)[0, 0]).item()
        return (1-mp.mpc(phi))/s
    for degree in (5, 8):
        for threshold in range(1, 11):
            truth = -math.expm1(d*math.log1p(-math.exp(-threshold)))
            value = float(mp.invertlaplace(tail_transform, threshold, method='talbot', degree=degree))
            finite = math.isfinite(value)
            clipped = min(1.0, max(0.0, value)) if finite else None
            rows.append(dict(degree=degree, t=threshold, true_value=truth,
                             raw_prediction=value if finite else None,
                             clipped_prediction=clipped,
                             raw_invalid=(not finite or not 0.0 <= value <= 1.0),
                             relative_error_percent=100*abs(value-truth)/truth if finite else None,
                             clipped_relative_error_percent=100*abs(clipped-truth)/truth if finite else None))
    summaries = []
    for degree in (5, 8):
        for end in (5, 8, 10):
            points = [r for r in rows if r['degree']==degree and r['t']<=end]
            raw_vals = [r['relative_error_percent'] for r in points]
            clipped_vals = [r['clipped_relative_error_percent'] for r in points]
            raw_mre = statistics.mean(raw_vals) if all(v is not None for v in raw_vals) else None
            clipped_mre = statistics.mean(clipped_vals) if all(v is not None for v in clipped_vals) else None
            summaries.append(dict(degree=degree, interval=f'1:{end}',
                                  mre_percent=clipped_mre, raw_mre_percent=raw_mre,
                                  clipped_mre_percent=clipped_mre,
                                  raw_invalid_count=sum(r['raw_invalid'] for r in points)))
    return rows, summaries

def validate_nn_config(config):
    for key in ('batch_size', 'train_freq', 'max_grad_sample', 'max_boundary_grad_sample'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['batch_size'] < 50:
        raise ValueError('NN4MGF boundary consistency requires batch_size >= 50')
    if not config['joint_rounds']:
        raise ValueError('At least one joint training round is required')
    for stage in config['joint_rounds']:
        if not isinstance(stage['epochs'], int) or stage['epochs'] <= 0:
            raise ValueError('epochs must be a positive integer')
        if stage['lr'] <= 0 or stage.get('T0', stage['epochs']) <= 0:
            raise ValueError('Invalid learning-rate schedule')


def worker(args):
    import torch
    import numpy as np
    import mpmath
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', method=args.method, dimension=args.dimension,
                  seed=args.seed if args.method=='hcn' else None, mode=args.mode, nn4mgf_commit=UPSTREAM,
                  torch=str(torch.__version__), numpy=np.__version__, mpmath=mpmath.__version__,
                  python=sys.version, cuda_runtime=torch.version.cuda,
                  measurement='CUDA synchronized update wall time; PyTorch allocator peaks',
                  protocol='HCN fixed endpoint / NN4MGF 300k schedule with upstream two-batch updates',
                  schedule_source='https://arxiv.org/html/2607.08091v1#A1',
                  seed_provenance='HCN explicit seed; NN4MGF RNG left unseeded by wrapper')
    try:
        report['source_lock'] = verify_sources()
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; no automatic CPU fallback')
        if args.mode != 'smoke' and args.device != 'cuda':
            raise RuntimeError('Resource measurements require CUDA; use smoke for CPU integration checks')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        report['device'] = device
        report['gpu'] = torch.cuda.get_device_name() if device=='cuda' else None
        report['gpu_total_memory_bytes'] = torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None
        torch.set_num_threads(args.cpu_threads)
        report['cpu_threads'] = args.cpu_threads
        report['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        if device=='cuda':
            try:
                info=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total',
                                     '--format=csv,noheader'],capture_output=True,text=True,timeout=10)
                report['nvidia_smi_inventory']=info.stdout.strip()
            except (OSError, subprocess.TimeoutExpired):
                report['nvidia_smi_inventory']=None
        if args.method == 'hcn':
            random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
            if device == 'cuda': torch.cuda.manual_seed_all(args.seed)
        d=args.dimension
        warmup = args.warmup if args.mode=='profile' else 0
        steps = (25000 if d==20 else 50000) if args.method=='hcn' else 300000
        if args.mode == 'smoke': steps=2
        if args.mode == 'profile': steps=args.steps+warmup
        meter=Meter(torch,warmup)
        os.chdir(out)
        if args.method == 'hcn':
            ns=load_hcn(d)
            model=ns['build_model'](args.seed)
            config=dict(ns['TRAIN_CONFIG'])
            if args.mode=='smoke':
                for k in ['batch_bar','batch_boundary','batch_monotonicity','batch_bdary','batch_mono']:
                    if k in config:config[k]=4
                ns['TRAIN_CONFIG']=config
            optimizer=torch.optim.Adam(model.parameters(),lr=config['lr_phase1' if d==20 else 'lr_stage1'])
            if d==20:
                batches=[config[k] for k in ['batch_bar','batch_boundary','batch_monotonicity']]
            else:
                batches=[config[k] for k in ['batch_bar','batch_bdary','batch_mono']]
            report['training_config']=config
            samples=sum(batches)
            if device=='cuda':torch.cuda.empty_cache()
            meter.sync(); started=time.perf_counter()
            for i in meter.iterations(range(steps)):
                if i==ns['LR_SWITCH_AFTER_STEP']:
                    for group in optimizer.param_groups:group['lr']=config['lr_phase2' if d==20 else 'lr_stage2']
                if d==20:ns['optimizer_step'](model,optimizer,*batches)
                else:ns['run_sequential_backward_training_step'](model,optimizer,ns['PARAMS'],config)
            meter.sync(); train_seconds=time.perf_counter()-started
        else:
            spec=importlib.util.spec_from_file_location('nn4mgf_pinned',HERE/'vendor/nn4mgf/fit_mgf.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            R=torch.eye(d,dtype=torch.float64)
            for i in range(1,d):R[i,i-1]=-1
            trainer=mod.MGFTrainer(d,-torch.ones(d),R+R.T,R,hidden_dim=64,dir='benchmark',
                                   x_min=-5,x_max=.5,y_min=-5,y_max=5)
            model=trainer.model
            batch=64 if args.mode=='smoke' else 16384
            config=dict(lb=-5,ub=.5,imag_lb=-5,imag_ub=5,excluded_boxes=[],full_gradient=False,
                        theta_eval=None,batch_size=batch,max_grad_sample=4 if args.mode=='smoke' else 1024,
                        max_boundary_grad_sample=4 if args.mode=='smoke' else 128,train_freq=2,
                        joint_rounds=[dict(epochs=steps,lr=1e-3,T0=100000,eta_min=1e-4)],
                        individual_rounds=None,lam_monotone=10.,lam_CR=10.,lam_growth=0.,
                        lam_zero_anchor=.1,lam_boundary_consistent=10.,anchor_set=None)
            if args.mode=='full':
                config.update(json.loads((HERE/'nn4mgf_paper_config.json').read_text()))
            if args.nn_config:
                overrides=json.loads(Path(args.nn_config).read_text())
                allowed={'joint_rounds','train_freq','batch_size','max_grad_sample','max_boundary_grad_sample'}
                if set(overrides)-allowed:
                    raise ValueError('Unsupported NN4MGF configuration fields')
                if args.mode!='full':
                    raise ValueError('--nn-config is for full runs only')
                config.update(overrides)
                steps=sum(r['epochs'] for r in config['joint_rounds'])
                report['protocol']='HCN fixed endpoint / explicitly supplied NN4MGF configuration'
                report['nn_config_sha256']=hashlib.sha256(Path(args.nn_config).read_bytes()).hexdigest()
            validate_nn_config(config)
            report['training_config']=config
            samples=config['batch_size']*config['train_freq']
            # Retain upstream train() arithmetic, optimizer and scheduler exactly.
            # The progress iterator measures each update including its scalar logging.
            mod.tqdm=meter.iterations
            Path('Plots/benchmark').mkdir(parents=True)
            if device=='cuda':torch.cuda.empty_cache()
            meter.sync();started=time.perf_counter();trainer.train(**config)
            meter.sync();train_seconds=time.perf_counter()-started
        report.update(meter.result(samples))
        report.update(training_call_seconds=train_seconds,total_updates=steps,
                      base_inputs_per_update=samples,total_sampled_base_inputs=steps*samples,
                      warmup_updates=warmup,
                      parameter_real_scalars=sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters()),
                      precision=str(next(model.parameters()).dtype))
        # Timing/peaks above are captured before checkpoint saving and tail evaluation.
        torch.save(dict(model_state_dict=model.state_dict(),report=report),out/'endpoint.pt')
        rows, summary=evaluate(model,args.method,d,device,torch)
        with (out/'tail_predictions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        report['tail_mre']=summary
        report['accuracy_role']='final fixed endpoint' if args.mode=='full' else 'integration diagnostic only'
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        dump(out/'report.json',report)
        raise
    dump(out/'report.json',report)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','profile','full'],default='smoke')
    p.add_argument('--dimensions',type=int,nargs='+',choices=[20,30],default=[20,30])
    p.add_argument('--methods',nargs='+',choices=['hcn','nn4mgf'],default=['hcn','nn4mgf'])
    p.add_argument('--seeds',type=int,nargs='+',default=[12345])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--output',default=None)
    p.add_argument('--nn-config',default=None,help='Optional reviewed JSON schedule for full NN4MGF runs')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--method',help=argparse.SUPPRESS)
    p.add_argument('--dimension',type=int,help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.steps<1 or args.warmup<0:p.error('steps must be positive and warmup nonnegative')
    if args.worker:
        worker(args);return
    verify_sources()
    dest=Path(args.output or ('benchmark_runs/'+time.strftime('%Y%m%d_%H%M%S'))).resolve()
    dest.mkdir(parents=True,exist_ok=False)
    reports=[]
    for d in args.dimensions:
        for seed in args.seeds:
            for method in args.methods:
                if method=='nn4mgf' and seed != args.seeds[0]: continue
                label=f'{method}_d{d}_seed{seed}' if method=='hcn' else f'{method}_d{d}_unseeded'
                out=dest/label
                env=os.environ.copy();env['MPLBACKEND']='Agg'
                if args.device=='cpu':env['CUDA_VISIBLE_DEVICES']=''
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--method',method,
                     '--dimension',str(d),'--seed',str(seed),'--mode',args.mode,'--device',args.device,
                     '--steps',str(args.steps),'--warmup',str(args.warmup),'--cpu-threads',str(args.cpu_threads),
                     '--output',str(out)]
                if args.nn_config and method=='nn4mgf':cmd.extend(['--nn-config',str(Path(args.nn_config).resolve())])
                print('Running',method,d,seed if method=='hcn' else 'unseeded',flush=True)
                with (dest/f'{label}.log').open('w') as f:
                    result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                report=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else dict(status='failed',method=method,dimension=d,seed=seed if method=='hcn' else None,error='Worker terminated; inspect log',returncode=result.returncode)
                reports.append(report);dump(dest/'summary.json',reports)
                print(report['status'],out,flush=True)
    fields=['method','dimension','seed','mode','status','mean_update_seconds','median_update_seconds',
            'measured_update_seconds','training_call_seconds','peak_allocated_bytes','peak_reserved_bytes',
            'base_inputs_per_second','total_sampled_base_inputs']
    with (dest/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(reports)
    print('Results:',dest)
    if any(r['status']!='completed' for r in reports):sys.exit(1)

if __name__=='__main__':main()
