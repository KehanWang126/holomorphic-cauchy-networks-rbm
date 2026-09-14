# Same-server HCN / NN4MGF benchmark

Run both methods sequentially on the same GPU. The runner uses the existing HCN
notebook functions and the unmodified, MIT-licensed NN4MGF training module pinned
in `sources.json`. It measures synchronized update time and peak PyTorch GPU
memory, saves final weights, and evaluates both Talbot Degrees 5 and 8.

## Setup and commands

From the repository root, use the existing `hcn-rbm` environment, then install
the two additional NN4MGF dependencies:

```bash
pip install -r benchmarks/requirements.txt

# Integration check: two updates with reduced batches; not a speed benchmark.
python benchmarks/run.py --mode smoke --device cuda --dimensions 20 30

# Original training batch sizes; 10 warm-up + 50 measured updates per method.
CUDA_VISIBLE_DEVICES=0 python benchmarks/run.py --mode profile --dimensions 20 30

# Full configured runs, one method at a time on the same GPU.
CUDA_VISIBLE_DEVICES=0 python benchmarks/run.py --mode full --dimensions 20 30
```

Use `--seeds 12345 22345 32345` for repeated runs, `--methods hcn` or
`--methods nn4mgf` for one method, and `--output /path/to/new-directory` to choose
an output directory. Every method/dimension/seed runs in a fresh process.
Use an idle GPU and the same environment for both methods. Outputs are never
overwritten. Run long jobs through the server's scheduler or a persistent shell.
CPU integration checks are available with `--mode smoke --device cpu`.

## Configurations and interpretation

| Method | Dimension | Updates in full mode | Base inputs per update | Total base inputs |
|---|---:|---:|---:|---:|
| HCN | 20 | 25,000 | 2,560 | 64,000,000 |
| HCN | 30 | 50,000 | 1,280 | 64,000,000 |
| NN4MGF public-code default | 20/30 | 100,000 | 32,768 | 3,276,800,000 |

HCN retains its notebook optimizer, losses, sampler, learning-rate switch,
and backward implementation. NN4MGF retains its `MGFTrainer.train()` method,
including AdamW, the cosine schedule, CR penalty, two sampled batches per update,
and its diagnostic calculations. Model precision is recorded in each report.
Training curves and intermediate checkpoint I/O from the HCN notebook driver
are not replayed; this harness compares training updates, then saves the final
state and computes tail errors. NN4MGF's own end-of-training plots remain enabled.

**The active public NN4MGF script configures 100,000 updates with `train_freq=2`.**
This is not the 100,000-plus-200,000 schedule reported in the paper. Consequently,
the default full run is a comparison against the pinned *public-code training
configuration*, not a reproduction of the paper's complete training schedule.
Do not label its total time as the original paper's runtime. If the authors
provide the complete schedule, pass a reviewed JSON with `--nn-config` in full
mode; `nn4mgf_public_config.json` shows the supported format. The runner records
all effective settings and the configuration hash. No additional stages are
invented automatically.

Profile mode preserves original batch sizes and scheduler period but stops early;
it measures initial training throughput and memory, not final accuracy or total
training time. Smoke mode reduces batches and is only an integration test.
No automatic batch-size reduction is performed after an out-of-memory error.
Each failed run has its own log/report; the other runs can still complete.

## Outputs

Each run produces `report.json`, `endpoint.pt`, and `tail_predictions.csv`.
The parent directory includes logs and `summary.csv` / `summary.json`.

- `mean_update_seconds`, `median_update_seconds`: synchronized wall time per
  optimizer update, including sampling, losses, backward, optimization and
  per-update logging. Warm-up is excluded only in profile mode.
- `measured_update_seconds`: sum of measured update times, excluding final plots,
  checkpoint saving and tail evaluation. This is the main training-time metric.
- `training_call_seconds`: outer training-call time, also including warm-up and
  any upstream final plots. It is separately labelled, not the primary ratio.
- `peak_allocated_bytes`, `peak_reserved_bytes`: absolute peaks of PyTorch's CUDA
  allocator during the measured window, including model and optimizer state.
  These are not whole-device `nvidia-smi` readings and exclude non-PyTorch memory.
- `base_inputs_per_second`: accounts for each method's actual sampling budget.
- `tail_mre`: per-degree interval MRE. Only full-mode results are final-endpoint
  accuracy measurements; short-run errors are integration diagnostics.

Compare methods within the same dimension, hardware, mode and inversion degree.
Report both time and final MRE for full runs. This runner does not preassign any
speedup or memory-saving factor.

## Provenance and local validation

NN4MGF source: https://github.com/zhangz73/NN4MGF at
`1956666902281f6ce4558ca82d8d15fa038f06fb`. The training module, original experiment
script and MIT license are retained under `vendor/nn4mgf/`.
The HCN notebook hashes are checked before running. Changes to either source
require explicit review and an updated lock.

```bash
python benchmarks/test_runner.py
```

The server GPU measurement remains to be executed; CPU smoke tests verify the
integration, not GPU speed, peak memory, or full-run convergence.
