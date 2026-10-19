# Holomorphic Cauchy Networks for Reflected Brownian Motion

Code, trained checkpoints, and experimental results for Holomorphic Cauchy
Networks (HCN), a complex neural approach to stationary tail computation for
reflected Brownian motion (RBM). Building on the BAR learning framework of
Dai and Zhang (2026), HCN uses holomorphic Cauchy activations and a compact
shared architecture. The experiments extend to 50 dimensions on a single
NVIDIA GeForce RTX 4080 Laptop GPU, with a high-dimensional sampling budget
of 64 million base inputs per run.

## Code and data

| Location | Contents |
|---|---|
| [notebooks/](notebooks/) | Training and evaluation notebooks for 2D, 20D, 30D, and 50D |
| [experiments/d20_activation_ablation/](experiments/d20_activation_ablation/) | Cauchy versus complex-tanh evaluation and six fixed checkpoints |
| [paper/](paper/) | Canonical manuscript tables, clipped predictions, and reproducible source-data audit |
| [results/](results/) | Archived raw predictions, checkpoint hashes, and training records |
| [scripts/](scripts/) | Rebuild and verification entry points; historical raw diagnostics remain archived |
| [tests/](tests/) | Checkpoint and result verification |
| [docs/](docs/) | Detailed reproduction instructions |

## Getting started

Create the reference environment and open Jupyter:

```bash
conda env create -f environment.yml
conda activate hcn-rbm
jupyter lab
```

The reference training environment uses Python 3.10.20 and PyTorch 2.5.1
with CUDA 12.1. The 20D fixed-checkpoint evaluator also runs on CPU.

Open the relevant notebook in `notebooks/` and select its run mode.
Main notebooks default to `smoke` for a short check. Use `train` for 2D,
30D, or 50D; `train_one` or `train_all` for 20D; and `evaluate`
to evaluate a completed run. The fixed training endpoints are 15,000
updates for 2D, 25,000 for 20D, 50,000 for 30D, and 100,000 for 50D.

For the activation comparison, open
[the evaluation notebook](experiments/d20_activation_ablation/d20_activation_ablation_evaluate.ipynb)
and select **Restart Kernel and Run All** to evaluate the six supplied
20D checkpoints. Detailed settings are in
[the reproduction guide](docs/reproducibility.md).

## Main results

MRE denotes mean relative error. Reported finite tail predictions are projected to [0, 1] before computing MRE; nonfinite outputs remain invalid. Raw predictions and raw invalid counts are preserved in [paper/data/](paper/data/) and are never used for checkpoint or seed selection. Ranges 1:5, 1:8, and 1:10 use integer
thresholds starting at 1. Degree 8 is the primary evaluation rule from
20D onward; Degree 5 is used for the 2D results and baseline comparisons.

### 2D

Both benchmarks use the fixed 15,000-update endpoint and Talbot Degree 5.

| Benchmark and range | HCN MRE (%) | NN4MGF MRE (%) |
|---|---:|---:|
| Skew-symmetric, 1:5 | 2.147 | 2.448 |
| Skew-symmetric, 1:8 | 2.213 | 8.050 |
| Skew-symmetric, 1:10 | 9.737 | 13.915 |
| Harrison, 1:5 | 1.422 | 6.684 |

### 20D

Degree-8 results at 25,000 updates. HCN entries are the mean MRE and sample
standard deviation across seeds `12345`, `22345`, and `32345`, in percent.

| Range | HCN MRE, mean ± SD | Archived NN4MGF MRE |
|---|---:|---:|
| 1:5 | 1.441% ± 1.322% | 2.802% |
| 1:8 | 7.632% ± 1.565% | 8.148% |
| 1:10 | 12.264% ± 3.678% | 18.908% |

The [20D result directory](results/d20_skew_symmetric_product_form/)
contains 60 HCN tail-probability evaluation records:
3 seeds × 2 inversion degrees × 10 thresholds.

### Cauchy versus complex tanh at 20D

Both activations use the same hidden widths, seed labels, 25,000-update
endpoint, and 64M-input budget. The Degree-8 clipped MREs below include all three prespecified seeds for both activations (`12345`, `22345`, `32345`). The raw predictions, including out-of-range values, remain available in the archived pointwise file and the paper audit.

| Activation | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|
| Cauchy | 1.441 | 7.632 | 12.264 |
| Complex tanh | 8.604 | 22.105 | 37.036 |

[Activation results](results/d20_activation_ablation/) retain the original raw evaluation. Canonical three-seed clipped summaries, individual predictions, and raw invalid counts are in [paper/data/](paper/data/).

### 30D

The fixed 50,000-update model uses seed `12345` and a 64M-input budget.
Under Degree 5, HCN has lower MRE than NN4MGF on all three ranges.

| Method | Degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| NN4MGF (reported benchmark) | 5 | 8.353 | 10.476 | 21.507 |
| HCN (comparison) | 5 | 3.429 | 10.107 | 19.084 |

The same weights evaluated with the primary Degree-8 rule give:

| Method | Degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| HCN (primary) | 8 | 1.683 | 6.546 | 8.872 |

### 50D

The 50D model runs on a single laptop GPU with 18,308 trainable real scalar
parameters. Results use seed `12345`, the fixed 100,000-update endpoint,
and the same 64M-input budget.

| Method | Degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| HCN (primary) | 8 | 1.495 | 6.367 | 8.011 |

Interval MREs remain below the corresponding 30D Degree-8 values as this
benchmark is extended to 50 dimensions.

### 10D non-product pilot

One fixed seed (`12345`) at step 25,000 gives Degree-8 clipped MRE 1.710% on 1:5 and 3.684% on 1:8, versus 19.409% for the independent-marginal comparator on 1:8. The simulation reference is an **unverified pilot**, not an exact joint-tail ground truth; this exploratory result does not establish a multi-seed performance claim.

## Trained checkpoints

| Experiment | Endpoint checkpoints | Update |
|---|---:|---:|
| 2D: two benchmarks | 2 | 15,000 |
| 20D: Cauchy and complex tanh | 6 | 25,000 |
| 30D | 1 | 50,000 |
| 50D | 1 | 100,000 |

Five additional 50D checkpoints support
[late-training analysis](results/d50_late_training_stability/).
Together with the 100,000-update endpoint, they cover six checkpoints
from the same run at 90,000–100,000 updates.

## Baseline source

NN4MGF denotes the method of Dai and Zhang (2026), following the name of
their [public code repository](https://github.com/zhangz73/NN4MGF).
The 2D and 30D comparisons use reported benchmark results; the 20D
comparisons use archived released-checkpoint evaluations.
Source details accompany the corresponding result files.

## Reproducing summaries

Run these commands from the repository root:

```bash
# Recompute and check the six fixed 20D checkpoints
python tests/test_d20_activation_ablation.py

# Verify source hashes and regenerate the canonical clipped tables (Python 3 stdlib)
python paper/audit/audit_results.py

# Check that the published tables agree with the regenerated audit
python scripts/build_paper_summaries.py
```

See [reproduction instructions](docs/reproducibility.md) and
[validation checks](docs/validation_checklist.md) for additional details.

## Server time and memory comparison

Use [benchmarks/](benchmarks/) to run HCN and pinned NN4MGF code on the same
GPU at 20D and 30D. It provides smoke, profiling, and full-configuration modes.
See the [benchmark instructions](benchmarks/README.md) for commands and interpretation.
HCN uses seed 12345; NN4MGF runs without an explicit seed for 300,000 updates,
retaining the public code's two batches per update. The measured 30D H100 profile gave 0.165 s/update and 7.805 GiB allocated for HCN versus 1.601 s/update and 77.815 GiB for NN4MGF (9.7× initial-update throughput). A 58.2× full-schedule ratio is an extrapolation, not completed full-run timing.
