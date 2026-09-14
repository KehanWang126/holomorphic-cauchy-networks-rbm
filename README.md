# Holomorphic Cauchy Networks for Reflected Brownian Motion


This repository contains the reproducibility code and audited fixed-checkpoint
artifacts for Holomorphic Cauchy Networks (HCNs) applied to stationary tail
estimation for reflected Brownian motion (RBM).


## Audited release scope


The current release contains six experiments:


1. a 2D skew-symmetric product-form benchmark;
2. a 2D Harrison non-product benchmark;
3. the 20D skew-symmetric product-form HCN main experiment;
4. a 20D fixed-budget Cauchy-versus-complex-tanh activation ablation; and
5. the 30D skew-symmetric product-form HCN main experiment; and
6. the 50D skew-symmetric product-form HCN main experiment.


Every reported model is evaluated at a prespecified endpoint. No ground-truth
tail probability is used for training, checkpoint selection, or seed selection.
Predictions are reported without clipping, and invalid or nonfinite outcomes are
recorded rather than silently discarded.


## Repository layout


```text
notebooks/
  d2_skew_symmetric_product_form.ipynb
  d2_harrison_nonproduct_global_cauchy.ipynb
  d20_skew_symmetric_product_form.ipynb
  d30_skew_symmetric_product_form.ipynb
  d50_skew_symmetric_product_form.ipynb
experiments/
  d20_activation_ablation/
    d20_activation_ablation_evaluate.ipynb
    checkpoints/
    reference/
results/
  d2_skew_symmetric_product_form/
  d2_harrison_nonproduct_global_cauchy/
  d20_skew_symmetric_product_form/
  d20_activation_ablation/
  d30_skew_symmetric_product_form/
  d50_skew_symmetric_product_form/
  d50_late_training_stability/
docs/
  reproducibility.md
  validation_checklist.md
```


The training notebooks are deliberately self-contained so that a reviewer can
inspect the benchmark, architecture, sampler, BAR objective, optimizer protocol,
and numerical inversion without following hidden project imports. The 20D
activation ablation is a separate evaluation-only package containing the six
frozen endpoint weights required for that comparison.


## Environment


Create the reference environment with Conda:


```bash
conda env create -f environment.yml
conda activate hcn-rbm
jupyter lab
```


The audited training environment used Python 3.10.20, PyTorch 2.5.1 with CUDA
12.1, and an NVIDIA GeForce RTX 4080 Laptop GPU. The 20D frozen-weight evaluator
also passed on CPU with PyTorch 2.5.1 and mpmath 1.3.0.


## Running the notebooks


The 2D, 20D, 30D, and 50D main notebooks default to a safe smoke test. In their
configuration cell, choose the documented mode:


- `smoke`: structural checks and two tiny optimizer steps;
- `train` (2D): the fixed 15,000-step protocol;
- `train_one` (20D): one explicitly selected paper seed;
- `train_all` (20D): all three paper seeds sequentially;
- `train` (30D/50D): the fixed 50,000- or 100,000-step protocol;
- `evaluate`: evaluate a completed run at the fixed endpoint.


The 30D release result uses the archived historical step-50,000 checkpoint. Its
evaluator accepts that audited legacy schema, verifies the published SHA-256,
and always writes recomputed results to a new output directory.


The 50D release result uses the archived historical step-100,000 checkpoint. Its
evaluator accepts that legacy schema only after verifying the frozen SHA-256 and
writes recomputed evaluations to a new output directory.


The 20D activation ablation never trains. From the repository root, run:


```bash
cd experiments/d20_activation_ablation
jupyter lab d20_activation_ablation_evaluate.ipynb
```


Then select **Restart Kernel and Run All**. Its audited default is CPU and it
evaluates all six bundled step-25,000 checkpoints.


The same fixed-weight checks can be run noninteractively from the repository root:


```bash
python tests/test_d20_activation_ablation.py
```


## Audited 2D results


The fixed step-15,000 Degree-5 results are:


| Benchmark and range | HCN MRE | NN4MGF reference MRE |
|---|---:|---:|
| 2D skew-symmetric, 1:5 | 2.147% | 2.448% |
| 2D skew-symmetric, 1:8 | 2.213% | 8.050% |
| 2D skew-symmetric, 1:10 | 9.737% | 13.915% |
| 2D Harrison, 1:5 | 1.422% | 6.684% |

The curated artifacts and exact endpoint hashes are under the corresponding
`results/d2_*` directories.

## Audited 20D main results

The 20D main result uses three prespecified seeds (`12345`, `22345`, `32345`),
the fixed step-25,000 endpoint, and Talbot Degree 8 as the primary evaluation.
The table reports the arithmetic mean MRE and sample standard deviation across
all three seeds in percentage points.

| Evaluation range | HCN MRE, mean ± sample SD | Archived NN4MGF MRE |
|---|---:|---:|
| 1:5 | 1.455% ± 1.306% | 2.802% |
| 1:8 | 7.641% ± 1.568% | 8.148% |
| 1:10 | 12.271% ± 3.690% | 18.908% |

Degree 5 is retained only as a numerical diagnostic and historical comparison
bridge. The main-result directory contains all 60 recomputed HCN tail-probability evaluation records
(3 seeds × 2 inversion degrees × 10 thresholds),
references to the three shared endpoint weights, aggregate tables, exact-transform
diagnostics, and a
SHA-256 manifest.

## Audited 20D activation ablation

The ablation compares Cauchy and complex-tanh networks with the same width,
training schedule, sample budget, seeds, and fixed endpoint. It is not an exact
parameter-count match: Cauchy has 16,388 trainable real scalar degrees of freedom
and tanh has 15,108.

The Degree-8 `1:8` Cauchy result is **7.641% ± 1.568%**. The tanh aggregate is
dominated by a catastrophic failure of seed `22345`; that seed and every extreme
prediction remain included. For the manuscript activation comparison, usability is reported separately:
Cauchy succeeds in 3/3 recorded runs and tanh in 2/3. The conditional accuracy
comparison uses the same two successful seed labels, 12345 and 32345:

| Activation | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|
| Cauchy | 1.933 | 7.465 | 10.308 |
| Complex tanh | 10.045 | 24.138 | 49.136 |

These values are in `paired_success_summary.csv`; `failure_summary.csv` records
the counts. The existing `multiseed_summary.csv` continues to include all three
seeds. Run `python scripts/build_paper_summaries.py` to rebuild the additional
paper summaries from the recorded predictions.

The six checkpoint SHA-256 hashes, 120 recomputed evaluation points, Talbot
diagnostics, and a 22-test CPU validation report are committed under
`experiments/d20_activation_ablation/` and `results/d20_activation_ablation/`;
the executable test suite is under `tests/`.

## Audited 30D main result

The 30D main result uses seed `12345` and the fixed step-50,000 endpoint.
The first 40,000 updates use learning rate `5e-5`; the final 10,000 use
`1e-5`. Batches `1024/128/128` give a total budget of 64 million sampled
base inputs.

Degree 5 provides the same-degree comparison with the reported NN4MGF
benchmark results. HCN has lower MRE on all three reporting ranges.

| Method | Talbot degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| NN4MGF (reported benchmark) | 5 | 8.353 | 10.476 | 21.507 |
| HCN (comparison) | 5 | 3.691 | 10.270 | 19.215 |

Degree 8 is our primary evaluation rule. With the same step-50,000 weights,
HCN gives:

| Method | Talbot degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| HCN (primary) | 8 | 1.685 | 6.547 | 8.873 |

## Audited 50D main result

The 50D experiment runs on a single NVIDIA GeForce RTX 4080 Laptop GPU,
using seed `12345` and the fixed step-100,000 endpoint. The first 80,000
updates use learning rate `5e-5`; the final 20,000 use `1e-5`. Batches
`512/64/64` maintain the 64-million-base-input budget. Talbot Degree 8
is the primary evaluation rule.

| Method | Talbot degree | 1:5 MRE (%) | 1:8 MRE (%) | 1:10 MRE (%) |
|---|---:|---:|---:|---:|
| HCN (primary) | 8 | 1.504 | 6.373 | 8.015 |

The model uses 18,308 trainable real scalar parameters. Its interval MREs
are lower than the corresponding 30D Degree-8 results, maintaining accuracy
as this benchmark is extended from 30 to 50 dimensions.

## Checkpoint inventory

The repository stores each primary or ablation endpoint weight once. It contains 10 unique endpoint checkpoints: two 2D endpoints, six 20D Cauchy/complex-tanh endpoints, one 30D endpoint, and one 50D endpoint. The 20D main-result manifest references the shared Cauchy weights under `experiments/d20_activation_ablation/checkpoints/` instead of duplicating them.

For the 50D within-run late-training stability diagnostic, `results/d50_late_training_stability/` adds checkpoints at steps 90k, 92k, 94k, 96k, and 98k. The step-100k endpoint is reused from `results/d50_skew_symmetric_product_form/checkpoint_step100000.pt`. Thus the repository contains 15 checkpoint files in total while the late-trajectory tables cover steps 90k through 100k. These six checkpoints document within-run late-training behavior; they are not independent seeds.

## Baseline provenance

The 2D published NN4MGF comparison values are traced to commit
`1956666902281f6ce4558ca82d8d15fa038f06fb` of
[`zhangz73/NN4MGF`](https://github.com/zhangz73/NN4MGF). The 20D NN4MGF files are
archived released-checkpoint predictions with their provenance recorded locally;
the NN4MGF checkpoint itself was not supplied and is not claimed to have been
re-evaluated by this repository.

## Reproducibility and release status

See [docs/reproducibility.md](docs/reproducibility.md) for the staged verification
protocol and [docs/validation_checklist.md](docs/validation_checklist.md) for the
release gates.

The 2D, 20D, 30D, and 50D entries listed above are the audited public subset.
Every archived checkpoint is released with a fixed SHA-256 and a dependency-light
integrity test.

## Manuscript diagnostics

Run `python scripts/build_signed_diagnostics.py` to reconstruct all 80 signed
error decompositions in `results/diagnostics/d20_signed_errors.csv`. This joins
recorded exact-transform, HCN, and archived NN4MGF predictions and checks the
additive identity before rounding. It does not train or run checkpoint inference.

NN4MGF denotes the method of Dai and Zhang (2026), following the name of their
public code repository. Exact-transform inversion error measures the error from
applying Talbot to the exact transform; it is not a lower bound on the final
learned-transform tail error. Legacy internal fields containing `floor` retain
their names for compatibility.

The 30D/50D implementations backpropagate the loss branches sequentially,
accumulating gradients before a single optimizer update. Together with smaller
batches, this controls simultaneous graph storage while additional updates
maintain the sampling budget. No same-hardware speedup ratio is inferred.

The 50D session-clock reconstruction is documented in
`results/d50_skew_symmetric_product_form/training_runtime.json` and
`runtime_segments.csv`. The last session's 16.98 hours is not a cumulative clock.
