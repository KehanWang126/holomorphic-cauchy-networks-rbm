# Holomorphic Cauchy Networks for Reflected Brownian Motion

This repository contains the reproducibility code for a fixed-capacity
Holomorphic Cauchy Network (HCN) approach to stationary tail estimation for
reflected Brownian motion (RBM). The project is being prepared for journal
submission and is released incrementally after each experiment passes the same
audit and reproduction checks.

## Current scope

The first audited experiment is the two-dimensional skew-symmetric
product-form benchmark:

- coordinate-shared holomorphic Cauchy architecture;
- 15,236 trainable real scalar degrees of freedom;
- seed 12345;
- prespecified endpoint of 15,000 optimizer updates;
- 64.05 million sampled constraints;
- Talbot Degree 5 evaluation;
- raw predictions with no clipping;
- no ground-truth tails in training or endpoint selection.

The Harrison, 20D, 30D, and 50D experiments will be added only after their
public notebooks and artifacts pass the same checks.

## Repository layout

```text
notebooks/
  d2_skew_symmetric_product_form.ipynb
docs/
  reproducibility.md
  validation_checklist.md
results/
  d2_skew_symmetric_product_form/
```

The notebook is deliberately self-contained so that a reviewer can inspect
the architecture, sampler, BAR objective, training protocol, and numerical
inversion without following hidden imports.

## Quick start

Create the reference environment with Conda:

```bash
conda env create -f environment.yml
conda activate hcn-rbm
jupyter lab
```

Open `notebooks/d2_skew_symmetric_product_form.ipynb` and choose one mode in
the configuration cell:

- `RUN_MODE = "smoke"`: two optimizer steps; this is the safe default.
- `RUN_MODE = "train"`: the fixed 15,000-step paper protocol, followed by
  Degree-5 evaluation.
- `RUN_MODE = "evaluate"`: evaluate a checkpoint produced by this cleaned
  notebook without retraining.

For the formal reproduction, restart the kernel, set `RUN_MODE = "train"`,
and use **Run All**. The notebook creates a new timestamped directory under
`outputs/`; it never overwrites an earlier run.

## Audited reference result

The previously audited production run gives the following fixed-endpoint
Degree-5 mean relative errors (MREs). These values are reproduction targets,
not early-stopping criteria.

| Evaluation range | HCN | Published NN4MGF |
|---|---:|---:|
| \(t=1{:}5\) | 2.147% | 2.448% |
| \(t=1{:}8\) | 2.213% | 8.050% |
| \(t=1{:}10\) | 9.737% | 13.915% |

The audited HCN run contains no invalid raw prediction. The exact checkpoint
hash and pointwise reference values are kept out of the training path and are
used only for post-run auditing.

Fresh training runs on a different PyTorch, CUDA, driver, or GPU stack need
not be bitwise identical. Acceptance is based on the frozen protocol,
successful numerical audits, absence of invalid predictions, and comparable
fixed-endpoint accuracy. Evaluation of an unchanged checkpoint should be
numerically identical in the same software environment.

## Reproducibility and provenance

See [docs/reproducibility.md](docs/reproducibility.md) for the two-stage
verification procedure and [docs/validation_checklist.md](docs/validation_checklist.md)
for the release gate.

The published NN4MGF comparison values are traced to commit
`1956666902281f6ce4558ca82d8d15fa038f06fb` of
[`zhangz73/NN4MGF`](https://github.com/zhangz73/NN4MGF), specifically
`Plots/d=2/skewed_symmetry/tables.txt`.

## Status

Work in progress. Do not treat intermediate repository revisions as the final
paper artifact.

