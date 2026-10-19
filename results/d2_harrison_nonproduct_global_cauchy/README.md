# 2D Harrison non-product reference run

This directory contains the curated artifacts from a clean execution of
[`d2_harrison_nonproduct_global_cauchy.ipynb`](../../notebooks/d2_harrison_nonproduct_global_cauchy.ipynb).
The interior log-transform combines a shared-coordinate Holomorphic Cauchy
branch with a centered full-input Holomorphic Cauchy branch, so it is not
restricted to a product-form transform.

## Frozen protocol

- seed: `12345`;
- optimizer updates: `15,000`;
- sampled constraints: `64,050,000`;
- trainable real scalar degrees of freedom: `15,236`;
- evaluation: Talbot Degree 5 at (t=1,\ldots,5);
- raw predictions preserved; reported finite tails use the paper's [0, 1] clipping convention (no value changes in this run);
- no ground-truth tail values in training or endpoint selection.

## Audited result

| Method | Degree-5 MRE, (t=1{:}5) | Invalid predictions |
|---|---:|---:|
| HCN, fixed step 15,000 | **1.422%** | 0 |
| Published NN4MGF | 6.684% | 0 |

The full-input interaction residual is nonzero both after training and in the
read-only evaluation (`0.0478711` maximum absolute residual over the fixed
audit points). All 15,236 real scalar degrees of freedom participate in the
forward path, and the coordinate, boundary, and global branches all receive
nonzero gradients.

Training took 2,109.60 seconds on the recorded NVIDIA GeForce RTX 4080 Laptop
GPU environment. The final checkpoint SHA256 is

```text
26a2c69fd797662ec1cb44c0aea200be25942b6dc5ac518d06bf2620051aef54
```

`summary_degree5.json` and `pointwise_degree5.csv` contain the original raw
metrics (identical to clipped MRE here). `nonseparability_audit.json`, `parameter_budget_audit.json`,
`talbot_precision_audit.json`, and `preflight_audit.json` contain the structural
and numerical checks. The recorded evaluation uses mpmath 1.3.0 at 50 decimal
digits and passes complex128/float64 precision auditing. Evaluation leaves the
checkpoint hash unchanged.

No reliable closed-form exact joint transform was available in the audited
Harrison sources, so this release does not fabricate an exact-transform Talbot
floor. Exact Harrison tails are instead computed from the benchmark density
integral after training. Only the prespecified step-15,000 model is included.
