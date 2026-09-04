# 2D skew-symmetric product-form reference run

This directory contains the curated artifacts from a clean execution of
[`d2_skew_symmetric_product_form.ipynb`](../../notebooks/d2_skew_symmetric_product_form.ipynb).
It is the fixed-endpoint reference run used for the two-dimensional
skew-symmetric product-form comparison.

## Frozen protocol

- seed: `12345`;
- optimizer updates: `15,000`;
- sampled constraints: `64,050,000`;
- trainable real scalar degrees of freedom: `15,236`;
- evaluation: Talbot Degree 5;
- raw predictions, without clipping;
- no ground-truth tail values in training or endpoint selection.

## Audited result

| Evaluation range | HCN MRE | Published NN4MGF MRE |
|---|---:|---:|
| \(t=1{:}5\) | 2.147% | 2.448% |
| \(t=1{:}8\) | 2.213% | 8.050% |
| \(t=1{:}10\) | 9.737% | 13.915% |

There are no invalid raw predictions. Training took 1,738.53 seconds on the
recorded NVIDIA GeForce RTX 4080 Laptop GPU environment. The final checkpoint
SHA256 is

```text
3010b9805adc5b9fd1eb8bc1b1efb39e19ab4043fb755c628c206ff82cefe928
```

`summary_degree5.json` and `pointwise_degree5.csv` contain the reported
metrics. `run_config.json`, `sample_budget.json`, `preflight_audit.json`, and
`environment.json` record the production protocol and numerical checks.
`manifest.sha256` covers every committed artifact in this directory except
the manifest itself.

Only the prespecified step-15,000 model is included. Exploratory checkpoints,
checkpoint-accuracy trajectories, and Degree-8 diagnostics are intentionally
excluded from this two-dimensional benchmark release.
