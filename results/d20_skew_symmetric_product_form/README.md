# 20D skew-symmetric product-form main results

This directory contains the audited HCN main result at the prespecified
step-25,000 endpoint for seeds 12345, 22345, and 32345.

## Primary Degree-8 result

| Range | Mean MRE (%) | Sample SD (percentage points) | Invalid raw predictions |
|---|---:|---:|---:|
| 1:5 | 1.440570 | 1.321982 | 1 |
| 1:8 | 7.631897 | 1.564846 | 1 |
| 1:10 | 12.263597 | 3.678306 | 1 |

The table uses clipped finite predictions for MRE and retains raw invalid counts across all three seeds. Historical CSV summaries and figures in this directory still use the original raw convention; use [paper/data/](../../paper/data/) for canonical clipped pointwise and aggregate tables. Degree 5 is diagnostic.

## Files

- The three Cauchy endpoint weights are stored once in
  [`../../experiments/d20_activation_ablation/checkpoints/`](../../experiments/d20_activation_ablation/checkpoints/),
  shared by the main experiment and activation ablation;
- `checkpoint_manifest.json`: seed, endpoint, provenance, and SHA-256 identity;
- `pointwise.csv`: 60 recomputed points (3 seeds × 2 degrees × 10 values of `t`);
- `per_seed_summary.csv`: each seed and interval separately;
- `multiseed_summary.csv`: arithmetic mean and sample SD (`ddof=1`);
- `exact_transform_*.csv`: Talbot numerical-inversion diagnostics;
- `nn4mgf_archived_summary.csv`: archived NN4MGF released-checkpoint reference;
- `cauchy_degree5.png`, `cauchy_degree8.png`: raw predictions and errors;
- `training_seed12345/`: the uninterrupted replacement run's configuration,
  training trajectory, runtime, sampler, parameter, RBM, and preflight audits;
- `evaluation_report.json`: evaluation-scope metadata; and
- `manifest.sha256`: hashes of every curated file in this directory.

The seed-12345 endpoint was newly trained without interruption. Seeds 22345 and
32345 retain the previously audited endpoints. No intermediate checkpoint was
evaluated or ranked for this release.
