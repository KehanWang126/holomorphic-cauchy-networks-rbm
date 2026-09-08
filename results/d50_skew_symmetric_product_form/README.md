# 50D skew-symmetric product-form result

This directory contains the single prespecified seed-12345 checkpoint at global
optimizer step 100,000. The run starts from a fresh random initialization, uses
learning rate `5e-5` for steps 1--80,000 and `1e-5` for steps 80,001--100,000,
and never uses a tail probability to select a checkpoint.

## Primary result

Talbot Degree 8 is primary. Degree 5 is retained only as a numerical inversion
diagnostic. Predictions are raw and are not clipped.

| Range | HCN MRE | Exact-transform Talbot floor |
|---|---:|---:|
| `t=1:5` | 1.504% | 0.596% |
| `t=1:8` | 6.373% | 0.469% |
| `t=1:10` | 8.015% | 0.608% |

The Degree-8 prediction at `t=2` is `1.000477639318975`; this slight overshoot
is retained and counted as one invalid raw tail. No failed point is removed from
the MRE.

The released checkpoint has SHA-256
`669c9bc1b2188241e47e78ee5a23bfaa690a86233529b8eef340daf11cf79b0a`.
It is the exact step-100,000 file read by the independent 80k-to-100k diagnostic.
The separately exported final checkpoint has identical contents in all 77 tensor
storage blocks but different serialization metadata, so it is recorded only in
`source_provenance.json` and is not duplicated here.

The release assembler did not rerun model inference. It curated the stored
fixed-checkpoint diagnostic, verified the checkpoint hash against the diagnostic
record, and preserved the original raw predictions. The public notebook can
recompute the evaluation from the bundled checkpoint.

The training history contains 90 scheduled records: every 1,000 steps through
80,000 and every 2,000 steps thereafter. Its final cumulative elapsed time is
recorded in `training_runtime.json`.
