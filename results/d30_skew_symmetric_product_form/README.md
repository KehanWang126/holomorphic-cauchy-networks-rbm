# 30D skew-symmetric product-form result

This directory contains the single prespecified seed-12345 checkpoint at global
optimizer step 50,000. It is the historical point-matched run that continues the
same Adam and RNG state from step 20,000. Intermediate checkpoints and the later
50k-to-60k exploratory branch are not release endpoints.

## Primary result

Talbot Degree 8 is primary. The table reports MRE after clipping finite predictions to [0, 1]; the files in this directory preserve the historical raw evaluations. Full canonical tables and raw invalid counts are in [paper/data/](../../paper/data/).

| Range | HCN MRE | Exact-transform Talbot floor |
|---|---:|---:|
| `t=1:5` | 1.683% | 0.152% |
| `t=1:8` | 6.546% | 0.178% |
| `t=1:10` | 8.872% | 0.375% |

The checkpoint has SHA-256 `0bfcd0dc11779fabd477fcf5ee9bf07772e555a9b10776618dc4e47a1106be1c`. The stored endpoint predictions
were cross-checked exactly between the historical training record and an
independent checkpoint-diagnostic CSV. The release assembler did not rerun the
model; the public notebook can recompute the evaluation from the bundled weight.

The `elapsed_seconds` column in `training_history.csv` is inherited from the
historical staged run and should not be interpreted as one uninterrupted 50k
wall-clock measurement.
