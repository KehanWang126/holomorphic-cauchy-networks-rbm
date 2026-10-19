# 50D late-training stability (90k–100k)

This supplementary diagnostic uses all archived checkpoints at 90k, 92k, 94k,
96k, 98k, and 100k from the same seed-12345 training run. The main result remains
the fixed 100k endpoint. This is a within-run trajectory analysis.

The five earlier weights are stored here. The 100k weight is shared with
[`../d50_skew_symmetric_product_form/checkpoint_step100000.pt`](../d50_skew_symmetric_product_form/checkpoint_step100000.pt).
Paths in `source_checkpoint_hashes.csv` are relative to this directory.

The CSV values are preserved from the supplied historical diagnostic archive
`d50_tail_diagnostic_80k_to_100k_20260813_144549.zip`, restricted to the complete
90k–100k interval. No model was retrained. SHA-256 identities were checked against
the supplied source weights. Local machine paths have been replaced by relative
repository paths. MRE columns in the CSVs are fractions unless marked percent.

For Degree 8, clipped MRE over t=1:8 ranges from 5.299% to 8.104% across the six checkpoints; the 100k fixed endpoint has 6.367%. These checkpoints are one training trajectory, not independent seeds, and are not selected by tail accuracy. The CSVs here preserve historical raw diagnostics; canonical clipped trajectory errors and raw invalid counts are in [paper/data/](../../paper/data/). Degree 5 remains diagnostic.

Files include pointwise predictions, per-checkpoint summary, exact-transform
Talbot reference values, checkpoint hashes, and a file-integrity manifest.
The main notebook evaluator intentionally enforces the fixed 100k endpoint;
it is not a general evaluator for these supplementary earlier checkpoints.
