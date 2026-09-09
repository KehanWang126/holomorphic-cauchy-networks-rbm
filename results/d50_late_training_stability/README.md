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

For Degree 8, MRE over t=1:8 ranges from 5.312% to 8.116%; over t=1:10 it
ranges from 6.318% to 12.682%. The tail interval has greater variation; these
results should not be described as identical accuracy across checkpoints.
Raw invalid predictions remain included. Degree 5 is retained as a diagnostic.

Files include pointwise predictions, per-checkpoint summary, exact-transform
Talbot reference values, checkpoint hashes, and a file-integrity manifest.
The main notebook evaluator intentionally enforces the fixed 100k endpoint;
it is not a general evaluator for these supplementary earlier checkpoints.
