# 20D activation-ablation results

These are the outputs of the audited CPU run of the six fixed step-25,000
checkpoints in `../../experiments/d20_activation_ablation/`.

- 2 methods × 3 seeds × 2 Talbot degrees × 10 tail locations = 120 points;
- no training during evaluation;
- no prediction clipping or seed filtering;
- all checkpoint hashes unchanged;
- 22 automated tests passed with no failure or error.

`pointwise.csv` is the authoritative record. `summary.csv` reports each
method/seed separately, and `multiseed_summary.csv` reports the arithmetic mean
and sample standard deviation (`ddof=1`) over all three seeds. The extremely
large Degree-8 tanh aggregate is caused by the retained seed-22345 failure and is
not a missing-value or plotting artifact.

Degree 8 is primary. Degree 5 is retained as an inversion diagnostic and
historical comparison bridge. `nn4mgf_archived_summary.csv` is based on archived
predictions, not a model re-evaluated by this repository.

Every curated file is covered by `manifest.sha256`.
