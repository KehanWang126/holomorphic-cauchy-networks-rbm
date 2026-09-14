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


### Paper summary synchronization

The archived all-three-seed aggregates retain every failure and extreme value.
The manuscript additionally reports usable-run counts and Degree-8 conditional
means over common successful seeds 12345 and 32345. Rebuild these separate files
with `python scripts/build_paper_summaries.py` from the repository root.
Rebuild the 80 signed error decompositions with
`python scripts/build_signed_diagnostics.py`. Both scripts use recorded data,
without retraining, selecting a new endpoint, or clipping predictions.
