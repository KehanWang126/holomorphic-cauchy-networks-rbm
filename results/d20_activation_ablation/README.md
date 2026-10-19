# 20D activation-ablation results

These are the outputs of the audited CPU run of the six fixed step-25,000
checkpoints in `../../experiments/d20_activation_ablation/`.

- 2 methods × 3 seeds × 2 Talbot degrees × 10 tail locations = 120 points;
- no training during evaluation;
- raw predictions and old raw summaries retained for checkpoint audit;
- all checkpoint hashes unchanged;
- 22 automated tests passed with no failure or error.

`pointwise.csv` preserves the original raw predictions; `summary.csv` and `multiseed_summary.csv` preserve the old raw errors. The manuscript's canonical clipped summaries are in [paper/data/](../../paper/data/): Degree-8 1:8 MRE is 7.631897% ± 1.564846 pp for Cauchy and 22.105385% ± 9.942438 pp for complex tanh, each over all three seeds. Tanh seed 22345 has an extreme raw prediction, is clipped for reported error, and is **not excluded**. Raw invalid counts remain visible in the paper audit.

Degree 8 is primary. Degree 5 is retained as an inversion diagnostic and
historical comparison bridge. `nn4mgf_archived_summary.csv` is based on archived
predictions, not a model re-evaluated by this repository.

Every curated file is covered by `manifest.sha256`.



The old `paired_success_summary.csv` and `failure_summary.csv` are historical conditional/raw reports. They are not current manuscript tables. Regenerate the canonical clipped results with `python paper/audit/audit_results.py` from the repository root.
