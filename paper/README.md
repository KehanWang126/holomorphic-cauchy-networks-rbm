# Manuscript data and clipping convention

The tables in `data/` are the canonical reported results. For every finite real tail estimate, reporting uses `min(1, max(0, raw_prediction))`; relative errors and MREs are recomputed from this value at full source precision. Raw estimates, out-of-bounds flags, and both raw and clipped errors remain in `data/pointwise_clipped.csv`. Nonfinite predictions stay invalid. Endpoints, seeds, reference values, and inversion degrees are not selected after evaluation.

Rebuild and verify the tables from the bundled, immutable source CSVs using Python 3 standard library:

```bash
python paper/audit/audit_results.py
python scripts/build_paper_summaries.py
```

The audit checks SHA-256 identities in `audit/selection_manifest.json` and produces `audit/results/`. The second command verifies the published copies in `data/` match these rebuilt files. `data/per_run_summary.csv` and `data/aggregate_summary.csv` contain both raw and clipped MRE and raw invalid counts. `data/signed_error_components_raw.csv` and `data/signed_error_components_clipped.csv` expose both conventions. `audit/inputs/` are original tabular source records; the full checkpoint and training provenance remains under `../results/` and `../experiments/`.

The historical `../results/` CSV summaries and plots still reflect the original raw convention and remain hash-verified as source evidence. The old activation `paired_success_summary.csv` and `failure_summary.csv` represent a retired conditional two-seed analysis; manuscript results include all three 20D Cauchy and tanh seeds. The 10D non-product seed-12345 simulation reference is an unverified pilot, and the H100 profile extrapolation is not a completed full-schedule training run.
