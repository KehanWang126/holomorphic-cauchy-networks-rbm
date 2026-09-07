# Release validation checklist

## Repository-wide integrity

- [ ] Every notebook opens as valid nbformat v4 JSON.
- [ ] Committed notebooks contain no execution outputs.
- [ ] No notebook depends on state from an earlier interactive session.
- [ ] No absolute local user path is embedded in committed code or metadata.
- [ ] Every referenced relative path exists.
- [ ] Every curated artifact matches its `manifest.sha256` entry.
- [ ] Raw timestamped run directories and Jupyter autosave files are excluded.

## 2D release gate

- [ ] Seed is fixed at 12345.
- [ ] Endpoint is fixed at step 15,000.
- [ ] Parameter count is 15,236 real scalar degrees of freedom.
- [ ] Total sampled constraints are 64,050,000.
- [ ] Adam state is preserved across the learning-rate change after step 12,000.
- [ ] Talbot Degree 5 is the primary 2D protocol.
- [ ] Harrison passes its full-input nonseparability audit.
- [ ] Ground truth is absent from training and checkpoint selection.
- [ ] Raw predictions are not clipped.
- [ ] Checkpoint SHA-256 is unchanged by evaluation.

## 20D main-experiment gate

- [ ] Only `notebooks/d20_skew_symmetric_product_form.ipynb` is released.
- [ ] The corrected dimension audit uses `PARAMS.mu.numel()` rather than the
      nonexistent `PARAMS.d` attribute.
- [ ] Seeds are exactly 12345, 22345, and 32345.
- [ ] Endpoint is fixed at step 25,000 with no intermediate evaluation.
- [ ] Parameter count is 16,388 real scalar degrees of freedom.
- [ ] Each seed uses exactly 64,000,000 sampled constraints.
- [ ] Adam state is preserved across the learning-rate change after step 20,000.
- [ ] Degree 8 is primary and Degree 5 is labeled diagnostic.
- [ ] All 60 HCN evaluation points are present.
- [ ] Three-seed aggregation uses sample SD with `ddof=1`.
- [ ] Invalid predictions are retained and counted.
- [ ] The three endpoint hashes match `checkpoint_manifest.json`.

## 20D activation-ablation gate

- [ ] The evaluator contains no training mode or optimizer step.
- [ ] Exactly six method/seed checkpoint pairs are present.
- [ ] All six endpoint steps are 25,000 and hashes match the manifest.
- [ ] The comparison is described as same-width and same-budget, not exactly
      parameter-matched.
- [ ] Cauchy is evaluated from weights with PyTorch complex128.
- [ ] Tanh is evaluated from weights with the audited NumPy complex128 forward
      map and does not call the crashing PyTorch complex-tanh kernel.
- [ ] All 120 evaluation points are present.
- [ ] No seed or failed point is excluded from aggregation.
- [ ] Nonfinite errors make an interval undefined rather than being removed.
- [ ] The NN4MGF reference is labeled archived and not re-evaluated.
- [ ] All 22 automated checks pass.

## 30D main-experiment gate

- [ ] Only `notebooks/d30_skew_symmetric_product_form.ipynb` is released.
- [ ] Seed is fixed at 12345 and the endpoint is fixed at step 50,000.
- [ ] Parameter count is 17,028 trainable real scalar degrees of freedom.
- [ ] The 1024/128/128 batches and 50,000 updates total 64,000,000 constraints.
- [ ] Adam state is preserved across the learning-rate change after step 40,000.
- [ ] Sequential branch backward is used without changing the summed gradient.
- [ ] Intermediate files are recovery-only and never selected by tail accuracy.
- [ ] Degree 8 is primary; Degrees 5 and 6 are diagnostics.
- [ ] Ground truth and the published table appear only after training code.
- [ ] Raw invalid or nonfinite predictions are retained and counted.
- [ ] The historical schema is `d30_general_pointmatched50k_v1` and the fixed
      checkpoint SHA-256 is verified before any legacy pickle-capable load.
- [ ] The stored fixed-endpoint predictions agree between training history and
      the independent diagnostic table for every shared point.
- [ ] The 50k-to-60k branch and retrospective 29k/49k minima are excluded.
- [ ] The final checkpoint hash is unchanged by evaluation.
- [ ] `python tests/test_d30_notebook.py` passes.

## Formal-run acceptance

- [ ] Preflight audits pass before training or evaluation.
- [ ] Training, where requested, completes without NaN, Inf, or an unhandled
      pole-proximity failure.
- [ ] Evaluation produces configuration, environment, runtime, pointwise,
      summary, figure, and hash records.
- [ ] Results are comparable at the prespecified endpoint without selecting a
      more favorable intermediate step, seed, or Talbot degree.
