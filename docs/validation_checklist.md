# Release validation checklist

## Notebook integrity

- [ ] Notebook opens as valid nbformat v4 JSON.
- [ ] A fresh kernel can execute all cells in order.
- [ ] No cell depends on variables from an earlier interactive session.
- [ ] No absolute local path is embedded in code or outputs.
- [ ] The committed notebook contains no execution outputs.

## Frozen experiment identity

- [ ] Benchmark is `d2_skew_symmetric_product_form`.
- [ ] Seed is 12345.
- [ ] Parameter count is 15,236 real scalar DOF.
- [ ] Formal endpoint is exactly 15,000 updates.
- [ ] Total sampled constraints are exactly 64,050,000.
- [ ] Initial learning rate is \(5\times10^{-5}\).
- [ ] Learning rate changes to \(10^{-5}\) after step 12,000.
- [ ] Adam state is preserved across the learning-rate change.

## Scientific safeguards

- [ ] Training never reads exact tails or published predictions.
- [ ] No ground-truth early stopping or checkpoint selection occurs.
- [ ] Only the prespecified step-15,000 model is saved and reported.
- [ ] Primary 2D evaluation uses Talbot Degree 5 only.
- [ ] Predictions are raw and are not clipped to \([0,1]\).
- [ ] Invalid and nonfinite values are reported rather than replaced.
- [ ] Checkpoint SHA256 is unchanged by evaluation.

## Formal-run acceptance

- [ ] Preflight audits pass.
- [ ] Training finishes without NaN, Inf, or pole-proximity failure.
- [ ] Degree-5 invalid count is zero.
- [ ] Fixed-endpoint accuracy is comparable to the audited reference.
- [ ] Output configuration, environment, runtime, pointwise results, summary,
      figure, and checkpoint hash files are present.

