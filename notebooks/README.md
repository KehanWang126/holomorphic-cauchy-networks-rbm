# Public experiment notebooks

The current release contains exactly one public notebook for each training
experiment:

- `d2_skew_symmetric_product_form.ipynb`: 2D product-form benchmark, seed 12345,
  fixed step 15,000, and Talbot Degree 5 evaluation.
- `d2_harrison_nonproduct_global_cauchy.ipynb`: 2D Harrison non-product
  benchmark with a centered full-input HCN branch, seed 12345, fixed step 15,000,
  and Talbot Degree 5 evaluation.
- `d20_skew_symmetric_product_form.ipynb`: 20D coordinate-shared HCN, three
  prespecified seeds, fixed step 25,000, and Degree-5/8 Talbot diagnostics with
  Degree 8 designated as primary.
- `d30_skew_symmetric_product_form.ipynb`: 30D coordinate-shared HCN, seed
  12345, fixed step 50,000, matched 64M sampled-constraint budget, recovery-only
  intermediate checkpoints, and Degree-5/6/8 diagnostics with Degree 8 primary.
  Its evaluator also accepts the hash-verified historical point-matched 50k
  checkpoint committed with the audited result.
- `d50_skew_symmetric_product_form.ipynb`: 50D coordinate-shared HCN, seed
  12345, fixed step 100,000, matched 64M sampled-constraint budget, recovery-only
  intermediate checkpoints, and Degree-5/8 diagnostics with Degree 8 primary.
  Its evaluator accepts only the hash-verified historical step-100,000 endpoint
  committed with the audited result bundle.

The 20D filename above is the sole retained main-experiment version. Earlier
duplicate drafts and the draft containing the invalid `PARAMS.d` audit access are
not part of the repository.

The activation ablation is intentionally separate from the main training
notebook. Its evaluation-only notebook and six frozen checkpoints are in
`../experiments/d20_activation_ablation/`.

Jupyter autosave files under `.ipynb_checkpoints/` are excluded. Committed
notebooks contain no execution output.
