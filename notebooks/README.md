# Public experiment notebooks

Each notebook is self-contained and defaults to a short smoke test. Paper
training protocols must be selected explicitly in the configuration cell.

The current release contains:

- `d2_skew_symmetric_product_form.ipynb`: the 2D skew-symmetric product-form
  benchmark with the prespecified 15,000-step endpoint and Degree-5
  evaluation.
- `d2_harrison_nonproduct_global_cauchy.ipynb`: the 2D Harrison benchmark with
  a centered full-input non-product Holomorphic Cauchy branch, the same
  15,236-real-DOF budget, a prespecified 15,000-step endpoint, and Degree-5
  evaluation.

Jupyter autosave files under `.ipynb_checkpoints/` are deliberately excluded.
