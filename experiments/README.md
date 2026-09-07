# Controlled experiments

This directory contains self-contained controlled comparisons that should not be
mixed into the main training notebooks.

- `d20_activation_ablation/`: evaluation-only Cauchy-versus-complex-tanh
  comparison using six frozen step-25,000 endpoints. Run from within that
  directory so the notebook can resolve its manifest, weights, and reference
  files without machine-specific paths.
