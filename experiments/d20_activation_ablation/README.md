# 20D fixed-endpoint activation ablation

This directory is a self-contained, evaluation-only package. It does not expose
a training mode and does not select a checkpoint or seed from observed accuracy.

## Run

From this directory, start Jupyter and execute the notebook from a fresh kernel:

```bash
jupyter lab d20_activation_ablation_evaluate.ipynb
```

Use **Restart Kernel and Run All**. The default configuration:

- runs on CPU;
- loads Cauchy and complex-tanh checkpoints for seeds 12345, 22345, and 32345;
- verifies every checkpoint against `checkpoint_manifest.json`;
- evaluates only the fixed step-25,000 endpoint;
- recomputes Degree-5 and Degree-8 predictions for `t=1,...,10`;
- writes a new timestamped directory under `outputs/`; and
- never trains or overwrites an earlier result.

The frozen tanh weights are loaded and audited with PyTorch, while their diagonal
affine-tanh-affine transform is evaluated in NumPy complex128. This is
mathematically identical to the original forward map and reproduces the supplied
tanh predictions, but avoids the observed Windows PyTorch native complex-tanh
kernel crash. Cauchy remains evaluated directly in PyTorch complex128.

## Interpretation

Degree 8 is primary; Degree 5 is a numerical diagnostic and historical comparison
bridge. Predictions are not clipped. Invalid, negative, above-one, extreme, or
nonfinite outcomes are recorded rather than removed. Aggregate standard deviations
use `ddof=1` over all three prespecified seeds.

This is a same-width and same-sampled-constraint-budget comparison, not an exact
parameter match. Cauchy has 16,388 trainable real scalar degrees of freedom and
tanh has 15,108. Each method/seed training run used 64,000,000 sampled constraints.

The NN4MGF file under `reference/` is an archived released-checkpoint evaluation.
Its original checkpoint was not supplied, so this notebook does not claim to
re-evaluate it.

The audited output is committed separately under
`../../results/d20_activation_ablation/`. `validation_cpu.json` records 22 passing
tests and a complete 120-point CPU evaluation.

Run the self-contained regression and negative tests from the repository root:

```bash
python tests/test_d20_activation_ablation.py
```
