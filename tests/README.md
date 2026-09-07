# Validation tests

Run the 20D frozen-checkpoint validation from the repository root:

```bash
python tests/test_d20_activation_ablation.py
```

The test loads all six fixed step-25,000 endpoints, recomputes the complete
120-point grid, compares it with the committed reference output, verifies hashes
and metadata, and exercises failure cases. It does not train or alter a checkpoint.

Run the static 30D notebook release checks with:

```bash
python tests/test_d30_notebook.py
```

This verifies the fixed seed, 50k endpoint, matched 64M sample budget, model
parameter count, learning-rate schedule, evaluation policy, provenance, clean
cell state, hash-gated support for the archived historical checkpoint, and
absence of excluded 60k/retrospective-selection paths.

Verify the committed 30D result bundle without importing PyTorch:

```bash
python tests/test_d30_release.py
```

This checks every manifest entry, the exact endpoint hash, the complete
Degree-5/6/8 pointwise grid, training-history agreement, reported MREs, the
Degree-8 Talbot floor, and the recorded public smoke result.
