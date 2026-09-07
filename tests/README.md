# Validation tests

Run the 20D frozen-checkpoint validation from the repository root:

```bash
python tests/test_d20_activation_ablation.py
```

The test loads all six fixed step-25,000 endpoints, recomputes the complete
120-point grid, compares it with the committed reference output, verifies hashes
and metadata, and exercises failure cases. It does not train or alter a checkpoint.
