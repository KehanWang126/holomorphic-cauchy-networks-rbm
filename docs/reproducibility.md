# Reproducibility protocol

Each audited 2D release is validated in two distinct stages. Keeping them
separate makes failures diagnosable.

## Stage A: checkpoint regression

Evaluate the archived, audited endpoint without changing its bytes. The
evaluation must:

1. load the fixed step-15,000 model;
2. use Talbot Degree 5 on the benchmark's frozen evaluation grid
   (\(t=1,\ldots,10\) for skew-symmetric and \(t=1,\ldots,5\) for Harrison);
3. report raw predictions without clipping;
4. reproduce the stored pointwise values and MREs in the reference software
   environment;
5. leave the SHA256 checkpoint hash unchanged.

This stage checks the cleaned evaluation and reporting path independently of
training variability.

## Stage B: clean end-to-end run

Restart the kernel, set `RUN_MODE = "train"`, and run every notebook cell in
order. A successful run must:

1. pass the architecture, anchor, analytic-Jacobian, gradient, and Cauchy-pole
   preflight checks; Harrison additionally passes full-input interaction
   residual checks before and after training;
2. count exactly 15,236 trainable real scalar degrees of freedom;
3. execute exactly 15,000 optimizer updates;
4. sample 51.24M BAR, 6.405M boundary, and 6.405M monotonicity constraints;
5. preserve the Adam optimizer state when the learning rate changes after
   step 12,000;
6. save only the prespecified step-15,000 endpoint;
7. introduce exact tails and the public baseline only after training;
8. produce finite raw Degree-5 predictions with no clipping;
9. confirm that evaluation does not modify the saved checkpoint.

The audited reference runs included in this repository used Python 3.10.20,
PyTorch 2.5.1 with CUDA 12.1, and an NVIDIA GeForce RTX 4080 Laptop GPU. Local
training times were 1,738.53 seconds for skew-symmetric and 2,109.60 seconds
for Harrison. Runtime is reported only as a local computational record, not
as a hardware-fair timing comparison with NN4MGF.

## Why both stages are needed

An exact checkpoint regression proves that code cleanup did not change the
evaluation. A fresh run proves that a user can reproduce the full protocol
from an uninitialized model. GPU kernels and software versions can prevent
bitwise-identical retraining, so a different new checkpoint hash alone is not
a failure.
