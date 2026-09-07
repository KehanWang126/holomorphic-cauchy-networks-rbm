# Reproducibility protocol

The release separates checkpoint regression from fresh training. This distinction
keeps numerical evaluation, training variability, and checkpoint selection from
being conflated.

## Stage A: fixed-checkpoint regression

Stage A loads an archived endpoint without changing its bytes. It must:

1. verify the checkpoint identity and SHA-256 hash before loading;
2. verify the benchmark, method, seed, model configuration, and endpoint metadata;
3. recompute every prespecified evaluation point;
4. report raw predictions without clipping;
5. retain invalid, negative, above-one, extreme, and nonfinite outcomes;
6. use no best-checkpoint or best-seed selection; and
7. confirm that the checkpoint hash is unchanged after evaluation.

For 2D, the fixed endpoint is step 15,000 and the primary inversion uses Talbot
Degree 5. For 20D, the endpoint is step 25,000, Degree 8 is primary, and Degree 5
is retained only as a numerical diagnostic and historical comparison bridge.

The 20D activation evaluator verifies six endpoints: Cauchy and complex tanh for
seeds 12345, 22345, and 32345. It evaluates 120 points in total. A complete CPU
run with PyTorch 2.5.1 and mpmath 1.3.0 passed 22 automated tests.

### Windows-safe tanh evaluation

The supplied tanh checkpoints are read and structurally audited with PyTorch.
Their frozen diagonal affine-tanh-affine map is then evaluated with NumPy
complex128. This produces the same supplied pointwise predictions while avoiding
the observed Windows native PyTorch complex-tanh kernel termination at extreme
Talbot nodes. It is a fresh forward computation from the weights, not a lookup of
archived tanh predictions. The Cauchy endpoints remain evaluated in PyTorch
complex128.

## Stage B: clean training reproduction

Restart the kernel, select the documented training mode, and execute every cell
in order.

### 2D requirements

A successful 2D run must:

1. pass architecture, anchor, analytic-Jacobian, gradient, and pole-proximity
   checks;
2. count exactly 15,236 trainable real scalar degrees of freedom;
3. execute exactly 15,000 Adam updates for seed 12345;
4. sample 51.24M BAR, 6.405M boundary, and 6.405M monotonicity constraints;
5. preserve optimizer state when the learning rate changes after step 12,000;
6. save and report only the prespecified step-15,000 endpoint;
7. introduce exact tails and the public baseline only after training; and
8. produce finite, unclipped Degree-5 predictions.

The Harrison run additionally verifies its centered full-input global branch and
nonzero interaction residual.

### 20D main-experiment requirements

A successful 20D run must:

1. use only the canonical `d20_skew_symmetric_product_form.ipynb` notebook;
2. pass RBM, boundary-mass, architecture, analytic-Jacobian, gradient, and
   Cauchy-pole preflight checks;
3. count exactly 16,388 trainable real scalar degrees of freedom;
4. use only the prespecified seeds 12345, 22345, and 32345;
5. execute exactly 25,000 Adam updates per seed;
6. sample 51.2M BAR, 6.4M boundary, and 6.4M monotonicity constraints per seed;
7. preserve Adam state when the learning rate changes after step 20,000;
8. save and report only the step-25,000 endpoint;
9. keep exact tails and baselines outside training and endpoint selection; and
10. evaluate Degree 8 as primary without clipping or seed exclusion.

The included new seed-12345 training run used the same frozen protocol and
replaced its interrupted historical predecessor. Seeds 22345 and 32345 retain
their previously audited fixed endpoints.

## Talbot diagnostic

The exact-transform files apply the same stock `mpmath.invertlaplace` Talbot call
to the analytic transform. They quantify the numerical inversion error at the
specified degree. They are not a mathematical lower bound on learned-model error:
transform approximation and inversion errors can reinforce or cancel.

The outer `mp.workdps(100)` context does not imply 100-digit model inference.
Stock mpmath chooses internal Talbot precision from `degree`, while neural
evaluations use complex128.

## Baseline provenance

The 2D published NN4MGF values are traced to the public NN4MGF repository and
commit recorded in the root README. The 20D NN4MGF entries are archived
released-checkpoint predictions. The corresponding weight file was not supplied,
so the repository reports them separately and does not claim that they were
re-evaluated or trained under a matched computational budget.

## Interpreting fresh runs

GPU kernels, drivers, and library versions can prevent bitwise-identical fresh
training. A new checkpoint hash is therefore not expected to match the archived
one. Acceptance of a fresh training reproduction is based on the frozen protocol,
successful audits, complete fixed-endpoint reporting, and comparable accuracy.

In contrast, evaluating an unchanged archived checkpoint must preserve its hash
and reproduce the reference pointwise values within the stated numerical
tolerance.
