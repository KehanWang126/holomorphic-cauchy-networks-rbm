# Reproducibility protocol

The release separates checkpoint regression from fresh training. This distinction
keeps numerical evaluation, training variability, and checkpoint selection from
being conflated.

## Stage A: fixed-checkpoint regression

Stage A loads an archived endpoint without changing its bytes. It must:

1. verify the checkpoint identity and SHA-256 hash before loading;
2. verify the benchmark, method, seed, model configuration, and endpoint metadata;
3. recompute every prespecified evaluation point;
4. retain the raw prediction and report a separate clipped value for every finite prediction;
5. retain and count negative, above-one, extreme, and nonfinite raw outcomes; nonfinite outputs remain invalid;
6. use no best-checkpoint or best-seed selection; and
7. confirm that the checkpoint hash is unchanged after evaluation.

For 2D, the fixed endpoint is step 15,000 and the primary inversion uses Talbot
Degree 5. For 20D, the endpoint is step 25,000, Degree 8 is primary, and Degree 5
is retained only as a numerical diagnostic and historical comparison bridge.
For 30D, the endpoint is step 50,000 and Degree 8 is primary; Degrees 5 and 6
quantify low-degree inversion effects.
For 50D, the endpoint is step 100,000 and Degree 8 is primary; Degree 5 is
retained only as a numerical diagnostic. Legacy loading is enabled only for the
exact archived endpoint whose SHA-256 is frozen in the public notebook.

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
8. produce Degree-5 raw predictions and compute reported MRE from their clipped values.

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
10. evaluate Degree 8 as primary, clipping finite predictions for reported MRE without excluding any seed.

The included new seed-12345 training run used the same frozen protocol and
replaced its interrupted historical predecessor. Seeds 22345 and 32345 retain
their previously audited fixed endpoints.

### 30D main-experiment requirements

A successful 30D run must:

1. use seed 12345 and preserve the exact Adam and RNG state across the historical
   step-20,000 continuation boundary;
2. execute exactly 50,000 Adam updates with batches 1024/128/128;
3. expose BAR, boundary, and monotonicity branches to 51.2M, 6.4M, and 6.4M
   sampled constraints, respectively;
4. preserve Adam state when the learning rate changes after step 40,000;
5. use sequential branch backward only to reduce peak memory;
6. report only the prespecified step-50,000 checkpoint;
7. keep recovery checkpoints, exact tails, and public baselines outside model
   selection; and
8. preserve raw Degree-5/6/8 predictions, reporting clipped MRE with Degree 8 primary.

The released historical endpoint uses checkpoint schema
`d30_general_pointmatched50k_v1` and SHA-256
`0bfcd0dc11779fabd477fcf5ee9bf07772e555a9b10776618dc4e47a1106be1c`.
Its stored predictions were cross-checked between the training history and the
independent 35k-to-50k diagnostic table. The public notebook also exposes a
consolidated clean-training path, but that path is not substituted for the
archived endpoint. The later 50k-to-60k continuation and retrospective 29k/49k
minima are excluded from the main result.

### 50D main-experiment requirements

A successful 50D run must:

1. use seed 12345 and start from a fresh initialization;
2. execute exactly 100,000 Adam updates with batches 512/64/64;
3. expose BAR, boundary, and monotonicity branches to 51.2M, 6.4M, and 6.4M
   sampled constraints, respectively;
4. preserve Adam state when the learning rate changes after step 80,000;
5. use sequential branch backward to control peak memory;
6. report only the prespecified step-100,000 checkpoint;
7. keep recovery checkpoints and exact tails outside model selection; and
8. preserve raw Degree-5/8 predictions, reporting clipped MRE with Degree 8 primary.

The historical run identifies its endpoint with schema
`d50_general_pointmatched100k_v1` and variant
`d50_controlled_general_scaling_pure_general`. The released diagnostic endpoint
has SHA-256
`669c9bc1b2188241e47e78ee5a23bfaa690a86233529b8eef340daf11cf79b0a`.
The original 80k-to-100k diagnostic is retained only as provenance for the fixed
100k endpoint; it is not used for checkpoint selection.

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


### Paper summary synchronization

The original files under `results/` remain hash-verified raw checkpoint evaluations. They are retained as source evidence, including any out-of-range values and historical raw summaries. The manuscript tables instead use the reproducible, independently hash-checked post-processing under `paper/audit/`. For each finite real tail prediction, it reports `min(1, max(0, raw_prediction))`; nonfinite results are flagged and never silently substituted. Interval MRE is computed per seed and averaged across all prespecified seeds (sample SD with `ddof=1`). The 20D tanh seed 22345 is included, even where its raw prediction is extreme.

From the repository root, run `python paper/audit/audit_results.py` to validate bundled input hashes and reconstruct pointwise, per-run, and aggregate tables. `python scripts/build_paper_summaries.py` additionally compares rebuilt tables against the published `paper/data/` copies. The old `results/d20_activation_ablation/paired_success_summary.csv` and `failure_summary.csv` document an earlier conditional/raw convention and are not manuscript results. Signed raw and clipped decompositions are both in `paper/data/`.

The 10D non-product seed-12345 result uses an unverified pilot simulation reference. Neither this result nor the H100 profile extrapolation is a completed, independently verified 30D non-product reference or a full-run runtime measurement.
