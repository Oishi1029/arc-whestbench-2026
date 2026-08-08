"""wmc4 + the layer-1 EXACT-MOMENT ANCHOR at full covariance strength.  `_ANCHOR = "full"`.

Official harness, `whest run --split full --n-mlps 200 --runner subprocess`, the SAME 200
MLPs for both, same session, this machine:

                                       wmc4 (live)        this
    Adjusted Final-Layer Score            2.24e-07     2.14e-07     1.047x
    Raw Final-Layer MSE                   3.26e-07     3.09e-07     1.055x
    All-Layers MSE                        6.06e-07     5.02e-07     1.207x
    Mean Compute Utilization             0.6898789    0.6963142
    Failed MLPs                            0 / 200      0 / 200

The all-layers number moves most because the layer-1 row is now returned EXACTLY, not
sampled.  The scored (final-layer) number is the one that counts, and 200 MLPs cannot
resolve it -- see below, this is the whole methodological point of this file.

===========================================================================================
THE MECHANISM
-------------
x is exactly N(0, I), so z_1 = W_0^T x is exactly jointly Gaussian with covariance
S = W_0^T W_0, and layer 1's post-ReLU moments are closed-form with NO sampling error:

    sigma_i = ||W_0[:, i]||
    mu_i    = sigma_i / sqrt(2 pi)
    sd_i    = sigma_i sqrt((pi - 1) / (2 pi))
    C_ij    = (sigma_i sigma_j / 2pi)[ rho (pi/2 + arcsin rho) + sqrt(1 - rho^2) ] - mu_i mu_j
              rho_ij = S_ij / (sigma_i sigma_j)          [the arc-cosine kernel]

The ensemble is pushed onto them:  Y' = (Y - mean Y) A + mu, with A = Cov_emp^{-1/2} C^{1/2}
("full"), diag(sd / sd_emp) ("diag"), or I ("mean").

THE MEASUREMENT -- AND WHY IT NEEDS n = 1000
--------------------------------------------
PAIRED, common random numbers, EQUAL COMPUTE (each variant's k is cut by its own extra cost,
so an MSE ratio here IS a score ratio), real competition MLPs of the `full` split against the
dataset's own 1e9-sample ground truth, at the real operating point k = 64,512:

    variant            mean-MSE ratio   t(mean)   t(log)   win rate   95% CI (bootstrap)
    base = wmc4 stats       1.0000         --       --        --              --
    + mean anchor           1.0355       +3.81     +4.65     57.7%     [1.017, 1.054]
    + diagonal anchor       1.0422       +4.66     +5.03     57.2%     [1.025, 1.061]
    + FULL covariance       1.0832       +5.45     +5.57     58.7%     [1.053, 1.113]

    full vs diag            1.0393       +3.29     +2.79     55%
    full vs mean            1.0461       +3.63     +2.92     54%

So the brief's 1.09-1.13x claim sits at the TOP of the confidence interval, the point
estimate is 1.083x, and the claim that the FULL covariance version is materially different
from the mean-only one is CORRECT: full beats mean-only by 1.046x at t = +3.63.

**The cheap variant is not the right answer.**  The diagonal anchor is free and captures half
the gain; the full one costs 3.4% of the budget and still wins by 1.039x at t = +3.29.

TWO REASONS THIS WAS PREVIOUSLY MEASURED AS NEUTRAL, BOTH METHODOLOGICAL
------------------------------------------------------------------------
RESEARCH.md 5 and 5b measured the mean-only anchor at 0.994x (t = -0.57, n = 250) and the
mean+covariance anchor at 1.06x (n = 8), and concluded neutral.  Both readings were right
about their own experiment and wrong about this one:

  * **WRONG OPERATING POINT.**  `runoff.py` equalises compute at 9.5% of budget, which is
    k = 5,900 -- 11x smaller than the real k = 70,312.  The anchor's value is strongly
    k-dependent, because the full transform fits p = n(n+1)/2 = 32,896 free parameters from
    k samples and the overfitting penalty goes as (1 + p/k).  Re-measured at k = 5,900, this
    same code gives: mean 1.015x, diag 1.002x, **full 0.825x -- materially HARMFUL**.  At
    k = 64,512 it is 1.083x.  The sign of the full anchor flips with k.  Any mechanism whose
    benefit scales with k is invisible to a 9.5%-of-budget run-off.

  * **NOT ENOUGH MLPs.**  The per-MLP effect is heavy-tailed.  On the same running sample the
    full anchor measured 1.034x at n = 125, 1.036x at n = 200, **1.018x at n = 300** (t =
    +0.70, median ratio 0.9965, win rate 48.7% -- it looked like a clean null), and 1.083x at
    n = 1000.  Two disjoint 300-MLP halves of the same split disagreed at 1.018x vs 1.111x.
    The "unpaired comparisons under 200 MLPs cannot resolve 25%" rule of thumb is optimistic
    here: even PAIRED, 300 MLPs could not resolve an 8% effect.

WHAT IT COSTS (flopscope 0.10.0, width 256 / depth 32, MEASURED)
----------------------------------------------------------------
The anchored ensemble is never materialised.  With Y' = (Y - m) A + mu the layer-2
pre-activation is Y' W_1 = Y (A W_1) + (mu - m A) W_1, so the transform folds into W_1 as one
n x n weight product plus one broadcast row add -- there is NO k n^2 transform pass, only the
k n^2 empirical Gram that A is built from.

    variant   extra per-sample        extra fixed       % of sized budget
    mean      n                       8 n^2                   0.02%
    diag      3 n                     8 n^2                   0.03%
    full      Gram 102,239 / sample   60 n^3 + 88 n^2         3.37%

The Gram is a (256, k) @ (k, 256) product and Strassen applies to it as well: 102,239 per
sample at 2 levels against 131,056 plain, a 22% saving on the dominant line.  The two float64
eigendecompositions are 0.37%.  Net effect on the pass:  k falls 72,712 -> 70,312 (-3.30%),
executed FLOPs rise 1.2%, residual wall time is unchanged (about ten extra numpy calls per
MLP).  Predicted adjusted-score ratio from the measured variance-constant improvement
(C_none / C_full = 1.116) and the exact k and utilisation ratios: **1.065x**.  Harness
measured 1.047x at n = 200, which that n cannot separate from 1.065x.

The brief costed this at ~0.3% of budget.  That is the two eigendecompositions only; the
k n^2 Gram that feeds them is 10x larger and is the real price.

THE HEADLINE NUMBER, MEASURED AT THE EXACT SHIPPED SAMPLE COUNTS
----------------------------------------------------------------
The table above equalises compute through a model.  This one does not: it runs the base at
k = 72,712 and the anchored variant at k = 70,312 -- the two values `_sample_count` actually
returns -- over all 1000 MLPs of the `full` split, paired:

    raw final-layer MSE ratio     1.0744    paired t = +5.15   win 57.0%
                                            95% CI (bootstrap) [1.0469, 1.1028]
    measured utilisation ratio    1.0119    (0.6665 -> 0.6744 FLOPs, n = 200)
    ADJUSTED SCORE RATIO          1.0618

So: **1.062x on the adjusted score, at t = +5.15 over 1000 MLPs.**  The two n = 200
confirmations (official harness 1.047x, paired flopscope 1.035x at t = +1.13) are both
consistent with it and neither can resolve it, which is the point.

TWO DEFECTS FOUND AND FIXED, BOTH WORTH RECORDING
--------------------------------------------------
(1) A HARD-CODED CONSTANT, WRONG IN THE 4TH FIGURE.  sqrt((pi-1)/(2pi)) was typed as
    0.5825213767497766 instead of 0.583819370103549, an error of -0.22%.  It measured **24x
    WORSE than no anchor at all** (raw MSE 3.87e-06 against 1.59e-07).  The diagonal anchor
    multiplies every layer-1 fluctuation by sd_exact / sd_emp; ReLU is positively homogeneous;
    so a systematic rescaling of layer 1 propagates undamped through all 31 remaining layers.
    A moment anchor has NO error tolerance in the moment it anchors to.  This is the same
    mechanism that makes anchoring to ANALYTIC moments at layers >= 2 catastrophic
    (RESEARCH.md: 0.75x at L=2, 0.10x at L=8, where sigma is ~9% wrong).  Now derived from
    `math.pi`, not transcribed.

(2) A RUNT BLOCK.  The anchor is built per ensemble block.  At wmc4's `_BLOCK` = 65,536 the
    shipped k = 70,024 split into 65,536 + 4,488, and the 4,488-sample runt was being anchored
    with p/k = 7.3 -- fitting 32,896 parameters to 4,488 samples.  MEASURED over 8 real MLPs:
    two blocks 2.2158e-07 against one block 2.1449e-07, a 3.3% penalty from the runt alone.
    `_BLOCK` is now 131,072 so the whole depth-32 ensemble is one block, and
    `_MIN_ANCHOR_BLOCK` = 32,768 declines to anchor any short block that still occurs.

SAFETY
------
Unchanged in kind from wmc4 and re-measured with the anchor on.  `k` is still sized as if
NOTHING prunes, now including the anchor's exact Gram and eigendecomposition cost, so the
sized spend is still a strict upper bound.  The anchor cannot break `_step`'s exact dead-row
pruning: a row of `yt` that is identically zero contributes nothing to Y (A W_1) whatever A
is, and its constant part is already inside the shift.  Any numerical trouble in building A
returns None and the pass runs unanchored.  Effective compute (FLOPs + 1e11 * residual) as a
fraction of budget, all cases finite and correctly shaped:

    all-positive weights (nothing EVER dead)   0.9325   <-- binding case (0.9098 is FLOPs)
    depth 5 / 8 / 10                           0.9218 / 0.9213 / 0.9015
    normal He init                             0.7028
    weights x10 / x0.01                        0.7085 / 0.5494
    rank-1 / all-negative / all-zero           0.4420 / 0.2402 / 0.0783
    depth 1 / 2 / 3 / 64                       0.2577 / 0.5712 / 0.7260 / 0.5635
    width 250 / 255 / 300 (odd, unpaddable)    0.6318 / 0.6034 / 0.7097
    width 4 depth 2, width 1 depth 32          0.0006 / 0.0017

Set `_ANCHOR` to None to recover wmc4 exactly: the anchor cost terms become identically zero,
`_build_anchor` returns before touching an array, and `k` is unchanged.

===========================================================================================

--- everything below this line is wmc4.py, unchanged apart from the anchor hooks ---

REPLICATION of the Strassen leg of AIcrowd forum topic 18106 ("Team SOX", sub 319341,
adjusted 1.551e-07), grafted onto our own wmc3.py.

PROVENANCE.  Everything below the line `--- wmc3 ---` is wmc3.py unchanged.  The new
material is the fast matrix multiply and the exact cost model that sizes it.  Source for
the idea: `main.pdf` attached to topic 18106, section 3.2: "The dense portion is evaluated
with Strassen matrix multiplication [24], while the prefix of cold columns is processed row
by row".  That one clause is the only place their write-up describes a matmul that is
cheaper than 2*m*k*n MACs, and it is the leg of their recipe we did not already have.

WHY THIS IS THE LEG THAT MATTERS.  Their published operating point decomposes cleanly:
    N = 84,992 samples, raw MSE 2.18e-07  ->  variance constant C = MSE*N = 1.85e-02
    ours:  k = 57,800,   raw MSE 4.01e-07 ->  C = 2.32e-02          (1.25x worse)
    their per-sample cost 0.71*2.72e11/84,992 = 2.27e6 FLOPs
    our   per-sample cost 0.657*2.72e11/57,800 = 3.09e6 FLOPs       (1.36x worse)
    1.25 * 1.36 = 1.70x, against the 1.74x gap actually observed.
So roughly HALF their advantage over us is a pure cost-per-sample effect, and Strassen is
the only mechanism in their write-up big enough to explain it.

WHAT IS STATED vs WHAT IS INFERRED.  Stated: Strassen for the dense block; a cold-column
prefix multiplied row-by-row with row buckets on the ladder (0,1,2,4,8); dead/kink/on
classification at |alpha| = 3 with a pilot re-classification; on-neurons folded linearly
through layers 31-32; antithetic scrambled Sobol; N = 84,992.  Inferred by us: the
recursion depth, the leaf size, and the fact that the counted-FLOP saving survives the
flopscope meter at all (they never quote a FLOP breakdown).

FLOPSCOPE PRICES STRASSEN HONESTLY -- MEASURED, NOT ASSUMED.
`flopscope.numpy.matmul` bills `2*m*k*n - m*n` and elementwise add/sub and `concatenate`
bill 1 per element at float32.  So a Strassen level really does reduce the meter, and the
closed-form model `_mm_cost` below reproduces the meter to the FLOP:

    A(256x256) @ B(256x65536), float32           predicted        measured
      direct                                   8,573,157,376   8,573,157,376
      1 level                                  7,574,994,944   7,574,994,944   0.8836x
      2 levels                                 6,767,734,784   6,767,734,784   0.7894x
      3 levels                                 6,177,113,088   6,177,113,088   0.7205x
      4 levels                                 5,862,848,256   5,862,848,256   0.6839x
      5 levels                                 5,942,292,544   5,942,292,544   0.6931x  (worse)

This is not a billing artefact: Strassen genuinely performs fewer multiplications.  It is
the same arithmetic the meter would charge a hand-written O(n^2.807) kernel.

WHAT IT COSTS.  Accuracy: max relative error of the product against a float64 reference
rises 9.09e-07 (direct float32) -> 1.34e-06 (2 levels) -> 1.86e-06 (3 levels).  Wall time:
numpy's own time inside a counted op is NOT residual (residual = wall - flopscope backend -
flopscope overhead), so the extra numpy calls are largely free; only our Python glue is
billed.  Both effects are measured end-to-end below, not argued.

RESULT -- OFFICIAL HARNESS, `whest run --split full --n-mlps 200 --runner subprocess`,
this machine, the same 200 MLPs for both:

                                       wmc3 (live lineage)       this
    Adjusted Final-Layer Score                   2.61e-07     2.32e-07     1.125x
    Raw Final-Layer MSE                          3.99e-07     3.51e-07
    All-Layers MSE                               7.55e-07     6.67e-07
    Mean Compute Utilization                    0.6570208    0.6638290
    Residual Wall Time / MLP                                    10.5 ms
    Failed MLPs                                   0 / 200      0 / 200
    Worst MLP                                    1.17e-06     1.45e-06

PAIRED, 250 real competition MLPs of the `full` split, common MLPs, exact flopscope 0.10.0
accounting against the dataset's own 1e9-sample ground truth:

                              wmc3          this
    adjusted score        2.6373e-07    2.3569e-07    1.1190x   paired t = +4.93
    raw final-layer MSE   4.0078e-07    3.5423e-07    1.1314x   paired t = +5.34
    compute utilisation       0.6619        0.6698
    worst MLP              2.001e-06     1.665e-06

The two accountings agree to 0.5%.  The raw-MSE gain IS the whole gain: the pass is 11.6%
cheaper per sample, so the same worst-case sizing buys 11.6% more samples, and MSE = C/k
converts that one-for-one.  Nothing statistical changed.

WHY ONLY ONE LEVEL, WHEN TWO IS 10.7% CHEAPER STILL.  Because residual wall time is the
binding constraint, not FLOPs.  Deeper recursion multiplies the number of Python-level
numpy calls by 7 per level (28 calls per product at 1 level, ~217 at 2, ~1500 at 3), and
that glue is billed at 1e11 FLOP-equivalents per second.  Measured, 16 real MLPs, offline
flopscope accounting, adjusted score and residual wall time per MLP:

    levels   chunk    adjusted     util    residual/MLP    vs 0 levels
      0      16,384   2.1390e-07  0.6618       5.8 ms        1.000x
      1      65,536   1.7410e-07  0.6675      11.6 ms        1.229x   <-- shipped
      2      16,384   1.7990e-07  0.7111     110.1 ms        1.189x
      2      65,536   1.7665e-07  0.6967      70.7 ms        1.211x
      3      65,536   1.8206e-07  0.8055     311.8 ms        1.175x

Level 3 has the lowest raw MSE of the five (2.27e-07, since it buys the most samples) and
the WORST adjusted score but one, because 312 ms of residual is 11.5% of the whole budget.
Level 2 in a single 131,072-sample block -- the configuration that would have amortised the
Python glue best -- does not run at all: it trips flopscope's 60 s wall-clock limit
(`TimeExhaustedError: zeros: wall-clock time 62.961s exceeds limit 60.000s`).

This is worth stating plainly because it is the general lesson: under a score of the form
`flops + 1e11 * residual_seconds`, an asymptotically faster algorithm is only worth its
asymptotics if its constant-factor bookkeeping stays inside counted primitives.  Strassen
at one level is 12 extra numpy calls per layer; at three levels it is ~1500, and the
interpreter overhead alone outruns the arithmetic it saves.

BUDGET SAFETY -- unchanged in kind, re-measured.  `k` is still sized as if NOTHING prunes,
now against the exact Strassen cost model, so the sized spend is still a strict upper bound
on the executed spend.  Effective compute (FLOPs + 1e11 * residual) as a fraction of budget
on the pathological inputs:

    all-positive weights (nothing EVER dead)   0.9170   <-- the binding case
    rank-1 weights (maximal collapse)          0.9144
    depth 5 / 8 / 10 / 64                      0.917 / 0.917 / 0.883 / 0.522
    normal He init                             0.6698
    weights x10 (float32 overflow)             0.6692
    weights x0.01 (underflow)                  0.5272
    all-negative weights (everything dead)     0.2150
    all-zero weights                           0.0474
    width 250 / 255 / 300 (odd, unpaddable)    0.7208 / 0.6392 / 0.6776
    width 4 depth 2, width 1 depth 32          0.0004 / 0.0020

Every case returns a finite (depth, width) array and none reaches the budget.  Odd widths
matter here in a way they did not before: `_smm` checks divisibility at every level and
falls back to the plain product, so a width the recursion cannot split costs exactly what
wmc3 costs -- it degrades, it does not fail.

WHAT THIS DOES *NOT* REPLICATE, AND WHAT THAT IS WORTH.  Two legs of 18106 are still open:

  * The packed cold-column prefix (their section 3.2.1).  Their own Table 2 puts the cold
    prefix at 35.3% of the layer width on a plain ReLU stack, carrying ~3 nonzeros per
    sample row.  Replacing a dense 90-column block with ~3 gathered columns per row is
    arithmetically worth about 1.5x on the layer product -- larger than Strassen -- but
    flopscope bills a gather at 4 FLOPs per element against 1 for arithmetic, and the row
    bucketing needs a per-row sort.  Not attempted here.  Note it is the unbiased version
    of our own `_MIN_FIRE > 1` epsilon-pruning knob, which wmc3 measured at 1.30x and
    rejected precisely because dropping rarely-firing neurons is biased; 18106 keeps the
    saving and pays for the rare firings exactly.
  * Antithetic scrambled Sobol in place of our whitened antithetic Gaussian ensemble.
    That is the variance axis, worth 1.25x by the decomposition above.

DIVISIBILITY.  Strassen needs all three dimensions even at every level.  Rather than pad
with zero rows, the pruning masks are ROUNDED UP to a multiple of 2^levels by putting a few
already-dead neurons back into the alive set.  That is strictly conservative -- a neuron
that is computed instead of masked can only reduce the mask bias -- and it keeps every
weight block a genuine sub-block with no padding arithmetic.  `k` is rounded down to a
multiple of 2^(levels+1) so every chunk length is divisible too.

--- everything below this line is wmc3.py, unchanged ---

Descendant of `work/mine/wmc2.py` (live submission, adjusted 2.700e-07).  Everything
statistical is unchanged -- whitened ensemble, antithetic pairing, float32, canonical
per-MLP seeding, MSE = C/k with p = 1.  Every change below is a pure COST reduction, which
is the only channel that moves the adjusted score for a 1/k estimator:

        score = C * cost_per_sample / budget          (flat in k above the 0.1 floor)

MEASURED
--------
OFFICIAL HARNESS, `whest run --split full --n-mlps 200 --runner subprocess`, the same 200
MLPs for both, this machine:

                                        wmc2 (live)        this
    adjusted final-layer score            3.23e-07      2.61e-07      1.2375x
    raw final-layer MSE                   4.04e-07      3.99e-07
    all-layers MSE                        7.65e-07      7.55e-07
    mean compute utilisation             0.8006086     0.6570208
    residual wall time / MLP                            5.3 ms
    failed MLPs                           0 / 200       0 / 200
    worst MLP                             1.72e-06      1.17e-06

The live submission scores 3.23e-07 on this split and 2.700e-07 on the leaderboard's, so
the same 1.2375x puts this at a projected **2.18e-07** there.

PAIRED, 250 real competition MLPs of the `full` split, exact flopscope 0.10.0 accounting
against the dataset's own 1e9-sample ground truth (both estimators seed from mlp.seed, so
the MLPs pair exactly):

                              wmc2 (live)      this
    adjusted score            3.2428e-07    2.6343e-07     1.2342x   paired t = +14.58
    raw final-layer MSE       4.0475e-07    4.0078e-07     paired t = +1.15  (n.s.)
    compute utilisation          0.7932       0.6550
    residual wall time / MLP     21.8 ms       6.1 ms

The two accountings agree to 0.3%.  The raw MSE is unchanged (t = +1.15, n.s.): this is
entirely a cost reduction, and the 1.8% of extra samples the cheaper pass buys at the same
worst-case sizing happens to offset the mask bias almost exactly.

THE THREE MECHANISMS, in decreasing order of size
-------------------------------------------------

(B) TWO-SIDED (LEAD-BLOCK) PRUNING -- the big one.
    wmc2 prunes only the *input rows* of each layer matmul: rows of the transposed
    activation block `yt` that are identically zero over the chunk.  It still pays
    2 * a_in * n * c to compute all n *output* columns, and only discovers at the NEXT
    layer that ~25% of them were dead.  No exact certificate can fix that -- knowing
    max_t z_j <= 0 requires computing z_j.

    So the first `_LEAD` samples of the ensemble are run at full output width (exactly as
    wmc2 does today, one-sided exact pruning), and the alive set observed there is FROZEN
    as a static mask for the remaining samples.  Each tail matmul then costs
    2 * a_in * a_out * c.  The lead samples are ordinary ensemble members -- they still
    contribute to the estimate -- so the mask costs nothing to build; we merely forgo the
    output-side saving on `_LEAD / k` ~ 1.8% of the pass.

    The reduced weight blocks W[l][ix_(mask[l-1], mask[l])] are gathered ONCE, outside the
    chunk loop (~4 * a_in * a_out FLOPs each, ~3e7 total).  Per-layer sample sums are
    accumulated in the reduced index space and scattered to full width once at the end via
    `fnp.bincount` (8 FLOPs per entry -- flopscope arrays are immutable, so a per-chunk
    scatter would need a one-hot matmul; a static mask avoids the problem entirely).

    This introduces a genuine BIAS -- a neuron silent over the lead block but alive later
    is treated as zero -- and the bias does NOT shrink with k.  It is bounded by the lead
    size: _LEAD = 1024 is the measured knee (512 gives the same saving with 6x the bias,
    2048/4096 give less saving for less bias).  Layer `depth-1` is NEVER masked on the
    output side: that is the scored layer, and masking bias there would land directly on
    the score.

(C) FINAL-LAYER OUTPUT-COLUMN CLASSIFICATION.
    The lead block also yields the full-width pre-activation Z of the scored layer.  Per
    column, with a `_FINAL_MARGIN`-sigma safety band:

        on   := min_t Z_j >  +margin * sd_j   -> ReLU is the identity for this neuron, so
                                                 mean_s ReLU(z_j) = w_j . mean_s(y_{L-1})
                                                 EXACTLY (the sample mean commutes with a
                                                 linear map).  One dot product, no samples.
        dead := max_t Z_j <  -margin * sd_j   -> emit 0.
        kink := everything else               -> pay the full k-sample column.

    ~45% of the scored layer's columns classify at margin 1.0, which is ~1.5% of the pass.
    NOTE what this is NOT: identifying a neuron as linear cannot reduce the VARIANCE of a
    mean (the two estimators are bit-identical when the classification is right).  The
    entire gain is cost.  Measured misclassification at margin 1.0: 0.25 false-on and 0.50
    false-dead per MLP, dMSE/MSE = -7.7e-07, i.e. below the noise.

(A) GRAM HALVING -- free and exact.
    The ensemble is antithetic, x = [xh ; -xh], so x^T x == 2 * xh^T xh *identically*.
    The whitener's Gram pass is 3.46% of the whole budget and this halves it, with zero
    bias and zero variance change (flopscope-verified 1.073676e9 -> 5.368054e8).

THE KNOB SWEEP THAT SET THE CONSTANTS
-------------------------------------
48 real MLPs, each variant compared against ITS OWN exact reference -- the same code with
the mask and the classification disabled (`_LEAD` above k), which is the identical
ensemble at the identical k, so the difference is the mask/classification error alone and
the Monte-Carlo noise cancels exactly.  `bias^2` is mean((pred - pred_exact)^2).

    variant                        gain vs exact ref   bias^2 mean   worst    % of MSE
    lead 1024, mask only                  1.2410x       1.500e-09   4.03e-09    0.47%
    lead 1024 + classify 1.0 sd           1.2570x       1.499e-09   4.03e-09    0.47%   <-- shipped
    lead 1024 + classify 0.5 sd           1.2602x       1.501e-09   4.04e-09    0.47%
    lead 1024 + classify 2.0 sd           1.2481x       1.500e-09   4.03e-09    0.47%
    lead  512 + classify 1.0 sd           1.2754x       8.298e-09   6.30e-08    2.58%
    lead 2048 + classify 1.0 sd           1.2179x       2.983e-10   9.03e-10    0.09%
    lead 1024 + classify, _MIN_FIRE 2     1.2716x       4.502e-09   1.45e-08    1.39%
    lead 1024 + classify, _MIN_FIRE 4     1.3040x       1.694e-08   5.26e-08    5.18%

Three things this settles.

  * The scored-layer classification is genuinely free of error.  bias^2 is 1.499e-09 with
    it and 1.500e-09 without -- identical to four digits, and identical again at 0.5 and
    2.0 sd.  All of the bias is the mask; the classification contributes none of it.  That
    is the commutation identity showing up as a measurement: when the classification is
    right the two estimators are the same number, and at a 1-sd band it is essentially
    always right.  Margin 0.5 would buy another 0.25% for no measured bias, but 1.0 is
    where the misclassification counts are 0.25 false-on / 0.50 false-dead per MLP, and
    0.25% is not worth loosening a certificate on a private split.

  * `_LEAD = 1024` is the knee, sharply.  512 buys 1.5% more and takes 5.5x the mean bias
    and 16x the WORST-MLP bias (6.3e-08, a fifth of that MLP's whole MSE).  2048 halves
    the gain of the mask for a bias that was already 0.47%.

  * `_MIN_FIRE > 1` (epsilon-pruning -- mask a neuron that fires fewer than `_MIN_FIRE`
    times in the lead block rather than never) is implemented and left at **1**, the EXACT
    "never fired" test.  It is real: _MIN_FIRE=4 measures 3.7% better here.  It is not
    taken because the bias it buys that with does not shrink with k, is 11x larger in the
    mean and 13x larger on the worst MLP, and cannot be certified on the private split --
    where the only thing separating a 1.3040x from a disaster is that the private MLPs
    behave like these 48.  An exact test needs no such assumption.  Flip the constant to 2
    or 4 to take it; the measurement above is the whole argument either way.

BUDGET SAFETY -- UNCHANGED AND SLIGHTLY IMPROVED
------------------------------------------------
k is still sized as if NOTHING prunes: alive fraction 1.0 at every layer, mask = identity.
The mask can only ever make the pass cheaper, so the sized spend remains a strict upper
bound on the executed spend for ANY input, including a pathological private MLP with no
dead neurons.  Gram halving lowers the worst case, so at the same `_TARGET_UTILISATION`
the same worst-case bound buys ~1.8% more samples.  The overrun cliff (predictions zeroed
AND the multiplier forced to 1.0, ~100,000x) stays unreachable by construction.

The fallback ladder still contains EXACTLY ONE expensive rung.  At ~80% real spend a
second Monte-Carlo attempt after a float32 overflow reaches ~1.9x of budget and is fatal;
the head layers are checked for divergence early so a doomed pass is abandoned ~20% in.

Stress-tested through the real flopscope accounting (effective compute, i.e. FLOPs plus
1e11 * residual wall time, as a fraction of the budget):

    all-positive weights (nothing EVER dead)   0.9121   <-- the binding case, and the one
    normal He init                             0.6595       the sizing is built for
    rank-1 weights (maximal collapse)          0.3625
    weights x10 (float32 overflow)             0.6396
    weights x0.01 (underflow)                  0.5285
    all-negative weights (everything dead)     0.0737
    all-zero weights                           0.0459
    depth 1 / 2 / 5 / 7 / 8 / 9 / 10 / 64      0.297 / 0.489 / 0.914 / 0.912 / 0.912 /
                                               0.912 / 0.872 / 0.548
    width 4 depth 2, width 1 depth 32          0.0004 / 0.0044

Every case returns a finite (depth, width) array and none reaches the budget.

Shallow MLPs (fewer than `_MIN_TAIL_LAYERS` layers after the head split -- e.g. the 2-layer
probe MLP that `whest validate` uses) take the legacy wmc2 chunked path: there is nothing
for a lead-block mask to amortise over, and the special-casing would be untested code on
the contract check.

ONE OBJECTION THIS FILE ANSWERS, BECAUSE IT IS THE OBVIOUS ONE
--------------------------------------------------------------
"An always-on neuron is exactly linear, so treating it as linear must reduce the error."
It does not, and this is the single most important thing to understand about mechanism (C).
For a linear map the sample mean commutes:  mean_s[(W^T y)_i] == (W^T mean_s[y])_i.  The
"exact linear" estimator and the plain sample mean are therefore the SAME NUMBER with the
SAME variance -- bit-identical, not merely close.  Identifying a neuron as linear cannot
reduce the variance of a mean.  Measured here as the bias^2 column of the sweep below
being unmoved to four digits by turning the classification on.

The classification is worth exactly what it costs to skip a column, and nothing more.  The
same disposes of the whole "collapse a run of always-on layers into one matmul" idea: that
needs every neuron in the run to be on, and the always-on fraction peaks at ~0.25 in the
last layer, so the probability is ~0.25^256.  Everything that pays in this file pays
through FLOPs.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# Sub-block for the masked tail of the network.  wmc2's measured knee was 4096, because
# smaller blocks made its per-chunk EXACT detector prune harder.  That trade-off is gone:
# the mask is static, so chunk size changes no FLOP at all and only the Python iteration
# count, which is billed as residual wall time at 1e11 FLOP/s.  Measured over 24 real MLPs,
# identical flops (0.6592 of budget) at every setting, residual wall time
# 4096 -> 7.8 ms | 8192 -> 5.8 ms | 16384 -> 4.9 ms | 32768 -> 6.9 ms.
_CHUNK = 65_536

# Layers 0.._SPLIT_LAYER are ~99% alive, so they are run as one block over the whole
# ensemble: chunking them would cost iterations and prune nothing.
_SPLIT_LAYER = 6

# Samples run at full output width to build the static mask.  Measured knee.
_LEAD = 1024

# Minimum number of firings in the lead block for a neuron to survive the mask.
# 1 == the EXACT "never crossed zero over the lead block" test.  See docstring.
_MIN_FIRE = 1

# Safety band, in units of the column's own sd, for classifying scored-layer columns as
# always-on / always-dead.  None disables the classification.
_FINAL_MARGIN = 1.0

# Minimum lead width for the scored-layer classification to be trusted at all.
_MIN_CLS_LEAD = 256

# The masked tail needs at least this many layers after the head split to be worth it.
_MIN_TAIL_LAYERS = 3

# Upper bound on the ensemble block held in memory at once (samples).  Raised from wmc4's
# 65,536 so that the whole depth-32 ensemble (k = 70,312) is ONE block: the anchor is built
# per block, and splitting the ensemble both shortens the sample it is estimated from and
# leaves a runt block behind.  At 70,312 samples x 256 the block is 72 MB and the head runs
# in 2.7 s per MLP, nowhere near flopscope's 60 s wall-clock limit; the limit was measured
# at 131,072 columns, which only the shallow (large-k) cases reach -- see BUDGET SAFETY.
_BLOCK = 131_072

# Fraction of the FLOP budget targeted by the WORST-CASE (nothing-prunes) cost model.
# See "BUDGET SAFETY".  Real executed spend lands near 0.67 with the mask active.
_TARGET_UTILISATION = 0.92

_MIN_SAMPLES = 512
_MAX_SAMPLES = 400_000

_EIGH_PER_CUBE = 9.0
_MATMUL_PER_CUBE = 2.0
_RNG_F32_PER_ELEMENT = 16.0

# ---------------------------------------------------------------------------
# Layer-1 exact-moment anchor.  See the module docstring for the measurement.
# ---------------------------------------------------------------------------
#
# x is exactly N(0, I), so z_1 = W_0^T x is exactly jointly Gaussian with covariance
# S = W_0^T W_0, and the post-ReLU moments of layer 1 are known in closed form:
#
#     sigma_i = ||W_0[:, i]||
#     mu_i    = sigma_i / sqrt(2 pi)
#     sd_i    = sigma_i * sqrt((pi - 1) / (2 pi))
#     C_ij    = (sigma_i sigma_j / 2pi) [ rho (pi/2 + arcsin rho) + sqrt(1 - rho^2) ] - mu_i mu_j
#
# The ensemble can therefore be pushed onto those moments exactly:
#
#     y1 <- (y1 - mean(y1)) A + mu          A = Cov_emp^{-1/2} Cov_exact^{1/2}   ("full")
#           (y1 - mean(y1)) D + mu          D = diag(sd / sd_emp)                ("diag")
#           (y1 - mean(y1))   + mu                                               ("mean")
#
# `None` disables it and makes this file wmc4.py exactly.  MEASURED: see the docstring.
# "full" is shipped: 1.0832x on the adjusted score, paired t = +5.45 over 1000 real MLPs,
# and it beats "diag" by 1.0393x at t = +3.29, so the cheap variant is NOT the right call.
_ANCHOR = "full"

# Eigenvalue floor for the "full" mode's two symmetric square roots, relative to the
# mean eigenvalue.  Only used when _ANCHOR == "full".
_ANCHOR_EIG_FLOOR = 1e-8

# Derived, not transcribed.  A hand-typed value of _RELU_SD_FACTOR that was wrong in the
# 4th significant figure (0.58252 for 0.58382, -0.22%) measured 24x WORSE than no anchor at
# all: the diagonal anchor multiplies every layer-1 fluctuation by sd_exact / sd_emp, so a
# systematic error in sd_exact is a systematic rescaling of the whole ensemble that ReLU's
# positive homogeneity carries all the way to layer 32.  This is the one place in the file
# where a constant has to be exact, so it is computed.
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_RELU_SD_FACTOR = math.sqrt((math.pi - 1.0) / (2.0 * math.pi))

# Smallest block the anchor will be built from.  Blocks are anchored INDEPENDENTLY, so each
# block's layer-1 mean is exactly mu and hence so is the average over all blocks, whatever
# the block sizes -- but the "full" transform fits n(n+1)/2 = 32,896 free parameters from
# the block's own samples, and a SHORT block therefore overfits badly.
#
# This is not hypothetical.  At _BLOCK = 65,536 the shipped k = 70,024 splits into 65,536 +
# 4,488, and the 4,488-sample runt was being anchored with p/k = 7.3.  MEASURED over 8 real
# MLPs: two blocks 2.2158e-07 against one block 2.1449e-07, a 3.3% penalty for the runt
# alone.  `_BLOCK` is now large enough to hold the whole depth-32 ensemble, and this floor
# is a backstop that simply declines to anchor any short block that still occurs.
_MIN_ANCHOR_BLOCK = 32_768

# ---------------------------------------------------------------------------
# The replicated piece: Strassen for the layer matmuls.
# ---------------------------------------------------------------------------

# Recursion depth.  0 disables Strassen entirely and makes this file wmc3.py.
_LEVELS = 2

# Stop recursing when the small (weight) dimensions fall below this, whatever `_LEVELS`
# says.  Guards the pruned layers, whose blocks are ~190 wide rather than 256.
_MIN_DIM = 48

_MOD = 1 << _LEVELS if _LEVELS > 0 else 1


def _mm_cost(m: int, k: int, n: int, lv: int) -> float:
    """EXACT flopscope 0.10.0 cost of `_smm(A, B)` for A (m,k), B (k,n), float32.

    matmul bills 2*m*k*n - m*n; add/sub and concatenate bill 1 per element.  The three
    overhead terms are the 5 A-side sums (m/2 x k/2), the 5 B-side sums (k/2 x n/2), the 8
    C-side sums (m/2 x n/2), and the two-stage concatenate that reassembles the result
    (2*m*n).  Verified against the meter to the FLOP at 1..5 levels; see the module
    docstring.
    """
    if lv <= 0 or (m & 1) or (k & 1) or (n & 1) or m < _MIN_DIM or k < _MIN_DIM:
        return 2.0 * m * k * n - m * n
    h, i, j = m // 2, k // 2, n // 2
    return (7.0 * _mm_cost(h, i, j, lv - 1)
            + 5.0 * h * i + 5.0 * i * j + 8.0 * h * j + 2.0 * m * n)


def _smm(a, b, lv: int = _LEVELS):
    """A @ B by Strassen-Winograd recursion, falling back to `@` when it cannot apply.

    The fallback conditions are checked at every level, so a block with an odd or small
    dimension simply costs what the plain product costs -- the estimator stays correct for
    every shape, including the 2-layer probe MLP the contract check uses.
    """
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[1])
    if lv <= 0 or (m & 1) or (k & 1) or (n & 1) or m < _MIN_DIM or k < _MIN_DIM:
        return a @ b
    h, i, j = m >> 1, k >> 1, n >> 1
    a11, a12, a21, a22 = a[:h, :i], a[:h, i:], a[h:, :i], a[h:, i:]
    b11, b12, b21, b22 = b[:i, :j], b[:i, j:], b[i:, :j], b[i:, j:]
    r = lv - 1
    m1 = _smm(a11 + a22, b11 + b22, r)
    m2 = _smm(a21 + a22, b11, r)
    m3 = _smm(a11, b12 - b22, r)
    m4 = _smm(a22, b21 - b11, r)
    m5 = _smm(a11 + a12, b22, r)
    m6 = _smm(a21 - a11, b11 + b12, r)
    m7 = _smm(a12 - a22, b21 + b22, r)
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=1)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=1)
    return fnp.concatenate([top, bot], axis=0)


def _round_up_idx(keep, width: int):
    """Grow a boolean keep-set to a multiple of `_MOD` by putting dropped entries back.

    Returns an index array, or None when everything survives.  Adding entries back is
    always safe: a neuron that is computed rather than masked contributes its true value,
    so this can only reduce the lead-block mask's bias.  All arithmetic here is
    width-sized (256 elements), i.e. ~1e-7 of the budget.
    """
    idx = fnp.nonzero(keep)[0]
    a = int(idx.shape[0])
    if a >= width:
        return None
    need = (-a) % _MOD
    if need:
        if a + need >= width:
            return None
        rest = fnp.nonzero(fnp.logical_not(keep))[0]
        idx = fnp.sort(fnp.concatenate([idx, rest[:need]]))
    return idx


def _anchor_fixed_cost(width: int) -> float:
    """Once-per-block small-matrix cost of the anchor (0 when it is off).

    mean/diag: sigma from W_0 (2 n^2), the fused D W_1 (n^2) and the shift W_1^T v (2 n^2).
    full: additionally S = W_0^T W_0 (2 n^3 f32), the arc-cosine kernel (~80 n^2 -- flopscope
    prices `arcsin` at 36 FLOPs/element in float64, MEASURED), two float64 `eigh`
    (18 n^3 each, MEASURED) with their reassembly matmuls (4 n^3 each), A = X Y (4 n^3) and
    A W_1 (2 n^3).  Priced at 60 n^3 + 88 n^2, i.e. 1.01e9 at width 256 -- 0.37% of budget
    against a measured true cost near 8.7e8.
    """
    if _ANCHOR is None:
        return 0.0
    w = float(width)
    small = 8.0 * w * w
    if _ANCHOR != "full":
        return small
    return small + 80.0 * w * w + 60.0 * w ** 3


def _anchor_per_sample_cost(width: int) -> float:
    """Per-sample cost of the anchor (0 when it is off).

    mean: the shift broadcast-add on the layer-2 pre-activation, n.
    diag: + the second-moment reduction over the block, 2 n.
    full: + the empirical Gram of the layer-1 block, 2 n^2 -- one whole extra k n^2 pass,
          which at depth 32 is ~3% of the budget.  That is the entire cost argument.
    """
    if _ANCHOR is None:
        return 0.0
    w = float(width)
    if _ANCHOR == "mean":
        return w
    if _ANCHOR == "diag":
        return 3.0 * w
    # The Gram runs once per block at the block's own length; its per-sample cost is
    # LARGEST at the shortest block the anchor will accept, so pricing there is an upper
    # bound for every block the loop can actually produce.
    return 3.0 * w + _mm_cost(width, _MIN_ANCHOR_BLOCK, width, _LEVELS) / _MIN_ANCHOR_BLOCK


def _fixed_cost(width: int, depth: int, blocks: int = 1) -> float:
    """eigh + whitener assembly + fusion into W_0, plus the one-time index machinery.

    The `48 * depth * width^2` term is deliberate slack for the once-per-MLP costs the
    mask introduces: the `ix_` gathers of the reduced weight blocks (4 * a_in * a_out each,
    worst case 4 * width^2), the scored-layer column gather, the bincount scatters and the
    lead-block classification reductions.  At width 256 / depth 32 it is 1.0e8 FLOPs,
    0.04% of the budget, against a true worst case near 3.6e7.
    """
    cube = float(width) ** 3
    return (_EIGH_PER_CUBE * cube + 2.0 * _MATMUL_PER_CUBE * cube
            + 16.0 * float(width) ** 2
            + 48.0 * float(depth) * float(width) ** 2
            # The anchor's small-matrix work is redone once per _BLOCK, so it is charged per
            # block.  `blocks` is driven to a fixed point in `_sample_count` below.
            + float(blocks) * _anchor_fixed_cost(width))


def _per_sample_cost(width: int, depth: int) -> float:
    """Cost of one sample assuming NOTHING prunes (every alive fraction 1.0).

    A strict upper bound on the executed cost, which is what makes the budget bound hold
    for every possible input.

      draw, float32                 16 * n / 2
      antithetic concat + negate    3 * n
      Gram pass  xh^T xh            n^2            <-- HALVED; x^T x == 2 * xh^T xh
      fused layer 0                 2*n^2 - n  + n (relu) + n (sum)
      layers 1..depth-1             2*n^2 + 8*n
          = matmul (2*n^2 - n) + relu (n) + sum (n) + 6*n of slack covering the row max,
            the nonzero, and the worst affordable gather.  When nothing is dead every
            gather is skipped entirely, so this line is a genuine upper bound.
    """
    w = float(width)
    # Worst case for the layer product: nothing prunes, so every layer is a full
    # (width, width, chunk) product.  `_CHUNK` is the SMALLEST chunk the loop ever runs at
    # full rate, and `_mm_cost` per sample is monotonically decreasing in the chunk length
    # (the A-side sums do not scale with it), so pricing at `_CHUNK` is conservative.
    mm = _mm_cost(width, width, _CHUNK, _LEVELS) / float(_CHUNK)
    draw = _RNG_F32_PER_ELEMENT * w * 0.5 + 3.0 * w
    gram = w * w
    layer0 = mm + 2.0 * w
    rest = float(depth - 1) * (mm + 8.0 * w)
    return draw + gram + layer0 + rest + _anchor_per_sample_cost(width)


def _sample_count(budget: int, width: int, depth: int) -> int:
    per = _per_sample_cost(width, depth)
    # The anchor's fixed cost is per BLOCK and the block count depends on k, so drive the
    # pair to a fixed point.  `blocks` only ever increases here, and a larger `blocks` gives
    # a smaller k, so the loop terminates with blocks >= ceil(k / _BLOCK) -- the bound holds.
    blocks = 1
    for _ in range(4):
        spend = _TARGET_UTILISATION * float(budget) - _fixed_cost(width, depth, blocks)
        k = int(spend / per)
        k = max(_MIN_SAMPLES, min(_MAX_SAMPLES, k))
        need = (k + _BLOCK - 1) // _BLOCK
        if need <= blocks:
            break
        blocks = need
    # Even (the antithetic pairing must be exact) AND a multiple of 2*_MOD, so that every
    # chunk the loop hands to `_smm` -- including the trailing partial one -- has a length
    # divisible by 2^_LEVELS and the recursion never degrades to the plain product.
    step = 2 * _MOD
    return max(step, (k // step) * step)


def _exact_layer1_sigma(w0):
    """sigma_i = ||W_0[:, i]||.  Exact, because x is exactly N(0, I) so z_1 ~ N(0, W_0^T W_0)."""
    return fnp.sqrt(fnp.maximum(fnp.sum(w0 * w0, axis=0),
                                fnp.asarray(1e-30, dtype=fnp.float32)))


def _sym_pow(m, p: float, floor):
    """Symmetric matrix power via eigh, with a floor on the eigenvalues."""
    ev, u = fnp.linalg.eigh(m)
    return (u * fnp.power(fnp.maximum(ev, floor), p)) @ u.T


def _full_anchor_transform(yt, m, sigma, w0, width: int, inv):
    """A = Cov_emp^{-1/2} Cov_exact^{1/2} for the layer-1 post-ReLU block.

    Row-vector convention: the anchored ensemble is Y' = (Y - m) A + mu, so A is applied on
    the RIGHT of a sample row.  Cov_exact is the rectified-Gaussian (arc-cosine) kernel of
    S = W_0^T W_0, which is exact here because z_1 is exactly Gaussian.

    The n^2-sized work is float64 (an inverse square root of a 256x256 matrix is the one
    place in this estimator where float32 is genuinely marginal); the k n^2 Gram that feeds
    it stays float32, which is where all the cost is.  Priced in `_anchor_fixed_cost`.
    """
    # Strassen applies to the Gram too: (w, nb) @ (nb, w).  Measured per-sample cost at
    # width 256, 2 levels: 102,239 against 131,056 plain, a 22% saving on this line.
    ce = fnp.asarray(_smm(yt, yt.T) * inv, dtype=fnp.float64) - fnp.outer(
        fnp.asarray(m, dtype=fnp.float64), fnp.asarray(m, dtype=fnp.float64))
    sig = fnp.asarray(sigma, dtype=fnp.float64)
    s = fnp.asarray(w0.T @ w0, dtype=fnp.float64)
    ss = fnp.outer(sig, sig)
    rho = fnp.clip(s / ss, -1.0, 1.0)
    mu = sig * (1.0 / _SQRT_2PI)
    cx = (ss * (rho * (0.5 * math.pi + fnp.arcsin(rho))
                + fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0)))
          * (1.0 / (2.0 * math.pi))) - fnp.outer(mu, mu)
    # Relative floor, with an absolute backstop so a degenerate MLP (all-zero weights, every
    # layer-1 neuron dead) yields a finite -- if useless -- A rather than inf, and is then
    # rejected by the finiteness check in `_build_anchor`.
    floor = max(_ANCHOR_EIG_FLOOR * float(fnp.trace(cx)) / float(width), 1e-30)
    a = _sym_pow(ce, -0.5, floor) @ _sym_pow(cx, 0.5, floor)
    return fnp.asarray(a, dtype=fnp.float32)


def _build_anchor(yt, sum0, w0, w1, width: int):
    """Fuse the layer-1 exact-moment anchor into W_1.  Returns (W_1', shift, mu) or None.

    The anchored ensemble is never materialised.  With Y' = (Y - m) A + mu the layer-2
    pre-activation is

        Y' W_1 = Y (A W_1) + (mu - m A) W_1

    so the whole anchor costs one n x n weight transform and one broadcast row add.  That
    also keeps `_step`'s exact dead-row pruning EXACT: a row of `yt` that is identically
    zero contributes nothing to `Y (A W_1)` whatever A is, and its constant part is already
    inside the shift.

    `sum0` is the row sum of `yt`, which the estimator computes anyway for the layer-1
    output row, so the empirical mean costs nothing.
    """
    if _ANCHOR is None:
        return None
    nb = int(yt.shape[1])
    if nb < _MIN_ANCHOR_BLOCK:
        return None
    try:
        inv = fnp.asarray(1.0 / float(nb), dtype=fnp.float32)
        m = sum0 * inv
        sigma = _exact_layer1_sigma(w0)
        mu = sigma * fnp.asarray(1.0 / _SQRT_2PI, dtype=fnp.float32)
        if _ANCHOR == "mean":
            w1f, ma = w1, m
        elif _ANCHOR == "diag":
            sd = sigma * fnp.asarray(_RELU_SD_FACTOR, dtype=fnp.float32)
            var = fnp.maximum(fnp.sum(yt * yt, axis=1) * inv - m * m,
                              fnp.asarray(1e-30, dtype=fnp.float32))
            d = sd / fnp.sqrt(var)
            w1f = w1 * fnp.reshape(d, (width, 1))
            ma = m * d
        elif _ANCHOR == "full":
            a = _full_anchor_transform(yt, m, sigma, w0, width, inv)
            w1f = a @ w1
            ma = a.T @ m
        else:
            return None
        shift = w1.T @ (mu - ma)
        if not (bool(fnp.all(fnp.isfinite(w1f))) and bool(fnp.all(fnp.isfinite(shift)))):
            return None
        return w1f, shift, mu
    except Exception:
        return None


def _mean_propagation(mlp: MLP) -> fnp.ndarray:
    """Cheap, extremely robust analytic fallback (~2e8 FLOPs, scores ~9.5e-05).

    Marginal Gaussian assumption per neuron, mean and variance only, no inter-neuron
    correlation.  Far worse than the Monte-Carlo path, but it never touches a matrix
    decomposition and so cannot fail the way the main path might.
    """
    width = mlp.width
    mu = fnp.zeros(width, dtype=fnp.float32)
    var = fnp.ones(width, dtype=fnp.float32)
    rows = []
    for w in mlp.weights:
        w32 = fnp.asarray(w, dtype=fnp.float32)
        mu_pre = w32.T @ mu
        var_pre = fnp.maximum((w32 * w32).T @ var, 1e-20)
        sigma = fnp.sqrt(var_pre)
        alpha = mu_pre / sigma
        cdf = flops.stats.norm.cdf(alpha)
        pdf = flops.stats.norm.pdf(alpha)
        mu = mu_pre * cdf + sigma * pdf
        ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sigma * pdf
        var = fnp.maximum(ez2 - mu * mu, 1e-20)
        rows.append(mu)
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Whitened + antithetic MC with two-sided lead-block pruning of the forward pass."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        # Submission-level RNG.  Per-MLP randomness is seeded from mlp.seed inside
        # predict() so the submission reproduces exactly under the grader's seed.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Monte Carlo, degrading to a ~2e8-FLOP analytic estimate and then to zeros.

        EXACTLY ONE expensive rung.  Retrying the Monte-Carlo pass after a failure would
        double the spend and blow the overrun cliff at this operating point (measured:
        1.88x of budget on an MLP whose activations overflow float32).  There is also
        nothing to gain from such a retry -- the masked pass and the unpruned pass fail
        for the same reasons.
        """
        try:
            out = self._predict_mc(mlp, budget)
            if bool(fnp.all(fnp.isfinite(out))):
                return out
        except Exception:
            pass
        try:
            out = _mean_propagation(mlp)
            if bool(fnp.all(fnp.isfinite(out))):
                return out
        except Exception:
            pass
        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)

    # -- main path ----------------------------------------------------------

    def _predict_mc(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        k = _sample_count(int(budget), width, depth)
        zero = fnp.asarray(0.0, dtype=fnp.float32)

        # Antithetic pairing: X = [xh ; -xh].  Every odd empirical moment is then exactly
        # zero, and the whitening below makes the second moment exactly identity.
        # Measured on 750 real MLPs with common random numbers: 1.122x lower final-layer
        # MSE, paired t = +3.03.  Antithetic sampling ALONE is harmful (0.82x).
        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = rng.standard_normal((half, width), dtype=fnp.float32)
        x = fnp.concatenate([xh, -xh], axis=0)

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f = self._fused_first_layer(xh, w0, k, width)
        weights = [fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights]

        split = min(_SPLIT_LAYER, depth - 1)
        n_tail = depth - 1 - split
        if n_tail < _MIN_TAIL_LAYERS:
            return self._legacy(x, w0f, w0, weights, k, width, depth, split, zero)

        # Accumulators.
        head_tot = None                       # (split+1, width) sample sums
        masks = None                          # layer -> index array or None (= all alive)
        wr = None                             # layer -> reduced weight block
        lead_sums = None                      # layer -> full-width sums from the lead block
        red = None                            # layer -> reduced sums from the masked chunks
        red_fin = None                        # scored layer, reduced to the kink columns
        kink_idx = None
        on_idx = None
        w_on = None

        for base in range(0, k, _BLOCK):
            xb = x[base:base + _BLOCK]                   # basic slice: a view, costs 0

            # --- head: layers 0..split over the whole block ------------------
            # Carried transposed, y as (width, samples): the pruning gather is then a
            # contiguous row gather and the alive-detection max a contiguous row reduction.
            yt = fnp.maximum(_smm(w0f.T, xb.T), zero)
            head = [fnp.sum(yt, axis=1)]
            anchor = _build_anchor(yt, head[0], w0, weights[1], width) if depth > 1 else None
            if anchor is not None:
                # The layer-1 row is then EXACTLY mu, so report mu rather than the sample
                # mean.  Scaled by the block length so the /k at the end undoes it.
                head[0] = anchor[2] * fnp.asarray(float(yt.shape[1]), dtype=fnp.float32)
            for layer in range(1, split + 1):
                if layer == 1 and anchor is not None:
                    yt = self._step(yt, anchor[0], width, zero, anchor[1])
                else:
                    yt = self._step(yt, weights[layer], width, zero)
                head.append(fnp.sum(yt, axis=1))
            head_block = fnp.stack(head, axis=0)
            head_tot = head_block if head_tot is None else head_tot + head_block

            # Early abort on divergence.  An MLP whose activations overflow float32 is
            # already non-finite here, ~7 layers in; bailing now costs ~20% of the pass.
            if not bool(fnp.all(fnp.isfinite(yt))):
                raise ValueError("non-finite activations in the head layers")

            nb = int(yt.shape[1])
            start0 = 0

            # --- lead block: full output width, builds the static mask --------
            if masks is None:
                lead = min(_LEAD, nb)
                masks, lead_sums, zfin = self._lead_pass(
                    yt[:, :lead], weights, split, depth, width, zero)
                start0 = lead

                use_cls = (_FINAL_MARGIN is not None and lead >= _MIN_CLS_LEAD
                           and nb > lead)
                if use_cls:
                    kink_idx, on_idx = self._classify_final(zfin, width)
                wr, w_on = self._reduced_weights(
                    weights, masks, split, depth, width, kink_idx, on_idx)

            # --- masked chunks -----------------------------------------------
            m0 = masks[split]
            for start in range(start0, nb, _CHUNK):
                sl = yt[:, start:start + _CHUNK]
                y = sl if m0 is None else sl[m0]
                part = []
                for layer in range(split + 1, depth - 1):
                    y = fnp.maximum(_smm(wr[layer].T, y), zero)
                    part.append(fnp.sum(y, axis=1))
                if wr[depth - 1] is not None:
                    yf = fnp.maximum(_smm(wr[depth - 1].T, y), zero)
                    sf = fnp.sum(yf, axis=1)
                    red_fin = sf if red_fin is None else red_fin + sf
                if red is None:
                    red = part
                else:
                    red = [a + b for a, b in zip(red, part)]

        inv = fnp.asarray(1.0 / float(k), dtype=fnp.float32)
        rows = [head_tot[i] * inv for i in range(split + 1)]

        for layer in range(split + 1, depth - 1):
            tot = lead_sums[layer]
            if red is not None:
                tot = tot + self._scatter(masks[layer], red[layer - split - 1], width)
            rows.append(tot * inv)

        rows.append(self._final_row(
            lead_sums[depth - 1], red_fin, kink_idx, on_idx, w_on,
            rows[depth - 2], width, inv))

        out = fnp.stack(rows, axis=0)
        if not bool(fnp.all(fnp.isfinite(out))):
            raise ValueError("non-finite Monte-Carlo estimate")
        if tuple(out.shape) != (depth, width):
            raise ValueError("unexpected output shape")
        return out

    # -- pieces -------------------------------------------------------------

    def _lead_pass(self, ylead, weights, split, depth, width, zero):
        """Run the lead samples at FULL output width, recording the alive set per layer.

        Identical arithmetic to wmc2's tail: only the *input* rows are pruned, by the exact
        "silent for every sample in this block" test, which is an algebraic identity.  The
        alive sets it computes on the way are exactly the mask the chunk loop then freezes.
        """
        lead = int(ylead.shape[1])
        masks = {split: self._alive_idx(ylead, width)}
        sums = {}
        y = ylead
        idx = masks[split]
        zfin = None
        for layer in range(split + 1, depth):
            w = weights[layer]
            z = (w.T @ y) if idx is None else (w[idx].T @ y[idx])
            y = fnp.maximum(z, zero)
            sums[layer] = fnp.sum(y, axis=1)
            if layer == depth - 1:
                zfin = z
            else:
                idx = self._alive_idx(y, width)
                masks[layer] = idx
        return masks, sums, zfin

    @staticmethod
    def _alive_idx(y, width: int):
        """Index array of the neurons kept by the mask, or None when every one survives.

        `y` is post-ReLU so its row maxima are >= 0, and a row maximum of exactly 0 means
        that neuron is silent for every sample in the block -- the EXACT test.  With
        `_MIN_FIRE > 1` this loosens to a firing-count threshold, which trades a little
        more saving for bias that does not shrink with k; left at 1 by default.
        """
        if _MIN_FIRE <= 1:
            keep = fnp.max(y, axis=1) > 0.0
        else:
            keep = fnp.count_nonzero(y > 0.0, axis=1) >= _MIN_FIRE
        return _round_up_idx(keep, width)

    @staticmethod
    def _classify_final(zfin, width: int):
        """Split the scored layer's columns into kink / always-on / always-dead.

        `zfin` is the FULL-width pre-activation of the scored layer over the lead block.
        The band is `_FINAL_MARGIN` times each column's own sd, so the classification is
        scale-free and its error rate is set by how far the column sits from the kink,
        not by the absolute size of the activations.
        """
        sd = fnp.std(zfin, axis=1)
        band = sd * fnp.asarray(float(_FINAL_MARGIN), dtype=sd.dtype)
        on = fnp.min(zfin, axis=1) > band
        dead = fnp.max(zfin, axis=1) < -band
        kink = fnp.logical_not(fnp.logical_or(on, dead))
        # Round the kink set up to a multiple of `_MOD` so the scored layer's product is
        # Strassen-eligible too.  The columns pulled back in are on/dead ones that are now
        # simply SAMPLED instead of shortcut, which is the conservative direction.
        kink_idx = _round_up_idx(kink, width)
        if kink_idx is None:
            return None, None
        promoted = fnp.bincount(kink_idx, minlength=width) > 0
        on_idx = fnp.nonzero(fnp.logical_and(on, fnp.logical_not(promoted)))[0]
        return kink_idx, (on_idx if int(on_idx.shape[0]) > 0 else None)

    @staticmethod
    def _reduced_weights(weights, masks, split, depth, width, kink_idx, on_idx):
        """Gather every reduced weight block ONCE, outside the chunk loop.

        `fnp.ix_` costs 4 * a_in * a_out and is cheaper than the two-step `W[i][:, j]`
        gather (measured 1.44e5 vs 3.49e5 for a 200x180 block).
        """
        wr = {}
        for layer in range(split + 1, depth - 1):
            a = masks[layer - 1]
            b = masks[layer]
            w = weights[layer]
            if a is None and b is None:
                wr[layer] = w
            elif b is None:
                wr[layer] = w[a]
            elif a is None:
                wr[layer] = w[:, b]
            else:
                wr[layer] = w[fnp.ix_(a, b)]

        a = masks[depth - 2]
        w = weights[depth - 1]
        if kink_idx is None:
            wr[depth - 1] = w if a is None else w[a]
        elif int(kink_idx.shape[0]) == 0:
            wr[depth - 1] = None
        elif a is None:
            wr[depth - 1] = w[:, kink_idx]
        else:
            wr[depth - 1] = w[fnp.ix_(a, kink_idx)]

        w_on = None if on_idx is None else w[:, on_idx]
        return wr, w_on

    @staticmethod
    def _scatter(idx, vals, width: int):
        """Reduced-index sums back to full width.  `bincount` is 8 FLOPs per entry.

        flopscope arrays are immutable, so an in-place scatter is impossible; doing it once
        at the end (rather than per chunk) keeps this off the hot path entirely.
        """
        if idx is None:
            return vals
        if int(idx.shape[0]) == 0:
            return fnp.zeros(width, dtype=fnp.float32)
        out = fnp.bincount(idx, weights=vals, minlength=width)
        return fnp.asarray(out, dtype=fnp.float32)

    def _final_row(self, lead_fin, red_fin, kink_idx, on_idx, w_on, mean_prev,
                   width: int, inv):
        """Assemble the scored layer.

        kink columns  : lead-block sum + masked-chunk sum, over all k samples.
        always-on     : w_j . mean(y_{L-1}) -- exact, because ReLU is the identity there
                        and the sample mean commutes with a linear map.  This uses ALL k
                        samples through `mean_prev`, so it is not a truncation.
        always-dead   : zero, and the lead-block sum is already exactly zero for them.
        """
        if kink_idx is None:
            tot = lead_fin if red_fin is None else lead_fin + red_fin
            return tot * inv
        kink_lead = lead_fin[kink_idx]
        tot = kink_lead if red_fin is None else kink_lead + red_fin
        out = self._scatter(kink_idx, tot, width) * inv
        if on_idx is not None:
            out = out + self._scatter(on_idx, w_on.T @ mean_prev, width)
        return out

    # -- legacy path, for MLPs too shallow for the mask to amortise ----------

    def _legacy(self, x, w0f, w0, weights, k, width, depth, split, zero):
        totals = None
        for base in range(0, k, _BLOCK):
            xb = x[base:base + _BLOCK]
            yt = fnp.maximum(_smm(w0f.T, xb.T), zero)
            head = [fnp.sum(yt, axis=1)]
            anchor = _build_anchor(yt, head[0], w0, weights[1], width) if depth > 1 else None
            nb0 = int(yt.shape[1])
            if anchor is not None:
                head[0] = anchor[2] * fnp.asarray(float(nb0), dtype=fnp.float32)
            for layer in range(1, split + 1):
                if layer == 1 and anchor is not None:
                    yt = self._step(yt, anchor[0], width, zero, anchor[1])
                else:
                    yt = self._step(yt, weights[layer], width, zero)
                head.append(fnp.sum(yt, axis=1))
            if not bool(fnp.all(fnp.isfinite(yt))):
                raise ValueError("non-finite activations in the head layers")
            nb = int(yt.shape[1])
            tail = None
            for start in range(0, nb, _CHUNK):
                y = yt[:, start:start + _CHUNK]
                part = []
                # `split` is >= 1 whenever depth >= 2, so layer 1 is always in the head above
                # and the anchor never needs re-applying here.
                for layer in range(split + 1, depth):
                    y = self._step(y, weights[layer], width, zero)
                    part.append(fnp.sum(y, axis=1))
                if not part:
                    break
                block = fnp.stack(part, axis=0)
                tail = block if tail is None else tail + block
            head_block = fnp.stack(head, axis=0)
            block = head_block if tail is None else fnp.concatenate([head_block, tail], axis=0)
            totals = block if totals is None else totals + block
        out = totals * fnp.asarray(1.0 / float(k), dtype=fnp.float32)
        if not bool(fnp.all(fnp.isfinite(out))):
            raise ValueError("non-finite Monte-Carlo estimate")
        if tuple(out.shape) != (depth, width):
            raise ValueError("unexpected output shape")
        return out

    @staticmethod
    def _step(yt: fnp.ndarray, w32: fnp.ndarray, width: int, zero, shift=None) -> fnp.ndarray:
        """One ReLU layer on a transposed block, with EXACT input-row pruning.

        Dropping the all-zero rows of `yt` together with the matching rows of `w32` leaves
        w32.T @ yt exactly unchanged -- an algebraic identity, so the estimator is
        untouched and only the MAC count falls.  `shift` is the anchor's constant row (see
        `_build_anchor`); it is added to the pre-activation, and because the dropped rows
        are identically zero the pruning stays exact with the anchor in place.
        """
        alive = _round_up_idx(fnp.max(yt, axis=1) > 0.0, width)
        if alive is not None:
            z = _smm(w32[alive].T, yt[alive])
        else:
            z = _smm(w32.T, yt)
        if shift is not None:
            z = z + fnp.reshape(shift, (int(z.shape[0]), 1))
        return fnp.maximum(z, zero)

    @staticmethod
    def _fused_first_layer(xh: fnp.ndarray, w0: fnp.ndarray, k: int, width: int):
        """Return G^{-1/2} W_0, the whitener fused into the first weight matrix.

        Fusing replaces a k*n^2 transform of the ensemble with an n^3 transform of the
        weights.  The Gram matrix is formed from the antithetic HALF: since x = [xh ; -xh],
        x^T x == 2 * xh^T xh identically, which halves a 3.5%-of-budget line item at zero
        cost in accuracy.  On any numerical trouble we return W_0 unchanged, which is still
        an unbiased estimator -- just with ~1.6x the variance.
        """
        try:
            scale = fnp.asarray(2.0 / float(k), dtype=fnp.float32)
            gram = (xh.T @ xh) * scale
            evals, evecs = fnp.linalg.eigh(gram)
            evals = fnp.maximum(evals, 1e-6)
            whitener = (evecs * fnp.power(evals, -0.5)) @ evecs.T
            fused = whitener @ w0
            if not bool(fnp.all(fnp.isfinite(fused))):
                return w0
            return fused
        except Exception:
            return w0
