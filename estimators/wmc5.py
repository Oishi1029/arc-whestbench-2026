"""wmc4 + the FULL exact layer-1 anchor (exact mean AND exact arc-cosine covariance).

WHAT IS NEW
-----------
The input is exactly N(0, I), so the first pre-activation z_1 = W_0^T x is EXACTLY jointly
Gaussian with covariance S = W_0^T W_0.  Both post-ReLU moments of layer 1 are therefore
closed-form, with no sampling error at all:

    mu_i     = sigma_i / sqrt(2 pi)                              sigma_i = ||W_0[:,i]||
    Sigma_ij = (sigma_i sigma_j / 2 pi) [ rho_ij (pi/2 + arcsin rho_ij)
                                          + sqrt(1 - rho_ij^2) ] - mu_i mu_j
    Sigma_ii = sigma_i^2 (pi - 1) / (2 pi)          (set exactly, not via the formula)

with rho = S / (sigma sigma^T).  The k-sample ensemble Y_1 is pushed onto BOTH:

    Y_1 <- (Y_1 - 1 m^T) A + 1 mu^T ,      A^T Cov_emp A = Sigma ,    m = mean(Y_1)

so the transformed ensemble has EXACTLY the right layer-1 mean and EXACTLY the right
layer-1 covariance, and only chaos of degree >= 3 in Y_1 is left to sample.

Both moment formulas were verified against 4e6 brute-force samples on a real competition
MLP: max |mu_MC - mu_exact| = 1.02e-03 against a 3-sigma MC band of 1.2e-03, and
max |Cov_MC - Sigma| = 2.2e-03 against a 4.5-sigma band of 2.2e-03.  They are right.

THE PRIOR NEGATIVE, AND WHY THIS IS NOT IT
------------------------------------------
RESEARCH.md 5b measured the MEAN-only anchor as neutral (1.05x, paired t = -0.57, 3 of 8
MLPs worse) and a Mehler-series mean+covariance anchor at 1.06x on 8 self-generated MLPs.
This file re-measures the mean-only variant on 40 real competition MLPs on top of the LIVE
statistics (whitened + antithetic) and confirms the negative -- 0.98x, t = -0.47.  The
diagonal-only (mean + exact marginal variance) variant is also neutral, 0.97x, t = -0.63.
The FULL covariance anchor is the one that is different, and it is different by a lot.

THE IDENTITY THAT MAKES IT CHEAP
--------------------------------
Naively the anchor costs a full-ensemble n x n Gram, (Y - 1 m^T)^T (Y - 1 m^T), i.e.
2 n^2 k FLOPs -- 3.8% of the budget, which would eat a third of the gain.  It does not have
to.  The ensemble is antithetic, x = [xh ; -xh], so Z = [Zh ; -Zh] and the post-ReLU block
splits as Yh = ReLU(Zh), Yh' = ReLU(-Zh).  With U = Yh + Yh' = |Zh| and V = Yh - Yh' = Zh,

    Yh^T Yh + Yh'^T Yh' = (U^T U + V^T V) / 2          (algebraic identity, exact)
    sum(Yh) + sum(Yh')  = sum(U)                        (algebraic identity, exact)

so  M2 = (U^T U + V^T V) / (2k)  and  m = sum(U) / k.  Both Grams are over HALF the
ensemble.  Better still, V^T V needs no matmul at all: the whitener is defined by
G = (2/k) xh^T xh and W_0f = G^{-1/2} W_0, so

    V^T V = Zh^T Zh = W_0^T G^{-1/2} (xh^T xh) G^{-1/2} W_0 = (k/2) W_0^T W_0 = (k/2) S

EXACTLY -- the same S the exact moments already need.  The whole empirical covariance
therefore costs one n x n Gram over half the ensemble, half the naive price.  Measured
against the direct computation on a real MLP: mean identity relative error 1.8e-16,
covariance identity relative error 1.6e-06 (that residue is the float32 whitener, four
orders of magnitude below the 1/sqrt(k) ~ 4e-03 error the anchor exists to remove).

AND IT IS FUSED, SO APPLYING IT IS FREE
---------------------------------------
The transform is never materialised.  Layer 2's pre-activation is

    Z_2 = ((Y_1 - 1 m^T) A + 1 mu^T) W_1 = Y_1 (A W_1) + 1 (W_1^T mu - (A W_1)^T m)^T

so A is folded into W_1 once (one n^3 matmul) and the correction is a single broadcast
column add.  Crucially the exact input-row pruning of layer 2 still applies unchanged: a
row of Y_1^T that is identically zero contributes nothing to Y_1 (A W_1) either, and the
offset is a precomputed constant.  Nothing on the hot path changes shape.

Net marginal cost of the whole anchor: one half-ensemble Gram (priced at the plain, i.e.
non-Strassen, rate for safety) + ~5 length-n passes + ~2e8 FLOPs of once-per-MLP linear
algebra.  At width 256 / depth 32 that moves k from 72,712 to 71,208 -- 2.1% -- so the
anchor has to beat 1.021x before it is worth anything at all.

MEASURED -- 250 real competition MLPs of the `full` split, PAIRED
-----------------------------------------------------------------
Every variant sized from the SAME budget by wmc4's own cost model, so a variant that pays
for a covariance pass gets correspondingly fewer samples; common random numbers, so the
MLPs and the base draws pair exactly.  Ground truth is the dataset's own 1e9-sample
`final_means`.

    variant                                  k        mean MSE     ratio    paired t
    whitened + antithetic (wmc4's stats)   72,696    3.1308e-07    1.0000x      --
    + exact mean + exact FULL covariance   71,592    2.9317e-07    1.0679x    +2.64
    the same, with the whitener dropped    71,880    2.8983e-07    1.0802x    +3.01

and, on the same 250 MLPs at n = 40 as a screen, the two variants the prior negative was
actually about:

    + exact mean only                                             0.9815x    -0.47
    + exact mean and exact marginal VARIANCES (diagonal only)      0.9731x    -0.63

So the earlier finding replicates exactly: the mean anchor is neutral, and so is the
diagonal.  It is the OFF-DIAGONAL of the arc-cosine covariance -- the 32,640 exactly-known
pairwise correlations that no amount of marginal information contains -- that carries the
whole effect.  That is why 5b's Mehler-series version found only 1.06x on 8 MLPs and could
not tell it from noise, and why "full" is a materially different claim from "mean".

IN THE ACTUAL ESTIMATOR -- the number that decides whether to ship
------------------------------------------------------------------
The table above is the mechanism in isolation, in plain numpy.  Wired into this file, with
the lead-block mask, the final-layer classification, Strassen and the real flopscope meter
all present, PAIRED against wmc4 (both estimators seed from `mlp.seed`, and k differs by
only 2%, so the two ensembles share ~98% of their draws).

The FIRST 250 MLPs of the split gave +4.7% at paired t = +1.54, which does NOT clear the
|t| > 2.5 bar this project uses.  So the test was run out to the whole 1,000-MLP split in
four independent 250-MLP blocks, exactly as RESEARCH.md 5c did for the antithetic term:

    block         MLPs        adjusted ratio      paired t
    1             1 - 250         1.0472x          +1.54
    2           251 - 500         1.0719x          +2.56
    3           501 - 750         1.0772x          +2.23
    4           751 - 1000        1.0627x          +1.93

Four blocks, four positive.  Pooled over the 750 MLPs the estimator had never been sized
or tuned against (blocks 2-4, exact per-MLP arrays, no summary-statistic approximation):

                              wmc4          this
    adjusted score        2.1819e-07    2.0383e-07    1.0705x   paired t = +3.83
    raw final-layer MSE   3.2928e-07    3.0709e-07    1.0723x   paired t = +3.83
                                                       log-ratio  t = +4.71
    FLOP utilisation          0.6695        0.6699    (worst single MLP 0.767)
    residual wall / MLP        83.9 ms       79.1 ms
    failed MLPs              0 / 750       0 / 750

t = +3.83 on n = 750.  330 of 750 MLPs still got worse -- the anchor buys its average
through the tail rather than across the board, which is why 250 MLPs could not resolve it
and why the mean-only version's "3 of 8 got worse" was never evidence of anything.

OFFICIAL HARNESS, `whest run --split full --n-mlps 200 --runner subprocess`, both
estimators run on the SAME machine on the same afternoon, the same 200 MLPs:

                                       wmc4         this
    Adjusted Final-Layer Score       2.41e-07     2.13e-07
    Raw Final-Layer MSE              3.26e-07     3.10e-07     1.052x
    All-Layers MSE                   6.06e-07     4.99e-07     1.214x
    Mean Compute Utilization           0.7430       0.6884
    Failed MLPs                       0 / 200      0 / 200
    Worst MLP                        1.51e-06     1.10e-06

Read the RAW MSE column, not the adjusted one: the machine was carrying other jobs and the
two runs drew different amounts of residual wall time (wmc4's utilisation came out 0.743
against its own quiet-machine record of 0.664), so the adjusted ratio here is measuring
the load as much as the estimator.  The raw ratio, 1.052x, is load-independent, and it
agrees with the paired flopscope figure for exactly these MLPs -- the harness's first 200
sit inside block 1, which measured 1.049x.  The honest headline number is the 1.0705x from
the 750 MLPs above, where the accounting is exact and the blocks are independent.

The All-Layers MSE falls 21% -- that part is not a statistical claim at all: layer 1's row
is now the closed-form mu, and layers 2 onwards are propagated from an ensemble whose
first two moments are exact.

Dropping the whitener is 1.15% better than keeping it (paired t = +0.86, n.s.) -- the two
mechanisms overlap, as they should, since anchoring y_1's covariance subsumes most of what
whitening x buys.  The whitener is KEPT anyway: it is what makes `Zh^T Zh == (k/2) S`
exact, it costs ~2e8 FLOPs once the Gram is being computed regardless, and it is the
behaviour the estimator falls back to when the anchor declines.  A 1.15% difference at
t = +0.86 is not a reason to give that up.

NUMERICS
--------
A = L^{-T} M^T with L L^T = Cov_emp + ridge I and M M^T = Sigma + ridge I, both by
CHOLESKY (2 n^3 / 3 each, against 9 n^3 for an eigh), then one triangular-free
`linalg.solve`.  Then A^T Cov_emp A = M L^{-1} Cov_emp L^{-T} M^T = M M^T = Sigma.
Cholesky is valid here because both matrices are genuinely SPD: Sigma is an arc-cosine
kernel Gram (measured condition number 12.6 at width 256, He init) and Cov_emp is a
k >> n sample covariance of a full-support ensemble.  The ridge is relative
(1e-6 * trace / n) so it is scale-free, and every failure mode -- a non-SPD matrix, a
non-finite A, an A with an implausibly large entry -- drops the anchor and leaves the
estimator exactly equal to wmc4.  The decompositions run in float64; only the k-sized
Gram is float32.

SAFETY -- every wmc4 property preserved
---------------------------------------
  * canonical seeding, `fnp.random.default_rng(mlp.seed)`, untouched.
  * EXACTLY ONE expensive rung in the fallback ladder; the anchor adds no retry.
  * k is still sized from the runtime `budget` argument against a WORST-CASE cost model in
    which nothing prunes, and the anchor's cost is priced into that model whether or not it
    is actually taken -- so the sized spend remains a strict upper bound.
  * the anchor is skipped, not failed, whenever it does not apply: depth < 3, a shallow
    MLP on the legacy path, a whitener that did not converge, k above `_ANCHOR_MAX` (the
    anchor needs the whole ensemble in one block for the antithetic identity), or any
    numerical trouble.  Skipping only ever makes the pass cheaper.
  * layer 1's reported row becomes mu_exact, which is exact rather than sampled.

Stress-tested through the real flopscope meter, 35 cases.  Every one returns a finite
(depth, width) array; none fails.  COUNTED FLOPS as a fraction of the budget (the residual
wall-time term is omitted here because the measurement machine was running six competing
jobs at load average 9.5, which inflates residual 50x and says nothing about the
estimator):

    depth 8  (the binding case)               0.9107   <- wmc4's own figure here is 0.9170
    all-positive weights (nothing EVER dead)  0.9055   <- wmc4 0.9170
    rank-1 weights (maximal collapse)         0.9047   <- wmc4 0.9144
    depth 10 / 16 / 20 / 23 / 32 / 64         0.884 / 0.808 / 0.766 / 0.726 / 0.642 / 0.519
    depth 1 / 2 / 3 / 5                       0.254 / 0.407 / 0.560 / 0.866
    normal He init, 256x32                    0.6415
    width 250 / 255 / 300 / 512 (odd, big)    0.637 / 0.622 / 0.692 / 0.680
    weights x10 (float32 overflow)            0.6520
    weights x0.01 (underflow)                 0.5214
    all-negative weights (everything dead)    0.1994
    all-zero weights                          0.0609
    W_0 with 256 IDENTICAL columns            0.3503   <- Sigma exactly singular, rho == 1
    W_0 with 128 ZERO columns                 0.6152   <- sigma_i == 0 for half the neurons
    width 4 depth 2, width 1 depth 32         0.0002 / 0.0000

The worst case is 0.911 against wmc4's 0.917: the anchor makes the bound TIGHTER, not
looser, because k is cut by 2.1% while the anchor's own Gram is billed at the plain rate
and executed at the Strassen rate.  The two degenerate-covariance cases are the ones the
anchor has to survive rather than crash on -- identical columns make rho == 1 off the
diagonal and a zero column makes sigma_i == 0, both of which make Sigma singular.  Both
return finite predictions: the ridge carries the first, and the `trace > 0` and
`max|A| < 1e3` guards decline the anchor outright when it cannot be trusted.

--- everything below is wmc4.py's own docstring, unchanged, for provenance ---

REPLICATION of the Strassen leg of AIcrowd forum topic 18106 ("Team SOX", sub 319341,
adjusted 1.551e-07), grafted onto wmc3.py.  Strassen matrix multiplication for the layer
products, with the exact flopscope cost model that sizes it; two-sided (lead-block) pruning
of the forward pass; final-layer output-column classification into kink / always-on /
always-dead; Gram halving via the antithetic identity x^T x == 2 xh^T xh.  See
work/mine/wmc4.py for the full measurement record of those four mechanisms.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_CHUNK = 65_536
_SPLIT_LAYER = 6
_LEAD = 1024
_MIN_FIRE = 1
_FINAL_MARGIN = 1.0
_MIN_CLS_LEAD = 256
_MIN_TAIL_LAYERS = 3
_BLOCK = 65_536
_TARGET_UTILISATION = 0.92
_MIN_SAMPLES = 512
_MAX_SAMPLES = 400_000

_EIGH_PER_CUBE = 9.0
_MATMUL_PER_CUBE = 2.0
_RNG_F32_PER_ELEMENT = 16.0

# --- the anchor -------------------------------------------------------------

# Master switch.  False makes this file bit-identical to wmc4 apart from the sizing.
_ANCHOR = True

# The antithetic identity needs column j and column j + k/2 of the same activation block,
# so the whole ensemble must live in ONE head block.  Above this the anchor is dropped and
# the estimator degrades to wmc4.  Sized to cover the depth-32 operating point (k ~ 71.4k)
# with headroom, and kept below the 131,072 that was measured to trip flopscope's 60 s
# per-op wall-clock limit at two Strassen levels.
_ANCHOR_MAX = 98_304

# Relative ridge on both covariances before the Cholesky factorisations.
_ANCHOR_RIDGE = 1e-6

# Reject the anchor if the fused transform has an entry this large -- the signature of a
# near-singular empirical covariance that the ridge did not catch.
_ANCHOR_MAX_ENTRY = 1.0e3

# Minimum width for the anchor to be worth its n^3 terms.
_ANCHOR_MIN_WIDTH = 32

# Once-per-MLP FLOPs the anchor adds, as a multiple of width^3.  Measured budget:
# S = W_0^T W_0 (2), the closed-form moments (~0.3), two Choleskys (2 x 2/3), the solve
# (~5.4), A @ W_1 (2), the two offset matvecs (~0).  ~10.4; carried at 24 for slack.
_ANCHOR_FIXED_PER_CUBE = 24.0

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_HALF_PI = 0.5 * math.pi
_INV_2PI = 1.0 / (2.0 * math.pi)
_RELU_VAR = (math.pi - 1.0) / (2.0 * math.pi)

# --- Strassen ---------------------------------------------------------------

_LEVELS = 2
_MIN_DIM = 48
_MOD = 1 << _LEVELS if _LEVELS > 0 else 1


def _mm_cost(m: int, k: int, n: int, lv: int) -> float:
    """EXACT flopscope 0.10.0 cost of `_smm(A, B)` for A (m,k), B (k,n), float32."""
    if lv <= 0 or (m & 1) or (k & 1) or (n & 1) or m < _MIN_DIM or k < _MIN_DIM:
        return 2.0 * m * k * n - m * n
    h, i, j = m // 2, k // 2, n // 2
    return (7.0 * _mm_cost(h, i, j, lv - 1)
            + 5.0 * h * i + 5.0 * i * j + 8.0 * h * j + 2.0 * m * n)


def _smm(a, b, lv: int = _LEVELS):
    """A @ B by Strassen-Winograd recursion, falling back to `@` when it cannot apply."""
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
    """Grow a boolean keep-set to a multiple of `_MOD` by putting dropped entries back."""
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


# --- cost model -------------------------------------------------------------

def _fixed_cost(width: int, depth: int, anchor: bool) -> float:
    cube = float(width) ** 3
    base = (_EIGH_PER_CUBE * cube + 2.0 * _MATMUL_PER_CUBE * cube
            + 16.0 * float(width) ** 2
            + 48.0 * float(depth) * float(width) ** 2)
    return base + (_ANCHOR_FIXED_PER_CUBE * cube if anchor else 0.0)


def _per_sample_cost(width: int, depth: int, anchor: bool) -> float:
    """Cost of one sample assuming NOTHING prunes (every alive fraction 1.0).

    The anchor term prices the half-ensemble Gram U^T U at the PLAIN 2-FLOPs/MAC rate --
    2 * width^2 * (k/2) / k = width^2 per sample -- even though `_smm` executes it for
    ~0.79 of that.  Overpricing keeps the bound strict.  The 5 * width covers building
    U = Yh + Yh', its row sum, the broadcast offset add at layer 2 and slack.
    """
    w = float(width)
    mm = _mm_cost(width, width, _CHUNK, _LEVELS) / float(_CHUNK)
    draw = _RNG_F32_PER_ELEMENT * w * 0.5 + 3.0 * w
    gram = w * w
    layer0 = mm + 2.0 * w
    rest = float(depth - 1) * (mm + 8.0 * w)
    extra = (w * w + 5.0 * w) if anchor else 0.0
    return draw + gram + layer0 + rest + extra


def _size(budget: int, width: int, depth: int, anchor: bool) -> int:
    spend = _TARGET_UTILISATION * float(budget) - _fixed_cost(width, depth, anchor)
    k = int(spend / _per_sample_cost(width, depth, anchor))
    k = max(_MIN_SAMPLES, min(_MAX_SAMPLES, k))
    step = 2 * _MOD
    return max(step, (k // step) * step)


def _sample_count(budget: int, width: int, depth: int):
    """Return (k, use_anchor).  The two are decided together so the sizing and the executed
    path always agree: if the anchored k would not fit the single-block requirement we fall
    back to the unanchored k, which is LARGER, so the anchor could not have been taken at
    that k either.  Never the other way round.
    """
    k_plain = _size(budget, width, depth, False)
    if not _ANCHOR or width < _ANCHOR_MIN_WIDTH or depth < 3:
        return k_plain, False
    split = min(_SPLIT_LAYER, depth - 1)
    if depth - 1 - split < _MIN_TAIL_LAYERS:
        return k_plain, False          # legacy path; the anchor is not wired into it
    k_anchor = _size(budget, width, depth, True)
    if k_anchor > _ANCHOR_MAX or k_anchor < 4 * width:
        return k_plain, False
    return k_anchor, True


def _mean_propagation(mlp: MLP) -> fnp.ndarray:
    """Cheap, extremely robust analytic fallback (~2e8 FLOPs, scores ~9.5e-05)."""
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
    """Whitened + antithetic MC, two-sided lead-block pruning, exact layer-1 anchor."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Monte Carlo, degrading to a ~2e8-FLOP analytic estimate and then to zeros.

        EXACTLY ONE expensive rung: retrying the Monte-Carlo pass after a failure would
        double the spend and blow the overrun cliff at this operating point.
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
        k, want_anchor = _sample_count(int(budget), width, depth)
        zero = fnp.asarray(0.0, dtype=fnp.float32)

        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = rng.standard_normal((half, width), dtype=fnp.float32)
        x = fnp.concatenate([xh, -xh], axis=0)

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f, whitened = self._fused_first_layer(xh, w0, k, width)
        weights = [fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights]

        split = min(_SPLIT_LAYER, depth - 1)
        n_tail = depth - 1 - split
        if n_tail < _MIN_TAIL_LAYERS:
            return self._legacy(x, w0f, weights, k, width, depth, split, zero)

        # The antithetic identity U = Yh + Yh' needs the whole ensemble in one block.
        use_anchor = bool(want_anchor and whitened and k <= _ANCHOR_MAX)
        blk = k if use_anchor else _BLOCK

        head_tot = None
        masks = None
        wr = None
        lead_sums = None
        red = None
        red_fin = None
        kink_idx = None
        on_idx = None
        w_on = None
        mu_exact = None

        for base in range(0, k, blk):
            xb = x[base:base + blk]

            yt = fnp.maximum(_smm(w0f.T, xb.T), zero)
            head = [fnp.sum(yt, axis=1)]

            if use_anchor:
                w1a, off1, mu_exact = self._build_anchor(
                    yt, w0, weights[1], k, half, width)
                if w1a is None:
                    use_anchor = False
                    mu_exact = None
                else:
                    yt = self._step_offset(yt, w1a, off1, width, zero)
                    head.append(fnp.sum(yt, axis=1))

            for layer in range(2 if mu_exact is not None else 1, split + 1):
                yt = self._step(yt, weights[layer], width, zero)
                head.append(fnp.sum(yt, axis=1))
            head_block = fnp.stack(head, axis=0)
            head_tot = head_block if head_tot is None else head_tot + head_block

            if not bool(fnp.all(fnp.isfinite(yt))):
                raise ValueError("non-finite activations in the head layers")

            nb = int(yt.shape[1])
            start0 = 0

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
        if mu_exact is not None:
            # Layer 1's mean is known exactly; the ensemble was pushed onto it, so this is
            # what the anchored ensemble's sample mean IS, to float precision.
            rows[0] = mu_exact

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

    # -- the anchor ---------------------------------------------------------

    @staticmethod
    def _exact_moments(w0, width: int):
        """Closed-form (mu, Sigma, S) for y_1 = ReLU(W_0^T x), x ~ N(0, I).

        All of it is n^2 or n^3 work -- ~4e7 FLOPs at width 256, 0.015% of the budget.
        The diagonal of Sigma is written from the exact marginal identity rather than the
        rho = 1 limit of the kernel, which is numerically delicate.
        """
        s32 = w0.T @ w0
        s = fnp.asarray(s32, dtype=fnp.float64)
        dg = fnp.diagonal(s)
        sig = fnp.sqrt(fnp.maximum(dg, 0.0))
        mu = sig * fnp.asarray(1.0 / _SQRT_2PI, dtype=fnp.float64)
        den = sig[:, None] * sig[None, :]
        safe = fnp.maximum(den, 1e-300)
        rho = fnp.clip(s / safe, -1.0, 1.0)
        m2 = (den * _INV_2PI) * (rho * (_HALF_PI + fnp.arcsin(rho))
                                 + fnp.sqrt(fnp.maximum(1.0 - rho * rho, 0.0)))
        cov = m2 - mu[:, None] * mu[None, :]
        diag_exact = fnp.asarray(_RELU_VAR, dtype=fnp.float64) * dg
        cov = cov - fnp.diag(fnp.diagonal(cov)) + fnp.diag(diag_exact)
        return mu, cov, s

    def _build_anchor(self, yt, w0, w1, k: int, half: int, width: int):
        """Fold the exact layer-1 mean+covariance anchor into W_1.

        Returns (A W_1, offset, mu_exact), or (None, None, None) to decline.  Declining is
        always safe: the caller then runs wmc4's ordinary layer-1 step.
        """
        try:
            # U = Yh + Yh' = |Zh|.  Both halves are basic slices, i.e. views.
            u = yt[:, :half] + yt[:, half:]
            m32 = fnp.sum(u, axis=1) * fnp.asarray(1.0 / float(k), dtype=fnp.float32)
            uu = fnp.asarray(_smm(u, u.T), dtype=fnp.float64)

            mu, cov_ex, s = self._exact_moments(w0, width)
            m = fnp.asarray(m32, dtype=fnp.float64)

            # M2 = (U^T U + Zh^T Zh) / 2k  with  Zh^T Zh = (k/2) S  exactly.
            m2 = (uu + s * fnp.asarray(0.5 * float(k), dtype=fnp.float64)) \
                * fnp.asarray(1.0 / (2.0 * float(k)), dtype=fnp.float64)
            cov_emp = m2 - m[:, None] * m[None, :]

            eye = fnp.eye(width, dtype=fnp.float64)
            r_emp = float(fnp.trace(cov_emp)) * (_ANCHOR_RIDGE / float(width))
            r_ex = float(fnp.trace(cov_ex)) * (_ANCHOR_RIDGE / float(width))
            if not (r_emp > 0.0 and r_ex > 0.0):
                return None, None, None
            lo = fnp.linalg.cholesky(cov_emp + eye * r_emp)
            hi = fnp.linalg.cholesky(cov_ex + eye * r_ex)
            # A = L^{-T} M^T  =>  A^T Cov_emp A = M M^T = Sigma.
            a = fnp.linalg.solve(lo.T, hi.T)
            if not bool(fnp.all(fnp.isfinite(a))):
                return None, None, None
            if float(fnp.max(fnp.abs(a))) > _ANCHOR_MAX_ENTRY:
                return None, None, None

            a32 = fnp.asarray(a, dtype=fnp.float32)
            mu32 = fnp.asarray(mu, dtype=fnp.float32)
            w1a = a32 @ w1
            off = w1.T @ mu32 - w1a.T @ m32
            if not (bool(fnp.all(fnp.isfinite(w1a))) and bool(fnp.all(fnp.isfinite(off)))):
                return None, None, None
            return w1a, off, mu32
        except Exception:
            return None, None, None

    @staticmethod
    def _step_offset(yt, w1a, off, width: int, zero):
        """Layer 2's pre-activation from the ANCHORED layer-1 ensemble, without ever
        materialising it.  Exact input-row pruning is unaffected: an identically-zero row
        of `yt` contributes nothing to `yt^T (A W_1)` either.
        """
        alive = _round_up_idx(fnp.max(yt, axis=1) > 0.0, width)
        if alive is not None:
            z = _smm(w1a[alive].T, yt[alive])
        else:
            z = _smm(w1a.T, yt)
        return fnp.maximum(z + off[:, None], zero)

    # -- wmc4 pieces, unchanged ---------------------------------------------

    def _lead_pass(self, ylead, weights, split, depth, width, zero):
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
        if _MIN_FIRE <= 1:
            keep = fnp.max(y, axis=1) > 0.0
        else:
            keep = fnp.count_nonzero(y > 0.0, axis=1) >= _MIN_FIRE
        return _round_up_idx(keep, width)

    @staticmethod
    def _classify_final(zfin, width: int):
        sd = fnp.std(zfin, axis=1)
        band = sd * fnp.asarray(float(_FINAL_MARGIN), dtype=sd.dtype)
        on = fnp.min(zfin, axis=1) > band
        dead = fnp.max(zfin, axis=1) < -band
        kink = fnp.logical_not(fnp.logical_or(on, dead))
        kink_idx = _round_up_idx(kink, width)
        if kink_idx is None:
            return None, None
        promoted = fnp.bincount(kink_idx, minlength=width) > 0
        on_idx = fnp.nonzero(fnp.logical_and(on, fnp.logical_not(promoted)))[0]
        return kink_idx, (on_idx if int(on_idx.shape[0]) > 0 else None)

    @staticmethod
    def _reduced_weights(weights, masks, split, depth, width, kink_idx, on_idx):
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
        if idx is None:
            return vals
        if int(idx.shape[0]) == 0:
            return fnp.zeros(width, dtype=fnp.float32)
        out = fnp.bincount(idx, weights=vals, minlength=width)
        return fnp.asarray(out, dtype=fnp.float32)

    def _final_row(self, lead_fin, red_fin, kink_idx, on_idx, w_on, mean_prev,
                   width: int, inv):
        if kink_idx is None:
            tot = lead_fin if red_fin is None else lead_fin + red_fin
            return tot * inv
        kink_lead = lead_fin[kink_idx]
        tot = kink_lead if red_fin is None else kink_lead + red_fin
        out = self._scatter(kink_idx, tot, width) * inv
        if on_idx is not None:
            out = out + self._scatter(on_idx, w_on.T @ mean_prev, width)
        return out

    def _legacy(self, x, w0f, weights, k, width, depth, split, zero):
        totals = None
        for base in range(0, k, _BLOCK):
            xb = x[base:base + _BLOCK]
            yt = fnp.maximum(_smm(w0f.T, xb.T), zero)
            head = [fnp.sum(yt, axis=1)]
            for layer in range(1, split + 1):
                yt = self._step(yt, weights[layer], width, zero)
                head.append(fnp.sum(yt, axis=1))
            if not bool(fnp.all(fnp.isfinite(yt))):
                raise ValueError("non-finite activations in the head layers")
            nb = int(yt.shape[1])
            tail = None
            for start in range(0, nb, _CHUNK):
                y = yt[:, start:start + _CHUNK]
                part = []
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
    def _step(yt: fnp.ndarray, w32: fnp.ndarray, width: int, zero) -> fnp.ndarray:
        alive = _round_up_idx(fnp.max(yt, axis=1) > 0.0, width)
        if alive is not None:
            return fnp.maximum(_smm(w32[alive].T, yt[alive]), zero)
        return fnp.maximum(_smm(w32.T, yt), zero)

    @staticmethod
    def _fused_first_layer(xh: fnp.ndarray, w0: fnp.ndarray, k: int, width: int):
        """Return (G^{-1/2} W_0, whitened_ok).

        `whitened_ok` is the licence for the anchor's `Zh^T Zh == (k/2) S` identity: it
        holds only if the whitener really is G^{-1/2} with no eigenvalue clamped, so the
        floor is checked rather than silently applied.
        """
        try:
            scale = fnp.asarray(2.0 / float(k), dtype=fnp.float32)
            gram = (xh.T @ xh) * scale
            evals, evecs = fnp.linalg.eigh(gram)
            ok = bool(fnp.min(evals) > 1e-3)
            evals = fnp.maximum(evals, 1e-6)
            whitener = (evecs * fnp.power(evals, -0.5)) @ evecs.T
            fused = whitener @ w0
            if not bool(fnp.all(fnp.isfinite(fused))):
                return w0, False
            return fused, ok
        except Exception:
            return w0, False
