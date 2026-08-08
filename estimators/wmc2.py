"""Whitened + antithetic Monte Carlo with EXACT dead-neuron pruning of the forward pass.

WHAT THIS IS
------------
Statistically identical to `work/mine/wmc.py` (whitened, antithetic, float32 Monte Carlo).
The only change is that the forward pass skips multiply-accumulates whose result is
*provably* zero, which buys ~1.15x more samples per FLOP and moves the adjusted score by
the same factor.  The error scaling is unchanged: MSE = C/k exactly, p = 1.  This is a
constant-factor cost reduction, not a better estimator.

MEASURED (official harness, `whest run --split full --n-mlps 100 --runner subprocess`):

                                  this        wmc.py (incumbent)
    adjusted final-layer score    3.16e-07    4.14e-07
    raw final-layer MSE           3.95e-07    4.14e-06
    all-layers MSE                7.36e-07    7.07e-06
    compute utilisation           0.8045      0.0997
    residual wall time / MLP      29.8 ms     1.4 ms
    failed MLPs                   0 / 100     0 / 100

The 1.31x above is inflated by 100-MLP sampling noise in the raw MSE.  The robust figure
comes from a 1,000-MLP paired common-random-numbers run-off on the full split (each MLP's
small-k ensemble is a nested subset of its large-k ensemble):

    raw MSE  wmc.py k=6,150   3.8029e-06     (harness on 1,000 MLPs: 3.84e-06)
    raw MSE  this   k=58,784  3.9508e-07     (exact-1/k prediction: 3.9786e-07)
    adjusted ratio                1.15x
    paired t (adjusted)          +4.33       -> real, not noise
    log(MSE*k) old - new    +0.013 +- 0.023  -> 1/k scaling holds exactly, t = +0.58

So: a real, statistically confirmed, algebraically exact 1.15x -- and 6.3x short of the
5e-08 raw-MSE bar this design was aimed at.  It does not close that gap and cannot: see
"WHAT THIS DOES NOT DO" at the bottom.

THE IDENTITY
------------
Let Y_l be the post-ReLU activation block for a block of samples at layer l, and
A_l = { j : max_t Y_l[t, j] > 0 } the neurons that fire for at least one sample in the
block.  For j not in A_l the whole column is identically zero, so

        Y_l @ W_{l+1}  ==  Y_l[:, A_l] @ W_{l+1}[A_l, :]

*exactly* -- an algebraic identity on the block, not an approximation.  Zero bias, zero
variance change; only the FLOP count moves.  Verified: measured final-layer MSE is
3.488e-07 for the pruned path and 3.488e-07 for the unpruned path at identical k
(agreement to 4 significant figures; the residual difference is float32 summation order).

WHY IT PAYS
-----------
The effective rank of the activation matrix collapses with depth in these networks, and
with it the fraction of neurons that ever fire.  Measured over 8 real competition MLPs at
k = 59,400, mean alive fraction per layer:

    block  |  L0    L4    L8    L12   L16   L20   L24   L28   L30  |  mean(L0..30)
     1024  | 1.00  0.98  0.91  0.84  0.81  0.77  0.74  0.72  0.72  |  0.836
     4096  | 1.00  0.99  0.94  0.87  0.83  0.80  0.77  0.75  0.75  |  0.859
    59400  | 1.00  1.00  0.96  0.91  0.87  0.84  0.81  0.78  0.78  |  0.889

Smaller blocks prune harder (a neuron only has to stay silent over CH samples) but need
more Python-level iterations, and iteration overhead is billed as residual wall time at
1e11 FLOP/s -- which at this budget is 2.72e9 FLOPs per 27 ms.  Two measured consequences
shape the implementation:

  1. CH = 4096 is the knee.  Measured effective-compute fraction over 4 real MLPs:
       CH=2048 -> 0.8399 | CH=4096 -> 0.8352 | CH=8192 -> 0.8387 | CH=16384 -> 0.8414
     (against 0.9489 unpruned).  The published design's CH=1024 is on the wrong side of it.

  2. Layers 0-6 are ~99% alive, so chunking them buys nothing and costs iterations.  They
     are run as one block over the whole ensemble and only layers 7..31 are chunked.
     Measured at CH=4096: residual 49.1 ms without the split, 36.4 ms with it.
     Effective-compute fraction: 0.8394 without the split, 0.8352 with it.

  3. The forward pass is carried TRANSPOSED, y as (width, samples).  Then the pruning
     gather is a contiguous ROW gather and the alive-detection max is a contiguous row
     reduction, instead of strided column operations.  Same FLOPs, measured ~25% less
     residual wall time (CH=8192: 35.1 ms row-major vs 26.5 ms transposed).

BUDGET SAFETY -- THE NON-NEGOTIABLE PART
----------------------------------------
k is sized as if NOTHING prunes (alive fraction 1.0), plus the detection and gather
overhead.  The executed spend is therefore bounded above by the sized spend for *any*
input, including a pathological private MLP with no dead neurons.  The overrun cliff --
predictions zeroed and multiplier forced to 1.0, ~100,000x -- is unreachable by
construction, and no property of the public MLPs is baked into the sizing.

Because the score is flat in k above the floor, sizing conservatively costs essentially
nothing: the pruning gain is banked as a *lower multiplier* at unchanged k rather than as
extra samples, and the two are worth the same.  Measured: sizing target 0.95 gives
k=58,784 at utilisation 0.8345, target 0.92 gives k=56,926 at utilisation 0.8045; the
score coefficient u/k is 1.4197e-05 and 1.4190e-05 respectively -- indistinguishable.  The
safety is free, so take it.

Adversarial cases, run through the real cost model (see the report accompanying this file):

    all-positive weights (nothing EVER dead)   flops 0.910 B, effective 0.920 B
    rank-1 weights (maximal collapse)          flops 0.524 B, effective 0.535 B
    all-zero weights                           flops 0.061 B, effective 0.068 B
    weights x10 (float32 overflow)             flops 0.871 B, effective 0.879 B

None exceeds the budget.  The binding case is the first, and it is the one the sizing is
built for.

ONE TRAP WORTH RECORDING.  An earlier revision kept `wmc.py`'s fallback ladder, whose
second rung re-ran Monte Carlo with pruning disabled.  At 10% of budget that is harmless;
at this operating point it is fatal -- an MLP whose activations overflow float32 makes the
first attempt raise, and the pair of attempts reached 1.88x of budget, which zeroes the
predictions and forces the multiplier to 1.0.  The ladder now contains exactly one
expensive rung, and the head layers are checked for divergence early so a doomed pass is
abandoned ~20% in rather than at the end.

WHY FULL BUDGET AND NOT THE 10% FLOOR
--------------------------------------
The score is flat above the floor for any 1/k estimator, so the level of utilisation is
free -- but the pruning saving is only *visible* above the floor.  At the floor the
multiplier is clamped to 0.1 and a cheaper forward pass buys nothing unless k is sized
optimistically, which reintroduces exactly the overrun risk this design removes.  Running
at full budget is also far more robust to residual wall time, because 1e11*R is an
absolute charge: at a 2.3e11 spend a 36 ms residual costs 1.6%, at a 2.7e10 spend it would
cost 13%.

COST MODEL, measured directly against flopscope 0.10.0 (numpy 2.2.6), float32, n = 256:

    (M,K)@(K,N) matmul            2*M*N*K - M*N        (NOT 2*M*N*K)
    elementwise / reduction       1 FLOP per element
    fancy-index gather            4 FLOPs per element gathered      <-- NOT 1
    nonzero(v), v of length n     n
    eigh(n,n)                     9*n^3
    standard_normal, float64      32 FLOPs per element
    standard_normal, float32      16 FLOPs per element              <-- half price
    concatenate                   1 FLOP per element written
    basic slice x[a:b]            0

matmul, dot, einsum and tensordot are all priced identically, in float32, float64 and
float16 alike per MAC (float64 is 2x float32; float16 gets no discount).  There is no
cheap shape and no cheap alias.

WHAT THIS DOES NOT DO
---------------------
It does not change the error scaling, and the 1,000-MLP run-off above confirms MSE = C/k
to within t = +0.58 of exact.  At the full budget that lands at 3.95e-07 raw MSE against
a 5e-08 target -- 7.9x short.  Closing that requires an estimator whose error falls faster
than 1/k, which no rearrangement of this Monte-Carlo pass can provide: the pruning is a
cost reduction and nothing else.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# Sub-block for the pruned tail of the network.  Measured knee; see module docstring.
_CHUNK = 4096

# Layers 0.._SPLIT_LAYER are ~99% alive, so they are run as one block over the whole
# ensemble: chunking them would cost iterations and prune nothing.
_SPLIT_LAYER = 6

# Upper bound on the block held in memory at once (samples).  At the competition budget
# k ~ 5.9e4 < _BLOCK, so the head runs in a single block; the cap only exists so memory
# stays bounded if the budget or shape ever changes.
_BLOCK = 65_536

# Fraction of the FLOP budget targeted by the WORST-CASE (nothing-prunes) cost model.
# Real executed spend lands near 0.80.  See "BUDGET SAFETY".
#
# 0.92 rather than 0.99: above the floor the score depends on k only through how the
# fixed cost and the residual-time charge amortise over it, which together are ~2% of the
# spend, so 7 points of unclaimed headroom cost ~0.3% of score.  In exchange the
# worst-case residual wall-time margin becomes (1 - 0.92) * 2.72e11 / 1e11 = 218 ms
# against a measured 42 ms, i.e. the grader can be 5x slower than this machine and still
# not reach the overrun cliff even on an MLP where nothing at all is dead.
_TARGET_UTILISATION = 0.92

_MIN_SAMPLES = 512
_MAX_SAMPLES = 400_000

_EIGH_PER_CUBE = 9.0
_MATMUL_PER_CUBE = 2.0
_RNG_F32_PER_ELEMENT = 16.0


def _fixed_cost(width: int) -> float:
    """eigh + whitener assembly + fusion into W_0, plus slack for the small vector ops."""
    cube = float(width) ** 3
    return _EIGH_PER_CUBE * cube + 2.0 * _MATMUL_PER_CUBE * cube + 16.0 * float(width) ** 2


def _per_sample_cost(width: int, depth: int) -> float:
    """Cost of one sample assuming the pruning saves NOTHING (alive fraction 1.0).

    A strict upper bound on the executed cost, which is what makes the budget bound hold
    for every possible input.

      draw, float32                 16 * n / 2
      antithetic concat + negate    3 * n
      Gram pass  x^T x              2 * n^2        (the -n^2/k term dropped: conservative)
      fused layer 0                 2*n^2 - n  + n (relu) + n (sum)
      layers 1..depth-1             2*n^2 + 8*n
          = matmul (2*n^2 - n) + max (n) + relu (n) + sum (n) + 6*n of slack covering the
            worst affordable gather (4*n for the activation block plus the weight-row
            gather amortised over the chunk).  When nothing is dead the gather is skipped
            entirely, so this line is a genuine upper bound.
    """
    w = float(width)
    draw = _RNG_F32_PER_ELEMENT * w * 0.5 + 3.0 * w
    gram = 2.0 * w * w
    layer0 = 2.0 * w * w + w
    rest = float(depth - 1) * (2.0 * w * w + 8.0 * w)
    return draw + gram + layer0 + rest


def _sample_count(budget: int, width: int, depth: int) -> int:
    spend = _TARGET_UTILISATION * float(budget) - _fixed_cost(width)
    k = int(spend / _per_sample_cost(width, depth))
    k = max(_MIN_SAMPLES, min(_MAX_SAMPLES, k))
    return (k // 2) * 2          # even, so the antithetic pairing is exact


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
    """Whitened + antithetic MC whose forward pass skips provably-zero neurons."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        # Submission-level RNG.  Per-MLP randomness is seeded from mlp.seed inside
        # predict() so the submission reproduces exactly under the grader's seed.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        """Monte Carlo, degrading to a ~2e8-FLOP analytic estimate and then to zeros.

        The fallback ladder deliberately contains only ONE expensive path.  Retrying the
        Monte-Carlo pass after a failure -- which is what `work/mine/wmc.py` effectively
        allows, harmlessly, because it runs at 10% of budget -- would double the spend and
        blow the overrun cliff at this operating point.  Measured: an MLP whose activations
        overflow float32 makes the main path raise, and a retry ladder with a second
        Monte-Carlo attempt reaches 1.88x of budget.  There is also nothing to gain from
        such a retry: the pruned pass is an algebraic identity of the unpruned one, so
        anything that breaks one breaks the other.
        """
        try:
            out = self._predict_mc(mlp, budget, prune=True)
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

    def _predict_mc(self, mlp: MLP, budget: int, prune: bool = True) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        k = _sample_count(int(budget), width, depth)
        zero = fnp.asarray(0.0, dtype=fnp.float32)

        # Antithetic pairing: X = [xh ; -xh].  The empirical mean and every odd moment are
        # then exactly zero, so no centring is needed; the whitening below makes the second
        # moment exactly identity too.  Measured on 750 real competition MLPs with common
        # random numbers: 1.122x lower final-layer MSE, paired t = +3.03.  Antithetic
        # sampling ALONE is harmful here (0.82x) -- it only pays once the ensemble is also
        # whitened.
        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = rng.standard_normal((half, width), dtype=fnp.float32)
        x = fnp.concatenate([xh, -xh], axis=0)

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        w0f = self._fused_first_layer(x, w0, k, width)
        weights = [fnp.asarray(w, dtype=fnp.float32) for w in mlp.weights]

        split = min(_SPLIT_LAYER, depth - 1)
        chunk = _CHUNK if prune else _BLOCK
        totals = None

        for base in range(0, k, _BLOCK):
            xb = x[base:base + _BLOCK]                       # basic slice: a view, costs 0

            # --- head: layers 0..split over the whole block ------------------
            # Carried transposed, y as (width, samples): the pruning gather is then a
            # contiguous row gather and the alive-detection max a contiguous row reduction.
            yt = fnp.maximum(w0f.T @ xb.T, zero)
            head = [fnp.sum(yt, axis=1)]
            for layer in range(1, split + 1):
                yt = self._step(yt, weights[layer], width, zero, prune)
                head.append(fnp.sum(yt, axis=1))

            # Early abort on divergence.  An MLP whose activations grow fast enough to
            # overflow float32 will already be non-finite here, ~7 layers in; bailing now
            # costs ~20% of the pass instead of all of it.  Costs 2*width*samples.
            if not bool(fnp.all(fnp.isfinite(yt))):
                raise ValueError("non-finite activations in the head layers")

            # --- tail: layers split+1..depth-1, chunked so pruning bites ------
            nb = int(yt.shape[1])
            tail = None
            for start in range(0, nb, chunk):
                y = yt[:, start:start + chunk]
                part = []
                for layer in range(split + 1, depth):
                    y = self._step(y, weights[layer], width, zero, prune)
                    part.append(fnp.sum(y, axis=1))
                # A shallow MLP (depth <= _SPLIT_LAYER + 1) leaves the tail range empty.
                # fnp.stack([]) raises, which used to send the whole main path into the
                # mean-propagation fallback -- silently, and in particular during
                # `whest validate`, whose probe MLP is 2 layers deep. The competition
                # shape is depth 32 so this never fired in scoring, but it meant the
                # main path was never actually exercised by the contract check.
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
    def _step(yt: fnp.ndarray, w32: fnp.ndarray, width: int, zero, prune: bool) -> fnp.ndarray:
        """One ReLU layer on a transposed activation block, dead neurons skipped.

        `yt` is (width, samples) and post-ReLU, so its row maxima are >= 0 and a row
        maximum of exactly 0 means that neuron is silent for every sample in the block.
        Dropping those rows of `yt` together with the matching rows of `w32` leaves
        w32.T @ yt exactly unchanged -- an algebraic identity, so the estimator is
        untouched and only the MAC count falls.
        """
        if prune:
            alive = fnp.nonzero(fnp.max(yt, axis=1))[0]
            if int(alive.shape[0]) < width:
                return fnp.maximum(w32[alive].T @ yt[alive], zero)
        return fnp.maximum(w32.T @ yt, zero)

    @staticmethod
    def _fused_first_layer(x: fnp.ndarray, w0: fnp.ndarray, k: int, width: int) -> fnp.ndarray:
        """Return G^{-1/2} W_0, the whitener fused into the first weight matrix.

        Fusing replaces a k*n^2 transform of the ensemble with an n^3 transform of the
        weights.  On any numerical trouble we return W_0 unchanged, which is still an
        unbiased estimator -- just with ~1.6x the variance.
        """
        try:
            gram = (x.T @ x) / fnp.asarray(float(k), dtype=fnp.float32)
            evals, evecs = fnp.linalg.eigh(gram)
            evals = fnp.maximum(evals, 1e-6)
            whitener = (evecs * fnp.power(evals, -0.5)) @ evecs.T
            fused = whitener @ w0
            if not bool(fnp.all(fnp.isfinite(fused))):
                return w0
            return fused
        except Exception:
            return w0
