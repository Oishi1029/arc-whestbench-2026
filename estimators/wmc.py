"""Whitened (moment-matched) Monte-Carlo estimator for WhestBench.

RATIONALE (all figures measured locally on 2026-08-04; see work/RESEARCH.md)
---------------------------------------------------------------------------
The score is  final_layer_mse * max(0.1, effective_compute / flop_budget).

Because the multiplier FLOORS at 0.1, every FLOP up to 10% of the budget is
free, and beyond that point the score is exactly flat for any estimator whose
MSE falls as 1/k: score = (C/k) * (c*k/B) = C*c/B.  So the optimum sits right
at the 10% line, and the shipped covariance-propagation baseline -- which spends
1.33% of the budget and scores 8.37e-06 -- is leaving ~7.5x of free compute
unused.  That, not a cleverer series, is the main opening.

Two measured facts drove this design:

  1. Analytic covariance propagation has hit a wall near 6e-05 final-layer MSE.
     Making its post-ReLU covariance update EXACT does not help: the update
     Cov(ReLU(u_i), ReLU(u_j)) = sum_k (rho^k / k!) c_k(i) c_k(j)  (Mehler)
     is exact for jointly Gaussian pre-activations, and the shipped "gain"
     heuristic Phi(a_i)Phi(a_j)cov_ij is precisely its order-1 truncation.
     Carrying the expansion to order 8 reproduces brute-force pair covariances
     to 4 decimals yet leaves the 32-layer result slightly WORSE (6.5e-05 vs
     5.9e-05).  The residual error is joint non-Gaussianity compounding with
     depth, which no pairwise formula can fix.
  2. Plain Monte Carlo at the free-compute ceiling beats it by ~15x, and unlike
     any analytic scheme it is unbiased -- its error is pure variance.

So: spend the free allowance on sampling, then attack the variance.

VARIANCE REDUCTION -- moment-matched inputs.  The input law N(0, I) is known
exactly, so the sample ensemble is forced to match it exactly in its first two
moments:  X <- (X - mean X) G^{-1/2},  G = cov(X).  An affine re-standardisation
of an i.i.d. Gaussian ensemble is still a valid N(0, I) ensemble, but its first-
and second-moment sampling error is now identically zero.  Measured: 1.56x lower
final-layer MSE than plain MC at equal cost, and markedly lower spread across
MLPs (worst-case 6.8e-06 vs 1.4e-05).

The whitening transform is FUSED into the first weight matrix -- X @ (G^{-1/2} W_0)
rather than (X @ G^{-1/2}) @ W_0 -- which turns a k*n^2 pass into an n^3 one and
buys back ~3% more samples.

PRECISION.  flopscope 0.10.0 prices arithmetic by dtype; measured directly,
float32 matmul costs 2.0 FLOPs/MAC against float64's 4.0.  Running in float32
therefore doubles the affordable sample count, halving the Monte-Carlo variance.
256-term dot products and means over ~6*10^3 samples sit comfortably inside
float32 accuracy, and the whitened Gram matrix is well conditioned by
construction (eigenvalues within roughly (1 +- 2 sqrt(n/k))^2).

BUDGET SAFETY.  Exceeding the budget zeroes the prediction AND forces the
multiplier to 1.0 -- a ~100,000x penalty -- so the sample count is derived from
the `budget` argument at run time against a cost model calibrated by direct
measurement against flopscope 0.10.0 (see _CALIBRATION below), targeted at 9.5%,
and every stage degrades to a cheaper estimator rather than raising.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# ---------------------------------------------------------------------------
# Cost model, calibrated by direct measurement against flopscope 0.10.0
# (numpy 2.2.6 backend), all in float32, width n = 256:
#
#   matmul                  2.0 FLOPs per multiply-accumulate
#   elementwise (relu, +)   1.0 FLOP  per element
#   reductions (sum, mean)  1.0 FLOP  per element read
#   eigh(n, n)              150,994,944  ==  9 * n^3
#   (n,n) @ (n,n)            33,488,896  ==  2 * n^3
#
# Per sample we pay one Gram pass, one fused first layer, and (depth - 1)
# remaining layers -- i.e. (depth + 1) passes of 2*n^2 -- plus a relu and a
# running sum per layer (2*n each).
# ---------------------------------------------------------------------------
_MAC = 2.0
_EIGH_PER_CUBE = 9.0
_MATMUL_PER_CUBE = 2.0

# Sit just under the 10% floor. BELOW the floor the score is 0.1 * C/k with k proportional to
# utilisation, so score ~ 0.1/u: every unused percent of the free allowance costs a proportional
# percent of score. ABOVE it the score is flat, and overshooting costs only ~1% per 1% (unlike a
# *budget* overrun, which zeroes the prediction and forces multiplier 1.0 -- ~100,000x).
# The penalty is therefore mild and near-symmetric around u = 0.10, so aim close to it.
#
# Measured residual wall time is 1.0 ms/MLP = 0.037% of budget, so 0.1% of headroom covers a
# grader core ~3x slower than this machine and 0.4% covers ~10x slower.
_TARGET_UTILISATION = 0.099

_MIN_SAMPLES = 512
_MAX_SAMPLES = 400_000


def _fixed_cost(width: int) -> float:
    """One eigh plus two (n,n)@(n,n) products (the whitener and its fusion)."""
    cube = float(width) ** 3
    return _EIGH_PER_CUBE * cube + 2.0 * _MATMUL_PER_CUBE * cube


def _per_sample_cost(width: int, depth: int) -> float:
    w = float(width)
    passes = float(depth) + 1.0          # Gram + fused layer 0 + (depth - 1) layers
    return _MAC * w * w * passes + 2.0 * w * float(depth)


def _sample_count(budget: int, width: int, depth: int) -> int:
    spend = _TARGET_UTILISATION * float(budget) - _fixed_cost(width)
    k = int(spend / _per_sample_cost(width, depth))
    return max(_MIN_SAMPLES, min(_MAX_SAMPLES, k))


def _mean_propagation(mlp: MLP) -> fnp.ndarray:
    """Cheap, extremely robust analytic fallback (~2e8 FLOPs).

    Marginal Gaussian assumption per neuron, tracking mean and variance only and
    ignoring inter-neuron correlation. Scores ~9.5e-05 -- far worse than the
    Monte-Carlo path, but it never touches a matrix decomposition and so cannot
    fail the way the main path might on a pathological MLP.
    """
    width = mlp.width
    zero = fnp.asarray(0.0, dtype=fnp.float32)
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
        _ = zero
    return fnp.stack(rows, axis=0)


class Estimator(BaseEstimator):
    """Moment-matched Monte Carlo, sized to the free share of the FLOP budget."""

    def __init__(self) -> None:
        self._setup_rng = None

    def setup(self, ctx: SetupContext) -> None:
        # Submission-level RNG. Per-MLP randomness is seeded from mlp.seed inside
        # predict() so the submission reproduces exactly under the grader's seed.
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        try:
            return self._predict_mc(mlp, budget)
        except Exception:
            pass
        try:
            return _mean_propagation(mlp)
        except Exception:
            return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)

    # -- main path ----------------------------------------------------------

    def _predict_mc(self, mlp: MLP, budget: int) -> fnp.ndarray:
        width, depth = mlp.width, mlp.depth
        # Force an even sample count so the antithetic pairing below is exact and `scale`
        # matches the true row count.
        k = (_sample_count(int(budget), width, depth) // 2) * 2
        zero = fnp.asarray(0.0, dtype=fnp.float32)
        scale = fnp.asarray(1.0 / float(k), dtype=fnp.float32)

        # Antithetic pairing: X = [xh ; -xh].  The empirical mean and every odd moment are
        # then exactly zero by construction, so no centring is needed, and the subsequent
        # whitening makes the second moment exactly identity too.  Measured on 750 real
        # competition MLPs with common random numbers: 1.122x lower final-layer MSE,
        # paired t = +3.03 -- a real effect, and it costs nothing.
        #
        # Note antithetic sampling ALONE is harmful here (measured 0.82x; ReLU networks are
        # not odd, so f(x) and f(-x) stay positively correlated).  It only pays once the
        # ensemble is also whitened -- the two together match N(0, I) in every moment up to
        # third order.  See work/RESEARCH.md §5c.
        rng = fnp.random.default_rng(mlp.seed)
        half = k // 2
        xh = fnp.asarray(rng.standard_normal((half, width)), dtype=fnp.float32)
        x = fnp.concatenate([xh, -xh], axis=0)

        w0 = fnp.asarray(mlp.weights[0], dtype=fnp.float32)
        first = self._fused_first_layer(x, w0, k, width)

        rows = []
        y = fnp.maximum(first, zero)
        rows.append(fnp.sum(y, axis=0) * scale)
        for idx in range(1, depth):
            w32 = fnp.asarray(mlp.weights[idx], dtype=fnp.float32)
            y = fnp.maximum(y @ w32, zero)
            rows.append(fnp.sum(y, axis=0) * scale)

        out = fnp.stack(rows, axis=0)
        if not bool(fnp.all(fnp.isfinite(out))):
            raise ValueError("non-finite Monte-Carlo estimate")
        return out

    @staticmethod
    def _fused_first_layer(x: fnp.ndarray, w0: fnp.ndarray, k: int, width: int) -> fnp.ndarray:
        """Return X @ (G^{-1/2} W_0), i.e. the whitened ensemble through layer 0.

        Fusing the whitener into W_0 replaces a k*n^2 transform with an n^3 one.
        On any numerical trouble we fall back to the plain (un-whitened) layer,
        which is still an unbiased estimator -- just with ~1.6x the variance.
        """
        try:
            gram = (x.T @ x) / fnp.asarray(float(k), dtype=fnp.float32)
            evals, evecs = fnp.linalg.eigh(gram)
            evals = fnp.maximum(evals, 1e-6)
            whitener = (evecs * fnp.power(evals, -0.5)) @ evecs.T
            fused = whitener @ w0
            if not bool(fnp.all(fnp.isfinite(fused))):
                return x @ w0
            return x @ fused
        except Exception:
            return x @ w0
