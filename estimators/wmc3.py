"""Whitened + antithetic Monte Carlo with TWO-SIDED lead-block pruning.

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
_CHUNK = 16_384

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

# Upper bound on the ensemble block held in memory at once (samples).
_BLOCK = 65_536

# Fraction of the FLOP budget targeted by the WORST-CASE (nothing-prunes) cost model.
# See "BUDGET SAFETY".  Real executed spend lands near 0.67 with the mask active.
_TARGET_UTILISATION = 0.92

_MIN_SAMPLES = 512
_MAX_SAMPLES = 400_000

_EIGH_PER_CUBE = 9.0
_MATMUL_PER_CUBE = 2.0
_RNG_F32_PER_ELEMENT = 16.0


def _fixed_cost(width: int, depth: int) -> float:
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
            + 48.0 * float(depth) * float(width) ** 2)


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
    draw = _RNG_F32_PER_ELEMENT * w * 0.5 + 3.0 * w
    gram = w * w
    layer0 = 2.0 * w * w + w
    rest = float(depth - 1) * (2.0 * w * w + 8.0 * w)
    return draw + gram + layer0 + rest


def _sample_count(budget: int, width: int, depth: int) -> int:
    spend = _TARGET_UTILISATION * float(budget) - _fixed_cost(width, depth)
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
            return self._legacy(x, w0f, weights, k, width, depth, split, zero)

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
            yt = fnp.maximum(w0f.T @ xb.T, zero)
            head = [fnp.sum(yt, axis=1)]
            for layer in range(1, split + 1):
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
                    y = fnp.maximum(wr[layer].T @ y, zero)
                    part.append(fnp.sum(y, axis=1))
                if wr[depth - 1] is not None:
                    yf = fnp.maximum(wr[depth - 1].T @ y, zero)
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
            idx = fnp.nonzero(fnp.max(y, axis=1))[0]
        else:
            cnt = fnp.count_nonzero(y > 0.0, axis=1)
            idx = fnp.nonzero(cnt >= _MIN_FIRE)[0]
        return None if int(idx.shape[0]) >= width else idx

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
        kink_idx = fnp.nonzero(kink)[0]
        on_idx = fnp.nonzero(on)[0]
        if int(kink_idx.shape[0]) >= width:
            return None, None
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

    def _legacy(self, x, w0f, weights, k, width, depth, split, zero):
        totals = None
        for base in range(0, k, _BLOCK):
            xb = x[base:base + _BLOCK]
            yt = fnp.maximum(w0f.T @ xb.T, zero)
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
        """One ReLU layer on a transposed block, with EXACT input-row pruning.

        Dropping the all-zero rows of `yt` together with the matching rows of `w32` leaves
        w32.T @ yt exactly unchanged -- an algebraic identity, so the estimator is
        untouched and only the MAC count falls.
        """
        alive = fnp.nonzero(fnp.max(yt, axis=1))[0]
        if int(alive.shape[0]) < width:
            return fnp.maximum(w32[alive].T @ yt[alive], zero)
        return fnp.maximum(w32.T @ yt, zero)

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
