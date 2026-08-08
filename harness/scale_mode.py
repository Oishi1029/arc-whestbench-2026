"""Is the common-scale error mode SYSTEMATIC (correctable) or NOISE (not)?

BACKGROUND
  The frontier workflow measured that 25.9% of the final-layer MSE lies in a single
  "common-scale" mode -- error proportional to the truth, e ~ alpha * t -- against 0.39%
  expected if the error were isotropic in 256 dimensions. A 66x enrichment. A further 8.6%
  sits in a constant offset. Together ~1.5x of the error lives in two directions.

  Interpretation: some draws of the random ensemble happen to carry slightly more "energy"
  than average, and that scale error propagates multiplicatively through all 32 layers, so
  every neuron ends up over- or under-estimated by roughly the same RELATIVE amount.

THE QUESTION THAT DECIDES EVERYTHING
  Is alpha a property of the MLP (systematic -> predictable from the weights -> correctable,
  and shippable as a precomputed artefact), or a property of the random draw (pure noise ->
  a different alpha every time -> nothing to learn)?

  Monte Carlo is unbiased, so the null hypothesis is "pure noise". But whitening and the
  antithetic pairing deliberately perturb the ensemble, and the ReLU is nonlinear, so a
  systematic component is not impossible.

METHOD -- a variance decomposition, which settles it cleanly
  Run the SAME estimator on the SAME MLP with R independent random streams. Then
      total Var(alpha) = between-MLP variance (systematic) + within-MLP variance (noise)
  and the systematic share is the intraclass correlation
      ICC = (MS_between - MS_within) / (MS_between + (R-1) * MS_within).

  ICC near 0  -> alpha is pure sampling noise. Unpredictable in principle. Lineage CLOSED.
  ICC large   -> alpha is a stable property of the MLP. Worth trying to predict it, and the
                 achievable gain is bounded by the systematic share.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/binyong/Library/CloudStorage/GoogleDrive-binyongbong1029@gmail.com/"
                   "My Drive/HACKATHONS/ARC White-Box Estimation Challenge/work")
from offline_bench import FLOP_BUDGET, load_split, mc_flops  # noqa: E402
from runoff import whiten_transform  # noqa: E402

N_MLPS = 150
REPS = 5


def estimate(W, rng, k):
    n = W[0].shape[0]
    half = k // 2
    xh = rng.standard_normal((half, n), dtype=np.float32)
    x = np.concatenate([xh, -xh]).astype(np.float32)
    t = whiten_transform(x)
    y = np.maximum(x @ (t @ W[0]), 0.0)
    for w in W[1:]:
        y = np.maximum(y @ w, 0.0)
    return y.mean(0, dtype=np.float64)


def decompose(e, t):
    """Split the error into: scale mode (along t), constant offset, and the rest."""
    ones = np.ones_like(t) / math.sqrt(len(t))
    tn = t / np.linalg.norm(t)
    a_scale = float(e @ tn)                 # component along the truth direction
    a_off = float(e @ ones)                 # component along the constant vector
    tot = float(e @ e)
    return a_scale, a_off, tot


if __name__ == "__main__":
    mlps = load_split("full", limit=N_MLPS)
    n, d = 256, 32
    k = int((0.095 * FLOP_BUDGET - 13.0 * n ** 3) / mc_flops(1, n, d, 1.0))
    k = (k // 2) * 2
    print(f"{len(mlps)} real MLPs x {REPS} independent streams, k={k:,}\n")

    alphas = np.zeros((len(mlps), REPS))
    offs = np.zeros((len(mlps), REPS))
    frac_scale, frac_off = [], []

    for i, m in enumerate(mlps):
        t = m["final_means"]
        tn_norm = np.linalg.norm(t)
        for r in range(REPS):
            rng = np.random.default_rng((m["seed"] % (2 ** 63)) + 7919 * r)
            e = estimate(m["weights"], rng, k) - t
            a_s, a_o, tot = decompose(e, t)
            # alpha is the RELATIVE scale error: e ~ alpha * t
            alphas[i, r] = a_s / tn_norm
            offs[i, r] = a_o
            frac_scale.append(a_s ** 2 / tot)
            frac_off.append(a_o ** 2 / tot)
        if (i + 1) % 30 == 0:
            print(f"  ... {i+1}/{len(mlps)}")

    print()
    print("=" * 74)
    print("1. HOW CONCENTRATED IS THE ERROR?  (isotropic baseline = 1/256 = 0.39%)")
    print("=" * 74)
    print(f"  energy in the scale mode (along truth) : {np.mean(frac_scale):7.2%}"
          f"   ({np.mean(frac_scale)*256:5.1f}x isotropic)")
    print(f"  energy in the constant offset          : {np.mean(frac_off):7.2%}"
          f"   ({np.mean(frac_off)*256:5.1f}x isotropic)")

    def icc(a):
        gm = a.mean()
        ms_b = REPS * ((a.mean(1) - gm) ** 2).sum() / (a.shape[0] - 1)
        ms_w = ((a - a.mean(1, keepdims=True)) ** 2).sum() / (a.shape[0] * (REPS - 1))
        return (ms_b - ms_w) / (ms_b + (REPS - 1) * ms_w), ms_b, ms_w

    print()
    print("=" * 74)
    print("2. IS IT SYSTEMATIC OR NOISE?   (the decisive question)")
    print("=" * 74)
    for nm, a in (("scale alpha", alphas), ("offset", offs)):
        v, msb, msw = icc(a)
        print(f"  {nm:12s} ICC = {v:+.4f}   MS_between={msb:.3e}  MS_within={msw:.3e}")
        print(f"               per-MLP mean |alpha| = {np.abs(a.mean(1)).mean():.3e}, "
              f"within-MLP sd = {a.std(1, ddof=1).mean():.3e}")
    print()
    print("  ICC ~ 0  => pure sampling noise, unpredictable in principle => LINEAGE CLOSED.")
    print("  ICC >> 0 => stable per-MLP property => worth learning to predict.")

    print()
    print("=" * 74)
    print("3. UPPER BOUND ON THE PRIZE")
    print("=" * 74)
    print("  If an ORACLE removed the scale mode and the offset entirely:")
    rem = 1.0 - np.mean(frac_scale) - np.mean(frac_off)
    print(f"     MSE x {rem:.3f}  =>  {1/rem:.2f}x better  (this is the ABSOLUTE ceiling,")
    print("     achievable only with perfect knowledge; a real predictor gets ICC x that.)")
