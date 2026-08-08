"""Paired run-off between the estimator variants, on the REAL competition MLPs.

Method
------
Every variant sees the SAME MLPs and the SAME base random draws (common random numbers), so
the *difference* between variants has far lower variance than either estimate alone. This is
what makes a 250-MLP comparison decisive when a 100-MLP independent comparison is not
(see work/RESEARCH.md §7b: the mini split's own RNG noise is 10.4%, so it cannot resolve
anything under ~21%).

Each variant is given the same FLOP budget (9.5% of 2.72e11) and its sample count k is derived
from its own cost, so variants that pay for extra passes get correspondingly fewer samples --
an honest equal-compute comparison.

Variants
--------
  mine      whitened MC                        (work/mine/wmc.py)
  anti      antithetic + whitened              (the workflow winner's core claim)
  mean1     whitened + exact layer-1 mean anchor
  anti+mean1 both                              (= the workflow winner's algorithm)
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/Users/binyong/Library/CloudStorage/GoogleDrive-binyongbong1029@gmail.com/"
                   "My Drive/HACKATHONS/ARC White-Box Estimation Challenge/work")
from offline_bench import FLOP_BUDGET, load_split, mc_flops  # noqa: E402

N_MLPS = 250
TARGET = 0.095
SQRT2PI = math.sqrt(2 * math.pi)
FIXED = 13.0 * 256 ** 3


def k_for(extra_passes: float, n: int, d: int) -> int:
    per = mc_flops(1, n, d, extra_passes)
    return max(1, int((TARGET * FLOP_BUDGET - FIXED) / per))


def whiten_transform(x: np.ndarray) -> np.ndarray:
    g = (x.T @ x).astype(np.float64) / x.shape[0]
    ev, U = np.linalg.eigh(g)
    return ((U * np.maximum(ev, 1e-8) ** -0.5) @ U.T).astype(np.float32)


def forward(y: np.ndarray, Ws) -> np.ndarray:
    for w in Ws:
        y = np.maximum(y @ w, 0.0)
    return y.mean(0, dtype=np.float64)


def run_variant(W, base, antithetic: bool, mean1: bool) -> np.ndarray:
    """base: a (kmax, n) float32 pool of N(0,1) draws — shared across variants."""
    n = W[0].shape[0]
    d = len(W)
    extra = 1.0
    k = k_for(extra, n, d)
    if antithetic:
        half = k // 2
        x = np.concatenate([base[:half], -base[:half]]).astype(np.float32)
    else:
        x = base[:k].astype(np.float32)
        x = x - x.mean(0)
    t = whiten_transform(x)
    y = np.maximum(x @ (t @ W[0]), 0.0)
    if mean1:
        # z_1 = W_0^T x is EXACTLY Gaussian when x is N(0,I), so E[ReLU(z_1)]_i is exact.
        sigma = np.sqrt((W[0].astype(np.float64) ** 2).sum(0))
        y = (y - y.mean(0) + (sigma / SQRT2PI)).astype(np.float32)
    return forward(y, W[1:])


if __name__ == "__main__":
    mlps = load_split("full", limit=N_MLPS)
    n, d = mlps[0]["weights"][0].shape[0], len(mlps[0]["weights"])
    kmax = k_for(1.0, n, d)
    print(f"{len(mlps)} real competition MLPs, width={n} depth={d}, k={kmax:,} "
          f"(equal compute at {TARGET:.1%} of budget)")
    print("common random numbers across variants -> paired comparison\n")

    variants = {
        "mine (whitened)":    dict(antithetic=False, mean1=False),
        "anti":               dict(antithetic=True,  mean1=False),
        "mean1":              dict(antithetic=False, mean1=True),
        "anti+mean1 (winner)": dict(antithetic=True, mean1=True),
    }
    errs = {name: [] for name in variants}

    for i, m in enumerate(mlps):
        rng = np.random.default_rng((m["seed"] % (2 ** 63)) ^ 0xA5A5)
        base = rng.standard_normal((kmax, n), dtype=np.float32)   # SHARED across variants
        tgt = m["final_means"]
        for name, kw in variants.items():
            p = run_variant(m["weights"], base, **kw)
            errs[name].append(float(np.mean((p - tgt) ** 2)))
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(mlps)}")

    print()
    print("=" * 78)
    ref = np.array(errs["mine (whitened)"])
    print(f"{'variant':22s} {'mean MSE':>12} {'adjusted':>12} {'vs mine':>9} {'paired p':>10}")
    for name in variants:
        v = np.array(errs[name])
        mse = v.mean()
        line = f"{name:22s} {mse:12.4e} {mse*0.1:12.4e} {ref.mean()/mse:8.3f}x"
        if name != "mine (whitened)":
            dif = ref - v                       # >0 means the variant is better
            se = dif.std(ddof=1) / math.sqrt(len(dif))
            t = dif.mean() / se if se > 0 else 0.0
            line += f"  t={t:+6.2f}"
            line += "  SIGNIFICANT" if abs(t) > 2.5 else "  (n.s.)"
        print(line)
    print("=" * 78)
    print("t is the paired t-statistic vs 'mine'. |t| > 2.5 => a real difference, not noise.")
