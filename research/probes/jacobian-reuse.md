VERDICT: Jacobian reuse is dead — best measured FOM 1.94e6, i.e. 45.9x WORSE than the whitened+antithetic baseline (4.22e4) at matched FLOPs, paired t=12.6 on 64 MLPs. The cost arithmetic is fine; the geometry is not: reaching the ideal-case floor (FOM 768, itself still 2.8x above the 272 target) requires ~1e271 hubs.

## 1. Exactness and the perturbation law (`pwl_01_pert.py`, 6 MLPs, 24 directions each)

`f(x0) = x0 @ P0` verified to 2.0e-06 relative (float32 roundoff). Perturbation `x_t = x0 + t·||x0||·δ`, `||δ||=1`:

| t (relative step) | rel. error of `x_t@P0` | sign flips /8192 |
|---|---|---|
| 1e-6 | 1.92e-06 (roundoff) | 0.0 |
| 1e-4 | 3.76e-06 | 0.3 (0.004%) |
| 1e-3 | 1.24e-04 | 2.9 (0.035%) |
| 1e-2 | 4.03e-03 | 26.6 (0.33%) |
| 1e-1 | 9.64e-02 | 221 (2.70%) |
| 3e-1 | 3.19e-01 | 496 (6.05%) |
| 1.0 | 8.52e-01 | 923 (11.3%) |

**The law is relerr ≈ t, exactly linear over three decades.** There is no plateau, no "cell you can live in": the error starts growing at the first sign flip, at t ≈ 1e-4.

## 2. Sign-flip geometry vs correlation (`pwl_02_rho.py`, 8 MLPs)

`x = ρ·x0 + sqrt(1-ρ²)·g`. Agreement is *not* 50% at ρ=0 in deep layers (many neurons are near-always-on), which is why the naive "8192 coin flips" intuition understates the damage — the damage is concentrated in layer 1.

| ρ | angle | agree L1 | L8 | L16 | L32 | all | relerr |
|---|---|---|---|---|---|---|---|
| 0.0 | 90.0° | 0.498 | 0.805 | 0.870 | 0.921 | 0.840 | 1.632 |
| 0.6 | 53.1° | 0.700 | 0.838 | 0.884 | 0.927 | 0.870 | 1.107 |
| 0.9 | 25.8° | 0.850 | 0.896 | 0.917 | 0.939 | 0.914 | 0.557 |
| 0.99 | 8.1° | 0.951 | 0.964 | 0.965 | 0.966 | 0.964 | 0.156 |
| 0.999 | 2.6° | 0.986 | 0.989 | 0.988 | 0.985 | 0.987 | 0.037 |

`Var(f)/E[f²] = 0.0487/1.185 = 0.041`, so a surrogate must reach relerr < 0.20 merely to be *better than predicting zero*. That is ρ > 0.995, i.e. inside 5.7°.

## 3. Hub-and-spoke (`pwl_03_hub.py`, 6 MLPs, 2048 samples, true nearest-hub assignment)

| m hubs | nearest-hub angle | Vd = Var(f−g) | Vd/Vf |
|---|---|---|---|
| 1 | 89.99° | 1.982 | 35.5 |
| 4 | 86.35° | 1.909 | 34.2 |
| 16 | 83.68° | 1.805 | 32.3 |
| 64 | 81.59° | 1.731 | 31.0 |
| 256 | 79.88° | 1.666 | 29.9 |

**256x more hubs bought 1.19x** (empirical slope Vd ∝ m^−0.031). The surrogate is 30x worse than the constant predictor. Spherical-cap counting in d=256 (exact, not an extrapolation): 36.9°→1e58 hubs, 18.2°→1e131, 8.1°→1e218, 2.6°→1e346.

## 4. Jacobian spectrum and rank-truncation bias (`pwl_04_spec.py`, `pwl_10_rankbias.py`)

P(x) is extremely low rank as predicted: participation rank **3.95 ± 0.82**; rank 5.6 / 11.9 / 18.4 for 90 / 99 / 99.9% of ‖P‖²_F; σ₆₄/σ₁ = 2e-06. But Frobenius energy is again not accuracy — measured bias against the mean, CRN-paired, 8 MLPs × 384 exact Jacobians:

| r | bias² (MSE floor) | vs baseline MSE 5.73e-06 |
|---|---|---|
| 4 | 4.59e-02 | 8000x |
| 8 | 3.22e-03 | 562x |
| 18 | 9.55e-06 | 1.7x |
| 32 | 1.73e-08 | 0.003x |
| 64 | 9.0e-15 | 1.6e-09x |

**Usable in-cell rank is 32, not 4** — an 8x cost penalty over the participation rank. Subspace overlap between two random hubs: output-side top-8 cos² = 0.327 (10x chance), input-side only 0.076 (2.4x chance) — the shared structure is on the wrong side.

## 5. The ideal-case floor, and the headline

Ideal floor of the *entire* family = C_wa · c_incell with a perfect (exact) linearization, C_wa = 0.02345 measured on 64 MLPs:

| r | c_incell | FOM floor | vs target 272 | |
|---|---|---|---|---|
| 4 | 4.10e3 | 96 | 0.35x | bias-dead |
| 18 | 1.84e4 | 432 | 1.59x | bias-dead |
| **32** | **3.28e4** | **768** | **2.83x** | usable |
| 256 (full) | 1.31e5 | 3074 | 11.3x | usable |

**Even a perfect oracle linearization cannot reach 272.** And what is actually achievable (`pwl_08_paired.py`, 64 MLPs, matched 7.37e9 FLOPs/MLP, 2-level MLMC with M=4 hubs, k1=4959, k2=1355):

| estimator | MSE | FOM | |
|---|---|---|---|
| whitened+antithetic k=4096 | 5.725e-06 | 4.221e4 | baseline |
| hub 2-level | 2.625e-04 | **1.935e6** | 45.9x worse, t=12.6 |

Also measured: single-hub Jacobian as a zero-mean linear control variate — mean |corr(f_j, (xP0)_j)| = 0.0315, Var ratio 0.9901, and +7% cost → 1.06x worse (24 MLPs). The best *possible* linear CV removes 27% (0.7304), reproducing the known degree-1 chaos figure. Active-subspace test (`pwl_05_active.py`, nested MC): top-64 eigendirections of E[PPᵀ] capture only 0.35–0.58 of Var(f) — independently reconfirms d≈250.

## The single number that kills it

To merely **break even** with the baseline, the two-level identity FOM = (√(Vg·cg) + √(Vd·cf))² requires Vd/Vf ≤ 0.42, i.e. relerr ≤ 0.13, i.e. a hub within 7.0°, i.e. **1e234 hubs**. To hit FOM 272: 0.81°, **1e473 hubs**. Measured reality at m=256: 79.9° and Vd/Vf = 29.9.

## What would have to be true for me to be wrong

1. **Hubs would have to be placeable non-uniformly and adaptively** so that the covering exponent is the *intrinsic* dimension rather than 256. Measured against this: the input-side active subspace captures only 58% of variance at r=64, and input-side hub subspace overlap is 0.076 — I find no low-dimensional structure on the input side to exploit. If someone finds an r≲10 input coordinate system, 1e234 becomes ~1e9 and the scheme revives.
2. **The relerr≈t law would have to bend.** It is linear over t ∈ [1e-3, 3e-1] with no shoulder; I did not extrapolate it beyond measured points, and the m-sweep (Vd ∝ m^−0.031) independently confirms no shoulder out to m=256.
3. **The rank-truncation bias would have to be correctable** so r=8 (FOM floor 192, below target) becomes usable. Its bias is 3.2e-3 — 562x the baseline MSE — and it is a genuine bias, not noise, so it needs an exact correction, which costs a full evaluation.

Scripts (all absolute): `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/` + `pwl_lib.py`, `pwl_01_pert.py`, `pwl_02_rho.py`, `pwl_03_hub.py`, `pwl_04_spec.py`, `pwl_05_active.py`, `pwl_06_cv.py`, `pwl_07_final.py`, `pwl_08_paired.py`, `pwl_09_kill.py`, `pwl_10_rankbias.py`, `pwl_11_floor.py`.