All measurements complete. Here is the report.

**VERDICT — the chaos spectrum is measured, and degree-exactness is priced out by ~100x, not by a hair: the only affordable exactness level is D=5 (33,153 nodes, 22% of budget), it leaves S(5)/C_plain = 0.331 ± 0.029 of the variance alive, so FOM ≥ 2.1e4 (c=1.25e6) / 3.1e4 (c=1.8e6) — 78–113x above the 272 target and only 1.5–2.4x better than the live submission. D=7 needs 2.86e6 nodes = 18.9x the ENTIRE budget for one evaluation.**

## 1. The measured curve R(rho) — 120 MLPs, 65 values of rho, 6.55e5 forwards/MLP

Circle design: each sample draws an independent random 2-plane span(x,w) and evaluates 128 points u_t = cos(t·pi/64)x + sin(t·pi/64)w, giving 128 pairs at every lag simultaneously. Exact ground-truth means m["F"] used as the centring constant (no plug-in bias). Files: `hs_lib.py`, `hs_p1_rcurve.py`, `hs_rcurve.npz`, `hs_p4.log`.

| quantity | value | SE |
|---|---|---|
| R(1) = C_plain | **5.12429e-02** | 3.3e-05 |
| R(-1) | -4.44888e-03 | 2.9e-05 |
| max SE on R(rho)/R(1) over all 65 rho | **5.6e-04** | (target was <2e-3) |
| R(0)/R(1) (must be exactly 0 — validation) | +9.0e-05 | 3.2e-04 ✓ |
| odd mass Σ_{d odd} c_d /R1 = (R1−R(−1))/2R1 | **0.54341** | model-free |
| even mass Σ_{d even} c_d /R1 | **0.45659** | model-free |

Selected points (R/R1): 0.99880→0.98279, 0.98079→0.85458, 0.92388→0.66809, 0.70711→0.35902, 0.38268→0.13849, 0.09802→0.02753, −0.70711→−0.09142, −0.98079→−0.08801.

## 2. The spectrum c_1..c_20 (constrained inversion + two independent cross-checks)

Hand-written Lawson–Hanson NNLS (`hs_nnls.py`) on 62 rho values, 60 individual degrees + 27 log-spaced groups to d=8000 + an unresolved lump, hard constraint Σc_d = R(1). Validated on synthetic spectra (`hs_p0b_synth.py`): S(1),S(3),S(5) recovered to 0.3%, 0.3%, 2%.

| d | c_d/R1 (NNLS) | boot SE | across lam 0→1e12 |
|---|---|---|---|
| 1 | **0.25682** | 0.00048 | 0.0976 .. 0.2568 |
| 2 | **0.22852** | 0.00499 | 0.0937 .. 0.2285 |
| 3 | **0.09505** | 0.00293 | 0.0899 .. 0.1459 |
| 4 | 0.04385 | 0.02480 | 0.0439 .. 0.1164 |
| 5 | 0.02948 | 0.01393 | 0.0295 .. 0.0854 |
| 6 | 0.06408 | 0.03872 | 0.0243 .. 0.0729 |
| 7 | 0.04957 | 0.02183 | 0.0057 .. 0.0672 |
| 8–15 | 0.000–0.014 each | ±0.01–0.03 | not individually identified |
| 16,17 | 0.054, 0.043 | ±0.02 | not individually identified |
| Σ_{d>60} | 0.0483 | — | unresolved lump 0.0004 |

**Only c_1, c_2, c_3 are individually identified.** Beyond d=3 the individual c_d are regularisation-dominated (ranges span 3–10x); the *partial sums* are not. That is the honest statement, and it is enough.

Cross-checks on c_1, c_2 (three independent routes agreeing):

| route | c1/R1 | c2/R1 |
|---|---|---|
| NNLS inversion | 0.25682 ± 0.00048 | 0.22852 ± 0.00499 |
| weighted odd/even polynomial fit, R'(0) & R''(0)/2, rho ≤ 0.62 | 0.25685 ± 0.00040 | 0.23408 ± 0.02804 |
| **direct Stein projection** ‖E[x f(x)ᵀ]‖²_F/n, split-half unbiased, no inversion at all (`hs_p3_c1direct.py`) | **0.25750** | — |

Its own R(1) came out 5.12438e-02 vs 5.12429e-02 from the circle design — 0.002% agreement across two completely different estimators. **The "26% of the variance at chaos degree 1" claim is confirmed: 25.7%.**

## 3. Independent cross-check against the achieved variance constants (`hs_p2_constants.py`, 120 MLPs x 40 reps; `hs_p7_verify.py` through core's own API)

Predictions from the spectrum, leading order in 1/k: C_plain = R(1); C_anti = R(1)+R(−1); C_white = R(1)−c1−c2; C_wa = R(1)+R(−1)−2c2.

| estimator | C measured @k=1024 | @k=4096 | via core API @k=4096 | spectral prediction | error |
|---|---|---|---|---|---|
| plain | 5.05004e-02 | **5.11870e-02** | 5.275e-02 | 5.12429e-02 | **0.1%** |
| antithetic | 4.63684e-02 | 4.76335e-02 | — | 4.67940e-02 | 1.8% |
| whitened | 2.61847e-02 | 2.65052e-02 | — | 2.60857e-02 | 1.6% |
| whitened+antithetic | 2.34763e-02 | **2.38254e-02** | **2.34512e-02** | 2.28036e-02 | 4.5% |

- **c1+c2 = C_plain − C_white = 2.46819e-02 measured, vs 2.48720e-02 from the spectrum — 0.8% agreement.** The whitening reading in the brief is correct.
- **The "c_3 = C_whitened − C_wa" reading in the brief is wrong by 1.8x.** The correct identity is C_white − C_wa = Σ_{odd d≥3} c_d − Σ_{even d≥4} c_d = 0.28659·R1 − 0.22807·R1 = 2.999e-03, measured 2.680e-03. The actual c_3 = 0.09505·R1 = **4.87e-03**, not 2.68e-03. Antithetic kills *all* odd degrees, not just 3, and pays a 2x penalty on the even ones.
- **C_wa = 1.886e-02 as quoted in the brief is not reproducible.** Two independent measurements give 2.345e-02 and 2.383e-02 (paired plain-vs-wa ratio 2.207, t=11.8, n=480). The leading-order floor for that estimator class is 2·Σ_{even d≥4} c_d = 2.28e-02. Whatever produced 1.886e-02, it is not `core.draw_wa` on these 120 MLPs. If the live FOM 4.99e4 was computed from 1.886e-02, it is understated by ~25%.

## 4. THE DELIVERABLE — S(D), FOM(D), and the Möller wall

S(D) point estimate + bootstrap SE + LP band (extremal max/min of the partial sum subject to c≥0 and chi2 ≤ chi0 + bootstrap-p95, i.e. calibrated on the actual correlated noise, not a diagonal chi2). `hs_p5_final.py`, `hs_p5.log`.

| D | S(D)/R1 | boot SE | LP band | var. red. | FOM @c=1.8e6 | FOM @c=1.25e6 |
|---|---|---|---|---|---|---|
| 1 | 0.74310 | 0.00048 | 0.735–0.751 | 1.35x | 68541 | 47598 |
| 2 | 0.51458 | 0.00514 | 0.496–0.535 | 1.94x | 47464 | 32961 |
| **3** | **0.42189** | 0.00744 | 0.379–0.479 | 2.37x | **38914** | **27023** |
| **5** | **0.33126** | 0.02930 | 0.238–0.425 | 3.02x | **30555** | **21219** |
| 7 | 0.26739 | 0.02953 | 0.165–0.379 | 3.74x | 24663 | 17127 |
| 9 | 0.21132 | 0.02193 | 0.126–0.342 | 4.73x | 19491 | 13536 |
| 15 | 0.19936 | 0.02232 | 0.070–0.273 | 5.02x | 18388 | 12770 |
| 32 | 0.09389 | 0.00845 | 0.021–0.183 | 10.7x | 8660 | 6014 |
| 64 | 0.04108 | 0.00571 | 0.002–0.102 | 24.3x | 3789 | 2631 |
| 128 | 0.02162 | 0.00361 | 0.000–0.062 | 46.3x | 1994 | 1385 |
| 256 | 0.00676 | 0.00354 | 0.000–0.042 | 148x | 623 | 433 |

**Target FOM ≤ 272 requires S(D)/R1 ≤ 0.00295 (c=1.8e6) or ≤ 0.00425 (c=1.25e6), i.e. a 235–339x variance reduction over plain MC.**

The Möller wall, k ≥ C(256+m, m) for D = 2m+1, against k_max = B/c = 1.51e5 (c=1.8e6) / 2.18e5 (c=1.25e6):

| D | m | nodes ≥ | cost of ONE rule (c=1.8e6) | vs budget | affordable? | best FOM |
|---|---|---|---|---|---|---|
| 3 | 1 | 2.570e+02 | 4.63e+08 | 0.002x | YES | 38914 (LP-min 34977) |
| **5** | 2 | **3.3153e+04** | 5.97e+10 | **0.22x** | **YES (last rung)** | **30555 (LP-min 21950)** |
| 7 | 3 | 2.8622e+06 | 5.15e+12 | **18.9x** | **NO** | — |
| 9 | 4 | 1.8604e+08 | 3.35e+14 | 1231x | NO | — |
| 13 | 6 | 4.2407e+11 | 7.63e+17 | 2.8e6x | NO | — |

**Empirical anchor — I built a real degree-3-exact rule and measured it** (`hs_p6_rule3.py`, 120 MLPs x 64 rotations): 2n=512 nodes ±sqrt(256)·q_i on a uniformly random orthonormal basis.

| | measured |
|---|---|
| C_rule = MSE·k (k=512, one rotation) | **2.42265e-02** |
| spectral price S(3) | 2.16189e-02 |
| **C_rule / S(3)** | **1.12** — the S(D)·c pricing is accurate to 12% for an actual rule |
| irreducible bias² · k (rotational randomisation cannot remove it) | 8.51e-04 (3.5% of C) |
| C_plain at same k | 5.15153e-02 → **only 2.13x reduction**, paired t = −14.1 |
| FOM | **4.36e+04** (c=1.8e6) / **3.03e+04** (c=1.25e6) |

A genuine degree-3 rule is statistically **tied with** whitened+antithetic (2.42e-02 vs 2.35e-02) and 160x/111x above target.

**What degree D would be needed, and what it costs:** S(D)/R1 ≤ 0.00425 is reached at D ≈ 600–750 by the measured large-D local exponent (point estimate; the LP band cannot exclude D as low as ~96 nor bound it above). That is m ≈ 300–375, giving C(256+m, m) ≈ **1e170 – 1e185** nodes against an affordable 2.2e5. Even at the absurdly generous D*=501 the node count is 1e151.

## 5. Does "C(D) ∝ D^-0.574" survive? No — it is a local slope at D∈{1,3}, and it is the wrong slope everywhere else

Two independent readings of the tail. First, the model-free log-log slope q of R(1)−R(rho) vs eps=1−rho (which measures S(D) ~ D^-q at D ~ 1/eps, with no inversion at all):

| eps range | D probed | q | SE |
|---|---|---|---|
| 0.404–0.486 | ~2 | 0.3937 | 0.0002 |
| 0.197–0.259 | 3–5 | 0.4560 | 0.0002 |
| 0.096–0.142 | 7–10 | 0.5076 | 0.0001 |
| 0.030–0.058 | 17–33 | 0.5914 | 0.0001 |
| 0.011–0.030 | 33–92 | 0.6647 | 0.0001 |
| 0.0012–0.011 | 92–830 | **0.7932** | 0.0000 |

Second, local exponents of the inverted S(D):

| window | exponent | naive D* for FOM=272 |
|---|---|---|
| D ∈ [1,3] (the window the old law was fitted in) | **−0.5153** | 4.58e+04 |
| D ∈ [4,9] | −0.7078 | 3.81e+03 |
| D ∈ [15,64] | −1.0550 | 7.49e+02 |
| D ∈ [64,256] | −1.2093 | 6.32e+02 |

S(D) is **not** a power law — the local exponent steepens monotonically from −0.39 to −1.21 across the resolvable range. The old −0.574 is a fair estimate *of the D∈{1,3} local slope* (I measure −0.515 there) and nothing more; extrapolating it four decades overstates D* by 20–70x. My data resolve at most D ~ 1/(1−rho_max) = 830, so I will not quote a D* beyond ~8000, and I do not need to: **the correction is irrelevant, because the node-count wall bites at D=7, two rungs below the cheapest D* anyone has proposed.**

## What would have to be true for me to be wrong

1. **A degree-D-exact rule beats its own residual variance.** I priced FOM(D) = S(D)·c, which assumes the annihilated estimator's residual behaves like iid noise. A rule whose nodes *anti-correlate* the surviving degrees could beat S(D). The measured degree-3 rule came in at 1.12x S(3), i.e. slightly *worse*, so this would need a qualitatively different construction. If someone finds one at D=5 it would still need a 78x further gain to hit 272.
2. **The Möller bound k ≥ C(256+m,m) is not tight for the specific measure/functional here.** It is a lower bound on nodes for *full* degree-(2m+1) exactness. A rule exact only on the degrees that actually carry mass (a "sparse-chaos" rule) could be far cheaper. But c_1 alone is 25.7% and lives in a 256-dim space, and 45.7% sits in even degrees — there is no measured sparsity to exploit. Nothing in my spectrum is concentrated.
3. **c_2 is materially larger than 0.229.** The whitened-estimator identity C_plain − C_white = c1+c2 agrees to 0.8%, and three routes agree on c_1 to 0.3%, so this would require both the R(rho) machinery and the direct Stein estimator to be wrong in the same direction. I consider it closed.
4. **The unresolved tail is not where I put it.** The LP band at D≥96 admits S(D)=0, i.e. the data alone cannot exclude that everything sits below degree 96. That is the genuine identifiability limit, and it *helps* the annihilator story — but D=96 needs m=47, C(303,47) ≈ 1e50 nodes. It changes nothing.
5. **The cost model changes.** Everything scales linearly in c. Even at a hypothetical c = 1e5 (12.5x below the metered Strassen floor), FOM(5) = 1.7e3, still 6x above target.

Artefacts (all absolute paths, `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/`): `hs_lib.py` (circle design), `hs_nnls.py` (Lawson–Hanson + grouped design), `hs_p0b_synth.py` (inversion validation), `hs_p1_rcurve.py` → `hs_rcurve.npz` + `hs_p1.log`, `hs_p2_constants.py` → `hs_p2.log`, `hs_p3_c1direct.py`, `hs_p4_analyse.py` → `hs_p4.log` + `hs_spectrum.npz`, `hs_p5_final.py` → `hs_p5.log`, `hs_p6_rule3.py` → `hs_p6.log`, `hs_p7_verify.py`.