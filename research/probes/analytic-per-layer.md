All measurements complete. Here is the report.

**VERDICT: Analytic propagation beats MC on the *mean* for layers 1–9 at k=4096 and 1–2 at full budget, but exploiting it is worth at most 1.017x (t=2.73) and is NET NEGATIVE once the 6.56e9-FLOP analytic pass is metered. Best FOM = 3.96e4 idealized / 4.12e4 metered vs baseline 4.03e4 — 146x short of the 272 target. This is a 1.0x, not a 100x.**

---

## 1. Analytic propagation, implemented and validated

`/private/tmp/claude-501/.../scratchpad/ap_lib.py`. I derived and numerically verified the exact **non-central** bivariate ReLU second moment (the shipped arc-cosine kernel is only the zero-mean case, and `mu/sigma` reaches 4.3 by layer 32, so non-centrality is not optional):

```
E[ReLU(u)ReLU(v)] = (m1 m2 + r s1 s2)·Phi2(p,q;r)
                  + m1 s2 phi(q)Phi(lam) + m2 s1 phi(p)Phi(kap)
                  + s1 s2 w phi(p) phi(kap)
p=m1/s1, q=m2/s2, w=sqrt(1-r^2), kap=(q-rp)/w, lam=(p-rq)/w
```

| check | result |
|---|---|
| reduces to arc-cosine kernel at m=0 | max abs diff **1.8e-15** |
| vs 2.0e8-sample MC | max abs err 2.6e-4 (MC se 3.0e-4) |
| `Phi2` (Drezner–Wesolowsky, nq=24×3 panels) | 6.1e-10 |
| quadrature convergence nq24→nq64 | 4.9e-17 |
| **pure-analytic final-layer MSE, 40 MLPs** | **5.917e-05** (expected 6–8e-05 ✓) |

Note: the Owen-T form of `Phi2` is 0/0 at h=k=0 — exactly the layer-1 case. That bug silently returns 0.5 instead of 1/4+asin(r)/2π; use the DW form.

## 2. Per-layer accuracy (new measurement, 40 MLPs vs the exact `A` bake)

| l | RMS abs err | rel | MC se (k=4096) | rel | ratio an/MC | rel sd err | mu/sd |
|---|---|---|---|---|---|---|---|
| 1 | 2.64e-05 | 0.0047% | 1.29e-02 | 2.29% | **0.002** | 0.000% | 0.00 |
| 2 | 1.57e-03 | 0.227% | 1.11e-02 | 1.60% | 0.142 | 0.208% | 0.68 |
| 3 | 2.75e-03 | 0.356% | 9.84e-03 | 1.27% | 0.280 | 0.606% | 0.98 |
| 4 | 3.68e-03 | 0.446% | 8.92e-03 | 1.08% | 0.413 | 1.00% | 1.23 |
| 6 | 4.95e-03 | 0.563% | 7.48e-03 | 0.851% | 0.662 | 2.12% | 1.68 |
| 8 | 5.87e-03 | 0.652% | 6.49e-03 | 0.720% | 0.905 | 3.32% | 2.00 |
| **9** | 6.06e-03 | 0.674% | 6.09e-03 | 0.677% | **0.995** ← crossover | 4.15% | 2.18 |
| 12 | 6.87e-03 | 0.719% | 5.32e-03 | 0.556% | 1.292 | 6.36% | 2.68 |
| 16 | 7.55e-03 | 0.792% | 4.62e-03 | 0.485% | 1.634 | 9.03% | 3.11 |
| 24 | 8.02e-03 | 0.814% | 3.89e-03 | 0.395% | 2.063 | 14.0% | 3.86 |
| 32 | 7.69e-03 | 0.827% | 3.39e-03 | 0.364% | 2.270 | 16.1% | 4.28 |

Layer 1 is **exact** — its 2.64e-05 residual is precisely the reference bake's own noise (sd(y₁)/√1e9 = 2.6e-05). Analytic error saturates at ~0.83%; it does *not* grow faster than the endpoint suggests.

**Crossover L\*** (analytic mean more accurate than the MC mean): k=4096 → **L\*=9**; k=16384 → **L\*=4**; k=151,111 (full budget) → **L\*=2**.

## 3. Hard anchoring, layers 1..L (40 MLPs, k=4096, bias/variance split from 8 reps)

| L | oracle VR | oracle b² | analytic VR | analytic b² | analytic FOM_eff |
|---|---|---|---|---|---|
| 2 | 1.049 | 1.4e-08 | 1.048 | 7.04e-07 | 2.28e5 |
| 4 | 1.205 | −1.8e-08 | 1.201 | 5.72e-06 | 1.59e6 |
| 8 | 1.655 | 4.3e-08 | 1.647 | 1.91e-05 | 5.22e6 |
| 16 | 2.744 | 1.7e-08 | 2.734 | 3.62e-05 | 9.87e6 |
| 31 | **47.54** | 1.2e-09 | 47.24 | 5.71e-05 | 1.55e7 |

The analytic anchor removes **exactly as much variance as the oracle** (47.2 vs 47.5) and is destroyed entirely by bias. The previous session's "L=2 0.75x, L=8 0.10x" was **not buggy** — I reproduce the same collapse.

Two mechanisms they (and the task framing) missed:

**(a) Partial anchors compound.** 31 layers at w≈0.5 gave b² = 6.09e-05 — the *entire* pure-analytic bias. The layer-l error is re-injected at every later layer, so a constant w converges geometrically to a hard anchor. Per-layer independent w calculus is invalid; the correct object is a **single-layer** anchor (which subsumes all shallower ones: single-anchor-at-16 Vrem = 0.6311 = cumulative-1..16 Vrem = 0.6307).

**(b) Structured error transmits ~10x better than noise** (24 MLPs, transmission gain G = |Δoutput|²/|δ|², equal norms):

| l | G(analytic err) | G(MC sampling law) | ratio |
|---|---|---|---|
| 2 | 1.89e-01 | 1.56e-02 | **12.1** |
| 3 | 1.87e-01 | 2.43e-02 | 7.7 |
| 8 | 3.97e-01 | 9.20e-02 | 4.3 |
| 16 | 4.88e-01 | 2.30e-01 | 2.1 |
| 31 | 9.24e-01 | 9.40e-01 | 1.0 |

So the per-layer crossover **overstates the usable depth by ~150x in squared output-MSE units**. That is the answer to "which": neither buggy anchoring nor faster-than-expected error growth — the metric was wrong.

## 4. Optimally weighted hybrid — the best this family can do (64 MLPs)

Single-layer anchor, w\* = Vrem/(Vrem+b²). Paired CRN at the **0.1B corner** (k=15112; the multiplier floors at 0.1, which is the optimal operating point for a biased estimator and score-neutral for the unbiased baseline):

| layer | w | C | VR(var only) | b² | score·B | **ratio** | t |
|---|---|---|---|---|---|---|---|
| — | baseline | 0.02237 | 1.000 | — | 4.049e4 | — | — |
| 31 | 0.011 | 0.02190 | 1.022 | 1.93e-08 | 3.994e4 | **1.0139** | **2.73** |
| 31 | 0.022 | 0.02142 | 1.044 | 4.57e-08 | 3.980e4 | **1.0173** | 1.72 |
| 31 | 0.044 | 0.02049 | 1.092 | 1.43e-07 | 4.078e4 | 0.9928 | −0.37 |
| 31 | 0.100 | 0.01822 | 1.228 | 6.63e-07 | 5.082e4 | 0.7967 | −5.94 |
| 31 | 1.000 | 0.00047 | 47.93 | 6.24e-05 | 1.698e6 | 0.0239 | −29.05 |
| 16 | 0.024 | 0.02169 | 1.032 | 3.04e-08 | 3.986e4 | 1.0158 | 1.55 |
| 8 | 0.029 | 0.02185 | 1.024 | 2.54e-08 | 4.002e4 | 1.0119 | 1.24 |

Optimal w is **0.011–0.022**. At full budget the same sweep gives w\*≈0.002 and a gain of **0.18%**.

## 5. The cost that kills it

Metered analytic pass (`ap_cost.py`, stated cost model): 31 × (7.50e7 kernel + 1.34e8 float64 matmuls) = **6.558e9 FLOPs**.

| operating point | k baseline | k hybrid | sample loss | gross gain | **net** |
|---|---|---|---|---|---|
| full B | 151,111 | 147,468 | 2.4% | 1.0018x | **0.978x** |
| 0.1B corner | 15,111 | 11,468 | **24.1%** | 1.023x | **0.776x** |

**FOM: baseline 4.03e4 → 3.96e4 if the analytic pass were free → 4.12e4 metered.** Target 272. Pure analytic alone scores 1.61e6 (40x worse than MC).

## 6. What would have to be true for me to be wrong

To hit FOM 272 at the 0.1B corner you need MSE ≤ 1e-08 at k=15,111, i.e. **both** C ≤ 1.51e-4 (a **148x** variance reduction) **and** b ≤ 1.0e-4 absolute = **0.0105% relative**.

| requirement | measured | shortfall |
|---|---|---|
| VR ≥ 148x from mean anchoring | oracle ceiling **46.5x** (single anchor l=31, Vrem=0.9785) | 3.2x short — mean anchoring alone cannot reach the target even with a perfect oracle; covariance anchoring is mandatory |
| bias ≤ 0.0105% | pure analytic mu₃₁: **0.844%** | 80x (6460x in b²) |
| " | one Gaussian step from the **exact** layer-31 state: **0.162%** | 15x (238x in b²) |
| " | 1-D plug-in with **exact** (mu₃₂,sigma₃₂): **0.112%** (b²·B = 3.3e5) | **11x (114x in b²)** |

The last row is the kill. The irreducible 1-D Gaussian-closure error, measured per layer with exact marginals fed in (8 MLPs, k=131072, 12 reps), is 0.223% at l=2, peaks at 0.259% (l=4), and decays only to 0.112% at l=32 — everywhere ≥ 10x too large. Restarting propagation from an exact deep state barely helps (0.725% from l=1 → 0.162% from l=31), because the error is front-loaded, not accumulated.

So I am wrong only if (i) a closure exists whose *per-layer marginal* error is under 0.0105% at comparable cost — the already-closed Hermite/cumulant results argue no; or (ii) the transmission-gain measurement is wrong and structured error is actually attenuated downstream — it measured >1 at every layer, median over 24 MLPs.

**Files:** `ap_lib.py` (verified moment kernel — reusable), `ap_verify2.py`, `ap_perlayer.py`, `ap_hybrid.py`, `ap_single.py`, `ap_amp.py`, `ap_floor.py`, `ap_final.py`, `ap_cost.py`, all in `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/`.