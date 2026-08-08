## 1. VERDICT

**The oracle-anchoring measurement replicates exactly — and it is an identity, not an estimator.** 95% of the 51x mean-anchoring gain comes from anchoring **layer 31 alone** (VR 48.1 of 50.6, t=+11.3, 64 MLPs), i.e. "if you are handed E[y_31] you nearly have E[y_32]". Priced honestly, the best self-contained anchored estimator has **FOM 1.14e5 (Phi 4.19e-07), 1.14x better than the whitened+antithetic baseline and 419x above the 272 target**; the covariance version is **0.92x, i.e. worse than baseline**. The target is not shown to be information-limited by this measurement.

## 2. MEASUREMENTS

**(1) Mean anchoring, L-sweep — 64 MLPs, k=4096, reps=2 (128 paired obs), CRN**
Baseline w+a: MSE 6.168e-06, C 2.526e-02, FOM 1.06e5. **L=1 → 1.04x, matching the known 1.09x sanity value.**

| L | MSE | VR | t | C=MSE·k | FOM (c=4.19e6) |
|---|---|---|---|---|---|
| 1 | 5.533e-06 | 1.04 | +1.49 | 2.266e-02 | 9.50e4 |
| 2 | 5.020e-06 | 1.15 | +4.61 | 2.056e-02 | 8.62e4 |
| 4 | 4.328e-06 | 1.33 | +6.66 | 1.773e-02 | 7.43e4 |
| 8 | 3.228e-06 | 1.79 | +8.43 | 1.322e-02 | 5.54e4 |
| 16 | 1.927e-06 | 3.00 | +10.83 | 7.893e-03 | 3.31e4 |
| 24 | 9.157e-07 | 6.31 | +13.03 | 3.751e-03 | 1.57e4 |
| 30 | 2.253e-07 | 25.64 | +14.55 | 9.227e-04 | 3.87e3 |
| **31** | **1.109e-07** | **52.09** | **+14.74** | **4.542e-04** | **1.90e3** |

C is flat in k (4.68e-4 / 4.38e-4 / 4.49e-4 / 4.62e-4 at k=1024/4096/16384/65536) → no bias floor. **Claim of 57x / C=4.5e-4 CONFIRMED.**

**(2) The 51x is one layer.** 64 MLPs, k=4096, reps=2, anchoring *only* the listed layers:

| anchored | MSE | VR | t |
|---|---|---|---|
| all 1..31 | 1.088e-07 | 50.61 | +11.27 |
| **only 31** | **1.146e-07** | **48.08** | +11.26 |
| only 28..31 | 1.129e-07 | 48.80 | +11.26 |
| 1..30 (31 excluded) | 2.220e-07 | 24.81 | +11.07 |
| only 30 | 2.337e-07 | 23.56 | +11.05 |
| only 1 | 5.075e-06 | 1.09 | +2.79 |

Single-layer perturbation gains (24 MLPs, k=16384, eps=3e-3): **g_31 = 0.950, every other layer g_l ≤ 0.013**; Σ B_l = 8.14e-6 vs all-layer B = 8.50e-6. The final answer is a near-linear function of mu_31 and essentially blind to mu_1..mu_30.

**(3) Covariance anchoring — 16 MLPs, k=4096, independent reference stream, float64 accumulation**

| reference | scheme | MSE | C | VR |
|---|---|---|---|---|
| — | baseline w+a | 8.235e-06 | 3.373e-02 | 1 |
| — | oracle mu only | 1.354e-07 | 5.544e-04 | 60.8 |
| 65k | oracle mu + diag Σ | 6.959e-08 | 2.851e-04 | 118.3 |
| 262k | oracle mu + diag Σ | 6.637e-08 | 2.719e-04 | 124.1 |
| 1M | oracle mu + diag Σ | 6.640e-08 | 2.720e-04 | 124.0 |
| 65k | oracle mu + FULL Σ | 2.657e-08 | 1.088e-04 | 310.0 |
| 262k | oracle mu + FULL Σ | 2.217e-08 | 9.082e-05 | 371.4 |
| **1M** | **oracle mu + FULL Σ** | **2.090e-08** | **8.561e-05** | **394.0** |
| 65k / 262k / 1M | ref-mu + FULL Σ (**no oracle**) | 3.894e-07 / 9.943e-08 / 4.190e-08 | — | 21.1 / 82.8 / **196.6** |

**Claim of 272-341x / C=8.5e-5 CONFIRMED** (C = 8.561e-5 at a 1M reference, saturating — not reference-limited). But the oracle-mu column is the only one that reaches it; with the reference supplying the mean too, VR collapses to 197 and keeps falling with a smaller reference.

**(4) DECISIVE: required anchor accuracy.** 32 MLPs, L=31, MSE fitted as B + C/k from k=4096 and 32768.

| perturbation | B (bias floor) | VR@k=4096 | VR@k=32768 |
|---|---|---|---|
| exact oracle | ≤ 5.9e-10 | 70.9 | 53.3 |
| eps=1e-5 | 7.0e-10 | 70.9 | 52.9 |
| eps=1e-4 | 1.023e-08 | 65.5 | 31.2 |
| eps=1e-3 | 9.464e-07 | 7.06 | **0.75** |
| eps=3e-3 | 8.500e-06 | **0.86** | 0.08 |
| eps=1e-2 | 9.436e-05 | 0.08 | 0.01 |
| eps=1e-1 | 9.317e-03 | 0.00 | 0.00 |

**B = 0.945·eps² over 5 decades**, identical for both parameterizations (A·(1+eps·u): 0.879-0.969). Therefore:
- **eps for MSE = 1e-8 (score 1e-9 at the 0.1 floor): eps\* = 1.03e-04.**
- **eps at which anchoring stops helping: 2.79e-3 at k=4096, 8.71e-4 at k=32768, 1.81e-4 at k=1e6** (measured crossings bracket both).
- Ensemble's own sample-mean accuracy at layer 31: 0.261/√k → 4.1e-3 at k=4096. **eps\* is 40x tighter.**
- The "≈1% accurate" analytic mu_l gives B = 9.45e-5: **9450x the 1e-8 target and 13x worse than plain whitened+antithetic MC.**

**(5) The real anchor, priced.** Anchor = an independent w+a MC estimate at k_ref (32 MLPs):

| k_ref | 4096 | 16384 | 65536 | 262144 | 1048576 |
|---|---|---|---|---|---|
| B | 5.268e-06 | 1.280e-06 | 3.149e-07 | 8.620e-08 | 2.372e-08 |
| B·k_ref | 2.158e-02 | 2.097e-02 | 2.063e-02 | 2.260e-02 | 2.487e-02 |
| VR@k=4096 | 1.37 | 5.35 | 17.62 | 39.32 | 59.02 |

B ∝ 1/k_ref over 2.4 decades ⇒ **C_anchor-channel = 2.16e-02, cost 4.06e6/sample ⇒ FOM_anchor = 8.76e4** — vs the w+a baseline's own FOM 1.30e5 on the same MLPs. Acquiring mu_31 is the *same commodity at the same price* as acquiring mu_32. Optimal two-stage split (k_ref/k = 7.3):

| estimator | FOM | Phi | vs baseline | vs target 272 |
|---|---|---|---|---|
| w+a baseline (32 MLPs) | 1.298e5 | 4.77e-07 | 1.00x | 477x |
| **self-contained mean-anchored** | **1.141e5** | **4.19e-07** | **1.14x** | **419x** |
| **self-contained full-cov anchored** | **1.534e5** | 5.64e-07 | **0.92x** | 564x |
| free oracle mu, anchoring uncharged | 1.75e3 | 6.43e-09 | 74x | 6.4x |
| free oracle mu+Σ, anchoring uncharged | 359 | 1.32e-09 | 362x | 1.3x |
| free oracle mu+Σ, anchoring **charged** | 647 | 2.38e-09 | 201x | 2.4x |

Two independent reasons the headline oracle number does not become a score: (a) B=1e-8 needs k_ref = 2.16e6 anchor samples = 8.76e12 FLOPs = **32x the entire 2.72e11 budget**, and the dataset's own oracle was baked with 1e9 samples/MLP = **15,404x the budget**; (b) even with a *free* oracle, the full-cov anchoring operator itself costs ≥31·n² = 2.03e6 FLOPs/sample (Gram) + 31·22n³/k (two eigh), pushing FOM from 359 to 647 — the quoted Phi=7.75e-10 charged nothing for it.

**(6) Bootstrap escape hatch closed.** 64 MLPs, k=4096, reps=2: split into J sub-streams, anchor each to the pooled (or leave-one-out) mean at every layer — zero extra samples. VR = 0.987 / 0.961 / 0.865 / 0.631 / 0.291 for J = 2/4/8/16/32 (t = −1.22 to −7.15); leave-one-out identical. Self-anchoring is a strict loss, monotonically worse with more streams.

Scripts: `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/anch_lib.py`, `anch_02_head.py`, `anch_06_pert2.py`, `anch_07_boot.py`, `anch_08_cov.py`, `anch_09_gain.py`, `anch_10_only.py` (outputs `anch_06.out`, `anch_07.out`, `anch_08.out`, `anch_09.out`).

## 3. WHAT WOULD HAVE TO BE TRUE FOR ME TO BE WRONG

1. **An anchor source exists whose cost is not proportional to its accuracy squared.** Every measurement here prices the anchor at C_anchor = 2.16e-2 with 1/k_ref scaling. A deterministic (non-MC) mu_31 with relative error < 1.03e-4 at sub-budget cost would flip the verdict instantly. The only such candidate measured — analytic Gaussian propagation at ~1% — is 100x too coarse (B = 9.4e-5), and the marginal-moment class is already closed at MSE 6.2e-07.
2. **My g_l breakdown is an artifact of anchoring all layers simultaneously.** It is not: anchoring *only* layer 31 gives VR 48.1 against 50.6 for all 31 layers, measured independently (t=+11.26, 64 MLPs). If someone finds a scheme where shallow anchors matter, the cheap-shallow-layer argument reopens — but the measured g_l ≤ 0.013 for l ≤ 30 says the information simply is not there.
3. **The anchoring operator can be made free.** I charged ≥31·n² FLOPs/sample for the covariance Gram (assuming the transform is folded into W_{l+1}, the cheapest form I can construct). If the empirical covariance can be forced without ever forming it, the free-oracle full-cov FOM returns to 359 — still 1.3x above target, and still requiring an oracle costing 15,404 budgets to bake.
4. **B ∝ eps² breaks below eps = 1e-5.** I did not measure below eps=1e-5 (B there is at the ground-truth's own noise floor, ≤6e-10). All quoted eps\* values are interpolations *inside* the measured range, and k_ref = 2.16e6 is 2.1x beyond the largest measured k_ref (1.05e6) — within one decade.
5. One caveat I could not eliminate: the "exact" oracle A is itself a 1e9-sample bake with eps ≈ 8e-6, so the measured oracle floor B ≤ 5.9e-10 is an upper bound limited by the dataset, not by the method. This does not affect any conclusion (it is 17x below the 1e-8 threshold) but it means the true oracle VR may be slightly above 394.