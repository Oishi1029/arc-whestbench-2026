## VERDICT

**The isotropy assumption is false by 203x at chaos degree 1 — but the hole it opens is worth 1.34x and is *already fully collected* by the whitening in the live baseline. The residual the baseline actually leaves is even-degree≥4 chaos, whose effective dimension I measured directly at d ≳ 224, so the "d ≈ 252" conclusion SURVIVES. Priced estimator: FOM 1.18e5 vs baseline 1.02e5 (0.87x, i.e. **worse**), paired on 64 MLPs, t = −1.71.**

### 1. Degree-1 chaos G = E[f xᵀ] is radically anisotropic (64 MLPs, split-half debiased)

Stein vs. direct Jacobian on 600 exact Jacobians: σ₁ 3.6557 vs 3.6481 (0.2%), corr 0.988, Euler identity J(x)x = f(x) to 6.2e-06. Fraction of ‖G‖²_F in the top-d right-singular subspace (U fitted on one split, energy measured on two independent others):

| d | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| measured | .7912 | .8646 | .9320 | **.9717** | .9840 | .9863 | .9887 | .9928 |
| isotropic d/256 | .0039 | .0078 | .0156 | .0312 | .0625 | .1250 | .2500 | .5000 |
| **ratio** | **203x** | 111x | 60x | **31x** | 15.7x | 7.9x | 4.0x | 2.0x |

Mean σ₂/σ₁ = 0.303. **The claim VALUE(d) ≤ 1/(a₁(1−d/256)) is quantitatively wrong**: at d=8 it predicts removal of 3.1% of the degree-1 variance, actual is 97.2%, so VALUE(8) = 1/(1−0.26·0.972) = **1.34x, not 1.008x**. Degree-2 (E[f(xxᵀ−I)], projected, split-half): 17.2% captured at d=8 vs isotropic 0.098% = **176x**, 89.1% at d=128 vs 25% = 3.6x.

### 2. Why the hole is empty: the ladder, and what the baseline already annihilates

Two independent methods (Hermite-kernel identity Σ_{|α|=p}h_α(x)h_α(x′)=(x·x′)^p/p!, and the four samplers) agree:

| | V₁ | V₂ | V_odd≥3 | V_even≥4 | V_total |
|---|---|---|---|---|---|
| Hermite kernel | 25.92% | 22.46% | — | — | 0.04865 |
| four samplers | 26.92% | 22.51% | 27.02% | **23.55%** | 0.04728 |

Whitening forces sample mean = 0 and sample covariance = I **exactly**, so it annihilates 100% of degrees 1 and 2 *in all 256 dimensions, for free*; antithetic kills all odd degrees. Measured C = MSE·k: plain 0.04728, anti 0.04356 (1.085x), whitened 0.02391 (1.977x; predicted 0.02441), whitened+anti 0.02227 (**2.123x**; predicted 1/(2·0.2355) = 2.123x). The degree-1 anisotropy is therefore worth **exactly zero** on top of the deployed baseline.

### 3. ANOVA effective dimension of the residual — the decisive number

κ(d) = fraction of V_even≥4 (= 23.0% of V_total = 100% of the baseline's remaining variance) lying inside S. Conditional sampling, antithetic inner batches, all cross-split unbiased, 64 MLPs (48 for d≥160).

| subspace | d | Var(E[f_e\|S])/V_tot | V₂(S)/V_tot | κ | ceiling 1/(1−κ) | deg-4 CV cost | **net** |
|---|---|---|---|---|---|---|---|
| G | 8 | .0252 | .0274 | −0.009 ±.015 | 0.99x | +0.02% | 0.99x |
| G | 64 | .0800 | .0862 | −0.027 ±.050 | 0.97x | +55% | 0.63x |
| act (E[JᵀJ]) | 8 | .0386 | .0386 | 0.000 ±.026 | 1.00x | +0.02% | 1.00x |
| act | 64 | .1629 | .1437 | 0.083 ±.094 | 1.09x | +55% | 0.70x |
| act | 128 | .2811 | .2001 | 0.354 ±.159 | 1.55x | +838% | 0.17x |
| act | 160 | .3448 | .2138 | 0.571 ±.183 | 2.33x | +2027% | 0.11x |
| act | 192 | .4021 | .2214 | 0.788 ±.201 | 4.73x | +4178% | 0.11x |
| act | 224 | .4473 | .2243 | **0.973 ±.213** | 37x | +7706% | 0.47x |
| rand | 32 | ρ=.0419 (all deg) | — | — | — | — | control ✓ |

κ is **near-linear in d** (+0.0066/dim, measured not extrapolated) — the degree-4+ residual is essentially isotropic. κ = 0.9945 (the 183x target) requires **d ≳ 224**, at which a degree-4-exact rule needs C(227,4) = **1.08e8** basis functions = 7706% of the forward-pass cost it is replacing.

### 4. Priced estimator (whitened+antithetic + cross-fitted degree-4 Hermite CV on top-8 of G, k = 65536, 64 MLPs, reps = 4, n = 256 paired)

| | MSE | C=MSE·k | c/sample | FOM | Φ |
|---|---|---|---|---|---|
| baseline (wa) | 3.4921e-07 | 0.02289 | 4.4646e6 | **1.022e5** | 3.76e-07 |
| + subspace CV | 3.6087e-07 | 0.02365 | 4.9922e6 (+11.8%) | **1.181e5** | 4.34e-07 |

Variance ratio 0.9677 (t = −1.71) — the CV removes **nothing**, and loses the M/k = 1.0% coefficient-estimation inflation. FOM ratio 0.865. Rescaled to the live cost basis (c = 1.8e6): baseline 4.12e4 ≈ the live 4.99e4; CV 4.76e4. **Target 272 is 151x away and this route moves the wrong direction.**

Ancillary negative: the subspace is **not free from the weights**. The raw product Π = W₁···W₃₂ captures only 16.1% of degree-1 energy at d=8 (vs 97.2% for the sampled G); |cos∠| between top singular vectors = 0.105. A pilot is mandatory.

### 5. What would have to be true for me to be wrong

- **The only real gap**: I tested two *principled* subspaces (top-d of E[∇f], top-d of E[JᵀJ]) plus a random control. Neither is optimized for degree-4 energy. A subspace built from the degree-4 chaos Gram directly could carry more; my κ(8) = 0.000 ±0.026 bounds only the tested subspaces. But the residual would have to be ~40x more degree-4-concentrated than degree-2 is (176x at d=8) *in a different subspace*, while degree-2 concentration itself already decays 176x→3.6x from d=8→128.
- κ(224) = 0.973 ± 0.213 is the loosest number here; "d ≳ 224" is honest, "d ≈ 252" is not separately confirmed. I measured d = 224 directly rather than extrapolating.
- If whitening did *not* exactly annihilate degrees 1–2, the degree-1 anisotropy would be worth 1.34x. Measured C_whitened = 0.02391 vs predicted 0.02441 (2%), so it does.

Scripts (all absolute): `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/` — `edim1_verify.py` (Stein↔Jacobian), `edim2_spec.py` (G spectrum), `edim3_degrees.py` (Hermite ladder), `edim4_anova.py` (ρ(d)), `edim5_base.py` (four-sampler decomposition), `edim6_kappa.py` + `edim9_bigd.py` (κ(d)), `edim7_free.py` (weight-product subspace), `edim8b_price.py` (priced estimator).