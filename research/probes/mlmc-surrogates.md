**VERDICT: no exactly-integrable surrogate moves FOM. Best measured FOM = 4.06e4 (= unchanged baseline); the best surrogate (layer-1 CV) lands at FOM 4.29e4, a 5.6% *regression* after its own cost. Target 272 needs 149x; the entire exactly-integrable class delivers 1.02x (t=0.64, n=240, not significant).**

## 1. Baseline (whitened+antithetic, 60 MLPs, k=8192, reps=4)
MSE 2.7555e-06 → C = MSE·k = 2.257e-02 → FOM_naive (c=4.2e6) 9.48e4, **FOM_live (c=1.8e6) 4.06e4**. Reduction needed to hit FOM 272: **149x**.

## 2. Surrogates, ranked by measured Var(f−g)/Var(f)

| # | surrogate g | Var(f−g)/Var(f) | paired pipeline gain | net FOM |
|---|---|---|---|---|
| 1 | **(a) affine in y₁**, E[g] exact = ‖W₁[:,i]‖/√(2π) | 0.599 (raw, plain x) | **1.016x, t=0.64, n=240** | 4.29e4 (**0.95x = worse**) |
| 1b | (a)+quadratic forms in y₁ (arc-cos kernel, exact) + 128 Jacobian-informed + 256 random ridges relu(vᵀx), p=1040 exact-mean controls | — | 1.054 → **1.060** (n=120, t≤1.9) | saturated; 784 extra exact controls buy 0.5% |
| 2 | **(b) single-hub linearisation** g=J(x₀)x | — | **exactly 0** — mean(g) over a whitened batch = 8.6e-08 (float32 roundoff) vs signal 1.03. Bit-identical. | 1.000x |
| 3 | (b) multi-hub, nearest by cos | 21.5 (c=0) · 6.4 (c=.9) · 1.50 (c=.99) · **0.505 (c=.999)** · 0.049 (c=.99999) | hubs needed: 1e92 / 1e217 / **1e344** / 1e599 | dead by 10³⁴⁴ |
| 4 | **(c) GP surrogate**, g=relu(ζ), ζ~N(μ₃₂,σ₃₂) **oracle marginals, best coupling over all couplings** | **0.0086–0.0115** (Σᵢ W₂² / trCov f, K=65536, 5 MLPs) | **ceiling 87–116x** | best case FOM **349** — still above 272, and unconstructible |
| 5 | **(d) low-rank W_l→rank r**, coupled on same x | r=16/32/64: **1.000** (Var(g)=0, net collapses) · r=128: 0.938 · r=192: 0.821 · r=224: 0.355 | MLMC FOM ratio ≥ **1.29** (best, r=128) | always worse |
| 6 | **(d) depth-truncated y_L** with *oracle* E[y_L] | R(L)=1.09/1.05/1.14/1.21/1.80/2.98/4.79/7.57/16.0/35.2/**67.3** at L=1/2/4/8/12/16/20/24/28/30/31 | MLMC FOM ratio min **1.017** (L=1); never <1 | closed |

## 3. Why (the two numbers that explain everything)

**Availability vs value are on opposite ends of the net.** Closed-form Gaussian integration over N(0,I₂₅₆) covers ridge functions h(vᵀx), *pairs* of ReLU ridges (arc-cosine kernel), and polynomials — i.e. exactly **one ReLU layer**; three-way ReLU products need trivariate orthant probabilities with no closed form. Measured value of an exact mean by layer: **L1 = 1.09x, L16 = 2.98x, L31 = 67.3x**. Exact means exist only where they are worth 1.09x.

**Whitening+antithetic already pins layer 1.** MSE(mean(y₁) − μ₁ᵉˣᵃᶜᵗ): plain 1.67e-04 (theory 1.653e-04) → wa **1.09e-05**, a 15.3x pre-existing reduction. That is why (a) has nothing left to correct. Structurally: relu(z)² = (z²+z|z|)/2, whitening pins the z² term exactly and antithetic annihilates z|z|, so mean‖y₁‖² is *already exact* — and ‖y_l‖² alone explains 85.6% of Var(f) (L1 21%, L32 85.6%).

**Ceilings, for calibration:** scalar oracle Σy_l saturates at 1.82x even at L=31; the full 256-vector oracle at L=31 gives FOM **604**; the perfect GP coupling gives FOM **349**. Both exceed the 272 target. The blocker for the GP class is measured non-Gaussianity of z₃₂: skew ≈ 0, **excess kurtosis +0.296 to +0.441**.

**The GP coupling deadlock, made concrete.** The only constructible exactly-Gaussian coupling is the rank/quantile map, and it makes mean(g) a function of (m̂,ŝ) only (measured batch-to-batch std of mean(g) = 9.2e-04, entering *solely* through the two moments) — it collapses to the (μ,σ) plug-in class whose floor is the already-closed MSE 6.2e-07. A value-based coupling fluctuates with the sample but is then not exactly Gaussian, so E[g] is not exactly computable. No middle ground was found.

## 4. What would have to be true for me to be wrong
1. **A closed form for E[∏ of ≥3 ReLU ridges] exists.** Even then it only extends exactness to layer 2, worth **1.05x**. You would need closed forms at layer ≥28 (16x) — a 28-fold ReLU composition integral.
2. **The W₂ bound is loose.** It can undershoot Var(f−g) only by ‖E[f−g]‖² ≈ 256·6.2e-07 = 1.6e-04, i.e. **0.05%** of the measured bound 0.29. Not loose.
3. **z₃₂ is nearer Gaussian than measured.** Excess kurtosis would have to fall below ≈0.05 for the GP class to reach 149x. Measured 0.296–0.441 at K=65536 across 5 MLPs — not sampling noise.
4. **Hubs can be placed adaptively far better than random packing.** Against this: at a *random* hub the layer-32 mask agreement is already **0.937–0.973** and yet Var(f−g)/Var(f) = 21.5. The residual 3–6% of flipped masks alone destroys the coupling, so better hub geometry in x-space is not the missing ingredient.

Scripts (all absolute): `/private/tmp/claude-501/-Users-binyong-.../scratchpad/sg_lib.py` (exact μ₁, arc-cosine Cov₁), `sg_02_pipe.py`, `sg_04b_oracle.py` (per-layer oracle ceiling map), `sg_05_lin.py` (bit-identity + e(c)), `sg_06_rank.py`, `sg_07_rich.py` (p=1040 exact-mean dictionary), `sg_08_gp.py` (W₂ ceiling), `sg_09_final.py` (headline paired + depth-MLMC closure).