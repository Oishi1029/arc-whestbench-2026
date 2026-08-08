All measurements complete.

**VERDICT: The degree-5 Mysovskikh cubature is exact to 1.8e-15 and lands at FOM 3.11e6 — 70.5x WORSE than the whitened+antithetic baseline it must beat (paired t = 15.9, 128 pairs, 64 MLPs). The class is dead, not just this construction: the degree-4 Hermite chaos carries only 35% of the baseline's residual variance, so the BEST CONCEIVABLE degree-5 rule (equal weights, ESS = node count) caps out at FOM 2.94e4 = 1.7x better than live and 108x above the 272 target. Closing degree-5, and degree-7 with it.**

---

## 1. The rule is real and exact (`d5_rule.py`, `d5_verify.py`)

Mysovskikh / Lu–Darmofal, n=256, 66,307 nodes = {0} ∪ {±r aᵢ} ∪ {±s b_ij}. Closed-form weights derived from the identities Σ_{i<j}(αᵢ+α_j)⁴ = (n−7)S₄ + 3S₂², Σ_{i<j}(αᵢ+α_j)² = (n−1)S₂:

  w₂s⁴ = 2(n−1)²/(n+1)²,  **w₁r⁴ = −n²(n−7)/(2(n+1)²)** — negative for every n > 7.

Verified after a random Householder/QR rotation, float64, worst relative error over all degrees ≤ 5:

| test | true | rule | rel err |
|---|---|---|---|
| x₁⁴ | 3 | 3.00000000 | 1.5e-16 |
| x₁²x₂² | 1 | 1.00000000 | 2.2e-16 |
| x₁³x₂ , x₁x₂x₃x₄ , x₁⁵ | 0 | ~1e-16 | ≤3.1e-16 |
| ‖x‖², ‖x‖⁴ | 256, 66048 | exact | 0.0 |
| (u·x)⁴, (u·x)²(v·x)² | — | — | 1.8e-15, 6.7e-16 |
| random deg-2/3/4/5 forms | — | — | ≤1e-16 |
| **worst, degree ≤5** | | | **1.835e-15** |
| [deg-6 x₁⁶ — must fail] | 15 | 14.662 | 2.3e-2 ✔ |

Structural find: the free radius parameter is pinned by **t = r² = n+2 = 258, where r = s = 16.0624** — the rule collapses to a single shell (a weighted spherical 5-design) plus an origin atom of weight 1/129. f(0)=0 for a bias-free ReLU net, so the origin is free: k = 66,306 evaluated nodes.

**A random rotation does NOT make it unbiased.** Randomising Q randomises only the angular part; the radial quadrature stays a fixed 2-shell rule. Since f is exactly positively homogeneous of degree 1, E_Q[rule] = (Σ w_p‖x_p‖)·E_u[g] exactly, so the relative bias is the scalar Σ w_p‖x_p‖/E‖x‖ − 1 = **−2.910e-3**, and 2.910e-3 is the *maximum over t* — it cannot be zeroed. Alone that is an MSE floor of 7.4e-6, 20x above the baseline MSE at the same node count. I removed it exactly by rescaling with E‖x‖ (legitimate: homogeneity is exact). Both variants are reported.

## 2. Headline measurement — 64 MLPs × 2 reps, common seeds (`d5_main.py`)

| estimator | MSE | k | c (FLOPs/node) | **FOM = MSE·k·c** | Φ |
|---|---|---|---|---|---|
| whitened+antithetic MC, k=66306 | 3.6955e-07 | 66306 | 2.065e6 | **5.060e4** | 1.860e-07 |
| **degree-5 Mysovskikh, corrected** | 2.6048e-05 | 66306 | 1.802e6 | **3.113e6** | 1.144e-05 |
| degree-5, uncorrected (raw rotated) | 3.2851e-05 | 66306 | 1.802e6 | 3.925e6 | 1.443e-05 |
| midpoint block only (λ=0) | 3.4231e-06 | 65792 | 1.802e6 | 4.059e5 | 1.492e-06 |
| simplex block only (λ=1) → deg-3 | 4.7004e-05 | 514 | 1.953e6 | 4.383e4 | 1.612e-07 |

**Paired, degree-5 vs W+A: MSE ratio 70.49x, t = 15.92, n = 128 pairs.** The baseline reproduces the live submission to 1.4% (measured C = MSE·k = 0.02450 vs 0.0277 quoted), so this is not a broken harness.

**Why it fails — measured effective sample size** (ESS ≡ C_baseline/MSE):

| node set | nodes | ESS | ESS/nodes |
|---|---|---|---|
| simplex block | 514 | 521 | **1.014×** |
| midpoint block | 65,792 | 7,158 | 0.109× |
| full degree-5 rule | 66,306 | 941 | **0.0142×** |

After the homogeneity reduction the rule is exactly `μ₁·(λ·mean_simplex + (1−λ)·mean_midpoint)`, and degree-5 exactness *forces* **λ = −0.9614** — a 2x extrapolation that puts ~half the weight mass on 514 of the 66,306 nodes. The λ-scan shows no member of the family is competitive:

| λ | −1.0 | **−0.9614 (deg-5)** | −0.5 | **−0.10 (oracle best)** | 0 | +1.0 |
|---|---|---|---|---|---|---|
| MSE ratio vs W+A | 76.5 | **70.5** | 19.7 | **7.66** | 9.26 | 127 |

Even the oracle-fitted best λ is 7.7x worse than plain MC.

## 3. Degree-3 control — 64 MLPs × 6 reps, 384 pairs (`d5_deg3.py`)

| estimator | MSE | k | c | FOM | vs live |
|---|---|---|---|---|---|
| rotated simplex m=1, corrected | 4.367e-05 | 514 | 1.953e6 | 4.383e4 | 1.14x |
| rotated simplex m=1, uncorrected | 4.429e-05 | 514 | 1.953e6 | 4.445e4 | 1.12x |
| **rotated simplex m=8, corrected** | 5.346e-06 | 4112 | 1.820e6 | **4.000e4** | **1.25x** |
| W+A MC k=514 | 4.493e-05 | 514 | 2.421e6 | 5.592e4 | 0.89x |
| W+A MC k=4112 | 5.496e-06 | 4112 | 2.107e6 | 4.762e4 | 1.05x |

Control passes: variance is statistically **indistinguishable** from whitening+antithetic (MSE ratio 0.972, t = −0.46; and 0.973, t = −0.52 at m=8). The machinery is correct.

Side result, honest but small: a stack of 8 rotated simplices under one Haar rotation is a *drop-in replacement for whitening* that is degree-3 exact, unbiased, and skips whitening's 2.65e5 FLOPs/sample (14.7% overhead) for 1.9e4 (1.1%). **FOM 4.00e4 vs live 4.99e4 = 1.25x, Φ 1.471e-07.** That beats the published best (4.2e4) but it is a cost lever, not a variance annihilator — 147x still short of target.

## 4. The ceiling that kills the whole class (`d5_chaos.py`, `d5_refit.py`)

Measured C(ρ) = Cov(f(x), f(x_ρ)) = Σ_d V_d ρ^d on 32 MLPs at K=32768, with ρ=±1 computed exactly:

| quantity | value |
|---|---|
| Var(f) per sample | 0.049834 |
| Cov(f(x), f(−x)) | −0.004719 |
| even mass Σ_{d even} V_d (exact) | 0.022557 (45.3%) |
| odd mass (killed exactly by antithetic) | 0.027277 (54.7%) |
| V₂ (killed by whitening) | 0.00940 |
| **V₄** | **0.00468** |
| tail d≥4 (= W+A residual) | 0.01316 |
| tail d≥6 | 0.00849 |

Consistency: predicted C_WA = 2(even − V₂) = 0.02632 vs measured 0.02450 — **7.4% agreement**, so the decomposition is real.

| | gain at full efficiency | best possible FOM | x-to-target |
|---|---|---|---|
| degree-3 rule | 1.00x | 4.41e4 | 162 |
| **degree-5 rule** | **1.55x** (bootstrap 1.455 ± 0.028; model-free bound ≤1.41) | **2.94e4** | **108** |
| degree-7 rule | 2.22x | 1.99e4 | 73 |
| **gain needed to reach 272** | **162x** | 272 | 1 |

Regularisation-independent: gain(5) = 1.551 for every α from 0 to 1e-2.

Two supporting arithmetic facts:
- Degree-4 exactness is 183,181,376 linear conditions (monomials of degree 4 in 256 vars); an N-node rule has 257N parameters, so a *generic* degree-5 rule needs N ≥ 712,768. Möller's bound is 66,305. **Any O(n²) degree-5 rule must therefore be an orbit of a large symmetry group** — which is precisely why its 66,306 nodes are generated by only 257 base directions and the ESS collapses to 941.
- For a **positive-weight** rule, setting v = u_q in Σ ω_p(u_p·v)⁴ = 3/(n(n+2)) gives ω_max ≤ 2.271e-5, hence N ≥ ESS ≥ 44,032 — at most a 1.5x ESS penalty. The 70x loss is entirely attributable to the *forced negative weights* (w₁r⁴ < 0 for all n > 7). Constructive positive degree-5 rules of O(n²) size in n=256 are not known, and even one would only buy 1.55x.

## 5. What would have to be true for me to be wrong

1. **V₄ would have to be ~99.4% of the W+A residual variance** instead of the measured 35.5%. My three independent handles (regularised fit 1.55, MLP bootstrap 1.455 ± 0.028, model-free interval bound ≤1.41) would all have to be wrong by two orders of magnitude. The 7.4% agreement between the predicted and measured baseline variance argues against that.
2. **A positive-weight degree-5 cubature with ~66k nodes would have to exist and be constructible.** That would recover the 47x ESS loss — but it only buys the 1.55x ceiling, landing at FOM 2.94e4 (108x from target). So even being wrong here does not reach 272.
3. My ρ-curve is measured at K=32768 on 32 MLPs; the small-ρ points have 5–14% relative noise, which is why I quote a range for gain(5) rather than a point. A 10x larger K would tighten 1.41–1.56 but cannot move it near 162.
4. **The one thing I did NOT rule out**: the whole analysis assumes the estimator averages i.i.d.-in-Q evaluations of the *unmodified* network. Anything that changes the integrand (internal moment anchoring, which the brief records at 272–341x with an oracle) is untouched by this ceiling — the chaos spectrum I measured is a property of f, and the only way past 1.55x is to change f, not to integrate it better.

Files: `/private/tmp/claude-501/.../scratchpad/d5_rule.py` (construction), `d5_verify.py` (exactness), `d5_tscan.py` (radius/bias structure), `d5_est.py` (estimators + cost model), `d5_main.py` (headline), `d5_deg3.py` (control), `d5_chaos.py` + `d5_refit.py` (ceiling). Logs: `d5_chaos.log`, `d5_deg3.log`; caches `d5_cache.npz`, `d5_chaos.npz`.