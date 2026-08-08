**VERDICT — the deterministic branch is dead by 360x, but the reason moves: the reconstruction step is NOT the blocker (max-entropy on 6 exact moments = bias² 5.08e-09, parameter-free, below the 1e-08 requirement). 100% of the blockage sits in acquiring the ORDER-1 moment, which is a bijection of the whole problem one layer up. Best measured deterministic estimator inside 2.72e10 FLOPs: raw MSE 3.60e-06 → adjusted 3.60e-07 (equiv FOM 9.8e4), 2.0x WORSE than the live submission and 360x above the 1e-08 requirement.**

---

## Measurement base
32 real MLPs, 2 independent MC streams × 1e6 samples each (plus a full second dataset at 2.5e5 for K-stability), float32 forward (f32-vs-f64 forward difference: MSE 8.6e-18, irrelevant), float64 moment accumulation to order 12. Bias/noise separated with `D = est − same-stream empirical mean`, two-stream variance subtraction; 8192 neuron-units per number.
**My MC provably converges to the dataset `F`** — MSE(2-stream avg vs F) = 5.20e-08 → 2.00e-08 → 2.26e-09 for K = 3.28e5 / 9.83e5 / 4.92e6 per stream, i.e. 23x drop for a 15x K increase. No floor; `F` is not the limiter.

## 1. Accuracy ladder, exact moments (32 MLPs, 8192 units)

| m | GC/Hermite bias² | ±se | vs 1e-08 |
|---|---|---|---|
| 2 | 1.1098e-06 | 1.7e-08 | 111x |
| 3 | 1.6128e-07 | 2.5e-09 | 16x |
| **4** | **2.3037e-08** | **3.6e-10** | **2.30x  ← minimum** |
| 5 | 3.0665e-08 | 4.8e-10 | 3.07x |
| 6 | 2.7340e-08 | 4.3e-10 | 2.73x |
| 7 | 3.2227e-08 | 5.3e-10 | 3.22x |
| 8 | 3.7452e-08 | 6.7e-10 | 3.75x |
| 9 | 1.6006e-07 | 3.3e-09 | diverging |
| 10 / 12 | 4.918e-07 / 5.12e-06 | | diverged |

- **Confirms 2.5e-08 at order 4** (I get 2.304e-08, 8% lower, ±1.6%).
- **Refutes the 1.867e-08 minimum at order 7**: order 7 = 3.22e-08, 1.4x *worse* than order 4. Orders 5–8 are a plateau at 2.7–3.7e-08; divergence starts at **order 9**, not 7.
- **Edgeworth ≡ Gram-Charlier here, bit-identical** (1.6128e-07 / 2.3037e-08 / …). There is no n^(−1/2) to reorder by, so the two truncations coincide by construction. That closes "try a proper Edgeworth".
- K-stable: m≤6 agrees to 3 digits between K=2.5e5 and K=1e6.

## 2. Better 1-D reconstructions from the same exact moments

| method | M=4 | M=6 | M=8 | M=10 |
|---|---|---|---|---|
| Gram-Charlier (closed form) | 2.304e-08 | 2.734e-08 | 3.745e-08 | 4.92e-07 |
| **Max-entropy (0 fitted params)** | 5.259e-08 | **5.077e-09** ±1.3e-10 | **8.148e-10** ±2.3e-11 | fails (grid) |
| Fitted Hermite, linear (p=2/4/6) | 1.606e-08 | 7.532e-09 | 3.082e-09 | K-unstable |
| Fitted Hermite, a-modulated (p=4/8/12) | 1.557e-08 | 3.596e-09 | 2.174e-09 | K-unstable |
| Gauss quadrature (Golub–Welsch) | 1.43e-05 (5 mom) | 8.33e-06 (7 mom) | 5.33e-06 (9 mom) | 3.76e-06 (11 mom) |

- **MAJOR: max-entropy with 6 exact moments reaches 5.08e-09**, below the 1e-08 requirement, with **no fitted constants**. K-stable: 5.785e-09 (K=2.5e5) vs 5.827e-09 (K=1e6) on a 6-MLP subset (ratio 0.993). M=8 → 8.15e-10, also K-stable (9.81e-10 vs 1.04e-09, ratio 0.945; 85% Newton convergence, rest falls back to GC-4).
- Fitted reconstructions are held out on 16 unseen MLPs and are K-stable up to M=8 (ratios 0.91–1.00). **M≥10 and the 33-parameter variant are K-UNSTABLE (ratios 0.29 and 0.07)** — I reproduced 7.0e-11 at M=12 and showed it moves **40x** when K changes 4x. The previously reported ~9.2e-09 at 20 coefficients on 4 MLPs sits in exactly that unstable regime; it is an O(1/K) plug-in artefact, not a real bias.
- **Gauss quadrature is refuted**: 3.76e-06 with 11 moments, 160x worse than GC at order 4 with 3x the moments. Cause measured on a standard normal: Gauss-rule error for ReLU decays only O(1/n) (0.101, 0.045, 0.029, 0.021, 0.017 at n=2,4,6,8,10 nodes) because no node lands on the kink.
- **Rigorous Markov–Krein minimax** (my own LP; sharp min/max of E[ReLU] over *all* laws matching the moments, 200 real neurons): mean squared half-width 3.43e-04 / 7.93e-05 / 3.95e-05 / 2.51e-05 / 1.73e-05 / 1.29e-05 at M=2/4/6/8/10/12, decay **M^−1.65**. So no moment method with M≤12 can *guarantee* better than 1.29e-05; the 5.08e-09 maxent result is 2500x inside the band and is pure exploitation of the He-init/depth-32 family. (Extrapolating to a 1e-08 guarantee gives ~900 moments — three decades of extrapolation, flagged as unreliable, but consistent with the m^−1.7 / 800-moment claim.)

## 3. Cost — where it actually dies

**(a) The accuracy burden is entirely on moments 1 and 2.** Perturbing each input of the order-6 fitted reconstruction (held-out, 16 MLPs):

| perturbed | 1e-5 | 1e-4 | 1e-3 | 1e-2 |
|---|---|---|---|---|
| μ (rel. to σ) | 3.66e-09 | 4.14e-09 | 5.20e-08 | 5.00e-06 |
| σ (relative) | 3.65e-09 | 3.65e-09 | 4.26e-09 | 8.10e-08 |
| c₃, c₄, c₅, c₆ | 3.65e-09 | 3.65e-09 | 3.65e-09 | **3.65–3.68e-09** |

Fitted law: MSE ≈ 0.048·(δμ/σ)². **μ must be known to 4.6e-04·σ; c₃…c₆ need only ~1% relative accuracy.**

**(b) Order 1 has nothing to sketch.** μᵢ = wᵢ·E[y₃₁] and W₃₂ is invertible, so the 256 order-1 marginal moments of z₃₂ are a *bijection* of E[y₃₁] — the entire problem one layer up. Measured transfer: E[y₃₁]→E[z₃₂] amplification **2.001** (= mean‖wᵢ‖²), E[z₃₂]→E[ReLU] attenuation **0.461** (= mean Φ(a)²), **net 0.922**. No attenuation, no shortcut. Randomised/sketched contraction can only address orders ≥3 — exactly the orders that need 1% accuracy and are already nearly free. **The hole does not exist.**

**(c) Sampled marginal moments give zero variance reduction** (64 MLPs, identical forward pass so identical c):

| k | C_plain | GC m=4 | m=6 | m=8 | m=10 |
|---|---|---|---|---|---|
| 4096 | 0.04320 | ×1.001 | ×1.005 | ×1.033 | ×1.381 |
| 16384 | 0.04640 | ×1.009 | ×1.015 | ×1.051 | ×1.718 |

Paired (CRN) plain vs GC-6: ratio 0.9956, **t = −5.09** (k=4096); 0.9872, **t = −6.60** (k=16384) — plain MC is significantly better. Cause: dR/dμ = Φ(a), so the plug-in inherits the full sample-mean variance.

## 4. Best deterministic estimator inside 2.72e10 FLOPs (metered)

| estimator | npts | cost/B | raw MSE | adjusted |
|---|---|---|---|---|
| axis cubature, degree 3 | 512 | 0.0079 | 4.767e-05 | 4.77e-06 |
| ±Hadamard rows, rotated (deg 3) | 512 | 0.0079 | 3.668e-05 | 3.67e-06 |
| rot-Hadamard stack ×4 | 2048 | 0.0315 | 1.114e-05 | 1.11e-06 |
| **rot-Hadamard stack ×12** | **6144** | **0.0946** | **3.602e-06** | **3.60e-07** |
| plain MC, same npts | 6144 | 0.0946 | 6.399e-06 | 6.40e-07 |
| radial(chi-n Gauss)×Hadamard | 6144 | 0.0946 | 1.238e-05 | 1.24e-06 |

Deterministic point sets scale as 1/npts (no acceleration); the fixed gain over plain MC is 1.78x. Best deterministic **raw MSE 3.60e-06 = 360x above the 1e-08 requirement**; adjusted 3.60e-07 ⇒ equivalent FOM 9.8e4, **2.0x worse than the live submission (FOM 4.99e4)** and 360x worse than the target. If the moments were free, maxent-M6 would give adjusted 5.1e-10 — at target — which is precisely the measure of how much the acquisition step costs.

## What would have to be true for me to be wrong

1. **A method delivering E[y₃₁] to rms 1.0e-04 (MSE 1.1e-08) within 2.7e10 FLOPs.** That is now the *only* requirement — orders 3–6 need 1% and cost almost nothing. My best deterministic value for that same quantity is MSE 3.6e-06, **330x short**. Anything claiming the deterministic branch works must produce this number.
2. My LP minimax bands are computed on a discretised support grid, which can only *narrow* them; 26% of true F values fall outside my M=10–12 bands, so the true minimax is wider and the negative is conservative.
3. Maxent M=8 runs at 85% Newton convergence (unconverged neurons fall back to GC-4); treat 8.1e-10 as ~1e-09. The M=6 headline (96% convergence, K-stable across 4x K) is solid.
4. My MC could share a bias with `F`. Ruled out to 2.3e-09 by the 1/K convergence test above.

Scripts (all absolute, prefix `dt_`): `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/` — `dt_01_moments.py` (moment collection), `dt_lib.py` (GC/Edgeworth/Wheeler), `dt_me2.py` (max-entropy), `dt_lp.py`+`dt_09_lp.py` (Markov–Krein LP), `dt_05_analysis.py` (ladder), `dt_18_kdep2.py` (K-stability), `dt_19_sens.py` (moment sensitivity), `dt_07_cost.py` (sampled-moment FOM), `dt_10_layers.py` (bijection/amplification), `dt_11_floor.py` (deterministic floor), `dt_15b_conv.py` (MC-vs-F convergence). Moment caches: `dt_mom32.npz`, `dt_mom32_k250k.npz`.