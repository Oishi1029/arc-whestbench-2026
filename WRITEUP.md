# Exactness is available exactly where it is worthless

### A measured map of the WhestBench estimator space, and the protocol that produced it

**ARC White-Box Estimation Challenge 2026 — Phase 1 algorithmic-contribution write-up**
Task: per-neuron mean activation of the final layer of a bias-free ReLU MLP, width 256, depth 32.

| | |
|---|---|
| Best public submission | **#325573**, adjusted final-layer score **1.730e-07** (raw final-layer MSE 2.55e-07, utilisation 0.678) |
| Estimator | `work/final/sweep.py` — whitened, antithetic, float32 Monte-Carlo sample mean with eight exact cost reductions |
| Reliability | 0/50 public MLPs failed; 0/1000 on the local `full` split; worst-case budget utilisation 0.9040 on adversarial inputs |
| Trajectory | 3.405e-07 → 2.700e-07 → 2.240e-07 → 1.834e-07 (#324358, rank #54) → **1.730e-07** |
| Constructible frontier | rank 4 at **4.62e-08**. We finish **3.7× above it** and cannot account for the gap — see §9. |

All numbers are dated 2026-08-07 unless stated. Where a number describes a competitor it is
arithmetic on figures **they published**, and it is labelled as such.

---

## Read this page only, if you read nothing else

**What we shipped.** A whitened, antithetic, float32 Monte-Carlo sample mean with eight exact cost
reductions. Adjusted final-layer score **1.730e-07** (submission #325573), 0/50 public MLPs failed,
0/1000 on the local split. The variance devices are worth **2.17×** over plain MC at equal compute;
the cost reductions roughly another **3.1×**. None of the ingredients is novel and we do not claim
them as such.

**What we contribute.** A measured map of where this problem's walls are, with the experiment that
located each. The organising result:

> Closed-form Gaussian integration in this problem reaches **exactly one ReLU layer**. The value of
> an exact layer mean is **1.13× at layer 1** and **50.8× at layer 31**. *Exactness is available
> exactly where it is worth ~1.1×, and would be worth ~50× exactly where it does not exist.*

Everything else is that antisymmetry in a particular coordinate system:

| result | § |
|---|---|
| Max-entropy reconstruction from **six exact moments** reaches bias² = **5.08e-09**, zero fitted parameters — so the deterministic branch's reconstruction step is *solved*, and the blockage is entirely in acquiring the first two moments of `z₃₂`, which are the answer one layer up | 3.1 |
| A **model-free** bound (no fit, no extrapolation) puts the constructible frontier at cubature **degree 5** — the last rung that fits the budget — and a 10× improvement at **degree 7**, which costs **18.9 budgets for one evaluation**. The cost of exactness steps by 86× between the rung that suffices and the rung above it | 4.2 |
| We **built** the degree-5 rule (66,307 nodes, exact to 1.8e-15) and measured it **70× worse**. Degree-5 exactness *forces* `w₁r⁴ = −n²(n−7)/(2(n+1)²) < 0`, collapsing effective sample size to **941 of 66,306**. Any positive-weight rule would pay ≤1.5× | 4.3 |
| An **exact** multilevel telescoping identity — every level exact, no coarse surrogate — that loses anyway by a two-line allocation argument, for **any** plausible weight profile (12.3–16.5×) | 4.6 |
| A published mechanism reproduced and its stated explanation refuted: the sample mean **commutes** with a linear map, measured **bit-identical, 0.000e+00**, so identifying a neuron as linear cannot reduce variance. The real gain is cost, and at fixed budget the two are perfectly confounded | 5 |
| The general form: **any estimator that is a smooth function of the same sample's own statistics re-derives the sample mean.** Three independent measured instantiations | 2.4 |
| **ICC across independent streams** as the discriminator between a correctable bias and heteroscedastic noise. 30.67% of our error energy lies along one direction, 78.5× isotropic, with **ICC = −0.011**. We were about to build the corrector | 7.3 |

**Two open problems we could not close, stated precisely in §9.** (A) Is there a *positive-weight*
degree-5 cubature rule on `N(0, I₂₅₆)` with `O(n²)` nodes? (B) Can `μ₃₁` be obtained to `1.03e-04`
relative, and the layer-31 second-moment matrix to `3e-3`, below the cost of estimating `μ₃₂`
directly?

**What we get wrong.** §7 lists six conclusions of ours that later measurement overturned,
including two headline figures in this document's own earlier draft that we could not re-derive and
have replaced. We finish **3.7×** above the measured frontier and cannot account for the gap.

---

## Abstract

The scoring rule `Φ = mse × max(0.1, compute/B)` is flat in sample count above its floor, so an
estimator has exactly two knobs: the variance constant `C` and the per-sample cost `c`. We map
both to their walls and report where the walls are, because we could not get past them.

The organising result is an antisymmetry. Closed-form Gaussian integration in this problem
reaches **exactly one ReLU layer** — ridge functions, pairs of ridges through the arc-cosine
kernel, and polynomials; three-way ReLU products need trivariate orthant probabilities, which have
no closed form. But the value of an exact layer mean rises steeply with depth: anchoring the
ensemble onto exact means gives **1.13× at layer 1 and 50.8× at layer 31**. Layer 1 is the only
one we can actually have, and shipping it is worth **1.083×** (paired, n = 1000, `t` = +5.45).
*Exactness is available exactly where it is worth ~1.1×, and would be worth ~50× exactly where it
does not exist.* That antisymmetry, and not a node count, is the structure of the problem.

Four independent probes terminate on the same vector, `μ₃₁ = E[y₃₁]`. On the deterministic side
the reconstruction step is **solved**: a maximum-entropy density matching six exact marginal
moments of the final pre-activation reaches **bias² = 5.08e-09**, with zero fitted parameters —
comfortably inside the accuracy any large score improvement needs. Moments 3–6 are effectively
free. The blockage is the first two: the order-1 moment is, because `W₃₂` is invertible, *the
answer one layer up*, and the order-2 moment is a functional of `E[y₃₁y₃₁ᵀ]` that no bijection
reduces to it. On the stochastic side, whitening plus antithetic pairing is exactly a randomised
degree-3 cubature rule (confirmed by building a real degree-3 rule, which merely ties it), and a
model-free bound on the surviving Wiener chaos puts the required exactness degree at **D ≥ 15**
against a last affordable rung of **D = 5**. We built the degree-5 rule — 66,307 nodes, verified
exact to 1.8e-15 — and measured it **70.5× worse**, for a structural reason: degree-5 exactness
forces `w₁r⁴ = −n²(n−7)/(2(n+1)²) < 0`, collapsing effective sample size to 941 of 66,306 nodes.

We also give: an exact multilevel telescoping identity in which every level is exact, and the
two-line allocation argument that kills it anyway for any plausible weight profile; the
refutation of a published mechanism whose stated explanation is empty for a sample mean while its
real (cost-side) gain reproduces; a general trap — *any estimator that is a smooth function of the
same sample's own statistics re-derives the sample mean* — with three measured instantiations; and
the measurement protocol without which none of the above is decidable. Six of our own conclusions
are retracted in place, with the measurement that caught each.

---

## Contribution statement

**This is a negative-results, bounds and methodology contribution, not a new estimator.** What we
ship is a whitened, antithetic, float32 Monte-Carlo sample mean with structural cost reductions.
Every ingredient is standard; their combination is worth **2.17×** over plain Monte Carlo at
equal compute (§6.1, three independent routes agreeing inside 3%), and the cost reductions roughly
another 3.1× — but we do not claim any of it as novelty.

The results we believe are new:

1. **The availability–value antisymmetry** (§2), with the oracle-value ladder that measures it and
   the identity that explains why layer 31 is worth ~50× and is unreachable.
2. **Maximum-entropy reconstruction from six exact moments reaches bias² = 5.08e-09** (§3.1) —
   the deterministic branch's reconstruction step is not the blocker, and locating the blocker
   precisely on one vector is a sharper statement than "deterministic methods don't work".
3. **The model-free exactness-degree bound** (§4.2): `D ≥ [ln(1−S) − ln R(ρ)]/ln(1/ρ)` from
   `c_d ≥ 0`, with no inversion, no fit and no extrapolation. It places the constructible frontier
   at degree 5 — the last rung inside the budget — and a 10× improvement at degree 7, which costs
   18.9 budgets per evaluation. It replaces our own earlier extrapolated `D ≈ 1.3e4`, a ~900×
   correction.
4. **The negative-weight obstruction to degree-5 cubature** (§4.3), with the built rule, the
   closed-form weight, the ESS collapse, and the positive-weight counterfactual that attributes
   the entire 47× loss to weight negativity rather than to construction quality.
5. **The exact multilevel identity and its two-line refutation** (§4.6).
6. **The commutation identity** — a sample mean commutes with a linear map, so identifying a
   neuron as linear cannot change the estimate or its variance (measured bit-identical,
   0.000e+00) — and the general trap it is an instance of (§2.4, §5).
7. **Intraclass correlation across independent streams** as the discriminator between a correctable
   bias and heteroscedastic noise (§7.3, §8.4). We were about to build the corrector.

**Every number here is a real measurement**, on real competition MLPs of the public `full` split
against the dataset's own 1e9-sample ground truth unless explicitly flagged. Appendix A maps each
one to a script, a split, a sample count and a design. Appendix B reports the *reproducibility
spread* of our own central constants across ten independent probes, because a write-up that quotes
`C` to four digits and cannot reproduce the second is worse than one that quotes a band.

---

## 1. The problem, and the one lever the scoring rule removes

A bias-free ReLU MLP of width `n = 256`, depth `d = 32`, He init `W_l ~ N(0, 2/n)`, is handed to
the estimator in full. Inputs `y₀ = x ~ N(0, I_n)`; `z_l = W_lᵀ y_{l−1}`, `y_l = ReLU(z_l)`. Only
the final layer is scored, against a budget `B = 2.72e11` FLOPs per MLP.

Fix the scale first. On the 100 public `mini` MLPs, the mean over neurons of `t_i²` is **0.9093** —
exactly the MSE of the all-zeros baseline. So MSE 4e-07 is a relative RMS error of **6.6e-04**: a
six-significant-figure problem dressed as a regression problem.

### 1.1 Two regimes, and only one of them matters

```
Φ = mse_final × max(0.1, (F + λR)/B),        λ = 1e11 FLOP/s
```

For an estimator with fixed cost `f`, per-sample cost `c`, `k` samples and `MSE = C/k`:

```
Φ(k) = 0.1·C/k                    below the floor      — falls like 1/k
     = C·c/B + C·f/(kB)           above it             — flat in k
```

At our operating point `f/(ck) = 0.13%`, so "flat" is flat to a tenth of a percent. We measured it
rather than asserting it — whitened + antithetic MC, **the same 120 `full`-split MLPs at every
budget share**:

| budget share `u` | `k` | raw final MSE | `C = MSE·k` | **adjusted score** |
|---|---|---|---|---|
| 0.10 | 6,214 | 3.457e-06 | 2.148e-02 | **3.457e-07** |
| 0.20 | 12,478 | 1.770e-06 | 2.209e-02 | **3.540e-07** |
| 0.40 | 25,008 | 9.156e-07 | 2.290e-02 | **3.663e-07** |
| 0.80 | 50,066 | 4.755e-07 | 2.380e-02 | **3.804e-07** |

Eight times the budget buys 7.27× lower MSE and a score **10% worse**. OLS log–log slope
`p = 0.950`; an independent sweep with the pruned estimator gave `p = 1.008`. Both bracket `p = 1`.

**So `B` is not a knob. `C` and `c` are the only two, and they are worth exactly the same.** We call
this the **cost/variance confound**: at fixed budget, `raw MSE = C·c/(uB)`, so a 1.2× cheaper sample
and a 1.2× lower-variance sample are *indistinguishable in raw MSE*. Only a comparison at equal `k`
on the same ensemble separates them. This is the mechanism behind the refutation in §5 and the
reason §8.3 insists on separating the channels in every measurement.

The same algebra inverts the naive operating point. Below the floor the multiplier is clamped
regardless of *actual* spend, while `k` must still be sized against a worst case in which nothing
prunes — so a data-dependent saving earns **nothing**. Above the floor the multiplier tracks actual
spend and the saving is credited in full. Our estimator therefore sizes `k` at worst-case `u = 0.92`
and executes at `u = 0.68`.

Finally the asymmetry that sets every safety margin: overshooting the floor costs ≈1% of score per
1% of overshoot; overrunning the *budget* zeroes the predictions **and** forces the multiplier to
1.0, a factor of ≈1e5. Gentle on one side, a cliff on the other. Every cost model in this document
is therefore a strict upper bound, never an estimate.

### 1.2 What `C` is, and how well we know it

`C` is not fitted. For plain MC, `MSE = (1/k)·mean_i Var(y_{d−1,i})`, so the variance constant *is*
the single-forward-pass per-neuron variance of the scored layer. Our best determination comes from
an Ornstein–Uhlenbeck circle design — 120 MLPs × 65 correlation values, each sample drawing an
independent random 2-plane:

> **`C_plain = R(1) = 5.12429e-02`, SE 3.3e-05**, corroborated by an independent Stein-projection
> route at 5.12438e-02 — **agreement to 0.002%**.

Our shipped estimator's constant is `C ≈ 2.32e-02`, i.e. **2.21× below plain MC** — attributable
to whitening and antithetic pairing, which §6.1 measures at 2.17× by three independent routes.

**But see Appendix B.** Across ten independent probes, nominally the same whitened+antithetic
estimator produced `C` anywhere from **0.02227 to 0.02526** — an 11% spread, driven by `k`, by
MLP count, and by whether the reference was centred on exact ground truth. We report the band, not
a fourth digit, and we recommend everyone else do the same.

---

## 2. The organising result: availability and value sit at opposite ends of the network

Every variance annihilator needs a function with an **exactly known Gaussian expectation** that is
**cheap to evaluate**. That is the entire class. Enumerating it is short:

| class | exactly known mean? | cost | measured value |
|---|---|---|---|
| polynomials of degree ≤ D, as node exactness | yes | `C(256+m, m)` nodes | D = 5 is the last affordable rung; ceiling 3.02× (§4) |
| polynomials of degree ≤ D, as explicit control variate | yes | `C(256+D, D)` FLOPs/sample; D = 4 is 44× a forward pass | priced out |
| functions of a `d`-dimensional projection | yes (`d`-dim quadrature) | cheap | residual chaos has effective dimension `d ≳ 224` (§4.4) |
| functionals of layer 1 (ReLU mean, arc-cosine covariance) | **yes, closed form** | free | **1.083×** (§6.3) |
| hidden-layer means or covariances at depth | **no** | — | worth **50.8×** at layer 31 |

### 2.1 Closed-form Gaussian integration reaches exactly one ReLU layer

`z₁ = W₀ᵀx` is *exactly* jointly Gaussian, so layer 1's post-ReLU mean and full covariance are
closed-form: `σ_i = ‖W₀[:,i]‖`, `μ_i = σ_i/√(2π)`, and

```
C_ij = (σ_i σ_j / 2π)[ ρ(π/2 + arcsin ρ) + √(1−ρ²) ] − μ_i μ_j,     ρ_ij = S_ij/(σ_i σ_j)
```

— the arc-cosine kernel. It stops there. Pairs of ReLU ridges are integrable because a bivariate
orthant probability is elementary; **three-way ReLU products need a trivariate orthant probability,
which has no closed form.** From layer 2 onward the joint law leaves the Gaussian class and nothing
in it has a known mean.

*A trap worth recording, since it is invisible until it bites:* the shipped arc-cosine kernel is the
**zero-mean** case only, and `μ/σ` reaches 4.3 by layer 32. We derived and verified the exact
**non-central** bivariate ReLU second moment (reduces to arc-cosine at `m = 0` to 1.8e-15; matches
2e8-sample MC within its own standard error). Use the **Drezner–Wesolowsky** form of `Φ₂` — the
Owen-T form is 0/0 at `h = k = 0`, which is exactly the layer-1 case, and silently returns 0.5
instead of `¼ + asin(r)/2π`.

### 2.2 The value ladder, measured

Anchoring the sample ensemble at hidden layer `L` onto the exact means and continuing, then reading
off the variance reduction (oracle: the anchors come from the dataset's own 1e9-sample
`all_layer_means`, so no estimator can produce them):

| anchored layer | 1 | 4 | 8 | 16 | 24 | 30 | **31** |
|---|---|---|---|---|---|---|---|
| variance reduction | **1.13×** | 1.30× | 1.76× | 3.03× | 6.43× | 25.8× | **50.8×** |

Replicated independently twice (60 MLPs → 50.8×; 64 MLPs → 50.6×). Anchoring *all* of layers 1–31
gives 51.2×; anchoring **layer 31 alone** gives 48.1× (`t = +11.26`). Single-layer perturbation
gains are `g₃₁ = 0.950` with every other `g_l ≤ 0.013`.

> **Exactness is available only where it is worth ~1.1×, and would be worth ~50× only where it
> does not exist.** Availability and value sit at opposite ends of the network. That, and not a
> node count, is why a large improvement is hard here.

*(Definitions matter and we state ours: **50.8× is "hard-anchor layers 1–31 onto oracle means,
measure the MSE ratio at fixed `k`"**, 60 MLPs at k = 4,096. Across our own corpus the layer-31
figure ranges **46.5–52.1** purely on definitional choices — single-layer vs cumulative anchor,
hard vs weighted, and the value of `k`. A separate design entirely, a depth-truncated multilevel
surrogate fed an oracle `E[y_L]`, gives a steeper ladder on the same networks — 1.09× at layer 1
rising to 67.3× at layer 31. We quote the anchoring ladder throughout because it is the one whose
table we print; the two designs agree on the shape and disagree on the level by ~30%, which is
itself worth knowing. Any anchoring number quoted without its definition is meaningless — see
Appendix B.)*

### 2.3 Why layer 31 is worth 50× — it is an identity, not an estimator

For a bias-free net, `ReLU(z) = (z + |z|)/2` and the sample mean commutes with `W₃₂ᵀ` exactly, so

```
Ȳ₃₂ = ½ ( W₃₂ᵀ Ȳ₃₁ + mean|z₃₂| )
```

The layer-31 mean is not *information arriving* at layer 31; it is one of two terms in an exact
decomposition of the answer. Which is why it is worth 50× — and why it is not obtainable.

**Required accuracy.** Perturbing the layer-31 anchor by relative `eps` gives
`bias² = 0.945·eps²` over five decades. Reaching raw MSE ≤ 1e-08 needs `eps* = 1.03e-04` — **40×
tighter than the ensemble's own layer-31 sample-mean accuracy** (`0.261/√k`, = 4.1e-03 at
k = 4,096). And `W₃₂` is invertible, so the 256 order-1 marginal moments of `z₃₂` *are* `E[y₃₁]`
under a linear bijection: measured amplification `E[y₃₁] → E[z₃₂]` is 2.001 (= mean `‖w_i‖²`) and
attenuation `E[z₃₂] → E[ReLU]` is 0.461 (= mean `Φ(α)²`), net **0.922**. There is no attenuation to
hide behind and no shortcut: the oracle is not "not yet obtained", **it is the problem, one layer
up**.

Priced honestly, acquiring `μ₃₁` is the same commodity at the same price as acquiring `μ₃₂`. The
best *self-contained* anchored estimator we could build measures **1.14×** over its own baseline.

### 2.4 The general trap

> **Any construction whose estimator is a smooth function of the same sample's own statistics
> re-derives the sample mean.**

For `z` with the measured distribution, the projection of ReLU onto polynomials of degree ≤ 6
carries **99.2%** of `Var(ReLU(z))`. So *exact* moments 1–6 would be worth ~130× — but a plug-in
built from the **sample's own** moments has, by the delta method, the variance of the *projection*
(99.2%), not of the residual (0.78%). Three independent measured instantiations:

| instantiation | measured | conditions |
|---|---|---|
| always-on neurons "collapse" to a linear map (§5.1) | **0.000e+00** — bit-identical | 24 MLPs at k = 6,000; 8 MLPs at k = 24,000, float64 |
| single-hub Jacobian control variate `g = J(x₀)x` | **exactly 0.000**; mean over a whitened batch 8.6e-08 (float32 roundoff) against a signal of 1.03 | — |
| Gram–Charlier plug-in from the sample's own marginal moments | **0.9956× (t = −5.09)** at k = 4,096; **0.9872× (t = −6.60)** at k = 16,384 | paired CRN, 64 MLPs — plain MC significantly *better* |
| bootstrap self-anchoring, `J` sub-streams | 0.987 / 0.961 / 0.865 / 0.631 / **0.291×** for J = 2/4/8/16/32 | `t` = −1.22 to −7.15 |

Each of these looked, before measurement, like a different idea. They are one idea. We flag it
because it is the cheapest available filter on a candidate estimator: *if you can write it as a
smooth function of statistics your own sample produced, do not build it.*

---

## 3. The deterministic branch: the reconstruction is solved, the input is not

### 3.1 Maximum entropy from six exact moments reaches bias² = 5.08e-09

Given the *exact* first six marginal moments of each final pre-activation `z₃₂,i`, fit the
maximum-entropy density and integrate `ReLU` against it. Measured on 32 MLPs, 8,192 neuron-units
per number, moments taken as MC-converged:

| method | order | bias² | vs the 1e-08 requirement |
|---|---|---|---|
| Gram–Charlier / Hermite | 2 | 1.1098e-06 ± 1.7e-08 | 111× |
| | 3 | 1.6128e-07 ± 2.5e-09 | 16× |
| | **4** | **2.3037e-08 ± 3.6e-10** | 2.30× — the GC minimum |
| | 6 | 2.7340e-08 ± 4.3e-10 | 2.73× |
| | 7 | 3.2227e-08 ± 5.3e-10 | 3.22× |
| | 9 | 1.6006e-07 ± 3.3e-09 | diverging |
| **maximum entropy** | 4 | 5.259e-08 | 5.26× |
| | **6** | **5.077e-09 ± 1.3e-10** | **0.51× — under the requirement** |
| | 8 | 8.148e-10 ± 2.3e-11 | 0.08× |

**Zero fitted parameters.** Stable to 0.7% when the reference moment bake is changed by 4×.

Three corrections to our own earlier record fall out of this table. The Gram–Charlier minimum is at
order **4**, not order 7 — order 7 is 1.4× *worse*, and divergence starts at order **9**.
**Edgeworth and Gram–Charlier are bit-identical here**, because there is no `n^(−1/2)` to reorder
by, which closes "try a proper Edgeworth".

*Scope, stated plainly:* a Markov–Krein LP shows that no method using `M ≤ 12` moments can
*guarantee* better than 1.29e-05 on an arbitrary distribution with these moments. Our 5.08e-09 sits
2500× inside that guaranteed band. It is therefore **exploitation of the He-init depth-32 family,
not a general theorem** — which is fine for this benchmark and must not be quoted as more.
The M = 8 row converged for only 85% of neurons (the rest fall back to GC-4); treat 8.1e-10 as
"about 1e-09".

### 3.2 The entire accuracy burden is on the first moment

Perturbing the fitted law's inputs on held-out MLPs: `MSE ≈ 0.048 · (δμ/σ)²`, so `μ` must be known
to **4.6e-04 · σ**, while `c₃…c₆` need only ~1% relative accuracy (perturbing them by 1e-2 moves
MSE from 3.65e-09 to 3.65–3.68e-09). And by §2.3, the order-1 moment `E[z₃₂,i] = w_i·μ₃₁` is the
answer one layer up.

**The precise claim, since the loose one is tempting and wrong.** Moments 3–6 really are
effectively free: perturbing them by 1% moves the reconstruction's bias² from 3.65e-09 to at most
3.68e-09. The order-1 moment is the dominant blockage and is `μ₃₁` under the bijection of §2.3. But
the **order-2** moment is not free either — it must be supplied to ≈3e-3 relative accuracy, and
`σ_i² = w_iᵀ E[y₃₁y₃₁ᵀ] w_i` is a functional of the layer-31 *second-moment matrix*, which no
bijection reduces to `μ₃₁`. Our own anchoring measurements say the same thing from the other side:
an oracle layer-31 *mean* is worth ~50×, but only the mean *plus* full covariance reaches the
variance target. So the honest statement is **two objects, not one** — and the second is the larger
of the two to acquire.

The nearest constructible substitute fails by a wide margin: ~1%-accurate analytic Gaussian
propagation as the anchor gives bias² = 9.45e-05, i.e. **9,450× the target and 13× worse than plain
whitened+antithetic MC**. Per-layer, analytic propagation is more accurate than sampling only up to
a crossover at **layer 9** (analytic/MC accuracy ratio 0.002 at l = 1, 0.995 at l = 9, 2.27 at
l = 32) — and even that overstates usable depth, because structured analytic error transmits
downstream ~12× better than sampling noise at l = 2 (median over 24 MLPs at equal norms). Priced
through the meter, the analytic pass costs 6.56e9 FLOPs and the whole route is **net negative**.

### 3.3 Cumulant propagation — ARC's own method, over budget at this depth

The companion paper (arXiv:2605.05179; Wu, Lecomte, Winer, Robinson, Hilton, Christiano, ARC)
tracks joint cumulants to order `K`. The shipped `examples/03_covariance_propagation.py` **is** its
`K = 2` instance — Algorithm 2 diffs line-for-line against the shipped file. So there is no free win
from "fixing" the baseline, and `K = 3` promises an `n`-fold error reduction.

Cost at `n = 256, L = 32` against `B = 2.72e11`, from the paper's own Table 1 and Appendix J:

| variant | FLOPs | share of budget |
|---|---|---|
| K = 2 basic | 3.81e09 | 1.40% |
| **K = 3, cheapest honest reading** | **3.55e11 – 4.56e11** | **130% – 168%** |
| K = 3 factorized | 5.36e11 | 197% |
| K = 4 (any) | ≥ 4.4e13 | ≥ 16,000% |

The cost model validates: Table 1 predicts `K = 2` at 1.40% of budget, and we measure `examples/03`
at 1.32–1.33% through the official harness on 1,000 MLPs — **agreement to 6%**. Two consequences are
counter-intuitive at this depth. **Factorization hurts:** factorized cost scales as `30n³L²` against
basic `~(7/3)n⁴L`, crossing at `L ≈ 7n/90 ≈ 20`, so at `L = 32` the paper's headline "factor of `n`
speedup" does not apply — which the paper's own Appendix J confirms. And **flopscope will not grant
the paper's discount**: its counts assume ideal symmetric-tensor kernels (`β₃ = 7/18`), whereas a
naive dense `K = 3` in numpy is `6n⁴L ≈ 8.25e11`, 3.0× budget.

Accuracy: calibrating `c₂` from the shipped `K = 2` and extrapolating at *constant* `c_K`
(optimistic — the paper says `c_K` grows with `K`) gives `K = 3` at raw MSE ≈ 1.0e-05, **2.9× worse
than our estimator at the compute floor**, for 48–168% of the budget.

**The paper says so itself.** Appendix D: *"This depth scaling is worse than Monte Carlo sampling."*
Appendix C: the algorithms perform poorly at low width, especially when `L` is large. §6.2: they
underperform sampling at 8 hidden layers once `K` reaches 4. The deepest network anywhere in the
paper is `L = 12`; Phase 1 is `L = 32`. Appendix S.1.1 gives the reason — the theory rests on `W`
being large and unstructured, so the CLT makes pre-activations near-Gaussian with
`E[κ_r²] = O(n^{1−r})`. **That decay fails exactly when the activation distribution concentrates
onto a few directions, which is the rank collapse of §7.2 (`r90 = 8` at layer 31). The two negative
results are the same phenomenon.**

---

## 4. The stochastic branch, closed five ways

### 4.1 What we ship is already a randomised degree-3 cubature rule

Expand the scored quantity in the multivariate Hermite (Wiener chaos) basis of `N(0, I_n)`. Forcing
an empirical moment forces the sample mean of every Hermite polynomial of that degree to its true
value, so each device is exactly a statement about which chaos it annihilates:

- **whitening** (`mean_s x = 0`, `mean_s xxᵀ = I` exactly) removes degrees 1 and 2, **in all 256
  dimensions, for free**;
- **antithetic** (the ensemble is `{±x}`) removes every odd degree;
- together, `MSE = 2·Σ_{even d ≥ 4} V_d / k`.

The spectral predictions and the measurements agree to a few percent — 120 MLPs × 40 reps:

| | measured `C` (k = 4,096) | spectral prediction | error |
|---|---|---|---|
| plain | 5.11870e-02 | 5.12429e-02 | 0.1% |
| antithetic | 4.76335e-02 | 4.67940e-02 | 1.8% |
| whitened | 2.65052e-02 | 2.60857e-02 | 1.6% |
| whitened + antithetic | 2.38254e-02 | 2.28036e-02 | 4.5% |

And the framing was tested rather than asserted: we **built** a genuine degree-3-exact rule
(`2n = 512` nodes at `±√256·q_i` on a uniformly random orthonormal frame, 120 MLPs × 64 rotations)
and measured `C_rule = 2.42265e-02` against whitened+antithetic's 2.35e-02. **The real rule merely
ties the deployed pair.** There is nothing left on the degree-3 rung.

### 4.2 A model-free bound: the target needs degree ≥ 15

With `R(ρ)/R(1) = Σ_d c_d ρ^d` and `c_d ≥ 0`, a degree-`D`-exact rule leaving surviving mass `S`
must satisfy `R(ρ) ≥ (1−S)ρ^D` for *every* `ρ`, hence `D ≥ [ln(1−S) − ln R(ρ)]/ln(1/ρ)`. No
inversion, no fit, no extrapolation. Measured two independent ways agreeing to 2% (larger run:
120 MLPs × 65 values of ρ, SE ≤ 5.6e-4):

| ρ | 0.90 | 0.98 | 0.99 | 0.995 | 0.999 | 0.9999 |
|---|---|---|---|---|---|---|
| `1 − R(ρ)/R(1)` | .4089 | .1682 | .1065 | .0650 | .01742 | .002133 |
| mean chaos degree | 4.09 | 8.41 | 10.65 | 12.99 | 17.42 | **21.33** |

Because `C = S·C_plain`, any target score fixes a required surviving mass and hence a required
degree. Evaluating the bound over the ρ table above (it is maximised at ρ = 0.98 for the tighter
targets, ρ = 0.90 for the looser one), against the Möller node-count wall `k ≥ C(256+m, m)` for
`D = 2m+1` and a per-node cost of 1.8e6 FLOPs:

| target | required `S` | **`D ≥`** | rung | nodes | cost | affordable? |
|---|---|---|---|---|---|---|
| **the measured frontier, 4.62e-08 (3.74× from us)** | **12.1%** | **3.77** | **5** | **33,153** | **0.22× budget** | **✓** |
| 10× on our shipped score | 4.53% | 6.82 | 7 | 2,862,209 | **18.9× the entire budget for one evaluation** | ✗ |
| adjusted 1e-09 | 0.26% | 14.95 | 15 | 1.59e13 | — | ✗ |

Read the top row carefully, because it is the most useful thing in this section. **The
constructible frontier sits exactly at the last affordable rung.** Degree 5 costs 22% of the
budget and, at full efficiency, would buy 3.02× over plain MC — enough. Degree 7, one rung up,
costs nineteen budgets for a single evaluation. There is no gentle slope here: the cost of
exactness steps by a factor of 86 between the rung that suffices and the rung above it.

So the question this benchmark actually poses on the stochastic side is narrow and concrete:
**is there a positive-weight degree-5 cubature rule at n = 256 with O(n²) nodes?** §4.3 is what
happens when you build the known one.

The surviving-variance table shows why nothing lower will do: `S(D)/R(1)` = 0.743 / 0.515 / 0.422 /
0.331 / 0.267 / 0.211 at `D` = 1/2/3/5/7/9, reaching 0.094 only at `D = 32` and 0.0068 at
`D = 256`. The variance is not concentrated at low degree.

*(An earlier draft of this section attached `S ≤ 0.29% ⇒ D ≥ 14.7` to "a 10× improvement". That
`S` is the requirement for an adjusted score of 1e-09 — about 173× below what we ship, and a
target §9 explicitly abandons. The bound was right; the goal it was labelled with was wrong by a
factor of ~17, which made the wall look far more distant than it is. The corrected table is
strictly worse news for the 1e-09 story and strictly better news as a research direction.)*

> **This corrects our own earlier claim of `D ≈ 1.3e4`**, which came from fitting `C(D) ∝ D^−0.574`
> on `D ∈ {1,3}` and pushing it four orders out. `S(D)` is not a power law — the model-free log-log
> slope steepens monotonically from 0.394 to 0.793 across the range. The exponent was wrong by
> ~900×. The conclusion is unchanged, because the node count explodes either way; but the argument
> that supported it was not sound and we replace it.

### 4.3 We built the degree-5 rule. It is 70× worse, and the reason is structural

Mysovskikh / Lu–Darmofal at `n = 256`: **66,307 nodes**, verified after a random Householder/QR
rotation to a worst relative error of **1.835e-15** over every monomial of degree ≤ 5, and
correctly *failing* at degree 6 (`x₁⁶` gives 14.662 against a true 15). Head-to-head at the
identical node count `k = 66,306`, paired on 128 pairs (64 MLPs × 2 reps, common seeds):

| estimator | MSE | FOM = `C·c` |
|---|---|---|
| whitened + antithetic MC | 3.6955e-07 | 5.060e4 |
| **degree-5 Mysovskikh** | **2.6048e-05** | **3.113e6 — 70.5× worse, `t` = 15.92** |

**The failure is forced by exactness, not by our construction.** Degree-5 exactness pins the
weights:

```
w₂s⁴ = 2(n−1)²/(n+1)²  > 0,        w₁r⁴ = −n²(n−7)/(2(n+1)²)  < 0   for every n > 7
```

The measured effective sample size collapses to **941 of 66,306 nodes** (0.0142×). By contrast, any
*positive*-weight degree-5 rule obeys `ω_max ≤ 2.271e-5`, hence `ESS ≥ 44,032` — at most a 1.5×
penalty. **The entire ~47× loss is attributable to forced weight negativity**, and the mixing
coefficient `λ = −0.9614` is an algebraic consequence of exactness, not a tuning choice. Even
oracle-selecting `λ` (which uses the answer, so is not an estimator) bottoms out at 7.66× worse.

Two further facts close the class. First, **a random rotation does not make the rule unbiased**:
positive homogeneity makes the radial part deterministic, leaving a relative bias of −2.910e-3 and
an MSE floor of 7.4e-06, removable only by rescaling with `E‖x‖`. Second, the *best conceivable*
degree-5 rule — equal weights, ESS equal to the node count — gains **1.55×** (MLP bootstrap
1.455 ± 0.028; model-free bound ≤ 1.41). Against the 3.02× that §4.2 says degree 5 would need to
deliver to reach the frontier, 1.55× is short by about 2× — so even a *perfect* degree-5 rule is
not obviously sufficient, and the one that exists is 70× the wrong side of it.

*One positive by-product, kept because it is nearly free:* a stack of 8 rotated simplices under one
Haar rotation is a drop-in ensemble that skips whitening's 14.7% overhead for 1.1%, measuring
**1.25×** on the FOM against the live cost basis. It is a cost win, not a variance win — the
degree-3 rotated simplex is statistically indistinguishable from whitening+antithetic
(`t = −0.46`, `t = −0.52`).

### 4.4 The subspace hole is empty — and it is empty for an interesting reason

The degree-1 structure is *violently* anisotropic. The Stein matrix `G = E[f xᵀ]` (validated against
600 exact Jacobians to 0.2%, and against the Euler identity `J(x)x = f(x)` to 6.2e-06) holds
**79.1% of its energy in one direction and 97.2% in eight**, against an isotropic 3.1% — a **203×
violation** of the isotropy premise our own earlier analysis had assumed.

**And it is worth exactly zero**, because whitening already annihilates degrees 1 and 2 in all 256
dimensions for free. The residual the baseline actually leaves is even-degree-≥4 chaos, whose ANOVA
effective dimension measures **`d ≳ 224`**. A priced estimator on the top-8 subspace comes out at
**0.87× — worse** (`t = −1.71`), and a degree-4-exact rule on a 224-dimensional subspace would need
`C(227,4) = 1.08e8` basis functions, 7,706% of a forward pass.

This is the mechanism behind an earlier observation we could not explain: in a stratification sweep,
**a random direction stratifies as well as the network's own dominant direction** (0.26% vs 0.19%
of removable residual variance). The surviving variance is isotropic *even though the function is
not*, because the anisotropic part has already been removed for free.

*Honest scope:* `d ≳ 224` is the defensible claim; the κ measurement at d = 224 carries a 22%
error bar and runs on 48 MLPs rather than 64. A subspace built directly from the degree-4 chaos
Gram was not tested — that is the one real gap in this section.

### 4.5 Quasi-Monte Carlo is dead here, and the spectrum predicted it

Pure-numpy Joe–Kuo Sobol (validated: exact base-2 stratification, `ppf` round-trips to 6 decimals),
digitally shifted, antithetic, whitened, paired against the pseudorandom baseline on 120 `full`-split
MLPs: **0.94× at k = 4,096 (`t` = −1.03); 1.018× at k = 32,768 (`t` = +0.23).** Exactly what an
effective dimension of ≳224 predicts. Low effective *rank* of the activations (§7.2) does not imply
low effective *dimension* of the integrand, and we had conflated the two.

### 4.6 Multilevel is closed by an identity, not an inequality

`ReLU(z) = (z+|z|)/2` plus commutation of the sample mean with `Wᵀ` gives, for a bias-free net,

```
E[y₃₂] = Σ_{l=1..32} 2^{−(33−l)} (W₃₂ᵀ ··· W_{l+1}ᵀ) E|z_l|          (exact)
```

**Every level is exact.** There is no coarse surrogate — which is precisely what defeated every
previous multilevel attempt on this problem, and is why we thought this one would work.

It loses anyway, by arithmetic. Level `l` needs only `l` layers, so under cost `Σ n_l·l` the
optimal allocation `n_l ∝ √(w_l/l)` gives `MSE_min = (Σ√(w_l l))²/K`, while plain MC gets
`n_l = k` at *every* level for the price of one pass. The ratio is

```
MSE_multilevel / MSE_plain = ( Σ_l √(w_l · l) )² / 32
```

and it is hand-checkable. By Cauchy–Schwarz it can never exceed `(Σ_l l)/32 = 16.5×`; at a uniform
weight profile it is **14.9×**; at the measured profile (7.8% per layer at layers 3–4 falling to
1.9% per layer at layer 31) it is **12.3×**, reproducing the 12× we computed from the full weight
table by a completely different route. Multilevel wins only if essentially all the variance sits in
the first seven layers — uniform weight on layers 1–8 already gives 1.04× — and it does not.

> **Plain MC gets every level for the price of one pass because the same samples serve all levels;
> multilevel pays `l` per sample per level.** The exactness of the identity buys nothing against
> that.

*We flag one weakness deliberately:* the per-layer weight table `w_l` behind the 12.3× is not
shipped; only the two anchor points above are recorded. That is why the paragraph is written around
the profile-independence of the conclusion (12.3–16.5× across the entire admissible range) rather
than around a single number.

### 4.7 The rest of the stochastic ledger, measured and closed

| route | best measured | verdict |
|---|---|---|
| exactly-integrable surrogates (layer-1 affine + quadratic + 1,040 exact-mean controls) | 1.06× gross, **0.95× net** | closed |
| single-hub Jacobian control variate | **exactly 0.000** | reduces to the sample mean (§2.4) |
| multi-hub piecewise linearisation | 256× more hubs buy 1.19×; extrapolated requirement 10²⁷¹–10³⁴⁴ hubs | closed |
| Jacobian region reuse | 45.9× worse, `t` = 12.6 | closed |
| low-rank surrogate MLMC | `Var(f−g)/Var(f)` = 1.000 at r = 16/32/64, 0.938 at 128, 0.821 at 192, 0.355 at 224 ⇒ **≥1.29× worse at every r** | closed |
| GP surrogate, oracle marginals, best coupling | ceiling 87–116×, best-case FOM 349 — still above the required 272, **and unconstructible** | closed even as an oracle |
| orthoplex sampler at radius `√n` | O(1) radial bias, bias² ≈ 6e-7, `k`-independent; with the exact correction `E‖x‖/√n = 0.99902392` it is statistically identical to whitening (`t` = −0.46) | not shipped |

*A genuinely instructive trap from the last row:* the "layer-1 collapse" variant, which reuses one
shared rotation with per-stack sign randomisation, has a **fixed-frame variance floor** — `C` grows
∝ `k` (measured 0.185 → 0.740 from k = 4,096 → 16,384, `t` = −15). An estimator whose variance
constant *increases* with sample count looks fine at small `k` and is worthless at the operating
point. Independent rotations are mandatory.

---

## 5. A correct result with a wrong explanation

The best **publicly written-up** method on the Phase 1 board is AIcrowd topic 18106 — adjusted
1.551e-07, raw final-layer MSE 2.18e-07 at utilisation 0.71, method published in full.

*How we characterise it, and what is ours rather than theirs.* We do not have permission to
reproduce their write-up, so we separate the two carefully. **What their write-up states**, as we
read it: classify neurons as dead / always-on / kink at a threshold on `|α|`, with a pilot
re-classification; **treat the always-on neurons as linear in the last two layers**; evaluate the
dense block with Strassen multiplication and the cold-column prefix row by row; draw antithetic
scrambled Sobol points; `N = 84,992`. **What is our own generalisation**: that if treating a neuron
as linear helps, then runs of always-on layers should collapse into a precomputed matrix product
with *zero sampling variance*, and variance should scale with the kink count rather than the width.

That generalisation is the natural reading, it is what we set out to build, and it is what §5.1–5.3
refute. The narrow published version is refuted by the same identity — folding always-on neurons
linearly through the last two layers is *bit-identical* to running them through the ReLU — but we
want to be exact about whose claim is whose, because the strong version is ours.

We adopted the route. It works. **Linearity cannot be the source of the gain, in either version.**

### 5.1 The commutation identity

The estimator is a **sample mean**, and the sample mean commutes with any linear map:
`mean_s[(Wᵀy_s)_j] = (Wᵀ mean_s[y_s])_j` exactly, for every `s`-set. So if neuron `j` never crosses
zero over the ensemble, `mean_s[ReLU(z_j,s)]` and `mean_s[z_j,s]` are not merely close — they are
*the same sum of the same k floats*. Measured on real `full`-split MLPs, all reductions in float64:
max absolute difference over always-on columns is **0.000e+00** at 24 MLPs × k = 6,000 and again at
8 MLPs × k = 24,000. Bit-identical, not "identical to rounding".

### 5.2 The mechanism's target class is the class that dominates the error

Classifying scored-layer columns empirically and attributing squared error against the 1e9-sample
truth:

| | 24 MLPs, k = 6,000 | 8 MLPs, k = 24,000 |
|---|---|---|
| always-on: share of neurons / of squared error | 26.8% / **67.3%** | 24.6% / **53.5%** |
| dead: share of neurons / of squared error | 27.4% / **0.00%** | 26.3% / **0.00%** |
| per-neuron error, always-on ÷ kink | 3.51× | 2.30× |

*(Provenance: this re-run is recorded in the `wmc3.py` docstring, not in a committed driver. Our
chronological log records an earlier run of the same measurement at 25% / **72.7%** with a 5.2×
per-neuron ratio. The always-on set shrinks as `k` grows — "never crossed zero" is a statement
about the draw — so the exact shares are `k`-dependent. The ordering is not, and it is the ordering
that carries the argument.)*

The class the mechanism targets dominates the error and the mechanism moves it by exactly zero; the
class it does "solve" contributes 0.00%, because ReLU already emits exact zeros there.

The strong reading — a whole always-on *layer* collapsing — needs every neuron on, which under an
independence approximation is `P ≈ 0.268²⁵⁶ ≈ 1e-146`. The weak reading is well-defined and we
priced it: eliminating layer `l` on its always-on set `A` trades two matmuls for three, giving a
cost ratio `(3 − 2f)/2`, **below 1 only for `f > 1/2`**. Measured `f` rises monotonically with depth
and **peaks at 0.294** in the last layer. Plugging the measured layer-30 numbers: 61,907 vs 57,960
MAC-units, **6.8% worse**. The family is closed on cost grounds, independently of §5.1, by a
criterion anyone can check on their own network in one line.

Two steelmen, both on real MLPs: **freezing** always-on neurons at a propagated value mid-network is
**65–270× worse** (it destroys the sample-to-sample covariance downstream layers integrate against);
and a Gaussian plug-in fed *exact empirical* `μ` and `σ` is **2.2× worse than plain MC** — §2.4
again.

### 5.3 Why "their raw MSE is better" is not counter-evidence

This is the cost/variance confound of §1.1. Arithmetic on **their published** `N`, MSE and
utilisation, against our then-current estimator (dated: 2026-08-05):

| | topic 18106 (published) | ours, then (`wmc3`) | ratio |
|---|---|---|---|
| samples `N` | 84,992 | 57,766 | |
| raw final MSE | 2.18e-07 | 4.01e-07 | |
| `C = MSE·N` (variance) | 1.85e-02 | 2.32e-02 | **1.25×** |
| per-sample cost (FLOPs) | 2.27e06 | 3.09e06 | **1.36×** |

`1.25 × 1.36 = 1.70×` against the 1.74× public gap then observed. **This is inference from published
figures, not a measurement of their code**, and it establishes only that a cost explanation
suffices — not that theirs is the cost we computed.

We then replicated the one mechanism in their write-up large enough to explain the cost half —
Strassen matrix multiplication — and measured **1.1190× paired (`t` = +4.93, 250 MLPs)**, shipping it
as submission **#324358** (1.834e-07, rank #54). And we tested their one leg we had not replicated,
scrambled-Sobol sampling, and found it worth nothing here (§4.5). The public gap is now 1.12×, and
§8.2 shows a 50-MLP leaderboard cannot resolve 1.12× at all: our own full-split number for the same
estimator is 2.24e-07 against their public 1.551e-07.

### 5.4 The lesson

The published result is real, we reproduced its direction, and we shipped it. What is wrong is the
explanation, and the failure mode is general: **the mechanism was stated in the language of variance
reduction, the observable it moved was cost, and under a fixed-budget `1/k` estimator those two are
perfectly confounded in every raw-MSE number.** Believing the stated mechanism would have sent us to
build the layer collapse (needs `f > 1/2`, measures `f ≤ 0.294`) and to freeze always-on neurons
(65–270× worse). Testing *why* it worked converted a refuted 1.74× story into a verified 1.23×, and
redirected the remaining effort to FLOP accounting.

We were not immune to the same error — see §7.

---

## 6. What we ship, and what each part is worth

### 6.1 The component ledger

All accuracy numbers are against the dataset's own 1e9-sample `final_means`; all comparisons are
paired with common random numbers on real `full`-split MLPs **at equal FLOP cost**, each variant's
`k` derived from its own flopscope-priced cost.

| component | channel | isolated gain | measurement |
|---|---|---|---|
| whitened ensemble | variance `C` | **1.869×** | 120 MLPs × 3 reps, n = 360 paired CRN, `t` = **+9.36** |
| antithetic pairing, on top | variance `C` | **1.160×** | same run, `t` = **+3.23**; independently 1.122× (`t` = +3.03, n = 750) |
| float32 rather than float64 | cost `c` | **2.000×** | exact, from measured flopscope pricing |
| exact dead-column pruning | cost `c` | **1.15×** | 1,000 MLPs paired, `t` = **+4.33**; `log(MSE·k)` unchanged, `t` = +0.58 |
| lead-block masking + classification + Gram halving | cost `c` | **1.234×** | 250 MLPs paired in a real `BudgetContext`, `t` = **+14.58**; raw-MSE `t` = +1.15 (n.s.) |
| **Strassen (shipped, #324358)** | cost `c` | **1.119×** | 250 MLPs paired, `t` = **+4.93**; harness 200 MLPs, 0/200 failed |

Whitening + antithetic together reach **2.17× over plain MC**, by three routes that agree inside
3%: composing the two rows above gives 1.869 × 1.160 = **2.168×**; a direct paired run against
plain MC gives **2.207×** (`t` = +11.8, n = 480); and the independent 120-MLP × 40-rep constant
table of §4.1 gives 5.11870e-02 / 2.38254e-02 = **2.148×**. Three separately-run measurements of
the same quantity landing within 3% is the strongest single validation in this document, and we
quote the range rather than any one of them.

> **Correction to our own record.** Earlier drafts of this ledger led with "1.886×, `t` = +8.11,
> 400 MLPs" for whitening and "1.151×, `t` = +2.77" for antithetic, and multiplied them to 2.171×.
> We could not re-derive any of those four figures from a committed run when we audited this
> document, and 1.886 is numerically the variance constant `C = 1.886e-02` from an internal report
> whose other numbers we have since retracted. We replace them with the figures above, which do
> trace. The conclusion is unchanged to within 0.2%; the provenance is not, and that is the
> difference this document is about.

### 6.2 The cost floor, measured

`matmul` is **99.15%** of the pass. The entire non-matmul surface is 0.85% and the four obvious
items inside it are worth 0.25% combined. flopscope's metering is honest, and we checked rather
than assumed:

- `matmul((k,m)@(m,o))` costs `k·o·(2m−1)` **exactly**, verified to the FLOP on six shapes;
  the naive 32-layer chain is **4,194,304 FLOPs/sample**, exactly.
- `einsum('ki,kj->ij', X, X)` gets an exact **1.992×** symmetry discount and is **bit-identical**
  to `X.T @ X` (max difference 0.000e+00) — flopscope detects the repeated operand and bills only
  the unique entries of the symmetric output. Passing a *copy* is billed in full.
- **No dtype is cheaper than float32.** float16, int8, int16, int32 and uint8 all meter at
  2.0 FLOPs/MAC; float64 is 4.0. float16 additionally introduces a 1.27e-06 MSE bias floor.
- **The one dtype arbitrage that would have been worth a free 2× does not exist.** flopscope's rate
  table prices complex64 at the same 1.0 rate as float32, which invites packing two real matmuls
  into one complex matmul. Measured on `(256,256)@(256,4096)`: float32 535,822,336; complex64
  **2,145,386,496** — exactly 4× float32, the true real-multiply count. The meter is honest.
  *(Note the two unit conventions in play: flopscope's rate table is per-element; our 2.0/4.0
  figures are FLOPs per MAC.)*

**Strassen is capped, and the cap is not an implementation defect.** flopscope charges Strassen's
additions at 1.0/element, and Winograd's 15 additions is proven minimal for a rank-7 ⟨2,2,2⟩
algorithm, so no deeper recursion escapes: the metered chain bottoms out at **1.686×** against
naive. And the metered optimum is not the *score* optimum, because depth `d` costs `7^d`
Python-level leaf calls billed at λ = 1e11 FLOP/s:

| Strassen depth | 0 | 1 | **2** | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| metered FLOPs/sample | 130,816 | 114,945 | 101,507 | 90,534 | 82,307 | 77,515 |
| **residual**-equivalent FLOPs/sample | 142 | 717 | 2,662 | 22,124 | 74,196 | **432,427** |
| **total** | 130,958 | 115,662 | **104,169** | 112,658 | 156,503 | 509,941 |

Depth 5 saves 24,000 metered FLOPs/sample and spends 432,000 residual — an **18:1 loss** (133:1 at
chunk 8,192). Depths 1 and 2 are statistically tied and the ordering is *not stable against machine
load*; the shipped code uses 2.

> **Under a score of the form `flops + λ·residual_seconds`, an asymptotically faster algorithm is
> worth its asymptotics only if its constant-factor bookkeeping stays inside counted primitives.**

### 6.3 Four more exact identities, and one variance gain

**`sweep.py`, submission #325573, adjusted 1.730e-07 — our best public score.** It adds four exact
algebraic identities to `wmc4`. Nothing statistical changes; each is an identity or an exact
re-pricing of the same arithmetic:

| identity | saving | why it is exact |
|---|---|---|
| antithetic mirror of layer 0 | 2.1% | `ReLU(−z) = ReLU(z) − z` pointwise in exact IEEE arithmetic, so half of layer 0's matmul was recomputing the negative of the other half |
| symmetric Gram via aliased-operand einsum | 1.3% | bitwise identical to `xᵀx`; supersedes Strassen on the Gram (0.502× beats 0.779×) |
| Strassen in the lead pass | 0.3% | the one place still calling plain `@` |
| Cholesky whitener instead of `eigh` | 0.09% | **distributionally exact**: `x L^{−T} = x R^{−1} = Q` from the thin QR, and both the Cholesky and symmetric roots are Haar on the Stiefel manifold, hence identical in law. Measured raw-MSE ratio 1.0006×, `t` = +0.06 — a clean null, as predicted |

Paired on **1,000** real MLPs: FLOPs/MLP 1.7906e11 → 1.7207e11 (3.90% cheaper, exact and
deterministic on every machine and every MLP of this shape), adjusted score **1.0834×**, `t` = +6.13,
0/1000 failed. Binding adversarial utilisation improved 0.9170 → 0.9040. It predicted a public score
of 1.69e-07 and delivered **1.730e-07** — a 2.4% miss, which is as well as a deterministic cost model
can be expected to transfer through a 50-MLP draw.

**`anchor-cheap.py`, submission #325574**, folds the exact layer-1 full-covariance anchor of §2.1
into `W₁`. Paired, **n = 1000**, at the real operating point `k = 64,512`:

| variant | mean-MSE ratio | `t` | 95% CI |
|---|---|---|---|
| + mean anchor | 1.0355 | +3.81 | [1.017, 1.054] |
| + diagonal anchor | 1.0422 | +4.66 | [1.025, 1.061] |
| **+ full covariance** | **1.0832** | **+5.45** | [1.053, 1.113] |

This is a genuine variance-channel gain, and it is the subject of §7.5 — **the effect flips sign
below `k ≈ 6,000`** (0.825× at k = 5,900), which is why we measured it as neutral three times before
we measured it correctly.

**Composing the two.** The cost leg and the variance leg are independent mechanisms on the same
base, so we merged them. One thing composes better than either alone: the anchor's dominant cost is
a symmetric Gram of the layer-1 block, and it takes the same symmetry-aware einsum discount the
whitener's Gram uses — measured `einsum("ik,jk->ij", yt, yt)` at **0.5020×** of `yt @ yt.T`,
**bit-identical** (max difference 0.0), against the 0.79× Strassen the standalone anchor paid. The
anchor therefore costs 2.3% of the sample budget inside the merged build rather than 3.4%.

Measured paired at the real operating point, against `sweep.py`:

| n MLPs | ratio | `t`(mean) | `t`(log) | win rate |
|---|---|---|---|---|
| 240 | 1.0455 | +1.59 | +1.36 | 48.3% |
| **1000** | **1.0619** | **+3.67** | **+3.28** | **52.0%** |

We report both rows because the first one is the interesting one. At n = 240 the point estimate was
already there and the composition **failed our own significance bar** (`|t| > 2.5`), with a win rate
*below* 50% — the mean gain sitting in the tail while the median MLP was unchanged. We did not ship
it on that evidence. At n = 1000 it clears, the point estimate having barely moved and `t` having
scaled as √n. The anchor delivers 1.062× on top of `sweep` against 1.083× on top of `wmc4`, so the
two legs are close to but not exactly multiplicative. Harness check on 100 MLPs: 2.09e-07 → 2.03e-07,
all-layers 5.72e-07 → 4.69e-07, 0/100 failed, worst-case adversarial utilisation 0.9094.

**And it scored 1.920e-07 publicly, the worst of our four best builds.** We are reporting a
mechanism with `t` = +5.45 at n = 1000 that the 50-MLP public board ranks last. We believe the
paired offline measurement and not the board, for the reason in §8.2 — but the honest form of that
belief is to say which measurement we are overruling and why, rather than to quote only the one
that agrees with us. The two mechanisms in this section are orthogonal (cost × variance) and their
composition is the obvious next build.

### 6.4 Sizing `k`, and why the cliff is unreachable

`k` is sized from a **worst-case cost model in which nothing prunes** — alive fraction 1.0 at every
layer, mask = identity, plus explicit slack. The mask can only make the pass cheaper, so the sized
spend is a strict upper bound on the executed spend **for any input**, with no property of the
public MLPs baked in.

Stress-tested through real flopscope accounting (effective compute / budget): all-positive weights
(nothing ever dead — the binding case) **0.9040** for `sweep.py` as shipped, 0.9170 for `wmc4`; rank-1 0.9144; depth 5/8/10/64 =
0.917/0.917/0.883/0.522; He init 0.6698; weights ×10 (float32 overflow) 0.6692; all-negative 0.2150;
all-zero 0.0474; odd widths 250/255/300 = 0.7208/0.6392/0.6776. **Every case returns a finite
`(depth, width)` array and none reaches 1.0.** Odd widths matter: the recursion checks divisibility
at every level and falls back to the plain product, so a width it cannot split degrades rather than
fails.

**The trap we hit and recorded:** a fallback ladder whose second rung re-runs Monte Carlo is
harmless at 10% of budget and *fatal* at 80% — a float32-overflow MLP makes the first attempt raise
and the pair reaches **1.88× of budget**, zeroing the prediction and forcing multiplier 1.0. The
ladder must contain exactly one expensive rung.

### 6.5 A route we did not take

The scoring rule bills residual wall time at λ = 1e11 FLOP/s, so arithmetic executed outside
flopscope's instrumentation is materially cheaper than instrumented arithmetic. We measured the
mechanism directly — §6.2's depth table is exactly this channel, priced — and **we chose not to
exploit it**: Phase 1 rank comes from a private re-evaluation on a frozen stack; the starter kit
tells participants not to; and it is the opposite of what an algorithmic-contribution prize
rewards. Every FLOP saving reported in this document is a counted-FLOP saving, and §5.3's Strassen
replication is included precisely *because* flopscope prices it honestly, which we verified by
measurement rather than assuming.

**We make no claim about whether anyone else used this channel.** We looked, our first reading was
wrong, and we retract it in §7.6.

---

## 7. Six conclusions of ours that were wrong, and what caught each

The most transferable thing we have. Each was held, acted on, and withdrawn.

**7.1 "The sampling lineage is closed."** We measured `Φ` insensitive to `k` and concluded sampling
was finished. `Φ = C·c/B` falls if *either* factor falls, and the sweep said nothing about `c` —
where every gain for the next two days came from. *Caught by:* asking what the measurement actually
established. "Sampling is closed" and "`Φ` is insensitive to `k`" are different statements.

**7.2 "Rank truncation will pay."** Effective rank collapses with depth (`r90`: 171 → 8 from layer 0
to 31), so projecting late layers to rank `r` should turn a `k·n²` pass into `~2k·n·r + r·n²`.
Truncating layers ≥ 16 and comparing **on the same ensemble** to isolate pure truncation bias gives
8.87e-05 at `r = 48` — **211× our noise floor**. The rank that suffices is 128–192, where
`2r/n = 1.5` and truncation costs *more* than the dense pass. *Caught by:* comparing a squared bias
against a variance. **`r99 = 47` says 99% of the variance lives in 47 directions; it says nothing
about whether the discarded 1% is harmless for a quantity needed to 0.07% relative accuracy.**
Energy rank and accuracy rank differ here by ~4×.

**7.3 "The error's dominant direction is a correctable bias."** 30.67% of error energy lies along
the truth direction — 78.5× isotropic — and an oracle removing it gains 1.69×. We ran `R = 5`
independent streams on the same MLP and decomposed the scale coefficient's variance: **ICC =
−0.0107** (and −0.0132 for the offset), the signature of an exactly zero systematic component;
per-MLP mean `|α| = 5.2e-04` against a within-MLP sd of 1.4e-03. *Caught by:* five extra forward
passes. **Concentration is a statement about `Σ`; correctability is a statement about `E[e]`.** For
an unbiased sample mean the bias is zero *by construction*, yet the energy fraction along any `u`
is `u'Σu/tr(Σ)`, which equals `1/n` only if `Σ ∝ I` — and ReLU outputs are heteroscedastic, giving
`Σt⁴/(Σt²)² = 5.8×` isotropic from a calculation with no bias in it. Evaluating the full zero-bias
prediction against measurement: 59.56% predicted vs 63.24% measured on the truth direction, 19.45%
vs 20.76% on the constant — **agreement to 6%, with zero bias assumed anywhere.**

**7.4 "Extending the covariance baseline's series will help."** The shipped gain heuristic is
*exactly* the order-1 truncation of an exact Mehler–Hermite series with elementary coefficients
(`c₁ = σΦ(α)`, `c_k = σ He_{k−2}(−α)φ(α)`). Extending it is `O(n²)` per order. Full 32-layer
propagation at `K` = 1/2/4/8/12 gives 5.95e-05, 6.54e-05, 6.55e-05, 6.54e-05, 6.54e-05 — making the
per-layer algebra exact makes the answer **10% worse**. *Caught by:* reading the companion paper
afterwards, where our `K ≥ 2` variant is its Appendix E *ablation*, shown flat in width. **The axis
we varied is not the axis that governs the error.**

**7.5 "The exactly-known-mean control-variate family is capped at 1.001×."** Measured with a
320-stream oracle regression at `k ≈ 3,000`. **The effect is `k`-dependent and flips sign at
`k ≈ 6,000`**: the same anchor measures 0.825× at k = 5,900 and **1.083× at k = 64,512** (paired,
n = 1000, `t` = +5.45). The mechanism is transparent in hindsight — the transform fits
`p = n(n+1)/2 = 32,896` parameters from `k` samples and the penalty goes as `(1 + p/k)`. *Caught
by:* measuring at the shipped operating point instead of the run-off's convenient 9.5%-of-budget
setting. This one fact retroactively explains **three** separate null readings we had accumulated,
and the bound it replaced was wrong by 8× its own claimed headroom.

**7.6 "Some top entries bill arithmetic through residual wall time."** Inferred from forum reports
plus our own arithmetic about what the channel is worth, and propagated into a draft of this
document. We then checked the actual evaluation data and **it did not support the inference** — the
entries in question run genuinely metered compute. We withdrew it, and we make no claim about any
participant's methods. *Caught by:* going and measuring the thing we had inferred rather than the
thing convenient to believe. (The same pass turned up a 10× arithmetic error of our own: a quantity
reported as "63× above the requirement" was 6.3×.)

**A seventh, of a different kind.** Three times we wrote a completeness claim — "the *only*
remaining lever is a sub-cubic matmul", "*every* subsequent gain came from cost", "`r ≈ 8` is
*untested*" — and each was falsified within days by our own later measurement. And in auditing this
document for publication we found two headline figures we could not re-derive from any committed
run (§6.1) and five §4 drivers we had marked reproducible and could not produce (Appendix A). The
words `only`, `every`, `none` and `reproducible` are the ones to grep for before publishing.

---

## 8. How to measure anything on this benchmark

Every negative result above was produced by an apparatus, not by an idea. On this benchmark the
apparatus is the hard part.

### 8.1 The public dataset ships its own ground truth

Each row carries `final_means` and `all_layer_means`, baked at 1e9 samples per MLP. Accuracy is
therefore measurable offline, in plain numpy, against the *exact* MLPs and *exact* targets the
harness uses: ~60 s for 100 MLPs against the harness's ~25 min for 1,000. This is not a proxy — the
whitened → whitened+antithetic step measured **1.123×** through the official harness against
**1.122×** predicted offline (0.1%), and the harness bills 1.765e11 FLOPs/MLP against our offline
accounting's 1.7646e11 (**0.02%**).

Our first round of experiments used self-generated MLPs and self-computed 2e6-sample MC targets —
targets noisier than the estimators under test. Two of this document's retractions trace directly to
that. **Do not do this; the ground truth is already in the file.**

*One caveat with teeth:* the reference is itself a 1e9-sample Monte-Carlo bake. Layer 1's "exact"
analytic residual of 2.64e-05 **is** the bake's own noise (`sd(y₁)/√1e9 = 2.6e-05`). Several oracle
floors in this document are bounded by the reference, not by truth, and we say so where it matters.

### 8.2 Measure your split's noise floor before you trust any ranking

Same estimator, same 100 mini-split MLPs, only the RNG stream changing: MSE = 5.04, 4.07, 4.23,
3.98, 4.70, 3.92 (×1e-06) — **relative sd 10.4% for a single arm.** On 400 `full`-split MLPs,
per-MLP MSE has relative sd 0.957 and worst/mean = 10.0, and the correlation between two independent
streams on the same MLP is **0.181** — only ~18% of per-MLP error variance is a property of the
network.

Translating to A/B resolution, anchored on the measured paired SE of 5.45% at n = 400:

| n MLPs | 50 (leaderboard) | 100 (mini) | 250 | 400 | 1000 (`full`) |
|---|---|---|---|---|---|
| **2σ band on the ratio** | **~30%** | **22–29%** | 14% | 11% | **6.9%** |

**The public leaderboard's 50 MLPs cannot resolve two entries differing by less than ~30%.** We were
burned by exactly this: a competing design's 19% mini-split "advantage" evaporated to 3% on 1,000
MLPs. It is also why §5.3 declines to read anything into a 1.12× public gap, and why we do not cite
our own leaderboard movements as evidence for anything.

### 8.3 Pairing: what it buys, and what it does not

Our internal notes claimed "common random numbers is what separates signal from noise." Half right,
and the wrong half matters. Same contrast, three designs, 400 `full`-split MLPs:

| design | SE of the mean MSE difference | `t` |
|---|---|---|
| D1 disjoint MLP sets, independent streams | 2.818e-07 | +1.90 |
| D2 same MLPs, independent streams | 2.410e-07 | +2.22 |
| D3 same MLPs, common random numbers | 2.396e-07 | +2.77 |

**Running both arms on the same MLPs is worth 1.17× in SE. Sharing the random stream on top is
worth 1.01× — nothing**, because the antithetic arm consumes the pool differently from the
reference, so the two ensembles are not the same and their errors barely co-move.

Where pairing *is* transformative is when the difference contains a deterministic component: in the
masking verification, the **adjusted-score** paired `t` was **+14.58** while the **raw-MSE** paired
`t` on the same run was **+1.15 (n.s.)**. Utilisation is a near-deterministic function of the
weights; MSE is draw noise.

**The operational rule follows from the cost/variance confound: separate the channels.** Price cost
deterministically in FLOPs — one evaluation per MLP shape suffices — then test only for *accuracy
neutrality*. Variance changes need hundreds to thousands of MLPs whether or not you pair.

### 8.4 Checklist

1. Score against the dataset's shipped `final_means`, never a self-computed MC target.
2. Measure your split's noise floor first; refuse to rank anything inside its 2σ band.
3. Use the same MLPs for both arms (1.17× SE). Do not assume CRN adds more — measure it.
4. Price cost deterministically; test accuracy separately and only for neutrality.
5. **Measure at the operating point you will ship at.** A run-off at 9.5% of budget uses `k` 11×
   smaller than production, and any mechanism whose benefit scales with `k` is invisible to it
   (§7.5). Any mechanism fitting `p` parameters from `k` samples has a `(1 + p/k)` penalty that
   moves the *sign* of the result.
6. Before building a corrector for a concentrated error mode, compute its ICC across streams
   (§7.3). `R = 5` forward passes is far cheaper than the corrector.
7. Before adding a cost-for-bias knob, evaluate `gain = (c/c')/(1 + k·b²/C)` rather than the cost
   ratio alone. Ours turns over immediately: masking a third of every layer buys 1.28× of cost for
   37× of error.
8. Adopt a significance bar and hold it. Ours is **|t| > 2.5** on a per-MLP paired statistic. One
   of our three Aug-7 builds measured 1.0547× at `t` = +2.04 and was shipped as a hedge, not as a
   claim.
9. Confirm the winner once on the real harness, for budget and failure safety — overruns and
   per-MLP failures are invisible offline and cost ~1e5 when they happen.
10. Watch memory. At `k ≈ 57,000` one sample array is ~58 MB and the harness holds several; an
    unbounded `--split full` run froze a development machine.

---

## 9. The open problem, stated precisely

We finish **3.7×** above the measured constructible frontier — twelve independent teams are piled
at adjusted 4.6e-08 to 1.1e-07, and rank 4 reached 4.62e-08 in only 115 submissions. Nothing in
this document explains that 3.7×, and we say so rather than dressing our ceiling up as the
problem's.

What we can offer is a sharpened statement of where the remaining factor must live. Since
`Φ = C·c/B`, and since §4 bounds `C` on the entire exactly-known-mean, cheap-to-evaluate class
(1.06× gross / 0.95× net) while §6.2 bounds metered `c` from below, the frontier is not reachable by
rearranging either axis. Two concrete open problems fall out, and we state them precisely because
that is more useful than a verdict.

> **(A) The stochastic one.** Is there a **positive-weight** degree-5 cubature rule on `N(0, I₂₅₆)`
> with `O(n²)` nodes? §4.2 puts the frontier exactly at degree 5, the last rung that fits in the
> budget (33,153 nodes, 22% of it). §4.3 shows the known Mysovskikh construction is destroyed by
> weights that exactness *forces* negative — ESS 941 of 66,306 — while any positive-weight rule
> would pay at most 1.5×. This is a question about cubature, not about neural networks, and it has
> a definite answer we do not know.
>
> **(B) The deterministic one.** Obtain `μ₃₁ = E[y₃₁]` to relative accuracy `1.03e-04` — 40×
> tighter than the sample mean of the same ensemble delivers — *and* the layer-31 second-moment
> matrix to ≈3e-3, both at a cost materially below estimating `μ₃₂` directly. Given those,
> §3.1's six-moment maximum-entropy reconstruction closes the rest with zero fitted parameters.

We do not know whether (B) is possible. We do know it is not circular *by definition*: the required
accuracy and the required cost are both quantified, and a method that beats either is a real
advance. We also know four things it cannot be: analytically propagated (13× worse than the MC
it would replace), a smooth function of the same sample's statistics (§2.4), a low-dimensional
projection (§4.4), or a moment plug-in built from sampled moments (§2.4).

---

## Appendix A: Reproduction manifest

**Code.** The complete solution, every estimator in the lineage, and the ten probe reports cited
throughout are released at `<REPOSITORY URL>` under `<OSI LICENSE>`. Each estimator's docstring
carries the measurements for the mechanism it adds, so a claim in this document and the code that
produced it sit next to each other. All paths below are relative to that repository root.

**Environment.** Python 3.10.20, numpy 2.2.6, flopscope 0.10.0, whestbench 0.14.0 — the announced
grader stack, CPU only. Estimator imports are limited to `math`, `flopscope`, `flopscope.numpy`,
`whestbench` and stdlib.

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/whest-starterkit"
```

**Official harness** (the only thing that measures budget safety and failures):

```bash
uv run --project work/whest-starterkit whest run --estimator <file.py> --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split full --n-mlps <N> --runner subprocess
```

### A.1 Submission log

| # | date | estimator | adjusted score | rank | failed | util |
|---|---|---|---|---|---|---|
| #323440 | 2026-08-04 | `work/mine/wmc.py` — whitened + antithetic | 3.405e-07 | #121 | 0/50 | 10.01% |
| #323892 | 2026-08-05 | `work/mine/wmc2.py` — + exact dead-column pruning | 2.700e-07 | #80 | 0/50 | ~80% |
| #324076 | 2026-08-05 | `work/mine/wmc3.py` — + a-priori lead-block masking | 2.240e-07 | #68 | 0/50 | 65.99% |
| #324358 | 2026-08-05 | `work/mine/wmc4.py` — + Strassen | 1.834e-07 | #54 | 0/50 | 70.74% |
| **#325573** | 2026-08-07 | **`work/final/sweep.py` — + four exact identities (§6.3)** | **1.730e-07** | | **0/50** | **67.8%** |
| #325572 | 2026-08-07 | `work/mine/wmc5.py` — + layer-1 mean+covariance anchor | 1.860e-07 | | 0/50 | 70.5% |
| #325574 | 2026-08-07 | `work/final/anchor-cheap.py` — layer-1 full covariance (§6.3) | 1.920e-07 | | 0/50 | 70.8% |

**The last four rows are an instance of §8.2 and we report them as one.** Their whole spread is
**11%**, against a ~30% 2σ resolution limit on a 50-MLP board. In particular both anchor builds came
in *above* `wmc4`'s raw MSE on the public 50 despite a paired offline reading of **1.083× better at
n = 1000, `t` = +5.45**. We take the n = 1000 paired measurement as the better predictor of a fresh
draw and the public ordering among these four as uninformative — which is the same rule we applied
when the public board moved in our favour.

### A.2 Number → source

| § | claim | source | split / n / k | re-runnable? |
|---|---|---|---|---|
| 1 | mean `t²` = 0.9093 | `work/offline_bench.py` | mini / 100 | yes |
| 1.1 | budget sweep, `p` = 0.950 | scratch driver over `offline_bench.score` | full / 120 | **no — scratch**; method fully specified in §1.1 |
| 1.2 | `C_plain` = 5.12429e-02, SE 3.3e-05 | OU circle design, `hs_p2_constants.py` (see note) | full / 120 × 65 ρ | **no — scratch** |
| 2.2 | anchoring ladder, VR 50.8× / 50.6× | `work/adversarial/oracle-anchoring.md` | full / 60 and 64 / k = 4,096, reps 2, CRN | **yes** |
| 2.3 | `bias² = 0.945·eps²`, `eps*` = 1.03e-04 | same | full / 64 | **yes** |
| 2.4 | commutation 0.000e+00 | recorded in `work/mine/wmc3.py` docstring | full / 24 at k=6,000; 8 at k=24,000 | **no — scratch driver** |
| 2.4 | GC plug-in 0.9956× (`t` = −5.09) | `work/adversarial/deterministic-closure.md` | full / 64 / k = 4,096 and 16,384, paired CRN | **yes** |
| 3.1 | max-entropy ladder, 5.077e-09 | same | full / 32 / 8,192 units per number | **yes** |
| 3.2 | `MSE ≈ 0.048(δμ/σ)²` | same, held-out | full / 16 | **yes** |
| 3.2 | analytic-propagation crossover at layer 9 | `work/adversarial/analytic-per-layer.md` | full / 40 / k = 4,096 | **yes** |
| 3.3 | cumulant cost table, `c₂` | arithmetic on arXiv:2605.05179 Table 1 / App. J | — | arithmetic given in §3.3 |
| 4.1 | spectral `C` predictions | `hs_p2_constants.py`, `hs_p7_verify.py` (see note) | full / 120 × 40 reps | **no — scratch** |
| 4.1 | built degree-3 rule, `C` = 2.42265e-02 | `hs_p6_rule3.py` (see note) | full / 120 × 64 rotations | **no — scratch** |
| 4.2 | `R(ρ)` table, `D ≥ 3.77 / 6.82 / 14.95` | `hs_p5_final.py` (see note) | full / 120 × 65 ρ | **no — scratch** |
| 4.3 | degree-5 build + 70.5× | `d5_tscan.py` (see note), `work/adversarial/degree5-cubature.md` | full / 64 × 2 reps, 128 pairs | **no — scratch** |
| 4.4 | Stein matrix, 203× anisotropy; `d ≳ 224` | `work/adversarial/active-subspace.md` | full / 64 (48 for d ≥ 160) | **yes** |
| 4.5 | Sobol 0.94× / 1.018× | post-crash score push, `work/RESEARCH.md` §7h.12 | full / 120 / k = 4,096 and 32,768, paired | **yes** |
| 4.6 | multilevel identity + allocation | derivation in §4.6; ratio recomputed here from the two published weight anchors | — | **partly — the full `w_l` table is not shipped; see §4.6** |
| 5.2 | error attribution, `f > 1/2` | `work/gap/cost.py` docstring | full | **no — scratch driver** |
| 5.3 | 18106 decomposition | arithmetic on their **published** `N`, MSE, utilisation, 2026-08-05 | — | arithmetic given in §5.3 |
| 6.1 | whitening 1.869×, antithetic 1.160× | `work/RESEARCH.md` §7h.0 | full / 120 × 3 reps, n = 360 paired | **no — scratch driver**; the harness (`runoff.py`) is committed |
| 6.2 | metering identities, Strassen depth table | `work/adversarial/cost-floor.md` | one layer, chunk 65,536 and 8,192 | **yes** |
| 6.3 | `sweep.py` 1.0834×, `t` = +6.13 | `work/final/sweep.py` docstring | full / 1000, paired | **yes** |
| 6.3 | anchor 1.0832×, `t` = +5.45 | `work/final/anchor-cheap.py` docstring | full / 1000 / k = 64,512, paired CRN | **yes** |
| 6.4 | budget stress cases | `work/mine/wmc4.py` docstring | synthetic pathological weights | **no — scratch** |
| 7.2 | rank table, truncation bias | scratch driver | full, same-ensemble | **no — scratch** |
| 7.3 | ICC, energy fractions | `work/scale_mode.py` | full / 150 × R=5 | **yes** |
| 7.4 | Mehler orders 1–12 | `work/experiments/mehler.py` | self-generated MLP, 400k target | yes — **but see the caveat below** |
| 8.2 | split noise floors | `work/offline_bench.py`, `seed_offset` 0–5 | mini / 100 | **yes** |
| 8.3 | D1/D2/D3 designs | scratch driver | full / 400 | **no — scratch** |

**On the "no — scratch" rows.** These were run interactively against the two committed harnesses
(`offline_bench.py`, `runoff.py`) and the numbers are recorded in the docstring of the file that
implements the mechanism, but the driver was not committed. We say that rather than imply a
reproducibility we cannot deliver. Every such row states its split, `n`, `k` and comparison design.

**On the five rows marked "(see note)".** These are the §4 chaos-spectrum and cubature
measurements — among the strongest results in this document — and their drivers
(`hs_p2_constants.py`, `hs_p5_final.py`, `hs_p6_rule3.py`, `hs_p7_verify.py`, `d5_tscan.py`) were
written to a working directory that has since been cleared. **We could not recover them, and an
earlier draft of this appendix wrongly marked all five re-runnable.** What survives is the full
measurement record in `work/adversarial/chaos-spectrum.md` and
`work/adversarial/degree5-cubature.md`, which state the design, the node construction, the
verification tolerances, the MLP and rotation counts and the standard errors — enough to rebuild
each experiment, but not the same thing as shipping the code. We flag it prominently because §4 is
the part of this document a reader would most want to check, and it is currently the least
checkable. Losing a driver is an ordinary mistake; marking it reproducible afterwards is not, and
that is the error we are correcting here.

**On §7.4 specifically.** The Mehler experiment predates our offline harness and uses a
self-generated He-init MLP against a self-computed 400k-sample target. Its `K = 1` value (5.95e-05)
is 26% below the 8.03e-05 we measure for `examples/03` through the official harness. Only the
*relative* `K = 1 → K ≥ 2` comparison, within one experiment on one network, is claimed.

**Full logs.** `work/RESEARCH.md` (chronological research log, including every conclusion this
document retracts), `work/LITERATURE.md`, `work/adversarial/*.md` (ten probe reports), `STATE.md`.

---

## Appendix B: The reproducibility spread of our own constants

A write-up that quotes `C` to four significant figures and cannot reproduce the second is worse than
one that quotes a band. Across ten independent probes, the **same** whitened+antithetic estimator
produced:

| `C = MSE·k` | conditions |
|---|---|
| 0.02227 | 64 MLPs, four-sampler decomposition |
| 0.02237 | 64 MLPs, k = 15,112 |
| 0.02257 | 60 MLPs, k = 8,192, 4 reps |
| 0.02289 | 64 MLPs, k = 65,536, 4 reps, n = 256 paired |
| 0.02322 | 120 MLPs × 3 reps, n = 360 paired |
| 0.02345 | 64 MLPs |
| 0.02383 | 120 MLPs × 40 reps, k = 4,096 |
| 0.02526 | 64 MLPs, k = 4,096, 2 reps |

**An 11% spread**, systematically ordered by `k` and by whether the reference was centred on exact
ground truth. Three practical consequences: (i) never quote "the" variance constant without its `n`
and `k`; (ii) a 1.1× claim supported by two unpaired runs from different probes is inside this
spread and means nothing; (iii) the figure of merit `FOM = C·c` inherits the spread *and* a second
one from `c` — baseline FOMs across our own corpus range 4.03e4 to 1.06e5 purely because `c` differs
and was not always stated. **Any FOM quoted without its cost basis is meaningless.** We quote
per-probe values with their conditions rather than a corpus average.

The same applies to the layer-31 anchoring number, which ranges 46.5–52.1 across the corpus
depending on whether the anchor is single-layer or cumulative, hard or weighted, and at which `k`.
We state our definition in §2.2.

---

## Appendix C: AI-assistance disclosure

This work was carried out with the assistance of Claude (Anthropic), used under the participant's
direction for code generation, experiment execution, analysis and drafting. The competition rules
expressly permit this: development of Solution code with the assistance of LLMs or other AI tools is
permitted under §5.7, and the Rules' Technical Writeup provision states that writeups "may be
drafted with the assistance of LLMs or other AI tools, provided that the Participant takes
responsibility for the accuracy and completeness of the final writeup."

All experimental design decisions, all judgements about what to ship, and the decision to reject the
residual-wall-time billing channel (§6.5) were made by the participant. The participant has reviewed
the contents of this document and takes full responsibility for the accuracy and completeness of
every claim and measurement reported in it.
