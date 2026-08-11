# Research log — WhestBench estimator development

> Session 1, 2026-08-04. Every number below is a real measurement; the scripts that
> produced them are in `work/experiments/`, and the official-harness numbers come from
> `whest run --runner subprocess` on the full 100-MLP public mini split.
> This log is the source material for the Phase 1 algorithmic-contribution write-up (due Aug 17).

## 1. Reading the scoring function

```
adjusted_final_layer_score = final_layer_mse × max(0.1, effective_compute / flop_budget)
effective_compute          = flops_used + λ · residual_wall_time_s,   λ = 1e11 FLOP/s
flop_budget                = 2.72e11 per MLP
```

Two structural observations dominate everything that follows.

**(a) The multiplier floors at 0.1, so ~10% of the budget is free.** Any estimator spending
less than 2.72e10 effective FLOPs is charged the same 0.1 multiplier as one spending exactly
that much.

**(b) Above the floor, the score is exactly flat for any 1/k estimator.** For Monte Carlo,
MSE = C/k and cost = c·k, so for utilisation ≥ 0.1

```
score = (C/k) · (c·k / B) = C·c / B        — independent of k
```

The optimum therefore sits *exactly at the 10% line*: below it more compute strictly helps,
above it more compute is pure risk for zero gain. **The shipped covariance-propagation
baseline spends 1.33% of the budget.** It leaves ~7.5× of free compute unused, and that —
not a cleverer series — turned out to be the whole opening.

Also: only the **final layer** is scored. `all_layers_mse` is a diagnostic.

## 2. Where the analytic baseline's error actually lives

`work/experiments/diag.py`. Since E[y_l,i] = E[ReLU(z_l,i)] depends only on the *marginal*
law of z_l,i, the error splits into (i) error in the predicted marginal moments and
(ii) intrinsic non-Gaussianity of z_l,i. Feeding **true** MC-measured marginal moments into
the exact Gaussian ReLU formula isolates (ii):

| quantity (width 256, depth 32, MC reference 400k samples) | final-layer MSE |
|---|---|
| covariance propagation (shipped `examples/03`) | 5.96e-05 |
| Gaussian formula fed with **true** marginal moments | **4.06e-06** |
| MC noise floor of the diagnostic itself | 5.1e-07 |

So ~93% of the baseline's error is mis-estimated moments, not non-Gaussianity. The predicted
mean is good (0.83% relative error at the final layer) but **the predicted standard deviation
is 11.4% off**, and that error grows monotonically with depth (0.09% at layer 1 → 11.4% at
layer 32).

## 3. Negative result: making the covariance update exact does not help

`work/experiments/mehler.py`. For jointly Gaussian pre-activations, the post-ReLU covariance
has an exact Mehler/Hermite expansion

```
Cov(ReLU(u_i), ReLU(u_j)) = Σ_{k≥1} (ρ_ij^k / k!) · c_k(i) · c_k(j),
      c_k(i) = E[ReLU(μ_i + σ_i a) He_k(a)],   a ~ N(0,1)
```

**The shipped "gain" heuristic `Φ(α_i)Φ(α_j)·cov_ij` is precisely the order-1 truncation**,
because c_1(i) = σ_i Φ(α_i). Carrying the expansion further reproduces brute-force pair
covariances to four decimals (verified against 4M-sample MC on random pairs) — but through
32 layers it makes the final answer *slightly worse*:

| truncation order K | final-layer MSE |
|---|---|
| 1 (= shipped baseline) | 5.95e-05 |
| 2 | 6.54e-05 |
| 4 | 6.55e-05 |
| 8 | 6.54e-05 |
| 12 | 6.54e-05 |

**Interpretation.** The pairwise formula is not the bottleneck; the *joint* Gaussian surrogate
is. Order-1's under-estimate of the covariance happens to partially cancel the compounding
surrogate error, and removing it exposes the real bias. Analytic propagation of this family
is stuck around 6e-05 regardless of how exact its per-layer algebra is.

*(This is the most interesting finding of the session and the strongest candidate for the
algorithmic-contribution write-up: it identifies a wall, explains it, and shows the shipped
baseline is a first-order truncation of a series that does not pay to extend.)*

## 4. What actually works: spend the free budget on sampling

`work/experiments/hybrid.py`, mean final-layer MSE over 5 MLPs, all at the free ceiling:

| estimator | MSE | vs baseline |
|---|---|---|
| covariance propagation | 1.08e-04 | 1.0× |
| **plain Monte Carlo, k = 6,484** | **7.45e-06** | **14.4×** |
| antithetic MC | 9.11e-06 | 11.8× |
| MC moments → analytic ReLU mean | 9.36e-06 | 11.5× |

Monte Carlo is also **unbiased** — its error is pure variance, so unlike any analytic scheme
it has no floor. Antithetic sampling *hurts*: ReLU networks are not odd, so f(x) and f(−x)
stay positively correlated and k/2 antithetic pairs beat neither k independent draws.

## 5. Variance reduction: moment-matched (whitened) inputs

`work/experiments/vr.py`. The input law N(0, I) is known exactly, so force the ensemble to
match it exactly in its first two moments: `X ← (X − mean X) · G^{−1/2}`, `G = cov(X)`. An
affine re-standardisation of an i.i.d. Gaussian ensemble is still a valid N(0, I) ensemble,
but its first- and second-moment sampling error is now identically zero.

| estimator (7 MLPs, equal cost) | MSE | vs plain MC |
|---|---|---|
| plain MC | 6.38e-06 | 1.00× |
| **whitened MC** | **4.09e-06** | **1.56×** |
| whitened + Rao-Blackwellised final layer | 5.76e-06 | 1.11× |
| 50/50 blend | 4.52e-06 | 1.41× |

Whitening also tightens the spread across MLPs markedly (worst case 6.8e-06 vs 1.4e-05),
which matters because the private re-evaluation draws fresh MLPs.

The whitener is **fused into the first weight matrix** — `X @ (G^{−1/2} W_0)` instead of
`(X @ G^{−1/2}) @ W_0` — turning a k·n² pass into an n³ one and buying back ~3% more samples.

## 5c. Antithetic pairing helps — but ONLY on top of whitening

§4 measured antithetic sampling as *harmful* (9.11e-06 vs 6.38e-06 for plain MC): ReLU networks
are not odd, so f(x) and f(−x) stay positively correlated and k/2 antithetic pairs lose to k
independent draws. That conclusion was right for plain MC and **wrong once whitening is
applied** — the two together match N(0, I) exactly in moments 1 through 3 (antithetic kills every
odd moment by construction; whitening fixes the second), which neither does alone.

Measured by **paired comparison with common random numbers** on real competition MLPs from the
`full` split — same MLPs, same base draws, equal FLOP budget, so the *difference* has far lower
variance than either estimate:

| variant | mean MSE (250 MLPs) | ratio vs whitened | paired t |
|---|---|---|---|
| whitened (ours) | 4.46e-06 | 1.000× | — |
| whitened + antithetic | 4.03e-06 | 1.106× | +1.30 (n.s.) |
| whitened + exact layer-1 mean | 4.49e-06 | 0.994× | −0.57 (n.s.) |
| whitened + antithetic + layer-1 mean | 3.87e-06 | 1.151× | +1.98 (n.s.) |

At n = 250 nothing clears significance, so the antithetic term was re-tested alone at n = 750:

| | mean MSE (750 MLPs) | ratio | paired t |
|---|---|---|---|
| whitened | 4.4029e-06 | 1.000× | — |
| **whitened + antithetic** | **3.9255e-06** | **1.122×** | **+3.03 — significant** |

**Adopted.** It costs nothing (the negation and concatenate are ~3e6 FLOPs against a 2.6e10
budget). The layer-1 mean anchor was *not* adopted — it is neutral here, independently
confirming §5b.

This is also a caution about the multi-agent design round: one candidate's headline advantage
came almost entirely from this antithetic term, and its 100-MLP margin over ours (19%) sat
inside the split's own 21% noise band (§7b). The paired test is what separated the real
ingredient from the noise.

## 5b. Negative result: exact layer-1 moment matching adds nothing

`work/experiments/layer1.py`. The input is exactly N(0, I), so layer 1's pre-activation
`z_1 = W_0ᵀx` is **exactly** jointly Gaussian with covariance `W_0ᵀW_0`. Its post-ReLU mean is
therefore exact in closed form, and its post-ReLU covariance is exact via the Mehler expansion
of §3 — which *is* valid here, unlike at layers ≥ 2. So the sample ensemble can be pushed onto
its exactly-known layer-1 moments. Two variants, 8 MLPs:

| variant | MSE | vs whitened MC |
|---|---|---|
| whitened MC (baseline) | 3.28e-06 | 1.00× |
| + exact layer-1 **mean** (essentially free) | 3.13e-06 | 1.05× |
| + exact layer-1 mean **and covariance** (costs 2 extra k·n² passes) | 3.10e-06 | 1.06× |

Both gains are noise-level — the per-MLP results disagree in sign (3 of 8 got *worse*).
**Input whitening already captures essentially all of the available moment-matching gain**;
imposing exact information one layer deeper buys nothing. Not adopted.

## 6. flopscope 0.10.0 cost calibration (measured directly)

| operation | measured cost | model |
|---|---|---|
| matmul, float32 | 130,816,000 for (1000×256)@(256×256) | **2.0 FLOPs/MAC** |
| matmul, float64 | 261,632,000 | **4.0 FLOPs/MAC** |
| matmul, float16 | 130,816,000 | 2.0 FLOPs/MAC — **no discount below float32** |
| relu / add, float32 | 1.0 per element | 1.0 |
| sum / mean | 1.0 per element read | 1.0 |
| `eigh` (256×256) | 150,994,944 | 9·n³ |
| (n,n)@(n,n) | 33,488,896 | 2·n³ |

**float32 is exactly half the price of float64**, which doubles the affordable sample count
and therefore halves the MC variance. float32 forward-pass error is 9.2e-13 MSE against
float64 — five orders of magnitude below our target, so it is free accuracy-wise.

**float16 was ruled out**: no price advantage at all, and it introduces a 1.27e-06 MSE bias
floor (a third of our current total error).

## 7. Result — `work/mine/wmc.py`

Official harness, **1,000-MLP `full` split**, subprocess runner (see §7b for why the mini split
must not be used for this):

| | shipped `examples/03` | ours v1 (whitened) | **ours v2 (+ antithetic)** |
|---|---|---|---|
| **adjusted final-layer score** | 8.03e-06 | 4.47e-07 | **3.98e-07** |
| raw final-layer MSE | 8.03e-05 | 4.47e-06 | 3.98e-06 |
| all-layers MSE | 5.51e-05 | 8.22e-06 | 7.55e-06 |
| score multiplier | 0.1000 (floored) | 0.1000 | 0.1000 (floored) |
| compute utilisation | 1.32% | 9.54% | 9.53% |
| residual wall time | — | — | 1.0 ms/MLP |
| **failed MLPs** | 0 / 1000 | 0 / 1000 | **0 / 1000** |

**20.2× better than the strongest shipped baseline**, zero failures across a thousand MLPs.

The v1 → v2 step measured 1.123× through the official harness, against the 1.122× predicted by
the offline paired test of §5c — the two methods agree to within 0.1%, which validates
`work/offline_bench.py` as a stand-in for 25-minute harness runs.

A final revision raised `_TARGET_UTILISATION` from 0.095 to 0.099 (see §7c: score ∝ 0.1/u below
the floor, so the unused allowance was costing ~4%):

| v3 (submitted), 1,000-MLP full split | |
|---|---|
| **adjusted final-layer score** | **3.85e-07** |
| raw final-layer MSE | 3.84e-06 |
| all-layers MSE | 7.26e-06 |
| compute utilisation | 9.93% |
| score multiplier | 0.10002 |
| residual wall time | 1.03 ms/MLP |
| **failed MLPs** | **0 / 1000** |

**20.9× better than the incumbent.** The full progression: 8.03e-06 (incumbent) → 4.47e-07
(whitened MC) → 3.98e-07 (+ antithetic) → 3.85e-07 (+ utilisation).

The multiplier came out at 0.10002 rather than exactly 0.1, i.e. a handful of MLPs crossed the
free line — this costs 0.025% and is immaterial. It confirms the operating point is genuinely at
the knee.

## 7b. METHODOLOGY: the 100-MLP mini split cannot resolve differences under ~21%

`work/offline_bench.py`. The public dataset ships its own ground truth — every row carries
`final_means` (the scored layer, baked at n_samples = **1e9**) and `all_layer_means`. So
estimator *accuracy* can be measured offline in numpy against the exact competition MLPs and
exact targets, in ~60 s instead of the harness's ~25 min. (This also supersedes the
methodology of `work/experiments/*`, which used self-generated MLPs and self-computed 2e6-sample
Monte-Carlo targets.)

Running the **same estimator** on the **same 100 MLPs**, changing only the RNG stream:

| stream | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| final-layer MSE | 5.04e-06 | 4.07e-06 | 4.23e-06 | 3.98e-06 | 4.70e-06 | 3.92e-06 |

mean 4.32e-06, sd 4.51e-07 → **relative sd of the 100-MLP estimate is 10.4%**.

**Consequence: two candidates differing by less than ~21% (2σ) on the mini split are not
distinguishable.** The per-MLP error distribution is heavy-tailed (worst MLP ~5× the mean), so
the mean over only 100 MLPs is a noisy statistic.

This retroactively invalidates any ranking done on the mini split alone — including the
3.16e-07 / 3.52e-07 / 3.76e-07 ordering of the three workflow candidates against ours, whose
whole spread (19%) sits inside one 2σ interval. **All comparisons must be made on the 1,000-MLP
`full` split**, where the sd falls by √10 to ~3.3% and differences above ~7% become meaningful.

It also sets a ceiling on what public-leaderboard tuning is worth: the public score is one draw
from this distribution, and Phase 1 rank is decided by a *private* re-evaluation on fresh MLPs
regardless. Chase genuine variance reduction, not leaderboard decimals.

## 7c. The 1,000-MLP run-off (official harness, subprocess runner)

| estimator | adjusted score | raw final MSE | utilisation | failed |
|---|---|---|---|---|
| `examples/03` (incumbent) | 8.03e-06 | 8.03e-05 | 1.32% | 0/1000 |
| ours, whitened only | 4.47e-07 | 4.47e-06 | 9.54% | 0/1000 |
| workflow candidate `kink-targeted-whitened-antithetic-mc` | **3.69e-07** | 3.69e-06 | 9.81% | 0/1000 |
| workflow candidate `hermite-anchored-moment-matched-mc` | 3.80e-07 | 3.80e-06 | 9.33% | 0/1000 |

Both candidates' mini-split numbers (3.16e-07, 3.52e-07) were optimistic by ~15% — exactly the
noise band of §7b. On 1,000 MLPs the two are 3% apart, i.e. tied.

**The residual gap to ours decomposes into two things, both now understood:**
1. the antithetic term, worth 1.122× (§5c) — now adopted;
2. **compute utilisation**: they sit at 9.81% / 9.33% against our 9.54%.

Point 2 is a scoring subtlety worth stating plainly. Below the floor the score is
`0.1 × C/k` with `k ∝ u`, so **score ∝ 0.1/u — every unused percent of the free allowance costs
a proportional percent of score.** Above the floor it is flat. The penalty is therefore mild and
roughly symmetric around u = 0.1, and the optimum is to sit as near 0.1 as the residual-wall-time
uncertainty allows. Measured residual wall time is 0.92 ms/MLP (0.034% of budget), so even a 10×
slower grader core leaves ample room; targeting ~0.099 is safe, and overshooting is not a cliff
(it costs ~1% of score per 1% of overshoot, unlike a *budget* overrun which costs ~100,000×).

## 7d. THE CEILING, MEASURED — why this whole approach caps at ~3.4e-07

After submission #323440 came back at rank #121 (leader 5.10e-08), the obvious question is whether
we can simply buy our way up by spending more budget. Measured directly, whitened+antithetic MC on
120 real competition MLPs:

| budget share | k | raw final MSE | improvement vs 10% | **adjusted score** |
|---|---|---|---|---|
| 10% | 6,214 | 3.457e-06 | 1.00× | **3.457e-07** |
| 20% | 12,478 | 1.770e-06 | 1.95× | **3.540e-07** |
| 40% | 25,008 | 9.156e-07 | 3.78× | **3.663e-07** |
| 80% | 50,066 | 4.755e-07 | 7.27× | **3.804e-07** |

MSE falls as ~1/k (in fact a touch *worse* than 1/k — 7.27× for 8× the budget, as whitening's
benefit saturates), so **the adjusted score is flat, and very slightly worsening.** No amount of
compute moves this estimator off ~3.4e-07.

**This is the ceiling of every sampling-based approach, and we are sitting on it.** The leader's
raw MSE of 5.21e-08 at ~full budget is 6.5× better than MC at equal compute — about 42× in
effective sample count. That is not a tuning gap; it is a different class of estimator.

**Consequence for all future work:** the only designs worth building are ones with a credible
argument for error scaling *better* than 1/k. Everything else is a rearrangement of a solved
problem. The leading candidate remains the rank collapse (§8) → low effective dimension → QMC/
dimension-reduction territory, where 1/k² MSE scaling is achievable in principle.

## 7e. CORRECTION to §7d — "the sampling lineage is closed" was a false inference

§7d measured MSE·k = C = 2.375e-02 with exponent p = 1.008 and concluded the lineage was closed.
**The measurement is right; the inference was wrong.** It proves only that we cannot win by
*spending more compute*. The compute-invariant figure of merit is

```
Φ = adjusted score = C · c / B          (C = variance constant, c = cost per sample)
```

which falls if *either* C or c falls. §7d proved nothing about c. Dead-column pruning (§8) had
already exploited exactly that channel; the conclusion should have been "the *variance* axis is
closed", not "sampling is closed". Public evidence puts the sampler frontier near 1.5e-07, i.e. we
were ~2× above it, not sitting on it.

## 7f. The on/dead/kink route — mechanism refuted, cost gain real

The published recipe (AIcrowd topic 18106, adjusted 1.551e-07) is: classify neurons as **dead**,
**always-on** (ReLU is the identity ⇒ exactly linear), or **kink**; collapse runs of always-on
layers into exact linear maps with zero sampling variance; sample only the kink set.

**The stated mechanism does not work, and the refutation is clean.** For a mean, the sample mean
commutes with a linear map: if neuron *i* never crosses zero on the draw, `mean_s[ReLU(z_i)]` and
`mean_s[z_i]` are *the same sum*. Measured: **bit-identical, max difference 0.000e+00** in float64
— not "identical up to rounding". Identifying a neuron as linear cannot reduce variance in a mean.

Three further measurements bury it:

| finding | number |
|---|---|
| always-on neurons' share of final-layer MSE | **72.7%** (25% of neurons, 5.2× the per-neuron error of kink) |
| dead neurons' share of MSE | **0.00%** — ReLU already emits exact zeros |
| freezing always-on neurons at their sample mean mid-network | **65–270× worse** |
| P(all 256 neurons of the best layer are on) | 0.255²⁵⁶ ≈ **1e-152** |
| steelman: Gaussian plug-in with exact empirical μ, σ | **2.2× worse** than plain MC |

So the mechanism's entire target class is the class that dominates the error, and it moves it by
exactly zero; the class it does "solve" already contributes nothing. A run-collapse never happens.

**But the route still pays — through cost, not variance,** exactly as the objection predicted.
Verified independently, 320 full-split MLPs, paired with common random numbers, inside a real
`BudgetContext`:

| | adjusted | raw MSE | utilisation | residual/MLP | over budget |
|---|---|---|---|---|---|
| `work/mine/wmc2.py` (live) | 3.1389e-07 | 3.9437e-07 | 0.799 | 15.3 ms | 0/320 |
| `work/kink/combined.py` | **2.5523e-07** | 3.9130e-07 | 0.656 | 4.8 ms | 0/320 |

**1.2298×, paired t = +16.87.** Raw-MSE paired t = **+1.05 (n.s.)** — the accuracy is unchanged and
the entire gain is a cheaper pass. Bias from the a-priori masking: 0.41% of MSE, of which only ~18%
is k-independent (~0.08%); accuracy difference against the 1e9-sample truth is not measurable
(paired t = +0.12). Adversarial worst-case utilisation 0.9130 (depth-5 MLP), below the live
submission's own 0.9149.

**The lesson worth writing up:** a mechanism can be wrong in its stated explanation and still
produce a real, reproducible gain by a different route. Testing *why* something works, before
building on it, is what converted a refuted 1.74× story into a verified 1.23×.

## 7g-RETRACTION (2026-08-07) — ⚠️ THE CONCLUSION BELOW IS WITHDRAWN

**§7g's measurements stand. Its conclusion — "no constructible estimator reaches 1e-09" — is
false, and must not appear in the write-up.** The evidence that overturned it came from
per-submission evaluation data; the figures themselves are withheld here (see below).

> **[REDACTED IN THE PUBLIC RELEASE.]** This passage reported per-submission evaluation
> telemetry for named participants. We withdrew the inference it supported, we make no claim about any
> participant's methods, and we do not think per-entrant operational data belongs in a
> public document regardless of what it shows. The technical conclusion that survives is
> stated without it: on the billing-independent figure of merit `C_eff = MSE·N`, the
> leading entries sit two to three orders of magnitude below us — they have a real method.

2. **C_eff is billing-independent, and they are 200–450× below us on it.** That axis is the one
   §7g itself declares decisive. They have a real method.

3. **§7g contains an arithmetic error that inflated the apparent gap tenfold.** It reports the
   oracle-fed deterministic closure (bias² = 6.3e-08) as "63× above the requirement" of 1e-08.
   It is **6.3×**. And measured against the actual leader — adjusted 4.00e-10 at the 0.1 floor,
   i.e. raw MSE 3.63e-09 — the oracle closure is **17× short, not 63× and not 8,000×.** A
   deterministic method carries no N and sits at the floor for free.

**What survives.** The two-annihilator enumeration is sound *for unbiased averaging estimators*.
It says nothing about the **deterministic branch**, which §7g closed on a single extrapolation
(`C(D) ∝ D^−0.574` fitted on D ∈ {1,3} and pushed four orders out) plus the mis-scaled comparison
above. It also holds per-sample cost fixed at a full forward pass, while §8 measures r90 = 8 at
layer 31 — any estimator whose late layers cost `2knr` instead of `kn²` moves C_eff linearly and
is unpriced by §7g.

**The gap is ~17× and it is engineering, not exponents.** It sits in one identified place: the
quality of the moments feeding a final-layer Gaussian/Edgeworth closure.

> **[REDACTED IN THE PUBLIC RELEASE.]** This passage discussed a hypothesis about how leading
> entries might be achieving their scores, together with some platform internals. We withdrew the
> hypothesis, and the rest is not ours to publish.


---

## 7g. (SUPERSEDED — measurements retained, conclusion withdrawn; see 7g-RETRACTION above)
### The attainability analysis — why 1e-09 looked unreachable on 256×32

A maximum-effort search over four independent method classes — spherical cubature under positive
homogeneity, exact orthant/sign-pattern decomposition, offline amortisation across MLPs, and
exotic (characteristic-function / transfer-operator) formulations — returned **all four dead**,
each with a numerical probe on real competition MLPs. What it produced instead is the sharpest
result of the whole project.

**The requirement.** Because N_free·c = 0.1B identically, both branches of `max(0.1, ·)` collapse to
one budget-independent condition. Any estimator averaging N evaluations of per-node cost c needs

```
C_eff = MSE·N ≤ 1e-9 · B / c        and, since MSE = bias² + var,   bias² ≤ 1e-8
```

At our c = 2.645e6 that is **C_eff ≤ 1.03e-4** — 183× below our measured C = 1.886e-2. Node count
is irrelevant to the score; only the product C_eff·c is.

**The deterministic branch is closed by construction.** Best constructible D(W) is covariance
propagation at bias² = 8.0e-5. Feeding it oracle inputs no estimator can produce — the exact
1e9-sample mean, sd, skewness and kurtosis of every final pre-activation — still lands at
**6.3e-08, 63× above the requirement**. Cumulant propagation, the only route to those inputs, costs
168–227% of budget. **No constructible deterministic method is within 8,000× of the requirement,
and no oracle within 63×.**

**The stochastic branch admits exactly two known variance annihilators, and both are priced out.**

| annihilator | cost law | what 1e-09 demands |
|---|---|---|
| polynomial exactness to degree D | ≥ dim P₍D/2₎ = C(256+s, s) nodes — 257 at D=3 (where we are), 33,153 at D=5, 9.7e9 at D=11 | measured tail C(D) ∝ D^−0.574 ⇒ **D ≈ 1.3e4**, i.e. ~10^13000 nodes |
| exact integration over a d-dim subspace | ~K^d cells | 26% of variance sits at chaos degree 1, so VALUE(d) ≤ 1/(a₁(1−d/256)) ⇒ **d ≈ 252** (177 after whitening), ~3,600^177/177! cells |

**Every idea in the search — orthants, fibres, spherical designs, divergence identities, transfer
operators, contour methods — is one of those two constructions in different coordinates.**

> **The gap is not one of insight but of exponents.** Exactness in this problem is bought in units
> that cost C(256+s, s) nodes or K^d cells, and 1e-09 requires ~10^13000 of the first or ~10^600 of
> the second.

**Two independent cross-checks validate the picture.** Rao-Blackwellising the full sign pattern
equals radial integration, measured 1.035×; the chaos spectrum independently predicts 1.052× — two
unrelated measurements agreeing to 1.6%. And positive homogeneity, exact and free, buys nothing
precisely *because* it removes the one direction along which the function was already linear.

**Stated plainly for the write-up:** an adjusted score of 4.00e-10 implies C_eff = 4.1e-5, requiring
polynomial exactness to degree ≈ 4e4. Under this analysis no constructible estimator produces it at
width 256, depth 32.

**One free item did survive**, not because it approaches the target but because it costs nothing:
the **random orthoplex {±bᵢ}** is an *exact, unbiased* 3-design (ours is only a randomised one). It
removes the O(1/k) whitening bias and drops the eigh entirely (9n³ = 1.5e8). Measured 1.019× —
head-to-head against our whitening at identical FLOPs, C = 2.1970e-2 vs 2.1995e-2, paired t = −0.01,
i.e. the two are statistically identical in variance and the orthoplex wins only on cost.

## 8. Ruled out / open

**Ruled out with evidence:** higher-order Mehler expansion (§3); antithetic sampling (§4);
Rao-Blackwellising the final layer, alone or 50/50 blended (§5); exact layer-1 moment matching,
mean or mean+covariance (§5b); float16 (§6).

### ~~Best open lead — rank truncation~~ **CLOSED 2026-08-05. It does not work.**

> I flagged the rank collapse below as the highest-value remaining idea. It was tested properly
> and it is **illusory**. Truncating layers ≥16 to rank r and comparing on the *same* ensemble
> (isolating pure truncation bias, no sampling noise):
>
> | r | truncation-bias MSE | × the 4.2e-07 noise floor |
> |---|---|---|
> | 48 | 8.87e-05 | **211×** |
> | 64 | — | 128× |
> | 128 | — | 9.6× |
> | 192 | under the floor | — but r=192 **costs more** than not truncating (n/2r = 0.67×) |
>
> **The lesson: energy rank and the rank needed for 0.07% relative accuracy in the mean differ by
> about 4×.** r99 = 47 says 99% of the *variance* lives in 47 directions; it does not say the
> discarded 1% is harmless for a quantity we need to six significant figures. I conflated the two.
> Do not spend time here.

The underlying measurement, kept because it is still the right description of the geometry —
effective rank of the centred k×256 activation matrix (energy fraction captured by top r
directions) on a real competition MLP:

| layer | r for 90% | r for 99% | r for 99.9% |
|---|---|---|---|
| 0 | 171 | 244 | 255 |
| 7 | 58 | 162 | 206 |
| 15 | 31 | 115 | 168 |
| 23 | 14 | 66 | 126 |
| 31 | **8** | **47** | 111 |

The *cost* argument was right in form — per-layer cost would fall from k·n² to ~2k·n·r + r·n²,
paying whenever r < n/2 — but the accuracy premise was wrong, as measured above.

### What DID work: exact dead-column pruning (adopted 2026-08-05)

A neuron that is dead (ReLU output zero for every sample in a chunk) contributes exactly nothing
to the next layer, so its column can be skipped. This is **exact, not approximate** — verified at
identical k on real MLPs, pruned vs unpruned MSE ratio 1.000014 (float32 summation order only).
It buys ~1.17× by making each sample cheaper, so more samples fit the budget.

Note it does **not** change the error scaling: measured p = 1.008, i.e. still exactly 1/k. It is a
constant-factor win, not a regime change.

**Why it must run at high utilisation (a subtlety that reverses the §7c reasoning):** below the
0.1 floor the multiplier is clamped regardless of *actual* spend, so a cheaper forward pass earns
nothing — yet `k` must still be sized against the worst case (nothing prunes), leaving samples on
the table. Above the floor the multiplier tracks actual spend, so the saving is credited directly.
Hence `_TARGET_UTILISATION = 0.92` (worst-case sizing; real spend lands ~0.80). The 8 points of
unclaimed headroom cost ~0.3% of score and buy 218 ms of residual-wall-time margin against a
measured 42 ms — the grader can be 5× slower and still not reach the cliff.

**A trap recorded from that work:** a fallback ladder whose second rung re-runs Monte Carlo is
harmless at 10% of budget and *fatal* at 80% — a float32-overflow MLP makes the first attempt
raise and the pair reaches 1.88× of budget, zeroing the prediction and forcing multiplier 1.0.
The ladder must contain exactly one expensive rung.

### The sampling lineage is now closed

C = MSE·k = **2.375e-02** (measured, two independent estimates agree). Reaching the leader's
5e-08 would need k ≈ 475,000 — **8.3× more effective samples than the full budget buys** — and
measured p = 1.008 means there is no superlinear scaling to exploit. **No amount of sampling
cleverness reaches 5e-08.** Getting there requires a fundamentally different estimator.

### ~~The error is not isotropic~~ **CLOSED 2026-08-05. It is noise, not bias.**

`work/scale_mode.py`. The concentration is real and even larger than first reported — on 150 real
MLPs, **30.7% of the final-layer error energy lies along the truth direction** (78.5× the isotropic
1/256 baseline) and **10.0% along the constant vector** (25.6×). An oracle that removed both would
gain **1.69×**.

But concentration is not correctability. Running the same estimator on the same MLP with 5
independent random streams and decomposing the variance of the scale coefficient α:

| | ICC | MS_between | MS_within |
|---|---|---|---|
| scale α | **−0.0107** | 2.096e-06 | 2.214e-06 |
| offset | **−0.0132** | 1.466e-04 | 1.568e-04 |

**ICC ≈ 0** (slightly negative, the signature of exactly zero systematic component): between-MLP
variance is no larger than within-MLP variance. α is a property of the *random draw*, not of the
MLP. Per-MLP mean |α| = 5.2e-04 against a within-MLP sd of 1.4e-03 — the "signal" is a third of
its own noise. **There is nothing to learn and nothing to ship.**

**The methodological lesson, which is worth more than the result:** *error energy concentrating in
a direction is not evidence of a correctable bias.* Monte-Carlo error here is heteroscedastic —
neurons with larger means have proportionally larger error — so the error vector aligns with the
truth direction *by construction*, with no bias whatsoever. If e_i has sd ∝ t_i, the expected
energy fraction along t̂ is Σt⁴/(Σt²)², which is many times 1/n for any spread-out t. The 78.5×
enrichment is fully explained by that and implies nothing. Only the variance decomposition
separates the two, and it says noise.

**With this closed, the sample-mean lineage is finished.** ~2.9e-07 is our ceiling.

**Other open leads, in rough order of expected value:**
1. A control variate with an *exactly known* expectation. Layer 1 is exact in closed form
   (E[y_1,i] = ‖W_0[:,i]‖/√(2π)), so the layer-1 sampling error is observable — the question
   is whether it correlates usefully with the final-layer error and whether the regression
   coefficient can be estimated without spending the gain on noise.
2. Higher-order moment matching of the input ensemble (third moments), or a stratified /
   quasi-random construction in 256 dimensions.
3. Re-deriving the cost model per-MLP from `budget` at run time rather than from calibrated
   constants, so the estimator adapts if ARC re-pins the budget or the conversion rate.

**Risk to watch:** the forum post says Phase 2 may require *all* numerical work to run through
flopscope and may cap residual wall time. Neither would affect this estimator — it is already
entirely inside flopscope primitives with 0.006 s of residual wall time per 100 MLPs — but a
change to the 0.1 multiplier floor would invalidate the whole compute-allocation argument.

---

# 7h. THE ADVERSARIAL PASS (2026-08-07) — §7g's conclusion survives; its argument does not

An independent session was tasked with **breaking** §7g. Ten measured probes on real `full`-split
MLPs against the dataset's own 1e9-sample truth, plus a self-verification round. Verdict:

> **Nothing reaches 1e-09. Nothing reaches even 1.1×.** The one positive result the pass produced
> — a 1.30× cost gain — was refuted on residual wall time by the verification round. §7g's
> *conclusion* stands; two of its three legs were wrong and are replaced below.

Everything is in the budget-invariant figure of merit **FOM = C·c** (Φ = FOM/2.72e11):
plain MC 1.75e5 · **live submission 4.99e4** · published best 4.2e4 · **target ≤ 272**.

Full probe reports: `work/adversarial/*.md`.

## 7h.0 Harness calibration

Before anything else, the offline harness was re-validated against every published constant
(120 MLPs × 3 reps, n = 360 paired):

| | measured here | previously logged |
|---|---|---|
| C_plain | 5.035e-02 | 5.219e-02 |
| C_whitened | 2.695e-02 | 2.701e-02 |
| C_whitened+antithetic | 2.322e-02 | — |
| whitening gain | 1.869×, t = +9.36 | 1.93× |
| antithetic gain on top | **1.160×, t = +3.23** | 1.122×, t = +3.03 |

## 7h.1 The whole problem is one vector: μ₃₁

Four probes starting from unrelated premises all terminated on the same object.

Anchoring the sample ensemble at hidden layer L onto the exact `all_layer_means`, then continuing
(replicated independently twice: 60 MLPs → VR 50.8×, and 64 MLPs → VR 50.6×):

| anchored | 1 | 4 | 8 | 16 | 24 | 30 | **31** | 1..30 (not 31) | all 1..31 |
|---|---|---|---|---|---|---|---|---|---|
| VR | 1.13× | 1.30× | 1.76× | 3.03× | 6.43× | 25.8× | **50.8×** | 26.2× | 51.2× |

**95% of the celebrated 57× comes from anchoring layer 31 alone**; single-layer perturbation gains
are g₃₁ = 0.950 with every other g_l ≤ 0.013. This is not information arriving — it is the identity
`Ȳ₃₂ = ½(W₃₂ᵀ Ȳ₃₁ + mean|z₃₂|)`, in which the sample mean commutes with W₃₂ᵀ exactly. **It is an
identity, not an estimator.**

**Required anchor accuracy.** Perturbing the layer-31 anchor by relative eps gives
`MSE(eps) = 1.111e-07 + 0.79·eps²` (quadratic confirmed over three decades). Reaching raw
MSE ≤ 1e-08 needs the anchor's contribution below the sample mean's 5.54e-06 by **554× in
mean-squared error** (23.5× in RMS, i.e. 554× more samples) — on μ₃₁, the same species of quantity
as the answer. The oracle is not "not yet obtained"; **it is the problem, one layer up.**

**And a perfect μ₃₁ oracle is still not enough.** C = 4.54e-4 ⇒ FOM ≈ 925 at the best real c,
i.e. Φ ≈ 3.4e-09 — above target and behind leaderboard #2. Only the *full covariance* oracle
reaches it (C = 8.56e-5 at a 1M-sample reference, VR 394×, confirming §A4.3's 272–341×), and its
constructible version (reference means rather than oracle) measures **0.92×, worse than baseline**.

The deterministic branch lands in the same place from the opposite direction. **The reconstruction
step is solved:** maximum-entropy density matching 6 exact marginal moments of z₃₂ gives
bias² = **5.08e-09**, below the 1e-08 requirement, with zero fitted parameters (M = 8 → 8.15e-10;
K-stable to 0.7%). 100% of the blockage is acquiring the **order-1** moment,
`E[z₃₂,i] = wᵢ·μ₃₁`. Same wall.

*Corrections to the §A6 ladder in passing:* the Gram–Charlier minimum is order **4** (2.304e-08,
confirming the 2.5e-08 figure), not order 7; order 7 is 3.22e-08, i.e. 1.4× **worse**; divergence
starts at order **9**. Edgeworth and Gram–Charlier are bit-identical here (there is no n^(−1/2) to
reorder by), which closes "try a proper Edgeworth".

## 7h.2 The recurring trap, stated generally

For z with the measured a-distribution, the projection of ReLU onto polynomials of degree ≤ 6
carries **99.2%** of Var(ReLU(z)); the residual is 0.78%. So *exact* moments 1–6 would be worth
~130× — but a plug-in built from the **sample's own** moments has, by the delta method, variance
equal to the *projection's* variance (99.2%), not the residual's. Gain over the plain sample mean:
1.008×. This is the same trap as §7f's always-on collapse (bit-identical, 0.000e+00) and as the
single-hub Jacobian control variate measured this session (mean over a whitened batch = 8.6e-08,
float32 roundoff, against a signal of 1.03).

> **Any construction whose estimator is a smooth function of the same sample's own statistics
> re-derives the sample mean.**

## 7h.3 Cubature — §7g's exponent is wrong by ~900×, and it does not matter

§7g concluded `D ≈ 1.3e4` from `C(D) ∝ D^−0.574` fitted on D ∈ {1,3} and pushed four orders out.

**Model-free replacement — no inversion, no extrapolation.** With `R(ρ)/R(1) = Σ_d c_d ρ^d`,
`c_d ≥ 0`, a degree-D-exact rule leaving surviving mass S must satisfy `R(ρ) ≥ (1−S)ρ^D` for
*every* ρ, hence `D ≥ [ln(1−S) − ln R(ρ)]/ln(1/ρ)`. Measured two independent ways agreeing to 2%
(the larger run: 120 MLPs × 65 values of ρ, SE ≤ 5.6e-4 on R(ρ)/R(1)):

| ρ | 0.90 | 0.98 | 0.99 | 0.995 | 0.999 | 0.9999 |
|---|---|---|---|---|---|---|
| 1 − R(ρ)/R(1) | .4089 | .1682 | .1065 | .0650 | .01742 | .002133 |
| (1−R)/(1−ρ) → mean chaos degree | 4.09 | 8.41 | 10.65 | 12.99 | 17.42 | **21.33** |

The target needs S ≤ 0.29% of C_plain ⇒ **D ≥ 14.7**. Möller: degree 15 ⇒ m = 7 ⇒
k ≥ C(263,7) = **1.59e13 nodes** against ~1.5e5 affordable — short by **1.1e8×**. The true
requirement is D ≈ 15, not 1.3e4; the conclusion is unchanged because the node count explodes
either way.

**Degree 5 was built and measured, not argued.** Mysovskikh / Lu–Darmofal at n = 256 (66,307
nodes), verified exact to 1.835e-15 on every monomial of degree ≤ 5 and correctly failing at
degree 6:

| estimator | MSE | k | c | FOM |
|---|---|---|---|---|
| whitened+antithetic MC | 3.696e-07 | 66,306 | 2.065e6 | 5.06e4 |
| **degree-5 Mysovskikh** | 2.605e-05 | 66,306 | 1.802e6 | **3.11e6 — 70.5× worse, t = 15.9** |

**Why, and it is structural:** the degree-5 weights are forced negative,
`w₁r⁴ = −n²(n−7)/(2(n+1)²) < 0` for every n > 7, so effective sample size collapses to **941** of
66,306 nodes. Two further facts close the class: (i) a random rotation does **not** make the rule
unbiased — positive homogeneity makes the radial part deterministic, relative bias −2.910e-3,
removable only by rescaling with E‖x‖; (ii) the *best conceivable* degree-5 rule (equal weights,
ESS = node count) caps at **1.55×** (bootstrap 1.455 ± 0.028; an independent probe measured
S(5)/C_plain = 0.331 ± 0.029, i.e. ~1.4×), and degree 7 at 2.22×, against the **162× needed**.
Degree 7 also costs 2.86e6 nodes = **18.9× the entire budget** for one evaluation.

## 7h.4 The "d ≈ 252" claim survives a premise that is false by 203×

§7g's `VALUE(d) ≤ 1/(a₁(1−d/256))` assumes isotropic chaos. **It is not isotropic.** The degree-1
matrix G = E[f xᵀ] (Stein; validated against 600 exact Jacobians to 0.2%, Euler identity
J(x)x = f(x) to 6.2e-06) holds 79.1% of its energy in **one** direction and 97.2% in eight, against
d/256 = 3.1% — a 203× violation. So VALUE(8) = 1.34×, not 1.008×.

**And the hole is empty**, because whitening already annihilates degrees 1 and 2 exactly, in all
256 dimensions, for free. Measured C: plain 0.04728 → antithetic 0.04356 → whitened 0.02391 →
whitened+antithetic 0.02227, against 1/(2·0.2355) = 2.123× predicted from the Hermite-kernel
decomposition — three-digit agreement. The residual the baseline actually leaves is even-degree-≥4
chaos, whose ANOVA effective dimension measures **d ≳ 224**. A priced estimator on the top-d
subspace comes out at 0.87× (worse), t = −1.71. **The conclusion survives its own broken premise.**

## 7h.5 Multilevel is closed by an identity, not an inequality

`ReLU(z) = (z+|z|)/2` plus commutation of the sample mean with Wᵀ gives, for a bias-free net,

```
E[y_32] = Σ_{l=1..32} 2^{-(33-l)} (W_32ᵀ ··· W_{l+1}ᵀ) E|z_l|          (exact)
```

Every level is **exact** — no coarse surrogate, which is what defeated every previous attempt.
Level l needs only l layers, so the optimal allocation under cost Σ n_l·l gives
min MSE = (Σ_l √(w_l·l))²/(32k). The per-layer weights were independently re-measured here
(7.8%/layer at layers 3–4 falling to 1.9%/layer at layer 31 — reproducing §A5(a)'s 7.9% → 1.7% by
a completely different method), giving

> **multilevel / plain = 32·Σw_l / (Σ√(w_l·l))² = 0.082 — i.e. 12× WORSE than plain MC.**

Plain MC gets n_l = k at *every* level for the price of one pass because the same samples serve all
levels; multilevel pays l per sample per level. Arithmetic, not a bound.

## 7h.6 The cost axis — a claimed 1.30× that the verification round REFUTED

A probe reported the cost axis worth **1.30×** (best metered c = 2.032e6, Φ 1.409e-07) and
corrected `BUDGET_BOUND_REPORT`'s 1.25e6 floor as wrong by 1.72× because **flopscope charges
Strassen's additions at 1.0/element**. The metering is correct and was reproduced independently:

- `matmul((k,m)@(m,o))` = `k·o·(2m−1)` **exactly** (6 shapes, exact to the FLOP)
- naive 32-layer chain = **4,194,304**/sample exactly
- `einsum('ki,kj->ij', X, X)` gets an exact **1.992× symmetry discount**, bit-identical to `X.T@X`
  (max diff 0.000e+00)
- no dtype is cheaper than float32 — float16/int8/int16/int32/uint8 all meter at 2.0 FLOPs/MAC
- Strassen-Winograd in **block-tuple form** (never calling `concatenate`), 32-layer chain:

| depth | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| metered/sample | 4,194,304 | 3,686,912 | 3,257,728 | 2,908,064 | 2,647,384 | **2,498,522** | 2,506,926 | 2,756,930 |
| vs naive | 1.000× | 1.138× | 1.287× | 1.442× | 1.584× | **1.679×** | 1.673× | 1.521× |

reproducing the probe to 0.5%. float32 accuracy penalty is negligible (MSE 1.1e-13 at d=1 rising
to 2.5e-12 at d=5, against a 1e-08 requirement).

**But the probe counted instrumented FLOPs only.** The score bills
`effective_compute = flops_used + 1e11 · residual_wall_time_s`, and depth d costs 7^d Python-level
leaf calls. Measured `residual_wall_time_s` from the meter itself, one layer, chunk k = 65,536:

| depth | leaf calls | metered/sample | **residual**-FLOPs/sample | **TOTAL/sample** |
|---|---|---|---|---|
| 0 | 1 | 130,816 | 142 | 130,958 |
| 1 | 7 | 114,945 | 717 | 115,662 |
| **2** | 49 | 101,507 | 2,662 | **104,169 ← optimum** |
| 3 | 343 | 90,534 | 22,124 | 112,658 |
| 4 | 2,401 | 82,307 | 74,196 | 156,503 |
| 5 | 16,807 | 77,515 | 432,427 | 509,941 |

> **Depth 2 — which `work/mine/wmc4.py` already uses (`_LEVELS = 2`) — is the effective-compute
> optimum.** Depth 5 saves 24,000 metered FLOPs/sample and spends 430,000 residual FLOPs/sample:
> a 18:1 loss. Confirmed at chunk 8,192 as well (there the loss is 133:1).

**The cost axis is closed, and the live submission is already sitting on its optimum.** The file's
own docstring said so ("wall time is the binding constraint, not FLOPs"); this measures it. Nothing
here is shippable.

## 7h.7 Everything else, measured and dead

| route | best measured | verdict |
|---|---|---|
| exactly-integrable surrogates (layer-1 affine + quadratic + 1040 exact-mean controls) | 1.06× gross, **0.95× net** | closed |
| single-hub Jacobian control variate g = J(x₀)x | **exactly 0.000** | reduces to the sample mean |
| multi-hub piecewise linearisation | 256× more hubs buy 1.19×; 10²⁷¹–10³⁴⁴ hubs needed | closed |
| Jacobian region reuse | 45.9× worse, t = 12.6 | closed |
| low-rank surrogate MLMC | ≥1.29× worse at every r | closed |
| analytic anchoring of early layers | crossover at layer 9; ≤1.017×, **net negative** once the 6.56e9-FLOP analytic pass is metered | closed |
| GP surrogate, oracle marginals, best coupling over all couplings | ceiling 87–116×, best-case **FOM 349 — still above 272** | closed even as an oracle |

*Useful by-product, worth keeping:* the shipped arc-cosine kernel is the **zero-mean** case only,
and μ/σ reaches 4.3 by layer 32. The exact **non-central** bivariate ReLU second moment was derived
and verified (reduces to arc-cosine at m = 0 to 1.8e-15; matches 2e8-sample MC within its own SE).
Use the Drezner–Wesolowsky form of Φ₂ — the Owen-T form is 0/0 at h = k = 0, exactly the layer-1
case, and silently returns 0.5 instead of 1/4 + asin(r)/2π.

## 7h.8 Third-error audit of BUDGET_BOUND_REPORT

A third error of the same species was assumed to exist. Several were found. **None flips the
conclusion; all should be corrected before the Aug 17 write-up.**

| § | claim | stated | recomputed | load-bearing |
|---|---|---|---|---|
| A4.1 | LP D=5 surviving C | 4.364e-3 | **1.063e-2** (2.44×) | yes — the LP table is unsolvable from its own stated inputs |
| A4.1 | LP D=5 / D=32 max VR | 10.8× / 570× | **4.43× / 28.6×** | yes |
| A6 | minimax moments for the target | ~800 | **~201** — the 0.1 multiplier is applied zero times | no (still astronomical) |
| A6 | cheapest K=3 cumulant cost | 2.529 B | **1.676 B** (the companion paper's own fitted polynomials) | yes — margin shrinks from 2.5× to **1.18×** |
| A6 | K=2 cost | 0.0079 B | 0.0140 B | no |
| A8 | 2.4× arbitrage on the "Route-1 floor" | 8.6e-09 | 3.23e-10 | yes — A8 is computed from the 2.07e-8 that A4.3 itself retracts |
| §7g | exactness degree needed | D ≈ 1.3e4 | **D ≈ 15** (≈900×) | yes — conclusion unchanged, see 7h.3 |
| — | this session's own cost probe | 1.30× available | **1.00×** — residual wall time ignored | yes |

Möller node counts all check exactly (C(257,1) = 257, C(258,2) = 33,153, C(259,3) = 2,862,209,
C(260,4) = 186,043,585), as does "m = 3 is 26× over budget" (26.10×).

The K = 3 cumulant margin of 1.18× is the closest anything came to being affordable. It does not
matter: over-budget zeroes the prediction, and even at order 3 the *oracle-fed* reconstruction is
1.61e-07 — sixteen times the 1e-08 requirement.

## 7h.9 The leaderboard from outside (public pages, 2026-08-07)

| rank | entrant | adjusted | entries | final MSE |
|---|---|---|---|---|
| 1 | dpskv5 | 4.00e-10 | 285 | 3.6e-09 |
| 2 | joe_wanza | 1.00e-09 | 994 | 4.0e-09 |
| 3 | huang_chung_yi | 4.40e-09 | 469 | 2.99e-08 |
| **4** | **ednacob** | **4.62e-08** | **115** | 9.11e-08 |
| 5–15 | … | 5.81e-08 → 1.08e-07 | | |
| 54 | **us** | 1.834e-07 | | |

Twelve independent teams are piled against a wall at 4.6e-08 – 1.1e-07; then a **10.5× empty gap**;
then three entries alone. Every measurement in §7h says the constructible frontier is exactly that
band. In a field of 646 with ~14,000 submissions, an algorithmic frontier produces a continuum, not
a cliff with three points past it. **Note we are 4.0× off the frontier itself — that gap is real,
and is where any remaining Phase-1 effort should go.**

Recorded as structure, not as a claim about anyone. Forum topic 18099 (user Kerensa) documents a
`flopscope.numpy` path on which work "report[s] zero instrumented FLOPs", and jtel quotes
instrumented shares "below 0.001" for top submissions against ~0.93 for ordinary ones; topic 18108
is a participant recommendation to neutralise the wall-time channel. No organiser reply appears in
either thread. Against that, our own check of the leading entries' evaluation data did **not**
support the forum claims. **The picture is mixed, cannot be resolved from outside, and we publish
no per-entrant figures.** Rules §5.4/§5.5
already settle it internally: the withheld 50 and the fresh Private Re-evaluation suite.

## 7h.10 The statement that replaces §7g's two-annihilator enumeration

Every variance annihilator needs a function with an **exactly known Gaussian expectation** that is
**cheap to evaluate**. That is the entire class:

| class | known mean? | cost | measured value |
|---|---|---|---|
| polynomials ≤ D, as node exactness | yes | C(256+m,m) nodes | D=5 affordable → ≤1.55× ceiling, and its only construction is 70× worse; D ≥ 15 needed |
| polynomials ≤ D, as explicit control variate | yes | C(256+D,D) FLOPs/sample; D=4 is 44× a forward pass | priced out |
| functions of a d-dim projection | yes (d-dim quadrature) | cheap | residual chaos has effective dimension d ≳ 224 |
| functionals of layer 1 (ReLU mean, arc-cosine covariance) | yes, closed form | free | **1.09×** |
| OU-filter control variate Σ aⱼ T_{ρⱼ}f with Σ aⱼ = 0 | **yes, exactly 0** | nested MC; one inner draw caps at 2× for 3 forward passes | net loss |
| hidden-layer means/covariances, or any surrogate built from f elsewhere | **no** | — | worth up to 394×, and obtaining the mean *is* the problem |

> Closed-form Gaussian integration reaches exactly **one ReLU layer** — ridge functions h(vᵀx),
> *pairs* of ReLU ridges via the arc-cosine kernel, and polynomials. Three-way ReLU products need
> trivariate orthant probabilities, which have no closed form. Measured value of an exact layer
> mean: **L1 = 1.09×, L16 = 2.98×, L31 = 67.3×.**
>
> **Exact means exist only where they are worth 1.09×, and are worth 67× only where they do not
> exist.** Availability and value sit at opposite ends of the network. That, and not a node count,
> is why 1e-09 is out of reach.

## 7h.11 What this changes for the write-up

1. §7g's conclusion is **kept**; its three supporting arguments are **replaced** by 7h.3, 7h.4 and
   7h.10, all of which are measured rather than extrapolated.
2. The corrections in 7h.8 must be applied to `BUDGET_BOUND_REPORT.md` before anything is
   published from it. Two of its numbers (A4.1's LP table, A8's arbitrage line) do not survive.
3. The strongest genuinely new results, in order: the **exact multilevel identity** and its 12×
   negative (7h.5); the **negative-weight obstruction** to Möller-size degree-5 cubature (7h.3);
   the **max-entropy reconstruction reaching 5.08e-09 from 6 exact moments**, which relocates the
   deterministic blocker entirely onto the order-1 moment (7h.1); and the **availability-vs-value
   antisymmetry** (7h.10).
4. **No submission was produced.** Nothing measured this session beats `work/mine/wmc4.py`, so the
   harness was not re-run and the Phase-1 nomination pair is unchanged.

## 7h.12 Post-crash score push (2026-08-07 evening) — two negatives, two ready submissions

**Sobol/QMC input draw: dead, measured.** Pure-numpy Joe-Kuo Sobol (validated: exact base-2
stratification, ppf round-trips to 6 decimals), digitally shifted, antithetic, whitened —
paired against the pseudorandom baseline on 120 full-split MLPs: **0.94× at k=4096 (t=−1.03),
1.018× at k=32768 (t=+0.23)**. Exactly what the chaos spectrum predicts (residual variance at
effective dimension ≳224). Closes SOX's one unreplicated leg: their edge is not Sobol, and
part of the public 1.18× gap to them is 50-MLP leaderboard noise (our own full-split wmc4
number is 2.24e-07 against their public 1.551e-07).

**Orthoplex sampler: variance-neutral, cost-marginal, and one real trap found.** At node radius
√n the estimator has an O(1) radial bias (f is homogeneous; E‖x‖ ≠ √n): bias² ≈ 6e-7,
k-independent, measured 0.80×/0.61× at k=4k/16k. With the exact correction E‖x‖/√n =
0.99902392 (lgamma) it is statistically identical to whitening (0.978×, t=−0.46; 0.996×,
t=−0.05), replicating §7g. The layer-1-collapse variant (one shared rotation, per-stack
sign/permutation randomisation) has a **fixed-frame variance floor** — C grows ∝k (measured
0.185→0.740 from k=4k→16k, t=−15) — so independent rotations are mandatory and the net win
reverts to ~1.02× with integration risk. Not shipped.

**Found un-submitted, harness-verified work from the parallel session (`work/final/`):**
- `sweep.py` — wmc4 + four exact cost identities (antithetic layer-0 mirror 2.1%, symmetric
  einsum Gram 1.3%, lead-pass Strassen 0.3%, Cholesky whitener 0.09%). Official harness,
  200 MLPs: **2.11e-07 vs wmc4's 2.31e-07 (1.095×)**, 0/200 failed, binding adversarial case
  improved 0.9170 → 0.9040. Its Strassen-depth-3 negative (wall-time, 1.010× at t=+0.42,
  sign-flips under contention) independently matches §7h.6.
- `anchor-cheap.py` — wmc4 + exact layer-1 full-covariance anchor folded into W₁. Paired,
  n=1000, real operating point k=64,512: **1.083× (t=+5.45)**, with the k-dependence that
  explains every earlier "neutral" reading (the effect flips sign below k≈6,000: 0.825× at
  k=5,900). Harness 200: 2.14e-07 vs 2.24e-07.

Both validated and packaged: `submissions/2026-08-07-sweep-v7.tar.gz`,
`submissions/2026-08-07-anchor-v8.tar.gz`. **Mechanisms are orthogonal (cost × variance);
if both grade, the default top-2 nomination becomes a genuine two-mechanism hedge.
Next: merge them (expected ~1.18× over wmc4, public ≈1.55e-07) behind the full adversarial
suite before any submission of the merge.**

## 7h.13 The circularity, attacked directly and closed (2026-08-08)

Per Bong's directive the max-entropy door was attacked head-on. Three measurements, sequential.

**1. The 5.08e-09 claim is real — replicated to three digits.** The probe's pipeline was
reconstructed from its transcript (its scratchpad files were lost in the crash) and re-run from
fresh moment caches: Hermite-tilted Gaussian exponential family, dual Newton with active-set
shrinking, bias/noise separated via per-stream `est − empirical-mean` residuals (the design that
makes 5e-09 resolvable at all — recon and sample mean share samples, cancelling MC noise).

| M (moments) | bias², K=250k | bias², K=1e6 | K-stable? |
|---|---|---|---|
| 4 | 5.259e-08 | 5.228e-08 | ✓ |
| **6** | **5.077e-09 ± 1.3e-10** | **5.096e-09 ± 1.3e-10** | ✓ (ratio 1.004) |
| **8** | **8.148e-10 ± 2.3e-11** | 8.586e-10 ± 2.3e-11 | ✓ (ratio 1.05) |
| 10 | 1.70e-10 | (noise-limited) | at resolution floor |

M=6 is below the 1e-08 requirement for a 1e-09 score; M=8 is below even the 4e-09 raw
requirement for *matching the leader* at the 0.1 floor. **The reconstruction problem is solved,
with zero fitted parameters.**

**2. The plug-in is the sample mean — measured, not argued.** At the real operating point
(k = 65,536), paired on 48 MLPs against the 1e9-sample truth, versus the plain ReLU sample mean
of the *same* samples (V0 = 9.977e-07):

| variant | moments from | MSE | ratio V0/V | t |
|---|---|---|---|---|
| V1 | all from the same sample (honest, constructible) | 1.0022e-06 | **0.9954** | −5.80 |
| V2 | shape c₃..c₆ from a free 2e6-sample reference | 1.0025e-06 | **0.9951** | −5.11 |
| V3 | σ and shape free; only m₁ from the sample | 9.959e-07 | **1.0018** | +1.26 (n.s.) |

The honest plug-in is slightly *worse* than the sample mean (the delta-method nonlinearity
penalty, −0.46%, significant). Handing the estimator every moment of order ≥ 2 for free (V3)
leaves it statistically indistinguishable from the sample mean. **The reconstruction consumes
exactly the information the sample mean already provides.**

**3. Therefore the door opens into the same room.** The maxent result relocates the entire
deterministic blockage onto m₁ = wᵢ·μ₃₁ (§7h.1); the plug-in measurement shows the relocation is
total: with everything else free, the estimator IS the sample mean of m₁, and m₁ needs 554× less
mean-squared error than that sample mean provides (§7h.1). Every constructible route to a better
m₁ was separately measured dead (upstream anchoring cascades nothing; multilevel is 12× worse by
identity; analytic propagation is 83× short in accuracy; K≥3 transport is 1.18–1.68× over budget
and 16× short in accuracy even oracle-fed).

> **Closure statement for the write-up.** The deterministic branch of WhestBench decomposes as
> (reconstruction from marginal moments) ∘ (acquisition of those moments). The first factor is
> solved — 5.1e-09 from six exact moments, parameter-free, replicated. The second factor is the
> original problem: acquiring m₁ of the final layer to the required accuracy *is* estimating
> E[y₃₂], one linear map removed. A solved outer problem composed with the original inner problem
> is the original problem. No door remains on this branch; what remains is the constructibility
> frontier at ~1.5e-07, which the corrected §A8 shows is consistent with the entire visible
> leaderboard band at ranks 4–15 once the metric's bounded wall-time conversion (1.7–2.4×) is
> priced in.

### 7h.9-addendum — **[REDACTED IN THE PUBLIC RELEASE]**

> This entry connected a named participant's leaderboard movement to a compute signature we had
> inferred. Whatever hedge it carried, it was an insinuation about an identifiable person, resting
> on evidence we had already found unreliable, and it should not have been written. The operative conclusion needs none of it:
> **a public score is a 50-MLP statistic and is not finely interpretable** (§7b, and §8.2 of
> the write-up), which is why every decision here rests on paired offline measurement at
> n = 1000 instead.

