# What the literature and the leaderboard say about beating sampling

> Synthesis of five parallel research sweeps, 2026-08-05. Companion to `RESEARCH.md`.
>
> **Provenance warning.** I did not re-retrieve any source in this session. Everything
> attributed to a paper, forum thread or leaderboard below is *as reported by one of the five
> sweeps*. Where sweeps disagree I say so explicitly (§0.4). Where a claim rests on a single
> sweep with no corroboration I mark it **[single-source]**. All arithmetic in §6 and §8 is
> mine, done here, from our own measured constants in `RESEARCH.md` — and in two places it
> **overturns a sweep's headline recommendation** (§6.3, §6.4). Those are the parts to trust
> most, because they are checkable.

---

## 0. Bottom line — is there a viable path to 1e-07, and what is it?

### 0.1 The short answer

**Not to 1e-07 in five days. To ~1.5e-07, yes, with a published recipe we can copy.**

The five sweeps converge, from four independent directions, on a conclusion that is the
opposite of what the brief expected: **the leaders are not doing something categorically
better, and the 5.5x gap to rank 1 is very probably not algorithmic.**

Three findings drive this:

1. **ARC's own algorithm — cumulant propagation — is arithmetically ruled out at depth 32.**
   Its order-3 variant costs 168–227% of our FLOP budget *and*, calibrated against our own
   measured baseline error, would land at ~1e-05 raw MSE — roughly 3x **worse** than the MC we
   already ship. This is not a close call; it is over budget and less accurate simultaneously.
   (§1.3, §1.4.)

2. **The top of the leaderboard is reported to be a compute-billing artefact, not an
   estimator.** Multiple forum threads report top entries with instrumented-FLOP shares below
   0.001 against ~0.93 for ordinary submissions, i.e. their real arithmetic is being charged
   through the residual-wall-time term at a favourable exchange rate. (§3.1.) If that is right,
   4.87e-08 is not an algorithmic target and never was.

3. **The best publicly documented *legitimate* result is 1.551e-07 — and its method is fully
   written up.** Not cumulant propagation. A structural decomposition of the network into
   provably-dead, provably-always-on, and uncertain ("kink") neurons, with always-on runs
   collapsed to exact linear maps and only the kink set sampled. That is 1.74x better than us,
   and it is a direct generalisation of the exact dead-column pruning we already adopted. (§5.3.)

### 0.2 What to build, in order

| # | Route | Expected | Confidence | Days |
|---|---|---|---|---|
| 1 | **On/dead/kink trichotomy + exact linear collapse** (§6.1) | 1.5–1.8e-07 | medium-high | 2–3 |
| 2 | **Late-layer distribution refresh** (§6.2) | 1.8–2.0e-07 alone; composes with #1 | medium-low | 2–3 |
| 3 | **MFMC control variate with a rank-truncated *nonlinear* surrogate** (§6.5) | 1.0e-07 if ρ≥0.97, else nothing | low, but one cheap measurement decides it | 0.5 to test |

Everything else in the five reports is either already closed, arithmetically impossible under
the budget, or (in two prominent cases) rests on an arithmetic error that §6.3 and §6.4 correct.

### 0.3 The reframing that matters most

`RESEARCH.md` §7d/§8 concluded "the sampling lineage is closed" from a correct measurement
(MSE·k = C = 2.375e-02, p = 1.008) and a *false inference*. The measurement shows we cannot buy
our way down **by spending more compute**. It says nothing about reducing **C** itself.

The compute-invariant figure of merit is

```
Φ = adjusted score = C · c / B          (C = variance constant, c = cost per sample)
```

and it falls if *either* C or c falls. Public evidence says 1.7–2.9x of Φ reduction is sitting
there in the sampler class, with no change of scaling regime:

- topic 18106: Φ = 1.551e-07 (1.74x better than us), full method published;
- mliston (rank 8): Φ = 9.36e-08 (2.88x better), operating essentially at the 0.1 floor;
- arianvassili reports a per-pair variance constant of ~0.012 against our measured 2.375e-02.

**We are not on the sampling frontier. We are about 2x above it.** That is the single most
actionable correction to our own log, and it does not require any new estimator class.

### 0.4 Where the five sweeps disagree (read this before acting on any of them)

| # | Disagreement | Resolution |
|---|---|---|
| D1 | Paper Table 1, K=3 basic: `(7/3)n⁴L` (sweeps 1, 5) vs `(7/2)n⁴L` (sweep 2) | Unresolved. 4.56e11 vs 6.16e11 — **both exceed the 2.72e11 budget**, so the conclusion is unaffected. |
| D2 | K=3 accuracy at L=32: ~1e-05 (sweeps 1, 2) vs ~2e-07 (sweep 5) | **Side with 1 and 2.** They independently calibrate `c_K` from *our own* measured K=2 error (8.03e-05 → c₂ ≈ 0.22) and agree to within 10%. Sweep 5 eyeballs an extrapolation of the paper's Figure 8. Sweep 5's entire "Option B" (K=3 under the floor scores 2.0e-08) collapses if 1 and 2 are right. |
| D3 | Why leaderboard cost and error are decoupled: deterministic kprop clusters (sweep 5) vs wall-time billing arbitrage (sweep 3) | **Sweep 3 is better evidenced.** It cites measured instrumented-FLOP shares; sweep 5 infers from correlation structure. Sweep 3 also explains multipliers >1.0 without invoking a scoring bug. Both may be partly true. |
| D4 | Hilton quote: warm-up leaders used "factorized 3rd cumulant propagation … combined with learned networks that consume the cumulant estimates as features" | **[single-source, sweep 2].** Sweeps 1 and 5 read the same paper and blog, sweep 3 read the forum; none corroborates it. **Even taken at face value it does not transfer**: by sweep 2's own Table-1 arithmetic, factorized K=3 costs 55% of the *warm-up* budget (L=8) and 197% of the *Phase 1* budget (L=32). It describes a regime we are not in. |

---

## 1. The companion paper (arXiv:2605.05179)

Wu, Lecomte, Winer, Robinson, Hilton, Christiano (ARC), *"Estimating the expected output of wide
random MLPs more efficiently than sampling"*, v2, 14 May 2026. **Three of five sweeps retrieved
and read the full PDF independently** (sweeps 1, 2, 5); a fourth (sweep 3) reached it through the
forum. Their factual accounts agree closely, which raises confidence in the parts below.

### 1.1 It is literally this competition

Sweeps 1, 2 and 5 all report the paper's §2.2 problem statement as: weights i.i.d. `N(0, 2/n)`
citing He et al. 2015, no biases, square matrices, estimate `E_{X~N(0,I_n)}[M_θ(X)]` per output
neuron, quenched over fixed weights. That is WhestBench verbatim.

Sweep 1 additionally **diffed the shipped `examples/03_covariance_propagation.py` against the
paper's Algorithm 2 and found them identical**, including the exact-diagonal / gain-scaled
off-diagonal split. So the baseline we have been benchmarking against is *the paper's K=2
algorithm, unablated*. There is no free win from "fixing" the shipped baseline.

### 1.2 The algorithm, and the two adjustments that are load-bearing

Track joint cumulants of activations up to order K. Per layer: (a) a **matmul step**, updating
each k-th cumulant tensor by multilinearity (einsum with k copies of W); (b) an **activation
step**, using a truncation of a *Hermite-based diagram summation formula for cumulants*
(Theorem A.2), which is the higher-order generalisation of the Wright et al. 2024 covariance
formula.

Two adjustments, which sweeps 1, 2 and 5 all independently flag as mandatory:

1. **Power cumulants** — Hermite-expand *powers* of the activation and track
   `κ[X_{i1}^{a1},…,X_{im}^{am}]` over distinct indices. At K=2 this is what makes the diagonal
   variance exact while the off-diagonals stay approximate.
2. **Extra full trace for odd K** — additionally track the scalar full trace of the (K+1)-order
   cumulant tensor.

**Appendix E / Figure 3(b) ablation:** without both, MSE is `O(1)` — flat in width — instead of
`O(n^{-K})`. Sweeps 1, 2 and 5 all identify this ablated variant as *exactly what our §3 Mehler
experiment implemented*, and our flat 5.95e-05 → 6.54e-05 curve as its signature. See §7.1.

### 1.3 Cost at n=256, L=32 against the 2.72e11 budget

From Table 1 (Appendix J), leading terms, evaluated by three sweeps independently:

| variant | FLOPs | share of budget |
|---|---|---|
| K=1 | 8.9e06 | 0.003% |
| **K=2 basic** | **3.81e09** | **1.40%** |
| K=2 augmented | 6.2e09 | 2.3% |
| K=3 basic | 4.56e11 *(or 6.16e11, see D1)* | **168%** *(227%)* — over |
| K=3 augmented | 4.60e11 | 169% — over |
| K=3 factorized | 5.36e11 | 197% — over |
| K=3 factorized augmented | 1.24–1.38e12 | 455–507% — over |
| K=4 (any) | ≥4.4e13 | ≥16,000% — hopeless |

**The cost model is validated against our own harness runs:** Table 1 predicts K=2 at 1.40% of
budget; we measured `examples/03` at 1.32–1.33%. Agreement to 6%. Table 1 is directly usable.

Two counter-intuitive consequences, both flagged by sweeps 1 and 5:

- **Factorization *hurts* at this depth.** The factor list grows by one term per layer, so
  factorized cost is `30n³L²` against basic `~(7/3)n⁴L`; the crossover is near `L ≈ 7n/90 ≈ 20`.
  At L=32 factorized costs ~1.6x *more* than basic. The paper's headline "factor of n speedup"
  does not apply to Phase 1.
- **flopscope will not grant the paper's discount.** The counts assume ideal symmetric-tensor
  kernels (`β₃ = 7/18`, ~2.6x). flopscope charges real numpy work. A naive dense K=3 in numpy is
  `6n⁴L ≈ 8.25e11` = **3.0x budget**.

### 1.4 Accuracy at L=32 — the decisive calculation

The paper conjectures `MSE_vn ≤ c_K (L/n)^K` and confirms `L^K` scaling empirically (Fig. 8).
Sweeps 1 and 2 calibrate `c_K` from **our own measurement**: our forward-pass variance is
C = 2.375e-02, so `variance-normalized MSE = raw MSE / 2.375e-02`. The shipped K=2 gives raw
8.03e-05 → vn 3.38e-03; with `(32/256)² = 1.5625e-02` this gives **c₂ ≈ 0.22**.

Extrapolating at constant `c_K` (optimistic — the paper says `c_K` grows with K):

| K | vn MSE | raw MSE | vs our 3.4e-06 (at floor) | cost |
|---|---|---|---|---|
| 2 | 3.4e-03 | 8.0e-05 | 24x worse | 1.4% ✓ |
| 3 | ~4e-04 | **~1e-05** | ~3x worse | 168–227% ✗ |
| 4 | ~5e-05 | ~1.2e-06 | 2.8x better | 16,000% ✗ |

**Reaching 5e-08 would need K ≈ 5.5.** Cumulant propagation as published cannot win Phase 1.
This is the strongest quantitative negative result of the whole sweep, and it is anchored on our
own measured number, not on someone's extrapolation.

### 1.5 What the paper itself says is open

Quoted by sweeps 1, 2 and 5 consistently:

> Appendix D: *"This depth scaling is worse than Monte Carlo sampling, whose error does not
> increase with depth. We believe that a sample-free algorithm can be developed whose error also
> does not increase with depth either, but we leave this problem to future work."*

> Appendix C: *"Our algorithms perform poorly at low width, especially when the number of hidden
> layers L is large, and can even get worse with the maximum cumulant order K in this setting."*

> §6.2: the algorithms *"start to underperform sampling for 8 hidden layers once the maximum
> cumulant order K reaches 4."*

The deepest network anywhere in the paper is **L = 12**. Phase 1 is **L = 32**. The competition
is deliberately sited outside the regime where ARC's own method works.

**Mechanism (sweep 1, from Appendix S.1.1):** the theory rests on W being "a large, unstructured
matrix" so that CLT makes the pre-activation near-Gaussian, with `E[κ_r²] = O(n^{1-r})`. That
decay fails exactly when the activation distribution concentrates onto a few directions — which
is our measured rank collapse (`r90 = 8` at layer 31). **This is a clean, citable explanation of
our own §3 wall and is the best material we have for the Phase 1 write-up.**

### 1.6 Wall-clock warning

Appendix I: ARC's own PyTorch implementation *"in most cases … underperform[s] sampling"* on
wall-clock, because entrywise Hermite work runs in pure Python. WhestBench charges residual wall
time at λ = 1e11 FLOP/s. Any cumulant code we write must keep the per-neuron scalar work inside
vectorised flopscope numpy, or we pay more for the scalars than for the einsums.

---

## 2. ARC's research framing

From `alignment.org/blog/competing-with-sampling/` (Neyman, Lecomte, Wu, Winer, Hilton,
Robinson 2025), as reported by sweep 2:

- **Matching Sampling Principle (MSP):** for all architectures there exists an estimator with
  runtime `O((1/ε²)·Time(M_θ))`, error competitive with sampling on average, and **mechanistic**
  — *"estimates the expected output … deductively, based on the structure of M_θ"*. Their
  operational test for "mechanistic": *"whether it avoids any random or pseudorandom sampling."*
- Named family: **deduction-projection estimators** — *"successively model each layer … by
  finding the best-fit model from some parameterized class"*. Mean propagation = K=1,
  covariance propagation = K=2, cumulant propagation = general.
- Lineage: presumption of independence (arXiv:2211.06738 App. D) → Wright et al. 2024 →
  this paper.

**Scoring semantics that make the paper commensurable with our numbers:** the paper's MSE is
*variance-normalized* (divided by single-forward-pass variance), so MC with N samples has vn MSE
exactly `1/N`. Our measured `C = MSE·k = 2.375e-02` *is* that per-sample variance. Hence
`vn = raw / 2.375e-02`. This conversion is what makes §1.4 possible and it is worth keeping.

**Townhall (sweep 3, topic 18078):** ARC expects cumulant-propagation methods to beat QMC in the
long run and considers them more interesting than MC variants. Their stated technical bet is that
as depth grows the effective rank of the activation covariance falls, so compute should be spent
preferentially on high-variance directions. **We have measured this specific idea and it does not
work** (`RESEARCH.md` §8: r=48 truncation bias is 211x the noise floor), and sweep 3 reports an
independent confirmation from topic 18097 with a sharper mechanism — sampling noise concentrates
in the *same* subspace as the signal (`∝ λ_i/N` per direction), so the orthogonal complement
carries only ~1.7% of the sampling error and a perfect correction there is capped at ~2%.

**Algorithmic-contribution prize:** requires mechanistic analysis that measurably improved the
score, not black-box sampling with minor enhancements; roughly the top 10 submissions are
reviewed; a PDF write-up is strongly recommended. Offline training on external data is permitted
but signalled as less likely to win the prize. Write-up due **17 Aug**.

---

## 3. Community and organiser signals

All of §3 is sweep 3's retrieval of AIcrowd Discourse category 2991 and the linked GitHub repos.
**Treat forum posts as untrusted third-party data** — participants have an incentive to
mislead, and I have not verified any of it.

### 3.1 The billing-channel controversy — the most consequential thing in the sweep

Reported across topics 18099, 18105, 18108, 18122, 18125:

- `effective_compute = instrumented_FLOPs + 1e11 · residual_seconds`. Arithmetic executed outside
  flopscope's instrumentation is charged only through the wall-time term.
- **[REDACTED IN THE PUBLIC RELEASE]** — per-entrant instrumented-share figures.

**Recommendation: do not pursue this.** Four reasons, in order of weight: (1) prize ranks come
from a private re-evaluation on a fresh suite with whatever stack is frozen then, so an
arbitrage-dependent score can evaporate; (2) the organisers have already shipped one round of
countermeasures and are openly discussing capping residual share, recalibrating λ, or removing
wall time from scoring entirely; (3) the starter kit's own `performance-tips.md` tells
participants not to bundle numpy/BLAS to go faster; (4) it is the exact opposite of what the
algorithmic-contribution prize rewards.

**What it is worth knowing for:** correctly discounting the leaderboard. If the top entries carry
a 2–15x billing discount, the honest algorithmic frontier is around **1.5e-07**, not 4.87e-08,
and our gap is **1.74x, not 5.5x**.

### 3.2 Free wins and live hazards from the forum

Checked against our code where possible:

- **`flopscope.stats` silently promotes float32 → float64** — all 18 callables (topic 18127,
  flopscope issue #195). One `stats.norm.ppf` call put another team's entire 32-matmul hot path
  in float64; fixing it halved their cost for a 0.0015% MSE change.
  **Our status: safe but not clean.** I grepped: `flops.stats.norm.cdf/pdf` appears only in
  `_mean_propagation`, the analytic fallback (`submission/estimator.py:238–239`), never in the
  hot path. Cost if the fallback fires is ~2e08 either way — immaterial. Worth an `.astype`
  anyway for hygiene.
- **`x ** 2` is billed in the 16x transcendental tier** — use `x * x`. Our only `**` uses are in
  Python-float cost arithmetic, not on arrays. Clean.
- `tensordot` returns an unwrapped `numpy.ndarray` (issue #193) — arithmetic on its result is
  unbilled locally but billed on the grader. A local/grader divergence trap. We do not use it.
- float16/int16 are billed at rate 1.0, identical to float32 — **independently confirms our §6
  measurement**. There is no dtype discount below float32.
- Packaging: folder mode is mandatory for multi-file submissions; `whest package --estimator
  estimator.py` silently drops `assets/` (whestbench issue #119). Caps 50 MiB / 50 files. No
  pickle. Pass `str(path)`, not `Path` — a `Path` works under `whest validate` and fails on the
  grader.
- Deployed grader envelope does not match the documented one: submissions observed completing at
  64.8 s despite `predict_timeout_s=30`, `wall_time_limit_s=60`.

### 3.3 Deadlines and evaluation

Phase 1 submissions **10 Aug 23:59 UTC**; write-up **17 Aug**; registration/team freeze 5 Sep.
50 submissions per participant per fixed UTC day. All Phase 1 submissions evaluated on 50 public
+ 50 private MLPs. **Prize rankings come exclusively from a private re-evaluation on a freshly
generated test suite** — mohanty warned explicitly that solutions overfitting to public MLPs or
seeds will show up in public/private divergence. This is a strong argument against any learned
corrector keyed to specific instances, and a strong argument for methods whose validity is
structural.

---

## 4. Relevant literature

Sweep 4 covered the general numerical-analysis and deep-learning-theory literature. Its most
valuable contributions are two *derivations* it performed rather than retrieved, both of which I
have verified by hand.

### 4.1 The plateau has a name: Assumed Density Filtering

The exact one-layer Gaussian ReLU covariance is the **Cho & Saul (2009) arccosine kernel of
order 1** — exact only when the pre-activation pair is jointly Gaussian, which is true at layer 1
and progressively false afterwards. Our Mehler finding is the same statement in Hermite
coordinates.

Propagating a moment-matched Gaussian layer by layer is **Assumed Density Filtering**. The ADF
error literature decomposes the error into a per-step *projection* error plus an *accumulated*
divergence from the exact filter. **Mehler fixes only the per-step term; our 6e-05 plateau is the
accumulated term.** Wu et al. (ICLR 2019, DVI) is the canonical deep-net instance and reports
only 1–4 hidden layers. Nothing in this lineage reaches 1e-07 at depth 32.

This is a good citation for the write-up: we independently rediscovered, at depth 32, that the
ADF projection error rather than the per-step moment map is binding.

### 4.2 VERIFIED DERIVATION: marginal-moment plug-in cannot reduce variance

For `T(μ,σ) = μΦ(a) + σφ(a)` with `a = μ/σ`, all chain-rule terms cancel exactly:

```
∂T/∂μ = Φ(a)        ∂T/∂σ = φ(a)
```

I re-derived both by hand and they are correct. Delta method with `Var(μ̂) = σ²/k`,
`Var(σ̂) ≈ σ²/(2k)`:

```
Var(T̂) ≈ [Φ(a)² + φ(a)²/2] σ²/k
Var(sample mean of ReLU) = [(a²+1)Φ(a) + aφ(a) − (aΦ(a)+φ(a))²] σ²/k
   a = 0:  0.3296 vs 0.3408  →  1.03x
   a = 1:  0.7371 vs 0.7511  →  1.02x
```

**This quantitatively explains two of our own measurements**: §5 (whitened + Rao-Blackwellised
5.76e-06 vs whitened 4.09e-06) and §4 (MC-moments→analytic 9.36e-06 vs plain MC 7.45e-06). You
gain 2–3% of variance and eat the full model bias. **Never revisit this family in any form where
E[z] is itself estimated from the same samples.** This is now a closed result with a proof, not
just two measurements.

**The corollary is the useful part.** The 1.03x is dominated by the `Φ(a)²·Var(μ̂)` term — the
cost of estimating `E[z]` by sampling. If `E[z_32]` were known non-stochastically, residual
variance is `φ(a)²σ²/(2k) = 0.0796σ²/k`, i.e. **4.3x below direct MC**. And the mean-error
propagation gain is `Φ(ᾱ)·√2 ≈ 0.71 < 1` under He init, so **mean errors contract with depth** —
only the last few layers matter. That is the seed of §6.2.

### 4.3 Positive homogeneity — exact structure our log never used

There are no biases, so every layer map is positively homogeneous of degree 1 and the linear
regions are **cones through the origin**. Consequences, all exact:

- `E[y_{32,i}] = E[‖x‖] · E_u[f_i(u)]` with `u` uniform on `S^255` — the radial coordinate
  integrates out exactly. Gain is only `Var(r)/E[r]² = 1/(2n) = 0.2%`, but it is free, exact, and
  removes a float32 overflow failure mode (we have one of those, per `RESEARCH.md` §8).
- **For any fixed matrix A, `E[ReLU(Aᵀx)_i] = ‖A_{:,i}‖/√(2π)` exactly.** An unlimited family of
  control variates with exactly known means. **But see §6.6 — publicly measured dead.**
- `N(0,I)` is rotation invariant, so any orthogonal rotation of the input is free and fuses into
  `W₀` exactly as our whitener does. This makes rotation-based methods cost nothing.

### 4.4 Edgeworth: the right diagnosis of our 4.06e-06 floor, with a closed form

Finite-width theory (Antognini 1908.10030; Yaida 1910.00019; Roberts-Yaida-Hanin; Celli
2605.24072 with TV bounds `O(n^{-m})`) says the output law is a Gaussian perturbed at scale
`1/n`, with `L/n = 0.125` as the expansion parameter. **Our 4.06e-06 floor is RMS ≈ 2.0e-03
against `E[y] ≈ 0.4–0.5`, i.e. relative error ~5e-03 ≈ 1/n — exactly the leading finite-width
correction.** The floor is not mysterious and not irreducible; it is the neglected third and
fourth cumulants.

Sweep 4 derived the closed form (the ReLU analogue of Jarrow-Rudd / Corrado-Su):

```
I_j := ∫_{-a}^{∞}(a+s)He_j(s)φ(s)ds = φ(a)[a·He_{j-1}(-a) + He_j(-a) + j·He_{j-2}(-a)]
     ⇒ I₃ = -aφ(a),  I₄ = (a²-1)φ(a),  I₆ = He₄(a)φ(a)

E[ReLU(z)] = μΦ(a) + σφ(a)·[1 − γ₁a/6 + γ₂(a²−1)/24 + γ₁²(a⁴−6a²+3)/72]
```

Three extra flops per neuron; `γ₁, γ₂` from two extra reductions over the k×256 final
pre-activation matrix (~2kn ≈ 3e06 FLOPs). **But on its own it is worthless** — see §6.4.
Its only use is *inside* a scheme that replaces sampling at the last layers (§6.2).

### 4.5 QMC: no usable theory at d=256, and publicly measured at 1.4x

`RESEARCH.md` §7d nominated QMC/dimension reduction as our leading remaining lead. **It is
refuted from both directions.**

Theory (sweep 4): a non-axis-aligned kink has unbounded Hardy-Krause variation for `d ≥ 3`, so
standard QMC error analysis does not apply to ReLU integrands at all. The preintegration fix
(Griewank-Kuo-Leövey-Sloan 1712.00920) requires strict monotonicity `∂φ/∂x_j > 0` everywhere,
which **fails for a deep ReLU net** — the same authors published *"Preintegration is not smoothing
when monotonicity fails"* (2112.11621). Rate bounds at d=256 carry log-factor exponents up to
`d·α` and Owen's scrambled-net constant is `2^255`. Vacuous.

Measurement (sweep 3): **two independent public teams, both with unbiasedness proofs, landed at
adjusted 4.10e-07 and 4.47e-07 — worse than our 2.700e-07.** One measured the RQMC prefix through
layers 1–31 at exactly **1.40x** versus iid MC. Scrambled Sobol tied with a rank-1 lattice.

Sweep 4's independent ceiling argument agrees: our whitening (a control variate on the first- and
second-order ANOVA components) bought 1.56x, implying the low-order part carries only ~36% of
variance; QMC attacks the same terms, so the realistic ceiling on top of whitening is ~1.5x.

**Verdict: closed. Do not spend the remaining days here.**

### 4.6 Ruled out on arithmetic

- **Exact enumeration over linear regions / orthant probabilities** (2503.22082): needs
  activation probabilities near 0 or 1 for entropy pruning to bite. At He init they are ≈0.5 for
  essentially all 8192 hidden neurons. Hopeless.
- **Polynomial chaos**: `C(256+p, p)` coefficients — 33k at p=2, 2.8e06 at p=3, per neuron per
  layer. Degree 1 *is* the Gaussian surrogate we have. Infeasible by orders of magnitude.
- **Sparse-grid Gauss-Hermite / unscented / cubature**: 2509.18712 Thm 3.1 proves sparse-grid
  Gauss-Hermite achieves only `O(N^{-α/2})` in Gaussian Sobolev space and that this is
  **unimprovable by any reweighting**. The unscented transform's 513 points match moments 1–3;
  our whitened+antithetic ensemble already matches moments 1–3 exactly with thousands of points.
  Degree-5 cubature at d=256 needs ~131k points. Dead on three counts.

---

## 5. What the leaderboard structure implies

Sweep 5 fetched the live top 20; sweep 3 fetched the top 25 and cross-referenced the forum.
Both back out the compute multiplier as `u = adjusted / MSE`. **The decoding is validated on us**:
our implied `u = 0.797` against `RESEARCH.md` §8's documented "`_TARGET_UTILISATION = 0.92`, real
spend lands ~0.80". Same submission, correct decoding.

### 5.1 The structure

> **[REDACTED IN THE PUBLIC RELEASE.]** A per-entrant table decoding each named
> account's implied compute multiplier. Withheld for the reason given in
> `RESEARCH.md` §7g-RETRACTION. The structural observation that survives, and that
> §5.2 below actually uses, is that a dozen entries cluster tightly in both cost and
> error — no per-account figures are needed for it.

### 5.2 Three readings, and which to believe

**(i) A dozen entries share fixed cost and fixed error** — twelve of the top 20 sit at
`u ∈ [0.42, 0.54]` with MSE ∈ [1.97e-07, 2.45e-07], and within that band higher `u` does *not*
give lower MSE (rk4: u=0.420, MSE 2.044e-07; rk14: u=0.538, MSE 2.126e-07). For any 1/k estimator
MSE must be `∝ 1/u`. Sweep 5 concludes: a deterministic algorithm run by a dozen people from a
common source.

**(ii) Two entries are over budget** (u = 1.124, 1.169) with no zero-prediction fallback. Sweep 5
reads this as "cost is not a free parameter, therefore deterministic". A forum thread (18129) asks
whether the adjusted score is using an uncapped multiplier, with no organiser reply.

**(iii) Sweep 3's reading: the cost column is dominated by residual wall time.** Then `u` is a
hardware artefact, decoupled from MSE by construction, and can exceed 1.0 without any scoring bug.

**I weight (iii) highest** because it rests on directly measured instrumented-FLOP shares rather
than on correlation structure, and because it explains (i) and (ii) simultaneously. But (i) and
(iii) are not exclusive — the cluster is real either way.

### 5.3 The two calibration points that actually matter

**topic 18106 — adjusted 1.551e-07, raw 2.18e-07, u = 0.71, method fully published.** This is the
best-documented legitimate result on the board, 1.74x better than us. Method: moment propagation
classifies every neuron as almost-certainly-dead / almost-certainly-on / kink; boundary neurons
refined with pilot samples; **on-neurons treated as linear in the final layers**; only kink
neurons fully sampled; antithetic Sobol through layer 30; matmuls accelerated by sorting columns
by firing rate and grouping rows by active-column count. Stated limitation: does not transfer well
under architectural changes.

**mliston — rank 8 at u = 0.116, i.e. essentially the 0.1 floor, Φ = 9.36e-08.** This is 2.88x
better than us *at our own operating point*. Note the arithmetic: if mliston were a 1/k
sampler with our C, raw MSE 8.08e-07 would need k ≈ 29,400, costing ~45% of budget, not 11.6%. So
either their variance constant is ~4x below ours, or the estimator is not a plain sampler. Either
way it is an existence proof that **~3x of Φ is available at the floor**.

*(Sweep 3 states mliston is "3.3x better at the same compute" by comparing our adjusted to their
raw. That is wrong arithmetic. The correct comparison is Φ to Φ: 2.88x. Sweep 5 gets it right.)*

### 5.4 What the structure rules out

There is **no 100x outlier**. The ladder from 4.87e-08 to 1.27e-07 is a smooth 2.6x spread with a
2.1x cliff down to us. Nobody has found a near-exact method. Combined with §1.4 (K=3 is over
budget and less accurate than our MC) and §3.1 (top entries may carry a 2–15x billing discount),
**the honest algorithmic frontier is ~1.5e-07 and we are 1.74x from it.**

---

## 6. Candidate method classes, ranked — with feasibility under our constraints

Constraints for every entry: CPU only, numpy 2.2.6 / flopscope 0.10.0 / whestbench 0.14.0 /
Python 3.10, **no SciPy/PyTorch/JAX/Numba**, FLOP-heavy work through flopscope primitives,
non-flopscope wall time charged at 1e11 FLOP/s, **2.72e11 FLOPs per MLP**, seeded from `mlp.seed`,
precomputed artefacts permitted (50 MiB / 50 files, folder mode, no pickle).

### 6.1 BUILD FIRST — On/dead/kink trichotomy with exact linear collapse

**What.** Generalise our exact dead-column pruning in both directions. Classify each neuron via
cheap moment propagation plus a pilot batch as: **dead** (contributes exactly zero), **always-on**
(ReLU is exactly the identity, so the neuron is *linear*), or **kink**. A run of layers whose
relevant neurons are all on collapses into a single precomputed 256×256 matrix product with
**zero sampling variance** — you multiply weight matrices instead of pushing k samples through
them. Only the kink set needs the ensemble.

**Why it can beat our C.** It does not reduce variance on a fixed integrand; it *shrinks the part
of the function that is stochastic at all*. Variance scales with the kink count, not with n. And
because the network is positively homogeneous and the activation direction collapses with depth
(our own `r90 = 8` at layer 31), the on/dead sets should grow with depth. This is the mechanism
behind topic 18106's measured 1.551e-07.

**Feasible?** **Yes, and it is cheaper than what we run now.** Classification is one moment-
propagation pass (~2e07 FLOPs, we already have `_mean_propagation`) plus a small pilot batch.
Collapsing a run of on-layers is a few `(256,256)@(256,256)` matmuls at 3.35e07 each — negligible
against 2.72e11 — and it *removes* `k·n²` work per collapsed layer. All flopscope-native: matmul,
comparisons, `argsort` (4·N·log₂N), `take`. **No scipy.** The sparsity-grouped matmul is the only
fiddly part: gathers bill at 4x/element against 1x for arithmetic, so it only pays when the active
fraction is well under ~1/2, which is why 18106 sorts and groups rather than gathering per row.

**Accuracy.** Publicly measured at **1.551e-07 adjusted**. A partial implementation (better
classification than our current exact-dead test, without the grouped-matmul optimisation) should
still capture most of the variance reduction.

**Risk.** Classifying a neuron as always-on introduces a bias `σφ(α) − μΦ(−α)`; the threshold on
α must come from a measured error budget, not a guess.

**FIRST EXPERIMENT (half a day, decides everything).** Histogram `α_i = μ_i/σ_i` per layer on real
competition MLPs using `offline_bench.py`. If a meaningful fraction of neurons have `α > 4`
(`P(off) < 3e-05`), the linear collapse pays and this route is live. If activation rates are
uniformly ≈0.5 — which sweep 4 asserts is the He-init default — the trichotomy degenerates to the
dead-pruning we already have and **this route is dead**. Our measured 1.17x pruning gain says dead
columns *do* exist, so rates are not uniformly 0.5; but the always-on tail is unmeasured.

### 6.2 BUILD SECOND — Late-layer distribution refresh

**What.** Push k samples to layer L₀; fit the joint law of `y_{L₀}` from the ensemble (mean +
covariance, Cholesky on 256×256); draw K ≫ k fresh particles from it; push only those through the
remaining `32 − L₀` layers.

**Why.** §4.2's contraction result says mean errors decay by `Φ(ᾱ)√2 ≈ 0.71` per layer, so only
the last few layers determine the answer — and this buys effective samples exactly there. Sweep
4's arithmetic: a full pass is `32·2n² = 4.2e06` FLOPs/sample, so k ≈ 60,000 at 92% budget.
Refresh at L₀ = 30 with k = 25,000 through 30 layers (7.9e10) plus K = 150,000 through 2 layers
(3.9e10) totals 1.2e11 — **~2.5x effective samples for half the budget**; L₀ = 31 gives ~5x.

**Feasible?** **Yes.** One covariance (`k·n²`), one Cholesky/`eigh` (~9n³ = 1.5e08, measured),
one `K×n @ n×n` draw, then normal forward passes. All flopscope primitives, no scipy.

**Accuracy.** The cost is the refresh law's non-Gaussianity, which is exactly our §2 diagnostic
quantity. Our marginal-only version measured 4.06e-06; a full-covariance one-layer version was
measured by another team at 1.16e-06. Taking the latter and 5x effective samples at the floor:
`3.4e-06/5 + 1.16e-06 ≈ 1.84e-06` → **adjusted ~1.84e-07**, i.e. ~1.85x. **Composes with §6.1**
(they attack different things: one shrinks the stochastic part, the other buys particles in the
layers that matter).

**Where §4.4's Edgeworth correction earns its keep.** A skewness-matched refresh (skew-normal, or
a two-component Gaussian mixture matched to the measured third moment along the top principal
directions) should cut the refresh bias by roughly `n^{-1/2}`. **This is the only place Edgeworth
is worth anything** — see §6.4.

**Risk.** Most moving parts of the three routes; L₀ must be swept (24, 28, 30, 31) offline against
the shipped 1e09-sample ground truth. Two to three days.

### 6.3 REJECT — Blending a deterministic analytic anchor with our MC (corrects sweep 5)

Sweep 5's headline "Option A" is: implement K=2 cumulant propagation (1.4% of budget, free under
the floor) and inverse-variance-blend it with our MC, for a claimed **1.7–2.4x**. **The arithmetic
does not survive contact with our own measurements.**

For an unbiased MC estimate with variance `v` and an *independent* deterministic estimate with
squared error `b²`, the blend gives `MSE = v·b²/(v+b²)`, capped at a factor `(1 + v/b²)`. With
`v = Φ/u` and `Φ = 3.4e-07` (our floor-operating constant, from §7d's measured 3.457e-07):

```
S(u) = u · Φ b² / (Φ + u b²)        minimised at u = 0.1  (the floor stays optimal)
```

| anchor | b² | S(0.1) | gain |
|---|---|---|---|
| **actual K=2 cumulant propagation** | **8.03e-05** *(measured, harness)* | 3.26e-07 | **1.04x** |
| K=3 cumulant propagation | ~1e-05 | 2.54e-07 | 1.34x — **but costs 168–227% of budget** |
| *hypothetical* marginal oracle | 4.06e-06 | 1.85e-07 | 1.84x — **not achievable** |

**Sweep 5's error:** it used `b² = 4.06e-06`, our §2 *oracle* number ("Gaussian formula fed with
**true** marginal moments"), as if it were K=2's achievable error. It is not. Sweep 1 verified
that the shipped `examples/03` **is** unablated K=2 with the power-cumulant adjustment already in
it — and it measures **8.03e-05** through the harness. The 4.06e-06 oracle requires true marginal
moments, which is precisely what no cheap analytic method delivers; K=2's predicted σ is 11.4% off
by layer 32 (`RESEARCH.md` §2).

**The general structural argument, which closes the whole family.** The blend formula requires the
anchor's error to be *independent* of the MC error. There are exactly two kinds of anchor:

- **weights-only analytic** (K=2 kprop): genuinely independent, but `b² ≈ 8e-05` — 24x our
  variance, so the blend is worth 4%;
- **MC-moment-fed analytic**: accurate, but ~perfectly correlated with the MC estimate, so
  blending gains nothing — and §4.2 *proves* it (delta method: 1.02–1.03x), which is exactly what
  our §4 and §5 measured.

**There is no anchor that is both accurate and independent.** Reject the entire shrinkage family.
This also retro-explains why our §5b layer-1 anchoring measured neutral.

*(Note the refresh of §6.2 is **not** subject to this argument — it is not a blend of two
estimates but a reallocation of particles, so it inherits the model bias directly rather than
through a correlation-sensitive weight.)*

### 6.4 REJECT AS A STANDALONE — Edgeworth correction of the final layer

Three sweeps promote this. **On its own it is worth nothing, and §4.2 says why.**

Our estimator's final-layer value is a **sample mean of ReLU, which is already unbiased**. There
is no bias to correct. The Edgeworth formula only helps if it *replaces* the sample mean by a
moment plug-in — and §4.2's verified derivation shows that trade buys 2–3% of variance while
eating the full model bias. Our §4 and §5 measured exactly that, twice.

**Feasible?** Trivially (2kn ≈ 3e06 FLOPs, Φ/φ via `erf` or a shipped table, no scipy). **Useful?**
Only *inside* §6.2, where it reduces a refresh bias that genuinely exists. Do not ship it standalone.

### 6.5 TEST CHEAPLY — MFMC control variate with a rank-truncated *nonlinear* surrogate

**What.** Two-model multifidelity MC: `Ê = mean_k(f) − β·(mean_k(g) − mean_K(g))`, where `g` is the
rank-r-truncated network — same ReLUs, truncated linear maps. Variance ratio

```
Var_MFMC / Var_MC = (√(1−ρ²) + √(w·ρ²))²,     w = cost_low / cost_high
```

**Why this reopens a route we closed.** `RESEARCH.md` §8 killed rank truncation because the
truncation **bias** at r=48 was 211x the noise floor. **In an MFMC estimator the bias cancels
identically in the telescoping difference — only ρ and w matter, and we never measured ρ.** The
bias measurement was the wrong measurement for this use.

At r=48, `w ≈ 2r/n = 0.375`, and even ρ=1 gives only 0.61x — so **r=48 genuinely cannot pay**,
confirming half of our closure. But r=8 (our measured `r90` at layer 31) gives `w ≈ 0.06`, and at
ρ = 0.97 the formula gives ~5x; at ρ = 0.99, ~7x.

**Feasible?** Yes. The low-fidelity pass is `2knr + rn²` per layer instead of `2kn²`, all matmuls;
the basis comes from one `eigh` (1.5e08, measured). Flopscope-native, no scipy. Mind the §8 trap:
the fallback ladder must contain exactly one expensive rung.

**Accuracy.** Entirely contingent on ρ. If ρ ≥ 0.97 at r ∈ {8,16}, this alone lands near 1e-07.
If ρ ≤ 0.9, it is worth <1.5x and should be dropped.

**MEASURE ρ FIRST — half a day.** Per-sample correlation between truncated and full network
outputs, for r ∈ {8, 16, 32}, on real competition MLPs. This one number decides the route.
Prior: topic 18097's claim that sampling noise concentrates in the *same* subspace as the signal
(orthogonal complement carries ~1.7% of the error) is *suggestive of high ρ*, but it is a
statement about eigendirections within one forward pass, not about a truncated-network surrogate.
Treat it as encouraging, not decisive.

### 6.6 REJECT — Control variate from the mean-field linearised network

Sweep 4's elegant observation: for any fixed A, `E[ReLU(Aᵀx)_i] = ‖A_{:,i}‖/√(2π)` **exactly**, so
the mean-field linearisation `A = W₀D₁W₁…D₃₁W₃₁` with `D_l = diag(Φ(α_l))` gives a control variate
with an exactly known mean at 1.1e09 FLOPs (0.4% of budget). Our `RESEARCH.md` §8 open lead #1 was
this idea restricted to layer 1.

**Killed by public measurement (sweep 3, topic 18085): input-anchored control variates fail
because the coupling correlation decays to 0.25 by depth 32, against a required ≥0.95.** At
ρ = 0.25 the gain is `1/(1−ρ²) = 1.07x`. A second team measured linearisation control variates at
1.0–1.2x. Two independent measurements, consistent, both against.

Note this does **not** kill §6.5: a rank-truncated *nonlinear* surrogate keeps all 32 ReLUs and is
a far better approximation than a linearisation. Different objects, different ρ.

### 6.7 REJECT — Cumulant propagation at K ≥ 3, in any variant

Over budget (168–507%) *and* less accurate than our current MC (~1e-05 vs 3.4e-06), per §1.3/§1.4.
Factorization makes it *worse* at L=32, not better. K=4 is 16,000% of budget. **Do not port
`mlp_kprop`.** Anyone who does will land above budget and below our current score.

The one honest caveat: sweep 5 disputes the accuracy figure (D2). If sweep 5 were right and K=3
landed at ~2e-07, the route would still fail on cost — you would need a 17x FLOP reduction to get
it under the 0.1 floor, which nobody has demonstrated. **The cost wall alone is sufficient.**

### 6.8 REJECT — Randomized QMC, sparse grids, cubature, polynomial chaos, exact enumeration

All covered in §4.5–4.6. QMC: no usable theory at d=256, publicly measured at 1.40x, two teams
landed *worse* than us. Sparse grids: provably suboptimal. Cubature/unscented: our existing
ensemble already matches moments 1–3 with more points. Chaos and enumeration: infeasible by
orders of magnitude. **This closes `RESEARCH.md` §7d's leading open hypothesis.**

### 6.9 REJECT ON POLICY — Residual-wall-time compute channel

Technically available (§3.1), worth ~1.7–2.4x post-v0.10.0, and probably the bulk of the gap to
rank 1. Rejected on four independent grounds listed in §3.1. Its value to us is diagnostic only:
it tells us the real target is ~1.5e-07, not 4.87e-08.

### 6.10 KEEP FOR THE WRITE-UP, NOT THE SCORE — Depth-robust mechanistic estimation

ARC states the open problem in Appendix D verbatim (§1.5), and Phase 1's depth 32 is that problem
instantiated. The strongest public attempt (topic 18097, trajectory-calibrated moment chain)
reached raw 5.89e-06 / adjusted 9.97e-07 and its author concedes sampling wins Phase 1's metric.
Research-grade, not a five-day build. But it is the natural subject of the **17 Aug** write-up, and
our §3 measurement plus the paper's Appendix S.1.1 mechanism (§1.5) is genuinely publishable
material: *the shipped K=2 baseline is the order-1 truncation of a series that does not pay to
extend, because the binding error is the accumulated ADF projection error, and the CLT assumption
underpinning the whole method fails at depth 32 exactly where we measured the rank collapse.*

---

## 7. What this CONFIRMS about our four negative results

### 7.1 Route 1 (higher-order Mehler) — measurement CONFIRMED, conclusion PARTLY REFUTED

**Confirmed, with the mechanism named twice over.** Sweep 4: the shipped formula is the Cho-Saul
arccosine kernel, exact only under joint Gaussianity, and the scheme is ADF whose error is
dominated by the *accumulated* projection term rather than the per-step map — so making the
per-step algebra exact cannot help. Sweep 1: Corollary A.3 proves the Wright et al. covariance
formula is the leading term of an expansion that is *exact* for Gaussian Y, so extending it adds
nothing except exposing the surrogate error. **Our §3 interpretation — "the pairwise formula is
not the bottleneck; the joint Gaussian surrogate is" — is exactly right and now has two
independent proofs.**

**Partly refuted.** Sweeps 1, 2 and 5 all identify our experiment as the paper's own **Appendix E
ablation**: applying the diagram summation formula to φ directly, without power cumulants and
without the odd-K full trace. Figure 3(b) shows that variant has MSE `O(1)` — flat in width. Our
5.95e-05 → 6.54e-05 sequence is that flat line. So the axis we tested (exactness of the order-2
formula) is not the axis that matters (**cumulant order K**), and our sentence "analytic
propagation of this family is stuck near 6e-05 regardless" is too strong.

**But the correction does not change the decision.** The right axis (K=3) is ruled out
independently, on cost (§1.3) and on accuracy (§1.4). We reached the right verdict on partly wrong
reasoning. **The write-up must state this honestly** — it is a better story than the original.

### 7.2 Route 2 (low-rank truncation) — CONFIRMED for standalone use, REOPENED for MFMC

**Confirmed and sharpened.** Topic 18097 supplies the mechanism we lacked: sampling noise
concentrates in the *same* subspace as the signal (`∝ λ_i/N` per direction), so the orthogonal
complement carries only ~1.7% of the sampling error and a perfect analytic correction there is
capped at ~2%. This also undercuts ARC's own stated townhall direction. Separately, sweep 1 notes
the paper's *factorized* representation is a **lossless** restructuring of a symmetric tensor, not
a lossy truncation of an activation matrix — the two are unrelated, so our closure of one does not
bear on the other.

**Reopened in one specific configuration.** We measured truncation **bias**. In an MFMC estimator
the bias cancels and only **ρ** matters, which we never measured. See §6.5. The r=48 half of our
closure survives (cost ratio alone forbids it); the r=8 case is untested.

### 7.3 Route 3 (exact layer-1 anchoring) — CONFIRMED three ways

Sweep 1: the paper's Appendix C states *"Our procedures are exact when there is only 1 hidden
layer"* — there is nothing to gain from anchoring the one layer everything already gets right.
Sweep 3: topic 18053 built an entire estimator on exactly this anchor
(`E[ReLU(z₁,ᵢ)] = ‖W₀[:,i]‖/√(2π)`), with a proper unbiasedness proof, and reached adjusted
4.10e-07 — **worse than ours**. Sweep 4: §4.2's delta-method derivation explains *why* moment
anchoring cannot pay, quantitatively reproducing both our §4 and §5 numbers. Closed, with a proof.

### 7.4 Route 4 (concentrated error mode) — CONFIRMED, with one important scope limit

Our ICC ≈ 0 result stands and the methodological lesson is right: *error energy concentrating in a
direction is not evidence of a correctable bias.* Nothing on the leaderboard suggests anyone is
exploiting a learned bias correction.

**Scope limit worth recording (sweep 2).** The ICC argument applies to **Monte-Carlo** error,
which is a property of the random draw. It says nothing about the error of a **deterministic**
estimator, which is a bit-identical function of the weights and therefore has ICC = 1 by
construction. If we ever ship an analytic component, its residual *is* learnable in principle.
This does not resurrect the learned-corrector route for our current estimator, but the argument
must not be over-generalised in the write-up.

### 7.5 Bonus confirmations

- **float16** — forum confirms float16/int16 bill at rate 1.0, identical to float32. Our §6
  measurement was right; there is no dtype discount below float32.
- **Our ~3.4e-07 sampling ceiling is real and independently reached.** arianvassili claims a floor
  of ~3.7e-07 (range 3.4–3.9e-07) with structural claims formalised in Lean 4/Mathlib; another
  participant reports reaching a similar floor with ~10% left. **But their per-pair variance
  constant is ~0.012 against our measured 2.375e-02 — 2x better.** The floor is real, and **we are
  2x above it.** That floor is also conditional on pushing every sample through the whole forward
  pass; §6.1 breaks that assumption structurally.
- **`RESEARCH.md` §7d's QMC hypothesis is refuted** by two independent public measurements and by
  the d=256 rate theory. Remove it from the open-leads list.

---

## 8. Honest assessment: what is reachable by 10 Aug, and what is Phase 2

Five days. Submission deadline 10 Aug 23:59 UTC; 50 submissions/day available; write-up 17 Aug.

### 8.1 The realistic target is 1.5e-07, not 1e-07

The 1e-07 framing in the brief came from the leaderboard's top entries. If §3.1 is right — and it
is the best-evidenced claim in the sweep — those entries carry a 2–15x compute-billing discount
and are not an algorithmic target. **The honest frontier is topic 18106's 1.551e-07**, and we are
1.74x away from it with the method written down.

### 8.2 Day plan

| Day | Work | Decides |
|---|---|---|
| 1 (half) | Histogram `α_i = μ_i/σ_i` per layer on real MLPs, `offline_bench.py` | Whether §6.1's always-on collapse exists at all |
| 1 (half) | Measure per-sample ρ between rank-r truncated and full network, r ∈ {8,16,32} | Whether §6.5 is 5x or nothing |
| 2–3 | Build §6.1: three-way classification + exact linear collapse of on-runs + kink-only sampling | Main expected gain |
| 3–4 | Build §6.2 if §6.1 lands early, or §6.5 if ρ came back ≥0.97 | Composition |
| 4 | 1,000-MLP `full`-split run-off against current submission, paired, common random numbers | Ship / don't ship |
| 5 | Harden fallback ladder (one expensive rung), utilisation targeting, submit | Safety |

**Methodological discipline from `RESEARCH.md` §7b, non-negotiable:** the 100-MLP mini split has a
10.4% relative sd, so differences under ~21% are invisible there. Every comparison goes on the
1,000-MLP `full` split with common random numbers and paired tests. We have already been burned
once by a 19% "advantage" that sat inside the noise band.

### 8.3 Probabilities, stated honestly

| Outcome by 10 Aug | Probability |
|---|---|
| Beat our current 2.700e-07 at all | **~65%** |
| Reach ≤ 1.8e-07 (top ~30) | ~40% |
| Reach ≤ 1.5e-07 (match the best documented legitimate method) | ~22% |
| Reach ≤ 1.0e-07 (top 10) | **~15%** |
| Reach ≤ 5e-08 (rank 1) without the billing channel | **<3%** |

The 65% is driven almost entirely by §6.1 being a *published recipe* rather than an invention, and
by the fact that it extends machinery we already have (dead-column pruning, moment propagation,
the whitened+antithetic sampler). The main failure mode is the §6.1 gating experiment coming back
negative — if activation rates really are near 0.5 throughout, the always-on set is empty and the
route degenerates to what we already ship.

### 8.4 Do not do these, even if a day frees up

Cumulant propagation at K≥3 (over budget and less accurate); K=2 as a shrinkage anchor (4%, §6.3);
Edgeworth as a standalone correction (0%, §6.4); randomized QMC (1.4x measured, two teams landed
worse than us); the mean-field linearised control variate (ρ=0.25 measured); low-rank truncation
as a *standalone* estimator (211x the noise floor, our own measurement); the residual-wall-time
billing channel (policy).

### 8.5 Phase 2 / write-up material

The genuinely valuable output of this sweep may not be a score. **The competition at L=32 is ARC's
own stated open problem** — "we believe a sample-free algorithm can be developed whose error also
does not increase with depth, but we leave this problem to future work" — and we have a measured
account of exactly where and why the shipped K=2 method breaks at that depth, plus the mechanism
(Appendix S.1.1's CLT assumption failing against our measured rank collapse), plus a correct
identification of our own §3 experiment as the paper's Appendix E ablation. That is a coherent,
honest, self-critical write-up, and the algorithmic-contribution prize reviews roughly the top 10
regardless of exact rank.

The two research directions worth carrying into Phase 2, neither buildable in five days:

1. **Depth-robust mechanistic estimation** — periodic re-anchoring or renormalisation of a
   cumulant expansion so truncation error stops compounding as `L^K`. Note the warning from topic
   18097: naive layerwise re-anchoring *compounds* bias, and re-anchoring the covariance to truth
   made things worse. Whatever the construction is, it is not that.
2. **Depth-linear factorized cumulant propagation** — killing the `L²` in `30n³L²` by recompressing
   the factor list to `O(n)` each layer would bring factorized K=3 from 197% of budget to ~6%.
   Publishable independently of leaderboard rank. But by §1.4 the resulting estimator would still
   be less accurate than our MC at L=32, so it is a *paper*, not a *submission*.
