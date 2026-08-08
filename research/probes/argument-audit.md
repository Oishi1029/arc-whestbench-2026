All measurements complete. Final report:

---

## 1. VERDICT

**A third arithmetic error of the same species exists (A6 minimax: "~800 moments" omits the 0.1 multiplier; correct value ~201, a 3.9x error), but the load-bearing failure is different and larger: the A4.1 LP table is unsolvable from its own stated inputs — D=5 max-VR is 4.43x, not 10.8x — and the A6 cumulant cost table is 1.5–3.1x off the ARC companion paper's own fitted FLOP polynomials, shrinking the "K≥3 is over budget" margin from 2.5x to 1.18x. Meanwhile the live leaderboard puts three entries at FOM 108.8 / 272 / 381, and the report's own strongest bound (FOM 210.8) sits *above* the leader — so it is already falsified as a bound. This probe builds no estimator; FOM unchanged at 4.99e4.**

## 2. MEASUREMENTS

### 2a. Arithmetic re-derivation — discrepancies above 1.2x (full 150-row table in `aud_01.out`)

| § | Claim | Stated | Recomputed | Ratio | Load-bearing? |
|---|---|---|---|---|---|
| **A6** | minimax: exact marginal moments for 4.00e-10 | **800** | **201** (exp −1.7) / 171 (OLS −1.92) | **0.25x** | error confirmed: 779.5 if 4.00e-10 is treated as *raw*. The 0.1 multiplier is applied **zero times**. Conclusion (astronomical) survives |
| **A4.1** | LP D=5 surviving C | **4.364e-3** | **1.063e-2** | **2.44x** | **yes** — line 5 moves 3.98e-8 → 9.69e-8 |
| **A4.1** | LP D=5 max VR | 10.8x | **4.43x** | 0.41x | yes |
| **A4.1** | LP D=32 max VR | 570x | **28.6x** | 0.05x | yes |
| **A6** | cheapest K=3 cumulant cost | 2.529 B (1.895 B "factorised") | **1.676 B** (paper Table 1, basic) | 0.66x | **yes** — margin to affordable is 1.18–1.68x, not 2.5x |
| **A6** | K=4 cost | 863.8 B | **274.9 B** | 0.32x | no (still hopeless) |
| **A6** | K=2 cost | 0.0079 B | **0.0140 B** | 1.77x | no (both under the 0.1 floor) |
| **A3** | grader-meter vs closed form agreement | 0.012% | **0.20%** (vs 4,202,752) / **0.40%** (vs 4,194,560) | 17–33x | no |
| **A6** | cubature 33,153 nodes cost | 51.1% of B | **30.2%** at `c_real` | 0.59x | no (uses naive `c`, inconsistent with rest) |
| **A8** | 2.4x arbitrage on "Route-1 floor" | 8.6e-9 (22x above target) | **3.23e-10 (0.81x — below target)** on the *retained* line-7 bound | — | **yes** — A8 is computed from the 2.07e-8 that A4.3 **retracts** |
| **A2** | C for whitened+antithetic | 1.886e-2 | **2.377e-2** (measured, 120 MLPs) | **1.26x** | yes — see 2b |
| A6 | minimax decay exponent | −1.7 | −1.92 (OLS), local −1.67→−2.01 | 1.13x | no |
| A6 | cost growth 2→3 | 384x | 320x (from the table's own 2.529/0.0079) | 0.83x | no |
| A6 | (L/n)^5 | 3.8e-5 | 3.05e-5 | 0.80x | no |
| A4.1/2 | LP base C | 4.706e-2 used | canonical is 5.219e-2 (A4's own choice) | 1.11x | no (conservative direction) |
| A10 | 257 subs at 50/day | "about six days" | 5.14 days | 0.86x | no |

**Everything else checks.** Möller node counts are exact (`C(257,1)=257`, `C(258,2)=33,153`, `C(259,3)=2,862,209`, `C(260,4)=186,043,585`); "m=3 is 26x over budget" recomputes to 26.10x; `k_max=109,653`; all of A7 (GT floor 5.219e-12 adjusted, N_crit 1.305e7, N_eff, −3σ), all of A9's Φ ladder, all of A0's ratios-to-target, `2^256=1.16e77`, `256^31=4.52e74`, `C(260,5)=9.53e9`/38.1 GB, and the ε_μ/ε_σ sensitivity chain all reproduce to <1%.

### 2b. The A4.1 LP is unsolvable from its own R(ρ) table

Two-phase simplex (Bland's rule, d=1..2000, 23 constraints), solution verified feasible (Σc=1.0000000, min c=0, max|resid| exactly at tolerance):

| tol on R(ρ) | max removable mass, D≤5 | surviving C | max VR | Φ at `c_real` |
|---|---|---|---|---|
| **0.002 (report's stated floor)** | **0.77411** | **1.063e-2** | **4.43x** | **9.69e-8** |
| 0.005 | 0.80702 | 9.08e-3 | 5.18x | 8.28e-8 |
| 0.02 | 0.86331 | 6.43e-3 | 7.32x | 5.87e-8 |
| 0.05 | 0.92723 | 3.42e-3 | 13.7x | 3.12e-8 |
| *report's claim* | *0.90727* | *4.364e-3* | *10.8x* | *3.98e-8* |

Bisection: the report's D=5 row **first becomes feasible at tolerance ≈ 0.0389** — 19.5x looser than its stated `max(3se, 0.002)`. I re-measured R(ρ) independently (12 MLPs, 24,000 pairs each) and it agrees with the report's table to 1–6% (ρ=0.3 worst at 0.943x), so the discrepancy is in the LP solve, not the data. My LP on my own R(ρ) at tol 0.005 gives D=5 VR **4.93x**, Φ 8.84e-8.

Corroborating coherence check: at D=3 the corrected ceiling is 3.16x and whitening+antithetic measures 2.21–2.77x — a 1.14–1.43x gap. Under the report's 5.41x ceiling the gap was an unexplained 2x.

**Direction: this makes the impossibility case *stronger* on the cubature branch** (line 5: 3.98e-8 → 9.69e-8, FOM 1.08e4 → 2.64e4). It is an error, not an opening.

### 2c. Variance constants re-measured (120 real full-split MLPs, 4 streams, k=4096/16384/32768)

| estimator | report | measured C | ratio |
|---|---|---|---|
| plain MC | 5.219e-2 | **0.049113** (se 2.5e-3) | 0.94 |
| whitened | 2.701e-2 | **0.026122** (se 1.0e-3) | 0.97 |
| **whitened + antithetic** | **1.886e-2** | **0.023769** (se 9.4e-4) | **1.26** |
| direct `mean_i Var(f_i)`, 40k samples | 5.219e-2 | 0.051242 (se 2.8e-3) | 0.98 |

C is flat in k (0.02315 / 0.02435 / 0.02381), confirming p=1. **The A2 "Theorem 1 validation" is circular for our own row:** only the product `C·c = 49,885` is observed. With the measured `C_wa = 2.377e-2` the implied live `c` is **2.099e6**, not the 2.645e6 stated (and A2's own k-sweep table already says C_wa ≈ 2.246e-2, 1.19x above the 1.886e-2 it then uses). The identity itself is sound algebra; the table does not independently confirm it.

A7's model-free layer-1 check reproduces: mean s²=1.999431 (He 2.000), exact layer-1 Var 0.6814962, GT MSE 6.857e-10, **N_eff = 9.939e8** — the 1e9 bake is confirmed and A7 is not binding.

### 2d. Scoring arithmetic vs the live board (fetched today)

| entry | rank | adjusted | raw MSE | mult = adj/raw | **FOM = adj·B** | subs |
|---|---|---|---|---|---|---|
| dpskv5 | 1 | 4.0e-10 | 3.6e-9 | 0.111 | **108.8** | 285 |
| joe_wanza | 2 | 1.0e-9 | 4.0e-9 | 0.250 | **272** | 994 |
| huang_chung_yi | 3 | 1.4e-9 | 1.41e-8 | 0.099 (floored) | **381** | 470 |
| ednacob | 4 | 4.62e-8 | 9.11e-8 | 0.507 | 1.26e4 | 115 |
| dstepanov | 5 | 5.81e-8 | 1.051e-7 | 0.553 | 1.58e4 | 461 |
| SOX (topic 18106) | — | 1.551e-7 | 2.18e-7 | 0.712 | 4.22e4 | — |
| ours (#324358) | 54 | 1.834e-7 | 2.593e-7 | 0.707 | 4.99e4 | — |

`Φ = C·c/B` reproduces every row exactly (adjusted = raw × max(0.1, u)); the identity is confirmed on five independently graded entries, not two. Two inconsistencies: (i) RESEARCH 7g-RET states dpskv5 is "at the 0.1 floor" with raw 3.628e-9, but 4.0e-10/3.6e-9 = **0.111 > 0.1** — it is *not* floored, a 10% inconsistency; (ii) topic 18106's utilisation is 0.7115, not the 0.739 recorded.

**The decisive number:** the report's strongest defensible bound (line 7, free exact μ **and full Cov** of all 31 hidden layers, at `c_real`) is Φ 7.75e-10 = **FOM 210.8**. The leader sits at **FOM 108.8 — 1.94x below it**, and #2 at 272 is within 1.3x. A bound that the public board already violates is not a bound.

### 2e. The ARC companion paper (arXiv 2605.05179) — Table 1, priced at n=256, L=32

The paper ships *exact fitted FLOP polynomials* (Appendix J, Table 1). Report vs paper:

| variant | paper's polynomial | at n=256,L=32 | report A6 | report/paper |
|---|---|---|---|---|
| K=2 basic (=factorized) | 7n³L + 26n²L | **0.0140 B** | 0.0079 B | 0.56x |
| K=3 basic | (7/3)n⁴L + 252n³L | **1.676 B** | 3.026 / 2.529 B | 1.51–1.81x |
| K=3 factorized | 30n³L² + 39n³L | **1.972 B** | 1.895 B | 0.96x |
| K=4 basic | (5/4)n⁵L + 224n⁴L | **274.9 B** | 863.8 B | 3.14x |

Three corrections: (i) the report's **"ideal β₃ = 7/18 → 1.177 B" is not an unattainable ideal — it is exactly the paper's fitted leading term** (7/3)n⁴L = 1.179 B; (ii) the report names the *factorized* variant the cheapest K=3, but at L=32 it is the **dearest** — the paper says so explicitly, *"at high enough depth… they use more FLOPs"* (Appendix J); (iii) applying the report's own Strassen-4 (1.462x) to the n⁴L matmul term brings K=3 basic to **1.304 B**. So the true margin on "every K≥3 is over budget" is **1.18–1.68x**, not 2.5–3x — a binary cliff decided by a ~30% modelling margin.

**But A6's verdict survives on accuracy, and the paper is the strongest evidence for it.** Theorem 5.2 gives MSE O(ε²) in O(n/ε²) vs MC's Θ(n²/ε²) — an n-fold (256x) advantage — *but only at fixed depth*. With cost ~ n^{K+1}L and error ~ c_K(L/n)^K, the invariant G = MSE×FLOPs scales as **n·L^{K+1}**, so the advantage falls by 8^K = 512x going from the paper's demo (L=4, "over 100x" better than MC) to L=32. Calibrating c₂=0.079 from the report's measured K=2 raw MSE 6.44e-5: the only affordable order at depth 32 is **K=2, floor 6.4e-5 raw / 6.4e-6 adjusted — four orders above target.** Phase 1's "4× deeper" multiplies the analytic error by 4^K while the budget scaled only linearly.

### 2f. Public reconnaissance — what is *stated* vs what I *infer*

**Stated (attributed, short quotes):**
- Topic **18106** (SOX, 27 Jul): *"adjusted score 1.551×10−7 and raw final-layer MSE 2.18e−7"*. Method: moment-propagation classification into dead / on / kink; *"antithetic Sobol samples through layer 30"*; on-neurons treated linearly in the last two layers; **columns sorted by firing rate, rows grouped by active-column count** to speed the matmul. Code withheld until close. Caveat: *"our method does not easily transfer with architectural changes"*.
- Topic **18099** (jtel): NumPy arrays reachable via `flopscope.numpy` do real work at **zero instrumented FLOPs**; their own submission 319062 at share 0.93 while several top entries were *"below 0.001"*.
- Topic **18108** (28 Jul): top competitors show *"instrumented/effective-compute ratios below 0.001"*; the board is *"significantly determined by who can build the fastest native backend"*. Proposes capping / repricing / FLOPs-only scoring.
- Topic **18122** (Mohanty, organiser): eval servers ship `flopscope-client`, so *"you wont be able to do operations directly on the array that bypass the accounting boundary"*; RemoteArray has no `.base`; *"it would be cheaper for you to use flopscope instead"*.
- Topic **18125** (organiser, **3 Aug**): *"copies, gathers, sorts, and other data-movement operations are no longer free"*; *"64-bit class costs 2x the 32-bit class"*; participants limited to *"one physical core (2vCPUs)"* while the flopscope backend gets seven; λ = 1e11 FLOP/s *"does not provide a second compute lane"*; `predict_timeout_s = 30.0`, `wall_time_limit_s = 60.0`; prizes from **50 private MLPs** on fresh seeds, up to 2 nominated submissions.
- Topic **18105** (27 Jul): a floor claim of *"roughly 3.7e-7 adjusted"*, machine-checked *"in Lean 4 / Mathlib with zero `sorry`"*. A respondent: *"the top rankers might have found quite a few FLOP arbitrages"*.
- No public code, gist or writeup describing a method below 1e-07 exists that I could find. The three public Phase-1 writeups I located (18063, 18085, 18097) are all in the 1e-6 range and none claims sub-1e-7.

**Inferred (mine):** the 18108/18099 "<0.001 instrumented share" observations predate the 3 Aug one-core + data-movement repricing; RESEARCH 7g-RET's 7 Aug telemetry (0.94 / 0.71 / 0.23 for ranks 1–3) is consistent with the channel having been substantially closed at the top, while #3 still bills 77% through residual time. The three sub-400-FOM entries sit at three *different* utilisations (0.111, 0.250, 0.099), which argues against a single shared billing artefact.

## 3. WHAT WOULD HAVE TO BE TRUE FOR ME TO BE WRONG

- **On the LP:** if the report's LP used a tolerance ≈ 0.039 (from a per-ρ standard error ~0.013, i.e. 3se with se≈0.013), its table is defensible and mine is the wrong comparison. But its stated rule is `max(3se, 0.002)` and my own R(ρ) has se well under 0.005 at 12 MLPs × 24k pairs, so 3se ≈ 0.039 would need per-ρ noise 8x larger than mine on 16 MLPs × 150k pairs. Also possible: the report's LP included the ρ→0 and ρ→1 endpoints as hard constraints I omitted, which would only tighten it further (moving *my* way).
- **On the cumulant costs:** if flopscope prices einsums materially above PyTorch's FlopCounterMode extension (the paper's meter), the report's higher numbers could be right. The report says its table is flopscope-metered; the paper's is a fitted exact polynomial. I did not re-meter either.
- **On the "800 moments":** if the intended target was raw MSE 4.00e-10 rather than adjusted, the figure is right — but every neighbouring row in A6 is quoted raw and compared to 4.00e-10 *adjusted*, and 49→800 is 1.21 decades, breaching the report's own one-decade extrapolation rule.
- **On the leader-vs-bound falsification:** if dpskv5's public-50 score does not survive the withheld-50 or the private re-run, FOM 108.8 is not a real method and the oracle bound is untested rather than violated. Rules §5.4/§5.5 already settle this at phase close.

## 4. IS 1e-09 REACHABLE?

**The strongest surviving reason to think yes: it is already on the board three times, at FOM 108.8 / 272 / 381, at three different budget utilisations, with the top two billing 71–94% of their compute through the instrumented meter — and the only thing the report interposes between "everything constructible lands at 1.5e-7" and the target is a *measured oracle*, which §A4.3 itself concedes "is never a lower bound on the score". That oracle's FOM (210.8) is already 1.94x above the leader's, so it is falsified as a bound by public data. The specific claim whose failure flips the conclusion is A9 item 3, `c ≥ 2.48e6 FLOPs per evaluation`** — the assumption that every estimator evaluation must cost a full 32-layer, width-256 forward pass. A9's own ladder shows Φ_oracle = 3.90e-10 at c = 1.25e6, i.e. **a single factor of 2 in `c` spans the entire remaining margin**, while every other link in the report carries ≥100x. And the one method description that is actually public (topic 18106) reduces cost precisely by *restructuring* the pass — sorting columns by firing rate, grouping rows by active-column count, collapsing on-neurons in the last two layers — a mechanism the report prices only as two multiplicative constants (1.154x pruning, 1.462x Strassen) and never bounds structurally. Against this, the strongest reason to think **no** is the companion paper's own scaling law: the analytic branch's figure of merit degrades as L^{K+1}, the only affordable cumulant order at depth 32 is K=2, and its accuracy floor is 6.4e-6 adjusted — so whatever the leaders are doing, arXiv 2605.05179 says it is not cumulant propagation.

**Files:** `aud_01_arith.py`/`aud_01.out` (150-claim re-derivation), `aud_02_C.py`/`aud_02.out` (variance constants, N_eff), `aud_03_lp.py`/`aud_03.out` (simplex + R(ρ) re-measurement), `aud_04_final.py`/`aud_04.out` (paper Table 1 pricing, leaderboard FOM), `aud_06_lpcheck.py` (LP feasibility verification + tolerance bisection), `paper.txt` (arXiv 2605.05179 full text) — all in `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/`.

**Sources:** [leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards) · [18106](https://discourse.aicrowd.com/t/18106) · [18099](https://discourse.aicrowd.com/t/18099) · [18105](https://discourse.aicrowd.com/t/18105) · [18108](https://discourse.aicrowd.com/t/18108) · [18122](https://discourse.aicrowd.com/t/18122) · [18125](https://discourse.aicrowd.com/t/18125) · [arXiv 2605.05179](https://arxiv.org/abs/2605.05179)