## VERDICT

**The cost axis is worth 1.30× against the live submission and no more: best metered c = 2.032e6 FLOPs/sample (2.064× below naive 4.194e6), giving FOM 3.83e4 (Φ 1.409e-07) at C = 1.886e-2 — still 141× above the 272 target. The report's "1.25e6 metered floor" is wrong by 1.72×: it assumes additions are free, and flopscope charges them at 1.0/element, which caps Strassen at 1.686× no matter how deep you recurse.**

---

## 1. Exact meter model (all metered, not modelled)

`matmul((k,m)@(m,o))` = `k·o·(2m−1)` — i.e. (2·inner−1) per output element, exactly. Verified at 8 shapes.

| primitive | charge | note |
|---|---|---|
| f32 matmul | `k·o·(2m−1)` | inner-dim splitting is meter-neutral (2 halves + 1 add = identical) |
| elementwise (`maximum`,`add`,`sub`) | 1.0/elem | |
| reduction (`max/sum` axis 0) | 1.0/elem of input | |
| `concatenate`, `stack`, `reshape`, `.copy()` | 1.0/elem of output | arrays are **immutable** — no `out=`/slice-assign |
| basic slice `X[:, :m]`, `W[p:]`, `.T` | **0** | |
| fancy/boolean gather | 4.0/elem | |
| `eigh(n)` = 9n³, `cholesky(n)` = n³/3, `inv` = 2n³, `solve` = 8n³/3 | | |
| `einsum('ki,kj->ij',X,X)` | `n(n+1)/2·(2k−1)` | **exact 2× symmetry discount vs `X.T@X`, bit-identical (max diff 0.0)** |
| `fnp.random.standard_normal` | 32/elem (f64) | core's convention uses raw `np.random` (0) |

**Baseline c** (32 layers, matmul + ReLU, nothing else) = `32·(256·511 + 256)` = **4,194,304 exactly.**

## 2. Strassen-Winograd — metered, all depths

Implemented on nested block tuples so no `concatenate`/`reshape` is ever charged; weight-side (B) transforms are k-independent and amortised per MLP. **My closed form matched the meter to the FLOP at d = 0…6.**

| d | leaf mults | A-adds | C-adds | adds % | /layer/sample | 32-layer chain | vs naive |
|---|---|---|---|---|---|---|---|
| 0 | 130,816 | 0 | 0 | 0.0% | 131,072 | 4,194,304 | 1.000× |
| 1 | 114,240 | 256 | 448 | 0.6% | 115,200 | 3,686,400 | 1.138× |
| 2 | 99,568 | 704 | 1,232 | 1.9% | 101,760 | 3,256,320 | 1.288× |
| 3 | 86,436 | 1,488 | 2,604 | 4.5% | 90,784 | 2,905,088 | 1.444× |
| 4 | 74,431 | 2,860 | 5,005 | 9.5% | 82,552 | 2,641,664 | 1.588× |
| **5** | 63,026 | 5,261 | 9,207 | 18.6% | **77,750** | **2,488,000** | **1.686×** |
| **6** | 51,471 | 9,463 | 16,560 | 33.5% | **77,750** | **2,488,000** | **1.686×** |
| 7 | 38,604 | 16,816 | 29,428 | 54.3% | 85,103 | 2,723,298 | 1.540× |

Mults fall as (7/8)^d, k-dependent additions grow as (7/4)^d. Flat optimum at d = 5–6. **Additions-free hypothetical = 1,446,577 (≈ the report's 1.25e6); metered reality 2,488,000. The claimed floor is optimistic by 1.72×.** Winograd's 15 additions is proven minimal for rank-7 ⟨2,2,2⟩, so this is not an implementation defect.

**Accuracy is not a constraint anywhere near the optimum** (40 MLPs, k=2048, whitened+antithetic):

| | MSE vs f32-exact chain | MSE vs ground truth F |
|---|---|---|
| f32 exact vs f64 | 3.955e-13 | — |
| sampling error (C/k) | — | **8.562e-06** |
| Strassen d=1 | 7.20e-14 | 8.5620e-06 |
| d=3 | 3.15e-13 | 8.5621e-06 |
| d=5 | **2.96e-12** | 8.5614e-06 |

d=5 numerical error is 2.9e6× below the sampling MSE and 17× below the dataset's own 5.2e-11 noise floor. Growth is ~2.4×/level, so even d=8 stays ≲5e-11. **Paired CRN, n=60 MLPs:** d=3 ratio 1.000018 (t=+1.03), d=4 ratio 0.999994 (t=−0.26). Indistinguishable.

## 3. Sparsity — element-level is dead, column-level is real

**Measured density is exactly 0.5000 at every one of the 32 layers** (mean 0.49984, 24 MLPs, k=4096) — as theory demands for a bias-free ReLU net. Element-level sparsity is unexploitable: flopscope charges dense matmuls by shape, and a gather at 4/elem to compact a per-row-varying pattern is impossible in one matrix. Block sparsity by sorting samples is hopeless — columns are ~independent at p=0.5, so a b×c all-zero block has probability ~2^−bc.

The **only** exploitable structure is whole-column death per chunk, which is large and grows with depth (batch directions collapse). Column-death per chunk, ordering taken from an **independent 1024-sample pilot** (no circularity):

| chunk | prefix-dead (free basic slice) | full dead set (needs 4/elem gather) | capture |
|---|---|---|---|
| 8 | 57.3 / 256 | 79.1 | 0.724 |
| 16 | 52.2 | 70.1 | 0.744 |
| 32 | 47.1 | 63.1 | 0.746 |
| 64 | 41.5 | 57.5 | 0.721 |
| 256 | 27.3 | 48.7 | 0.560 |
| 1024 | 13.1 | 42.1 | 0.312 |
| 4096 | 6.1 | 36.9 | 0.166 |

Per layer it runs 0 at layer 1 → 12.6 at layer 6 → 45.8 at layer 16 → 73.0 at layer 32 (chunk 64). **The full dead set loses to the prefix**: the Y-gather (4·c·m) plus W-row gather (4·m·256/c) exceeds the extra saving at every chunk ≤ 256, and at larger chunks it forces the Strassen weight-prep to be recomputed per chunk (1.35e6/layer at d=5 → 42,000/sample), which is fatal. Permuting the weights once per MLP by pilot activity (a 4/elem gather, 8.4e6 per MLP → 128/sample) turns the prefix into a **free** basic slice on both `Y[:, p:]` and `W[p:]`.

**Pruning is exactly bit-identical** at fixed permutation: 12/12 MLPs, max deviation **0.000e+00**. (The permutation itself reorders f32 accumulation: max deviation 1.43e-6, same order as ordinary f32 nondeterminism, MSE impact ~2.5e-13.)

Metered d=0 pruned chain on real MLPs vs my model (validates the model to 2.4% worst case):

| chunk | metered | model | mean live m |
|---|---|---|---|
| 16 | 3,383,462 (1.240×) | 3,395,072 | 206.3 |
| 32 | 3,486,598 (1.203×) | 3,475,968 | 212.6 |
| 64 | **3,602,967 (1.164×)** | 3,564,032 | 219.7 |
| 256 | 3,879,271 (1.081×) | 3,786,752 | 236.5 |

The 1.164× at chunk 64 reproduces the live submission's stated 1.17× exactly.

## 4. Other levers, measured

- **Fuse the whitening transform into W₁** (T@W₁ once per MLP, 3.35e7 amortised): removes the explicit `x@T` application. **Saves 131,072/sample.**
- **Gram via `einsum('ki,kj->ij')`**: 131,056 → **65,784/sample**, bit-identical. Together these take whitening's marginal cost from 262,912 to **66,304/sample**.
- **No symmetry discount exists for the layer matmul**: `X @ W_sym` = 267,911,168, identical to a general matmul (correct — symm is not cheaper in flops).
- **Always-on merge** (dual of dead-column pruning): measured |A| = 78.8 ≈ |D| = 79.4, |P| = 97.8 at chunk 8. Merging a layer pair saves `512|A| − |P|(2m−1)` ≈ 5.9e3/sample/pair ≈ **2.8% total — and it is circular**: certifying "always on" requires computing the very `z` you are skipping, and the interval bound over the nonneg orthant is vacuous (`z_j ≤ Σ ymax_i·w⁺_ij ≥ 0` always). **Closed.**
- **Output-dim pruning**: same circularity. Impossible. This is why pruning is linear in m, not quadratic.
- Micro: in-place ReLU is not available (immutable arrays); `reshape` costs 1/elem — avoid; keeping the Strassen block grid alive between layers avoids all reassembly when not pruning; when pruning you must re-materialise (2 concats = 512/sample/layer), which is already charged above.
- **Meter escape found and NOT used**: `np.ascontiguousarray(FlopscopeArray)` returns a raw `np.ndarray` at 0 charge, exiting the meter entirely. Reporting it as a meter defect; building on it would be meter-gaming.

## 5. Best achievable c and the resulting FOM

Joint optimum over (chunk, depth), with honest re-blocking (512/sample/layer), detection (128-col window, 128/sample/layer) and the model→meter calibration (×1.0244, worst case) applied:

| configuration | chain | whiten | O(1)/smp | pilot | **c** | ×naive | **FOM** | Φ |
|---|---|---|---|---|---|---|---|---|
| naive f32 chain | 4,194,304 | 0 | 0 | 0 | 4,194,304 | 1.000× | 7.910e4 | 2.908e-07 |
| naive + `core.whiten` unfused | 4,194,304 | 262,912 | 1,726 | 0 | 4,458,430 | 0.941× | 8.409e4 | 3.091e-07 |
| + fused whiten, sym-einsum gram | 4,194,304 | 66,304 | 1,726 | 0 | 4,262,590 | 0.984× | 8.039e4 | 2.956e-07 |
| prefix pruning only, chunk 16 | 3,478,021 | 66,304 | 7,044 | 9,568 | 3,561,193 | 1.178× | 6.716e4 | 2.469e-07 |
| Strassen d=6 only | 2,488,000 | 66,304 | 5,078 | 0 | 2,559,638 | 1.639× | 4.827e4 | 1.775e-07 |
| **BEST: chunk 64, d=6, pruned** | **1,957,656** | 66,304 | 4,025 | 4,048 | **2,032,290** | **2.064×** | **3.833e4** | **1.409e-07** |
| runner-up: chunk 32, d=5, pruned | 2,012,817 | 66,304 | 4,135 | 3,726 | 2,087,238 | 2.009× | 3.937e4 | 1.447e-07 |

At c = 2.032e6 the per-MLP budget buys k = 133,839 samples. Accuracy penalty at that c: **zero measurable** — pruning bit-identical, Strassen d=6 ≈ 8e-12 MSE against a 8.6e-6 sampling MSE.

**Against the live submission** (c_live = 4.99e4 / 1.886e-2 = **2,645,811**): gain **1.302×**, FOM 4.99e4 → **3.83e4** (Φ 1.834e-07 → 1.409e-07). That beats the published best (4.2e4) but is **141× above the 272 target.**

**The structural point.** Even granting the report's physically-unavailable "additions free" floor (1,446,577) *and* pruning on top (×0.82 → 1.19e6), FOM = 2.24e4 — still **82× above target**. And a single 256×256 matvec is 130,816 FLOPs, so 32 layers cannot be cheaper than ~4.19e6/32 per layer by any exact method. **The cost axis cannot get within ~80× of the target under any assumption. C must fall by ≳100× regardless of what happens to c.** So: the honest factor here is **1.3×, not 3×**.

## 6. What would have to be true for me to be wrong

1. **If live c really is 1.8e6** rather than the 2.646e6 implied by FOM 4.99e4 ÷ C 1.886e-2, my result is a regression and the live code holds a lever I did not find. But the brief's own three numbers are mutually inconsistent: I measured both named levers and they reproduce the stated factors (pruning 1.164× vs stated 1.17×; Strassen 1.138×/1.288× at d=1/2, bracketing the stated 1.19×) and they compose to 4.194e6/1.34 = **3.13e6**, not 1.8e6. 1.8e6 is also below my metered pruned-Strassen chain floor of 1.96e6.
2. **Alternative-basis Strassen** (Karstadt–Schwartz) cuts per-level additions 15 → 12 via a one-off O(kn log n) basis change. I did not implement it. If the k-dependent additions drop 11 → 8/level, the chain floor moves 2.488e6 → ~2.36e6 and the optimum shifts to d=7, giving c ≈ 1.93e6. **Ceiling ≤1.06×** — worth doing, not worth hoping for.
3. **More einsum symmetry discounts.** I tried `ki,ij->kj` (none) and `ki,kj->ij` (2×). A symmetric-Strassen syrk could shave the 66,304 gram to ~55,000. **Ceiling 0.5% of c.**
4. **Chunks below 8.** Prefix-dead increments are already shrinking (57.3/52.2/47.1/41.5 at 8/16/32/64) and chunk < 2^d is infeasible at the depth that dominates. I did not extrapolate.

Scripts (all absolute): `/private/tmp/claude-501/-Users-binyong-Library-CloudStorage-GoogleDrive-binyongbong1029-gmail-com-My-Drive-HACKATHONS-ARC-White-Box-Estimation-Challenge/8658eb30-d66f-4600-9519-719e14a4a4f2/scratchpad/` — `cst_lib.py` (block-tuple Strassen + exact cost formula), `cst_vec.py` (vectorised Strassen for accuracy), `cst_06_smeter.py` (formula vs meter), `cst_07_acc.py`, `cst_08_chunks.py`, `cst_14_paired.py`, `cst_16_prune.py`, `cst_17_exact.py`, `cst_18_final.py`.