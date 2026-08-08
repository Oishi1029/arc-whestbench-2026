# ARC White-Box Estimation Challenge 2026 — Phase 1 solution and research log

Estimating the per-neuron mean activation of the final layer of a bias-free ReLU MLP
(width 256, depth 32, He init) under standard-normal inputs, within an analytical FLOP budget.

**Entrant:** bin_yong_bong · **Phase 1 best public score:** adjusted final-layer **1.730e-07**
(submission #325573), 0/50 public MLPs failed.

The technical write-up is **[WRITEUP.md](WRITEUP.md)** — start there. It is a negative-results,
bounds and methodology contribution: a measured map of where this problem's walls are, with the
experiment that located each one.

## Submission lineage

Each estimator's **docstring carries the measurements for the mechanism it adds**, so a claim in
the write-up and the code that produced it sit next to each other.

| file | adds | public score |
|---|---|---|
| `estimators/wmc.py` | whitened + antithetic Monte Carlo | 3.405e-07 (#323440) |
| `estimators/wmc2.py` | + exact dead-column pruning | 2.700e-07 (#323892) |
| `estimators/wmc3.py` | + a-priori lead-block masking | 2.240e-07 (#324076) |
| `estimators/wmc4.py` | + Strassen multiplication | 1.834e-07 (#324358) |
| `estimators/wmc5.py` | + layer-1 mean & covariance anchor | 1.860e-07 (#325572) |
| `estimators/anchor-cheap.py` | layer-1 **full**-covariance anchor | 1.920e-07 (#325574) |
| **`estimators/sweep.py`** | wmc4 + four exact cost identities | **1.730e-07 (#325573)** |
| **`estimators/merge.py`** | sweep × anchor, both mechanisms | **1.803e-07 (#326022)** |

Paired at n = 1000 against `sweep.py`, `merge.py` measures 1.0619× at `t` = +3.67. On the 50-MLP
public split it scores 1.027× *behind* it. Write-up §8.2 explains why we believe the first number.

The two nominated for the private re-evaluation are `sweep.py` and `merge.py`.

## Measurement harness

Accuracy is measurable offline: the public dataset ships its own ground truth, `final_means`,
baked at 1e9 samples per MLP.

- `harness/offline_bench.py` — score an estimator against the shipped ground truth (~60 s for
  100 MLPs, against the official harness's ~25 min for 1,000)
- `harness/runoff.py` — paired run-off with common random numbers at equal FLOP cost
- `harness/scale_mode.py` — the ICC-across-streams decomposition of §7.3

Plus `experiments/` — the drivers behind specific write-up sections: `mehler.py` and `diag.py`
(§7.4, the Mehler-series negative), `gap_cost.py` (§5.2 and §6.2, the cost identities and the
op log), `variance.py` (§4.4, bit-identical to the shipped estimator with radial marginalisation
off) and `replicate.py` (§5.3, the Strassen replication).

Validated against the official harness to **0.1%** on an estimator ratio and **0.02%** on a FLOP
count. See write-up §8.

## Research record

- `research/RESEARCH.md` — the chronological experimental log, **including every conclusion the
  write-up retracts**
- `research/LITERATURE.md` — companion paper (arXiv:2605.05179) and forum analysis
- `research/probes/` — ten adversarial probe reports; the primary source for write-up §§2–4

## Reproducing

Requires the announced grader stack: Python 3.10.20, numpy 2.2.6, flopscope 0.10.0,
whestbench 0.14.0, CPU only.

```bash
whest run --estimator estimators/merge.py \
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 \
  --split full --n-mlps 200 --runner subprocess
```

Write-up Appendix A maps every number in the document to its script, split, sample count and
comparison design — and states plainly which ones are not re-runnable, and why.

## AI assistance

Developed with substantial assistance from Anthropic's Claude for code generation, experiment
execution, analysis and drafting, as permitted under competition rules §5.7. All design decisions,
all judgements about what to ship, and responsibility for the accuracy of every claim are the
participant's. See write-up Appendix C.

## Licence

MIT — see [LICENSE](LICENSE).
