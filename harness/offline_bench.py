"""Fast offline benchmark against the REAL competition MLPs and REAL ground truth.

Why this exists
---------------
`whest run --split full` takes ~25 minutes. But the public dataset ships the ground truth
itself — every row carries `final_means` (the scored layer) and `all_layer_means`, baked at
n_samples = 1e9 per MLP. So an estimator's ACCURACY can be measured offline in seconds, in
plain numpy, against the exact MLPs and exact targets the harness uses.

That matters because accuracy is the hard thing to iterate on; FLOP cost is deterministic and
computable in closed form from the flopscope 0.10.0 cost model (calibrated in work/RESEARCH.md
§6). So: iterate here, then confirm the winner once with the real harness.

This also fixes a methodological weakness in the first round of experiments
(work/experiments/*), which used self-generated MLPs and self-computed Monte-Carlo ground truth
at 2e6 samples — noisier targets, and not the competition's actual networks.

Usage
-----
    from offline_bench import load_split, score

    mlps = load_split("mini")            # or "full" (1000 MLPs)
    res  = score(my_estimator_fn, mlps, flops_per_mlp=2.6e10)
    print(res)

`estimator_fn(weights, rng) -> (width,) array` returns the FINAL-layer mean prediction.
`weights` is a list of `depth` float32 (width, width) arrays.
"""

from __future__ import annotations

import glob
import math

import numpy as np

FLOP_BUDGET = 2.72e11
FREE_LINE = 0.10 * FLOP_BUDGET
_SNAP = ("/Users/binyong/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026"
         "/snapshots/*/prepared/{split}/*.arrow")


def load_split(split: str = "mini", limit: int | None = None):
    """Load (weights, final_means, all_layer_means, seed) for each MLP in a split."""
    import pyarrow as pa

    out = []
    for path in sorted(glob.glob(_SNAP.format(split=split))):
        with pa.memory_map(path, "rb") as src:
            tbl = pa.ipc.open_stream(src).read_all()
        for row in tbl.to_pylist():
            out.append({
                "name": row["mlp_name"],
                "seed": int(row["mlp_seed"]),
                "weights": [np.asarray(w, dtype=np.float32) for w in row["weights"]],
                "final_means": np.asarray(row["final_means"], dtype=np.float64),
                "all_layer_means": np.asarray(row["all_layer_means"], dtype=np.float64),
            })
            if limit is not None and len(out) >= limit:
                return out
    return out


def adjusted_score(mse: float, flops_per_mlp: float) -> float:
    """The official metric: mse * max(0.1, effective_compute / flop_budget)."""
    return mse * max(0.1, flops_per_mlp / FLOP_BUDGET)


def mc_flops(k: int, width: int, depth: int, extra_passes: float = 1.0) -> float:
    """float32 Monte-Carlo cost under flopscope 0.10.0: 2 FLOPs/MAC, calibrated in RESEARCH.md §6.

    `extra_passes` counts non-layer k*n^2 passes (e.g. 1.0 for one Gram matrix).
    """
    w = float(width)
    return 2.0 * w * w * (float(depth) + extra_passes) * k + 2.0 * w * float(depth) * k


def samples_for(target_frac: float, width: int, depth: int, extra_passes: float = 1.0,
                fixed: float = 13.0 * 256 ** 3) -> int:
    """Largest k whose predicted cost lands on `target_frac` of the budget."""
    per = mc_flops(1, width, depth, extra_passes)
    return max(1, int((target_frac * FLOP_BUDGET - fixed) / per))


def score(estimator_fn, mlps, flops_per_mlp: float, seed_offset: int = 0, verbose: bool = True):
    """Mean final-layer MSE and adjusted score over `mlps`."""
    errs = []
    for i, m in enumerate(mlps):
        rng = np.random.default_rng(m["seed"] % (2 ** 63) + seed_offset)
        pred = np.asarray(estimator_fn(m["weights"], rng), dtype=np.float64)
        errs.append(float(np.mean((pred - m["final_means"]) ** 2)))
    mse = float(np.mean(errs))
    res = {
        "n_mlps": len(mlps),
        "final_layer_mse": mse,
        "adjusted": adjusted_score(mse, flops_per_mlp),
        "utilisation": flops_per_mlp / FLOP_BUDGET,
        "worst_mlp_mse": float(np.max(errs)),
        "best_mlp_mse": float(np.min(errs)),
    }
    if verbose:
        print(f"  n={res['n_mlps']:4d}  MSE={mse:.4e}  adjusted={res['adjusted']:.4e}  "
              f"util={res['utilisation']:.4f}  worst={res['worst_mlp_mse']:.3e}")
    return res


# --------------------------------------------------------------------------
# Reference estimators, for calibration against the known harness numbers.
# --------------------------------------------------------------------------

def whitened_mc(k: int):
    """The current submission's algorithm, in plain numpy. Matches work/mine/wmc.py."""
    def fn(W, rng):
        n = W[0].shape[0]
        x = rng.standard_normal((k, n), dtype=np.float32)
        x = x - x.mean(0)
        g = (x.T @ x).astype(np.float64) / k
        ev, U = np.linalg.eigh(g)
        t = ((U * np.maximum(ev, 1e-8) ** -0.5) @ U.T).astype(np.float32)
        y = np.maximum(x @ (t @ W[0]), 0.0)
        for w in W[1:]:
            y = np.maximum(y @ w, 0.0)
        return y.mean(0)
    return fn


def plain_mc(k: int):
    def fn(W, rng):
        n = W[0].shape[0]
        y = rng.standard_normal((k, n), dtype=np.float32)
        for w in W:
            y = np.maximum(y @ w, 0.0)
        return y.mean(0)
    return fn


if __name__ == "__main__":
    mlps = load_split("mini")
    n, d = mlps[0]["weights"][0].shape[0], len(mlps[0]["weights"])
    print(f"loaded {len(mlps)} MLPs, width={n} depth={d}")
    print(f"ground truth is the dataset's own final_means (baked at n_samples=1e9)")
    print()

    k_w = samples_for(0.095, n, d, extra_passes=1.0)
    k_p = samples_for(0.095, n, d, extra_passes=0.0)
    print(f"whitened MC   k={k_w:,}")
    score(whitened_mc(k_w), mlps, mc_flops(k_w, n, d, 1.0) + 13.0 * n ** 3)
    print(f"plain MC      k={k_p:,}")
    score(plain_mc(k_p), mlps, mc_flops(k_p, n, d, 0.0))
    print()
    print("Harness reference for work/mine/wmc.py on this same split: adjusted 3.76e-07.")
    print("If whitened MC above is close to that, the offline harness is calibrated.")
