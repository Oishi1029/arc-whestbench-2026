"""Error-source diagnostic for the WhestBench task.

E[y_l,i] = E[ReLU(z_l,i)] depends ONLY on the marginal law of z_l,i.
If z_l,i were exactly Gaussian(mu_i, sig_i), then E[ReLU(z)] = mu*Phi(a) + sig*phi(a), a = mu/sig.

So total error decomposes into
  (a) error in the PREDICTED marginal moments (mu_i, sig_i)   <- what covariance propagation gets wrong
  (b) intrinsic NON-GAUSSIANITY of z_l,i                      <- the ceiling of ANY Gaussian-based method

This script measures (b) directly, using Monte Carlo to get the TRUE marginal moments,
then asks: how well does the Gaussian formula fed with TRUE moments predict the TRUE mean?
Whatever error remains is irreducible for Gaussian methods -> tells us where to spend effort.
"""
import math
import numpy as np

rng = np.random.default_rng(0)
N, D = 256, 32
NS = 400_000  # MC samples for the reference


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def relu_mean_gauss(mu, sig):
    """E[ReLU(z)] for z ~ N(mu, sig^2)."""
    sig = np.maximum(sig, 1e-30)
    a = mu / sig
    return mu * norm_cdf(a) + sig * norm_pdf(a)


def build_mlp(seed):
    r = np.random.default_rng(seed)
    return [r.normal(0.0, math.sqrt(2.0 / N), size=(N, N)).astype(np.float64) for _ in range(D)]


def mc_layer_stats(W, nsamp, seed=1, batch=20_000):
    """True per-layer pre-activation mean/std and post-ReLU mean, by Monte Carlo."""
    r = np.random.default_rng(seed)
    s1 = np.zeros((D, N))          # sum of z
    s2 = np.zeros((D, N))          # sum of z^2
    sy = np.zeros((D, N))          # sum of relu(z)
    n = 0
    while n < nsamp:
        b = min(batch, nsamp - n)
        y = r.standard_normal((b, N))
        for l in range(D):
            z = y @ W[l]
            s1[l] += z.sum(0)
            s2[l] += (z * z).sum(0)
            y = np.maximum(z, 0.0)
            sy[l] += y.sum(0)
        n += b
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu, np.sqrt(var), sy / n


def cov_prop(W):
    """The shipped examples/03 algorithm, in plain numpy (gain heuristic)."""
    mu = np.zeros(N)
    cov = np.eye(N)
    rows_mu, rows_sig = [], []
    out = []
    for l in range(D):
        mu_pre = W[l].T @ mu
        cov_pre = W[l].T @ cov @ W[l]
        var_pre = np.maximum(np.diag(cov_pre), 1e-12)
        sig_pre = np.sqrt(var_pre)
        a = mu_pre / sig_pre
        pdf, cdf = norm_pdf(a), norm_cdf(a)
        mu_new = mu_pre * cdf + sig_pre * pdf
        ez2 = (mu_pre * mu_pre + var_pre) * cdf + mu_pre * sig_pre * pdf
        var_post = np.maximum(ez2 - mu_new * mu_new, 0.0)
        gain = np.where(sig_pre > 1e-12, cdf, 0.0)
        cov = np.outer(gain, gain) * cov_pre
        np.fill_diagonal(cov, var_post)
        rows_mu.append(mu_pre.copy())
        rows_sig.append(sig_pre.copy())
        out.append(mu_new.copy())
        mu = mu_new
    return np.array(out), np.array(rows_mu), np.array(rows_sig)


W = build_mlp(seed=7)
print(f"MC reference: {NS:,} samples, width={N} depth={D}")
mu_t, sig_t, y_t = mc_layer_stats(W, NS)
pred, mu_p, sig_p = cov_prop(W)

# (b) THE CEILING: Gaussian formula fed with the TRUE marginal moments
y_gauss_oracle = relu_mean_gauss(mu_t, sig_t)

L = D - 1  # the scored layer
mse = lambda a, b: float(np.mean((a - b) ** 2))

print()
print("=" * 68)
print("FINAL LAYER (the only scored layer)")
print("=" * 68)
print(f"  cov-prop MSE vs MC truth               : {mse(pred[L], y_t[L]):.4e}")
print(f"  GAUSSIAN-ORACLE MSE (true moments in)  : {mse(y_gauss_oracle[L], y_t[L]):.4e}   <- floor for any Gaussian method")
print()
print(f"  relative error in predicted mu (final) : {np.abs(mu_p[L]-mu_t[L]).mean()/ (np.abs(mu_t[L]).mean()+1e-30):.4%}")
print(f"  relative error in predicted sigma      : {np.abs(sig_p[L]-sig_t[L]).mean()/ (sig_t[L].mean()+1e-30):.4%}")
print()
print("per-layer: covprop_mse | gauss_oracle_mse | mu_relerr | sig_relerr")
for l in range(D):
    print(f"  L{l:02d}  {mse(pred[l], y_t[l]):.3e}   {mse(y_gauss_oracle[l], y_t[l]):.3e}   "
          f"{np.abs(mu_p[l]-mu_t[l]).mean()/(np.abs(mu_t[l]).mean()+1e-30):8.4%}   "
          f"{np.abs(sig_p[l]-sig_t[l]).mean()/(sig_t[l].mean()+1e-30):8.4%}")

# MC noise floor on the reference itself (so we don't chase noise)
_, _, y_t2 = mc_layer_stats(W, NS // 4, seed=99)
print()
print(f"MC noise floor of this diagnostic at final layer (NS/4 vs NS): {mse(y_t2[L], y_t[L]):.3e}")
