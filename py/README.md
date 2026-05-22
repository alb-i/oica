# OverICA (Python port)

Python port of the OverICA algorithm:

> Anastasia Podosinnikova, Amelia Perry, Alexander Wein, Francis Bach,
> Alexandre d'Aspremont, David Sontag. *Overcomplete Independent Component
> Analysis via SDP.* AISTATS 2019.

Please cite the paper if you use this code for research.

This is a straightforward translation of the original MATLAB implementation
(`../overica.m` and friends) to NumPy/SciPy/scikit-learn. It is not a binary
clone of the MATLAB code (no `mex` files, no proprietary toolboxes), but the
algorithm and default parameters match.

## Install

```bash
cd py
python -m pip install -r requirements.txt
```

The package is a plain folder; either run scripts from the `py/` directory or
add it to your `PYTHONPATH`.

## Quick start

```python
import numpy as np
from overica import overica, sample_mixing_matrix, sample_from_ica_with_uniform_sources, a_error

rng = np.random.default_rng(0)
p, k, n = 6, 10, 50_000          # overcomplete: k > p
ds = sample_mixing_matrix(p, k, rng=rng)
X, _ = sample_from_ica_with_uniform_sources(ds, n, rng=rng)

ds_est, Ds_est, times, Hs = overica(X, k, rng=rng)
print("cos-error:", a_error(ds_est, ds)[0])
```

A runnable demo lives in `examples/demo.py`.

## API

```python
overica(X, k, opts=None, rng=None)
```

* `X` – `(p, n)` data matrix with observations in columns.
* `k` – desired latent dimension (`k < p**2 / 4` for the paper guarantees).
* `opts` – optional dict with any of:
    * `sub`: `'gencov'` (default) or `'quad'` – subspace estimator
      (generalized covariance vs. quadricovariance/fourth-order cumulant).
    * `s`: number of generalized covariances is `s * k` (default `s = 5`).
    * `t`: scale parameter for `'gencov'`
      (default `0.05 / sqrt(max|cov(X^T)|)`).
    * `sdp`: `'semiada'` (default), `'ada'`, or `'clust'` – deflation strategy.
    * `ctype`: `'h'` (default, hierarchical) or `'km'` (k-means++) – clustering
      type used by `'clust'` and `'semiada'`.
* `rng` – optional `numpy.random.Generator` for reproducibility.

Returns `(ds_est, Ds_est, times, Hs)`:

* `ds_est` – `(p, k)` estimated mixing matrix.
* `Ds_est` – `(p*p, k)` matricized rank-one atoms (`vec(d_i d_i^T)`).
* `times` – dict with `total_time`, `cum_time`, `svd_time`, `sdp_time`.
* `Hs` – `(p*p, k)` orthonormal basis of the estimated subspace.

## Layout

* `overica/algorithm.py` – top-level `overica` driver.
* `overica/cumulants.py` – `gencov`, `quadricov`, `genquadricov`.
* `overica/sdp.py` – FISTA solver and the three deflation variants.
* `overica/sampling.py` – synthetic data sampling utilities.
* `overica/evaluation.py` – error metrics (`a_error`, `f_error`,
  `evaluation_perf`, `evaluation_recovery`).
* `examples/demo.py` – end-to-end demo.

## Implementation notes

Highlights of the translation from MATLAB to Python:

* **Pure NumPy/SciPy stack.** No `mex` files and no proprietary toolboxes.
  `sklearn.cluster.KMeans` is used for the `km` clustering option, and
  `scipy.cluster.hierarchy.linkage`/`fcluster` for the default hierarchical
  (`h`) clustering — matching MATLAB's `clusterdata` (single linkage,
  Euclidean distance, max-clust criterion).
* **Hungarian matching.** `scipy.optimize.linear_sum_assignment` replaces the
  bundled `HungarianBipartiteMatching.m` for atom matching in the evaluation
  module.
* **MATLAB-compatible flattening.** All matrix↔vector reshapes use Fortran
  (column-major) order to stay bit-compatible with MATLAB's `M(:)` and
  `reshape(v, p, p)` conventions, so the column indexing of `Ds_est`, `Hs`,
  and friends matches the reference implementation exactly.
* **No native code for `quadricov`.** The original `quadricov_in.cpp` MEX is
  replaced by a pure-NumPy implementation built on the rank-one outer-product
  trick (`V[:, i] = vec(x_i x_i^T)`, then `M4 = V V^T / n`) plus three
  `einsum` permutation terms.
* **Reproducible randomness.** Functions that draw random numbers accept an
  `rng=` argument (a `numpy.random.Generator` or a seed), so runs are
  reproducible — something the MATLAB version doesn't easily expose.
* **`opts` as a dict.** Unknown or invalid keys fall back to the same
  defaults that the MATLAB `check_opts` helper enforces.

### Verification

* `examples/demo.py` runs end-to-end (`p=6, k=8, n=50_000`) in roughly 1.4 s
  on a modern laptop.
* All four major code paths have been exercised: `sub ∈ {gencov, quad}` ×
  `sdp ∈ {semiada, ada, clust}`. The undercomplete sanity check (`p=k=5`)
  recovers atoms to cos-error ≈ 0.04.
* `gencov(X, 0)` matches the empirical covariance of `X` to machine
  precision, and `quadricov(X)` is symmetric with the expected shape and
  shrinks towards zero for Gaussian data as the sample size grows (its
  population value is zero).

### Try it

```bash
cd py
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python examples/demo.py
```
