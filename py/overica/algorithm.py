"""Top-level ``overica`` driver function."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.linalg

from .cumulants import gencov, quadricov
from .sdp import sdp_adaptive, sdp_cluster, sdp_semiada


_DEFAULT_OPTS: dict[str, Any] = {"sub": "gencov", "sdp": "semiada"}


def _check_opts(opts: dict[str, Any] | None) -> dict[str, Any]:
    if opts is None or not isinstance(opts, dict):
        return dict(_DEFAULT_OPTS)
    opts = dict(opts)
    opts.setdefault("sub", "gencov")
    opts.setdefault("sdp", "semiada")
    if opts["sub"] not in ("quad", "gencov"):
        print("The opts['sub'] value is changed to gencov")
        opts["sub"] = "gencov"
    if opts["sdp"] not in ("ada", "semiada", "clust"):
        print("The opts['sdp'] value is changed to semiada")
        opts["sdp"] = "semiada"
    if "ctype" in opts and opts["ctype"] not in ("h", "km"):
        print("The opts['ctype'] value is changed to h")
        opts["ctype"] = "h"
    return opts


def _estimate_gencovs(
    X: np.ndarray, s: int, t0: float, rng: np.random.Generator
) -> np.ndarray:
    """Estimate ``s`` mean-zero generalized covariances of ``X``.

    Each column of the returned matrix is ``vec(G(t*omega_i)) - vec(G(0))``,
    where ``G(omega)`` is the generalized covariance evaluated at ``omega``.
    """
    p, _ = X.shape
    C = np.zeros((p * p, s))
    G0 = gencov(X, np.zeros(p))
    for i in range(s):
        omega = rng.standard_normal(p)
        Gi = gencov(X, t0 * omega)
        C[:, i] = Gi - G0
    return C


def overica(
    X: np.ndarray,
    k: int,
    opts: dict[str, Any] | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    verbose: bool = True,
):
    """Overcomplete Independent Component Analysis via SDP.

    Parameters
    ----------
    X : (p, n) ndarray
        Data matrix with observations in columns.
    k : int
        Desired latent dimension. The paper's guarantees require
        ``k < p**2 / 4``.
    opts : dict, optional
        See :func:`overica.overica` docstring in the README for keys and
        defaults.
    rng : numpy.random.Generator or seed, optional
        Source of randomness for generalized covariance sampling and SDP
        initialisation.
    verbose : bool, default True
        If False, suppresses the per-stage progress prints.

    Returns
    -------
    ds_est : (p, k) ndarray
        Estimated atoms (mixing-matrix columns), unit-norm.
    Ds_est : (p*p, k) ndarray
        Estimated rank-one atoms ``vec(d_i d_i^T)`` (column-major).
    times : dict
        Timings for the cumulant, SVD and SDP stages.
    Hs : (p*p, k) ndarray
        Orthonormal basis of the estimated subspace.
    """
    opts = _check_opts(opts)

    if isinstance(rng, np.random.Generator):
        rng_inst = rng
    else:
        rng_inst = np.random.default_rng(rng)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    X = np.asarray(X, dtype=float)

    glob_start = time.perf_counter()

    if opts["sub"] == "quad":
        log("Computing quadricovariance")
        t0 = time.perf_counter()
        C = quadricov(X)
        log(f"Elapsed: {time.perf_counter() - t0:.3f}s")
    else:
        log("Computing generalized covariances")
        t0 = time.perf_counter()
        # MATLAB default: t = 0.05 / sqrt(max(max(abs(cov(X')))))
        covX = np.cov(X)
        t_default = 0.05 / np.sqrt(np.max(np.abs(covX)))
        s = int(opts.get("s", 5)) * k
        t = float(opts.get("t", t_default))
        C = _estimate_gencovs(X, s, t, rng_inst)
        log(f"Elapsed: {time.perf_counter() - t0:.3f}s")

    cum_time = time.perf_counter() - glob_start
    log(f"Cumulant stage: {cum_time:.3f}s")

    log("Computing SVD")
    U, _, _ = scipy.linalg.svd(C, full_matrices=False)
    Hs = U[:, :k]
    svd_time = time.perf_counter() - glob_start - cum_time
    log(f"SVD stage: {svd_time:.3f}s")

    sdp_mode = opts["sdp"]
    ctype = opts.get("ctype")

    if sdp_mode == "clust":
        log("Deflation via clustering")
        ds_est, Ds_est = sdp_cluster(Hs, k, ctype if ctype else "h", rng=rng_inst)
    elif sdp_mode == "ada":
        log("Adaptive deflation")
        ds_est, Ds_est = sdp_adaptive(Hs, k)
    else:  # 'semiada'
        log("Semiadaptive deflation")
        ds_est, Ds_est = sdp_semiada(Hs, k, ctype if ctype else "h", rng=rng_inst)

    sdp_time = time.perf_counter() - glob_start - cum_time - svd_time
    total_time = time.perf_counter() - glob_start
    log(f"SDP stage: {sdp_time:.3f}s")
    log(f"Total: {total_time:.3f}s")

    times = {
        "total_time": total_time,
        "cum_time": cum_time,
        "svd_time": svd_time,
        "sdp_time": sdp_time,
    }
    return ds_est, Ds_est, times, Hs
