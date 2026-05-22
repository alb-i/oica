"""Atom-recovery error metrics (port of the MATLAB ``helpers/`` evaluation)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalize_2(ds: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(ds, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return ds / norms


def _update_ds_est(ds_est: np.ndarray, ds: np.ndarray, perm: np.ndarray):
    ds_est = ds_est[:, perm]
    signs = np.sign(np.einsum("ij,ij->j", ds_est, ds))
    signs[signs == 0] = 1.0
    ds_est = ds_est * signs
    return ds_est, {"ds_est": ds_est, "perm": perm, "signs": signs}


def _perf_l2(ds_est: np.ndarray, ds: np.ndarray):
    k = ds.shape[1]
    F = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            sij = np.sign(ds_est[:, i] @ ds[:, j])
            if sij == 0:
                sij = 1.0
            F[i, j] = np.linalg.norm(sij * ds_est[:, i] - ds[:, j])
    row, col = linear_sum_assignment(F)
    cost = F[row, col].sum()
    perm = col[np.argsort(row)]
    _, perfout = _update_ds_est(ds_est, ds, perm)
    perf = cost / k / np.sqrt(2.0)
    return perf, perfout


def _perf_fro(ds_est: np.ndarray, ds: np.ndarray):
    F = -np.abs(ds_est.T @ ds)
    row, col = linear_sum_assignment(F)
    perm = col[np.argsort(row)]
    ds_est, perfout = _update_ds_est(ds_est, ds, perm)
    perf = np.linalg.norm(ds_est - ds, "fro") ** 2 / np.linalg.norm(ds, "fro") ** 2
    return float(perf), perfout


def _perf_l1(ds_est: np.ndarray, ds: np.ndarray):
    k = ds.shape[1]
    F = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            sij = np.sign(ds_est[:, i] @ ds[:, j])
            if sij == 0:
                sij = 1.0
            F[i, j] = np.sum(np.abs(sij * ds_est[:, i] - ds[:, j]))
    row, col = linear_sum_assignment(F)
    cost = F[row, col].sum()
    perm = col[np.argsort(row)]
    _, perfout = _update_ds_est(ds_est, ds, perm)
    perf = cost / k / 2.0
    return float(perf), perfout


def _perf_cos(ds_est: np.ndarray, ds: np.ndarray):
    k = ds.shape[1]
    norms_e = np.linalg.norm(ds_est, axis=0)
    norms_d = np.linalg.norm(ds, axis=0)
    cosmat = np.abs(ds_est.T @ ds) / np.outer(np.where(norms_e == 0, 1.0, norms_e),
                                              np.where(norms_d == 0, 1.0, norms_d))
    cosmat = np.clip(cosmat, 0.0, 1.0)
    F = np.arccos(cosmat)
    row, col = linear_sum_assignment(F)
    cost = F[row, col].sum()
    perm = col[np.argsort(row)]
    _, perfout = _update_ds_est(ds_est, ds, perm)
    perf = 2.0 * cost / np.pi / k
    return float(perf), perfout


_PERF_FNS = {1: _perf_l1, 2: _perf_l2, 3: _perf_fro, 4: _perf_cos}


def evaluation_perf(ds_est: np.ndarray, ds: np.ndarray, perf_type: int = 2):
    """Atom-recovery error after optimal sign-aware bipartite matching.

    ``perf_type`` selects the cost metric:

    * 1 – L1
    * 2 – L2 (default)
    * 3 – squared Frobenius
    * 4 – cosine / angular
    """
    if ds_est.shape != ds.shape:
        raise ValueError("ds_est and ds must have matching shapes")
    if perf_type not in _PERF_FNS:
        raise ValueError(f"unknown perf_type: {perf_type}")
    ds = _normalize_2(np.asarray(ds, dtype=float))
    ds_est = _normalize_2(np.asarray(ds_est, dtype=float))
    return _PERF_FNS[perf_type](ds_est, ds)


def a_error(ds_est: np.ndarray, ds: np.ndarray):
    """Angular (cosine) error, scaled to ``[0, 1]``."""
    return evaluation_perf(ds_est, ds, perf_type=4)


def f_error(ds_est: np.ndarray, ds: np.ndarray):
    """Squared Frobenius error (relative)."""
    return evaluation_perf(ds_est, ds, perf_type=3)


def evaluation_recovery(ds_est: np.ndarray, ds: np.ndarray, th: float):
    """Count how many atoms of ``ds`` are recovered within angular threshold ``th``.

    Returns ``(nrec, recov, perfout)`` where ``recov[i]`` is ``1`` if at least
    ``i+1`` atoms are recovered. Matches the MATLAB ``evaluation_recovery``.
    """
    ds = _normalize_2(np.asarray(ds, dtype=float))
    ds_est = _normalize_2(np.asarray(ds_est, dtype=float))
    k = ds.shape[1]
    norms_e = np.linalg.norm(ds_est, axis=0)
    norms_d = np.linalg.norm(ds, axis=0)
    cosmat = np.abs(ds_est.T @ ds) / np.outer(np.where(norms_e == 0, 1.0, norms_e),
                                              np.where(norms_d == 0, 1.0, norms_d))
    cosmat = np.clip(cosmat, 0.0, 1.0)
    F = np.arccos(cosmat)
    row, col = linear_sum_assignment(F)
    perm = col[np.argsort(row)]
    _, perfout = _update_ds_est(ds_est, ds, perm)

    test = np.empty(k)
    for i in range(k):
        c = abs(ds_est[:, perm[i]] @ ds[:, i]) / (
            np.linalg.norm(ds_est[:, perm[i]]) * np.linalg.norm(ds[:, i])
        )
        test[i] = np.arccos(np.clip(c, 0.0, 1.0))

    nrec = int(np.sum(test < th))
    # MATLAB: recov[i] = 1 iff at least i+1 atoms have error below threshold;
    # since the criterion depends only on the *total* count, this collapses to
    # a step function with ``nrec`` leading ones.
    recov = (nrec >= np.arange(1, k + 1)).astype(int)
    return nrec, recov, perfout
