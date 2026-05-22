"""Cumulant estimators used by OverICA.

All functions operate on data matrices ``X`` of shape ``(p, n)`` (observations
in columns), matching the original MATLAB convention.

Throughout, ``vec(M)`` means column-major flattening (MATLAB's ``M(:)``).
"""

from __future__ import annotations

import numpy as np


def _vec_F(M: np.ndarray) -> np.ndarray:
    """Column-major (Fortran-order) flatten, matching MATLAB ``M(:)``."""
    return np.asarray(M).reshape(-1, order="F")


def gencov(X: np.ndarray, omega: np.ndarray | float) -> np.ndarray:
    """Generalized covariance at point ``omega``, returned as ``vec(C)``.

    Parameters
    ----------
    X : (p, n) ndarray
        Data with observations in columns.
    omega : (p,) ndarray or scalar
        Evaluation point. If a scalar ``a`` is given, it is interpreted as
        ``a * ones(p) / p`` (matching the MATLAB behaviour).

    Returns
    -------
    c : (p*p,) ndarray
        Column-major vectorization of the generalized covariance matrix
        ``E_omega[X X^T] - E_omega[X] E_omega[X]^T``.
    """
    X = np.asarray(X, dtype=float)
    p, _ = X.shape

    omega = np.asarray(omega, dtype=float)
    if omega.ndim == 0 or omega.size == 1:
        omega = float(omega) * np.ones(p) / p
    omega = omega.reshape(p)

    proj = X.T @ omega                       # (n,)
    eproj = np.exp(proj)                     # (n,)
    s = eproj.sum()

    mean_omega = (X @ eproj) / s             # (p,)
    C = (X * eproj) @ X.T / s                # (p, p)
    C = C - np.outer(mean_omega, mean_omega)
    return _vec_F(C)


def quadricov(X: np.ndarray) -> np.ndarray:
    """Fourth-order cumulant (quadricovariance) of centred data.

    Implements the same quantity as the original ``quadricov_in.cpp`` MEX
    routine: the data is first centred, then the matrix ``Q`` of size
    ``(p*p, p*p)`` is returned with

    ``Q[a + b*p, c + d*p] = E[x_a x_b x_c x_d] - C_{ab} C_{cd}
                            - C_{ac} C_{bd} - C_{ad} C_{bc}``

    where ``C = X X^T / n`` is the (biased) covariance of the centred data.
    """
    X = np.asarray(X, dtype=float)
    p, n = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)

    # V[:, i] = vec(x_i x_i^T) in column-major (F) order.
    # Build (p, p, n) outer products then flatten the first two axes in F order.
    outers = Xc[:, None, :] * Xc[None, :, :]          # (p, p, n)
    V = outers.reshape(p * p, n, order="F")           # (p*p, n)

    M4 = (V @ V.T) / n                                # (p*p, p*p)
    C = (Xc @ Xc.T) / n                                # (p, p)

    # Subtract the three product-of-covariances terms.
    vecC = _vec_F(C)
    term1 = np.outer(vecC, vecC)                       # C_{ab} C_{cd}

    # Compute C_{ac} C_{bd} and C_{ad} C_{bc} via index permutations of a
    # (p, p, p, p) tensor, then flatten back to (p*p, p*p) in F order.
    C_ac_bd = np.einsum("ac,bd->abcd", C, C).reshape(p * p, p * p, order="F")
    C_ad_bc = np.einsum("ad,bc->abcd", C, C).reshape(p * p, p * p, order="F")

    return M4 - term1 - C_ac_bd - C_ad_bc


def genquadricov(X: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Fourth-order generalized cumulant at point ``u`` (complex-valued).

    Matricized in MATLAB's column-major convention:
    ``Q[(i1-1)*p + i2, (i3-1)*p + i4] = CUM(i1, i2, i3, i4)``, i.e. the first
    two and last two dimensions are flattened in column-major order.
    """
    X = np.asarray(X, dtype=float)
    p, N = X.shape
    u = np.asarray(u, dtype=float).reshape(p)

    Xu = X.T @ u                                      # (N,)
    expXu = np.exp(1j * Xu)                           # (N,)
    M0 = expXu.sum() / N
    M1 = (X @ expXu) / (N * M0)                       # (p,)

    Xc = X - M1[:, None]                              # "generalized zero-mean"

    # Weighted outer products w_n (x_n x_n^T)
    outers = Xc[:, None, :] * Xc[None, :, :]          # (p, p, N), real
    W = expXu                                          # (N,), complex
    M2 = np.tensordot(outers, W, axes=([2], [0])) / (N * M0)  # (p, p), complex

    V = outers.reshape(p * p, N, order="F")           # (p*p, N), real
    # M4 = (1/(N*M0)) * sum_n w_n vec(x_n x_n^T) vec(x_n x_n^T)^T
    M4 = (V * W[None, :]) @ V.T / (N * M0)            # (p*p, p*p), complex

    # Subtract the three M2 cross terms.
    vecM2 = M2.reshape(-1, order="F")
    term1 = np.outer(vecM2, vecM2)                    # M2_{ab} M2_{cd}
    M2_ac_bd = np.einsum("ac,bd->abcd", M2, M2).reshape(p * p, p * p, order="F")
    M2_ad_bc = np.einsum("ad,bc->abcd", M2, M2).reshape(p * p, p * p, order="F")

    return M4 - term1 - M2_ac_bd - M2_ad_bc
