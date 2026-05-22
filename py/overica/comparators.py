"""Baseline / comparator algorithms used in the reproduction scripts.

* :func:`fourier_pca` is a port of ``comparison/fpca/fourier_pca2.m``.

The MATLAB code base also references FOOBI (Lieven De Lathauwer's algorithm),
but the actual implementation is *not* included in this repository (only a
README pointing at the author). We therefore do not provide a Python FOOBI;
the reproduction scripts skip it gracefully when it isn't installed.
"""

from __future__ import annotations

import numpy as np

from .cumulants import genquadricov


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def fourier_pca(X: np.ndarray, k: int, *, rng=None) -> np.ndarray:
    """Fourier PCA (FPCA) overcomplete ICA baseline.

    Parameters
    ----------
    X : (p, n) ndarray
        Data with observations in columns.
    k : int
        Number of latent components to recover.
    rng : numpy.random.Generator or seed, optional
        Source of randomness for the projection direction.

    Returns
    -------
    ds : (p, k) ndarray
        Estimated unit-norm atoms.
    """
    rng = _as_rng(rng)
    X = np.asarray(X, dtype=float)
    p, _ = X.shape

    u = rng.standard_normal(p)
    u = u / np.linalg.norm(u)

    Q = genquadricov(X, u)
    Q1 = Q.real
    Q2 = Q.imag

    W, S, _ = np.linalg.svd(Q1, full_matrices=False)
    W = W[:, :k]

    q1 = W.T @ Q1 @ W
    q2 = W.T @ Q2 @ W

    # MATLAB ``q1/q2`` is right-division: q1 * inv(q2). Solve M q2 = q1.
    M = np.linalg.solve(q2.T, q1.T).T

    _, V = np.linalg.eig(M)
    C = W @ V  # (p, k), complex

    for j in range(k):
        c = C[:, j]
        a = c.real
        b = c.imag
        denom = float(np.sum(a * a - b * b))
        num = -2.0 * float(np.sum(a * b))
        # Avoid the degenerate denom == 0 case (atan handles inf gracefully,
        # but we still want a sane phase).
        theta = np.arctan2(num, denom) / 2.0 if denom != 0 else (np.pi / 4.0)
        while theta < 0:
            theta += np.pi
        while theta > 2 * np.pi:
            theta -= np.pi
        tmp = (np.exp(1j * theta) * c).real
        norm = np.linalg.norm(tmp)
        if norm > 0:
            C[:, j] = tmp / norm

    ds = np.zeros((p, k))
    for j in range(k):
        c = C[:, j].real
        rmat = c.reshape(p, p, order="F")
        u_svd, _, _ = np.linalg.svd(rmat, full_matrices=False)
        ds[:, j] = u_svd[:, 0]

    return ds
