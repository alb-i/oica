"""Synthetic data sampling utilities (ports of the ``sampling/`` directory)."""

from __future__ import annotations

import numpy as np


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def sample_mixing_matrix(p: int, k: int, *, rng=None) -> np.ndarray:
    """Sample a ``(p, k)`` Gaussian mixing matrix with unit-norm columns."""
    rng = _as_rng(rng)
    ds = rng.standard_normal((p, k))
    ds /= np.linalg.norm(ds, axis=0, keepdims=True)
    return ds


def sample_orthogonal_matrix(k: int, *, rng=None) -> np.ndarray:
    """Sample an orthogonal ``(k, k)`` matrix via the QR of a random matrix."""
    rng = _as_rng(rng)
    X = rng.random((k, k))
    Q, _ = np.linalg.qr(X)
    return Q


def sample_from_ica_with_uniform_sources(
    ds: np.ndarray, N: int, *, rng=None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample data from an ICA model with the original paper's source distribution.

    The reference MATLAB implementation uses
    ``alpha = rand(k, n) .* abs(randn(k, n))`` and ``x = ds * alpha``. This is
    a slightly heavy-tailed, non-Gaussian source distribution suitable for
    benchmarking ICA recovery.
    """
    rng = _as_rng(rng)
    ds = np.asarray(ds, dtype=float)
    p, k = ds.shape

    batch = 1000
    times = N // batch
    rest = N % batch

    X = np.empty((p, N))
    Alpha = np.empty((k, N))

    def _sample_batch(n):
        a = rng.random((k, n)) * np.abs(rng.standard_normal((k, n)))
        return ds @ a, a

    for i in range(times):
        x, a = _sample_batch(batch)
        sl = slice(i * batch, (i + 1) * batch)
        X[:, sl] = x
        Alpha[:, sl] = a

    if rest > 0:
        x, a = _sample_batch(rest)
        sl = slice(times * batch, times * batch + rest)
        X[:, sl] = x
        Alpha[:, sl] = a

    return X, Alpha
