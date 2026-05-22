"""SDP-based atom extraction and deflation for OverICA.

This module ports the contents of the MATLAB ``sdp/`` directory:

* :func:`solve_relaxation_mezcal_approx_fista` – inner FISTA solver for the
  trace-1 PSD relaxation.
* :func:`majorize_minimize` – outer majorize-minimize loop driving the FISTA
  solver towards a rank-one solution.
* :func:`extract_basis` – orthonormalise the subspace and produce its
  orthogonal complement.
* :func:`extract_largest_eigenvector` – top |eigenvalue| eigenvector of a
  (symmetric) matrix.
* :func:`adaptive_deflation` – greedy deflation using SVD of accumulated atoms.
* :func:`cluster_Dss` – sign-aligned clustering of candidate atoms.
* :func:`sdp_cluster`, :func:`sdp_adaptive`, :func:`sdp_semiada` – the three
  deflation strategies exposed by ``overica``.
* :func:`approx_ds_from_Ds`, :func:`Ds_from_ds` – helpers between the ``ds``
  (atoms in columns) and ``Ds = vec(d_i d_i^T)`` representations.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _vec_F(M: np.ndarray) -> np.ndarray:
    return np.asarray(M).reshape(-1, order="F")


def _mat_F(v: np.ndarray, p: int) -> np.ndarray:
    return np.asarray(v).reshape(p, p, order="F")


def _as_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def proj_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex ``{w >= 0, sum w <= 1}``.

    Algorithm of Duchi et al., as used by the original MATLAB code.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    v = np.where(v > 0, v, 0.0)
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    idx = np.arange(1, len(u) + 1)
    cond = u > (sv - 1.0) / idx
    if not np.any(cond):
        return np.zeros_like(v)
    rho = np.max(np.flatnonzero(cond)) + 1
    theta = max(0.0, (sv[rho - 1] - 1.0) / rho)
    return np.maximum(v - theta, 0.0)


def extract_largest_eigenvector(D: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvector of ``(D + D^T)/2`` with the largest |eigenvalue|."""
    D = (D + D.T) / 2.0
    w, V = np.linalg.eigh(D)
    idx = int(np.argmax(np.abs(w)))
    u = np.real(V[:, idx])
    return u, float(np.abs(w[idx]))


def extract_basis(Es: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Esbasis, Fsbasis)`` from a full QR of ``Es``.

    ``Esbasis`` is an orthonormal basis of the column span of ``Es`` (first
    ``k`` columns), ``Fsbasis`` spans its orthogonal complement.
    """
    Q, _ = np.linalg.qr(Es, mode="complete")
    return Q[:, :k], Q[:, k:]


# ---------------------------------------------------------------------------
# Inner FISTA solver and majorize-minimize loop
# ---------------------------------------------------------------------------


def solve_relaxation_mezcal_approx_fista(
    Fsbasis: np.ndarray,
    G: np.ndarray,
    mu: float,
    Dinit: np.ndarray,
    maxiter: int = 100,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """FISTA solver for the trace-1 PSD relaxation used by OverICA.

    Solves (approximately)
    ``min_D  -<G, D> + (mu/2) || Fsbasis^T vec(D) ||^2``
    over ``D`` symmetric PSD with eigenvalues on the simplex.
    """
    p2, _ = Fsbasis.shape
    p = int(round(np.sqrt(p2)))

    D = Dinit.copy()
    E = D.copy()
    L = mu
    t = 1.0

    primal_vals = np.zeros(maxiter)
    dual_vals = np.zeros(maxiter)

    for it in range(maxiter):
        temp = Fsbasis.T @ _vec_F(E)
        grad = -G + mu * _mat_F(Fsbasis @ temp, p)
        E = E - (1.0 / L) * grad

        tnew = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

        Es = (E + E.T) / 2.0
        w, U = np.linalg.eigh(Es)
        U = np.real(U)
        w = np.real(w)
        eproj = proj_simplex(w)
        Dnew = (U * eproj) @ U.T

        D = (D + D.T) / 2.0
        E = Dnew + ((t - 1.0) / tnew) * (Dnew - D)
        D = Dnew
        t = tnew

        if it % 10 == 0:
            temp = Fsbasis.T @ _vec_F(D)
            grad = -G + mu * _mat_F(Fsbasis @ temp, p)
            primal_vals[it] = -float(np.sum(G * D)) + 0.5 * mu * float(temp @ temp)
            dual_vals[it] = float(np.linalg.eigvalsh((grad + grad.T) / 2.0).min()) \
                - 0.5 * mu * float(temp @ temp)

            best_dual = dual_vals[: it + 1 : 10].max()
            if (primal_vals[it] - best_dual) < tolerance:
                break

    return D


def majorize_minimize(
    G: np.ndarray,
    Fs: np.ndarray,
    *,
    mu: float = 5.0,
    maxiter: int = 100,
    tolerance: float = 1e-3,
    nmmmax: int = 100,
) -> np.ndarray:
    """Outer MM loop wrapping :func:`solve_relaxation_mezcal_approx_fista`.

    Iterates until the resulting ``D`` has Frobenius norm at least 1 (i.e. has
    converged to a rank-one atom on the simplex), at which point it represents
    a candidate atom.
    """
    p = int(round(np.sqrt(Fs.shape[0])))
    Dinit = np.eye(p) / p
    D = Dinit.copy()

    for _ in range(nmmmax):
        if np.linalg.norm(D, "fro") >= 1.0:
            break
        u, _ = extract_largest_eigenvector(G)
        Ginit = np.outer(u, u)
        D = solve_relaxation_mezcal_approx_fista(
            Fs, Ginit, mu, Dinit, maxiter, tolerance
        )
        G = (D + D.T) / 2.0
    return D


# ---------------------------------------------------------------------------
# Conversion helpers between ds and Ds
# ---------------------------------------------------------------------------


def Ds_from_ds(ds: np.ndarray) -> np.ndarray:
    """``Ds[:, i] = vec(d_i d_i^T)`` (column-major)."""
    ds = np.asarray(ds, dtype=float)
    p, k = ds.shape
    Ds = np.empty((p * p, k))
    for i in range(k):
        Ds[:, i] = _vec_F(np.outer(ds[:, i], ds[:, i]))
    return Ds


def approx_ds_from_Ds(Ds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover atoms ``d_i`` as the leading eigenvector of each ``D_i``."""
    Ds = np.asarray(Ds, dtype=float)
    p2, k = Ds.shape
    p = int(round(np.sqrt(p2)))
    ds = np.empty((p, k))
    eigmaxes = np.empty(k)
    for i in range(k):
        D = _mat_F(Ds[:, i], p)
        u, e = extract_largest_eigenvector(D)
        eigmaxes[i] = e
        ds[:, i] = u
    return ds, eigmaxes


# ---------------------------------------------------------------------------
# Deflation strategies
# ---------------------------------------------------------------------------


def adaptive_deflation(
    Fs: np.ndarray, k: int, Ds_est: np.ndarray | None = None
) -> np.ndarray:
    """Greedy deflation that augments ``Ds_est`` until it has ``k`` columns."""
    p = int(round(np.sqrt(Fs.shape[0])))
    if Ds_est is None:
        Ds_est = np.zeros((p * p, 0))
    Ds_est = np.asarray(Ds_est, dtype=float).reshape(p * p, -1)

    if Ds_est.shape[1] >= k:
        return Ds_est

    kloc = k - Ds_est.shape[1]
    Fsloc = np.hstack([Fs, Ds_est])

    for i in range(kloc):
        U, _, _ = np.linalg.svd(Fsloc, full_matrices=True)
        ncols_to_keep = Fsloc.shape[1] - (kloc - i)
        Fsloc_new = U[:, :ncols_to_keep]
        # MATLAB index range: end-(kloc-i):end -> (kloc-i+1) trailing columns
        Esloc = U[:, ncols_to_keep : ncols_to_keep + (kloc - i + 1)]

        G = _mat_F(Esloc[:, 0], p)
        D = majorize_minimize(G, Fsloc_new)
        d_vec = _vec_F(D)[:, None]
        Ds_est = np.hstack([Ds_est, d_vec])
        Fsloc = np.hstack([Fsloc_new, d_vec])

    return Ds_est


def _extract_clusters(DD: np.ndarray, nclust: int, ctype: str) -> np.ndarray:
    """Return one representative atom per cluster.

    Matches the MATLAB behaviour: pick the *first* member of each cluster as
    the representative (rather than averaging, which is unstable due to sign
    flips and small intra-cluster scaling differences).
    """
    p = int(round(np.sqrt(DD.shape[0])))
    if ctype == "h":
        Z = linkage(DD.T, method="single", metric="euclidean")
        labels = fcluster(Z, t=nclust, criterion="maxclust")
    elif ctype == "km":
        km = KMeans(n_clusters=nclust, init="k-means++", n_init=10).fit(DD.T)
        labels = km.labels_ + 1
    else:
        raise ValueError(f"unknown clustering type: {ctype!r}")

    Ds = np.zeros((p * p, nclust))
    unique_labels = np.unique(labels)
    for i, lbl in enumerate(unique_labels[:nclust]):
        members = DD[:, labels == lbl]
        Ds[:, i] = members[:, 0]
    return Ds


def cluster_Dss(Dss: np.ndarray, k: int, ctype: str = "h") -> np.ndarray:
    """Sign-align candidate atoms in ``Dss`` and cluster them into ``k`` groups."""
    Dss = np.asarray(Dss, dtype=float).copy()
    D1 = Dss[:, 0]
    for i in range(1, Dss.shape[1]):
        Dss[:, i] = np.sign(D1 @ Dss[:, i]) * Dss[:, i]
    return _extract_clusters(Dss, k, ctype)


def _validate_ctype(ctype: str | None) -> str:
    if ctype is None:
        return "h"
    if ctype not in ("h", "km"):
        return "h"
    return ctype


def sdp_cluster(
    Hs: np.ndarray, k: int, ctype: str = "h", *, rng=None
) -> tuple[np.ndarray, np.ndarray]:
    """Clustering-based deflation.

    Draws ``3k`` random initialisations, solves the SDP for each, and clusters
    the resulting atoms into ``k`` groups.
    """
    ctype = _validate_ctype(ctype)
    rng = _as_rng(rng)

    p = int(round(np.sqrt(Hs.shape[0])))
    nclust = 3 * k

    _, Fs = extract_basis(Hs, k)
    Dss = np.zeros((p * p, nclust))
    for irep in range(nclust):
        u = rng.standard_normal(p)
        u = u / np.linalg.norm(u)
        G = np.outer(u, u)
        D = majorize_minimize(G, Fs)
        Dss[:, irep] = _vec_F(D)

    Ds_est = cluster_Dss(Dss, k, ctype)
    ds_est, _ = approx_ds_from_Ds(Ds_est)
    return ds_est, Ds_est


def sdp_adaptive(Hs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive deflation (no clustering)."""
    _, Fs = extract_basis(Hs, k)
    Ds_est = adaptive_deflation(Fs, k)
    ds_est, _ = approx_ds_from_Ds(Ds_est)
    return ds_est, Ds_est


def sdp_semiada(
    Hs: np.ndarray, k: int, ctype: str = "h", *, rng=None
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-adaptive deflation: cluster, then top up with adaptive deflation.

    Keeps only well-separated atoms from the clustering step (those whose
    maximum cosine similarity with the others is below ``0.8``) and completes
    the remaining atoms with :func:`adaptive_deflation`.
    """
    ctype = _validate_ctype(ctype)
    _, Ds_clust = sdp_cluster(Hs, k, ctype, rng=rng)

    G = np.abs(Ds_clust.T @ Ds_clust) - np.eye(k)
    mind = int(np.argmin(G.max(axis=0)))
    selected = [Ds_clust[:, mind]]
    for i in range(k):
        if i == mind:
            continue
        if G[:, i].max() < 0.8:
            selected.append(Ds_clust[:, i])
    Ds_est = np.column_stack(selected) if selected else np.zeros((Ds_clust.shape[0], 0))

    _, Fs = extract_basis(Hs, k)
    Ds_est = adaptive_deflation(Fs, k, Ds_est)
    ds_est, _ = approx_ds_from_Ds(Ds_est)
    return ds_est, Ds_est
