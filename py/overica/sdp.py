"""SDP-based atom extraction and deflation for OverICA.

This module ports the contents of the MATLAB ``sdp/`` and ``helpers/``
directories:

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

Each function below is preceded by a comment block containing the
corresponding MATLAB source (verbatim), annotated line-by-line.
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/proj_simplex.m  (author: John Duchi)
#
#   function w = proj_simplex(v)                        % min_w ||w-v||₂ s.t. sum w ≤ 1, w ≥ 0
#     v = (v > 0) .* v;                                 % zero out negative entries
#     u = sort(v,'descend');                            % sort remaining entries descending
#     sv = cumsum(u);                                   % running totals of u
#     rho = find(u > (sv - 1) ./ (1:length(u))', 1, 'last');   % largest index satisfying the dual
#     theta = max(0, (sv(rho) - 1) / rho);              % water-filling threshold
#     w = max(v - theta, 0);                            % project (subtract theta, clip to ≥ 0)
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/extract_largest_eigenvector.m
#
#   function [u,e] = extract_largest_eigenvector(D)
#     D = (D+D')/2;                                     % symmetrize (defensive)
#     [u,e] = eig(D);                                   % full eigendecomposition
#     %[u,e] = eigs(D,1);                               % (alternative: just the top eigenpair)
#     [a,b] = max(abs(diag(e)));                        % index of largest |eigenvalue|
#     u = real(u(:,b(1)));                              % corresponding (real) eigenvector
#     e = a(1);                                         % return the magnitude
#   end
# ----------------------------------------------------------------------------
def extract_largest_eigenvector(D: np.ndarray) -> tuple[np.ndarray, float]:
    """Eigenvector of ``(D + D^T)/2`` with the largest |eigenvalue|."""
    D = (D + D.T) / 2.0
    w, V = np.linalg.eigh(D)
    idx = int(np.argmax(np.abs(w)))
    u = np.real(V[:, idx])
    return u, float(np.abs(w[idx]))


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/extract_basis.m
#
#   function [Esbasis, Fsbasis] = extract_basis(Es, k)
#     [Q,~] = qr(Es);                                   % full QR; Q is (p²)×(p²)
#     Esbasis = Q(:,1:k);                               % orthonormal basis of span(Es)
#     Fsbasis = Q(:,k+1:end);                           % orthonormal basis of the orth. complement
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/solve_relaxation_mezcal_approx_fista.m
#
#   function D = solve_relaxation_mezcal_approx_fista(Fsbasis,G,mu,Dinit,maxiter,tolerance)
#     [d,k] = size(Fsbasis);                            % d = p², k = #orth-complement cols
#     d = sqrt(d);                                      % reuse d for p (side length of D)
#     D = Dinit;                                        % current iterate (p×p)
#     E = D;                                            % FISTA momentum iterate
#     L = mu;                                           % Lipschitz constant of the gradient
#     t = 1;                                            % FISTA momentum parameter
#     primal_vals = zeros(1,maxiter);
#     dual_vals = zeros(1,maxiter);
#     for iter = 1:maxiter
#       temp = Fsbasis' * E(:);                         %   project vec(E) onto orth complement
#       grad = -G + mu * reshape( Fsbasis * temp,d,d);  %   ∇f(E) = -G + μ · Π_F vec(E)
#       E = E - (1/L) * ( grad );                       %   gradient step
#       tnew = .5 * ( 1 + sqrt( 1 + 4 *t*t ) );         %   FISTA momentum update
#       [u,e] = eig((E+E')/2);                          %   symmetric eigendecomposition
#       u = real(u);
#       e = real(diag(e));
#       eproj = proj_simplex( e(:) );                   %   project eigenvalues onto simplex
#       Dnew = u * diag( eproj ) * u';                  %   reassemble: PSD + trace ≤ 1
#       D = (D+D')/2;                                   %   symmetrize previous D
#       E = Dnew + ( t - 1) / tnew * ( Dnew - D);       %   FISTA momentum extrapolation
#       D = Dnew;
#       t = tnew;
#       if mod(iter,10)==1                              %   every 10 iters, check the gap:
#         temp = Fsbasis' * D(:);
#         grad = -G + mu * reshape( Fsbasis * temp,d,d);
#         primal_vals(iter) = -sum(G(:).*D(:)) + mu/2 * sum( temp.^2 );   % primal value
#         dual_vals(iter) = min(real(eig(grad))) - (mu/2) * sum(temp.^2); % dual value
#         if ( (primal_vals(iter) - max(dual_vals(1:10:iter)) ) < tolerance )
#           break;                                      %     duality gap small enough → stop
#         end
#       end
#     end
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/majorize_minimize.m
#
#   function D = majorize_minimize(G, Fs)
#     p = sqrt( size(Fs,1) );                           % side length of D (Fs has p² rows)
#     Dinit = eye(p) / p;                               % initialize D = I/p (trace-1, PSD)
#     D = Dinit;
#     mu = 5;                                           % FISTA penalty weight
#     maxiter = 100;                                    % FISTA inner iterations
#     tolerance = 1e-3;                                 % FISTA duality-gap tolerance
#     nmmmax = 100;                                     % max number of MM iterations
#     iter = 1;
#     while norm(D, 'fro') < 1                          % loop until D is rank-1 (||D||_F = 1)
#       u = extract_largest_eigenvector(G);             %   majorant direction
#       Ginit = u*u';                                   %   rank-1 surrogate for G
#       D = solve_relaxation_mezcal_approx_fista( ...   %   solve the trace-1 PSD relaxation
#             Fs,Ginit,mu,Dinit,maxiter,tolerance);
#       G = (D+D')/2;                                   %   refresh G for next majorant
#       iter = iter + 1;
#       if iter > nmmmax, break; end                    %   bail out if too many MM iterations
#     end
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/Ds_from_ds.m
#
#   function Ds = Ds_from_ds(ds)
#     [p,k] = size(ds);                                 % p = obs dim, k = #atoms
#     Ds = zeros(p^2, k);                               % preallocate flattened atoms
#     for i=1:k
#       Ds(:,i) = vec( ds(:,i)*ds(:,i)' );              % vec(d_i d_i^T) (column-major)
#     end
#   end
# ----------------------------------------------------------------------------
def Ds_from_ds(ds: np.ndarray) -> np.ndarray:
    """``Ds[:, i] = vec(d_i d_i^T)`` (column-major)."""
    ds = np.asarray(ds, dtype=float)
    p, k = ds.shape
    Ds = np.empty((p * p, k))
    for i in range(k):
        Ds[:, i] = _vec_F(np.outer(ds[:, i], ds[:, i]))
    return Ds


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/approx_ds_from_Ds.m
#
#   function [ds, eigmaxes] = approx_ds_from_Ds(Ds)
#     k = size(Ds,2);                                   % number of atoms
#     p = sqrt(size(Ds,1));                             % observed dim
#     ds = zeros(p,k);
#     eigmaxes = zeros(k,1);
#     for i = 1:k
#       D = reshape( Ds(:,i),p,p );                     % unflatten i-th atom (column-major)
#       [u,e] = extract_largest_eigenvector(D);         % rank-1 approximation
#       eigmaxes(i) = e;
#       ds(:,i) = u;
#     end
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/adaptive_deflation.m
#
#   function Ds_est = adaptive_deflation(Fs, k, Ds_est)
#     p = sqrt( size(Fs,1) );                           % observed dim
#     if ~ (nargin==2 || nargin==3), error('wrong number of inputs'); end
#     if nargin==2, Ds_est = []; end                    % default: start with empty atom set
#     if size(Ds_est,2) < k                             % need more atoms?
#       kloc = k - size(Ds_est,2);                      %   how many more
#       Fsloc = [Fs Ds_est];                            %   augmented orth-complement basis
#       for i = 1:kloc
#         [uuu,~,~] = svd(Fsloc);                       %     SVD of current basis
#         Fsloc = uuu(:,1:end-(kloc-i+1));              %     keep main-subspace columns
#         Esloc = uuu(:,end-(kloc-i):end);              %     trailing cols form residual subspace
#         G = reshape(Esloc(:,1),p,p);                  %     initial G = first residual direction
#         D = majorize_minimize(G, Fsloc);              %     extract one more atom via MM
#         Ds_est = [Ds_est D(:)];                       %     append to atom set
#         Fsloc = [Fsloc D(:)];                         %     augment basis
#       end
#     end
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``extract_clusters`` inside sdp/cluster_Dss.m
#
#   function Ds_temp = extract_clusters(DD, nclust, ctype)
#     p = sqrt(size(DD,1));                             % observed dim
#     if strcmp(ctype,'h')                              % hierarchical clustering
#       cc = clusterdata(DD', nclust);                  %   single linkage, Euclidean, maxclust
#     end
#     if strcmp(ctype,'km')                             % k-means++
#       cc = kmeans(DD, nclust);
#     end
#     Ds_temp = zeros(p^2, nclust);
#     for i = 1:nclust
#       DDi = DD(:, logical(cc==i));                    %   atoms assigned to cluster i
#       Ds_temp(:,i) = DDi(:,1); %mean(DDi,2);          %   take the FIRST member as representative
#     end                                               %   (averaging is commented out: unstable)
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/cluster_Dss.m
#
#   function Ds_est = cluster_Dss(Dss, k, ctype)
#     % atoms are scale (and hence sign) invariant; align them first to
#     % point in the same direction before clustering.
#     D1 = Dss(:,1);                                    % reference atom
#     for i = 2:size(Dss,2)
#       Di = Dss(:,i);
#       Dss(:,i) = sign( D1'*Di) * Di;                  %   flip if pointing the other way
#     end
#     Ds_est = extract_clusters(Dss, k, ctype);         % cluster into k groups, take reps
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/sdp_cluster.m
#
#   function [ds_est, Ds_est] = sdp_cluster(Hs, k, ctype)
#     if ~( nargin==2 || nargin==3), error('Wrong number of inputs'); end
#     if nargin==2, ctype = 'h'; end                    % default = hierarchical
#     if nargin==3 && ~( strcmp( ctype,'h' ) || strcmp( ctype,'km' ) ), ctype = 'h'; end
#     p = sqrt( size(Hs,1) );                           % observed dim
#     nclust = 3*k;                                     % oversample 3× before clustering
#     [~, Fs] = extract_basis(Hs, k);                   % orthogonal-complement basis Fs
#     Dss = zeros(p^2, nclust);
#     for irep = 1:nclust
#       u = randn(p,1); u = u/norm(u);                  %   random unit init direction
#       G = u*u';                                       %   rank-1 surrogate
#       D = majorize_minimize(G, Fs);                   %   solve for one candidate atom
#       Dss(:,irep) = D(:);
#     end
#     Ds_est = cluster_Dss(Dss, k, ctype);              % cluster down to k atoms
#     ds_est = approx_ds_from_Ds(Ds_est);               % rank-1 approx each
#   end
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/sdp_adaptive.m
#
#   function [ds_est, Ds_est] = sdp_adaptive(Hs, k)
#     if nargin~=2, error('wrong input'); end
#     [~, Fs] = extract_basis(Hs, k);                   % orthogonal-complement basis
#     Ds_est = adaptive_deflation(Fs, k);               % greedy deflation, k atoms
#     ds_est = approx_ds_from_Ds(Ds_est);               % rank-1 approx each
#   end
# ----------------------------------------------------------------------------
def sdp_adaptive(Hs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive deflation (no clustering)."""
    _, Fs = extract_basis(Hs, k)
    Ds_est = adaptive_deflation(Fs, k)
    ds_est, _ = approx_ds_from_Ds(Ds_est)
    return ds_est, Ds_est


# ----------------------------------------------------------------------------
# MATLAB reference: sdp/sdp_semiada.m
#
#   function [ds_est, Ds_est] = sdp_semiada(Hs, k, ctype)
#     if ~( nargin==2 || nargin==3), error('Wrong number of inputs'); end
#     if nargin==2, ctype = 'h'; end                    % default = hierarchical
#     if nargin==3 && ~( strcmp( ctype,'h' ) || strcmp( ctype,'km' ) ), ctype = 'h'; end
#     [~, Ds_clust] = sdp_cluster(Hs, k, ctype);        % first pass: cluster-based atoms
#     % keep only atoms which are well separated
#     G = abs(Ds_clust'*Ds_clust) - eye(k);             % |cosines| between distinct atoms
#     [~, mind] = min(max(G));                          % atom most separated from the rest
#     Ds_est(:,1) = Ds_clust(:,mind);                   % seed with that atom
#     ind = 2;
#     for i = setdiff( 1:k, mind )                      % keep additional well-separated atoms
#       if max( G(:,i) ) < .8                           %   (max similarity below 0.8)
#         Ds_est(:,ind) = Ds_clust(:,i);
#         ind = ind + 1;
#       end
#     end
#     [~, Fs] = extract_basis(Hs, k);                   % top up with adaptive deflation
#     Ds_est = adaptive_deflation(Fs, k, Ds_est);
#     ds_est = approx_ds_from_Ds(Ds_est);
#   end
# ----------------------------------------------------------------------------
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
