"""Atom-recovery error metrics (port of the MATLAB ``helpers/`` evaluation).

Each public Python function is preceded by a comment block containing the
corresponding MATLAB source (verbatim), annotated line-by-line. Inline
``# MATLAB: ...`` comments inside each body further point each Python
statement back to its MATLAB counterpart and explain what it does.

The MATLAB code base uses a Hungarian-matching implementation
``HungarianBipartiteMatching`` shipped under ``helpers/``; we replace it with
``scipy.optimize.linear_sum_assignment`` (same algorithm, returns the row /
column index pairs of the optimal matching directly).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------------------
# MATLAB reference: nested helper ``normalize_2`` used by several .m files
#
#   function ds = normalize_2(ds)
#     k = size(ds,2);                                   % number of columns
#     for i = 1:k                                       % rescale each column to unit ℓ²-norm
#       ds(:,i) = ds(:,i) / norm(ds(:,i));
#     end
#   end
# ----------------------------------------------------------------------------
def _normalize_2(ds: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(ds, axis=0, keepdims=True)        # MATLAB: norm(ds(:,i))   (vectorised over i)
    norms = np.where(norms == 0, 1.0, norms)                 # (defensive: avoid div-by-zero on null columns)
    return ds / norms                                        # MATLAB: ds(:,i) = ds(:,i)/norm(ds(:,i))


# ----------------------------------------------------------------------------
# MATLAB reference: nested helper ``update_ds_est`` used by several .m files
#
#   function [ds_est, perfout] = update_ds_est(ds_est, ds, perm)
#     k = size(ds_est, 2);                              % number of atoms
#     ds_est = ds_est(:, perm);                         % reorder to align with ds
#     signs = zeros(k,1);
#     for i = 1:k                                       % for each aligned column:
#       signi = sign( ds_est(:,i)' * ds(:,i) );         %   sign(<d_est_i, d_i>)
#       signs(i) = signi;                               %   remember the flip
#       ds_est(:,i) = ds_est(:,i) * signi;              %   apply the sign flip
#     end
#     perfout = struct('ds_est', ds_est, 'perm', perm, 'signs', signs);   % return audit info
#   end
# ----------------------------------------------------------------------------
def _update_ds_est(ds_est: np.ndarray, ds: np.ndarray, perm: np.ndarray):
    ds_est = ds_est[:, perm]                                 # MATLAB: ds_est = ds_est(:, perm)
    # Per-column inner products via einsum (equivalent to sum(ds_est.*ds, axis=0)).
    signs = np.sign(np.einsum("ij,ij->j", ds_est, ds))       # MATLAB: signi = sign(ds_est(:,i)'*ds(:,i))
    signs[signs == 0] = 1.0                                  # (defensive: MATLAB sign(0)=0 → keep column)
    ds_est = ds_est * signs                                  # MATLAB: ds_est(:,i) = ds_est(:,i)*signi
    return ds_est, {"ds_est": ds_est, "perm": perm, "signs": signs}  # MATLAB: perfout = struct(...)


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``perf_l2`` inside helpers/evaluation_perf.m
#
#   function [perf, perfout] = perf_l2(ds_est, ds)
#     k = size(ds,2);                                   % number of atoms
#     F = zeros(k);                                     % cost matrix for Hungarian matching
#     for i=1:k
#       for j=1:k
#         sij = sign( ds_est(:,i)'*ds(:,j) );           %   align signs
#         F(i,j) = norm( sij*ds_est(:,i) - ds(:,j), 2); %   ℓ²-distance after sign flip
#       end
#     end
#     [matching, cost] = HungarianBipartiteMatching(F); % minimum-weight bipartite matching
#     [perm, ~] = find(sparse(matching));               % extract column permutation
#     [~, perfout] = update_ds_est(ds_est, ds, perm);   % reorder/sign-flip ds_est
#     perf = cost/k;                                    % mean per-atom ℓ²-error
#     perf = perf/sqrt(2);                              % rescale to [0, 1]
#   end
# ----------------------------------------------------------------------------
def _perf_l2(ds_est: np.ndarray, ds: np.ndarray):
    k = ds.shape[1]                                          # MATLAB: k = size(ds,2)
    F = np.zeros((k, k))                                     # MATLAB: F = zeros(k)
    for i in range(k):                                       # MATLAB: for i = 1:k
        for j in range(k):                                   # MATLAB:   for j = 1:k
            sij = np.sign(ds_est[:, i] @ ds[:, j])           # MATLAB: sij = sign( ds_est(:,i)'*ds(:,j) )
            if sij == 0:
                sij = 1.0
            F[i, j] = np.linalg.norm(sij * ds_est[:, i] - ds[:, j])  # MATLAB: F(i,j) = norm(sij*ds_est(:,i) - ds(:,j), 2)
    # MATLAB: HungarianBipartiteMatching(F)  →  optimal column permutation.
    row, col = linear_sum_assignment(F)
    cost = F[row, col].sum()                                 # MATLAB: cost = sum of matched entries
    perm = col[np.argsort(row)]                              # MATLAB: [perm,~] = find(sparse(matching))
    _, perfout = _update_ds_est(ds_est, ds, perm)            # MATLAB: [~,perfout] = update_ds_est(...)
    perf = cost / k / np.sqrt(2.0)                           # MATLAB: perf = cost/k; perf = perf/sqrt(2)
    return perf, perfout


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``perf_fro`` inside helpers/evaluation_perf.m
#
#   function [perf, perfout] = perf_fro(ds_est, ds)
#     matching = HungarianBipartiteMatching(-abs(ds_est'*ds));   % maximize |cosine|
#     [perm, ~] = find(sparse(matching));
#     [ds_est, perfout] = update_ds_est(ds_est, ds, perm);
#     perf = norm(ds_est-ds,'fro').^2 / norm(ds,'fro').^2;       % squared Frobenius (relative)
#   end
# ----------------------------------------------------------------------------
def _perf_fro(ds_est: np.ndarray, ds: np.ndarray):
    F = -np.abs(ds_est.T @ ds)                               # MATLAB: -abs(ds_est'*ds)  (minimize → maximize |cos|)
    row, col = linear_sum_assignment(F)                      # MATLAB: HungarianBipartiteMatching(...)
    perm = col[np.argsort(row)]                              # MATLAB: [perm,~] = find(sparse(matching))
    ds_est, perfout = _update_ds_est(ds_est, ds, perm)       # MATLAB: [ds_est, perfout] = update_ds_est(...)
    perf = np.linalg.norm(ds_est - ds, "fro") ** 2 / np.linalg.norm(ds, "fro") ** 2  # MATLAB: norm(ds_est-ds,'fro').^2 / norm(ds,'fro').^2
    return float(perf), perfout


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``perf_l1`` inside helpers/evaluation_perf.m
#
#   function [perf, perfout] = perf_l1(ds_est, ds)
#     k = size(ds,2);
#     F = zeros(k);                                     % cost matrix (ℓ¹-distances after sign flip)
#     for i = 1:k
#       for j = 1:k
#         sij = sign( ds_est(:,i)'*ds(:,j) );
#         F(i,j) = norm( sij*ds_est(:,i)-ds(:,j), 1);
#       end
#     end
#     [matching, cost] = HungarianBipartiteMatching(F);
#     [perm, ~] = find(sparse(matching));
#     [~, perfout] = update_ds_est(ds_est, ds, perm);
#     perf = cost/k;                                    % mean per-atom ℓ¹-distance
#     perf = perf/2;                                    % rescale to [0, 1]
#   end
# ----------------------------------------------------------------------------
def _perf_l1(ds_est: np.ndarray, ds: np.ndarray):
    k = ds.shape[1]                                          # MATLAB: k = size(ds,2)
    F = np.zeros((k, k))                                     # MATLAB: F = zeros(k)
    for i in range(k):                                       # MATLAB: for i = 1:k
        for j in range(k):                                   # MATLAB:   for j = 1:k
            sij = np.sign(ds_est[:, i] @ ds[:, j])           # MATLAB: sij = sign(ds_est(:,i)'*ds(:,j))
            if sij == 0:
                sij = 1.0
            F[i, j] = np.sum(np.abs(sij * ds_est[:, i] - ds[:, j]))  # MATLAB: F(i,j) = norm(sij*ds_est(:,i)-ds(:,j), 1)
    row, col = linear_sum_assignment(F)                      # MATLAB: HungarianBipartiteMatching(F)
    cost = F[row, col].sum()
    perm = col[np.argsort(row)]                              # MATLAB: [perm,~] = find(sparse(matching))
    _, perfout = _update_ds_est(ds_est, ds, perm)            # MATLAB: [~,perfout] = update_ds_est(...)
    perf = cost / k / 2.0                                    # MATLAB: perf = cost/k; perf = perf/2
    return float(perf), perfout


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``perf_cos`` inside helpers/evaluation_perf.m
#
#   function [perf, perfout] = perf_cos(ds_est, ds)
#     k = size(ds,2);
#     F = zeros(k,k);                                   % angular-distance cost matrix
#     for i = 1:k
#       for j = 1:k
#         cosij = abs( ds_est(:,i)'*ds(:,j) ) / ...     %   |cos(angle)| between atoms
#                  ( norm(ds_est(:,i)) * norm(ds(:,j)) );
#         loc = acos( cosij );                          %   angular distance
#         F(i,j) = real(loc);                           %   guard against tiny imag noise
#       end
#     end
#     [matching, cost] = HungarianBipartiteMatching(F);
#     [perm, ~] = find(sparse(matching));
#     [~, perfout] = update_ds_est(ds_est, ds, perm);
#     perf = 2*cost/pi;                                 % rescale total angle to [0, k]
#     perf = perf/k;                                    % per-atom mean, in [0, 1]
#   end
# ----------------------------------------------------------------------------
def _perf_cos(ds_est: np.ndarray, ds: np.ndarray):
    # Vectorised replacement of the MATLAB double-for loop building F(i,j).
    k = ds.shape[1]                                          # MATLAB: k = size(ds,2)
    norms_e = np.linalg.norm(ds_est, axis=0)                 # MATLAB: norm(ds_est(:,i))   (vectorised)
    norms_d = np.linalg.norm(ds, axis=0)                     # MATLAB: norm(ds(:,j))       (vectorised)
    cosmat = np.abs(ds_est.T @ ds) / np.outer(np.where(norms_e == 0, 1.0, norms_e),
                                              np.where(norms_d == 0, 1.0, norms_d))
                                                             # MATLAB: cosij = abs(ds_est(:,i)'*ds(:,j)) / (norm·norm)
    cosmat = np.clip(cosmat, 0.0, 1.0)                       # (numerical safety before acos)
    F = np.arccos(cosmat)                                    # MATLAB: F(i,j) = real(acos(cosij))
    row, col = linear_sum_assignment(F)                      # MATLAB: HungarianBipartiteMatching(F)
    cost = F[row, col].sum()
    perm = col[np.argsort(row)]                              # MATLAB: [perm,~] = find(sparse(matching))
    _, perfout = _update_ds_est(ds_est, ds, perm)            # MATLAB: [~,perfout] = update_ds_est(...)
    perf = 2.0 * cost / np.pi / k                            # MATLAB: perf = 2*cost/pi; perf = perf/k
    return float(perf), perfout


_PERF_FNS = {1: _perf_l1, 2: _perf_l2, 3: _perf_fro, 4: _perf_cos}


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/evaluation_perf.m (driver)
#
#   function [perf, perfout] = evaluation_perf(ds_est, ds, perf_type)
#     if ~( (nargin==2) || (nargin==3) )                % require 2 or 3 inputs
#       error('perf: wrong input')
#     end
#     if sum( sum( size(ds_est) == size(ds) ) ) ~= 2    % shapes must match exactly
#       error('perf: wrong input')
#     end
#     if nargin==2, perf_type = 2; end                  % default = ℓ²-metric
#     ds = normalize_2(ds);                             % unit-norm both
#     ds_est = normalize_2(ds_est);
#     switch perf_type                                  % dispatch on metric:
#       case 1, [perf, perfout] = perf_l1(ds_est, ds);
#       case 2, [perf, perfout] = perf_l2(ds_est, ds);
#       case 3, [perf, perfout] = perf_fro(ds_est, ds);
#       case 4, [perf, perfout] = perf_cos(ds_est, ds);
#       otherwise, error('perf: specify type!')
#     end
#   end
# ----------------------------------------------------------------------------
def evaluation_perf(ds_est: np.ndarray, ds: np.ndarray, perf_type: int = 2):
    """Atom-recovery error after optimal sign-aware bipartite matching.

    ``perf_type`` selects the cost metric:

    * 1 – L1
    * 2 – L2 (default)
    * 3 – squared Frobenius
    * 4 – cosine / angular
    """
    if ds_est.shape != ds.shape:                             # MATLAB: if sum(size==size)~=2, error('perf: wrong input')
        raise ValueError("ds_est and ds must have matching shapes")
    if perf_type not in _PERF_FNS:                           # MATLAB: otherwise, error('perf: specify type!')
        raise ValueError(f"unknown perf_type: {perf_type}")
    ds = _normalize_2(np.asarray(ds, dtype=float))           # MATLAB: ds = normalize_2(ds)
    ds_est = _normalize_2(np.asarray(ds_est, dtype=float))   # MATLAB: ds_est = normalize_2(ds_est)
    return _PERF_FNS[perf_type](ds_est, ds)                  # MATLAB: switch perf_type ... end


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/a_error.m
#
#   function [perf, perfout] = a_error(ds_est, ds)
#     perf_type = 4;                                    % cosine / angular metric
#     [perf, perfout] = evaluation_perf(ds_est, ds, perf_type);
#   end
# ----------------------------------------------------------------------------
def a_error(ds_est: np.ndarray, ds: np.ndarray):
    """Angular (cosine) error, scaled to ``[0, 1]``."""
    return evaluation_perf(ds_est, ds, perf_type=4)          # MATLAB: evaluation_perf(ds_est, ds, 4)


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/f_error.m
#
#   function [perf, perfout] = f_error(ds_est, ds)
#     perf_type = 3;                                    % squared Frobenius metric
#     [perf, perfout] = evaluation_perf(ds_est, ds, perf_type);
#   end
# ----------------------------------------------------------------------------
def f_error(ds_est: np.ndarray, ds: np.ndarray):
    """Squared Frobenius error (relative)."""
    return evaluation_perf(ds_est, ds, perf_type=3)          # MATLAB: evaluation_perf(ds_est, ds, 3)


# ----------------------------------------------------------------------------
# MATLAB reference: helpers/evaluation_recovery.m
#
#   function [nrec, recov, perfout] = evaluation_recovery(ds_est, ds, th)
#     if nargin ~= 3, error('wrong input'); end
#     ds = normalize_2(ds);                             % unit-norm both
#     ds_est = normalize_2(ds_est);
#     [perf, perfout] = perf_cos(ds_est,ds);            % angular errors of matched atoms
#     k = size(ds, 2);
#     recov = zeros(k,1);                               % per-rank recovery indicator
#     for i = 1:k
#       recov(i) = sum( perf < th ) >= i;               %   1 iff ≥ i atoms below threshold
#     end
#     nrec = sum( perf < th );                          % total number recovered
#   end
# (Note: the inner ``perf_cos`` here builds ``test`` — the vector of per-atom
# angles — and returns it as ``perf``, hence ``sum( perf < th )`` counts atoms.)
# ----------------------------------------------------------------------------
def evaluation_recovery(ds_est: np.ndarray, ds: np.ndarray, th: float):
    """Count how many atoms of ``ds`` are recovered within angular threshold ``th``.

    Returns ``(nrec, recov, perfout)`` where ``recov[i]`` is ``1`` if at least
    ``i+1`` atoms are recovered. Matches the MATLAB ``evaluation_recovery``.
    """
    ds = _normalize_2(np.asarray(ds, dtype=float))           # MATLAB: ds = normalize_2(ds)
    ds_est = _normalize_2(np.asarray(ds_est, dtype=float))   # MATLAB: ds_est = normalize_2(ds_est)
    k = ds.shape[1]                                          # MATLAB: k = size(ds, 2)
    # Cost matrix = angular distance between every (i,j) pair (≡ perf_cos build).
    norms_e = np.linalg.norm(ds_est, axis=0)
    norms_d = np.linalg.norm(ds, axis=0)
    cosmat = np.abs(ds_est.T @ ds) / np.outer(np.where(norms_e == 0, 1.0, norms_e),
                                              np.where(norms_d == 0, 1.0, norms_d))
    cosmat = np.clip(cosmat, 0.0, 1.0)
    F = np.arccos(cosmat)                                    # MATLAB perf_cos: F(i,j) = acos(cosij)
    row, col = linear_sum_assignment(F)                      # MATLAB: HungarianBipartiteMatching(F)
    perm = col[np.argsort(row)]                              # MATLAB: [perm,~] = find(sparse(matching))
    _, perfout = _update_ds_est(ds_est, ds, perm)            # MATLAB: [~,perfout] = update_ds_est(...)

    # ``test`` holds the per-atom angular distance after matching (this is
    # the ``perf`` vector returned by the MATLAB-side ``perf_cos`` here).
    test = np.empty(k)
    for i in range(k):                                       # MATLAB: implicit in perf_cos loop
        c = abs(ds_est[:, perm[i]] @ ds[:, i]) / (
            np.linalg.norm(ds_est[:, perm[i]]) * np.linalg.norm(ds[:, i])
        )
        test[i] = np.arccos(np.clip(c, 0.0, 1.0))

    nrec = int(np.sum(test < th))                            # MATLAB: nrec = sum( perf < th )
    # MATLAB: recov[i] = 1 iff at least i+1 atoms have error below threshold;
    # since the criterion depends only on the *total* count, this collapses to
    # a step function with ``nrec`` leading ones.
    recov = (nrec >= np.arange(1, k + 1)).astype(int)        # MATLAB: recov(i) = sum(perf<th) >= i
    return nrec, recov, perfout
