"""Synthetic data sampling utilities (ports of the ``sampling/`` directory).

Every function below carries the corresponding MATLAB source (verbatim) as a
comment block, with line-by-line annotations. Inline ``# MATLAB: ...``
comments inside each body point each Python statement back to the MATLAB
line(s) it implements (and explain what it actually does).
"""

from __future__ import annotations

import numpy as np


def _as_rng(rng) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# ----------------------------------------------------------------------------
# MATLAB reference: sampling/sample_mixing_matrix.m
#
#   function ds = sample_mixing_matrix(p,k)
#     ds = randn(p,k);                                  % p×k iid standard-normal matrix
#     for i=1:k                                         % for each of the k columns:
#       ds(:,i) = ds(:,i)/norm(ds(:,i),2);              %   rescale to unit ℓ²-norm
#     end                                               % end loop
#   end                                                 % end function
# ----------------------------------------------------------------------------
def sample_mixing_matrix(p: int, k: int, *, rng=None) -> np.ndarray:
    """Sample a ``(p, k)`` Gaussian mixing matrix with unit-norm columns."""
    rng = _as_rng(rng)
    ds = rng.standard_normal((p, k))                         # MATLAB: ds = randn(p,k)
    ds /= np.linalg.norm(ds, axis=0, keepdims=True)          # MATLAB: ds(:,i) = ds(:,i)/norm(...,2)
                                                             #          (vectorised: all columns at once)
    return ds


# ----------------------------------------------------------------------------
# MATLAB reference: sampling/sample_orthogonal_matrix.m
#
#   function V = sample_orthogonal_matrix(k)
#     X = rand(k);                                      % k×k matrix of iid U(0,1) entries
#     V = orth(X);                                      % orthonormal basis for span(X) (≡ Q of QR(X))
#   end                                                 % end function
# ----------------------------------------------------------------------------
def sample_orthogonal_matrix(k: int, *, rng=None) -> np.ndarray:
    """Sample an orthogonal ``(k, k)`` matrix via the QR of a random matrix."""
    rng = _as_rng(rng)
    X = rng.random((k, k))                                   # MATLAB: X = rand(k)
    Q, _ = np.linalg.qr(X)                                   # MATLAB: V = orth(X)  (Q is the QR's Q)
    return Q


# ----------------------------------------------------------------------------
# MATLAB reference: sampling/sample_from_ica_with_uniform_sources.m
#
#   function [X, Alpha] = sample_from_ica_with_uniform_sources(ds, N)
#     if ~( nargin==2 )                                 % require exactly 2 inputs
#       error('sample data: wrong number of inputs')    %   else abort
#     end
#     [p,k] = size(ds);                                 % p = observed dim, k = latent dim
#     X = zeros(p,N);                                   % preallocate observations
#     Alpha = zeros(k,N);                               % preallocate latent draws
#     n = 1000;                                         % batch size (for speed)
#     times = floor(N/n);                               % number of full batches
#     rest = mod(N,n);                                  % size of the final partial batch
#     for i=1:times                                     % full batches:
#       inds = (i-1)*n + 1:i*n;                         %   destination indices
#       [x, alpha] = sample_batch(n);                   %   draw one batch
#       X(:,inds) = x;                                  %   store observations
#       Alpha(:,inds) = alpha;                          %   store source values
#     end                                               % end full-batch loop
#     inds = times*n + 1 : times*n + rest;              % indices of the final partial batch
#     [x, alpha] = sample_batch(rest);                  % draw partial batch
#     X(:,inds) = x;                                    % store observations
#     Alpha(:,inds) = alpha;                            % store sources
#
#     function [x, alpha] = sample_batch(n)             % nested helper:
#       alpha = rand(k,n).*abs( randn(k,n) );           %   heavy-tailed non-Gaussian sources
#       x = sparse( ds*alpha );                         %   x = D * alpha (sparse for speed)
#     end                                               % end nested function
#   end                                                 % end outer function
# ----------------------------------------------------------------------------
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
    p, k = ds.shape                                          # MATLAB: [p,k] = size(ds)

    batch = 1000                                             # MATLAB: n = 1000  (batch size)
    times = N // batch                                       # MATLAB: times = floor(N/n)
    rest = N % batch                                         # MATLAB: rest = mod(N,n)

    X = np.empty((p, N))                                     # MATLAB: X = zeros(p,N)
    Alpha = np.empty((k, N))                                 # MATLAB: Alpha = zeros(k,N)

    def _sample_batch(n):
        # MATLAB nested fn: alpha = rand(k,n) .* abs(randn(k,n))   (heavy-tailed sources)
        a = rng.random((k, n)) * np.abs(rng.standard_normal((k, n)))
        # MATLAB nested fn: x = sparse(ds * alpha)                 (dense here; numpy is fast enough)
        return ds @ a, a

    for i in range(times):                                   # MATLAB: for i=1:times
        x, a = _sample_batch(batch)                          # MATLAB:   [x, alpha] = sample_batch(n)
        sl = slice(i * batch, (i + 1) * batch)               # MATLAB:   inds = (i-1)*n+1 : i*n
        X[:, sl] = x                                         # MATLAB:   X(:,inds) = x
        Alpha[:, sl] = a                                     # MATLAB:   Alpha(:,inds) = alpha

    if rest > 0:                                             # MATLAB: implicit (handles rest unconditionally)
        x, a = _sample_batch(rest)                           # MATLAB: [x, alpha] = sample_batch(rest)
        sl = slice(times * batch, times * batch + rest)      # MATLAB: inds = times*n+1 : times*n+rest
        X[:, sl] = x                                         # MATLAB: X(:,inds) = x
        Alpha[:, sl] = a                                     # MATLAB: Alpha(:,inds) = alpha

    return X, Alpha
