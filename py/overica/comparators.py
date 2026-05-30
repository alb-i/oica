"""Baseline / comparator algorithms used in the reproduction scripts.

* :func:`fourier_pca` is a port of ``comparison/fpca/fourier_pca2.m``.

The MATLAB code base also references FOOBI (Lieven De Lathauwer's algorithm),
but the actual implementation is *not* included in this repository (only a
README pointing at the author). We therefore do not provide a Python FOOBI;
the reproduction scripts skip it gracefully when it isn't installed.

Inline ``# MATLAB: ...`` comments in the body of :func:`fourier_pca` map each
Python statement back to the corresponding line of ``fourier_pca2.m``.
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


# ----------------------------------------------------------------------------
# MATLAB reference: comparison/fpca/fourier_pca2.m
#
#   function ds = fourier_pca2(X, k)
#     [p,~] = size(X);                                  % p = observed dimension
#     u = randn(p,1); u = u/norm(u);                    % random unit-norm direction
#     Q = genquadricov(X, u);                           % generalized 4th-order cumulant at u (complex)
#     Q1 = real(Q); Q2 = imag(Q);                       % split into real / imag parts
#     [W,S,~] = svd(Q1);                                % SVD of the real part
#     [~,inds] = sort(diag(S),'descend');               % ensure descending order
#     W = W(:,inds);                                    %   (svd returns them sorted already; defensive)
#     W = W(:,1:k);                                     % keep the top-k left singular vectors
#     q1 = W'*Q1*W;                                     % project Q1 into W-basis (k×k)
#     q2 = W'*Q2*W;                                     % same for Q2
#     M = q1/q2;                                        % matrix right-division: M = q1 · inv(q2)
#     [V,~] = eig(M);                                   % eigendecomposition of M
#     C = W*V;                                          % candidate atoms (complex, in p-space)
#     for j = 1:k                                       % phase-correct each column:
#       c = C(:,j);
#       a = real(c); b = imag(c);
#       theta = atan( -( 2 * sum(a.*b) ) / sum( a.^2 - b.^2 ) )/2;   % optimal rotation
#       while theta<0, theta = theta + pi; end          %   wrap into [0, 2π)
#       while theta>2*pi, theta = theta - pi; end
#       temp = real(exp(1i*theta)*c);                   %   rotate so the imag part vanishes
#       C(:,j) = temp/norm(temp);                       %   normalize to unit norm
#     end
#     ds = zeros(p,k);
#     for j = 1:k                                       % atom recovery from rank-1 matrix:
#       c = C(:,j);
#       [v,s,~] = svd( reshape(c,p,p) );                %   SVD of unflattened p×p matrix
#       [~,inds] = sort(diag(s), 'descend');
#       v = v(:,inds);
#       ds(:,j) = v(:,1);                               %   keep leading left singular vector
#     end
#   end                                                 % end function
# ----------------------------------------------------------------------------
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
    p, _ = X.shape                                           # MATLAB: [p,~] = size(X)

    u = rng.standard_normal(p)                               # MATLAB: u = randn(p,1)
    u = u / np.linalg.norm(u)                                # MATLAB: u = u/norm(u)

    Q = genquadricov(X, u)                                   # MATLAB: Q = genquadricov(X, u)   (complex)
    Q1 = Q.real                                              # MATLAB: Q1 = real(Q)
    Q2 = Q.imag                                              # MATLAB: Q2 = imag(Q)

    # SVD of the real part; numpy returns singular values already sorted
    # descending, so we skip the explicit MATLAB ``sort(diag(S),'descend')``.
    W, S, _ = np.linalg.svd(Q1, full_matrices=False)         # MATLAB: [W,S,~] = svd(Q1)
    W = W[:, :k]                                             # MATLAB: W = W(:,1:k)  (top-k subspace)

    q1 = W.T @ Q1 @ W                                        # MATLAB: q1 = W'*Q1*W
    q2 = W.T @ Q2 @ W                                        # MATLAB: q2 = W'*Q2*W

    # MATLAB ``q1/q2`` is right-division: q1 * inv(q2). Solve M q2 = q1.
    M = np.linalg.solve(q2.T, q1.T).T                        # MATLAB: M = q1/q2

    _, V = np.linalg.eig(M)                                  # MATLAB: [V,~] = eig(M)
    C = W @ V                                                # MATLAB: C = W*V   (p×k, complex atoms)

    for j in range(k):                                       # MATLAB: for j = 1:k   (phase-correct each col)
        c = C[:, j]                                          # MATLAB: c = C(:,j)
        a = c.real                                           # MATLAB: a = real(c)
        b = c.imag                                           # MATLAB: b = imag(c)
        denom = float(np.sum(a * a - b * b))                 # MATLAB: sum(a.^2 - b.^2)
        num = -2.0 * float(np.sum(a * b))                    # MATLAB: -2*sum(a.*b)
        # MATLAB uses atan(num/denom); we use atan2 to avoid blow-up when
        # denom == 0. The result is the same modulo the wrap-around below.
        theta = np.arctan2(num, denom) / 2.0 if denom != 0 else (np.pi / 4.0)  # MATLAB: theta = atan(...)/2
        while theta < 0:                                     # MATLAB: while theta<0, theta = theta+pi; end
            theta += np.pi
        while theta > 2 * np.pi:                             # MATLAB: while theta>2*pi, theta = theta-pi; end
            theta -= np.pi
        tmp = (np.exp(1j * theta) * c).real                  # MATLAB: temp = real(exp(1i*theta)*c)
        norm = np.linalg.norm(tmp)
        if norm > 0:                                         # (defensive: avoid div-by-zero)
            C[:, j] = tmp / norm                             # MATLAB: C(:,j) = temp/norm(temp)

    ds = np.zeros((p, k))                                    # MATLAB: ds = zeros(p,k)
    for j in range(k):                                       # MATLAB: for j = 1:k  (recover atoms via SVD)
        c = C[:, j].real
        rmat = c.reshape(p, p, order="F")                    # MATLAB: reshape(c, p, p)  (col-major)
        u_svd, _, _ = np.linalg.svd(rmat, full_matrices=False)  # MATLAB: [v,s,~] = svd(...)
        ds[:, j] = u_svd[:, 0]                               # MATLAB: ds(:,j) = v(:,1)  (top left singular vec)

    return ds
