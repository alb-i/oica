"""Cumulant estimators used by OverICA.

All functions operate on data matrices ``X`` of shape ``(p, n)`` (observations
in columns), matching the original MATLAB convention.

Throughout, ``vec(M)`` means column-major flattening (MATLAB's ``M(:)``).

Every Python function below is preceded by a comment block containing the
corresponding MATLAB source (verbatim), annotated line-by-line.
"""

from __future__ import annotations

import numpy as np


def _vec_F(M: np.ndarray) -> np.ndarray:
    """Column-major (Fortran-order) flatten, matching MATLAB ``M(:)``."""
    return np.asarray(M).reshape(-1, order="F")


# ----------------------------------------------------------------------------
# MATLAB reference: cumulants/gencov.m
#
#   function C = gencov(X, omega)
#     if numel(omega)==1                                % if omega is a scalar:
#       p = size(X,1);                                  %   read off the observed dim p
#       omega = omega*ones(p,1)/p;                      %   replace with that scalar / p, broadcast to ℝᵖ
#     end                                               % end scalar-handling branch
#     n = size(X,2);                                    % n = number of observations
#     proj = X'*omega;                                  % proj ∈ ℝⁿ: projection of each obs onto omega
#     eproj = exp(proj);                                % exponential reweighting (one weight per obs)
#     % genexp                                          % (comment in original: "generalized expectation")
#     Eomega = (X*eproj) / sum(eproj);                  % weighted mean E_ω[X] ∈ ℝᵖ
#     C = X*sparse(1:n,1:n,eproj)*X' ;                  % X · diag(eproj) · X' (sparse for speed)
#     C = C / sum(eproj);                               % normalize to weighted E_ω[X X']
#     C = C - Eomega * Eomega';                         % subtract outer product of the weighted mean
#     C = C(:);                                         % return as column-major vec, shape p²×1
#   end                                                 % end function
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference (driver): cumulants/quadricov.m
#
#   function Q = quadricov(X)
#     X = X - repmat(mean(X,2), 1, size(X,2));          % centre each row to zero mean
#     Q = quadricov_in(X);                              % call the MEX kernel (upper-triangular cumulant)
#     Q = (Q - diag(diag(Q)))' + Q;                     % mirror upper-tri to lower (avoid doubling diag)
#   end                                                 % end function
#
# MATLAB reference (kernel): cumulants/quadricov_in.cpp
#   /* Computes upper-triangular part of the matricized 4-th order cumulant. */
#   for (i = 0; i < n; i++) {                           // for each observation:
#     for (a, b in [0..p)²) {                           //   for each (a,b) pair:
#       tt = data[i][a] * data[i][b];                   //     tt = x_{i,a} · x_{i,b}
#       C[a][b] += tt;                                  //     accumulate biased covariance
#       temp[a + b*p] = tt;                             //     store vec(x_i x_i^T) entry (col-major)
#     }
#     for (a, b in upper triangle of [0..p²)²)          //   for each upper-tri (a,b):
#       Q[a][b] += temp[a] * temp[b];                   //     accumulate outer products
#   }
#   C /= n;                                             // normalize C and Q (raw 4-th moment)
#   Q /= n;
#   for (a, b, c, d in [0..p)⁴) {                       // subtract Gaussian "Wick" terms:
#     Q[a+b*p][c+d*p] -= C[a][b]*C[c][d]                //   - <x_a x_b> <x_c x_d>
#                      + C[a][c]*C[b][d]                //   - <x_a x_c> <x_b x_d>
#                      + C[a][d]*C[b][c];               //   - <x_a x_d> <x_b x_c>
#   }                                                   // (computed only for upper triangle)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# MATLAB reference: cumulants/genquadricov.m
#
#   function Q = genquadricov(X, u)
#     [p,N] = size(X);                                  % p = obs dim, N = number of samples
#     Xu = X'*u;                                        % Xu(n) = u'·x_n, shape (N,)
#     expXu = exp(1i*Xu);                               % complex weights e^{i u'x_n}
#     M0 = sum( expXu ) / N;                            % characteristic function value at u
#     M1 = (X * expXu ) / (N*M0);                       % generalized 1st moment ("mean")
#     XC = X - repmat( M1, 1, N );                      % "generalized zero-mean" data
#     M2 = zeros(p);                                    % accumulator for generalized 2nd moment
#     M4 = zeros(p*p);                                  % accumulator for generalized 4th moment
#     temp = zeros(p);                                  % scratch p×p matrix
#     for n = 1:N                                       % for each observation:
#       xn = XC(:,n);                                   %   centred sample
#       temp = xn*conj(ctranspose(xn));                 %   xn · xnᵀ  (note: transpose, NOT conj-transpose;
#                                                       %    MATLAB's ' is conj-transpose, hence the conj)
#       M2 = M2 + expXu(n)*temp;                        %   weight by characteristic-func value
#       M4 = M4 + (expXu(n)*temp(:))*conj(ctranspose(temp(:)));  %   outer product of vec(xx')
#     end                                               % end loop
#     M2 = M2 / (N*M0); M4 = M4 / (N*M0);               % normalize
#     Q = zeros(p*p);                                   % preallocate cumulant
#     for i3 = 1:p                                      % for each (i3, i4) index pair:
#       for i4 = 1:p
#         icol = (i3-1)*p+i4;                           %   column-major linear index of (i3,i4)
#         temp(:) = M4(:,icol);                         %   pull out the matching column of M4
#         temp = temp - M2(i3,i4)*M2 ...                %   subtract the three M2-product (Wick) terms;
#                    - M2(:,i3)*M2(i4,:) ...            %   uses M2 symmetric, so M2' is dropped here
#                    - M2(:,i4)*M2(i3,:);
#         Q(:,icol) = temp(:);                          %   store into output column
#       end
#     end
#   end                                                 % end function (returns complex p²×p²)
# ----------------------------------------------------------------------------
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
