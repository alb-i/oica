"""Cumulant estimators used by OverICA.

All functions operate on data matrices ``X`` of shape ``(p, n)`` (observations
in columns), matching the original MATLAB convention.

Throughout, ``vec(M)`` means column-major flattening (MATLAB's ``M(:)``).

Every Python function below is preceded by a comment block containing the
corresponding MATLAB source (verbatim), annotated line-by-line. Inline
``# MATLAB: ...`` comments inside each body point each Python statement back
to its MATLAB equivalent and explain what it does.
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
    p, _ = X.shape                                           # MATLAB: p = size(X,1); n = size(X,2)

    omega = np.asarray(omega, dtype=float)
    if omega.ndim == 0 or omega.size == 1:                   # MATLAB: if numel(omega)==1
        omega = float(omega) * np.ones(p) / p                # MATLAB:   omega = omega*ones(p,1)/p
    omega = omega.reshape(p)

    proj = X.T @ omega                                       # MATLAB: proj = X' * omega  (n-vector)
    eproj = np.exp(proj)                                     # MATLAB: eproj = exp(proj)  (n-vector of weights)
    s = eproj.sum()                                          # MATLAB: sum(eproj)  (normalising constant)

    mean_omega = (X @ eproj) / s                             # MATLAB: Eomega = (X*eproj)/sum(eproj)   (E_ω[X])
    C = (X * eproj) @ X.T / s                                # MATLAB: C = X*diag(eproj)*X' / sum(eproj)
                                                             #         (broadcast multiply instead of sparse diag)
    C = C - np.outer(mean_omega, mean_omega)                 # MATLAB: C = C - Eomega*Eomega'  (covariance)
    return _vec_F(C)                                         # MATLAB: C = C(:)  (column-major flatten)


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
    Xc = X - X.mean(axis=1, keepdims=True)                   # MATLAB driver: X = X - repmat(mean(X,2),1,n)

    # V[:, i] = vec(x_i x_i^T) in column-major (F) order.
    # Build (p, p, n) outer products then flatten the first two axes in F order.
    outers = Xc[:, None, :] * Xc[None, :, :]                 # MATLAB kernel: temp[a+b*p] = data[i][a]*data[i][b]
                                                             #   (vectorised across all i; gives shape (p,p,n))
    V = outers.reshape(p * p, n, order="F")                  #   (now V[:,i] = vec(x_i x_i^T), col-major)

    M4 = (V @ V.T) / n                                       # MATLAB kernel: Q[a][b] += temp[a]*temp[b]; Q /= n
                                                             #   (rank-1 outer-product accumulation → V @ V.T)
    C = (Xc @ Xc.T) / n                                      # MATLAB kernel: C[a][b] += data[i][a]*data[i][b]; C/=n

    # Subtract the three product-of-covariances terms — the Gaussian "Wick"
    # contractions that need to be removed to get the genuine 4-th cumulant.
    vecC = _vec_F(C)                                         # MATLAB: vec(C) (column-major)
    term1 = np.outer(vecC, vecC)                             # MATLAB: C[a][b] * C[c][d]   (vec(C) · vec(C)')

    # The other two contractions need an index permutation: we form
    # T[a,b,c,d] = C[a,c]·C[b,d] (resp. C[a,d]·C[b,c]) as a 4-D tensor and
    # then flatten back to (p*p, p*p) in column-major order so the index
    # layout matches MATLAB's Q[a+b*p][c+d*p].
    C_ac_bd = np.einsum("ac,bd->abcd", C, C).reshape(p * p, p * p, order="F")  # MATLAB: -C[a][c]*C[b][d]
    C_ad_bc = np.einsum("ad,bc->abcd", C, C).reshape(p * p, p * p, order="F")  # MATLAB: -C[a][d]*C[b][c]

    return M4 - term1 - C_ac_bd - C_ad_bc                    # MATLAB: Q = (already-mirrored upper-tri Q)


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
    p, N = X.shape                                           # MATLAB: [p,N] = size(X)
    u = np.asarray(u, dtype=float).reshape(p)

    Xu = X.T @ u                                             # MATLAB: Xu = X' * u   (length-N projection)
    expXu = np.exp(1j * Xu)                                  # MATLAB: expXu = exp(1i*Xu)   (complex weights)
    M0 = expXu.sum() / N                                     # MATLAB: M0 = sum(expXu)/N    (char. fn. at u)
    M1 = (X @ expXu) / (N * M0)                              # MATLAB: M1 = (X*expXu)/(N*M0)  (gen. mean ∈ ℝᵖ)

    Xc = X - M1[:, None]                                     # MATLAB: XC = X - repmat(M1,1,N)  (gen. zero-mean)

    # Weighted outer products w_n (x_n x_n^T) — vectorised replacement for
    # the MATLAB ``for n=1:N`` loop that accumulates M2 and M4.
    outers = Xc[:, None, :] * Xc[None, :, :]                 # MATLAB loop: temp = xn * xn^T   (per-sample p×p)
    W = expXu                                                # MATLAB loop: weight = expXu(n)
    M2 = np.tensordot(outers, W, axes=([2], [0])) / (N * M0) # MATLAB: M2 = M2 + expXu(n)*temp, then /(N*M0)

    V = outers.reshape(p * p, N, order="F")                  # MATLAB: temp(:) is vec(xn xn^T) in col-major order
    # M4 = (1/(N*M0)) * sum_n w_n vec(x_n x_n^T) vec(x_n x_n^T)^T
    M4 = (V * W[None, :]) @ V.T / (N * M0)                   # MATLAB: M4 += (expXu(n)*temp(:))*temp(:)' / (N*M0)

    # Subtract the three M2 cross (Wick) terms to get the genuine cumulant.
    vecM2 = M2.reshape(-1, order="F")                        # MATLAB: vec(M2) in column-major
    term1 = np.outer(vecM2, vecM2)                           # MATLAB: M2(i3,i4)*M2   →  vec(M2)·vec(M2)'
    M2_ac_bd = np.einsum("ac,bd->abcd", M2, M2).reshape(p * p, p * p, order="F")  # MATLAB: M2(:,i3)*M2(i4,:)
    M2_ad_bc = np.einsum("ad,bc->abcd", M2, M2).reshape(p * p, p * p, order="F")  # MATLAB: M2(:,i4)*M2(i3,:)

    return M4 - term1 - M2_ac_bd - M2_ad_bc                  # MATLAB: temp = M4(:,icol) - <three Wick terms>
