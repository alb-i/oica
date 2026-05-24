"""Top-level ``overica`` driver function.

The Python function below mirrors the MATLAB ``overica.m`` driver. Each
helper / branch is preceded by a comment block containing the corresponding
MATLAB source (verbatim), annotated line-by-line.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import scipy.linalg

from .cumulants import gencov, quadricov
from .sdp import sdp_adaptive, sdp_cluster, sdp_semiada


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``set_default_opts`` inside overica.m
#
#   function opts = set_default_opts
#     opts.('sub') = 'gencov';                          % default subspace estimator
#     opts.('sdp') = 'semiada';                         % default deflation strategy
#   end
# ----------------------------------------------------------------------------
_DEFAULT_OPTS: dict[str, Any] = {"sub": "gencov", "sdp": "semiada"}


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``check_opts`` inside overica.m
#
#   function opts = check_opts(opts)
#     if ~isstruct(opts), opts = set_default_opts; end                  % must be a struct
#     if ~isfield( opts, 'sub' ), opts.('sub') = 'gencov'; end          % default sub
#     if ~isfield( opts, 'sdp' ), opts.('sdp') = 'semiada'; end         % default sdp
#     if isfield( opts, 'ctype' )                                       % validate ctype
#       ctype = opts.ctype;
#       if ~( strcmp( ctype, 'h' ) || strcmp( ctype, 'km' ) )
#         disp('The opts.ctype value is changed to h')
#         opts.('ctype') = 'h';
#       end
#     end
#     if isfield( opts, 'sub' )                                         % validate sub
#       sub = opts.sub;
#       if ~( strcmp( sub, 'quad' ) || strcmp( sub, 'gencov' ) )
#         disp('The opts.sub value is changed to gencov')
#         opts.('sub') = 'gencov';
#       end
#     end
#     if isfield( opts, 'sdp' )                                         % validate sdp
#       sdp = opts.sdp;
#       if ~( strcmp( sdp, 'ada' ) || strcmp( sdp, 'semiada' ) || strcmp( sdp, 'clust' ) )
#         disp('The opts.sdp value is changed to semiada')
#         opts.sdp = 'semiada';
#       end
#     end
#   end
# ----------------------------------------------------------------------------
def _check_opts(opts: dict[str, Any] | None) -> dict[str, Any]:
    if opts is None or not isinstance(opts, dict):
        return dict(_DEFAULT_OPTS)
    opts = dict(opts)
    opts.setdefault("sub", "gencov")
    opts.setdefault("sdp", "semiada")
    if opts["sub"] not in ("quad", "gencov"):
        print("The opts['sub'] value is changed to gencov")
        opts["sub"] = "gencov"
    if opts["sdp"] not in ("ada", "semiada", "clust"):
        print("The opts['sdp'] value is changed to semiada")
        opts["sdp"] = "semiada"
    if "ctype" in opts and opts["ctype"] not in ("h", "km"):
        print("The opts['ctype'] value is changed to h")
        opts["ctype"] = "h"
    return opts


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``estimate_gencovs`` inside overica.m
#
#   function C = estimate_gencovs(X, s, t0)
#     [p, n] = size(X);                                 % p = obs dim, n = samples
#     t = t0;                                           % per-iteration scale
#     C = zeros(p^2, s);                                % preallocate gencov columns
#     G0 = gencov(X,zeros(p,1)); G0 = G0(:);            % baseline G(0) (= covariance)
#     for i = 1:s
#       omega = randn(p,1);                             %   random direction
#       if t0 == -1, t = choose_t(X'*omega, n/20); end  %   special: auto-pick t (rarely used)
#       omega = t*omega;                                %   rescale direction
#       Gi = gencov(X,omega);                           %   evaluate generalized covariance
#       C(:,i) = Gi(:) - G0;                            %   store the *difference* from G(0)
#     end                                               %   (subtraction kills the Gaussian part)
#   end
# ----------------------------------------------------------------------------
def _estimate_gencovs(
    X: np.ndarray, s: int, t0: float, rng: np.random.Generator
) -> np.ndarray:
    """Estimate ``s`` mean-zero generalized covariances of ``X``.

    Each column of the returned matrix is ``vec(G(t*omega_i)) - vec(G(0))``,
    where ``G(omega)`` is the generalized covariance evaluated at ``omega``.
    """
    p, _ = X.shape
    C = np.zeros((p * p, s))
    G0 = gencov(X, np.zeros(p))
    for i in range(s):
        omega = rng.standard_normal(p)
        Gi = gencov(X, t0 * omega)
        C[:, i] = Gi - G0
    return C


# ----------------------------------------------------------------------------
# MATLAB reference: overica.m (top-level driver)
#
#   function [ds_est, Ds_est, times, Hs] = overica( X, k, opts )
#     if ~( nargin==2 || nargin==3), error('Wrong number of inputs'); end
#     if nargin==2, opts = set_default_opts; end        % fill in defaults
#     if nargin==3, opts = check_opts(opts); end        % validate user options
#     globtt = tic;                                     % start global timer
#     % ---- Stage 1: subspace estimator (cumulant) ----
#     if strcmp( opts.sub, 'quad' )                     % fourth-order cumulant branch
#       disp('Computing quadricovariance')
#       tt=tic; C = quadricov(X); toc(tt)               %   C is p²×p² (symmetric)
#     end
#     if strcmp( opts.sub, 'gencov' )                   % generalized-covariance branch
#       disp('Computing generalized covariances')
#       tt = tic;
#       s = 5*k;                                        %   default: 5·k gencovs
#       t = 0.05/sqrt( max( max( abs( cov(X') ) ) ) );  %   default scale: tied to data variance
#       if isfield( opts, 's' ), s = opts.s*k; end      %   user override for #gencovs
#       if isfield( opts, 't' ), t = opts.t; end        %   user override for scale
#       C = estimate_gencovs(X, s, t);                  %   C is p²×s
#       toc(tt)
#     end
#     toc(globtt)
#     cum_time = toc(globtt);                           % time spent on cumulant
#     % ---- Stage 2: SVD to get the subspace basis Hs ----
#     disp('Computing SVD')
#     [CU,~,~] = svds(C,k);                             % top-k left singular vectors
#     Hs = CU(:, 1:k);                                  % p²×k orthonormal basis
#     toc(globtt)
#     svd_time = toc(globtt) - cum_time;                % time spent on SVD
#     % ---- Stage 3: deflation (clust / ada / semiada) ----
#     if strcmp( opts.sdp, 'clust' )
#       disp('Deflation via clustering')
#       if isfield( opts, 'ctype' )
#         [ds_est, Ds_est] = sdp_cluster(Hs, k, opts.ctype);
#       else
#         [ds_est, Ds_est] = sdp_cluster(Hs, k);
#       end
#     end
#     if strcmp( opts.sdp, 'ada' )
#       disp('Adaptive deflation')
#       [ds_est, Ds_est] = sdp_adaptive(Hs, k);
#     end
#     if strcmp( opts.sdp, 'semiada' )
#       disp('Semiadaptive deflation')
#       if isfield( opts, 'ctype' )
#         [ds_est, Ds_est] = sdp_semiada(Hs, k, opts.ctype);
#       else
#         [ds_est, Ds_est] = sdp_semiada(Hs, k);
#       end
#     end
#     toc(globtt)
#     sdp_time = toc(globtt) - cum_time - svd_time;     % time spent on deflation
#     total_time = toc(globtt);
#     times.('total_time') = total_time;                % return per-stage timings
#     times.('cum_time')   = cum_time;
#     times.('svd_time')   = svd_time;
#     times.('sdp_time')   = sdp_time;
#   end
# ----------------------------------------------------------------------------
def overica(
    X: np.ndarray,
    k: int,
    opts: dict[str, Any] | None = None,
    *,
    rng: np.random.Generator | int | None = None,
    verbose: bool = True,
):
    """Overcomplete Independent Component Analysis via SDP.

    Parameters
    ----------
    X : (p, n) ndarray
        Data matrix with observations in columns.
    k : int
        Desired latent dimension. The paper's guarantees require
        ``k < p**2 / 4``.
    opts : dict, optional
        See :func:`overica.overica` docstring in the README for keys and
        defaults.
    rng : numpy.random.Generator or seed, optional
        Source of randomness for generalized covariance sampling and SDP
        initialisation.
    verbose : bool, default True
        If False, suppresses the per-stage progress prints.

    Returns
    -------
    ds_est : (p, k) ndarray
        Estimated atoms (mixing-matrix columns), unit-norm.
    Ds_est : (p*p, k) ndarray
        Estimated rank-one atoms ``vec(d_i d_i^T)`` (column-major).
    times : dict
        Timings for the cumulant, SVD and SDP stages.
    Hs : (p*p, k) ndarray
        Orthonormal basis of the estimated subspace.
    """
    opts = _check_opts(opts)

    if isinstance(rng, np.random.Generator):
        rng_inst = rng
    else:
        rng_inst = np.random.default_rng(rng)

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    X = np.asarray(X, dtype=float)

    glob_start = time.perf_counter()

    if opts["sub"] == "quad":
        log("Computing quadricovariance")
        t0 = time.perf_counter()
        C = quadricov(X)
        log(f"Elapsed: {time.perf_counter() - t0:.3f}s")
    else:
        log("Computing generalized covariances")
        t0 = time.perf_counter()
        # MATLAB default: t = 0.05 / sqrt(max(max(abs(cov(X')))))
        covX = np.cov(X)
        t_default = 0.05 / np.sqrt(np.max(np.abs(covX)))
        s = int(opts.get("s", 5)) * k
        t = float(opts.get("t", t_default))
        C = _estimate_gencovs(X, s, t, rng_inst)
        log(f"Elapsed: {time.perf_counter() - t0:.3f}s")

    cum_time = time.perf_counter() - glob_start
    log(f"Cumulant stage: {cum_time:.3f}s")

    log("Computing SVD")
    U, _, _ = scipy.linalg.svd(C, full_matrices=False)
    Hs = U[:, :k]
    svd_time = time.perf_counter() - glob_start - cum_time
    log(f"SVD stage: {svd_time:.3f}s")

    sdp_mode = opts["sdp"]
    ctype = opts.get("ctype")

    if sdp_mode == "clust":
        log("Deflation via clustering")
        ds_est, Ds_est = sdp_cluster(Hs, k, ctype if ctype else "h", rng=rng_inst)
    elif sdp_mode == "ada":
        log("Adaptive deflation")
        ds_est, Ds_est = sdp_adaptive(Hs, k)
    else:  # 'semiada'
        log("Semiadaptive deflation")
        ds_est, Ds_est = sdp_semiada(Hs, k, ctype if ctype else "h", rng=rng_inst)

    sdp_time = time.perf_counter() - glob_start - cum_time - svd_time
    total_time = time.perf_counter() - glob_start
    log(f"SDP stage: {sdp_time:.3f}s")
    log(f"Total: {total_time:.3f}s")

    times = {
        "total_time": total_time,
        "cum_time": cum_time,
        "svd_time": svd_time,
        "sdp_time": sdp_time,
    }
    return ds_est, Ds_est, times, Hs
