"""Reproduce Figure 1 (phase transition) from the OverICA paper.

Port of ``reproduce_fig_1_phase_transition.m``. The MATLAB script uses CVX;
we use CVXPY instead. For each observed dimension ``p`` and latent dimension
``k``, the experiment samples a unit-norm mixing matrix ``ds`` and solves
``nruns`` instances of the rank-1 SDP relaxation

    maximize    traces' q
    subject to  B = sum_j q_j * vec^-1(Ds[:, j]),   trace(B) = 1,   B >> 0

where ``Ds[:, j] = vec(ds_j ds_j^T)`` and ``traces = Ds^T vec(u u^T)`` for a
random unit vector ``u``. A solve is counted as a *success* when one of the
atoms achieves ``Ds^T vec(B) ≈ 1``, i.e. the SDP recovers a true atom rather
than a mixture. The success rate is plotted as a heatmap over ``(p, k)``.

The full grid in the paper has 9 values of ``p`` and up to ~95 values of
``k`` with 10 runs each, which takes many hours. By default this script runs
a much smaller grid that still exhibits the phase transition; pass
``--full`` for the paper-scale configuration.

Each function below is preceded by the corresponding chunk of the MATLAB
source (verbatim) with line-by-line annotations, and Python statements carry
``# MATLAB: ...`` markers pointing back at their MATLAB equivalents.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _experiments import _ensure_dir, _cache_load, _cache_save  # noqa: E402

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from overica import Ds_from_ds, sample_mixing_matrix  # noqa: E402


def _import_cvxpy():
    try:
        import cvxpy as cp  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "This script needs cvxpy. Install it with "
            "`pip install cvxpy`."
        ) from e
    return cp


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``compute(p, ks, nruns)`` inside
# reproduce_fig_1_phase_transition.m, restricted to a single (p, k) cell.
#
#   ds = sample_mixing_matrix(p,k);                  % draw unit-norm mixing matrix
#   Ds = zeros(p^2,k);                               % flattened rank-1 atoms
#   for j = 1:k
#     Ds(:,j) = vec(ds(:,j)*ds(:,j)');               % vec(d_j d_j^T)  (col-major)
#   end
#   for irun = 1:nruns                               % independent trials:
#     u = randn(p,1); u = u/norm(u);                 %   random probe direction
#     G = u*u';                                      %   rank-1 probe matrix
#     traces = Ds' * vec(G);                         %   <D_j, G> for each atom
#     cvx_begin quiet                                %   CVX block (≡ CVXPY Problem)
#     expression B(p,p)                              %   B is an *expression*, not a variable
#     variable q(k,1)                                %   q is the only decision variable
#     maximize ( traces' * q  )                      %   objective: project onto true atoms
#     subject to                                     %   constraints:
#       B = zeros(p,p);                              %     build B = Σ_j q_j · reshape(Ds(:,j),p,p)
#       for j=1:k
#         B = B + q(j) * reshape( Ds(:,j), p,p );
#       end
#       trace(B) == 1                                %     trace-1 normalisation
#       B == semidefinite(p)                         %     PSD cone
#     cvx_end
#     if abs( max(Ds'*vec(B)) - 1 ) < 0.001          %   success: B is (≈) one of the true atoms
#       successes(i) = successes(i) + 1;
#     end
#   end
#   successes(i) = successes(i) / nruns;             % mean success rate at this (p,k)
# ----------------------------------------------------------------------------
def _phase_transition_cell(
    p: int, k: int, nruns: int, rng: np.random.Generator, cp_solver: str | None
) -> float:
    """Success fraction at a single ``(p, k)`` configuration."""
    cp = _import_cvxpy()

    ds = sample_mixing_matrix(p, k, rng=rng)                 # MATLAB: ds = sample_mixing_matrix(p,k)
    Ds = Ds_from_ds(ds)                                      # MATLAB: Ds(:,j) = vec(ds(:,j)*ds(:,j)')
    # Pre-build the p×p un-flattened atoms once so each CVXPY problem can
    # construct B without reshaping inside the inner loop.
    Dmats = [Ds[:, j].reshape(p, p, order="F") for j in range(k)]  # MATLAB: reshape(Ds(:,j), p, p)

    successes = 0
    for _ in range(nruns):                                   # MATLAB: for irun = 1:nruns
        u = rng.standard_normal(p)                           # MATLAB: u = randn(p,1)
        u = u / np.linalg.norm(u)                            # MATLAB: u = u/norm(u)
        G = np.outer(u, u)                                   # MATLAB: G = u*u'
        traces = Ds.T @ G.reshape(-1, order="F")             # MATLAB: traces = Ds' * vec(G)

        q = cp.Variable(k)                                   # MATLAB: variable q(k,1)
        # CVXPY expression equivalent of MATLAB's ``B = Σ q(j) * reshape(...)``:
        B = sum(q[j] * Dmats[j] for j in range(k))           # MATLAB: expression B(p,p); for j=1:k, B = B + q(j)*...
        constraints = [cp.trace(B) == 1, B >> 0]             # MATLAB: trace(B)==1; B == semidefinite(p)
        prob = cp.Problem(cp.Maximize(traces @ q), constraints)  # MATLAB: maximize ( traces' * q )
        # ``cvx_begin quiet ... cvx_end`` ⇒ ``prob.solve(verbose=False)``.
        try:
            if cp_solver is not None:
                prob.solve(solver=cp_solver, verbose=False)
            else:
                prob.solve(verbose=False)
        except cp.error.SolverError:
            # MATLAB CVX would surface a "Infeasible/Failed" status; we
            # treat that as a no-success trial.
            continue
        if q.value is None:
            continue

        B_val = sum(q.value[j] * Dmats[j] for j in range(k)) # MATLAB: B (value at solver exit)
        # MATLAB: if abs( max(Ds'*vec(B)) - 1 ) < 0.001 → counts as success.
        if abs(float((Ds.T @ B_val.reshape(-1, order="F")).max()) - 1.0) < 1e-3:
            successes += 1                                   # MATLAB: successes(i) = successes(i) + 1

    return successes / nruns                                 # MATLAB: successes(i) = successes(i) / nruns


# ----------------------------------------------------------------------------
# MATLAB reference: main loop in reproduce_fig_1_phase_transition.m
#
#   ptvals = zeros( length(ps), length(ks) );        % global phase-transition matrix
#   for i = 1:length(ps)
#     p = ps(i);
#     ksloc = ks4p{i};                               % per-p latent dim range
#     filepath = strcat( pwd, '/expres/pt/p', num2str(p),'.mat' );  % cache file
#     if exist( filepath, 'file' ) == 2              % cache hit:
#       rr = load(filepath);                         %   load it
#       successes = rr.successes;
#     else                                           % cache miss:
#       successes = compute(p, ksloc, nruns);        %   evaluate the (p, k) grid
#       save( filepath, 'successes' )                %   persist
#     end
#     for j = 1:length(ksloc)                        % fill the global matrix:
#       ptvals( logical(ps==p), logical(ks==ksloc(j)) ) = successes(j);
#     end
#   end
# ----------------------------------------------------------------------------
def run(
    ps: Sequence[int],
    ks_grid: Sequence[int],
    ks_max_per_p: dict[int, int],
    nruns: int,
    *,
    expdir: str,
    rng: np.random.Generator | None = None,
    cp_solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the success-rate matrix on the requested grid (with caching).

    ``ks_grid`` is a single shared list of latent dimensions; for each ``p``
    only the values ``k <= ks_max_per_p[p]`` are actually solved. Cells
    outside that range are left as NaN.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    _ensure_dir(expdir)
    ks_grid = list(ks_grid)
    # MATLAB: ptvals = zeros(length(ps), length(ks))  — we use NaN instead of
    # zero so we can render unevaluated cells (k > kmax(p)) as grey.
    pt = np.full((len(ps), len(ks_grid)), np.nan)

    for pi, p in enumerate(ps):                              # MATLAB: for i = 1:length(ps)
        kmax = ks_max_per_p[p]                               # MATLAB: ksloc = ks4p{i}  (max k for this p)
        ksloc = [k for k in ks_grid if k <= kmax]
        cache_path = os.path.join(expdir, f"pt_p{p}.pkl")    # MATLAB: filepath = strcat(pwd,'/expres/pt/p',num2str(p),'.mat')
        cached = _cache_load(cache_path)
        if cached is not None and list(cached["ks"]) == list(ksloc):  # MATLAB: if exist(filepath, 'file')==2
            successes = cached["successes"]                  # MATLAB: rr = load(filepath); successes = rr.successes
        else:
            successes = np.zeros(len(ksloc))
            for j, k in enumerate(ksloc):                    # MATLAB: for i = 1:n (inner loop in `compute`)
                t0 = time.perf_counter()
                successes[j] = _phase_transition_cell(p, k, nruns, rng, cp_solver)  # MATLAB: compute(p, ksloc, nruns)
                elapsed = time.perf_counter() - t0
                print(
                    f"  p={p:3d} k={k:4d}  success={successes[j]:.2f} "
                    f"(elapsed {elapsed:.1f}s)"
                )
            _cache_save(cache_path, {"ks": list(ksloc), "successes": successes})  # MATLAB: save(filepath, 'successes')

        for j, k in enumerate(ksloc):                        # MATLAB: for j = 1:length(ksloc)
            pt[pi, ks_grid.index(k)] = successes[j]          # MATLAB: ptvals(logical(ps==p), logical(ks==ksloc(j))) = ...

    return np.array(ps), np.array(ks_grid), pt


# ----------------------------------------------------------------------------
# MATLAB reference: nested ``make_plot`` in reproduce_fig_1_phase_transition.m
#
#   ff=figure; hold on                                       % open figure, hold for overlays
#     ...                                                    % big-font, fullscreen formatting
#     pcolor( ptvals ), colorbar, colormap(flipud(gray))     % phase-transition heatmap
#     shading flat                                           % flat shading
#     xlim([1 95]); ylim([1 9])                              % fixed limits to match paper grid
#     yys = .5:1:9.5;                                        % y-coords for the reference curves
#     ys  = [0 10:5:50];                                     % grid values of p (10..50)
#     xxs = (ys .* (ys+1) / 2) / 10;                         % blue curve: k = p(p+1)/2 (cells)
#     zzs = (ys.^2 / 4) / 10;                                % red curve : k = p^2 / 4  (cells)
#     plot( xxs, yys, 'Color','b', 'LineWidth',5 )            % blue: complete/undercomplete bound
#     plot( zzs, yys, 'Color','r', 'LineWidth',5 )            % red: paper's guarantee threshold
#     set(gca,'YTick', 1:9); set(gca,'YTickLabel', 10:5:50)   % axis ticks
#     set(gca,'XTick', 10:10:90); set(gca,'XTickLabel', 100:100:900)
#     xlabel('Latent Dimension ($k$)', 'Interpreter', 'latex')
#     ylabel('Observed Dimension ($p$)', 'Interpreter', 'latex')
#     box on
# ----------------------------------------------------------------------------
def make_plot(ps: np.ndarray, ks: np.ndarray, pt: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 5))                  # MATLAB: ff = figure; hold on
    cmap = plt.get_cmap("gray_r")                            # MATLAB: colormap(flipud(gray))
    cmap.set_bad(color="lightgrey")                          # (Python-only: render NaN cells grey)
    masked = np.ma.array(pt, mask=np.isnan(pt))
    # MATLAB uses ``pcolor(ptvals); shading flat``; we use imshow + nearest
    # interpolation because that handles NaNs cleanly and gives one pixel
    # per (p, k) cell, matching the paper's discrete grid.
    img = ax.imshow(
        masked, cmap=cmap, aspect="auto", origin="lower",
        norm=mcolors.Normalize(vmin=0, vmax=1),
        extent=(float(ks[0]), float(ks[-1]), float(ps[0]) - 0.5, float(ps[-1]) + 0.5),
        interpolation="nearest",
    )                                                        # MATLAB: pcolor(ptvals); shading flat
    fig.colorbar(img, ax=ax, label="success rate")           # MATLAB: colorbar

    # Theoretical reference curves. The MATLAB script plots k as a function
    # of p (``xxs``, ``zzs`` versus ``yys``), each scaled by 10 to fit the
    # cell grid. We plot p as a function of k directly so the curves overlay
    # neatly on top of imshow in physical units.
    ks_dense = np.linspace(ks.min(), ks.max(), 400)
    # k = p(p + 1) / 2  (blue, complete/undercomplete boundary)
    p_blue = (-1 + np.sqrt(1 + 8 * ks_dense)) / 2            # MATLAB: xxs = (ys .* (ys+1) / 2) / 10
    ax.plot(ks_dense, p_blue, color="tab:blue", linewidth=2, label=r"$k=p(p+1)/2$")  # MATLAB: plot(xxs, yys, 'b')
    # k = p^2 / 4  (red, the paper's guarantee threshold)
    p_red = 2 * np.sqrt(ks_dense)                            # MATLAB: zzs = (ys.^2 / 4) / 10
    ax.plot(ks_dense, p_red, color="tab:red", linewidth=2, label=r"$k=p^2/4$")       # MATLAB: plot(zzs, yys, 'r')

    ax.set_xlim(ks.min(), ks.max())                          # MATLAB: xlim([1 95])  (in cell units)
    ax.set_ylim(ps.min() - 0.5, ps.max() + 0.5)              # MATLAB: ylim([1 9])
    ax.set_xlabel(r"Latent dimension $k$")                   # MATLAB: xlabel('Latent Dimension ($k$)', ...)
    ax.set_ylabel(r"Observed dimension $p$")                 # MATLAB: ylabel('Observed Dimension ($p$)', ...)
    ax.set_yticks(ps)                                        # MATLAB: set(gca,'YTick',1:9); 'YTickLabel',10:5:50
    ax.set_title("Phase transition of the OverICA SDP")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run the paper-scale configuration (slow)")
    parser.add_argument("--nruns", type=int, default=None,
                        help="number of independent SDP solves per (p, k) cell")
    parser.add_argument("--expdir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "expres", "pt"),
                        help="cache directory for per-p results")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "expres", "fig1_phase_transition.png"),
                        help="output figure path")
    parser.add_argument("--solver", type=str, default=None,
                        help="CVXPY solver name (default: CVXPY auto-selects)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.full:
        # Paper-scale configuration straight from reproduce_fig_1_phase_transition.m.
        ps = list(range(10, 51, 5))                          # MATLAB: ps = 10:5:50
        ks_grid = list(range(10, 951, 10))                   # MATLAB: ks = 10:10:950
        ks_max_per_p = {                                     # MATLAB: ks4p{1..9} = 10:10:{50,120,...,950}
            10: 50, 15: 120, 20: 210, 25: 220, 30: 400,
            35: 500, 40: 650, 45: 800, 50: 950,
        }
        nruns = args.nruns if args.nruns is not None else 10 # MATLAB: nruns = 10
    else:
        # Quick demo that still shows the phase transition shape (Python-only).
        ps = [8, 10, 12, 14]
        ks_grid = list(range(4, 65, 4))
        ks_max_per_p = {8: 32, 10: 50, 12: 60, 14: 64}
        nruns = args.nruns if args.nruns is not None else 3

    print(f"ps = {ps}")
    print(f"ks_grid = {ks_grid}")
    print(f"nruns = {nruns}, full={args.full}")

    rng = np.random.default_rng(0)
    ps_arr, ks_arr, pt = run(                                # MATLAB: main `for i = 1:length(ps)` loop
        ps, ks_grid, ks_max_per_p, nruns,
        expdir=args.expdir, rng=rng, cp_solver=args.solver,
    )

    _ensure_dir(os.path.dirname(args.out) or ".")
    make_plot(ps_arr, ks_arr, pt, args.out)                  # MATLAB: make_plot(ptvals)


if __name__ == "__main__":
    main()
