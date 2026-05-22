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


def _phase_transition_cell(
    p: int, k: int, nruns: int, rng: np.random.Generator, cp_solver: str | None
) -> float:
    """Success fraction at a single ``(p, k)`` configuration."""
    cp = _import_cvxpy()

    ds = sample_mixing_matrix(p, k, rng=rng)
    Ds = Ds_from_ds(ds)  # (p*p, k), column-major
    Dmats = [Ds[:, j].reshape(p, p, order="F") for j in range(k)]

    successes = 0
    for _ in range(nruns):
        u = rng.standard_normal(p)
        u = u / np.linalg.norm(u)
        G = np.outer(u, u)
        traces = Ds.T @ G.reshape(-1, order="F")

        q = cp.Variable(k)
        B = sum(q[j] * Dmats[j] for j in range(k))
        constraints = [cp.trace(B) == 1, B >> 0]
        prob = cp.Problem(cp.Maximize(traces @ q), constraints)
        try:
            if cp_solver is not None:
                prob.solve(solver=cp_solver, verbose=False)
            else:
                prob.solve(verbose=False)
        except cp.error.SolverError:
            continue
        if q.value is None:
            continue

        B_val = sum(q.value[j] * Dmats[j] for j in range(k))
        if abs(float((Ds.T @ B_val.reshape(-1, order="F")).max()) - 1.0) < 1e-3:
            successes += 1

    return successes / nruns


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
    pt = np.full((len(ps), len(ks_grid)), np.nan)

    for pi, p in enumerate(ps):
        kmax = ks_max_per_p[p]
        ksloc = [k for k in ks_grid if k <= kmax]
        cache_path = os.path.join(expdir, f"pt_p{p}.pkl")
        cached = _cache_load(cache_path)
        if cached is not None and list(cached["ks"]) == list(ksloc):
            successes = cached["successes"]
        else:
            successes = np.zeros(len(ksloc))
            for j, k in enumerate(ksloc):
                t0 = time.perf_counter()
                successes[j] = _phase_transition_cell(p, k, nruns, rng, cp_solver)
                elapsed = time.perf_counter() - t0
                print(
                    f"  p={p:3d} k={k:4d}  success={successes[j]:.2f} "
                    f"(elapsed {elapsed:.1f}s)"
                )
            _cache_save(cache_path, {"ks": list(ksloc), "successes": successes})

        for j, k in enumerate(ksloc):
            pt[pi, ks_grid.index(k)] = successes[j]

    return np.array(ps), np.array(ks_grid), pt


def make_plot(ps: np.ndarray, ks: np.ndarray, pt: np.ndarray, out_path: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("gray_r")
    cmap.set_bad(color="lightgrey")  # NaN cells (k > p^2 etc.) render as grey
    masked = np.ma.array(pt, mask=np.isnan(pt))
    img = ax.imshow(
        masked, cmap=cmap, aspect="auto", origin="lower",
        norm=mcolors.Normalize(vmin=0, vmax=1),
        extent=(float(ks[0]), float(ks[-1]), float(ps[0]) - 0.5, float(ps[-1]) + 0.5),
        interpolation="nearest",
    )
    fig.colorbar(img, ax=ax, label="success rate")

    # Theoretical reference curves (drawn as p as a function of k).
    ks_dense = np.linspace(ks.min(), ks.max(), 400)
    # k = p(p + 1) / 2  (blue, complete/undercomplete boundary)
    p_blue = (-1 + np.sqrt(1 + 8 * ks_dense)) / 2
    ax.plot(ks_dense, p_blue, color="tab:blue", linewidth=2, label=r"$k=p(p+1)/2$")
    # k = p^2 / 4  (red, the paper's guarantee threshold)
    p_red = 2 * np.sqrt(ks_dense)
    ax.plot(ks_dense, p_red, color="tab:red", linewidth=2, label=r"$k=p^2/4$")

    ax.set_xlim(ks.min(), ks.max())
    ax.set_ylim(ps.min() - 0.5, ps.max() + 0.5)
    ax.set_xlabel(r"Latent dimension $k$")
    ax.set_ylabel(r"Observed dimension $p$")
    ax.set_yticks(ps)
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
        ps = list(range(10, 51, 5))
        ks_grid = list(range(10, 951, 10))
        ks_max_per_p = {
            10: 50, 15: 120, 20: 210, 25: 220, 30: 400,
            35: 500, 40: 650, 45: 800, 50: 950,
        }
        nruns = args.nruns if args.nruns is not None else 10
    else:
        # Quick demo that still shows the phase transition shape.
        ps = [8, 10, 12, 14]
        ks_grid = list(range(4, 65, 4))
        ks_max_per_p = {8: 32, 10: 50, 12: 60, 14: 64}
        nruns = args.nruns if args.nruns is not None else 3

    print(f"ps = {ps}")
    print(f"ks_grid = {ks_grid}")
    print(f"nruns = {nruns}, full={args.full}")

    rng = np.random.default_rng(0)
    ps_arr, ks_arr, pt = run(
        ps, ks_grid, ks_max_per_p, nruns,
        expdir=args.expdir, rng=rng, cp_solver=args.solver,
    )

    _ensure_dir(os.path.dirname(args.out) or ".")
    make_plot(ps_arr, ks_arr, pt, args.out)


if __name__ == "__main__":
    main()
