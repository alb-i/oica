"""Reproduce Figures 2, 3, 6, 7 (synthetic experiments) from the OverICA paper.

Port of ``reproduce_fig_2_3_6_7_synthetic_exps.m``. Four experiments are run:

* **Figure 2** — ``experiment_asymptotic`` at ``p = 10`` across many ``k``.
  Uses the *exact* atom subspace ``Hs`` (built from the true mixing matrix),
  so only the deflation step is exercised.
* **Figure 3 (left)** — ``experiment_fixedk`` at ``p = 15, k = 30`` across
  small sample sizes ``n = 1k..10k``.
* **Figures 3 & 7** — ``experiment_fixedk`` at ``p = 15, k = 30`` across
  larger sample sizes ``n = 10k..200k``.
* **Figure 6** — ``experiment_fixedn`` at ``p = 20`` across ``k = 20..100``
  with ``n = 100k``.

The original code compares against FOOBI (proprietary, not bundled here)
and Fourier PCA. We skip FOOBI with a warning and keep the others.

Even the "small" experiments take a long time on a laptop, so a ``--quick``
flag is provided that subsamples the sweeps. The MATLAB defaults are used
with ``--full``.

Each per-figure helper below carries the corresponding MATLAB block
(verbatim) as a comment, with inline ``# MATLAB: ...`` annotations linking
each Python statement back to its MATLAB equivalent.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _experiments import (  # noqa: E402
    experiment_asymptotic,
    experiment_fixedk,
    experiment_fixedn,
    filter_algs,
    plot_with_errorbars,
)


# ---------------------------------------------------------------------------
# Per-figure runners
# ---------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# MATLAB reference: FIGURE 2 block of reproduce_fig_2_3_6_7_synthetic_exps.m
#
#   randn('state',0); rand('state',0);                % reproducibility (seed = 0)
#   p = 10;                                           % observed dimension
#   ks = 5:5:60;                                      % latent-dim sweep
#   nrep = 10;                                        % independent repetitions per k
#   algs = {'rand', 'foobi', 'oica-semiada'};         % algorithms to compare
#   experiment_asymptotic( p, ks, nrep, algs )        % the actual experiment driver
# ----------------------------------------------------------------------------
def figure2(outdir: str, *, quick: bool) -> None:
    p = 10                                                   # MATLAB: p = 10
    ks = [5, 10, 20, 30, 40, 50, 60] if quick else list(range(5, 61, 5))  # MATLAB: ks = 5:5:60
    nrep = 3 if quick else 10                                # MATLAB: nrep = 10
    algs = filter_algs(["rand", "foobi", "oica-semiada"])    # MATLAB: algs = {'rand','foobi','oica-semiada'}
                                                             # (FOOBI is silently dropped if not installed)

    print(f"\n[Figure 2] experiment_asymptotic  p={p} ks={ks} nrep={nrep}")
    res = experiment_asymptotic(                             # MATLAB: experiment_asymptotic( p, ks, nrep, algs )
        p, ks, nrep, algs, expdir=os.path.join(outdir, "asymp10"),
        rng=np.random.default_rng(0),                        # MATLAB: randn('state',0); rand('state',0)
    )

    fig, (axA, axF) = plt.subplots(1, 2, figsize=(14, 5))
    plot_with_errorbars(
        axA, res["ks"], res["aerr"],
        xname=r"Latent Dimension ($k$)", yname="A-Error", greenlines_p=p,
    )
    plot_with_errorbars(
        axF, res["ks"], res["ferr"],
        xname=r"Latent Dimension ($k$)", yname="F-Error", greenlines_p=p,
    )
    fig.suptitle(f"Figure 2: asymptotic, p={p}")
    fig.tight_layout()
    out = os.path.join(outdir, "fig2_asymp.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------------
# MATLAB reference: FIGURE 3 (first two plots) block
#
#   randn('state',0); rand('state',0);                        % seed
#   p = 15;                                                   % obs dim
#   k = 30;                                                   % latent dim
#   nrep = 10;                                                % repetitions per (n)
#   algs = { 'foobi', 'oica', 'fpca', 'rand' };               % algorithms
#   ns = 1e3 : 1e3 : 1e4;                                     % sample-size sweep (small n)
#   isshort = 1;                                              % use the "short" cache subdir
#   experiment_fixedk( p, k, nrep, algs, ns, isshort )
# ----------------------------------------------------------------------------
def figure3_short(outdir: str, *, quick: bool) -> None:
    p = 15                                                   # MATLAB: p = 15
    k = 30                                                   # MATLAB: k = 30
    ns = [1000, 3000, 5000, 7000, 10000] if quick else list(range(1000, 10001, 1000))  # MATLAB: ns = 1e3:1e3:1e4
    nrep = 3 if quick else 10                                # MATLAB: nrep = 10
    algs = filter_algs(["foobi", "oica", "fpca", "rand"])    # MATLAB: algs = {'foobi','oica','fpca','rand'}

    print(f"\n[Figure 3 short] experiment_fixedk  p={p} k={k} ns={ns} nrep={nrep}")
    res = experiment_fixedk(                                 # MATLAB: experiment_fixedk(p, k, nrep, algs, ns, isshort)
        p, k, nrep, algs, ns,
        expdir=os.path.join(outdir, "fixk30p15_short"),      # MATLAB: isshort == 1  ⇒  separate cache dir
        rng=np.random.default_rng(0),                        # MATLAB: randn('state',0); rand('state',0)
    )

    fig, (axA, axF) = plt.subplots(1, 2, figsize=(14, 5))
    plot_with_errorbars(
        axA, res["ns"] / 1000, res["aerr"],
        xname=r"Sample size ($n$; thousands)", yname="A-Error",
    )
    plot_with_errorbars(
        axF, res["ns"] / 1000, res["ferr"],
        xname=r"Sample size ($n$; thousands)", yname="F-Error",
    )
    fig.suptitle(f"Figure 3 (short): p={p}, k={k}")
    fig.tight_layout()
    out = os.path.join(outdir, "fig3_short.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------------
# MATLAB reference: FIGURES 3 AND 7 block
#
#   randn('state',0); rand('state',0);                        % seed
#   ns = 1e4 : 1e4 : 2e5;                                     % larger sample sizes
#   algs = { 'foobi', 'oica', 'fpca', 'oica-quad-semiada' };  % adds quad+semiada variant
#   isshort = 0;                                              % use the "long" cache subdir
#   experiment_fixedk( p, k, nrep, algs, ns, isshort )
# ----------------------------------------------------------------------------
def figure3_and_7(outdir: str, *, quick: bool) -> None:
    p = 15                                                   # MATLAB: p = 15  (re-used from above block)
    k = 30                                                   # MATLAB: k = 30
    ns = ([10_000, 50_000, 100_000, 200_000] if quick
          else list(range(10_000, 200_001, 10_000)))         # MATLAB: ns = 1e4:1e4:2e5
    nrep = 3 if quick else 10                                # MATLAB: nrep = 10
    algs = filter_algs(["foobi", "oica", "fpca", "oica-quad-semiada"])  # MATLAB: algs = {'foobi','oica','fpca','oica-quad-semiada'}

    print(f"\n[Figures 3 & 7] experiment_fixedk  p={p} k={k} ns={ns} nrep={nrep}")
    res = experiment_fixedk(                                 # MATLAB: experiment_fixedk(p, k, nrep, algs, ns, isshort)
        p, k, nrep, algs, ns,
        expdir=os.path.join(outdir, "fixk30p15"),            # MATLAB: isshort == 0  ⇒  long cache dir
        rng=np.random.default_rng(0),                        # MATLAB: randn('state',0); rand('state',0)
    )

    fig, (axA, axF) = plt.subplots(1, 2, figsize=(14, 5))
    plot_with_errorbars(
        axA, res["ns"] / 1000, res["aerr"],
        xname=r"Sample size ($n$; thousands)", yname="A-Error",
    )
    plot_with_errorbars(
        axF, res["ns"] / 1000, res["ferr"],
        xname=r"Sample size ($n$; thousands)", yname="F-Error",
    )
    fig.suptitle(f"Figures 3 & 7: p={p}, k={k}")
    fig.tight_layout()
    out = os.path.join(outdir, "fig3_fig7.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------------
# MATLAB reference: FIGURE 4 - RUNTIME block (called "figure_runtime" in Python)
#
#   randn('state',0); rand('state',0);                        % seed
#   p = 20;                                                   % obs dim
#   ks = 20:20:100;                                           % latent-dim sweep
#   n = 1e5;                                                  % fixed sample size
#   nrep = 10;                                                % repetitions per k
#   algs = {'foobi', 'fpca', 'oica'};                         % runtime comparison
#   experiment_fixedn( p, ks, nrep, algs, n );
# ----------------------------------------------------------------------------
def figure_runtime(outdir: str, *, quick: bool) -> None:
    p = 20                                                   # MATLAB: p = 20
    ks = [20, 40, 60, 80, 100] if quick else list(range(20, 101, 20))   # MATLAB: ks = 20:20:100
    n = 20_000 if quick else 100_000                         # MATLAB: n = 1e5
    nrep = 2 if quick else 10                                # MATLAB: nrep = 10
    algs = filter_algs(["foobi", "fpca", "oica"])            # MATLAB: algs = {'foobi','fpca','oica'}

    print(f"\n[Figure 4 runtime] experiment_fixedn  p={p} ks={ks} n={n} nrep={nrep}")
    res = experiment_fixedn(                                 # MATLAB: experiment_fixedn(p, ks, nrep, algs, n)
        p, ks, nrep, algs, n,
        expdir=os.path.join(outdir, f"fixn{n}p{p}"),
        rng=np.random.default_rng(0),                        # MATLAB: randn('state',0); rand('state',0)
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plot_with_errorbars(
        axes[0], res["ks"], res["aerr"],
        xname=r"Latent Dimension ($k$)", yname="A-Error", greenlines_p=p,
    )
    plot_with_errorbars(
        axes[1], res["ks"], res["ferr"],
        xname=r"Latent Dimension ($k$)", yname="F-Error", greenlines_p=p,
    )
    plot_with_errorbars(
        axes[2], res["ks"], res["times"],
        xname=r"Latent Dimension ($k$)", yname="Runtime (s)",
        greenlines_p=p, log_y=True,
    )
    fig.suptitle(f"Runtime experiment: p={p}, n={n}")
    fig.tight_layout()
    out = os.path.join(outdir, "fig_runtime.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="use the paper-scale configuration (slow)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "expres"),
                        help="output / cache directory")
    parser.add_argument(
        "--only", choices=("fig2", "fig3_short", "fig3_fig7", "runtime"),
        nargs="*", default=None,
        help="only run the listed figures (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    # MATLAB ``reproduce_fig_2_3_6_7_synthetic_exps`` runs the four blocks
    # unconditionally; here we expose ``--only`` to select subsets, which is
    # the only structural deviation from the original script.
    args = parse_args()
    quick = not args.full
    os.makedirs(args.outdir, exist_ok=True)

    selected = args.only or ("fig2", "fig3_short", "fig3_fig7", "runtime")
    if "fig2" in selected:
        figure2(args.outdir, quick=quick)                    # MATLAB: % FIGURE 2 block
    if "fig3_short" in selected:
        figure3_short(args.outdir, quick=quick)              # MATLAB: % FIGURE 3 (first two plots)
    if "fig3_fig7" in selected:
        figure3_and_7(args.outdir, quick=quick)              # MATLAB: % FIGURES 3 AND 7
    if "runtime" in selected:
        figure_runtime(args.outdir, quick=quick)             # MATLAB: % FIGURE 4 - RUNTIME


if __name__ == "__main__":
    main()
