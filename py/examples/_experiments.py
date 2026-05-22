"""Shared infrastructure for the ``reproduce_*`` scripts.

Provides:

* A small algorithm registry (``ALGORITHMS``) mapping the MATLAB-style alg
  names to Python callables, with their pretty display names and plot styles.
* The three experiment drivers (``experiment_asymptotic``,
  ``experiment_fixedk``, ``experiment_fixedn``) — direct ports of the MATLAB
  scripts in ``../scripts/``, with on-disk caching so that interrupted runs
  can be resumed cheaply.
* ``plot_with_errorbars`` — a small matplotlib helper that imitates the
  median-line / min-max-errorbar style of ``make_single_plot_cells.m``.

The defaults are kept faithful to the MATLAB code, but the
``reproduce_*`` scripts override them with smaller "quick" configurations so
that the demos finish in a reasonable amount of time on a laptop.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

# Make the ``overica`` package importable when running the scripts directly.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from overica import (  # noqa: E402  (sys.path manipulation above)
    Ds_from_ds,
    a_error,
    evaluation_recovery,
    f_error,
    fourier_pca,
    overica,
    quadricov,
    sample_from_ica_with_uniform_sources,
    sample_mixing_matrix,
    sdp_adaptive,
    sdp_cluster,
    sdp_semiada,
)


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------


@dataclass
class AlgSpec:
    name: str          # short identifier used in the original MATLAB code
    pretty: str        # display name used in plot legends
    color: str         # matplotlib colour
    marker: str        # matplotlib marker
    linestyle: str     # matplotlib linestyle
    needs_quadricov: bool = False
    needs_raw_X: bool = True
    needs_Hs_gencov: bool = False  # synthetic-perfect Hs (asymptotic experiment)


ALGORITHMS: dict[str, AlgSpec] = {
    "oica": AlgSpec(
        "oica", "OverICA", color="#cc0000", marker="*", linestyle="-",
    ),
    "oica-semiada": AlgSpec(
        "oica-semiada", "OverICA", color="#cc0000", marker="*", linestyle="-",
        needs_raw_X=False, needs_Hs_gencov=True,
    ),
    "oica-clust": AlgSpec(
        "oica-clust", "OICA(QC)", color="#cc00cc", marker="^", linestyle="--",
        needs_raw_X=False, needs_Hs_gencov=True,
    ),
    "oica-ada": AlgSpec(
        "oica-ada", "OICA(QA)", color="#e6b800", marker="o", linestyle=":",
        needs_raw_X=False, needs_Hs_gencov=True,
    ),
    "oica-quad-semiada": AlgSpec(
        "oica-quad-semiada", "OverICA(Q)", color="#3366cc", marker="*",
        linestyle="-", needs_quadricov=True,
    ),
    "fpca": AlgSpec(
        "fpca", "Fourier PCA", color="#0033aa", marker="d", linestyle="-",
    ),
    "foobi": AlgSpec(
        "foobi", "FOOBI", color="#196619", marker="^", linestyle="-",
        needs_quadricov=True,
    ),
    "rand": AlgSpec(
        "rand", "RAND", color="#cc7a00", marker="*", linestyle=":",
        needs_raw_X=False,
    ),
}


def _foobi_unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "FOOBI is proprietary and is not bundled with this repository. "
        "Contact Lieven De Lathauwer to obtain the MATLAB code; the Python "
        "port skips this baseline."
    )


def _run_one(alg: str, k: int, payload: dict, rng: np.random.Generator) -> dict:
    """Run a single algorithm on a prepared payload and return ``ds_est`` + time."""
    t0 = time.perf_counter()
    if alg in ("oica", "oica-gencov-semiada"):
        opts = {"sub": "gencov", "s": 10, "sdp": "semiada"}
        ds_est, _, _, _ = overica(payload["X"], k, opts, rng=rng, verbose=False)
    elif alg == "oica-semiada":
        ds_est, _ = sdp_semiada(payload["Hs"], k, "h", rng=rng)
    elif alg == "oica-clust":
        ds_est, _ = sdp_cluster(payload["Hs"], k, "h", rng=rng)
    elif alg == "oica-ada":
        ds_est, _ = sdp_adaptive(payload["Hs"], k)
    elif alg == "oica-quad-semiada":
        ds_est, _ = sdp_semiada(payload["Hs_quad"], k, "h", rng=rng)
    elif alg == "fpca":
        ds_est = fourier_pca(payload["X"], k, rng=rng)
    elif alg == "foobi":
        ds_est = _foobi_unavailable()
    elif alg == "rand":
        ds_est = sample_mixing_matrix(payload["p"], k, rng=rng)
    else:
        raise ValueError(f"unknown algorithm {alg!r}")
    return {"ds_est": ds_est, "time": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_with_errorbars(
    ax,
    xs: np.ndarray,
    series: dict[str, dict[str, np.ndarray]],
    *,
    xname: str,
    yname: str,
    greenlines_p: int | None = None,
    log_y: bool = False,
) -> None:
    """Median line + min/max error bars, one series per algorithm.

    ``series`` maps an algorithm short name to a dict with ``{"med", "lo",
    "hi"}`` arrays. Unknown algorithm names use neutral defaults.
    """
    for alg, data in series.items():
        spec = ALGORITHMS.get(alg, AlgSpec(alg, alg, "black", "o", "-"))
        med = np.asarray(data["med"], dtype=float)
        lo = np.asarray(data["lo"], dtype=float)
        hi = np.asarray(data["hi"], dtype=float)
        yerr = np.vstack([med - lo, hi - med])
        ax.errorbar(
            xs, med, yerr=yerr, label=spec.pretty,
            color=spec.color, marker=spec.marker, linestyle=spec.linestyle,
            linewidth=2.0, markersize=8, capsize=3,
        )

    if greenlines_p is not None:
        p = greenlines_p
        for thr in (p, p * (p - 1) / 2, p * (p + 1) / 2, p ** 2 / 4):
            if xs[0] <= thr <= xs[-1]:
                ax.axvline(thr, color="green", linewidth=1.5, alpha=0.7)

    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_load(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _cache_save(path: str, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------


def _aggregate(values: np.ndarray) -> dict[str, np.ndarray]:
    """``values`` of shape ``(n_x, nrep)`` -> dict with med/lo/hi over reps."""
    return {
        "med": np.median(values, axis=1),
        "lo": np.min(values, axis=1),
        "hi": np.max(values, axis=1),
    }


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def experiment_asymptotic(
    p: int,
    ks: Sequence[int],
    nrep: int,
    algs: Sequence[str],
    *,
    expdir: str,
    rng: np.random.Generator | None = None,
):
    """Port of ``scripts/experiment_asymptotic.m``.

    The "asymptotic" experiment uses the *exact* atom subspace ``Hs`` (built
    from the true mixing matrix), so the only error source is the deflation
    step. Returns per-(alg, k) aggregated A-error / F-error / recovery curves.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    _ensure_dir(expdir)
    ks = list(ks)

    aerr = {alg: np.zeros((len(ks), nrep)) for alg in algs}
    ferr = {alg: np.zeros((len(ks), nrep)) for alg in algs}
    recveck = {alg: [np.zeros(k) for k in ks] for alg in algs}
    theta = np.arccos(0.99)

    for ki, k in enumerate(ks):
        cache_path = os.path.join(expdir, f"asymp_k{k}.pkl")
        cached = _cache_load(cache_path)

        if cached is None:
            ds_list = [sample_mixing_matrix(p, k, rng=rng) for _ in range(nrep)]
            Hs_list = []
            for ds in ds_list:
                Ds = Ds_from_ds(ds)
                U, _, _ = np.linalg.svd(Ds @ Ds.T, full_matrices=False)
                Hs_list.append(U[:, :k])

            results_per_alg: dict[str, list[dict]] = {alg: [] for alg in algs}
            for irep in range(nrep):
                payload = {"p": p, "Hs": Hs_list[irep], "X": None}
                for alg in algs:
                    print(f"  k={k:3d} rep={irep+1}/{nrep} alg={alg}")
                    results_per_alg[alg].append(_run_one(alg, k, payload, rng))

            cached = {"ds_list": ds_list, "results": results_per_alg}
            _cache_save(cache_path, cached)

        ds_list = cached["ds_list"]
        results_per_alg = cached["results"]

        for alg in algs:
            for irep in range(nrep):
                ds = ds_list[irep]
                ds_est = results_per_alg[alg][irep]["ds_est"]
                aerr[alg][ki, irep] = a_error(ds_est, ds)[0]
                ferr[alg][ki, irep] = f_error(ds_est, ds)[0]
                nrec, recov, _ = evaluation_recovery(ds_est, ds, theta)
                recveck[alg][ki] += recov

    return {
        "ks": np.array(ks),
        "aerr": {alg: _aggregate(aerr[alg]) for alg in algs},
        "ferr": {alg: _aggregate(ferr[alg]) for alg in algs},
        "recveck": {alg: recveck[alg] for alg in algs},
        "raw": {"aerr": aerr, "ferr": ferr},
        "p": p,
        "nrep": nrep,
    }


def _prepare_payload(
    p: int,
    k: int,
    n: int,
    rng: np.random.Generator,
    *,
    needs_quadricov: bool,
):
    ds = sample_mixing_matrix(p, k, rng=rng)
    X, _ = sample_from_ica_with_uniform_sources(ds, n, rng=rng)
    X = X - X.mean(axis=1, keepdims=True)
    payload = {"p": p, "ds": ds, "X": X}
    if needs_quadricov:
        C = quadricov(X)
        U, _, _ = np.linalg.svd(C, full_matrices=False)
        payload["Hs_quad"] = U[:, :k]
    return payload


def experiment_fixedk(
    p: int,
    k: int,
    nrep: int,
    algs: Sequence[str],
    ns: Sequence[int],
    *,
    expdir: str,
    rng: np.random.Generator | None = None,
):
    """Port of ``scripts/experiment_fixedk.m`` (sample-size sweep)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    _ensure_dir(expdir)
    ns = list(ns)
    needs_quadricov = any(ALGORITHMS[a].needs_quadricov for a in algs if a in ALGORITHMS)

    aerr = {alg: np.zeros((len(ns), nrep)) for alg in algs}
    ferr = {alg: np.zeros((len(ns), nrep)) for alg in algs}

    for ni, n in enumerate(ns):
        cache_path = os.path.join(expdir, f"fixk_n{n}.pkl")
        cached = _cache_load(cache_path)

        if cached is None:
            payloads = [
                _prepare_payload(p, k, n, rng, needs_quadricov=needs_quadricov)
                for _ in range(nrep)
            ]
            results_per_alg: dict[str, list[dict]] = {alg: [] for alg in algs}
            for irep, payload in enumerate(payloads):
                for alg in algs:
                    print(f"  n={n:7d} rep={irep+1}/{nrep} alg={alg}")
                    results_per_alg[alg].append(_run_one(alg, k, payload, rng))

            cached = {
                "ds_list": [pl["ds"] for pl in payloads],
                "results": results_per_alg,
            }
            _cache_save(cache_path, cached)

        ds_list = cached["ds_list"]
        results_per_alg = cached["results"]

        for alg in algs:
            for irep in range(nrep):
                ds = ds_list[irep]
                ds_est = results_per_alg[alg][irep]["ds_est"]
                aerr[alg][ni, irep] = a_error(ds_est, ds)[0]
                ferr[alg][ni, irep] = f_error(ds_est, ds)[0]

    return {
        "ns": np.array(ns),
        "aerr": {alg: _aggregate(aerr[alg]) for alg in algs},
        "ferr": {alg: _aggregate(ferr[alg]) for alg in algs},
        "raw": {"aerr": aerr, "ferr": ferr},
        "p": p,
        "k": k,
        "nrep": nrep,
    }


def experiment_fixedn(
    p: int,
    ks: Sequence[int],
    nrep: int,
    algs: Sequence[str],
    n: int,
    *,
    expdir: str,
    rng: np.random.Generator | None = None,
):
    """Port of ``scripts/experiment_fixedn.m`` (latent-dim sweep)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    _ensure_dir(expdir)
    ks = list(ks)
    needs_quadricov = any(ALGORITHMS[a].needs_quadricov for a in algs if a in ALGORITHMS)

    aerr = {alg: np.zeros((len(ks), nrep)) for alg in algs}
    ferr = {alg: np.zeros((len(ks), nrep)) for alg in algs}
    times = {alg: np.zeros((len(ks), nrep)) for alg in algs}

    for ki, k in enumerate(ks):
        cache_path = os.path.join(expdir, f"fixn_k{k}.pkl")
        cached = _cache_load(cache_path)

        if cached is None:
            payloads = [
                _prepare_payload(p, k, n, rng, needs_quadricov=needs_quadricov)
                for _ in range(nrep)
            ]
            results_per_alg: dict[str, list[dict]] = {alg: [] for alg in algs}
            for irep, payload in enumerate(payloads):
                for alg in algs:
                    print(f"  k={k:3d} rep={irep+1}/{nrep} alg={alg}")
                    results_per_alg[alg].append(_run_one(alg, k, payload, rng))

            cached = {
                "ds_list": [pl["ds"] for pl in payloads],
                "results": results_per_alg,
            }
            _cache_save(cache_path, cached)

        ds_list = cached["ds_list"]
        results_per_alg = cached["results"]

        for alg in algs:
            for irep in range(nrep):
                ds = ds_list[irep]
                res = results_per_alg[alg][irep]
                ds_est = res["ds_est"]
                aerr[alg][ki, irep] = a_error(ds_est, ds)[0]
                ferr[alg][ki, irep] = f_error(ds_est, ds)[0]
                times[alg][ki, irep] = res["time"]

    return {
        "ks": np.array(ks),
        "aerr": {alg: _aggregate(aerr[alg]) for alg in algs},
        "ferr": {alg: _aggregate(ferr[alg]) for alg in algs},
        "times": {alg: _aggregate(np.maximum(times[alg], 1e-6)) for alg in algs},
        "raw": {"aerr": aerr, "ferr": ferr, "times": times},
        "p": p,
        "nrep": nrep,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Utility used by the reproduction scripts
# ---------------------------------------------------------------------------


def filter_algs(algs: Sequence[str], available: Callable[[str], bool] | None = None) -> list[str]:
    """Drop unavailable algorithms from a list (e.g. FOOBI when absent)."""
    out = []
    for alg in algs:
        if alg == "foobi":
            # Always skip with a clear warning.
            print("[skip] FOOBI is proprietary and not bundled; skipping.")
            continue
        if available is not None and not available(alg):
            print(f"[skip] {alg} unavailable, skipping.")
            continue
        out.append(alg)
    return out
