"""End-to-end OverICA demo on synthetic data.

Samples an overcomplete ICA model with heavy-tailed sources, runs OverICA on
the observed data, and reports the angular and Frobenius recovery errors.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ``overica`` importable when running this script directly from ``py/``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from overica import (
    a_error,
    f_error,
    overica,
    sample_from_ica_with_uniform_sources,
    sample_mixing_matrix,
)


def main() -> None:
    rng = np.random.default_rng(0)

    p, k, n = 6, 10, 50_000  # overcomplete: k > p, and k < p^2 / 4 = 9 ... bump k to 8
    k = 8
    print(f"p = {p}, k = {k}, n = {n}")

    ds = sample_mixing_matrix(p, k, rng=rng)
    X, _ = sample_from_ica_with_uniform_sources(ds, n, rng=rng)

    ds_est, Ds_est, times, Hs = overica(X, k, rng=rng)

    a_err, _ = a_error(ds_est, ds)
    f_err, _ = f_error(ds_est, ds)
    print()
    print(f"angular (cos) error : {a_err:.4f}")
    print(f"frobenius error     : {f_err:.4f}")
    print(f"timings             : {times}")


if __name__ == "__main__":
    main()
