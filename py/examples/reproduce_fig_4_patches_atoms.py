"""Reproduce Figure 4 (atoms learned from CIFAR-10 patches).

Port of ``reproduce_fig_4_patches_atoms.m``. Extracts ``p × p`` patches from a
batch of CIFAR-10 images (converted to grayscale), runs OverICA with the
generalized-covariance subspace estimator and semi-adaptive deflation, and
plots the recovered atoms.

Usage:

    # Single batch
    python examples/reproduce_fig_4_patches_atoms.py \\
        --data /path/to/cifar-10-batches-py/data_batch_1

    # Multiple batches (shell glob expansion or explicit list)
    python examples/reproduce_fig_4_patches_atoms.py \\
        --data /path/to/cifar-10-batches-py/data_batch_*

    python examples/reproduce_fig_4_patches_atoms.py \\
        --data /path/data_batch_1 /path/data_batch_2 /path/data_batch_3

If ``--data`` is omitted the script falls back to a small synthetic image
set (independent draws of low-rank-plus-noise images). The synthetic mode is
only useful for verifying that the pipeline works end-to-end; the atom
visualisation is not interesting.

Pooling several batches produces many millions of patches, so by default the
script randomly subsamples down to ``--max-patches`` (1,000,000) before
running OverICA. Use ``--max-patches 0`` to disable subsampling.

The CIFAR-10 Python pickle format is described at
https://www.cs.toronto.edu/~kriz/cifar.html
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from overica import overica  # noqa: E402


_RGB_LUMA = np.array([0.2989, 0.5870, 0.1140])


def _load_cifar_batch(path: str) -> np.ndarray:
    """Load a CIFAR-10 ``data_batch_*`` pickle and return ``(N, 32, 32, 3) uint8``."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"]
    return data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def load_cifar_batches(paths: list[str]) -> np.ndarray:
    """Load and concatenate one or more CIFAR-10 batch pickles."""
    arrs = []
    for path in paths:
        print(f"loading CIFAR batch from {path}")
        arrs.append(_load_cifar_batch(path))
    return np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]


def _to_grayscale(imgs: np.ndarray) -> np.ndarray:
    return (imgs.astype(np.float64) @ _RGB_LUMA)


def extract_patches(imgs_gray: np.ndarray, p: int) -> np.ndarray:
    """Extract every ``p × p`` patch from each image. Returns ``(p*p, N_patches)``.

    Uses ``np.lib.stride_tricks.sliding_window_view`` for a fully vectorised
    extraction (the equivalent of MATLAB's ``nlfilter(gg, [p p], @(b){b})``).
    """
    N, H, W = imgs_gray.shape
    n_y, n_x = H - p + 1, W - p + 1
    windows = np.lib.stride_tricks.sliding_window_view(imgs_gray, (p, p), axis=(1, 2))
    # ``windows`` has shape (N, n_y, n_x, p, p). We want (p*p, N*n_y*n_x) with
    # the inner p×p patch flattened in column-major order to match MATLAB.
    flat = np.moveaxis(windows, (3, 4), (-1, -2))            # → (N, n_y, n_x, p, p) col-major
    return flat.reshape(N * n_y * n_x, p * p).T              # → (p*p, N_patches)


def _maybe_subsample(X: np.ndarray, max_patches: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample patch columns to ``max_patches`` (0 disables)."""
    if max_patches <= 0 or X.shape[1] <= max_patches:
        return X
    idx = rng.choice(X.shape[1], size=max_patches, replace=False)
    idx.sort()
    print(f"  subsampling {X.shape[1]} patches → {max_patches}")
    return X[:, idx]


def _synthetic_patches(p: int, n_patches: int, rng: np.random.Generator) -> np.ndarray:
    """Cheap stand-in when no CIFAR-10 data is provided."""
    p2 = p * p
    k_latent = max(2 * p, 8)
    ds_true = rng.standard_normal((p2, k_latent))
    ds_true /= np.linalg.norm(ds_true, axis=0, keepdims=True)
    coeffs = rng.random((k_latent, n_patches)) * np.abs(rng.standard_normal((k_latent, n_patches)))
    X = ds_true @ coeffs + 0.05 * rng.standard_normal((p2, n_patches))
    return X


def plot_atoms(ds: np.ndarray, p: int, a: int, b: int, out_path: str) -> None:
    """Tile the columns of ``ds`` (each a ``p × p`` patch) on an ``a × b`` grid."""
    import matplotlib.pyplot as plt

    k = ds.shape[1]
    d1 = ds[:, 0]
    for i in range(1, k):
        if d1 @ ds[:, i] < 0:
            ds[:, i] = -ds[:, i]
    # Normalize each column by its peak absolute value (matching MATLAB).
    peaks = np.max(np.abs(ds), axis=0, keepdims=True)
    peaks = np.where(peaks == 0, 1.0, peaks)
    ds = ds / peaks

    fig, axes = plt.subplots(a, b, figsize=(b, a))
    axes = np.atleast_2d(axes)
    for i in range(a * b):
        ax = axes.flat[i]
        ax.axis("off")
        if i < k:
            patch = ds[:, i].reshape(p, p, order="F")
            ax.imshow(patch, cmap="gray", interpolation="nearest")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="one or more CIFAR-10 batch files (e.g. "
                             "`--data data_batch_1 data_batch_2`); shell "
                             "globs work via expansion (e.g. `data_batch_*`)")
    parser.add_argument("--p", type=int, default=7, help="patch size")
    parser.add_argument("--k", type=int, default=150,
                        help="number of atoms to recover (latent dimension)")
    parser.add_argument("--n-images", type=int, default=None,
                        help="cap the total number of CIFAR images used "
                             "across all batches (default: all)")
    parser.add_argument("--max-patches", type=int, default=1_000_000,
                        help="cap the number of patches actually fed to "
                             "OverICA (default: 1,000,000; 0 disables)")
    parser.add_argument("--rows", type=int, default=10,
                        help="grid rows for the atom plot")
    parser.add_argument("--cols", type=int, default=15,
                        help="grid cols for the atom plot")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for patch subsampling and OverICA RNG")
    parser.add_argument("--out", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "expres", "fig4_atoms.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    p, k = args.p, args.k

    if args.data:
        imgs = load_cifar_batches(args.data)
        if args.n_images is not None:
            imgs = imgs[: args.n_images]
        print(f"  {imgs.shape[0]} images (across {len(args.data)} batch"
              f"{'es' if len(args.data) != 1 else ''}) "
              f"-> patches of size {p}x{p}")
        gray = _to_grayscale(imgs)
        X = extract_patches(gray, p)
        print(f"  X has shape {X.shape}")
        X = _maybe_subsample(X, args.max_patches, rng)
    else:
        print("no --data given; generating synthetic patches for a smoke test")
        X = _synthetic_patches(p, 50_000, rng)

    XC = X - X.mean(axis=1, keepdims=True)

    opts = {"sdp": "semiada", "t": 0.1}
    print(f"running OverICA with k={k}, opts={opts}, n_patches={X.shape[1]}")
    ds_est, _, times, _ = overica(XC, k, opts, rng=rng)
    print(f"timings: {times}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_atoms(ds_est, p, args.rows, args.cols, args.out)


if __name__ == "__main__":
    main()
