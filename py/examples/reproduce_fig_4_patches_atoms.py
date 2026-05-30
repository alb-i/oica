"""Reproduce Figure 4 (atoms learned from CIFAR-10 patches).

Port of ``reproduce_fig_4_patches_atoms.m``. Extracts ``p × p`` patches from a
batch of CIFAR-10 images (converted to grayscale), runs OverICA with the
generalized-covariance subspace estimator and semi-adaptive deflation, and
plots the recovered atoms.

The original MATLAB code is included verbatim as comment blocks above each
corresponding Python function, with line-by-line annotations. Inline
``# MATLAB: ...`` comments inside the bodies point each Python line back to
its MATLAB counterpart.

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
    # Implements MATLAB's `rgb2gray(reshape(rr, 32, 32, 3))` in batch form:
    # both use the BT.601 luma coefficients (0.2989·R + 0.5870·G + 0.1140·B).
    return (imgs.astype(np.float64) @ _RGB_LUMA)


# ----------------------------------------------------------------------------
# MATLAB reference: reproduce_fig_4_patches_atoms.m (nested ``get_patches``)
#
#   function X = get_patches(data, p)
#     sr = 32; sc = 32;                                 % CIFAR image size (rows × cols)
#     [N, ~] = size(data);                              % N = number of images in the batch
#     d = floor(p/2) + mod(p,2) - 1;                    % half-width index offset
#     step = (sr - 2*d - 1)*(sc - 2*d - 1);             % patches per image (= 32·32 with nlfilter padding)
#     X = zeros( p*p, N * step );                       % preallocate patch matrix (p²×N·step)
#     ind = 1;                                          % running column index into X
#     for n = 1:N                                       % for each image:
#       disp(['n = ', num2str(n)])                      %   progress print
#       rr = data(n,:);                                 %   1×3072 row (R…R G…G B…B)
#       gg = rgb2gray( reshape( rr, 32, 32, 3 ) );      %   reshape to 32×32×3 then convert to grayscale
#       pp = nlfilter(gg, [p p], @(block) {block});     %   slide a p×p window over every pixel,
#                                                       %   returning a 32×32 cell of p×p patches
#       for i = 1:32                                    %   for each window position:
#         for j = 1:32
#           patch = pp{i,j};                            %     pull out the patch (p×p, with zero padding)
#           X(:,ind) = patch(:);                        %     flatten column-major and store as a column
#           ind = ind + 1;                              %     advance write pointer
#         end
#       end
#     end
#   end
#
# MATLAB's ``nlfilter`` implicitly zero-pads the image so every pixel becomes
# a window centre (yielding 32·32 = 1024 patches per image). By default this
# Python port emits only the (32−p+1)² *interior* windows (= 676 patches per
# image for p=7), but with ``pad=True`` it reproduces the MATLAB behaviour
# exactly: pad ``floor((p-1)/2)`` rows/cols on the top/left and the rest on
# the bottom/right, so each image still yields ``H·W`` patches. The MATLAB
# code preallocates ``X`` with only ``N·step`` columns, but the inner loop
# writes ``1024·N`` of them — MATLAB silently grows the array on overrun, so
# the final ``X`` matches the padded count.
# ----------------------------------------------------------------------------
def extract_patches(
    imgs_gray: np.ndarray, p: int, *, pad: bool = False
) -> np.ndarray:
    """Extract every ``p × p`` patch from each image.

    Parameters
    ----------
    imgs_gray : (N, H, W) ndarray
        Grayscale images.
    p : int
        Patch side length.
    pad : bool, default False
        If True, zero-pad each image before extraction so that every pixel
        becomes a window centre — matching MATLAB's ``nlfilter`` (yields
        ``N·H·W`` patches). If False, only the ``N·(H−p+1)·(W−p+1)``
        *interior* windows are returned (no padding).

    Returns
    -------
    X : (p*p, N_patches) ndarray
        Patches stacked as columns, each flattened in column-major order
        (matching MATLAB's ``patch(:)``).
    """
    if pad:
        pad_lo = (p - 1) // 2                                # MATLAB nlfilter: floor((p-1)/2) on top/left
        pad_hi = (p - 1) - pad_lo                            #                  the rest on bottom/right
        imgs_gray = np.pad(                                  # MATLAB nlfilter: implicit zero-padding
            imgs_gray, ((0, 0), (pad_lo, pad_hi), (pad_lo, pad_hi)),
        )
    N, H, W = imgs_gray.shape                                # MATLAB: [N, ~] = size(data); sr = H; sc = W
    n_y, n_x = H - p + 1, W - p + 1                          # MATLAB: step = (sr-2d-1)*(sc-2d-1) (no padding)
                                                             #         or 32·32 = 1024 (with padding)
    windows = np.lib.stride_tricks.sliding_window_view(      # MATLAB: pp = nlfilter(gg, [p p], @(b){b})
        imgs_gray, (p, p), axis=(1, 2),                      #   one entry per window position, no Python loop
    )
    # ``windows`` has shape (N, n_y, n_x, p, p). We want (p*p, N*n_y*n_x) with
    # the inner p×p patch flattened in column-major order to match MATLAB.
    flat = np.moveaxis(windows, (3, 4), (-1, -2))            # MATLAB: equivalent of `patch(:)` (column-major)
    return flat.reshape(N * n_y * n_x, p * p).T              # MATLAB: X(:,ind) = patch(:); ind = ind + 1;


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


# ----------------------------------------------------------------------------
# MATLAB reference: reproduce_fig_4_patches_atoms.m (nested ``plot_atoms``)
#
#   function plot_atoms( ds, p, k, a, b )
#     d1 = ds(:,1);                                     % reference atom (first column)
#     for i = 2:k                                       % sign-align every other atom:
#       ds(:,i) = sign(d1'*ds(:,i))*ds(:,i);            %   flip if it points the other way
#     end
#     ds = ds./(ones(size(ds,1),1)*max(abs(ds)));       % normalize each column by its peak |value|
#     ff=figure; hold on                                % open a new figure, hold subsequent plots
#       screensize = get( groot, 'Screensize' );        %   query screen size
#       rr = min(screensize(3), screensize(4));         %   pick the smaller dimension
#       set(ff,'Position', [0 0 rr*b/a rr])             %   resize figure to keep cells square (b/a)
#       for i = 1:k                                     %   for each atom:
#         subplot(a,b,i)                                %     pick subplot cell (a×b grid, position i)
#         rr = reshape( ds(:,i), [p p] );               %     unflatten i-th atom to p×p (column-major)
#         pcolor(rr)                                    %     plot as a coloured grid
#         shading flat                                  %     flat shading (one colour per cell)
#         colormap('gray')                              %     grayscale palette
#         axis off                                      %     hide axes / ticks
#       end
#     hold off                                          % release plot hold
#   end
# ----------------------------------------------------------------------------
def plot_atoms(ds: np.ndarray, p: int, a: int, b: int, out_path: str) -> None:
    """Tile the columns of ``ds`` (each a ``p × p`` patch) on an ``a × b`` grid."""
    import matplotlib.pyplot as plt

    k = ds.shape[1]
    d1 = ds[:, 0]                                            # MATLAB: d1 = ds(:,1)
    for i in range(1, k):                                    # MATLAB: for i = 2:k
        if d1 @ ds[:, i] < 0:                                # MATLAB: sign(d1'*ds(:,i)) < 0
            ds[:, i] = -ds[:, i]                             # MATLAB: ds(:,i) = sign(...)*ds(:,i)
    # Normalize each column by its peak absolute value (matching MATLAB).
    peaks = np.max(np.abs(ds), axis=0, keepdims=True)        # MATLAB: max(abs(ds))   (row vector)
    peaks = np.where(peaks == 0, 1.0, peaks)                 # (defensive: avoid divide-by-zero)
    ds = ds / peaks                                          # MATLAB: ds./(ones(size(ds,1),1)*max(abs(ds)))

    fig, axes = plt.subplots(a, b, figsize=(b, a))           # MATLAB: ff = figure; set(ff,'Position', [...])
    axes = np.atleast_2d(axes)
    for i in range(a * b):                                   # MATLAB: for i = 1:k  (+ blank trailing cells)
        ax = axes.flat[i]                                    # MATLAB: subplot(a, b, i)
        ax.axis("off")                                       # MATLAB: axis off
        if i < k:
            patch = ds[:, i].reshape(p, p, order="F")        # MATLAB: rr = reshape(ds(:,i), [p p])
            ax.imshow(patch, cmap="gray", interpolation="nearest")
                                                             # MATLAB: pcolor(rr); shading flat; colormap('gray')
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
    parser.add_argument("--padded", action="store_true",
                        help="zero-pad each image so every pixel becomes a "
                             "patch centre (matches MATLAB's `nlfilter`, "
                             "yielding H·W patches per image instead of "
                             "(H-p+1)·(W-p+1))")
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


# ----------------------------------------------------------------------------
# MATLAB reference: reproduce_fig_4_patches_atoms.m (top-level driver)
#
#   function reproduce_fig_4_patches_atoms( data )
#     p = 7;                                            % patch size
#     X = get_patches(data, p);                         % p²×N matrix of grayscale patches
#     XC = X - repmat( mean(X,2), 1, size(X,2) );       % subtract per-feature mean (centre)
#     opts.('sdp') = 'semiada';                         % semi-adaptive deflation
#     opts.('t') = 0.1;                                 % gencov scale
#     k = 150;                                          % overcomplete latent dimension
#     rand('stat',0)                                    % seed legacy RNGs (reproducibility)
#     randn('stat',0)
#     ds_est = overica( XC, k, opts );                  % run OverICA
#     plot_atoms( ds_est, p, k, 10, 15 )                % 10×15 grid of atom thumbnails
#   end
#
# This Python ``main`` follows the same structure but adds CLI flags
# (`--p`, `--k`, `--rows`, `--cols`, `--seed`, ...) and supports loading
# multiple CIFAR batches plus optional random patch subsampling.
# ----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)                   # MATLAB: rand('stat',0); randn('stat',0)

    p, k = args.p, args.k                                    # MATLAB: p = 7; k = 150;

    if args.data:
        imgs = load_cifar_batches(args.data)                 # MATLAB: ``data`` argument (single batch)
        if args.n_images is not None:
            imgs = imgs[: args.n_images]
        print(f"  {imgs.shape[0]} images (across {len(args.data)} batch"
              f"{'es' if len(args.data) != 1 else ''}) "
              f"-> patches of size {p}x{p}")
        gray = _to_grayscale(imgs)                           # MATLAB: rgb2gray(reshape(rr, 32, 32, 3))
        X = extract_patches(gray, p, pad=args.padded)        # MATLAB: X = get_patches(data, p);
        print(f"  X has shape {X.shape}"
              + (" (zero-padded, H·W patches/image)" if args.padded
                 else " (interior windows only)"))
        X = _maybe_subsample(X, args.max_patches, rng)       # (no MATLAB equivalent: memory cap for pooled batches)
    else:
        print("no --data given; generating synthetic patches for a smoke test")
        X = _synthetic_patches(p, 50_000, rng)

    XC = X - X.mean(axis=1, keepdims=True)                   # MATLAB: XC = X - repmat(mean(X,2), 1, size(X,2))

    opts = {"sdp": "semiada", "t": 0.1}                      # MATLAB: opts.('sdp') = 'semiada'; opts.('t') = 0.1
    print(f"running OverICA with k={k}, opts={opts}, n_patches={X.shape[1]}")
    ds_est, _, times, _ = overica(XC, k, opts, rng=rng)      # MATLAB: ds_est = overica( XC, k, opts );
    print(f"timings: {times}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_atoms(ds_est, p, args.rows, args.cols, args.out)    # MATLAB: plot_atoms( ds_est, p, k, 10, 15 )


if __name__ == "__main__":
    main()
