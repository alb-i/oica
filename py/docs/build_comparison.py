"""Generate ``matlab_python_comparison.html``: a side-by-side MATLAB vs
Python comparison of every ported function in the OverICA codebase.

The output is a single self-contained HTML file with embedded CSS — no
external assets needed. Known MATLAB / NumPy / SciPy / scikit-learn /
CVXPY / Matplotlib identifiers are turned into hyperlinks to their
upstream documentation, so the reader can verify the port quickly.

Usage::

    python py/docs/build_comparison.py

The script reads files relative to the repository root, regardless of
where it is invoked from.
"""

from __future__ import annotations

import html
import os
import re
import textwrap
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parent / "matlab_python_comparison.html"


# ---------------------------------------------------------------------------
# Documentation URLs
# ---------------------------------------------------------------------------
#
# Each entry is (search-pattern, url). The pattern is matched literally as a
# word; for dotted Python attribute access (``np.linalg.norm``) the dots are
# escaped at substitution time. Patterns are matched longest-first so more
# specific names (``np.linalg.norm``) win over their prefixes (``np.linalg``).

MATLAB_DOCS: dict[str, str] = {
    # math / array creation
    "randn": "https://www.mathworks.com/help/matlab/ref/randn.html",
    "rand": "https://www.mathworks.com/help/matlab/ref/rand.html",
    "zeros": "https://www.mathworks.com/help/matlab/ref/zeros.html",
    "ones": "https://www.mathworks.com/help/matlab/ref/ones.html",
    "eye": "https://www.mathworks.com/help/matlab/ref/eye.html",
    "sqrt": "https://www.mathworks.com/help/matlab/ref/sqrt.html",
    "exp": "https://www.mathworks.com/help/matlab/ref/exp.html",
    "linspace": "https://www.mathworks.com/help/matlab/ref/linspace.html",
    # array manipulation
    "sort": "https://www.mathworks.com/help/matlab/ref/sort.html",
    "cumsum": "https://www.mathworks.com/help/matlab/ref/cumsum.html",
    "find": "https://www.mathworks.com/help/matlab/ref/find.html",
    "max": "https://www.mathworks.com/help/matlab/ref/max.html",
    "min": "https://www.mathworks.com/help/matlab/ref/min.html",
    "sign": "https://www.mathworks.com/help/matlab/ref/sign.html",
    "abs": "https://www.mathworks.com/help/matlab/ref/abs.html",
    "real": "https://www.mathworks.com/help/matlab/ref/real.html",
    "imag": "https://www.mathworks.com/help/matlab/ref/imag.html",
    "ctranspose": "https://www.mathworks.com/help/matlab/ref/ctranspose.html",
    "reshape": "https://www.mathworks.com/help/matlab/ref/reshape.html",
    "repmat": "https://www.mathworks.com/help/matlab/ref/repmat.html",
    "diag": "https://www.mathworks.com/help/matlab/ref/diag.html",
    "sparse": "https://www.mathworks.com/help/matlab/ref/sparse.html",
    "norm": "https://www.mathworks.com/help/matlab/ref/norm.html",
    "size": "https://www.mathworks.com/help/matlab/ref/size.html",
    "length": "https://www.mathworks.com/help/matlab/ref/length.html",
    "numel": "https://www.mathworks.com/help/matlab/ref/numel.html",
    "mean": "https://www.mathworks.com/help/matlab/ref/mean.html",
    "cov": "https://www.mathworks.com/help/matlab/ref/cov.html",
    "sum": "https://www.mathworks.com/help/matlab/ref/sum.html",
    "floor": "https://www.mathworks.com/help/matlab/ref/floor.html",
    "mod": "https://www.mathworks.com/help/matlab/ref/mod.html",
    "logical": "https://www.mathworks.com/help/matlab/ref/logical.html",
    "setdiff": "https://www.mathworks.com/help/matlab/ref/double.setdiff.html",
    "unique": "https://www.mathworks.com/help/matlab/ref/double.unique.html",
    # linear algebra
    "qr": "https://www.mathworks.com/help/matlab/ref/qr.html",
    "svd": "https://www.mathworks.com/help/matlab/ref/double.svd.html",
    "svds": "https://www.mathworks.com/help/matlab/ref/svds.html",
    "eig": "https://www.mathworks.com/help/matlab/ref/eig.html",
    "eigs": "https://www.mathworks.com/help/matlab/ref/eigs.html",
    "orth": "https://www.mathworks.com/help/matlab/ref/orth.html",
    "trace": "https://www.mathworks.com/help/matlab/ref/trace.html",
    # trig
    "atan": "https://www.mathworks.com/help/matlab/ref/atan.html",
    "acos": "https://www.mathworks.com/help/matlab/ref/acos.html",
    # io / files
    "pwd": "https://www.mathworks.com/help/matlab/ref/pwd.html",
    "exist": "https://www.mathworks.com/help/matlab/ref/exist.html",
    "load": "https://www.mathworks.com/help/matlab/ref/load.html",
    "save": "https://www.mathworks.com/help/matlab/ref/save.html",
    "strcat": "https://www.mathworks.com/help/matlab/ref/strcat.html",
    "num2str": "https://www.mathworks.com/help/matlab/ref/num2str.html",
    "strcmp": "https://www.mathworks.com/help/matlab/ref/strcmp.html",
    "disp": "https://www.mathworks.com/help/matlab/ref/disp.html",
    "error": "https://www.mathworks.com/help/matlab/ref/error.html",
    # timing & control
    "tic": "https://www.mathworks.com/help/matlab/ref/tic.html",
    "toc": "https://www.mathworks.com/help/matlab/ref/toc.html",
    # structures / types
    "isstruct": "https://www.mathworks.com/help/matlab/ref/isstruct.html",
    "isfield": "https://www.mathworks.com/help/matlab/ref/isfield.html",
    "struct": "https://www.mathworks.com/help/matlab/ref/struct.html",
    "cell": "https://www.mathworks.com/help/matlab/ref/cell.html",
    "nargin": "https://www.mathworks.com/help/matlab/ref/nargin.html",
    # plotting (used by reproduction scripts)
    "figure": "https://www.mathworks.com/help/matlab/ref/figure.html",
    "subplot": "https://www.mathworks.com/help/matlab/ref/subplot.html",
    "plot": "https://www.mathworks.com/help/matlab/ref/plot.html",
    "hold": "https://www.mathworks.com/help/matlab/ref/hold.html",
    "pcolor": "https://www.mathworks.com/help/matlab/ref/pcolor.html",
    "shading": "https://www.mathworks.com/help/matlab/ref/shading.html",
    "colormap": "https://www.mathworks.com/help/matlab/ref/colormap.html",
    "colorbar": "https://www.mathworks.com/help/matlab/ref/colorbar.html",
    "xlabel": "https://www.mathworks.com/help/matlab/ref/xlabel.html",
    "ylabel": "https://www.mathworks.com/help/matlab/ref/ylabel.html",
    "xlim": "https://www.mathworks.com/help/matlab/ref/xlim.html",
    "ylim": "https://www.mathworks.com/help/matlab/ref/ylim.html",
    "axis": "https://www.mathworks.com/help/matlab/ref/axis.html",
    "set": "https://www.mathworks.com/help/matlab/ref/set.html",
    "get": "https://www.mathworks.com/help/matlab/ref/get.html",
    "flipud": "https://www.mathworks.com/help/matlab/ref/flipud.html",
    "box": "https://www.mathworks.com/help/matlab/ref/box.html",
    "groot": "https://www.mathworks.com/help/matlab/ref/groot.html",
    # image processing
    "imread": "https://www.mathworks.com/help/matlab/ref/imread.html",
    "rgb2gray": "https://www.mathworks.com/help/images/ref/rgb2gray.html",
    "im2double": "https://www.mathworks.com/help/images/ref/im2double.html",
    "nlfilter": "https://www.mathworks.com/help/images/ref/nlfilter.html",
    # stats toolbox
    "kmeans": "https://www.mathworks.com/help/stats/kmeans.html",
    "clusterdata": "https://www.mathworks.com/help/stats/clusterdata.html",
    # CVX (not a MathWorks product, but always called from MATLAB)
    "cvx_begin": "http://cvxr.com/cvx/doc/basics.html",
    "cvx_end": "http://cvxr.com/cvx/doc/basics.html",
    "variable": "http://cvxr.com/cvx/doc/basics.html#variables",
    "expression": "http://cvxr.com/cvx/doc/basics.html#expressions",
    "maximize": "http://cvxr.com/cvx/doc/basics.html#objective",
    "minimize": "http://cvxr.com/cvx/doc/basics.html#objective",
    "semidefinite": "http://cvxr.com/cvx/doc/sdp.html",
    "vec": "http://cvxr.com/cvx/doc/funcref.html",
}

# Python identifiers (NumPy / SciPy / scikit-learn / CVXPY / Matplotlib /
# stdlib). Dotted names are matched literally.
PYTHON_DOCS: dict[str, str] = {
    # numpy core
    "np.asarray": "https://numpy.org/doc/stable/reference/generated/numpy.asarray.html",
    "np.array": "https://numpy.org/doc/stable/reference/generated/numpy.array.html",
    "np.zeros": "https://numpy.org/doc/stable/reference/generated/numpy.zeros.html",
    "np.empty": "https://numpy.org/doc/stable/reference/generated/numpy.empty.html",
    "np.ones": "https://numpy.org/doc/stable/reference/generated/numpy.ones.html",
    "np.eye": "https://numpy.org/doc/stable/reference/generated/numpy.eye.html",
    "np.full": "https://numpy.org/doc/stable/reference/generated/numpy.full.html",
    "np.arange": "https://numpy.org/doc/stable/reference/generated/numpy.arange.html",
    "np.linspace": "https://numpy.org/doc/stable/reference/generated/numpy.linspace.html",
    "np.sqrt": "https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html",
    "np.exp": "https://numpy.org/doc/stable/reference/generated/numpy.exp.html",
    "np.sort": "https://numpy.org/doc/stable/reference/generated/numpy.sort.html",
    "np.cumsum": "https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html",
    "np.flatnonzero": "https://numpy.org/doc/stable/reference/generated/numpy.flatnonzero.html",
    "np.max": "https://numpy.org/doc/stable/reference/generated/numpy.max.html",
    "np.min": "https://numpy.org/doc/stable/reference/generated/numpy.min.html",
    "np.maximum": "https://numpy.org/doc/stable/reference/generated/numpy.maximum.html",
    "np.minimum": "https://numpy.org/doc/stable/reference/generated/numpy.minimum.html",
    "np.argmax": "https://numpy.org/doc/stable/reference/generated/numpy.argmax.html",
    "np.argmin": "https://numpy.org/doc/stable/reference/generated/numpy.argmin.html",
    "np.argsort": "https://numpy.org/doc/stable/reference/generated/numpy.argsort.html",
    "np.abs": "https://numpy.org/doc/stable/reference/generated/numpy.absolute.html",
    "np.sign": "https://numpy.org/doc/stable/reference/generated/numpy.sign.html",
    "np.real": "https://numpy.org/doc/stable/reference/generated/numpy.real.html",
    "np.imag": "https://numpy.org/doc/stable/reference/generated/numpy.imag.html",
    "np.outer": "https://numpy.org/doc/stable/reference/generated/numpy.outer.html",
    "np.einsum": "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html",
    "np.tensordot": "https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html",
    "np.hstack": "https://numpy.org/doc/stable/reference/generated/numpy.hstack.html",
    "np.column_stack": "https://numpy.org/doc/stable/reference/generated/numpy.column_stack.html",
    "np.atleast_2d": "https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html",
    "np.where": "https://numpy.org/doc/stable/reference/generated/numpy.where.html",
    "np.clip": "https://numpy.org/doc/stable/reference/generated/numpy.clip.html",
    "np.mean": "https://numpy.org/doc/stable/reference/generated/numpy.mean.html",
    "np.cov": "https://numpy.org/doc/stable/reference/generated/numpy.cov.html",
    "np.sum": "https://numpy.org/doc/stable/reference/generated/numpy.sum.html",
    "np.diag": "https://numpy.org/doc/stable/reference/generated/numpy.diag.html",
    "np.isnan": "https://numpy.org/doc/stable/reference/generated/numpy.isnan.html",
    "np.arctan2": "https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html",
    "np.arccos": "https://numpy.org/doc/stable/reference/generated/numpy.arccos.html",
    "np.unique": "https://numpy.org/doc/stable/reference/generated/numpy.unique.html",
    "np.ma.array": "https://numpy.org/doc/stable/reference/generated/numpy.ma.array.html",
    # numpy.linalg
    "np.linalg.norm": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html",
    "np.linalg.qr": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html",
    "np.linalg.svd": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html",
    "np.linalg.eig": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html",
    "np.linalg.eigh": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html",
    "np.linalg.eigvalsh": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html",
    "np.linalg.solve": "https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html",
    # numpy.random
    "np.random.default_rng": "https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng",
    "np.random.Generator": "https://numpy.org/doc/stable/reference/random/generator.html",
    "rng.standard_normal": "https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html",
    "rng.random": "https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html",
    # numpy.lib.stride_tricks
    "np.lib.stride_tricks.sliding_window_view": "https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html",
    # scipy
    "scipy.linalg.svd": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html",
    "scipy.optimize.linear_sum_assignment": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html",
    "linear_sum_assignment": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html",
    "scipy.cluster.hierarchy.linkage": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html",
    "scipy.cluster.hierarchy.fcluster": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
    "linkage": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html",
    "fcluster": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html",
    # scikit-learn
    "KMeans": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
    # CVXPY
    "cp.Variable": "https://www.cvxpy.org/api_reference/cvxpy.expressions.html#cvxpy.expressions.variable.Variable",
    "cp.Problem": "https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Problem",
    "cp.Maximize": "https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Maximize",
    "cp.Minimize": "https://www.cvxpy.org/api_reference/cvxpy.problems.html#cvxpy.Minimize",
    "cp.trace": "https://www.cvxpy.org/api_reference/cvxpy.atoms.html#trace",
    "cp.error.SolverError": "https://www.cvxpy.org/tutorial/solvers/index.html",
    # matplotlib (used by reproduction scripts)
    "plt.subplots": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html",
    "plt.imshow": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html",
    "plt.get_cmap": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.get_cmap.html",
    "plt.close": "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.close.html",
    "mcolors.Normalize": "https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Normalize.html",
    # stdlib
    "time.perf_counter": "https://docs.python.org/3/library/time.html#time.perf_counter",
    "os.path.join": "https://docs.python.org/3/library/os.path.html#os.path.join",
    "os.makedirs": "https://docs.python.org/3/library/os.html#os.makedirs",
}


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_matlab_blocks(source: str) -> str:
    """Strip the ``# MATLAB reference: ...`` comment blocks from Python code.

    The blocks are delimited by lines that are either exactly ``# ----...---`` or
    start with ``# `` and only contain dashes. Everything between two such
    horizontal rules (inclusive) is stripped, including the body.
    """
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.lstrip()
        is_rule = bool(re.match(r"#\s*-{20,}\s*$", stripped))
        if is_rule:
            in_block = not in_block
            continue
        if in_block:
            continue
        out.append(line)
    return "".join(out)


def extract_python_function(source: str, name: str) -> str:
    """Return the body of ``def name(...):`` (and trailing definitions if any
    nested) from a Python source string. Includes the def line itself.
    """
    # Match ``def name(`` at column 0 (top-level only).
    pattern = re.compile(rf"^def {re.escape(name)}\b.*?(?=^def \w|\Z)",
                         re.MULTILINE | re.DOTALL)
    m = pattern.search(source)
    if not m:
        raise ValueError(f"could not find def {name} in source")
    body = m.group(0).rstrip() + "\n"
    return body


def extract_python_chunk(source: str, names: list[str]) -> str:
    """Extract several Python functions in order and concatenate them."""
    parts = []
    for n in names:
        parts.append(extract_python_function(source, n))
    return "\n\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Linkifier
# ---------------------------------------------------------------------------


def _linkify_segment(text: str, mapping: dict[str, str]) -> str:
    """HTML-escape ``text`` and wrap every recognised token in an <a>."""
    text = html.escape(text)
    items = sorted(mapping.items(), key=lambda kv: -len(kv[0]))
    for token, url in items:
        token_re = re.escape(token)
        pattern = re.compile(
            r"(?<![A-Za-z0-9_.])" + token_re + r"(?![A-Za-z0-9_])"
        )
        replacement = (
            f'<a href="{url}" target="_blank" rel="noopener">'
            f"{html.escape(token)}</a>"
        )
        text = pattern.sub(replacement, text)
    return text


def linkify(code: str, mapping: dict[str, str], *, comment_char: str) -> str:
    """HTML-escape ``code`` and turn each known identifier into an ``<a>``.

    Line comments (everything from ``comment_char`` onward, e.g. ``%`` for
    MATLAB or ``#`` for Python) are escaped but **not** linkified, so common
    English words that happen to be MATLAB function names (``hold``,
    ``find``, ``mean`` …) don't become spurious links.
    """
    out_lines: list[str] = []
    for line in code.splitlines():
        idx = line.find(comment_char)
        if idx == -1:
            out_lines.append(_linkify_segment(line, mapping))
        else:
            code_part = line[:idx]
            comment_part = line[idx:]
            out_lines.append(
                _linkify_segment(code_part, mapping)
                + html.escape(comment_part)
            )
    # Preserve trailing newline of the original input.
    suffix = "\n" if code.endswith("\n") else ""
    return "\n".join(out_lines) + suffix


# ---------------------------------------------------------------------------
# Comparison units
# ---------------------------------------------------------------------------


def _py(p: str) -> Path:
    return REPO_ROOT / p


SECTIONS = [
    # ---- Sampling ----
    {
        "id": "sample_mixing_matrix",
        "title": "sample_mixing_matrix",
        "blurb": "Draw a (p × k) Gaussian matrix with unit-norm columns.",
        "matlab_path": "sampling/sample_mixing_matrix.m",
        "python_path": "py/overica/sampling.py",
        "python_funcs": ["sample_mixing_matrix"],
    },
    {
        "id": "sample_orthogonal_matrix",
        "title": "sample_orthogonal_matrix",
        "blurb": "Sample a random orthogonal (k × k) matrix via the QR of a uniform-random matrix.",
        "matlab_path": "sampling/sample_orthogonal_matrix.m",
        "python_path": "py/overica/sampling.py",
        "python_funcs": ["sample_orthogonal_matrix"],
    },
    {
        "id": "sample_from_ica_with_uniform_sources",
        "title": "sample_from_ica_with_uniform_sources",
        "blurb": "Draw an ICA-mixed dataset using the paper's heavy-tailed source distribution.",
        "matlab_path": "sampling/sample_from_ica_with_uniform_sources.m",
        "python_path": "py/overica/sampling.py",
        "python_funcs": ["sample_from_ica_with_uniform_sources"],
    },
    # ---- Cumulants ----
    {
        "id": "gencov",
        "title": "gencov — Generalized Covariance",
        "blurb": "Estimate E_ω[X X^T] − E_ω[X] E_ω[X]^T using exponential reweighting.",
        "matlab_path": "cumulants/gencov.m",
        "python_path": "py/overica/cumulants.py",
        "python_funcs": ["gencov"],
    },
    {
        "id": "quadricov",
        "title": "quadricov — 4th-order Cumulant",
        "blurb": (
            "Fourth-order cumulant of centred data. The MATLAB driver is a "
            "thin wrapper around the <code>quadricov_in.cpp</code> MEX "
            "kernel; the Python port replaces the C kernel with a "
            "vectorised NumPy outer-product trick."
        ),
        "matlab_path": "cumulants/quadricov.m",
        "matlab_extra_path": "cumulants/quadricov_in.cpp",
        "python_path": "py/overica/cumulants.py",
        "python_funcs": ["quadricov"],
    },
    {
        "id": "genquadricov",
        "title": "genquadricov — Generalized 4th-order Cumulant",
        "blurb": "Generalized quadricovariance at evaluation point u (complex-valued).",
        "matlab_path": "cumulants/genquadricov.m",
        "python_path": "py/overica/cumulants.py",
        "python_funcs": ["genquadricov"],
    },
    # ---- SDP machinery ----
    {
        "id": "proj_simplex",
        "title": "proj_simplex — Projection onto the Probability Simplex",
        "blurb": "Duchi et al. linear-time projection onto {w ≥ 0, Σw ≤ 1}.",
        "matlab_path": "sdp/proj_simplex.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["proj_simplex"],
    },
    {
        "id": "extract_largest_eigenvector",
        "title": "extract_largest_eigenvector",
        "blurb": "Return the eigenvector of (D + D^T)/2 whose eigenvalue has the largest absolute value.",
        "matlab_path": "sdp/extract_largest_eigenvector.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["extract_largest_eigenvector"],
    },
    {
        "id": "extract_basis",
        "title": "extract_basis",
        "blurb": "Full QR to split a subspace and its orthogonal complement.",
        "matlab_path": "sdp/extract_basis.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["extract_basis"],
    },
    {
        "id": "solve_relaxation",
        "title": "solve_relaxation_mezcal_approx_fista — FISTA Solver",
        "blurb": (
            "Inner FISTA solver for the trace-1 PSD relaxation. Alternates a "
            "gradient step with an eigendecomposition + projection onto the "
            "probability simplex, and uses Nesterov momentum."
        ),
        "matlab_path": "sdp/solve_relaxation_mezcal_approx_fista.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["solve_relaxation_mezcal_approx_fista"],
    },
    {
        "id": "majorize_minimize",
        "title": "majorize_minimize",
        "blurb": "Outer majorize-minimize loop that iterates the FISTA solver until the iterate becomes rank-one.",
        "matlab_path": "sdp/majorize_minimize.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["majorize_minimize"],
    },
    {
        "id": "adaptive_deflation",
        "title": "adaptive_deflation",
        "blurb": "Greedy deflation that grows the atom set by one column per iteration.",
        "matlab_path": "sdp/adaptive_deflation.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["adaptive_deflation"],
    },
    {
        "id": "cluster_Dss",
        "title": "cluster_Dss",
        "blurb": "Sign-align candidate atoms and run hierarchical / k-means clustering, keeping one representative per cluster.",
        "matlab_path": "sdp/cluster_Dss.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["cluster_Dss", "_extract_clusters"],
    },
    {
        "id": "sdp_cluster",
        "title": "sdp_cluster",
        "blurb": "Clustering-based deflation: draws 3k candidates, clusters them down to k.",
        "matlab_path": "sdp/sdp_cluster.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["sdp_cluster"],
    },
    {
        "id": "sdp_adaptive",
        "title": "sdp_adaptive",
        "blurb": "Adaptive deflation, top-level wrapper.",
        "matlab_path": "sdp/sdp_adaptive.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["sdp_adaptive"],
    },
    {
        "id": "sdp_semiada",
        "title": "sdp_semiada",
        "blurb": "Semi-adaptive deflation: cluster first, keep well-separated atoms, then top up with adaptive deflation.",
        "matlab_path": "sdp/sdp_semiada.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["sdp_semiada"],
    },
    {
        "id": "Ds_from_ds",
        "title": "Ds_from_ds / approx_ds_from_Ds — Atom Conversion Helpers",
        "blurb": "Convert between the (p × k) atoms ds and the flattened (p² × k) rank-1 outer products Ds.",
        "matlab_path": "helpers/Ds_from_ds.m",
        "matlab_extra_path": "helpers/approx_ds_from_Ds.m",
        "python_path": "py/overica/sdp.py",
        "python_funcs": ["Ds_from_ds", "approx_ds_from_Ds"],
    },
    # ---- Top-level driver ----
    {
        "id": "overica",
        "title": "overica — Top-level Driver",
        "blurb": (
            "Stages: (1) cumulant estimation (quadricov or generalized "
            "covariances), (2) SVD to obtain a subspace basis Hs, (3) SDP "
            "deflation. Both the MATLAB and Python versions ship the same "
            "<code>check_opts</code> defaulting logic."
        ),
        "matlab_path": "overica.m",
        "python_path": "py/overica/algorithm.py",
        "python_funcs": ["_check_opts", "_estimate_gencovs", "overica"],
    },
    # ---- Evaluation ----
    {
        "id": "evaluation_perf",
        "title": "evaluation_perf — Atom-recovery Metrics",
        "blurb": (
            "Optimal sign-aware bipartite matching followed by an L1/L2/"
            "Frobenius/cosine cost. The MATLAB code uses its own "
            "<code>HungarianBipartiteMatching</code>; the Python port "
            "delegates to <code>scipy.optimize.linear_sum_assignment</code>."
        ),
        "matlab_path": "helpers/evaluation_perf.m",
        "python_path": "py/overica/evaluation.py",
        "python_funcs": [
            "_normalize_2",
            "_update_ds_est",
            "_perf_l2",
            "_perf_fro",
            "_perf_l1",
            "_perf_cos",
            "evaluation_perf",
        ],
    },
    {
        "id": "evaluation_recovery",
        "title": "evaluation_recovery + a_error / f_error",
        "blurb": "Count how many atoms are recovered within an angular threshold; plus the two convenience wrappers.",
        "matlab_path": "helpers/evaluation_recovery.m",
        "matlab_extra_paths": ["helpers/a_error.m", "helpers/f_error.m"],
        "python_path": "py/overica/evaluation.py",
        "python_funcs": ["a_error", "f_error", "evaluation_recovery"],
    },
    # ---- Comparators ----
    {
        "id": "fourier_pca",
        "title": "fourier_pca2 — Fourier PCA Baseline",
        "blurb": "Reference baseline used in the synthetic-data experiments.",
        "matlab_path": "comparison/fpca/fourier_pca2.m",
        "python_path": "py/overica/comparators.py",
        "python_funcs": ["fourier_pca"],
    },
    # ---- Reproduction scripts ----
    {
        "id": "reproduce_fig_1",
        "title": "reproduce_fig_1_phase_transition",
        "blurb": "Phase-transition heatmap of the rank-1 SDP relaxation. MATLAB uses CVX; Python uses CVXPY.",
        "matlab_path": "reproduce_fig_1_phase_transition.m",
        "python_path": "py/examples/reproduce_fig_1_phase_transition.py",
        "python_funcs": ["_phase_transition_cell", "run", "make_plot", "main"],
    },
    {
        "id": "reproduce_fig_234",
        "title": "reproduce_fig_2_3_6_7_synthetic_exps",
        "blurb": "Synthetic-data experiments (asymptotic / fixed-k / fixed-n). FOOBI is skipped because it is not publicly available.",
        "matlab_path": "reproduce_fig_2_3_6_7_synthetic_exps.m",
        "python_path": "py/examples/reproduce_fig_2_3_6_7_synthetic_exps.py",
        "python_funcs": ["figure2", "figure3_short", "figure3_and_7", "figure_runtime", "main"],
    },
    {
        "id": "reproduce_fig_4",
        "title": "reproduce_fig_4_patches_atoms",
        "blurb": "Learn p × p image atoms from CIFAR-10 patches. The Python port supports zero-padding (matching MATLAB's nlfilter) and multi-batch loading.",
        "matlab_path": "reproduce_fig_4_patches_atoms.m",
        "python_path": "py/examples/reproduce_fig_4_patches_atoms.py",
        "python_funcs": ["_to_grayscale", "extract_patches", "plot_atoms", "main"],
    },
]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


CSS = """
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  color: #1f2328;
  background: #f6f8fa;
  line-height: 1.55;
}
header {
  background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
  color: white;
  padding: 32px 48px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
}
header h1 {
  margin: 0 0 8px 0;
  font-size: 28px;
}
header p {
  margin: 4px 0;
  font-size: 14px;
  opacity: 0.85;
}
header a {
  color: #bbdefb;
}
main {
  max-width: 1600px;
  margin: 0 auto;
  padding: 32px 24px 80px 24px;
}
nav#toc {
  position: sticky;
  top: 0;
  background: white;
  border-bottom: 1px solid #d0d7de;
  padding: 12px 24px;
  z-index: 10;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  font-size: 13px;
}
nav#toc strong { margin-right: 12px; color: #57606a; }
nav#toc a {
  display: inline-block;
  margin: 0 6px 4px 0;
  padding: 2px 8px;
  background: #ddf4ff;
  color: #0969da;
  text-decoration: none;
  border-radius: 12px;
  border: 1px solid #b6e3ff;
}
nav#toc a:hover { background: #b6e3ff; }
h2 {
  margin-top: 32px;
  padding-bottom: 6px;
  border-bottom: 1px solid #d0d7de;
  font-size: 22px;
}
section.legend {
  background: white;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 16px 24px;
  margin-bottom: 24px;
}
section.legend p { margin: 4px 0; font-size: 14px; }
section.legend code {
  background: #eaeef2;
  padding: 0 4px;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px;
}
article.section {
  background: white;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  padding: 16px 24px 24px 24px;
  margin-bottom: 28px;
}
article.section h3 {
  margin: 0 0 4px 0;
  font-size: 19px;
}
article.section .blurb {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #57606a;
}
article.section .blurb code {
  background: #eaeef2;
  padding: 0 4px;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 13px;
}
.pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.pair > div {
  border: 1px solid #d0d7de;
  border-radius: 6px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.pair h4 {
  margin: 0;
  padding: 6px 12px;
  background: #f6f8fa;
  border-bottom: 1px solid #d0d7de;
  font-size: 13px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.pair h4 .lang-tag {
  font-weight: 700;
  letter-spacing: 0.3px;
}
.pair h4 .source-path {
  font-weight: 400;
  color: #57606a;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px;
}
.lang-matlab h4 { background: #fff3e0; border-bottom-color: #ffcc80; }
.lang-python h4 { background: #e8f5e9; border-bottom-color: #a5d6a7; }
pre {
  margin: 0;
  padding: 12px 16px;
  background: #ffffff;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12.5px;
  line-height: 1.45;
  overflow: auto;
  white-space: pre;
  flex: 1;
}
pre a {
  color: #0969da;
  text-decoration: none;
  border-bottom: 1px dotted #0969da;
}
pre a:hover {
  background: #ddf4ff;
  border-bottom-color: transparent;
}
table.glossary {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
table.glossary th, table.glossary td {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid #d0d7de;
  vertical-align: top;
}
table.glossary th { background: #f6f8fa; }
table.glossary code {
  background: #eaeef2;
  padding: 0 4px;
  border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px;
}
footer {
  text-align: center;
  font-size: 12px;
  color: #57606a;
  padding: 16px;
}
@media (max-width: 1100px) {
  .pair { grid-template-columns: 1fr; }
}
"""


def render_glossary() -> str:
    """Render the MATLAB ↔ Python equivalence table."""
    rows: list[tuple[str, str, str, str, str]] = [
        # (description, MATLAB token, MATLAB url, Python token, Python url)
        ("Standard normal draws", "randn",
         MATLAB_DOCS["randn"], "rng.standard_normal", PYTHON_DOCS["rng.standard_normal"]),
        ("Uniform draws", "rand",
         MATLAB_DOCS["rand"], "rng.random", PYTHON_DOCS["rng.random"]),
        ("Identity matrix", "eye",
         MATLAB_DOCS["eye"], "np.eye", PYTHON_DOCS["np.eye"]),
        ("Zero matrix", "zeros",
         MATLAB_DOCS["zeros"], "np.zeros", PYTHON_DOCS["np.zeros"]),
        ("Reshape (column-major)", "reshape",
         MATLAB_DOCS["reshape"], "ndarray.reshape(..., order='F')",
         "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"),
        ("Flatten as column-vector", "vec / M(:)",
         MATLAB_DOCS["vec"], "ndarray.reshape(-1, order='F')",
         "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html"),
        ("Norm", "norm",
         MATLAB_DOCS["norm"], "np.linalg.norm", PYTHON_DOCS["np.linalg.norm"]),
        ("QR decomposition", "qr",
         MATLAB_DOCS["qr"], "np.linalg.qr", PYTHON_DOCS["np.linalg.qr"]),
        ("Full SVD", "svd",
         MATLAB_DOCS["svd"], "np.linalg.svd / scipy.linalg.svd",
         PYTHON_DOCS["scipy.linalg.svd"]),
        ("Truncated SVD (top-k)", "svds",
         MATLAB_DOCS["svds"], "np.linalg.svd(...)[:, :k]",
         PYTHON_DOCS["np.linalg.svd"]),
        ("Symmetric eigendecomp", "eig",
         MATLAB_DOCS["eig"], "np.linalg.eigh", PYTHON_DOCS["np.linalg.eigh"]),
        ("Right division A/B", "A/B",
         "https://www.mathworks.com/help/matlab/ref/mrdivide.html",
         "np.linalg.solve(B.T, A.T).T", PYTHON_DOCS["np.linalg.solve"]),
        ("Cumulative sum", "cumsum",
         MATLAB_DOCS["cumsum"], "np.cumsum", PYTHON_DOCS["np.cumsum"]),
        ("Outer product", "u*u'",
         MATLAB_DOCS["ctranspose"], "np.outer", PYTHON_DOCS["np.outer"]),
        ("Sample covariance", "cov",
         MATLAB_DOCS["cov"], "np.cov", PYTHON_DOCS["np.cov"]),
        ("Hungarian bipartite matching", "HungarianBipartiteMatching (bundled .m file)",
         "https://en.wikipedia.org/wiki/Hungarian_algorithm",
         "scipy.optimize.linear_sum_assignment",
         PYTHON_DOCS["scipy.optimize.linear_sum_assignment"]),
        ("Hierarchical clustering (single linkage)", "clusterdata",
         MATLAB_DOCS["clusterdata"],
         "scipy.cluster.hierarchy.linkage + fcluster",
         PYTHON_DOCS["scipy.cluster.hierarchy.linkage"]),
        ("K-means++", "kmeans",
         MATLAB_DOCS["kmeans"], "sklearn.cluster.KMeans", PYTHON_DOCS["KMeans"]),
        ("Convex-optimisation modelling layer", "CVX",
         "http://cvxr.com/cvx/doc/", "CVXPY",
         "https://www.cvxpy.org/api_reference/"),
        ("Wall-clock timer", "tic / toc",
         MATLAB_DOCS["tic"], "time.perf_counter", PYTHON_DOCS["time.perf_counter"]),
        ("RGB → grayscale", "rgb2gray",
         MATLAB_DOCS["rgb2gray"], "np.dot([0.2989, 0.5870, 0.1140])",
         "https://numpy.org/doc/stable/reference/generated/numpy.dot.html"),
        ("Sliding-window patches", "nlfilter / im2col",
         MATLAB_DOCS["nlfilter"], "np.lib.stride_tricks.sliding_window_view",
         PYTHON_DOCS["np.lib.stride_tricks.sliding_window_view"]),
    ]
    lines = [
        "<table class='glossary'>",
        "<thead><tr><th>Concept</th><th>MATLAB</th><th>Python equivalent</th></tr></thead>",
        "<tbody>",
    ]
    for desc, mtok, murl, ptok, purl in rows:
        lines.append(
            "<tr>"
            f"<td>{html.escape(desc)}</td>"
            f"<td><a href='{murl}' target='_blank' rel='noopener'><code>{html.escape(mtok)}</code></a></td>"
            f"<td><a href='{purl}' target='_blank' rel='noopener'><code>{html.escape(ptok)}</code></a></td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def render_section(sec: dict) -> str:
    # ---- gather MATLAB sources ----
    matlab_path = sec["matlab_path"]
    matlab_paths = [matlab_path]
    if "matlab_extra_path" in sec:
        matlab_paths.append(sec["matlab_extra_path"])
    matlab_paths += sec.get("matlab_extra_paths", [])
    matlab_chunks = []
    for p in matlab_paths:
        src = read(REPO_ROOT / p)
        if len(matlab_paths) > 1:
            matlab_chunks.append(f"% ===== {p} =====\n{src}")
        else:
            matlab_chunks.append(src)
    matlab_source = "\n\n".join(matlab_chunks).rstrip() + "\n"
    matlab_html = linkify(matlab_source, MATLAB_DOCS, comment_char="%")

    # ---- gather Python sources ----
    python_src = read(REPO_ROOT / sec["python_path"])
    python_src = strip_matlab_blocks(python_src)
    python_chunk = extract_python_chunk(python_src, sec["python_funcs"])
    python_html = linkify(python_chunk, PYTHON_DOCS, comment_char="#")

    matlab_label = (
        ", ".join(matlab_paths) if len(matlab_paths) > 1 else matlab_path
    )

    return textwrap.dedent(f"""
    <article class="section" id="{sec['id']}">
      <h3>{html.escape(sec['title'])}</h3>
      <p class="blurb">{sec['blurb']}</p>
      <div class="pair">
        <div class="lang-matlab">
          <h4>
            <span class="lang-tag">MATLAB</span>
            <span class="source-path">{html.escape(matlab_label)}</span>
          </h4>
          <pre>{matlab_html}</pre>
        </div>
        <div class="lang-python">
          <h4>
            <span class="lang-tag">Python</span>
            <span class="source-path">{html.escape(sec['python_path'])}</span>
          </h4>
          <pre>{python_html}</pre>
        </div>
      </div>
    </article>
    """).strip()


def render_toc() -> str:
    items = [
        "<a href='#glossary'>Glossary</a>",
    ]
    for sec in SECTIONS:
        items.append(f"<a href='#{sec['id']}'>{html.escape(sec['title'].split('—')[0].strip())}</a>")
    return "<nav id='toc'><strong>Jump to</strong>" + " ".join(items) + "</nav>"


def render_html() -> str:
    body_sections = "\n\n".join(render_section(s) for s in SECTIONS)
    today = date.today().isoformat()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OverICA: MATLAB ↔ Python Port</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <h1>OverICA: MATLAB ↔ Python Side-by-side Comparison</h1>
    <p>
      Companion document for the Python port of
      <a href="https://github.com/anastasia-podosinnikova/oica" target="_blank" rel="noopener">Anastasia Podosinnikova's OverICA</a>
      that lives under <code>py/</code> in this repository.
    </p>
    <p>
      Every section pairs the original MATLAB source with its Python
      port. Identifiers in the code are <strong>clickable links</strong> to
      the upstream documentation (MathWorks / NumPy / SciPy / scikit-learn
      / CVXPY / Matplotlib), so the reader can verify the equivalence of
      each construct.
    </p>
    <p style="font-size: 12px;">Generated {today}.</p>
  </header>

  {render_toc()}

  <main>
    <h2 id="glossary">Reference glossary</h2>
    <section class="legend">
      <p>
        The table below collects the most common MATLAB constructs used in
        OverICA and their direct Python counterparts. Both columns link to
        the official documentation.
      </p>
      {render_glossary()}
      <p style="margin-top: 16px;">
        A few translation conventions used throughout:
      </p>
      <ul>
        <li>MATLAB's column-major <code>M(:)</code> and <code>reshape(v, p, p)</code>
            map to NumPy with <code>order='F'</code> to preserve bit-exact index layout.</li>
        <li><code>FOOBI</code> (proprietary algorithm from L. De Lathauwer) is
            not redistributed by this repo and is silently skipped by the
            Python comparators.</li>
        <li>Functions that draw random numbers take an explicit
            <code>rng=</code> argument in Python so experiments are
            reproducible — MATLAB's reproducibility hinges on the global
            <code>randn('state', 0)</code> / <code>rand('state', 0)</code> seeds.</li>
      </ul>
    </section>

    <h2>Function-by-function comparison</h2>
    {body_sections}
  </main>

  <footer>
    <p>
      This document is generated by
      <code>py/docs/build_comparison.py</code>. Edit the
      <code>SECTIONS</code> list there to add or reorder sections, and the
      <code>MATLAB_DOCS</code> / <code>PYTHON_DOCS</code> maps to add new
      doc links.
    </p>
  </footer>
</body>
</html>
"""


def main() -> None:
    OUT_PATH.write_text(render_html(), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
