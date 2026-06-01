# AGENTS.md

## Cursor Cloud specific instructions

### Product overview

This repository implements **OverICA** (overcomplete independent component analysis via SDP) from the AISTATS 2019 paper. There are two parallel implementations:

| Track | Location | Runtime |
|-------|----------|-----------|
| **Python port** (recommended for CI/agents) | `py/` | Python 3.12+ with NumPy/SciPy/scikit-learn |
| **Original MATLAB** | repo root (`.m` files, prebuilt `.mexmaci64`) | MATLAB with CVX for some figures; MEX recompile via `install(1)` |

There is no long-running server. “Running the app” means executing Python scripts or MATLAB `reproduce_*` drivers.

### Python development (primary in Cloud)

Standard commands live in `py/README.md`. Non-obvious points:

- Run commands from `py/` (or set `PYTHONPATH` to `py/`). Example scripts add `py/` to `sys.path` when run directly.
- Use a virtualenv at `py/.venv` if you prefer isolation. On Ubuntu, `python3 -m venv` requires the **`python3.12-venv`** apt package; without it, use `python3 -m pip install -r requirements.txt` into the environment Python instead.
- Headless plotting: set `MPLBACKEND=Agg` when running `examples/reproduce_*.py`.
- **CVXPY** is optional and only needed for `examples/reproduce_fig_1_phase_transition.py` (commented in `requirements.txt`).
- **FOOBI** and **CIFAR-10** are optional for full paper reproduction; quick configs and synthetic fallbacks work without them.

### Smoke / hello-world

```bash
cd /workspace/py
python3 -m pip install -r requirements.txt   # or: .venv/bin/pip install -r requirements.txt
python examples/demo.py
```

Expected: OverICA runs on synthetic data and prints angular/Frobenius errors plus timings (~0.2–2 s on a typical VM).

### Lint / tests

There is no project linter config or pytest suite. Practical checks:

```bash
cd /workspace/py
python -m compileall -q overica examples
python examples/demo.py
```

### MATLAB track (optional)

Not installed in the default Cloud VM image. If MATLAB is available: run `install` or `install(1)` in MATLAB from the repo root, then `reproduce_*` scripts. Outputs go to `expres/` (gitignored).
