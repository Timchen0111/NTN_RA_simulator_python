# Remote Python Environment Setup

This project is a Python simulation repo. The main simulation function is in `main.py`, and `lab.py` is the current single-run experiment script.

## 1. Clone the repo

```bash
git clone <your-repo-url>
cd Python_simulator
```

If you are copying the folder manually instead of cloning, make sure these data files are present:

- `fixed_satellite_pool.json`
- `group_ps_table.npz`
- `starlink_tle.txt`

The simulation validates that `group_ps_table.npz`, `fixed_satellite_pool.json`, and the TLE file match each other. If any one of them is missing or stale, regenerate the precomputed files locally before running the remote job.

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CPU-only remote machine, the default `torch` package is usually enough. If the remote machine has CUDA and you want GPU-enabled PyTorch, install the CUDA build from the official PyTorch selector before running `pip install -r requirements.txt`.

## 3. Run a quick smoke test

Use this first because it does not open plotting windows:

```bash
python -c "import main; print(main.main(0.5, 1, 100, [1, 1], 42, 0.01)[0:2])"
```

Expected behavior: the script prints simulation progress and finishes with throughput / packet-loss-rate values.

## 4. Run the current experiment

```bash
python lab.py
```

On a headless remote server, plotting windows from `matplotlib.pyplot.show()` may not display. Use one of these options:

```bash
MPLBACKEND=Agg python lab.py
```

or edit the experiment script to save figures with `plt.savefig(...)` instead of `plt.show()`.

## 5. Common remote issues

- Missing `fixed_satellite_pool.json` or `group_ps_table.npz`: these were previously ignored by git, so confirm they are committed and pushed.
- Missing or different `starlink_tle.txt`: `scenario_time.py` can download TLE data if the remote has internet, but the precomputed tables expect the same TLE hash. For reproducible runs, commit the matching `starlink_tle.txt`.
- `cvxpy` solver errors: install the full requirements first; the code falls back in some cases, but proposed selection mode depends on `cvxpy`.
- Tests named `backoff_control_test.py` and `load_estimator_test.py` import `old`, which is not currently tracked in this repo. Use the smoke test above for remote validation unless those tests are updated.
