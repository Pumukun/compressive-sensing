#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -d "venv" ]; then
	echo "creating virtual environment..."
	"$PYTHON" -m venv venv
	echo "done."
fi

if [ ! -f "requirements.txt" ]; then
	echo "requirements.txt not found." >&2
	exit 1
fi

# Call the venv interpreter directly: activate is unnecessary and does not
# survive the end of the script, and the previous version only activated the
# venv when $VIRTUAL_ENV was empty, so it could install packages outside it.
echo "upgrading pip..."
./venv/bin/python -m pip install --quiet --upgrade pip

echo "installing dependencies from requirements.txt..."
./venv/bin/python -m pip install --quiet -r requirements.txt
echo "done."

echo
echo "BLAS backend:"
./venv/bin/python - <<'PY'
# Algorithm speed is almost entirely determined by BLAS: the NumPy wheel from
# PyPI ships OpenBLAS (~300 GFLOPS on 12 cores), while a distro-packaged NumPy
# may be built against reference netlib (~8 GFLOPS) - nearly a 40x difference.
import numpy as np, time
n = 1000
a, b = np.random.rand(n, n), np.random.rand(n, n)
a @ b
t = time.perf_counter(); a @ b; dt = time.perf_counter() - t
print(f"  matmul {n}^3: {2 * n**3 / dt / 1e9:.0f} GFLOPS")
if 2 * n**3 / dt / 1e9 < 30:
    print("  WARNING: this looks like single-threaded netlib BLAS.")
    print("  Check that numpy came from PyPI rather than from distribution packages.")
PY

echo
echo "run tests:  ./venv/bin/python test/run_tests.py --help"
echo "run bench:  ./venv/bin/python bench/run_bench.py --help"
