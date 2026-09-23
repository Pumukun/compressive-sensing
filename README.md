# Compressive Sensing
Compressive Sensing algorithms for 2D data

# Overview
The algorithm implementations live in `framework/`.
Documentation describing the algorithms is in `doc/`.

Testing lives in `test/`. Every algorithm is driven by a single parameterised
runner, `test/run_tests.py`; the scripts `omp_test.py`, `sp_test.py`,
`cosamp_test.py`, `brgp_test.py` and `all_algs_test.py` are thin wrappers around
it. A set of source images is in `misc/`, but nothing stops you from using your
own data. :)

Quality and speed benchmarking is in `bench/run_bench.py`.

# Build

### Linux

```
$ ./setup.sh
```

### Windows

```
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
```

Speed is almost entirely determined by the BLAS backend behind NumPy. The wheel
from PyPI ships OpenBLAS (~300 GFLOPS on 12 cores); a distribution-packaged
NumPy may be built against reference netlib (~8 GFLOPS). `setup.sh` prints a
measurement and warns when the backend is slow.

# Running the tests

```
$ ./venv/bin/python test/run_tests.py --images lena.png --algorithms omp sp --K 10 20
$ ./venv/bin/python test/run_tests.py --jobs 8          # every image, 8 processes
$ ./venv/bin/python test/run_tests.py --dry-run         # print the grid and exit
```

Results are written to `test/images/` and to the `test/results.db` database.
PSNR against CR charts from the database contents:

```
$ ./venv/bin/python test/plot_test.py
```

With `--jobs > 1` the runner caps BLAS to one thread per process; otherwise N
processes of 12 threads each on 12 cores slow things down instead of speeding
them up.

# Benchmarking quality and speed

`bench/run_bench.py` records a metric baseline and acts as a gate for
optimizations: a change is acceptable only if PSNR drops by no more than 0.1 dB
and SSIM by no more than 0.002.

```
$ ./venv/bin/python bench/run_bench.py --save-baseline   # write the baseline
$ ./venv/bin/python bench/run_bench.py                   # compare against it
```

# Reproducibility

The measurement matrix is random, so every algorithm accepts a `seed`:

```python
from framework import omp, dct
result = omp("misc/lena.png", dct(256), M=128, K=20, seed=42)
```

`seed=None` (the default) means a random matrix and a non-reproducible result.

# Colour images

```python
result = omp("misc/4.1.05.png", dct(256), M=128, K=20, seed=42, color=True)
```

Channels are appended as extra columns, so they share the measurement matrix and
the Gram matrix and a colour frame is solved in a single pass. The runner exposes
this as `--color`.

# Measurement matrix

```python
result = omp("misc/lena.png", dct(256), M=128, K=20, seed=42, measurement="hadamard")
```

Available: `gaussian` (default), `bernoulli`, `hadamard`; a ready-made `M x N`
matrix may be passed instead of a name. The runner exposes `--measurement`.

On our image set the randomised Hadamard matrix yields **0.3 to 1.2 dB more**
than the Gaussian one at the same `M`. The column sign randomisation is
essential there: without it the Hadamard rows are structurally close to the
columns of the DCT basis, mutual coherence rises to 0.98 (against 0.375 for the
Gaussian matrix) and recovery collapses to about 6 dB. `hadamard` requires `N`
to be a power of two.

# Authors

[Gerda Vladislav](https://github.com/hitfot)
> hitfot@mail.ru
>
> Telegram: @hitfot

[Demchenko Grigorii](https://github.com/Pumukun)

[Kocherygina Anastasia](https://github.com/somniiium)

[Sabitova Alina](https://github.com/AlinaSAB)

[Shibanov Mikhail](https://github.com/Kar1ch)
