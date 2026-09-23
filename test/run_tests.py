#!/usr/bin/env python3
'''
Run Compressive Sensing algorithms over a set of images and a parameter grid.

The basis size N is taken from each image. Failed runs are reported, not
swallowed. Results are written to the database in one batched insert.
Parallelism uses processes, since the algorithms are CPU-bound.

Examples:
    python run_tests.py                                   # every algorithm, every image
    python run_tests.py --algorithms omp sp --K 10 20
    python run_tests.py --images lena.png --jobs 4
    python run_tests.py --dry-run                         # print the grid and exit
'''
import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter


def _limit_blas_threads_if_parallel() -> None:
    '''
    Cap BLAS to one thread per process when --jobs > 1.

    OpenBLAS grabs every core by default, so N processes of N threads each
    oversubscribe the machine. Must run before NumPy is imported.
    '''
    jobs = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--jobs' and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
        elif arg.startswith('--jobs='):
            value = arg.split('=', 1)[1]
        else:
            continue
        try:
            jobs = int(value)
        except ValueError:
            pass

    if jobs > 1:
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(var, '1')


_limit_blas_threads_if_parallel()

import cv2  # noqa: E402

TEST_DIR = Path(__file__).resolve().parent
ROOT = TEST_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))

import db  # noqa: E402
from framework import dct, omp, sp, cosamp, brgp, iht, lamp  # noqa: E402

# CoSaMP orders its arguments differently: (path, matrix, s, M), s playing K
ALGORITHMS = {
    'omp': lambda p, psi, m, k, **kw: omp(p, psi, m, k, **kw),
    'sp': lambda p, psi, m, k, **kw: sp(p, psi, m, k, **kw),
    'cosamp': lambda p, psi, m, k, **kw: cosamp(p, psi, k, m, **kw),
    'brgp': lambda p, psi, m, k, **kw: brgp(p, psi, m, k, **kw),
    'iht': lambda p, psi, m, k, **kw: iht(p, psi, m, k, **kw),
    'lamp': lambda p, psi, m, k, **kw: lamp(p, psi, m, k, **kw),
}


def image_shape(path: Path):
    '''Image dimensions, or None if the file cannot be read as an image.'''
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else image.shape


def build_jobs(algorithms, images, m_values, k_values, misc_dir, out_dir, seed,
               color=False, measurement='gaussian'):
    '''Expand the grid into a flat job list, skipping invalid combinations.'''
    jobs, skipped = [], []

    for image_name in images:
        path = misc_dir / image_name
        shape = image_shape(path)
        if shape is None:
            skipped.append((image_name, 'not readable as an image'))
            continue

        height, width = shape
        # N is set by the height: Phi is built over that dimension
        n = height
        ms = [n // 2] if m_values is None else m_values

        for m in ms:
            if m >= n:
                skipped.append((f'{image_name} M={m}', f'M >= N ({n})'))
                continue
            for alg in algorithms:
                for k in k_values:
                    if k > m:
                        skipped.append((f'{image_name} M={m} K={k}', 'K > M'))
                        continue
                    stem = image_name.rsplit('.', 1)[0]
                    out = out_dir / alg / stem / f'M{m}_K{k}.png'
                    jobs.append({
                        'alg': alg, 'image': image_name, 'stem': stem,
                        'path': str(path), 'out': str(out), 'n': n,
                        'm': m, 'k': k, 'height': height, 'width': width, 'seed': seed,
                        'color': color, 'measurement': measurement,
                    })

    return jobs, skipped


def run_job(job: dict) -> dict:
    '''One run. Exceptions are returned in the result rather than swallowed.'''
    started = perf_counter()
    try:
        psi = dct(job['n'])
        rec = ALGORITHMS[job['alg']](job['path'], psi, job['m'], job['k'],
                                     seed=job['seed'], color=job['color'],
                                     measurement=job['measurement'])

        Path(job['out']).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(job['out'], rec.get_Image()):
            raise IOError(f'could not write {job["out"]}')

        return {**job, 'ok': True, 'seconds': perf_counter() - started,
                'psnr': float(rec.get_PSNR()), 'ssim': float(rec.get_SSIM()),
                'cr': float(rec.get_CR())}
    except Exception as exc:
        return {**job, 'ok': False, 'seconds': perf_counter() - started,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--algorithms', nargs='+', choices=sorted(ALGORITHMS),
                        default=sorted(ALGORITHMS),
                        help='algorithms to run (default: all)')
    parser.add_argument('--images', nargs='+', default=None,
                        help='file names inside misc/ (default: all)')
    parser.add_argument('--M', nargs='+', type=int, default=None,
                        help='number of measurements (default: N/2 per image)')
    parser.add_argument('--K', nargs='+', type=int, default=[10, 20],
                        help='number of iterations / sparsity level')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for the measurement matrix; negative means random')
    parser.add_argument('--jobs', type=int, default=1,
                        help='number of processes (the algorithms are CPU-bound)')
    parser.add_argument('--color', action='store_true',
                        help='process images as colour')
    parser.add_argument('--measurement', default='gaussian',
                        choices=['gaussian', 'bernoulli', 'hadamard'],
                        help='measurement matrix type (hadamard needs N to be a power of two)')
    parser.add_argument('--out-dir', default=str(TEST_DIR / 'images'),
                        help='where to write reconstructed images')
    parser.add_argument('--no-db', action='store_true',
                        help='do not write to results.db')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the grid and exit')
    args = parser.parse_args()

    misc_dir = ROOT / 'misc'
    images = args.images or sorted(p.name for p in misc_dir.iterdir() if p.is_file())
    seed = None if args.seed < 0 else args.seed

    jobs, skipped = build_jobs(args.algorithms, images, args.M, args.K,
                               misc_dir, Path(args.out_dir), seed,
                               color=args.color, measurement=args.measurement)

    print(f'jobs: {len(jobs)}, skipped: {len(skipped)}, processes: {args.jobs}, seed: {seed}')
    for name, reason in skipped:
        print(f'  skip  {name}: {reason}')
    if args.dry_run:
        return 0
    if not jobs:
        print('nothing to run')
        return 1

    started = perf_counter()
    results = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                report(results[-1], done, len(jobs))
    else:
        for done, job in enumerate(jobs, 1):
            results.append(run_job(job))
            report(results[-1], done, len(jobs))

    failed = [r for r in results if not r['ok']]
    elapsed = perf_counter() - started

    if not args.no_db:
        db.create_table()
        db.add_results([
            (r['out'], r['stem'], r['alg'], r['psnr'], r['ssim'], r['cr'],
             r['k'], r['m'], r['height'], r['width'])
            for r in results if r['ok']
        ])
        print(f'\nrows written to the database: {len(results) - len(failed)} ({db.DB_PATH})')

    print(f'time: {elapsed:.1f} s, succeeded: {len(results) - len(failed)}, failed: {len(failed)}')
    for r in failed:
        print(f'\nFAILED {r["alg"]} {r["image"]} M={r["m"]} K={r["k"]}: {r["error"]}')
        print(r['traceback'])

    return 1 if failed else 0


def report(r: dict, done: int, total: int) -> None:
    head = f'[{done}/{total}] {r["alg"]:6s} {r["image"]:16s} M={r["m"]:<5d} K={r["k"]:<4d}'
    if r['ok']:
        print(f'{head} PSNR={r["psnr"]:6.2f} SSIM={r["ssim"]:.4f} CR={r["cr"]:7.3f} '
              f'{r["seconds"]:6.2f} s')
    else:
        print(f'{head} FAILED: {r["error"]}')


if __name__ == '__main__':
    sys.exit(main())
