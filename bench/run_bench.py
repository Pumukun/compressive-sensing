#!/usr/bin/env python3
'''
Reproducible quality and speed benchmark for the Compressive Sensing algorithms.

Acts as a gate for optimizations: any change must preserve quality (PSNR no more
than PSNR_TOL below the baseline, SSIM no more than SSIM_TOL below it) and report
how the timing moved.

Usage:
    python bench/run_bench.py --save-baseline   # write the baseline to bench/baseline.json
    python bench/run_bench.py                   # compare the current code against it
    python bench/run_bench.py --quick           # reduced grid
    python bench/run_bench.py --algorithms omp sp

Exit code 1 if any run fails the quality gate.
'''
import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framework import dct, omp, sp, cosamp, brgp  # noqa: E402

SEED = 42
BASELINE_PATH = Path(__file__).resolve().parent / 'baseline.json'

PSNR_TOL = 0.1    # dB, allowed drop against the baseline
SSIM_TOL = 0.002  # allowed SSIM drop

IMAGES_FULL = ['lena.png', '4.1.05.png', 'house.png', 'boat.512.png']
IMAGES_QUICK = ['lena.png', 'house.png']

K_VALUES_FULL = [10, 20]
K_VALUES_QUICK = [20]

# CoSaMP orders its arguments differently: (s, M), s playing the role of K
ALGORITHMS = {
    'omp': lambda p, psi, m, k: omp(p, psi, m, k, seed=SEED),
    'sp': lambda p, psi, m, k: sp(p, psi, m, k, seed=SEED),
    'cosamp': lambda p, psi, m, k: cosamp(p, psi, k, m, seed=SEED),
    'brgp': lambda p, psi, m, k: brgp(p, psi, m, k, seed=SEED),
}


def image_size(path: Path) -> int:
    '''Basis size N, taken from the image.'''
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'could not read {path}')
    return image.shape[0]


def run(algorithms, images, k_values):
    '''Run the grid. Returns {key: {psnr, ssim, cr, seconds}}.'''
    results = {}

    for image_name in images:
        path = ROOT / 'misc' / image_name
        n = image_size(path)
        psi = dct(n)
        m = n // 2

        for alg_name in algorithms:
            for k in k_values:
                key = f'{alg_name}|{image_name}|M{m}|K{k}'

                start = time.perf_counter()
                result = ALGORITHMS[alg_name](str(path), psi, m, k)
                seconds = time.perf_counter() - start

                results[key] = {
                    'psnr': round(float(result.get_PSNR()), 4),
                    'ssim': round(float(result.get_SSIM()), 6),
                    'cr': round(float(result.get_CR()), 4),
                    'seconds': round(seconds, 3),
                }
                print(f'  {key:38s} PSNR={result.get_PSNR():6.2f}  '
                      f'SSIM={result.get_SSIM():.4f}  {seconds:7.2f} s')

    return results


def compare(current, baseline):
    '''Check against the baseline. Returns the list of quality-gate violations.'''
    failures = []
    total_now = total_was = 0.0

    print(f'\n{"run":38s} {"dPSNR":>8s} {"dSSIM":>9s} {"was":>8s} {"now":>8s} {"speedup":>8s}')
    for key, now in sorted(current.items()):
        was = baseline.get(key)
        if was is None:
            print(f'  {key:36s} {"not in the baseline":>36s}')
            continue

        d_psnr = now['psnr'] - was['psnr']
        d_ssim = now['ssim'] - was['ssim']
        speedup = was['seconds'] / now['seconds'] if now['seconds'] > 0 else float('inf')
        total_now += now['seconds']
        total_was += was['seconds']

        bad = d_psnr < -PSNR_TOL or d_ssim < -SSIM_TOL
        if bad:
            failures.append((key, d_psnr, d_ssim))

        print(f'  {key:36s} {d_psnr:+8.3f} {d_ssim:+9.5f} '
              f'{was["seconds"]:8.2f} {now["seconds"]:8.2f} {speedup:7.2f}x'
              f'{"  <-- GATE" if bad else ""}')

    missing = sorted(set(baseline) - set(current))
    for key in missing:
        print(f'  {key:36s} {"missing from this run":>36s}')

    if total_now > 0:
        print(f'\n  total: was {total_was:.2f} s, now {total_now:.2f} s, '
              f'speedup {total_was / total_now:.2f}x')

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--save-baseline', action='store_true',
                        help='store the current results as the baseline')
    parser.add_argument('--quick', action='store_true', help='reduced grid')
    parser.add_argument('--algorithms', nargs='+', choices=sorted(ALGORITHMS),
                        default=sorted(ALGORITHMS),
                        help='algorithms to benchmark (default: all)')
    args = parser.parse_args()

    images = IMAGES_QUICK if args.quick else IMAGES_FULL
    k_values = K_VALUES_QUICK if args.quick else K_VALUES_FULL

    print(f'seed={SEED}, images={len(images)}, K={k_values}, '
          f'algorithms={", ".join(args.algorithms)}\n')
    current = run(args.algorithms, images, k_values)

    if args.save_baseline:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + '\n')
        print(f'\nbaseline written: {BASELINE_PATH.relative_to(ROOT)} ({len(current)} runs)')
        return 0

    if not BASELINE_PATH.exists():
        print(f'\nno baseline at {BASELINE_PATH.relative_to(ROOT)}, '
              f'run with --save-baseline first')
        return 1

    baseline = json.loads(BASELINE_PATH.read_text())
    failures = compare(current, baseline)

    if failures:
        print(f'\nQUALITY GATE FAILED ({len(failures)}): '
              f'tolerance PSNR -{PSNR_TOL} dB, SSIM -{SSIM_TOL}')
        return 1

    print('\nquality gate passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
