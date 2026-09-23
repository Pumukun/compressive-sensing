#!/usr/bin/env python3
'''
Plot PSNR against CR from the results stored in results.db.

    python plot_test.py                 # every algorithm, saved into test/plots/
    python plot_test.py --algorithms omp --show
'''
import argparse
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import plot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--algorithms', nargs='+',
                        default=['omp', 'sp', 'cosamp', 'brgp'])
    parser.add_argument('--show', action='store_true',
                        help='open windows instead of writing files')
    args = parser.parse_args()

    for alg in args.algorithms:
        plot.psnr_cr_by_alg(alg, show=args.show)
    return 0


if __name__ == '__main__':
    sys.exit(main())
