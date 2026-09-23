#!/usr/bin/env python3
'''
Run every algorithm. A thin wrapper around run_tests.py, where all the logic lives.

    python all_algs_test.py --jobs 8
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tests import main

if __name__ == '__main__':
    sys.exit(main())
