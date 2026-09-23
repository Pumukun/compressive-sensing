#!/usr/bin/env python3
'''
Run the LAMP algorithm. A thin wrapper around run_tests.py, where all the logic lives.

    python lamp_test.py --K 10 20 --jobs 4
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tests import main

if __name__ == '__main__':
    # Inserted first so an explicit --algorithms still wins
    sys.argv[1:1] = ['--algorithms', 'lamp']
    sys.exit(main())
