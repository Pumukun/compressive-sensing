import numpy as np
import math

from functools import lru_cache


@lru_cache(maxsize=None)
def dct(N: int) -> np.ndarray:
    '''
    Build a square DCT basis matrix of size N.

    Cached, since the matrix depends only on N. The result is shared by every
    caller and therefore marked read-only.
    '''
    n = np.arange(N)

    # mat[i, k] = cos(i * k * pi / N)
    mat_dct_1d: np.ndarray = np.cos(np.outer(n, n) * (math.pi / N))

    mat_dct_1d[:, 1:] -= mat_dct_1d[:, 1:].mean(axis=0, keepdims=True)
    mat_dct_1d /= np.linalg.norm(mat_dct_1d, axis=0, keepdims=True)

    mat_dct_1d.flags.writeable = False
    return mat_dct_1d
