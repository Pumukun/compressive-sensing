import numpy as np

from typing import Optional

'''Measurement matrices Phi of size MxN.'''


def gaussian(M: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    '''Gaussian measurement matrix with columns of unit expected norm.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, N)) / np.sqrt(M)


def bernoulli(M: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    '''Matrix of +-1/sqrt(M), with the same recovery guarantees as the Gaussian one.'''
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(M, N)) / np.sqrt(M)


def hadamard(M: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    '''
    Randomised Hadamard matrix: M random rows of the order-N Hadamard matrix
    with randomly flipped column signs. N must be a power of two.

    The sign randomisation is required, not cosmetic. Plain Hadamard rows are
    highly coherent with the DCT basis (coherence 0.98 against 0.375 for a
    Gaussian matrix) and recovery collapses; flipping the signs brings coherence
    to 0.276.
    '''
    if N & (N - 1) != 0:
        raise ValueError(f'a Hadamard matrix requires N to be a power of two, got N={N}')
    if M > N:
        raise ValueError(f'M={M} cannot exceed N={N}')

    order = int(np.log2(N))
    matrix = np.ones((1, 1))
    for _ in range(order):
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])

    rng = np.random.default_rng(seed)
    rows = rng.choice(N, size=M, replace=False)
    signs = rng.choice([-1.0, 1.0], size=N)

    return (matrix[rows] * signs) / np.sqrt(M)


BUILDERS = {
    'gaussian': gaussian,
    'bernoulli': bernoulli,
    'hadamard': hadamard,
}


def build(kind: str, M: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    '''Measurement matrix by name: gaussian (default), bernoulli, hadamard.'''
    try:
        builder = BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f'unknown measurement matrix type {kind!r}, '
            f'available: {", ".join(sorted(BUILDERS))}'
        ) from None

    return builder(M, N, seed)


def resolve(measurement, M: int, N: int, seed: Optional[int] = None) -> np.ndarray:
    '''
    Turn the algorithms' measurement argument into an MxN matrix.

    Accepts a type name or a ready-made matrix; seed is ignored for the latter.
    '''
    if isinstance(measurement, np.ndarray):
        if measurement.shape != (M, N):
            raise ValueError(
                f'the measurement matrix must be {M}x{N}, got '
                f'{measurement.shape[0]}x{measurement.shape[1]}'
            )
        return measurement

    return build(measurement, M, N, seed)
