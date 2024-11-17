import numpy as np
import math

from typing import Tuple

def dct(N: int):
    mat_dct_1d: np.ndarray = np.zeros((N, N))
    v = range(N)
    for k in range(0, N):  
        dct_1d = np.cos(np.dot(v, k * math.pi / N))
        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)
        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)
    return mat_dct_1d

def omp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:    
    """
    Implementation of the Orthogonal Matching Pursuit (OMP) algorithm 
    for solving the compressed sensing problem.

    Parameters:
        y (np.ndarray): Measurement vector (shape: Mx1).
        Phi (np.ndarray): Sensing matrix (shape: MxN).
        K (int): The number of non-zero elements in the recovered signal.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - result (np.ndarray): Recovered signal vector (shape: Nx1), 
            where non-zero elements correspond to the estimated signal.
            - Candidate (np.ndarray): Indices of the selected columns from the sensing matrix Phi.

    Example:
        >>> import numpy as np
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> Phi = np.random.rand(3, 5)
        >>> K = 2
        >>> result, Candidate = cs_omp(y, Phi, K)
        >>> print("Recovered signal:", result.flatten())
        >>> print("Selected indices:", Candidate[0])
    """
    residual: np.ndarray = y
    M, N = Phi.shape
    index: np.ndarray = np.zeros(N, dtype=int)

    for i in range(N):
        index[i] = -1

    result: np.ndarray = np.zeros((N, 1))

    for j in range(K):
        product: np.ndarray = np.fabs(np.dot(Phi.T, residual))
        pos: int = int(np.argmax(product))
        index[pos] = 1
        my: np.ndarray = np.linalg.pinv(Phi[:, index >= 0])
        a: np.ndarray = np.dot(my, y)
        residual = y - np.dot(Phi[:, index >= 0], a)

    result[index >= 0] = a
    Candidate: np.ndarray = np.where(index >= 0)

    return result, Candidate


