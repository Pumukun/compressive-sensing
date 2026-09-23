import numpy as np
import framework.metrics as metrics

from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, gram_chunk_size, lstsq,
                             read_image, solve_stack, to_column_block, to_uint8)
from typing import Optional, Tuple


def omp(image_path: str, matrix: np.ndarray, M: int, K: int, seed: Optional[int] = None,
        color: bool = False, measurement='gaussian') -> ImageCS:
    '''
    OMP over a 2D image.
        image_path - path to the image (grayscale or colour).
        matrix - NxN basis matrix.
        M - number of measurements.
        K - number of iterations.
        seed - RNG seed for the measurement matrix (None means random).
        color - process the image as colour.
        measurement - 'gaussian', 'bernoulli', 'hadamard', or a ready-made MxN matrix.
    '''
    image = read_image(image_path, color)

    im, channels = to_column_block(image)
    N = im.shape[0]

    Phi = resolve_measurement(measurement, M, N, seed)
    img_cs_1d = np.dot(Phi, im)

    Theta_1d = np.dot(Phi, matrix)

    sparse_rec_1d = cs_omp_columns(img_cs_1d, Theta_1d, K)

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res


def cs_omp_columns(Y: np.ndarray, Phi: np.ndarray, K: int,
                   return_support: bool = False):
    '''
    OMP over every column of the image at once.
        Y - MxW measurement matrix, one column per image column.
        Phi - MxN product of the basis and the measurement matrix.
        K - number of iterations.
        return_support - also return the (W, K) support matrix.

    Works through the Gram matrix G = Phi^T Phi, so correlations for all columns
    come from one matrix product and the least-squares systems are solved as a
    batch. Every column runs exactly K iterations, so they stay in lockstep.
    '''
    M, N = Phi.shape
    W = Y.shape[1]

    # More than M atoms cannot be independent: G[S, S] would be singular
    K = max(0, min(K, M, N))

    X = np.zeros((N, W))
    support_all = np.zeros((W, K), dtype=np.intp)
    if K == 0 or W == 0:
        return (X, support_all) if return_support else X

    G = np.dot(Phi.T, Phi)
    A0 = np.dot(Phi.T, Y)

    chunk = gram_chunk_size(W, K)

    for start in range(0, W, chunk):
        stop = min(start + chunk, W)
        _omp_chunk(G, A0[:, start:stop], X[:, start:stop], support_all[start:stop], K)

    return (X, support_all) if return_support else X


def _omp_chunk(G: np.ndarray, A0: np.ndarray, X: np.ndarray,
               support: np.ndarray, K: int) -> None:
    '''One chunk of columns; X and support are filled in place.'''
    N, W = A0.shape
    cols = np.arange(W)

    for k in range(K):
        # Phi^T (Y - Phi X) = A0 - G X
        correlation = A0 if k == 0 else A0 - np.dot(G, X)
        correlation = np.abs(correlation)

        if k:
            correlation[support[:, :k].T, cols] = -1.0

        support[:, k] = np.argmax(correlation, axis=0)

        index = support[:, :k + 1]
        gram = G[index[:, :, None], index[:, None, :]]
        rhs = A0[index, cols[:, None]]

        X[:] = 0.0
        X[index, cols[:, None]] = solve_stack(gram, rhs)


def cs_omp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Per-column OMP.
        y - the vector being compressed.
        Phi - MxN product of the basis and the measurement matrix.
        K - number of iterations.
    '''
    residual: np.ndarray = y
    M, N = Phi.shape
    index: np.ndarray = np.full(N, -1, dtype=int)

    result: np.ndarray = np.zeros((N, 1))
    a: np.ndarray = np.zeros((0, 1))

    for j in range(K):
        product: np.ndarray = np.fabs(np.dot(Phi.T, residual))
        product[index >= 0] = -1.0
        pos: int = int(np.argmax(product))
        index[pos] = 1
        support: np.ndarray = Phi[:, index >= 0]
        a = lstsq(support, y)
        residual = y - np.dot(support, a)

    result[index >= 0] = a
    Candidate: np.ndarray = np.flatnonzero(index >= 0)

    return result, Candidate
