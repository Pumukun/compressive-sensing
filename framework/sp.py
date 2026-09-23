import numpy as np
import framework.metrics as metrics

from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, gram_chunk_size, lstsq,
                             read_image, solve_stack, to_column_block, to_uint8,
                             top_k_indices, top_k_per_column)
from typing import Optional, Tuple


def sp(image_path: str, matrix: np.ndarray, M: int, K: int, seed: Optional[int] = None,
       color: bool = False, measurement='gaussian') -> ImageCS:
    '''
    Subspace Pursuit over a 2D image.
        image_path - path to the image (grayscale or colour).
        matrix - NxN basis matrix.
        M - number of measurements.
        K - number of iterations and support size.
        seed - RNG seed for the measurement matrix (None means random).
        color - process the image as colour.
        measurement - 'gaussian', 'bernoulli', 'hadamard', or a ready-made MxN matrix.

    K is both the iteration count and the support size. Merging grows the support
    to 2K atoms, so a meaningful result needs 2K <= M.
    '''
    image = read_image(image_path, color)

    im, channels = to_column_block(image)
    N = im.shape[0]

    Phi = resolve_measurement(measurement, M, N, seed)
    img_cs_1d = np.dot(Phi, im)

    Theta_1d = np.dot(Phi, matrix)

    sparse_rec_1d = cs_sp_columns(img_cs_1d, Theta_1d, K)

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res


def cs_sp_columns(Y: np.ndarray, Phi: np.ndarray, K: int,
                  return_support: bool = False):
    '''
    SP over every column of the image at once.
        Y - MxW measurement matrix, one column per image column.
        Phi - MxN product of the basis and the measurement matrix.
        K - number of iterations and support size.
        return_support - also return the (W, K) support matrix.

    As in OMP the work goes through the Gram matrix. The SP support changes size
    between K and 2K, while a batched solve needs a uniform shape, so it is
    padded to 2K with dummy atoms N..N+2K-1 that are mutually orthogonal and unit
    norm: their Gram block is the identity and their right-hand side is zero, so
    they contribute nothing.
    '''
    M, N = Phi.shape
    W = Y.shape[1]

    K = max(0, min(K, M, N))

    X = np.zeros((N, W))
    support_all = np.zeros((W, K), dtype=np.intp)
    if K == 0 or W == 0:
        return (X, support_all) if return_support else X

    P = 2 * K

    G = np.dot(Phi.T, Phi)
    A0 = np.dot(Phi.T, Y)

    extended = N + P
    G_ext = np.zeros((extended, extended))
    G_ext[:N, :N] = G
    G_ext[N:, N:] = np.eye(P)

    A0_ext = np.zeros((extended, W))
    A0_ext[:N] = A0

    chunk = gram_chunk_size(W, P)
    for start in range(0, W, chunk):
        stop = min(start + chunk, W)
        _sp_chunk(G, G_ext, A0[:, start:stop], A0_ext[:, start:stop],
                  X[:, start:stop], support_all[start:stop], K)

    return (X, support_all) if return_support else X


def _sp_chunk(G: np.ndarray, G_ext: np.ndarray, A0: np.ndarray, A0_ext: np.ndarray,
              X_out: np.ndarray, support_out: np.ndarray, K: int) -> None:
    '''One chunk of columns; X_out and support_out are filled in place.'''
    N, W = A0.shape
    P = 2 * K

    pad = N + np.arange(P)
    cols = np.arange(W)

    X_ext = np.zeros((N + P, W))
    X = X_ext[:N]

    index = np.tile(pad[:K], (W, 1))

    for j in range(K):
        correlation = np.abs(A0 if j == 0 else A0 - np.dot(G, X))

        top_k_idx = top_k_per_column(correlation, K)

        union = np.sort(np.concatenate([index, top_k_idx], axis=1), axis=1)

        # Duplicates get a distinct dummy atom each: repeated indices would make
        # the Gram block singular
        duplicate = np.zeros_like(union, dtype=bool)
        duplicate[:, 1:] = union[:, 1:] == union[:, :-1]
        union = np.where(duplicate, pad[None, :], union)

        gram = G_ext[union[:, :, None], union[:, None, :]]
        rhs = A0_ext[union, cols[:, None]]

        X_ext[:] = 0.0
        X_ext[union, cols[:, None]] = solve_stack(gram, rhs)

        index = top_k_per_column(np.abs(X), K)

    X_out[:] = X
    support_out[:] = index


def cs_sp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Per-column SP.
        y - the vector being compressed.
        Phi - MxN product of the basis and the measurement matrix.
        K - number of iterations.
    '''
    residual: np.ndarray = y
    (M, N) = Phi.shape
    index: np.ndarray = np.array([], dtype=int)

    x: np.ndarray = np.zeros((N, 1))

    for j in range(K):
        product = np.fabs(np.dot(Phi.T, residual))
        top_k_idx = top_k_indices(product, K)

        index = np.union1d(index, top_k_idx).astype(int)

        x = np.zeros((N,1))
        x_temp: np.ndarray = lstsq(Phi[:, index], y)
        x[index] = x_temp

        index = top_k_indices(np.fabs(x), K)

        residual = y - np.dot(Phi, x)

    return  x, index
