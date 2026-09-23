import numpy as np
import framework.metrics as metrics
from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, gram_chunk_size, lstsq,
                             read_image, solve_stack, to_column_block, to_uint8,
                             top_k_indices, top_k_per_column)
from typing import Optional, Tuple

def cosamp(image_path: str, matrix: np.ndarray, s: int, M: int, seed: Optional[int] = None,
           color: bool = False, measurement='gaussian') -> ImageCS:
    '''
    CoSaMP 2d function.
        image_path - path to the image (grayscale or colour).
        matrix - NxN basis matrix.
        s - sparsity.
        M - number of measurements.
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

    sparse_rec_1d = cs_cosamp_columns(img_cs_1d, s, Theta_1d)

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res

def cs_cosamp_columns(Y: np.ndarray, s: int, Phi: np.ndarray, epsilon: float = 1e-10,
                      K: int = 1000, patience: int = 3) -> np.ndarray:
    '''
    CoSaMP over every column of the image at once.
        Y - MxW measurement matrix (one column per image column).
        s - sparsity.
        Phi - MxN matrix, the product of the basis and the measurement matrix.
        epsilon, K, patience - see cs_cosamp.

    As in OMP and SP the work goes through the Gram matrix, with the support
    padded to a fixed size of 3s. Unlike them, columns stop at different
    iterations, so a set of still-active columns is maintained and the matrix
    products narrow as columns converge.

    The residual norm uses ||y - Phi x||^2 = ||y||^2 - 2 x'A0 + x'Gx, avoiding a
    separate Phi x product.
    '''
    M, N = Phi.shape
    W = Y.shape[1]

    s = max(0, min(s, M, N))

    X_best = np.zeros((N, W))
    if s == 0 or W == 0:
        return X_best

    P = 3 * s

    G = np.dot(Phi.T, Phi)
    A0 = np.dot(Phi.T, Y)

    extended = N + P
    G_ext = np.zeros((extended, extended))
    G_ext[:N, :N] = G
    G_ext[N:, N:] = np.eye(P)

    A0_ext = np.zeros((extended, W))
    A0_ext[:N] = A0

    y_norm_sq = np.einsum('ij,ij->j', Y, Y)

    chunk = gram_chunk_size(W, P)
    for start in range(0, W, chunk):
        stop = min(start + chunk, W)
        _cosamp_chunk(G, G_ext, A0[:, start:stop], A0_ext[:, start:stop],
                      y_norm_sq[start:stop], X_best[:, start:stop],
                      s, epsilon, K, patience)

    return X_best


def _cosamp_chunk(G: np.ndarray, G_ext: np.ndarray, A0: np.ndarray, A0_ext: np.ndarray,
                  y_norm_sq: np.ndarray, X_best: np.ndarray,
                  s: int, epsilon: float, K: int, patience: int) -> None:
    '''One chunk of columns; X_best is filled in place.'''
    N, W = A0.shape
    P = 3 * s
    pad = N + np.arange(P)

    active = np.arange(W)
    support = np.tile(pad[:s], (W, 1))
    best_norm = np.sqrt(y_norm_sq)
    prev_norm = best_norm.copy()
    stale = np.zeros(W, dtype=int)
    Gx = None

    for _ in range(K):
        if active.size == 0:
            break

        a0 = A0[:, active]
        cols = np.arange(active.size)

        correlation = np.abs(a0 if Gx is None else a0 - Gx)
        top_k_idx = top_k_per_column(correlation, 2 * s)

        union = np.sort(np.concatenate([support[active], top_k_idx], axis=1), axis=1)
        duplicate = np.zeros_like(union, dtype=bool)
        duplicate[:, 1:] = union[:, 1:] == union[:, :-1]
        union = np.where(duplicate, pad[None, :], union)

        gram = G_ext[union[:, :, None], union[:, None, :]]
        rhs = A0_ext[union, active[:, None]]

        X_ext = np.zeros((N + P, active.size))
        X_ext[union, cols[:, None]] = solve_stack(gram, rhs)
        x_full = X_ext[:N]

        # Keep the s largest components by magnitude
        keep = top_k_per_column(np.abs(x_full), s)
        x = np.zeros((N, active.size))
        x[keep.T, cols] = x_full[keep.T, cols]

        Gx = np.dot(G, x)

        # ||y - Phi x||^2 = ||y||^2 - 2 x'A0 + x'Gx
        residual_sq = (y_norm_sq[active]
                       - 2.0 * np.einsum('ij,ij->j', x, a0)
                       + np.einsum('ij,ij->j', x, Gx))
        residual_norm = np.sqrt(np.maximum(residual_sq, 0.0))

        improved = residual_norm < best_norm[active] - 1e-12
        better = active[improved]
        X_best[:, better] = x[:, improved]
        best_norm[better] = residual_norm[improved]
        stale[better] = 0
        stale[active[~improved]] += 1

        halt = ((residual_norm < epsilon)
                | (np.abs(residual_norm - prev_norm[active]) < epsilon)
                | (stale[active] >= patience))
        prev_norm[active] = residual_norm

        support[active] = keep

        alive = ~halt
        active = active[alive]
        Gx = Gx[:, alive]


def cs_cosamp(y: np.ndarray, s: int, Phi: np.ndarray, epsilon: float = 1e-10,
              K: int = 1000, patience: int = 3) -> np.ndarray:
    '''
    Per-column helper.
        y - the vector being compressed.
        s - sparsity.
        Phi - MxN matrix, the product of the basis and the measurement matrix.
        epsilon - error tolerance.
        K - maximum number of algorithm iterations.
        patience - consecutive iterations without a decrease in the residual
                   tolerated before stopping.
    '''
    residual: np.ndarray = y
    (M, N) = Phi.shape

    result: np.ndarray = np.zeros((N, 1))

    # The best iterate, not the last: the residual is non-monotone
    best_result: np.ndarray = result
    best_norm: float = float(np.linalg.norm(residual))
    stale: int = 0

    for j in range(K):

        product = np.fabs(np.dot(Phi.T, residual))
        top_k_idx = top_k_indices(product, 2 * s)
        top_k_idx = np.union1d(top_k_idx, result.nonzero()[0])

        phiT = Phi[:, top_k_idx]
        x: np.ndarray = np.zeros((N,1))

        x[top_k_idx] = lstsq(phiT, y)

        # Keep the s largest components by magnitude
        prune = np.ones(N, dtype=bool)
        prune[top_k_indices(np.abs(x), s)] = False
        x[prune] = 0

        result = x

        residual_old: np.ndarray = residual
        residual = y - np.dot(Phi, result)

        residual_norm: float = float(np.linalg.norm(residual))
        if residual_norm < best_norm - 1e-12:
            best_norm = residual_norm
            best_result = result
            stale = 0
        else:
            stale += 1

        # The third criterion catches the oscillating case, where neither of
        # the first two ever fires
        halt = (residual_norm < epsilon
                or np.linalg.norm(residual - residual_old) < epsilon
                or stale >= patience)

        if halt:
            break

    # The sign of the DCT coefficients must be preserved
    return best_result
