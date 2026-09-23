import numpy as np
import framework.metrics as metrics

from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, read_image,
                             to_column_block, to_uint8, top_k_indices,
                             top_k_per_column)
from typing import Optional


def iht(image_path: str, matrix: np.ndarray, M: int, K: int, seed: Optional[int] = None,
        color: bool = False, measurement='gaussian') -> ImageCS:
    '''
    Iterative Hard Thresholding over a 2D image.
        image_path - path to the image (grayscale or colour).
        matrix - NxN basis matrix.
        M - number of measurements.
        K - sparsity level.
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

    sparse_rec_1d = cs_iht_columns(img_cs_1d, Theta_1d, K)

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res


def _step_size(G: np.ndarray) -> float:
    '''
    Gradient step 1/L, with L the largest eigenvalue of G.

    IHT converges when the step does not exceed 1/||Phi||_2^2, and that norm
    equals the largest eigenvalue of the Gram matrix.
    '''
    if G.shape[0] == 0:
        return 1.0

    largest = float(np.linalg.eigvalsh(G)[-1])
    return 1.0 / largest if largest > 0 else 1.0


def cs_iht_columns(Y: np.ndarray, Phi: np.ndarray, K: int,
                   max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    '''
    IHT over every column of the image at once.
        Y - MxW measurement matrix, one column per image column.
        Phi - MxN product of the basis and the measurement matrix.
        K - sparsity level.
        max_iter - iteration cap.
        tol - relative change below which the iteration stops.

    Unlike the pursuit algorithms, IHT keeps no support and solves no
    least-squares problem: each iteration is a gradient step followed by a
    projection onto the K largest coefficients. The gradient of all columns is
    one matrix product through the Gram matrix, so the whole image advances in
    lockstep with no per-column state.
    '''
    M, N = Phi.shape
    W = Y.shape[1]

    K = max(0, min(K, M, N))

    X = np.zeros((N, W))
    if K == 0 or W == 0:
        return X

    G = np.dot(Phi.T, Phi)
    A0 = np.dot(Phi.T, Y)
    step = _step_size(G)

    cols = np.arange(W)

    for _ in range(max_iter):
        # Phi^T (Y - Phi X) = A0 - G X
        candidate = X + step * (A0 - np.dot(G, X))

        keep = top_k_per_column(np.abs(candidate), K)
        X_new = np.zeros_like(X)
        X_new[keep.T, cols] = candidate[keep.T, cols]

        change = np.linalg.norm(X_new - X)
        X = X_new

        if change <= tol * max(1.0, float(np.linalg.norm(X))):
            break

    return X


def cs_iht(y: np.ndarray, Phi: np.ndarray, K: int,
           max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    '''
    Per-column IHT.
        y - the vector being compressed.
        Phi - MxN product of the basis and the measurement matrix.
        K - sparsity level.
    '''
    M, N = Phi.shape
    x = np.zeros((N, 1))

    step = _step_size(np.dot(Phi.T, Phi))

    for _ in range(max_iter):
        candidate = x + step * np.dot(Phi.T, y - np.dot(Phi, x))

        x_new = np.zeros((N, 1))
        keep = top_k_indices(np.abs(candidate), K)
        x_new[keep] = candidate[keep]

        change = np.linalg.norm(x_new - x)
        x = x_new

        if change <= tol * max(1.0, float(np.linalg.norm(x))):
            break

    return x
