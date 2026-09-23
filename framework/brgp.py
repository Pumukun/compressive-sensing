import numpy as np
import framework.metrics as metrics
from framework.measurement import resolve as resolve_measurement
from framework.utils import (ImageCS, from_column_block, lstsq, read_image,
                             to_column_block, to_uint8, top_k_indices)
from typing import Optional, Tuple

from framework.omp import cs_omp, cs_omp_columns
from framework.sp import cs_sp, cs_sp_columns

def brgp(image_path: str, matrix: np.ndarray, M: int, K: int, seed: Optional[int] = None,
         color: bool = False, measurement='gaussian') -> ImageCS:
    '''
    BRGP over a 2D image.
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
    N, W = im.shape

    Phi = resolve_measurement(measurement, M, N, seed)
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    # Initial SP and OMP supports for the whole image in one batch
    _, support_sp = cs_sp_columns(img_cs_1d, Theta_1d, K, return_support=True)
    _, support_omp = cs_omp_columns(img_cs_1d, Theta_1d, K, return_support=True)

    for i in range(W):
        y = np.reshape(img_cs_1d[:, i], (M, 1))

        Candidate_BRGP = np.intersect1d(support_omp[i], support_sp[i])

        column_rec = cs_brgp(y, Theta_1d, K, Candidate_BRGP)
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = to_uint8(from_column_block(np.dot(matrix, sparse_rec_1d), channels))

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res

def cs_brgp(y: np.ndarray, Phi: np.ndarray, K: int, Candidate: np.ndarray, u: float = 0.8) -> np.ndarray:
    '''
    Per-column helper.
        y - the vector being compressed.
        Phi - MxN product of the basis and the measurement matrix.
        K - number of iterations.
        Candidate - initial candidate support set.
        u - expansion coefficient in (0, 1).
    '''
    (M, N) = Phi.shape

    Candidate = np.asarray(Candidate, dtype=int).ravel()

    def project(support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Least-squares projection of y onto the support; returns (x, residual).'''
        x_new = np.zeros((N, 1))
        if support.size:
            x_new[support] = lstsq(Phi[:, support], y)
        return x_new, y - np.dot(Phi, x_new)

    def expand(support: np.ndarray, r_cur: np.ndarray) -> np.ndarray:
        '''Atoms whose correlation with the residual exceeds a fraction u of the maximum.'''
        temp = np.abs(np.dot(Phi.T, r_cur))
        # flatnonzero, not np.where: on an (N, 1) array np.where returns a
        # (rows, cols) tuple and union1d would fold in a spurious index
        F = np.flatnonzero(temp > temp.max() * u)
        return np.union1d(support, F).astype(int)

    x, r = project(Candidate)
    Candidate_save = Candidate
    r_save = r

    Candidate = expand(Candidate, r)
    x, r = project(Candidate)

    while len(Candidate) < K:
        dis = np.linalg.norm(r - r_save)

        if dis < np.linalg.norm(y):
            Candidate_save = Candidate
            r_save = r

            new_candidate = expand(Candidate, r)
        else:
            #print('back off')
            Candidate = Candidate_save
            r = r_save
            new_candidate = Candidate

        if new_candidate.size == Candidate.size:
            # Expansion added nothing; take the best atom from the complement,
            # otherwise the loop never terminates
            Candidate_dif = np.setdiff1d(np.arange(N), Candidate)
            if Candidate_dif.size == 0:
                break

            temp = lstsq(Phi[:, Candidate_dif], y)
            best = Candidate_dif[int(np.argmax(np.abs(temp)))]
            new_candidate = np.union1d(Candidate, [best]).astype(int)

        Candidate = new_candidate
        x, r = project(Candidate)

    T = K
    while T > 0:
        product = np.fabs(np.dot(Phi.T, r))
        top_t_idx = top_k_indices(product, T)
        Candidate = np.union1d(Candidate, top_t_idx).astype(int)

        x_temp = lstsq(Phi[:, Candidate], y)
        index = top_k_indices(np.fabs(x_temp), K)
        x_temp = x_temp[index]
        Candidate = Candidate[index]

        x = np.zeros((N, 1))
        x[Candidate] = x_temp
        r = y - np.dot(Phi, x)

        T = int(np.floor(T * u))

    return x

