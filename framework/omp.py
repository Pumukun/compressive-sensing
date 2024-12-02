import numpy as np
import cv2
import math
import utils

from typing import Tuple


def omp(image_path: str, matrix: np.ndarray, M: int, K: int) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print("image size:", image.shape)

    H, W = image.shape
    N = H

    im = np.array(image)
    Phi = np.random.randn(M,N) / np.sqrt(M)
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(W):
        if i % 50 == 0:
            print('iteration: ', i)

        y = np.reshape(img_cs_1d[:, i], (M, 1))
        column_rec, Candidate = cs_omp(y, Theta_1d,K)
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = np.dot(matrix, sparse_rec_1d)

    cr: float = utils.compression_ratio(image, sparse_rec_1d)
    PSNR: float = utils.PSNR(image, img_rec)
    print("cr:  ", cr)
    print("PSNR:", PSNR)

    return img_rec



def dct(N: int) -> np.ndarray:
    mat_dct_1d: np.ndarray = np.zeros((N, N))
    v = range(N)

    for k in range(0, N):  
        dct_1d = np.cos(np.dot(v, k * math.pi / N))

        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)

        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

    return mat_dct_1d



def cs_omp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:    
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
    Candidate: Tuple[np.ndarray, ...] = np.where(index >= 0)

    return result, Candidate

