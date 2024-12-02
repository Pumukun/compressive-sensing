import numpy as np
import cv2
import math
import framework.metrics as metrics

from framework.utils import ImageCS
from typing import Tuple

def cosamp(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    N = H

    im = np.array(image)
    Phi = np.random.randn(M,N) / np.sqrt(M)
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N,N))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(N):
        if i % 50 == 0:
            print('iteration: ', i)

        y = np.reshape(img_cs_1d[:, i],(M, 1))
        column_rec, Candidate = cs_cosamp(y, Theta_1d, K)
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = np.dot(matrix, sparse_rec_1d)

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)

    return img_res

def cs_cosamp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:    
    residual: np.ndarray = y
    (M, N) = Phi.shape
    index: np.ndarray = np.array([])

    result: np.ndarray = np.zeros((N, 1))

    for j in range(K):
        product = np.fabs(np.dot(Phi.T, residual))
        top_k_idx = product.argsort(axis=0)[-2 * K:]

        index = np.union1d(index, top_k_idx).astype(int)

        x: np.ndarray = np.zeros((N,1))
        x_temp: np.ndarray = np.dot(np.linalg.pinv(Phi[:, index]),y)
        x[index] = x_temp

        index = np.fabs(x).argsort(axis=0)[-K:]

        residual = y - np.dot(Phi, x)

    return  x, index
