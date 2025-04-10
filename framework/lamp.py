import numpy as np
import cv2
import math
import framework.metrics as metrics

from framework.utils import ImageCS
from typing import Tuple

def lamp(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    N = H

    im = np.array(image)

    Phi = np.random.randn(M, N) / np.sqrt(M)
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(W):
        y = np.reshape(img_cs_1d[:, i], (M, 1))
        column_rec = cs_lamp(y, Theta_1d, K)
        sparse_rec_1d[:, i] = column_rec
    
    image_rec = np.dot(matrix, sparse_rec_1d)

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, image_rec)

    img_res = ImageCS(image_rec, cr=CR, psnr=PSNR)
    return img_res
    

    


def cs_lamp(y: np.ndarray, Phi: np.ndarray, K: int) -> np.ndarray:
    M, N = Phi.shape
    
    threshold = 1e-6 # the threshold should be chosen at each iteration adaptively...

    x_current = np.zeros(N, dtype=int)
    s_current = -1 * np.ones(N, dtype=int)

    for j in range(K):
        r_k = y - Phi @ x_current
        x_temp = Phi.T @ r_k + x_current
        #s_current = Determine MAP estimate of the support using graph cuts
        #x_current = Estimate target signal

        if np.linalg.norm(r_k) < threshold:
            break
    
    return x_current



