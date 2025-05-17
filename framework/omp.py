import numpy as np
import cv2
import math
import framework.metrics as metrics

from framework.utils import ImageCS
from typing import Tuple


def omp(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    '''
    OMP 2d функция. 
        image_path - путь к сжимаемому изображению (изображение квадратное, цветное/ЧБ). 
        matrix - базисная матрица размера NxN. 
        K - количество итераций алгоритма. 
        M - размер вспомогательной матрицы.
    by Vladislav Gerda
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    N = H

    im = np.array(image)

    Phi = np.random.randn(M,N) / np.sqrt(M) # Формирование матрицы измерений
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(W):

        y = np.reshape(img_cs_1d[:, i], (M, 1))
        column_rec, Candidate = cs_omp(y, Theta_1d, K)
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = np.dot(matrix, sparse_rec_1d) # Восстановление изображения

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)

    return img_res


def cs_omp(y: np.ndarray, Phi: np.ndarray, K: int) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    '''
    Вспомогательная функция сжатия векторов. 
        y - сжимаемый вектор. 
        Phi - матрица MxN, являющаяся произведением базисной матрицы и матрицы измерений. 
        K - Количество итераций алгоритма.
    by Vladislav Gerda
    '''  
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

