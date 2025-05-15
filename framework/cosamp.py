import numpy as np
import cv2
import math
import framework.metrics as metrics
from framework.utils import ImageCS
from typing import Tuple

def cosamp(image_path: str, matrix: np.ndarray, s: int, M: int) -> ImageCS:
    '''
    CoSaMP 2d function. image_path - путь к сжимаемому изображению (изображение квадратное, цветное/ЧБ). matrix - базисная матрица размера NxN. s - sparsity. M - размер вспомогательной матрицы
    by Vladislav Gerda
    '''
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    N = H

    im = np.array(image)
    Phi = np.random.randn(M,N) / np.sqrt(M) # Формирование матрицы измерений 
    img_cs_1d = np.dot(Phi, im) 

    sparse_rec_1d = np.zeros((N,N))
    Theta_1d = np.dot(Phi, matrix) 

    for i in range(N): 
        y = np.reshape(img_cs_1d[:, i],(M, 1))
        column_rec = cs_cosamp(y, s, Theta_1d)
        sparse_rec_1d[:, i] = np.reshape(column_rec, (N))

    img_rec = np.round(np.fabs(np.dot(matrix, sparse_rec_1d))).astype(np.uint8) # Восстановление изображения

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    SSIM: float = metrics.SSIM(image, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR, ssim=SSIM)

    return img_res

def cs_cosamp(y: np.ndarray, s: int, Phi: np.ndarray, epsilon: float = 1e-10, K: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Вспомогательная функция сжатия векторов. y - сжимаемый вектор. s - sparsity. Phi - матрица MxN, являющаяся произведением базисной матрицы и матрицы измерений. epsilon - допустимая погрешность. K - максимальное количество итераций алгоритма.
    by Vladislav Gerda
    ''' 
    residual: np.ndarray = y 
    (M, N) = Phi.shape
    index: np.ndarray = np.array([])

    result: np.ndarray = np.zeros((N, 1)) 

    for j in range(K):

        product = np.fabs(np.dot(Phi.T, residual)) 
        top_k_idx = product.argsort(axis=0)[-2 * s:] # Берем 2s наибольших элементов вектора
        top_k_idx = np.union1d(top_k_idx, result.nonzero()[0])

        phiT = Phi[:, top_k_idx]
        x: np.ndarray = np.zeros((N,1))

        x[top_k_idx], _, _, _ = np.linalg.lstsq(phiT, y) # Решение задачи наименьших квадратовы
        x[np.argsort(x)[:-s]] = 0
        result = x

        residual_old: np.ndarray = residual
        residual = y - np.dot(Phi, result)

        halt = (np.linalg.norm(residual - residual_old) < epsilon) or \
            np.linalg.norm(residual) < epsilon # Проверка критерев остановки
        
        if halt:
            break

    return  np.fabs(result)
