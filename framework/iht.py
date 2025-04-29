import cv2
import numpy as np
import math
import framework.metrics as metrics
from framework.utils import ImageCS
from typing import Tuple
from framework.utils import ImageCS
from omp import dct

# Оператор жёсткого порога
def thresholdOperator(K:int, x:np.ndarray):
    abs_x = np.abs(x)
    threshold = np.partition(abs_x.ravel(), -K)[-K]
    mask = abs_x >= threshold
    x[~mask] = 0
    return x

#восстонавление по столбцам
def cs_iht(y: np.ndarray, Phi: np.ndarray, K: int, step_size:float, max_iter: int = 60, tol: float = 1e-6) -> Tuple[
    np.ndarray, Tuple[np.ndarray, ...]]:

    M, N = Phi.shape
    x = np.zeros((N, 1))

    for i in range(max_iter):
        residual = y - np.dot(Phi,x)
        gradient = np.dot(Phi.T,residual)

        x_new = x + (step_size * gradient)
        x_new = thresholdOperator(K, x_new)

        # Проверка сходимости
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    Candidate = np.where(x.ravel() != 0)
    return x, Candidate

#основная функция(сжатие\разжатие изображения)
def iht(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    N = H

    im = np.array(image)

    # Генерация случайной матрицы Phi размером M x N, удовлетворяющей RIP для sigma_3s
    # Здесь конкретно Бернуллиевская матрица
    Phi = np.random.choice([-1, 1], size=(M, N)) / np.sqrt(M)
    img_cs_1d = np.dot(Phi, im)
    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    # безопасный шаг градиента
    step_size = 1.0 / np.linalg.norm(Phi, ord=2) ** 2

    # Обработка каждой колонки изображения по отдельности
    for i in range(W):
        print(i)
        y = np.reshape(img_cs_1d[:, i], (M, 1))
        column_rec, Candidate = cs_iht(y, Theta_1d, K, step_size)

        # Преобразуем в вектор и сохраняем в результирующую матрицу
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = np.dot(matrix, sparse_rec_1d)

    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, img_rec)
    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)

    return img_res
