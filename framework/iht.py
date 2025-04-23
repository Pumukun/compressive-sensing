import cv2
import numpy as np
import math
import framework.metrics as metrics
from framework.utils import ImageCS
from typing import Tuple


def iht(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    # Загружаем изображение в градациях серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape()          # Высота и ширина изображения
    N = H                       # Размерность сигнала (предполагаем, что базис имеет размер H)

    im = np.array(image)       # Преобразуем изображение в массив NumPy

    # Генерация случайной измерительной матрицы Phi размером M x N
    Phi = np.random.randn(M, N) / np.sqrt(M)

    # Сжатие изображения: применяем Phi к каждой колонке
    img_cs_1d = np.dot(Phi, im)  # Результат: сжатые измерения размером M x W

    # Матрица для хранения разреженного представления (после восстановления)
    sparse_rec_1d = np.zeros((N, W))

    # Предварительное вычисление Theta = Phi * Ψ (где Ψ — матрица преобразования, например DCT)
    Theta_1d = np.dot(Phi, matrix)

    # Обработка каждой колонки изображения по отдельности
    for i in range(W):
        # Извлекаем колонку сжатого сигнала y размером M x 1
        y = np.reshape(img_cs_1d[:, i], (M, 1))

        # Восстанавливаем разреженное представление x при помощи IHT
        column_rec, Candidate = cs_iht(y, Theta_1d, K)

        # Преобразуем в вектор и сохраняем в результирующую матрицу
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    # Восстанавливаем изображение: применяем Ψ к разреженному представлению
    img_rec = np.dot(matrix, sparse_rec_1d)

    # Вычисляем метрики сжатия и качества
    CR: float = metrics.CR(image, sparse_rec_1d)      # Compression Ratio
    PSNR: float = metrics.PSNR(image, img_rec)        # Peak Signal-to-Noise Ratio

    # Упаковываем результат в объект ImageCS
    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)

    return img_res

def dct(N: int) -> np.ndarray:
    mat_dct_1d: np.ndarray = np.zeros((N, N))
    v = range(N)

    for k in range(0, N):
        dct_1d = np.cos(np.dot(v, k * math.pi / N))

        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)

        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

    return mat_dct_1d