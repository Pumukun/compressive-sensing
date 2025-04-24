import cv2
import numpy as np
import math
import framework.metrics as metrics
from framework.utils import ImageCS
from typing import Tuple


def cs_iht(y: np.ndarray, Phi: np.ndarray, K: int, max_iter: int = 50, tol: float = 1e-6) -> Tuple[
    np.ndarray, Tuple[np.ndarray, ...]]:
    """
    Iterative Hard Thresholding (IHT) algorithm.

    Args:
        y: измеренный сигнал (M x 1)
        Phi: матрица измерений (M x N)
        K: максимальное количество ненулевых коэффициентов
        max_iter: максимальное число итераций
        tol: порог сходимости по ошибке

    Returns:
        Восстановленный вектор (N x 1), индексы ненулевых элементов
    """
    M, N = Phi.shape
    x = np.zeros((N, 1))  # начальное приближение
    step_size = 1.0 / np.linalg.norm(Phi, ord=2) ** 2  # безопасный шаг градиента

    for i in range(max_iter):
        residual = y - Phi @ x
        gradient = Phi.T @ residual
        x_new = x + step_size * gradient

        # Жесткий порог: оставим только K наибольших по модулю элементов
        abs_x = np.abs(x_new)
        threshold = np.partition(abs_x.ravel(), -K)[-K]
        mask = abs_x >= threshold
        x_new[~mask] = 0

        # Проверка сходимости
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    Candidate = np.where(x.ravel() != 0)
    return x, Candidate

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

image_path = "grayscale.jpg"  # Убедись, что изображение есть в этой папке
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
H, W = image.shape()

N = H        # Размерность DCT-базиса (по высоте)
M = int(N * 0.5)  # Количество измерений (например, 50% от исходного)
K = 40       # Уровень разреженности

# Создание базисной матрицы (например, DCT)
matrix = dct(N)

# Вызов IHT-функции
result = iht(image_path, matrix, M, K)

# Проверка результатов
print(f"PSNR: {result.psnr:.2f} dB")
print(f"CR: {result.cr:.2f}")

# Сохраняем восстановленное изображение
cv2.imwrite("reconstructed_iht.png", np.clip(result.data, 0, 255).astype(np.uint8))