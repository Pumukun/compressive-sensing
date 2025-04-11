import numpy as np
import cv2
import math
import framework.metrics as metrics
from framework.utils import ImageCS
from typing import Tuple

def lamp(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    print("[DEBUG] Начало выполнения функции lamp")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")
    print("[DEBUG] Изображение успешно загружено")

    H, W = image.shape
    N = H

    im = np.array(image)

    Phi = np.random.randn(M, N) / np.sqrt(M)
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(W):
        print(f"[DEBUG] Обработка столбца {i + 1}/{W}")
        y = np.reshape(img_cs_1d[:, i], (M, 1))
        column_rec = cs_lamp(y, Theta_1d, K)
        sparse_rec_1d[:, i] = column_rec
    
    image_rec = np.dot(matrix, sparse_rec_1d)
    print("[DEBUG] Восстановление изображения завершено")
    
    CR: float = metrics.CR(image, sparse_rec_1d)
    PSNR: float = metrics.PSNR(image, image_rec)
    print(f"[DEBUG] CR: {CR}, PSNR: {PSNR}")
    img_8u = (image_rec * 255).astype(np.uint8)

    img_res = ImageCS(img_8u)
    print("[DEBUG] Завершение функции lamp")
    return img_res


def cs_lamp(y: np.ndarray, Phi: np.ndarray, K: int, max_iters=30) -> np.ndarray:
    print("[DEBUG] Начало выполнения функции cs_lamp")
    M, N = Phi.shape

    if y.shape[0] != M:
        raise ValueError(f"Размерность y ({y.shape[0]}) не совпадает с количеством строк Phi ({M}).")
    
    threshold = 0.005 * np.linalg.norm(y)

    x_current = np.zeros(N, dtype=float)
    s_current = -1 * np.ones(N, dtype=int)

    for j in range(max_iters):
        r_k = y.flatten() - Phi @ x_current
        x_temp = Phi.T @ r_k + x_current
        s_current = estimate_support(x_temp)
        x_current = estimate_target(y, Phi, s_current, K)
        residual_norm = np.linalg.norm(r_k)
        if residual_norm < threshold:
            break
    
    print("[DEBUG] Завершение функции cs_lamp")
    return x_current

def create_lam_ij_1d(N, weight=1):
        lam_ij = np.zeros((N, N), dtype=float)
        for i in range(N-1):
            lam_ij[i, i+1] = weight
            lam_ij[i+1, i] = weight
        return lam_ij

def create_lam_i_1d(N, bias_value=0.0):
        return np.full((N,), bias_value, dtype=float)

def log_p_x_given_s(x_val, s_val, tau=30.0, sigma=0.01):
    if s_val == 1:
        return -np.log(2 * tau) - np.abs(x_val) / tau
    elif s_val == -1:
        return -np.log(np.sqrt(2 * np.pi) * sigma) - (x_val ** 2) / (2 * sigma ** 2)
    else:
        raise ValueError("s_val должен быть 1 или -1")
    
def neighbors_1d(i, N):
    neighbors = []
    if i  - 1 >= 0:
        neighbors.append(i - 1)
    if i + 1 < N:
        neighbors.append(i + 1)
    return neighbors

def estimate_support(x_temp, smooth_iterations=5):
    
    N = x_temp.shape[0]

    lam_i = create_lam_i_1d(N)
    lam_ij = create_lam_ij_1d(N)

    s = np.empty(N, dtype=int)

    for i in range(N):
        cost_pos = -(lam_i[i] + log_p_x_given_s(x_temp[i], +1))
        cost_neg = -(lam_i[i] + log_p_x_given_s(x_temp[i], -1))
        s[i] = +1 if cost_pos < cost_neg else -1


    for iteration in range(smooth_iterations):
        changed = False
        for i in range(N):
            neighbors = neighbors_1d(i, N)
            pairwise_sum = 0.0
            for j in neighbors:
                pairwise_sum += lam_ij[i, j] * s[j]
            
            cost_pos = -(lam_i[i] + log_p_x_given_s(x_temp[i], +1)) - pairwise_sum
            cost_neg = -(lam_i[i] + log_p_x_given_s(x_temp[i], -1)) + pairwise_sum

            new_state = +1 if cost_pos < cost_neg else -1
            if new_state != s[i]:
                changed = True
                s[i] = new_state
        if not changed:
            break
    return s


def estimate_target(y, Phi, s, K):
    N = Phi.shape[1]
    indices = np.where(s == 1)[0]
    if len(indices) == 0:
        return np.zeros(N)
    
    Phi_sub = Phi[:, indices]
    
    x_sub, residuals, rank, s_vals = np.linalg.lstsq(Phi_sub, y, rcond=None)

    x_full = np.zeros(N)
    x_full[indices] = x_sub.flatten()

    if K < N:
        abs_vals = np.abs(x_full)
        top_k_indices = np.argsort(-abs_vals)[:K]
        pruned = np.zeros(N)
        pruned[top_k_indices] = x_full[top_k_indices]
        return pruned
    else:
        return x_full