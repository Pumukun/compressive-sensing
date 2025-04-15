# lamp.py
import numpy as np
import cv2
import maxflow
from numpy.linalg import lstsq
import framework.metrics as metrics
from framework.utils import ImageCS

def compute_tau(x_temp, K_tilde):
    """
    Определяем адаптивный порог tau как 5*K_tilde-й по величине коэффициент по модулю.
    Если 5*K_tilde превышает число коэффициентов, используем минимальное значение.
    """
    N = x_temp.size
    num = min(5 * K_tilde, N)
    tau = np.sort(np.abs(x_temp))[-num]
    return tau

def solve_graph_cut(x_temp, H, W, tau, sigma0, pairwise_weight):
    """
    Решаем задачу MAP для вектора поддержки через графовый разрез.
    
    x_temp          – временная оценка сигнала (вектор длины N)
    tau             – адаптивный порог для функции полезности
    sigma0          – параметр унарного потенциала для состояния s = -1
    pairwise_weight – вес парного потенциала (для 4-связной решётки)
    
    Возвращает:
       s_opt – вектор поддержки с элементами из {-1, +1}
    """
    num_nodes = H * W
    g = maxflow.Graph[float](num_nodes, num_nodes * 4)
    g.add_nodes(num_nodes)

    # Унарные потенциалы (сдвигаем, чтобы значения были неотрицательными):
    U0 = (x_temp**2) / (2 * sigma0**2)  # потенциал для s = -1
    U1 = - (np.abs(x_temp) - tau) / tau   # потенциал для s = +1

    U0 = U0 - U0.min()
    U1 = U1 - U1.min()

    for i in range(num_nodes):
        g.add_tedge(i, U1[i], U0[i])

    # Добавляем попарные ребра для 4-связной решётки (с соседом справа и снизу)
    for i in range(H):
        for j in range(W):
            node_id = i * W + j
            if j + 1 < W:
                neighbor = i * W + (j + 1)
                g.add_edge(node_id, neighbor, pairwise_weight, pairwise_weight)
            if i + 1 < H:
                neighbor = (i + 1) * W + j
                g.add_edge(node_id, neighbor, pairwise_weight, pairwise_weight)
    
    _ = g.maxflow()
    labels = np.empty(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        labels[i] = g.get_segment(i)
    s_opt = 2 * labels - 1
    return s_opt

def prune_signal(x, K_tilde):
    """
    Оставляем только K_tilde коэффициентов с наибольшими по модулю значениями,
    Остальные коэффициенты обнуляем
    """
    x_pruned = np.zeros_like(x)
    if K_tilde <= 0:
        return x_pruned
    idx = np.argsort(-np.abs(x))
    sel = idx[:K_tilde]
    x_pruned[sel] = x[sel]
    return x_pruned

def lamp_reconstruction(y, Phi, K_tilde, max_iter=15, tol=1e-3, sigma0=0.02, pairwise_weight=0.75):
    """
    Восстановление сигнала по алгоритму LaMP с использованием MRF для оценки вектора поддержки.
    
    Параметры:
      y               – вектор измерений (размер M)
      Phi             – матрица измерений (размер M x N)
      K_tilde         – требуемое число ненулевых коэффициентов
      max_iter        – максимальное число итераций
      tol             – порог останова по норме остатка
      sigma0          – параметр унарного потенциала
      pairwise_weight – вес ребер в графе
    
    Возвращает:
      x – восстановленный сигнал (размер N)
      num_iters – число итераций, сделанных алгоритмом
    """
    M, N = Phi.shape
    H = W = int(np.sqrt(N))
    if H * W != N:
        raise ValueError("Невозможно интерпретировать сигнал как квадратное изображение.")

    x = np.zeros(N)
    s = -np.ones(N, dtype=int)
    
    num_iters = max_iter
    for k in range(max_iter):
        # Шаг 1: вычисляем остаток
        r = y - Phi @ x
        
        # Шаг 2: временная оценка сигнала
        x_temp = x + Phi.T @ r
        
        # Шаг 3: MAP-оценка поддержки через графовый разрез
        tau = compute_tau(x_temp, K_tilde)
        s_opt = solve_graph_cut(x_temp, H, W, tau, sigma0, pairwise_weight)
        s = s_opt
        
        # Шаг 4: решаем задачу наименьших квадратов только для выбранных коэффициентов
        S = (s == 1)
        x_new = np.zeros(N)
        if np.count_nonzero(S) > 0:
            Phi_S = Phi[:, S]
            t, _, _, _ = lstsq(Phi_S, y, rcond=None)
            x_new[S] = t
        
        # Применяем обрезание (pruning)
        x_new = prune_signal(x_new, K_tilde)
        
        r_new = y - Phi @ x_new
        if np.linalg.norm(r_new) < tol:
            num_iters = k + 1
            x = x_new
            break
        x = x_new

    return x, num_iters

def lamp(image_path, K_ratio=0.13, M_ratio=0.35) -> ImageCS:
    """
    Обработка изображения, создание переменных для последующий обработки алгоритмом LaMP
    
    Параметры:
      image_path – Путь к обрабатываемому изображению
      K_ratio    - Коэффициент искомой разреженности
      M_ratio    – Коэффициент количества измерений
    
    Возвращает:
      img_res - Восстанавленное изображение с метриками (класса ImageCS)
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение по пути {image_path}")
    
    H, W = img.shape
    if H != W:
        raise ValueError("Изображение должно быть квадратным.")
    
    img_norm = img.astype(np.float32) / 255.0
    dct_coeffs = cv2.dct(img_norm)
    x_dct_true = dct_coeffs.flatten()
    N = x_dct_true.size

    K_tilde = int(K_ratio * N)
    M = int(M_ratio * N)

    Phi = (np.random.randn(M, N).astype(np.float32)) / np.sqrt(M)
    y = Phi @ x_dct_true
    
    x_dct_rec, num_iters = lamp_reconstruction(y, Phi, K_tilde, max_iter=15)
    dct_rec = x_dct_rec.reshape(H, W)
    img_rec = cv2.idct(dct_rec)
    img_rec = np.clip(img_rec, 0, 1)
    img_rec = (img_rec * 255).astype(np.uint8)

    CR: float = metrics.CR(img, dct_rec)
    PSNR: float = metrics.PSNR(img, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)
    return img_res
