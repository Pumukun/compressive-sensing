import numpy as np
import cv2
import maxflow
from numpy.linalg import lstsq
import framework.metrics as metrics
from framework.utils import ImageCS


def lamp(image_path: str,
         K_ratio: float = 0.2,
         M_ratio: float = 0.35,
         block_width: int = 2) -> ImageCS:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение по пути {image_path}")
    H, W = img.shape
    if H != W:
        raise ValueError("Изображение должно быть квадратным.")

    img_f = img.astype(np.float32) / 255.0
    dct_full = cv2.dct(img_f)
    dct_rec = np.zeros_like(dct_full)

    stride = max(1, block_width - 1)

    N_block = H * block_width
    M_block = max(1, int(round(M_ratio * N_block)))
    Phi_block = np.random.randn(M_block, N_block).astype(np.float32) / np.sqrt(M_block)
    K_tilde_block = max(1, int(round(K_ratio * N_block)))

    for j0 in range(0, W, stride):
        j1 = min(j0 + block_width, W)
        Wb = j1 - j0
        block_true = dct_full[:, j0:j1].reshape(-1)

        if Wb == block_width:
            Phi = Phi_block
            Kt = K_tilde_block
        else:
            N_b = H * Wb
            M_b = max(1, int(round(M_ratio * N_b)))
            Phi = np.random.randn(M_b, N_b).astype(np.float32) / np.sqrt(M_b)
            Kt = max(1, int(round(K_ratio * N_b)))

        y = Phi.dot(block_true)
        x_rec_b = lamp_reconstruction_block(
            y, Phi, Kt, H, Wb
        )
        mat = x_rec_b.reshape(H, Wb)

        if j0 == 0:
            dct_rec[:, :Wb] = mat
        else:
            dct_rec[:, j0+1:j1] = mat[:, 1:]

    img_rec_f = cv2.idct(dct_rec)
    img_rec = np.clip(img_rec_f, 0, 1)
    img_out = (img_rec * 255).astype(np.uint8)


    CR = metrics.CR(img, dct_rec)
    PSNR = metrics.PSNR(img, img_out)
    return ImageCS(img_out, cr=CR, psnr=PSNR)


def lamp_reconstruction_block(y: np.ndarray,
                              Phi: np.ndarray,
                              K_tilde: int,
                              H: int,
                              W_block: int,
                              max_iter: int = 15,
                              tol: float = 1e-3,
                              sigma0: float = 0.02,
                              pairwise_weight: float = 1) -> np.ndarray:
    N_block = H * W_block
    x = np.zeros(N_block, dtype=float)

    for k in range(max_iter):
        r = y - Phi.dot(x)
        x_temp = x + Phi.T.dot(r)
        tau = compute_tau(x_temp, K_tilde)
        s_opt = solve_graph_cut(x_temp, H, W_block, tau, sigma0, pairwise_weight)
        S = (s_opt == 1)

        x_new = np.zeros_like(x)
        if S.any():
            Phi_S = Phi[:, S]
            t, *_ = lstsq(Phi_S, y, rcond=None)
            x_new[S] = t
        x_new = prune_signal(x_new, K_tilde)

        if np.linalg.norm(y - Phi.dot(x_new)) < tol:
            x = x_new
            break
        x = x_new
    return x


def compute_tau(x_temp: np.ndarray, K_tilde: int) -> float:
    N = x_temp.size
    num = min(5 * K_tilde, N)
    absx = np.abs(x_temp)
    return float(np.partition(absx, -num)[-num])

def prune_signal(x: np.ndarray, K_tilde: int) -> np.ndarray:
    x_pruned = np.zeros_like(x)
    if K_tilde <= 0:
        return x_pruned
    idx = np.argpartition(-np.abs(x), K_tilde - 1)[:K_tilde]
    x_pruned[idx] = x[idx]
    return x_pruned

def solve_graph_cut(x_temp: np.ndarray,
                    H: int,
                    W_block: int,
                    tau: float,
                    sigma0: float,
                    pairwise_weight: float) -> np.ndarray:
    N_block = H * W_block
    g = maxflow.Graph[float](N_block, N_block * 8)
    g.add_nodes(N_block)

    # unary potentials
    U0 = (x_temp**2) / (2 * sigma0**2)    # s=-1
    U1 = - (np.abs(x_temp) - tau) / tau   # s=+1
    U0 -= U0.min()
    U1 -= U1.min()

    for i in range(N_block):
        g.add_tedge(i, U1[i], U0[i])

    # pairwise edges with 8‑connectivity
    for i in range(H):
        for j in range(W_block):
            u = i * W_block + j
            # horizontal
            if j + 1 < W_block:
                g.add_edge(u, u + 1, pairwise_weight, pairwise_weight)
            # vertical
            if i + 1 < H:
                g.add_edge(u, u + W_block, pairwise_weight+1, pairwise_weight+1)
            # diagonal down-right
            if i + 1 < H and j + 1 < W_block:
                g.add_edge(u, u + W_block + 1, pairwise_weight-0.5, pairwise_weight-0.5)
            # diagonal down-left
            if i + 1 < H and j - 1 >= 0:
                g.add_edge(u, u + W_block - 1, pairwise_weight-0.5, pairwise_weight-0.5)

    g.maxflow()
    labels = np.array([g.get_segment(i) for i in range(N_block)], dtype=int)
    return 2 * labels - 1