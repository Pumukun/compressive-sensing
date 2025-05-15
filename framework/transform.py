import numpy as np

def dct(N: int) -> np.ndarray:
    '''
    Функция создания квадратной матрицы dct.
        N - размер матрицы.
    by Vladislav Gerda
    '''
    mat_dct_1d: np.ndarray = np.zeros((N, N))
    v = range(N)

    for k in range(0, N):  
        dct_1d = np.cos(np.dot(v, k * math.pi / N))

        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)

        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

    return mat_dct_1d


