import numpy as np
import pywt

def dct(N: int) -> np.ndarray:
    '''
    Функция создания квадратной базисной матрицы dct.
        N - размер матрицы.
    by Vladislav Gerda
    '''
    mat_dct_1d: np.ndarray = np.zeros((N, N))
    v = range(N)

    for k in range(N):  
        dct_1d = np.cos(np.dot(v, k * np.pi / N))

        if k > 0:
            dct_1d = dct_1d - np.mean(dct_1d)

        mat_dct_1d[:, k] = dct_1d / np.linalg.norm(dct_1d)

    return mat_dct_1d

def fft(N: int) -> np.ndarray:
    '''
    Функция создания квадратной базисной матрицы fft.
        N - размер матрицы.
    by Vladislav Gerda
    '''
    omega: float = np.exp(-2 * np.pi / N)
    j, k = np.meshgrid(np.arange(N), np.arange(N))
    W = omega ** (j * k)

    return W / np.sqrt(N)

def wavelet_matrix(wavelet_name: str, N: int) -> np.ndarray:
    '''
    Функция создания квадратной базисной матрицы выбранного вейвлета
        wavelet_name - название вейвлета. Возможные варианты: ('dbn', n - порядок вейвлета Добеши)
    '''
    wavelet = pywt.Wavelet(wavelet_name)

    identity = np.eye(N)
    wavelet_matrix = np.zeros((N, N))
    
    for i in range(N):
        col = identity[:, i]
        coeffs = pywt.wavedec(col, wavelet, level=pywt.dwt_max_level(N, wavelet), mode='zero')
        wavelet_col = np.concatenate(coeffs)
        wavelet_matrix[:, i] = wavelet_col[:N]
    
    return wavelet_matrix