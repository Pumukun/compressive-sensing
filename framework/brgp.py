import PIL.Image
import numpy as np
import cv2
import math
import framework.metrics as metrics
import PIL
from framework.utils import ImageCS

from framework.omp import cs_omp
from framework.sp import cs_sp

def brgp(image_path: str, matrix: np.ndarray, M: int, K: int) -> ImageCS:
    '''
    BRGP 2d функция. 
        image_path - путь к сжимаемому изображению (изображение квадратное, цветное/ЧБ). 
        matrix - базисная матрица размера NxN. 
        K - количество итераций алгоритма. 
        M - размер вспомогательной матрицы.
    by Grigory Demchenko
    '''
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = PIL.Image.open(image_path)
    image = image.convert("L")
    H, W = image.height, image.width
    N = H

    im = np.array(image)

    Phi = np.random.randn(M,N) / np.sqrt(M) # Формирование матрицы измерений
    img_cs_1d = np.dot(Phi, im)

    sparse_rec_1d = np.zeros((N, W))
    Theta_1d = np.dot(Phi, matrix)

    for i in range(W):
        y = np.reshape(img_cs_1d[:, i], (M, 1))

        _, Candidate_sp = cs_sp(y, Theta_1d, K)
        _, Candidate_omp = cs_omp(y, Theta_1d, K)
        Candidate_BRGP = np.intersect1d(Candidate_omp, Candidate_sp)

        column_rec = cs_brgp(y, Theta_1d, K, Candidate_BRGP)
        x_pre = np.reshape(column_rec, (N))
        sparse_rec_1d[:, i] = x_pre

    img_rec = np.dot(matrix, sparse_rec_1d)

    CR: float = metrics.CR(im, sparse_rec_1d)
    PSNR: float = metrics.PSNR(im, img_rec)

    img_res = ImageCS(img_rec, cr=CR, psnr=PSNR)

    return img_res

def cs_brgp(y: np.ndarray, Phi: np.ndarray, K: int, Candidate: np.ndarray, u: float = 0.8) -> np.ndarray:
    '''
    Вспомогательная функция сжатия векторов. 
        y - сжимаемый вектор. 
        Phi - матрица MxN, являющаяся произведением базисной матрицы и матрицы измерений. 
        K - Количество итераций алгоритма.
    by Vladislav Gerda
    '''  
    (M, N) = Phi.shape

    x = np.zeros((N, 1))                            
    x_temp = np.dot(np.linalg.pinv(Phi[:, Candidate]), y)
    x[Candidate] = x_temp

    r = y - np.dot(Phi, x)
    Candidate_save = Candidate
    r_save = r

    temp = np.abs(np.dot(Phi.T, r))
    max_value = max(temp)

    F = np.where(temp > max_value*u)

    Candidate = np.union1d(Candidate, F).astype(int)

    x = np.zeros((N, 1))
    x_temp = np.dot(np.linalg.pinv(Phi[:, Candidate]), y)
    x[Candidate] = x_temp

    r = y - np.dot(Phi, x)

    while len(Candidate) < K:
        dis = np.linalg.norm(r - r_save)

        if dis < np.linalg.norm(y):
            Candidate_save = Candidate
            r_save = r 

            temp = np.abs(np.dot(Phi.T, r))
            max_value = max(temp)
            F = np.where(temp > max_value * u)

            Candidate = np.union1d(Candidate, F).astype(int)
        else:
            #print('back off')
            Candidate = Candidate_save
            r = r_save

            Candidate_dif = np.setdiff1d(np.arange(0, K - 1, K), Candidate)                            
            temp = np.dot(np.linalg.pinv(Phi[:,Candidate_dif]),y)
            F = np.where(temp == max(temp))

            Candidate = np.union1d(Candidate,F).astype(int)

        x = np.zeros((N, 1))                            
        x_temp = np.dot(np.linalg.pinv(Phi[:, Candidate]), y)
        x[Candidate] = x_temp

        r = y - np.dot(Phi, x)

    T = K
    while T > 0:
        product = np.fabs(np.dot(Phi.T, r))
        top_t_idx = product.argsort(axis = 0)[-T:]
        Candidate = np.union1d(Candidate, top_t_idx).astype(int)

        x_temp = np.dot(np.linalg.pinv(Phi[:, Candidate]), y)
        index = np.fabs(x_temp).argsort(axis=0)[-K:]
        x_temp = x_temp[index]
        Candidate = Candidate[index]

        x = np.zeros((N, 1)) 
        x[Candidate] = x_temp
        r = y - np.dot(Phi, x)

        T = T * u
        T = np.floor(T).astype(int)

    return x

