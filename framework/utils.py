from math import log10, sqrt
import numpy as np
import cv2

class ImageCS():
    '''
    Класс для представления изображения с метриками сжатия

    Атрибуты:
        __matrix (np.ndarray): Данные изображения в формате numpy.ndarray
        __cr (float): Коэффициент сжатия изображения. По умолчанию 0.0
        __psnr (float): Пиковое отношение сигнал/шум. По умолчанию 0.0
        __ssim (float): Индекс структурного сходства. По умолчанию 0.0

    by Grigory Demchenko
    '''
    def __init__(self, matrix=np.ndarray((0,0)), cr: float=0.0, psnr: float=0.0, ssim: float=0.0):
        self.__matrix = matrix
        self.__cr: float = cr
        self.__psnr: float = psnr
        self.__ssim: float = ssim

    def get_Image(self) -> np.ndarray:
        return self.__matrix

    def get_CR(self) -> float:
        return self.__cr

    def get_PSNR(self) -> float:
        return self.__psnr

    def get_SSIM(self) -> float:
        return self.__ssim


    def set_Image(self, image: np.ndarray) -> None:
        self.__matrix = image

    def set_CR(self, cr) -> None:
        self.__cr = cr

    def set_PSNR(self, psnr) -> None:
        self.__psnr = psnr

    def set_SSIM(self, ssim) -> None:
        self.__ssim = ssim
