import numpy as np
from typing import Tuple
from math import log10, sqrt
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed):
    '''
    Функция вычисления метрики PSNR. original, compressed - оригинальное и сжатое изображение соответственно
    by Vladislav Gerda
    '''
    _psnr = psnr(original, compressed)
    return _psnr

def SSIM(original, compressed):
    metric = ssim(original, compressed)
    return metric

def CR(image_source: np.ndarray, image_compressed: np.ndarray) -> float:
    
    nonzero_source: Tuple[np.ndarray, ...] = image_source.nonzero()
    nonzero_compressed: Tuple[np.ndarray, ...] = image_compressed.nonzero()

    source_count: int = 0
    compressed_count: int = 0

    for i in nonzero_source:
        source_count += i.size

    for i in nonzero_compressed:
        compressed_count += i.size

    return source_count / compressed_count