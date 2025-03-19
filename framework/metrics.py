import numpy as np
from typing import Tuple
from math import log10, sqrt
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr


def PSNR(original, compressed):
    
    (M, N) = original.shape
    d1 = (original - compressed)
    print(np.max(original), np.max(compressed))
    s = 0
    for i in d1:
        for j in i:
            s += j**2
    mse = s / (M * N)
    print(sqrt(mse))
    # if (mse == 0):
    #     return 100
    # max_pixel = 255.0
    # psnr = 20 * log10(max_pixel / sqrt(mse))
    _psnr = psnr(original, compressed)
    return _psnr


def CR(image_source: np.ndarray, image_compressed: np.ndarray, print_image_array = False) -> float:
    if print_image_array:
        print("source:\n", image_source)
        print("comressed:\n", image_compressed)
    
    nonzero_source: Tuple[np.ndarray, ...] = image_source.nonzero()
    nonzero_compressed: Tuple[np.ndarray, ...] = image_compressed.nonzero()

    source_count: int = 0
    compressed_count: int = 0

    for i in nonzero_source:
        source_count += i.size

    for i in nonzero_compressed:
        compressed_count += i.size

    return source_count / compressed_count


def SSIM(original_path, compressed_path):
    original = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    compressed = cv.imread(compressed_path, cv.IMREAD_GRAYSCALE)
    #metric = ssim(original, compressed)
    metric = None

    return metric
