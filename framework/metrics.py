import numpy as np
from typing import Tuple
from math import log10, sqrt
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed):
    normilized_original = original / 255
    normilized_compressed = compressed / 255
    _psnr = psnr(normilized_original, normilized_compressed)
    # _psnr = psnr(original, compressed)
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


def SSIM(original, compressed):
    metric = ssim(original, compressed)
    return metric
