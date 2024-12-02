import numpy as np
from typing import Tuple
from math import log10, sqrt
import cv2
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


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
    metric = ssim(original, compressed)

    return metric
