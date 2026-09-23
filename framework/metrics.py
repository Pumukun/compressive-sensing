import numpy as np
from typing import Tuple
from math import log10, sqrt
import cv2 as cv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed):
    '''Peak signal-to-noise ratio between the source and the reconstructed image.'''
    _psnr = psnr(original, compressed)
    return _psnr

def SSIM(original, compressed):
    '''Structural similarity index between the source and the reconstructed image.'''
    # Colour images keep the channels on the last axis
    channel_axis = 2 if original.ndim == 3 else None
    metric = ssim(original, compressed, channel_axis=channel_axis)
    return metric

def CR(image_source: np.ndarray, image_compressed: np.ndarray) -> float:
    '''
    Compression ratio: non-zero elements of the source over non-zero elements
    of the sparse representation.
    '''
    source_count: int = int(np.count_nonzero(image_source))
    compressed_count: int = int(np.count_nonzero(image_compressed))

    if compressed_count == 0:
        return float('inf')

    return source_count / compressed_count
