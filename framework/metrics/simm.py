import cv2 as cv
from skimage.metrics import structural_similarity as ssim

def SSIM(original_path, compressed_path):
    original = cv.imread(original_path, cv.IMREAD_GRAYSCALE)
    compressed = cv.imread(compressed_path, cv.IMREAD_GRAYSCALE)
    metric = ssim(original, compressed)

    return metric