from math import log10, sqrt
import cv2
import numpy as np 

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def main():
    original_name = input('Введите название исходного изображения ')
    original = cv2.imread("../data/original/" + original_name) # Исходное изображение
    compressed_name = input("Введите название сжатого изображения ")
    compressed = cv2.imread("../data/compressed/" + compressed_name) # Сжатое изображение
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")

if __name__ == "__main__":
    main()

