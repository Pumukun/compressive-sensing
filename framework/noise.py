import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from random import randint 
import numpy.typing as npt

'''by Vladislav Gerda'''

def GaussianNoise(image_path: str, stdev: int = 5, show: bool = True) -> npt.NDArray[np.uint8]:
    '''
    Добавляет гауссов шум к изображению и применяет гауссово размытие

    Args:
        image_path: Путь к изображению
        stdev: Стандартное отклонение для гауссова размытия
        show: Показать оригинальное и зашумленное изображение pyplot

    Returns:
        Зашумленное изображение в виде массива numpy.
    '''
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    blur = cv.GaussianBlur(image, (stdev, stdev), 0)

    if show:
        plt.subplot(121), plt.imshow(image), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(blur), plt.title('Gaussian')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return blur

def PoissonNoise(image_path: str, show: bool = True) -> npt.NDArray[np.uint8]:
    '''
    Добавляет пуассоновский шум к изображению

    Args:
        image_path: Путь к изображению
        show: Показать оригинальное и зашумленное изображение pyplot

    Returns:
        Зашумленное изображение в виде массива numpy
    '''
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    noisy = image + np.random.poisson(image)

    if show:
        plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(noisy, cmap='gray'), plt.title('Poisson')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    return noisy

def SaltAndPepperNoise(image_path: str, show: bool = True, number_of_pixels: int = 1000) -> npt.NDArray[np.uint8]:
    '''
    Добавляет шум типа "соль и перец" к изображению

    Args:
        image_path: Путь к изображению
        show: Показать оригинальное и зашумленное изображение pyplot
        number_of_pixels: Количество пикселей для зашумления

    Returns:
        Зашумленное изображение в виде массива numpy
    '''
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    noisy = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    row, col = image.shape

    for i in range(number_of_pixels):
        y_coord = randint(0, row - 1)
        x_coord = randint(0, col - 1)

        noisy[y_coord][x_coord] = 255
    
    if show:
        plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(noisy, cmap='gray'), plt.title('Shot')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return noisy

def SpeckleNoise(image_path: str, show: bool = True, variance: float = 0.1) -> npt.NDArray[np.uint8]:
    '''
    Добавляет speckle шум к изображению

    Args:
        image_path: Путь к изображению.
        show: Показать оригинальное и зашумленное изображение pyplot
        variance: Дисперсия шума.

    Returns:
        Зашумленное изображение в виде массива numpy
    '''
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    shape = (256, 256)
    noise = np.random.normal(0, variance, size=shape)
    speckle_noise = noise * (256 * np.ones(shape))
    noisy = image + speckle_noise

    if show:
        plt.subplot(121), plt.imshow(image, cmap='grey'), plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(noisy, cmap='grey'), plt.title('Speckle')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return noisy
