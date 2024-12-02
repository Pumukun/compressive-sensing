import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from random import randint 

def GaussianNoise(image_path, stdev=5, show=True):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    blur = cv.GaussianBlur(image, (stdev, stdev), 0)

    if(show):
        plt.subplot(121),plt.imshow(image),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(123),plt.imshow(blur),plt.title('Gaussian')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return blur

def PoissonNoise(image_path, show=True):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    noisy = image + np.random.poisson(image)

    if(show):
        plt.subplot(121),plt.imshow(image, cmap='gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(noisy, cmap='gray'),plt.title('Poisson')
        plt.xticks([]), plt.yticks([])
        plt.show()
        
    return noisy

def SaltAndPepperNoise(image_path, show=True, number_of_pixels=1000):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    noisy = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    row, col = image.shape

    for i in range(number_of_pixels):
        y_coord = randint(0, row - 1)
        x_coord = randint(0, col - 1)

        noisy[y_coord][x_coord] = 255
    
    if(show):
        plt.subplot(121),plt.imshow(image, cmap='gray'),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(noisy, cmap='gray'),plt.title('Shot')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return noisy

def SpeckleNoise(image_path, show=True, variance=0.1):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    shape = (256, 256)
    noise = np.random.normal(0, variance, size=shape)
    speckle_noise = noise * (256 * np.ones(shape))
    noisy = image + speckle_noise

    if(show):
        plt.subplot(121),plt.imshow(image, cmap='grey'),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(noisy, cmap='grey'),plt.title('Speckle')
        plt.xticks([]), plt.yticks([])
        plt.show()

    return noisy
