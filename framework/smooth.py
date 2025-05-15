import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def Mean_filter(image_path, k, show=True):
  image = cv.imread(image_path)
  kernel = np.ones((k,k), np.float32)/(k**2)
  blur = cv.filter2D(image, -1, kernel)

  if(show):
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Mean_filter')
    plt.xticks([]), plt.yticks([])
    plt.show()

  return blur

def Median_filter(image_path, k, show=True):
  image = cv.imread(image_path)
  blur = cv.medianBlur(image, k)

  if(show):
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Median_filter')
    plt.xticks([]), plt.yticks([])
    plt.show()

  return blur

def Gaussian_filter(image_path, k, show=True):
  image = cv.imread(image_path)
  blur = cv.GaussianBlur(image, (k, k), 0)

  if(show):
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Gaussian_filter')
    plt.xticks([]), plt.yticks([])
    plt.show()

  return blur

def Bilateral_filter(image_path, k, show=True):
  image = cv.imread(image_path)
  blur = cv.bilateralFilter(image, k, 75, 75)

  if(show):
    plt.subplot(121),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Bilateral_filter')
    plt.xticks([]), plt.yticks([])
    plt.show()

  return blur