import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join

from framework import ImageCS, lamp, dct
import db
import unittest

M = [256]
K = [20, 30]

onlyfiles = [f for f in listdir("../ims/") if isfile(join("../ims/", f))]

im_cnt = 0

for image in onlyfiles:
    for m in M:
        for k in K:
            try:

                rec = lamp(f"../ims/{image}", dct(256), m, k)
                cv2.imwrite(f"./M{m}_K{k}.png", rec.get_Image())
            except:
                print(f"Error processing image {image} with M={m} and K={k}")
