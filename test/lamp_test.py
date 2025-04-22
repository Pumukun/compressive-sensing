import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join
from framework import ImageCS, lamp
import db


def main():
    # Список изображений для обработки
    image_list = ["../misc/lena.png"]
    output_folder = "../images/lamp"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in image_list:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        print(f"=======================RECONSTUCTING IMAGE {basename}.png=======================")
        rec = lamp(image_path)
        rec_path = os.path.join(output_folder, f"{basename}_reconstructed.png")
        cv2.imwrite(rec_path, rec.get_Image())
        print(f"METRICS: cr={rec.get_CR()}, psnr={rec.get_PSNR()}")
        print(f"Recover saved as: {rec_path}")
        print(f"============================END FOR {basename}.png==============================")


if __name__ == "__main__":
    main()