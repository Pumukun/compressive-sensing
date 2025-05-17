import os
import threading
from time import time
import argparse
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import db
from framework import ImageCS, dct, sp

parser = argparse.ArgumentParser()
parser.add_argument('-r', action='store_true', help='reset db, delete output images')
parser.add_argument('-s', action='store_true', help='skip process()')
args = parser.parse_args()

images: list[str] = ['lena.png', '4.2.01.png', '4.2.06.png', '4.2.07.png', '5.3.01.png']
M = [512, 1024]
K = list(range(20, 50, 2))

def process() -> None:
    for image in images:
        img = Image.open(f"../misc/{image}")
        _, h = img.size

        for k in K:
            new_pwd = f"images/sp/{image[:image.find(".png")]}/M{h}_K{k}.png"
            rec: ImageCS = sp(f"../misc/{image}", dct(h), h, k)
            cv2.imwrite(new_pwd, rec.get_Image())
            db.add_result(
                new_pwd,
                image,
                "sp",
                rec.get_PSNR(),
                rec.get_SSIM(),
                rec.get_CR(),
                k,
                h,
                h,
                h
            )

def plot() -> None:
    res = db.get_result_by_alg("sp")
    print(res)

    for name in images:
        psnr = []
        cr = []

        for i in res:
            if i[1] == name:
                psnr.append( i[4])
                cr.append(i[6])

        plt.plot(cr, psnr, color='red', marker='o')
        plt.title(name)
        plt.xlabel("Стеепнь сжатия (%)")
        plt.ylabel("PSNR, dB")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    if args.r:
        try:
            db.create_table()
            db.delete_all()
            print("----------------")
            print("database created")
            print("----------------")
        except:
            print("DB troubles")

        os.system("rm -rf images")
        os.mkdir("images")
        os.mkdir("images/sp")

        for image in images:
            os.mkdir(f"images/sp/{image[:image.find(".png")]}")

    if not args.s:
        process()

    plot()
