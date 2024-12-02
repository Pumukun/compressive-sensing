import numpy as np
import cv2

from framework import ImageCS, cosamp, dct

#M = [32, 64, 128, 256]
#K = [5, 10, 20, 30, 50, 70, 100, 120, 150, 170, 200]

M = [128]
K = [20]

im_cnt = 0
for m in M:
    for k in K:
        print(f"---------- IMAGE {im_cnt} ----------")
        print("M:   ", m)
        print("K:   ", k)

        rec = cosamp("../lena.png", dct(256), m, k)
        cv2.imwrite(f"lena_cosamp_M{m}_K{k}.png", rec.get_Image())
        print(f"CR: {rec.get_CR()}, PSNR: {rec.get_PSNR()}")

        im_cnt += 1


