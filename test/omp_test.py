import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join

from framework import ImageCS, omp, dct
import db

#M = [32, 64, 128, 256]
#K = [5, 10, 20, 30, 50, 70, 100, 120, 150, 170, 200]

#M = [128]
#K = [20]

M = [256]
K = range(10, 20, 2)

mm = []
kk = []
psnr = []
cr = []

onlyfiles = [f for f in listdir("../misc/") if isfile(join("../misc/", f))]
try:
    os.mkdir("images/omp")
    os.remove("images/omp")
except:
    pass

im_cnt = 0

for i in onlyfiles:
    try:
        os.mkdir(f"images/omp/{i}")
    except:
        print(i)
    for m in M:
        for k in K:
            try:

                rec = omp(f"../misc/{i}", dct(512), m, k)
                cv2.imwrite(f"images/omp/{i}/{i[:i.find(".png")]}_omp_M{m}_K{k}.png", rec.get_Image())

                print(f"---------- IMAGE {im_cnt} ----------")
                print("M:   ", m)
                print("K:   ", k)
                print(f"CR: {rec.get_CR()}, PSNR: {rec.get_PSNR()}")

                mm.append(m)
                kk.append(k)
                psnr.append(rec.get_PSNR())
                cr.append(rec.get_CR())

                db.add_result(f"images/omp/{i}/{i[:i.find(".png")]}_omp_M{m}_K{k}.png", "OMP", rec.get_PSNR(), rec.get_CR(), k, m, 512, 512)

                im_cnt += 1
            except:
                break

print("mm = ", mm)
print("kk = ", kk)
print("psnr = ", psnr)
print("cr = ", cr)


# for i in range(0, len(M)):
#     plt.plot(kk[i * len(K): (i + 1) * len(K)], cr[i * len(K): (i + 1) * len(K)], label=f"M = {M[i]}", marker="o")

plt.plot(cr, psnr, marker="o")

plt.xlabel("K")
plt.ylabel("PSNR")
plt.grid()
plt.legend()
plt.show()
