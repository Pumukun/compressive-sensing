import numpy as np
from matplotlib import pyplot as plt
import cv2

from framework import ImageCS, brgp, dct

#M = [32, 64, 128, 256]
#K = [5, 10, 20, 30, 50, 70, 100, 120, 150, 170, 200]

M = [256]
K = [20]

#M = [32, 64, 128]
#K = range(10, 50, 2)

mm = []
kk = []
psnr = []
cr = []

im_cnt = 0
for m in M:
    for k in K:
        print(f"---------- IMAGE {im_cnt} ----------")
        print("M:   ", m)
        print("K:   ", k)

        rec = brgp("../lena.png", dct(256), m, k)
        cv2.imwrite(f"lena_omp_M{m}_K{k}.png", rec.get_Image())
        print(f"CR: {rec.get_CR()}, PSNR: {rec.get_PSNR()}")

        mm.append(m)
        kk.append(k)
        psnr.append(rec.get_PSNR())
        cr.append(rec.get_CR())

        im_cnt += 1

print("mm = ", mm)
print("kk = ", kk)
print("psnr = ", psnr)
print("cr = ", cr)


for i in range(0, len(M)):
    plt.plot(kk[i * len(K): (i + 1) * len(K)], cr[i * len(K): (i + 1) * len(K)], label=f"M = {M[i]}", marker="o")

plt.xlabel("K")
plt.ylabel("PSNR")
plt.grid()
plt.legend()
plt.show()
