import omp as o
import numpy as np
import cv2

# rec = o.omp("../test.jpg", o.dct(256), 256, 50)
# rec = o.omp("../test.jpg", np.random.normal(0, 256), 256, 50)

M = [32, 64, 128, 256]
K = [5, 10, 20, 30, 50, 70, 100, 120, 150, 170, 200]

#rec = o.omp("../lena.png", o.dct(256), 256, 5)

im_cnt = 0
for m in M:
    for k in K:
        print(f"---------- IMAGE {im_cnt} ----------")
        print("M:   ", m)
        print("K:   ", k)

        rec = o.omp("../lena.png", o.dct(256), m, k)
        cv2.imwrite(f"lena_omp_{im_cnt}.png", rec)

        im_cnt += 1


