import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from os import listdir
from os.path import isfile, join
from framework import ImageCS, brgp, dct
import db

M = [256]
K = range(10, 20, 2)

try:
    db.create_table()
    db.delete_all()
    print("----------------")
    print("database created")
    print("----------------")
except:
    pass

images = [f for f in listdir("../misc/") if isfile(join("../misc/", f))]

try:
    os.remove("images/brgp")
    os.mkdir("images/brgp")
except:
    pass

im_cnt = 0
for i in images:
    try:
        os.mkdir(f"images/brgp/{i}")
    except:
        pass
    
    print(f"---------- IMAGE {i} ----------")

    for m in M:
        for k in K:
            try:
                rec = brgp(f"../misc/{i}", dct(512), m, k)
                cv2.imwrite(f"images/brgp/{i}/{i[:i.find(".png")]}_brgp_M{m}_K{k}.png", rec.get_Image())
                
                db.add_result(f"images/brgp/{i}/{i[:i.find(".png")]}_brgp_M{m}_K{k}.png", "BRGP", rec.get_PSNR(), rec.get_CR(), k, m, 512, 512)

            except:
                try:
                    rec = brgp(f"../misc/{i}", dct(256), m, k)
                    cv2.imwrite(f"images/brgp/{i}/{i[:i.find(".png")]}_brgp_M{m}_K{k}.png", rec.get_Image())

                    db.add_result(f"images/brgp/{i}/{i[:i.find(".png")]}_brgp_M{m}_K{k}.png", "BRGP", rec.get_PSNR(), rec.get_CR(), k, m, 256, 256)

                except:
                    break