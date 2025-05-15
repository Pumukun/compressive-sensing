import db
from framework import dct, brgp, omp, sp, cosamp
import os
import cv2
import threading
from time import time
import numpy as np

try:
    db.create_table()
    db.delete_all()
    print("----------------")
    print("database created")
    print("----------------")
except:
    print("DB troubles")

algorithms = [cosamp]
images = [f for f in os.listdir("../misc/") if os.path.isfile(os.path.join("../misc/", f))]
M = [512, 1024]
K = [i for i in range(1, 1000, 50)]

os.system("rm -rf images")
os.mkdir("images")

def processing_images(alg):
    global images, M, K
    t1 = time()
    for image in images:
        os.mkdir(f"images/{alg.__name__}/{image[:image.find(".png")]}")

        for m in M:
            for k in K:
                try:
                    rec = alg(f"../misc/{image}", dct(256), k, m)
                    cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                    print(rec.get_Image())
                    print(np.argmax(rec.get_Image()))
                    db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{image[:image.find(".png")]}", f"{alg.__name__}", rec.get_PSNR(), rec.get_SSIM(), rec.get_CR(), k, m, 256, 256)
                except:
                    try:
                        rec = alg(f"../misc/{image}", dct(512), k, m)
                        cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                        print(rec.get_Image())
                        print(np.max(rec.get_Image()))
                        db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{image[:image.find(".png")]}", f"{alg.__name__}", rec.get_PSNR(), rec.get_SSIM(), rec.get_CR(), k, m, 512, 512)
                    except:
                        try:
                            rec = alg(f"../misc/{image}", dct(1024), k, m)
                            cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                            print(rec.get_Image())
                            print(np.argmax(rec.get_Image()))
                            db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{image[:image.find(".png")]}", f"{alg.__name__}", rec.get_PSNR(), rec.get_SSIM(), rec.get_CR(), k, m, 1024, 1024)
                        except:
                            print(f"image {image} is not processed in algorithm {alg.__name__}")
                            break

    t2 = time()
    print(f"algorithm {alg.__name__} take {round((t2-t1) / 60), 2}min")

for alg in algorithms:
    os.mkdir(f"images/{alg.__name__}")
    x = threading.Thread(target=processing_images, args=(alg, ))
    x.start()