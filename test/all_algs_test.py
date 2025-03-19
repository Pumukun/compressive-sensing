import db
from framework import ImageCS, dct, brgp, omp, sp, cosamp
import os
import subprocess
import cv2

try:
    db.create_table()
    db.delete_all()
    print("----------------")
    print("database created")
    print("----------------")
except:
    print("DB troubles")

algorithms = [brgp, omp, sp, cosamp]
images = [f for f in os.listdir("../misc/") if os.path.isfile(os.path.join("../misc/", f))]
M = [256]
K = [i for i in range(10, 11)]

os.system("rm -rf images")
os.mkdir("images")

for alg in algorithms:
    print(f"---------- ALGORITHM {alg.__name__} ----------")
    os.mkdir(f"images/{alg.__name__}")

    for image in images:
        print(f"---------- IMAGE {image} ----------")
        os.mkdir(f"images/{alg.__name__}/{image[:image.find(".png")]}")

        for m in M:
            for k in K:
                try:
                    rec = alg(f"../misc/{image}", dct(256), m, k)
                    cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                    db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{alg.__name__}", rec.get_PSNR(), rec.get_CR(), k, m, 256, 256)
                except:
                    try:
                        rec = alg(f"../misc/{image}", dct(512), m, k)
                        cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                        db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{alg.__name__}", rec.get_PSNR(), rec.get_CR(), k, m, 512, 512)
                    except:
                        try:
                            rec = alg(f"../misc/{image}", dct(1024), m, k)
                            cv2.imwrite(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", rec.get_Image())
                            db.add_result(f"images/{alg.__name__}/{image[:image.find(".png")]}/M{m}_K{k}.png", f"{alg.__name__}", rec.get_PSNR(), rec.get_CR(), k, m, 1024, 1024)
                        except:
                            print(f"image {image} is not processed in algorithm {alg.__name__}")
                            break