import db
from matplotlib import pyplot as plt
import os

def psnr_cr_by_alg(alg):
    try:
        os.mkdir("plots")
        os.mkdir(f"plots/{alg}")
    except:
        print('dirs already created')
    data = db.get_result_by_alg(alg)
    prepared_data = dict()
    for row in data:
        if row[8] not in [*prepared_data]:
            prepared_data[row[8]] = dict()
        if row[1] not in [*prepared_data[row[8]]]:
            prepared_data[row[8]][row[1]] = []
        prepared_data[row[8]][row[1]].append([row[4], row[6]])
    for M in [*prepared_data]:
        for image in [*prepared_data[M]]:
            psnr = []
            cr = []
            for [_psnr, _cr] in prepared_data[M][image]:
                psnr.append(_psnr)
                cr.append(_cr)
            plt.plot(cr, psnr, marker='o')
            plt.xlabel("CR")
            plt.ylabel("PSNR")
            plt.grid()
            plt.legend()
            plt.show()
            # plt.savefig(f"plots/{alg}/M{M}_{image}.png")