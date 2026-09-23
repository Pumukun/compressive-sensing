import os
from pathlib import Path

import db
from matplotlib import pyplot as plt

# results column indices: id, original_image, pwd, algorithm, PSNR, SSIM, CR, K, M, height, width
COL_IMAGE, COL_PSNR, COL_SSIM, COL_CR, COL_K, COL_M = 1, 4, 5, 6, 7, 8

PLOTS_DIR = Path(__file__).resolve().parent.parent / 'plots'


def psnr_cr_by_alg(alg: str, show: bool = False) -> list:
    '''
    PSNR against CR for the given algorithm, one chart per (M, image) pair.

    show=False saves into test/plots/<alg>/ (the default); otherwise a window opens.
    Returns the list of saved paths.
    '''
    out_dir = PLOTS_DIR / alg
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict = {}
    for row in db.get_result_by_alg(alg):
        grouped.setdefault(row[COL_M], {}).setdefault(row[COL_IMAGE], []).append(
            (row[COL_CR], row[COL_PSNR])
        )

    if not grouped:
        print(f'no rows in the database for algorithm {alg!r} ({db.DB_PATH})')
        return []

    saved = []
    for m, by_image in sorted(grouped.items()):
        for image, points in sorted(by_image.items()):
            points = sorted(p for p in points if p[0] is not None and p[1] is not None)
            if not points:
                continue

            cr = [p[0] for p in points]
            psnr = [p[1] for p in points]

            fig, ax = plt.subplots()
            ax.plot(cr, psnr, marker='o', label=f'{alg}, M={m}')
            ax.set_xlabel('CR')
            ax.set_ylabel('PSNR, dB')
            ax.set_title(f'{alg} - {image}')
            ax.grid(True)
            ax.legend()

            if show:
                plt.show()
            else:
                path = out_dir / f'M{m}_{image}.png'
                fig.savefig(path, dpi=120, bbox_inches='tight')
                saved.append(path)
            plt.close(fig)

    if saved:
        print(f'charts saved: {len(saved)} in {out_dir}')
    return saved
