"""
Given a 2d image, do argmax along the y direction.
Then refine argmax value by nearby three points
to find 2nd order polynomial and its peak.
"""
import numpy as np


def refined_argmax(d, m):
    ym = np.ma.array(d, mask=~m).argmax(axis=0)

    ny = d.shape[0]

    # now we create a 3xn array of points near the max.
    ymm = np.vstack([np.clip(ym - 1, 0, ny-1), ym, np.clip(ym + 1, 0, ny-1)])

    xx = np.arange(len(ym))
    xmm = np.vstack([xx, xx, xx])
    kk = d[ymm, xmm]

    bad_columns = ymm[2] - ymm[0] < 2
    kk[:, bad_columns] = np.nan

    # find a,b,c (from v = a y^2 + b y + c) and do y@max = -b/2a
    a = .5 * (kk[2] + kk[0] - 2 * kk[1])
    b = .5 * (kk[2] - kk[0])
    dy = -b / a / 2.

    return ym, dy
