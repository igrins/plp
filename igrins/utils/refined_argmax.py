"""
Given a 2d image, do argmax along the y direction.
Then refine argmax value by nearby three points
to find 2nd order polynomial and its peak.
"""
import numpy as np


def find_peak_poly2d(kk):
    """
    find the peak location and peak value form

    y = ax^2 + bc + c
    where kk is a 3 values (or list of 3 values) given at [-1, 0, 1]

    returns x at the peak and peak it self
    """

    # find a,b,c (from v = a y^2 + b y + c) and do y@max = -b/2a
    c = kk[1]
    a = .5 * (kk[2] + kk[0] - 2 * c)
    b = .5 * (kk[2] - kk[0])

    x = -b / a / 2.

    peak = -1. * b * b / (4. * a) + c

    return x, peak


def test_peak():
    x, peak = find_peak_poly2d([-1, 1, 1])
    assert (x, peak) == (0.5, 1.25)


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
    c = kk[1]
    a = .5 * (kk[2] + kk[0] - 2 * c)
    b = .5 * (kk[2] - kk[0])

    dy = -b / a / 2.

    peak = 3. * b * b / (4. * a * a) + c
    dy, peak = find_peak_poly2d(kk)

    return ym, dy, peak
