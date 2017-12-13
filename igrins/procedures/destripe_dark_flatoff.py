import numpy as np
import scipy.ndimage as ni

from ..utils.image_combine import image_median
from ..procedures import destripe_helper as dh

from ..procedures.estimate_sky import (estimate_background,
                                       get_interpolated_cubic)

from ..igrins_libs.logger import logger


def _make_background_mask(dark1):
    # esimate threshold for the initial background destermination
    dark1G = ni.median_filter(dark1, [15, 1])
    dark1G_med, dark1G_std = np.median(dark1G), np.std(dark1G)

    f_candidate = [1., 1.5, 2., 4.]
    for f in f_candidate:
        th = dark1G_med + f * dark1G_std
        m = (dark1G > th)

        k = np.sum(m, axis=0, dtype="f") / m.shape[0]
        if k.max() < 0.6:
            break
    else:
        logger.warning("No suitable background threshold is found")
        m = np.zeros_like(m, dtype=bool)
        f, th = np.inf, np.inf

    k = dict(bg_med=dark1G_med, bg_std=dark1G_std,
             threshold_factor=k, threshold=th)
    return m, k


def make_background_mask(data_list):

    # subtract p64 usin the guard columns
    data_list1 = [dh.sub_p64_from_guard(d) for d in data_list]
    dark1 = image_median(data_list1)

    m, k = _make_background_mask(dark1)

    return m, k


def make_initial_dark(data_list, bg_mask):

    # subtract p64 with the background mask, and create initial background
    data_list2 = [dh.sub_p64_mask(d, bg_mask) for d in data_list]
    dark2 = image_median(data_list2)

    # subtract p64 using the background.
    data_list3 = [dh.sub_p64_with_bg(d, dark2) for d in data_list]
    dark3 = image_median(data_list3)

    return dark3


def model_bg(dark3, destripe_mask):

    # model the backgroound
    V = dark3
    di, min_pixel = 24, 40
    xc, yc, v, std = estimate_background(V, destripe_mask,
                                         di=di, min_pixel=min_pixel)

    nx = ny = 2048
    ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)

    return ZI3


def make_dark_with_bg(data_list, bg_model,
                      destripe_mask=None):

    data_list5 = [dh.sub_with_bg(d, bg_model, destripe_mask)
                  for d in data_list]

    flat5 = image_median(data_list5)
    return flat5


def make_flaton(data_list):
    data_list1 = [dh.sub_p64_from_guard(d) for d in data_list]

    flat_on = image_median(data_list1)

    return flat_on
