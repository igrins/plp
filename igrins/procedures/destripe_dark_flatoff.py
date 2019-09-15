import numpy as np

from ..utils.image_combine import image_median
from ..procedures import destripe_helper as dh

from ..procedures.estimate_sky import (estimate_background,
                                       get_interpolated_cubic)

from ..procedures.readout_pattern_helper import apply_rp_2nd_phase


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


def _sub_median_row_with_mask(d1, mask):
    k = np.ma.array(d1, mask=mask).filled(np.nan)

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis)')

        c = np.nanmedian(k, axis=1)

    return dh.sub_column(d1, c)


def _sub_with_bg_old(d, bg, destripe_mask=None):
    d0 = d - bg
    d1 = dh.sub_p64_with_mask(d0, destripe_mask)
    d2 = _sub_median_row_with_mask(d1, destripe_mask)
    return d2 + bg


def _sub_with_bg_201909(d, bg, destripe_mask=None):

    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis)')

        r = apply_rp_2nd_phase(d - bg, destripe_mask) + bg

    return r


def make_dark_with_bg(data_list, bg_model,
                      destripe_mask=None):

    data_list5 = [_sub_with_bg_201909(d, bg_model, destripe_mask)
                  for d in data_list]

    flat5 = image_median(data_list5)
    return flat5


def make_flaton(data_list):
    data_list1 = [dh.sub_p64_from_guard(d) for d in data_list]

    flat_on = image_median(data_list1)

    return flat_on
