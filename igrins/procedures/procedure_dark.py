from __future__ import print_function

from collections import namedtuple

import numpy as np
import scipy.ndimage as ni
import pandas as pd

from astropy.io.fits import Card, HDUList, PrimaryHDU

from ..utils.image_combine import image_median as stsci_median
from ..procedures import destripe_dark_flatoff as dh

from .. import get_obsset_helper, DESCS

from ..utils.json_helper import json_dumps

from .ascii_plot import (asciiplot_per_amp, pad_with_axes,
                         pad_yaxis_label, pad_xaxis_label,
                         pad_title, to_string, markers)
# from igrins.procedures.readout_pattern import (
#     PatternAmpP2,
#     PatternP64Zeroth,
#     PatternP64First,
#     PatternP64ColWise,
#     PatternColWiseBias,
#     PatternColWiseBiasC64,
#     PatternRowWiseBias,
#     PatternAmpWiseBiasC64
# )

from ..procedures.readout_pattern import pipes, apply as apply_pipe
from ..procedures.readout_pattern_guard import remove_pattern_from_guard

# from igrins.procedures.readout_pattern import (
#     sub_amp_p2,
#     sub_p64_slope,
#     sub_p64_pattern,
#     sub_col_median_slow,
#     sub_col_median,
#     sub_row_median,
#     sub_amp_bias_variation,
#     sub_p64_pattern_each
# )

# from igrins.procedures.readout_pattern import (
#     get_amp_p2,
#     get_p64_slope,
#     get_p64_pattern,
#     get_col_median_slow,
#     get_col_median,
#     get_row_median,
#     get_amp_bias_variation,
#     get_p64_pattern_each,
# )

# from igrins.procedures.readout_pattern import _apply


def make_pair_subtracted_images(obsset):

    hdu_list = obsset.get_hdus()

    n = len(hdu_list)
    cube = []
    for i in range(n - 1):
        cube.append(hdu_list[i].data - hdu_list[i+1].data)

    hdul = obsset.get_hdul_to_write(([], np.array(cube)))
    obsset.store(DESCS["PAIR_SUBTRACTED_IMAGES"], hdul, cache_only=False)


def apply_rp_2nd_phase(d, mask=None):
    if mask is None:
        mask = np.zeros(d.shape, dtype=bool)
    else:
        mask = mask.copy()

    mask[:4] = True
    mask[-4:] = True

    p = [pipes[k] for k in ['p64_1st_order',
                            'col_wise_bias_c64',
                            'amp_wise_bias_r2',
                            'col_wise_bias']]

    return apply_pipe(d, p, mask=mask)


def apply_rp_3rd_phase(d):
    p = [pipes[k] for k in ['p64_per_column',
                            'row_wise_bias',
                            'amp_wise_bias_c64']]

    return apply_pipe(d, p)


def make_guard_n_bg_subtracted_images(obsset):

    hdu_list = obsset.get_hdus()

    # n = len(hdu_list)
    cube0 = np.array([hdu.data
                     for hdu in hdu_list])

    cube = np.array([remove_pattern_from_guard(hdu.data)
                     for hdu in hdu_list])

    bg = np.median(cube, axis=0)

    cube1 = cube - bg

    bias_mask = obsset.load_resource_for("bias_mask")

    # cube20 = np.array([apply_rp_2nd_phase(d1) for d1 in cube1])
    cube2 = np.array([apply_rp_2nd_phase(d1, mask=bias_mask) for d1 in cube1])
    cube3 = np.array([apply_rp_3rd_phase(d1) for d1 in cube2])

    hdu_list = [([("EXTNAME", "DIRTY")], cube0),
                ([("EXTNAME", "GUARD-REMOVED")], cube1),
                ([("EXTNAME", "ESTIMATED-BG")], bg),
                ([("EXTNAME", "LEVEL2-REMOVED")], cube2),
                ([("EXTNAME", "BIAS-MASK")], bias_mask),
                ([("EXTNAME", "LEVEL3-REMOVED")], cube3)]
    hdul = obsset.get_hdul_to_write(*hdu_list)
    obsset.store(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"], hdul, cache_only=True)


def get_per_amp_stat(cube, namp=32, threshold=100):
    r = {}

    ds = cube.reshape((namp, -1))

    msk_100 = np.abs(ds) > threshold

    r["count_gt_threshold"] = np.sum(msk_100, axis=1)

    r["stddev_lt_threshold"] = [np.std(ds1[~msk1])
                                for ds1, msk1 in zip(ds, msk_100)]

    return r


def estimate_amp_wise_noise(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    dl = []

    obsids = obsset.get_obsids()

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data
        for obsid, c in zip(obsids, cube):
            qq = get_per_amp_stat(c)

            ka = dict(obsid=obsid, level=k)

            _ = [dict(amp=i,
                      stddev_lt_threshold=q1,
                      count_gt_threshold=q2, **ka)
                 for i, (q1, q2) in enumerate(zip(qq["stddev_lt_threshold"],
                                                  qq["count_gt_threshold"]))]

            dl.extend(_)

    obsset.store(DESCS["RO_PATTERN_SUB_STAT_JSON"],
                 dict(stat=dl), cache_only=False)


def analyze_amp_wise_fft(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    from .ro_pattern_fft import (get_amp_wise_rfft)

    hdul_amp = []

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data

        qq_amp = [get_amp_wise_rfft(c) for c in cube]
        hdul_amp.append(([("EXTNAME", k)], np.array([np.abs(qq_amp),
                                                     np.angle(qq_amp)])))

        # qq0_c64 = get_c64_wise_noise_spectrum(np.array(cube))

    hdul = obsset.get_hdul_to_write(*hdul_amp)
    obsset.store(DESCS["RO_PATTERN_AMP_WISE_FFT_IMAGES"],
                 hdul, cache_only=False)


def analyze_c64_wise_fft(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    from .ro_pattern_fft import (get_c64_wise_noise_spectrum)

    hdul_amp = []

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data

        qq_amp = [get_c64_wise_noise_spectrum(c) for c in cube]
        hdul_amp.append(([("EXTNAME", k)], np.array([np.abs(qq_amp),
                                                     np.angle(qq_amp)])))

        # qq0_c64 = get_c64_wise_noise_spectrum(np.array(cube))

    hdul = obsset.get_hdul_to_write(*hdul_amp)
    obsset.store(DESCS["RO_PATTERN_C64_WISE_FFT_IMAGES"],
                 hdul, cache_only=False)


def generate_white_noise_image(obsset):
    # obsset.load(DESCS["PAIR_SUBTRACTED_IMAGES"])
    pass


# def plot_ap_per_amp(v, v2=None, ymargin=0):
#     amp = list(range(1, 33))
#     fig = AFigure(shape=(70, 20), margins=(1, ymargin), mod_mode="value")
#     _ = fig.plot(amp, v, marker="_o")
#     fig.xlim(0, 32)
#     return fig


def make_ap_badpix_count(cnt):

    # m, nn = asciiplot_per_amp(v, height=8, mmin=0, mmax=8)
    m1, nn = asciiplot_per_amp(cnt, height=8, xfactor=1)

    ss10 = np.take([" ", markers["o"], "*"], m1)
    ss11, sl = pad_with_axes(ss10)
    ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    ss14, sl = pad_title(ss13, sl, "Badpixel count per amp")

    S = to_string(ss14)

    return S


def make_ap_v1_v2(v1, v2):

    mmin = min(min(v1), min(v2))
    mmax = max(max(v1), max(v2))
    # m, nn = asciiplot_per_amp(v, height=8, mmin=0, mmax=8)
    m1, nn = asciiplot_per_amp(v1, height=8, xfactor=1,
                               mmin=mmin, mmax=mmax)
    m2, nn = asciiplot_per_amp(v2, height=8, xfactor=1,
                               mmin=mmin, mmax=mmax)

    ss10 = np.take([" ", markers["o"], "*"], m1)
    ss11, sl = pad_with_axes(ss10)
    ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    ss14, sl = pad_title(ss13, sl, "noise per amp: Raw")

    ss20 = np.take([" ", markers["o"], "*"], m2)
    ss21, sl = pad_with_axes(ss20)
    ss23, sl = pad_xaxis_label(ss21, sl, "1", "32")
    ss24, sl = pad_title(ss23, sl, "Reduced: level2")
    # ss2, sl = pad_yaxis_label(ss1, sl, nn[0], nn[-1])

    # S = "\n".join(["".join(sl) for sl in [::-1]])
    S = to_string(np.hstack([ss14, ss24]))

    return S


def print_out_stat_summary(obsset):

    dl = obsset.load(DESCS["RO_PATTERN_SUB_STAT_JSON"])

    df = pd.DataFrame(dl["stat"])
    g = df.groupby(['level', 'amp']).mean()[["count_gt_threshold",
                                             "stddev_lt_threshold"]]

    l1 = g.loc['DIRTY']
    cnt = l1["count_gt_threshold"].values
    std_dirty = l1["stddev_lt_threshold"].values

    l2 = g.loc['LEVEL2-REMOVED']
    std_lvl2 = l2["stddev_lt_threshold"].values

    S = make_ap_badpix_count(cnt)
    print()
    print(S)

    S = make_ap_v1_v2(std_dirty, std_lvl2)
    print()
    print(S)


def test():
    from igrins import get_obsset, DESCS

    # obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))
    obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))

    dl = obsset.load(DESCS["RO_PATTERN_SUB_STAT_JSON"])

    df = pd.DataFrame(dl["stat"])
    ss = df.groupby(['level', 'amp']).mean()
    g = ss[["count_gt_threshold", "stddev_lt_threshold"]]

    l1 = g.loc['DIRTY']
    cnt = l1["count_gt_threshold"].values
    v1 = l1["stddev_lt_threshold"].values

    l2 = g.loc['LEVEL2-REMOVED']
    v2 = l2["stddev_lt_threshold"].values

    S = make_ap_badpix_count(cnt)
    print()
    print(S)

    S = make_ap_v1_v2(v1, v2)
    print()
    print(S)


    # mmin = min(min(v1), min(v2))
    # mmax = max(max(v1), max(v2))
    # # m, nn = asciiplot_per_amp(v, height=8, mmin=0, mmax=8)
    # m1, nn = asciiplot_per_amp(v1, height=8, xfactor=1,
    #                            mmin=mmin, mmax=mmax)
    # m2, nn = asciiplot_per_amp(v2, height=8, xfactor=1,
    #                            mmin=mmin, mmax=mmax)

    # ss10 = np.take([" ", markers["o"], "*"], m1)
    # ss11, sl = pad_with_axes(ss10)
    # ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    # ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    # ss14, sl = pad_title(ss13, sl, "noise per amp: Raw")

    # ss20 = np.take([" ", markers["o"], "*"], m2)
    # ss21, sl = pad_with_axes(ss20)
    # ss23, sl = pad_xaxis_label(ss21, sl, "1", "32")
    # ss24, sl = pad_title(ss23, sl, "Reduced")
    # # ss2, sl = pad_yaxis_label(ss1, sl, nn[0], nn[-1])

    # S = "\n".join(["".join(sl) for sl in np.hstack([ss14, ss24])[::-1]])

    # fig = plot_ap_per_amp(g["count_gt_threshold"].values, ymargin=200)
    # fig.ylim(0, 3000)

if __name__ == '__main__':
    test()
