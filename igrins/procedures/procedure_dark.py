from __future__ import print_function

from collections import namedtuple

import numpy as np
import scipy.ndimage as ni

from astropy.io.fits import Card, HDUList, PrimaryHDU

from ..utils.image_combine import image_median as stsci_median
from ..procedures import destripe_dark_flatoff as dh

from .. import get_obsset_helper, DESCS

from ..utils.json_helper import json_dumps

from igrins.procedures.readout_pattern import (
    PatternAmpP2,
    PatternP64Zeroth,
    PatternP64First,
    PatternP64ColWise,
    PatternColWiseBias,
    PatternColWiseBiasC64,
    PatternRowWiseBias,
    PatternAmpWiseBiasC64
)

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


def generate_white_noise_image(obsset):
    # obsset.load(DESCS["PAIR_SUBTRACTED_IMAGES"])
    pass

def derive_pattern_noise(obsset):

    pipe = OrderedDict(amp_wise_bias_r2=PatternAmpP2,
                       p64_0th_order=PatternP64Zeroth,
                       col_wise_bias_c64=PatternColWiseBiasC64,
                       p64_1st_order=PatternP64First,
                       col_wise_bias=PatternColWiseBias,
                       p64_per_column=PatternP64PatternP64ColWise,
                       row_wise_bias=PatternRowWiseBias,
                       amp_wise_bias_c64=PatternAmpWiseBiasC64)

    hdu = obsset.load_fits_sci_hdu((DESCS["PAIR_SUBTRACTED_IMAGES"]))

    d = hdu.data[0]

    patterns = OrderedDict()
    for k, p in pipe.items():
        p = p(d)
        
    pass

def do_psd(obsset):
    pass

def estimate_std_per_amp(obsset):
    pass
