
# refactored from "correct_distortion.py"

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d


def _get_max_height(bottom_up_solutions, xx):
    """
    maximum height is measured by sampleing at xx positions.
    """
    max_height = 0

    for c in bottom_up_solutions:
        bottom = Polynomial(c[0][1])(xx)
        up = Polynomial(c[1][1])(xx)

        _height = up - bottom
        max_height = max(int(np.ceil(max(_height))), max_height)

    return max_height


def get_shifted(cleaned_data, bottom_up_solutions, normalize=False, height=0):

    acc_data = np.add.accumulate(cleaned_data, axis=0)
    ny, nx = acc_data.shape
    yy = np.arange(ny)
    xx = np.arange(0, nx)

    # Do not use assume_sorted, it results in incorrect interpolation.
    d0_acc_interp = [interp1d(yy, dd,
                              bounds_error=False)
                     for dd in acc_data.T]

    bottom_up_list = []

    if height == 0:
        height = _get_max_height(bottom_up_solutions, xx)

    d_factor = 1./height

    for c in bottom_up_solutions:
        bottom = Polynomial(c[0][1])(xx)
        up = Polynomial(c[1][1])(xx)
        dh = (up - bottom) * d_factor  # * 0.02

        bottom_up = zip(bottom - dh, up)
        bottom_up_list.append(bottom_up)

    d0_shft_list = []

    # for c in cent["bottom_up_solutions"]:
    for bottom_up in bottom_up_list:

        yy_list = [np.linspace(y1, y2, height+1)
                   for (y1, y2) in bottom_up]
        d0_acc_shft = np.array([intp(yy) for yy, intp
                                in zip(yy_list, d0_acc_interp)]).T

        # d0_shft = np.empty_like(d0_acc_shft)
        d0_shft = d0_acc_shft[1:, :]-d0_acc_shft[:-1, :]
        if normalize:
            d0_shft = d0_shft/[yy[1]-yy[0] for yy in yy_list]
        d0_shft_list.append(d0_shft)

    return d0_shft_list


def prepare_data_for_interp(data, order_map):
    # order_map is used to null the inter-order area
    msk = (order_map > 0) & np.isfinite(data)
    data[~msk] = 0.

    data[~np.isfinite(data)] = 0.

    return data, msk


def get_rectified_2dspec(data, order_map, bottom_up_solutions,
                         conserve_flux=False, height=0):

    # sl = slice(0, 2048), slice(0, 2048)
    data = data.copy()
    # resume from context does not work unless copying the data

    cleaned_data, msk = prepare_data_for_interp(data, order_map)

    d0_shft_list = get_shifted(cleaned_data, bottom_up_solutions, height=height)
    msk_shft_list = get_shifted(msk, bottom_up_solutions,
                                normalize=conserve_flux, height=height)

    return d0_shft_list, msk_shft_list


def main():
    import igrins
    from igrins import DESCS

    recipe_log = igrins.load_recipe_log("20170215")
    obsset = igrins.get_obsset("20170215", "H", recipe_log.iloc[13])

    hdu_debug = obsset.load(DESCS["DEBUG_IMAGE"], postfix="_simple")
    data = hdu_debug[0].data
    synth = hdu_debug[1].data

    bottom_up_solutions_ = obsset.load_resource_for("aperture_definition")
    bottom_up_solutions = bottom_up_solutions_["bottom_up_solutions"]

    hdu_ordermap = obsset.load_resource_sci_hdu_for("ordermap")
    ordermap = hdu_ordermap.data

    k = get_rectified_2dspec(data - synth, ordermap, bottom_up_solutions,
                             conserve_flux=True, height=0)


if __name__ == '__main__':
    main()
