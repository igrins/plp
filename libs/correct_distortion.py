import numpy as np
import libs.fits as pyfits
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial

class ShiftX(object):
    AXIS = 1

    def __init__(self, shiftx_map):
        self.shiftx_map = shiftx_map

        iy0, ix0 = np.indices(shiftx_map.shape)

        self.ix = ix0 - shiftx_map
        self.ix0 = ix0[0]

    def __call__(self, d0):

        d0_acc = np.add.accumulate(d0, axis=self.AXIS)

        # Do not use assume_sorted, it results in incorrect interpolation.
        d0_acc_shft = np.array([interp1d(xx, dd,
                                         bounds_error=False,
                                         )(self.ix0) \
                                for xx, dd in zip(self.ix, d0_acc)])

        d0_shft = np.empty_like(d0_acc_shft)
        d0_shft[:,1:] = d0_acc_shft[:,1:]-d0_acc_shft[:,:-1]

        return d0_shft


def get_flattened_2dspec(data, order_map, bottom_up_solutions):

    #sl = slice(0, 2048), slice(0, 2048)

    msk = (order_map > 0) & np.isfinite(data)
    data[~msk] = 0.

    data[~np.isfinite(data)] = 0.


    from scipy.interpolate import interp1d

    def get_shifted(data):

        acc_data = np.add.accumulate(data, axis=0)
        ny, nx = acc_data.shape
        yy = np.arange(ny)
        xx = np.arange(0, nx)

        # Do not use assume_sorted, it results in incorrect interpolation.
        d0_acc_interp = [interp1d(yy, dd,
                                  bounds_error=False) \
                         for dd in acc_data.T]

        bottom_up_list = []
        max_height = 0
        for c in bottom_up_solutions:
            bottom = Polynomial(c[0][1])(xx)
            up = Polynomial(c[1][1])(xx)
            dh = (up - bottom) * 0.05

            bottom_up = zip(bottom - dh, up + dh)
            bottom_up_list.append(bottom_up)

            height = up - bottom
            max_height = max(int(np.ceil(max(height))), max_height)

        d0_shft_list = []

        #for c in cent["bottom_up_solutions"]:
        for bottom_up in bottom_up_list:
            #p_bottom = Polynomial(c[0][1])
            #p_up = Polynomial(c[1][1])

            #height = p_up(xx) - p_bottom(xx)
            #bottom_up = zip(p_bottom(xx), p_up(xx))

            #max_height = int(np.ceil(max(height)))

            d0_acc_shft = np.array([intp(np.linspace(y1, y2, max_height+1)) \
                                    for (y1, y2), intp in zip(bottom_up, d0_acc_interp)]).T


            #d0_shft = np.empty_like(d0_acc_shft)
            d0_shft = d0_acc_shft[1:,:]-d0_acc_shft[:-1,:]
            d0_shft_list.append(d0_shft)

        return d0_shft_list


    d0_shft_list = get_shifted(data)
    msk_shft_list = get_shifted(msk)

    return d0_shft_list, msk_shft_list

if __name__ == "__main__":
    d = pyfits.open("../outdata/20140525/SDCH_20140525_0016.combined_image.fits")[0].data

    msk = np.isfinite(pyfits.open("../outdata/20140525/SDCH_20140525_0042.combined_image.fits")[0].data)

    d[~msk] = np.nan

    slitoffset = pyfits.open("../calib/primary/20140525/SKY_SDCH_20140525_0029.slitoffset_map.fits")[0].data

    d[~np.isfinite(slitoffset)] = np.nan


    # now shift
    msk = np.isfinite(d)
    d0 = d.copy()
    d0[~msk] = 0.

    shiftx = ShiftX(slitoffset)

    d0_shft = shiftx(d0)
    msk_shft = shiftx(msk)

    variance = d0
    variance_shft = shiftx(variance)

    d0_flux = d0_shft / msk_shft
