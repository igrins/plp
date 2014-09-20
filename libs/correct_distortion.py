import numpy as np
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d

class ShiftX(object):
    AXIS = 1

    def __init__(self, shiftx_map):
        self.shiftx_map = shiftx_map

        iy0, ix0 = np.indices(shiftx_map.shape)

        self.ix = ix0 - shiftx_map
        self.ix0 = ix0[0]

    def __call__(self, d0):

        d0_acc = np.add.accumulate(d0, axis=self.AXIS)

        d0_acc_shft = np.array([interp1d(xx, dd,
                                         bounds_error=False,
                                         assume_sorted=False)(self.ix0) \
                                for xx, dd in zip(self.ix, d0_acc)])

        d0_shft = np.empty_like(d0_acc_shft)
        d0_shft[:,1:] = d0_acc_shft[:,1:]-d0_acc_shft[:,:-1]

        return d0_shft


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
