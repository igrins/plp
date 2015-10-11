import astropy.io.fits as pyfits
from stsci_helper import stsci_median
import numpy as np

def make_combined_image_sky(helper, band, obsids):
    """
    simple median combine with destripping. Suitable for sky.
    """
    filenames, basename, master_obsid = helper.get_base_info(band, obsids)

    hdu_list = [pyfits.open(fn)[0] for fn in filenames]
    _data = stsci_median([hdu.data for hdu in hdu_list])

    from get_destripe_mask import get_destripe_mask
    destripe_mask = get_destripe_mask(helper, band, obsids)

    from destriper import destriper
    from estimate_sky import estimate_background, get_interpolated_cubic

    xc, yc, v, std = estimate_background(_data, destripe_mask,
                                         di=48, min_pixel=40)
    nx = ny = 2048
    ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)
    ZI3 = np.nan_to_num(ZI3)

    d = _data - ZI3
    mask=destripe_mask | ~np.isfinite(d)
    stripes = destriper.get_stripe_pattern64(d, mask=mask,
                                             concatenate=True,
                                             remove_vertical=False)

    return d - stripes
