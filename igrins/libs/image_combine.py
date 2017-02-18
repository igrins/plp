import astropy.io.fits as pyfits
from stsci_helper import stsci_median
import numpy as np

def make_combined_image_sky_deprecated(helper, band, obsids):
    """
    simple median combine with destripping. Suitable for sky.
    """
    filenames, basename, master_obsid = helper.get_base_info(band, obsids)

    hdu_list = [pyfits.open(fn)[0] for fn in filenames]
    _data = stsci_median([hdu.data for hdu in hdu_list])

    from get_destripe_mask import get_destripe_mask
    destripe_mask = get_destripe_mask(helper, band, obsids)

    data = destripe_sky(_data, destripe_mask)

    return data


def make_combined_image_sky_deprecated(helper, band, obsids, frametypes=None):

    from load_fits import get_hdus, get_combined_image
    hdus = get_hdus(helper, band, obsids)

    if frametypes is None: # do A-B
        sky_data_ = get_combined_image(hdus) / len(hdus)
    else:
        a_and_b = dict()
        for frame, hdu in zip(frametypes, hdus):
            a_and_b.setdefault(frame.upper(), []).append(hdu)

        a = get_combined_image(a_and_b["A"]) / len(a_and_b["A"])
        b = get_combined_image(a_and_b["B"]) / len(a_and_b["B"])

        sky_data_ = a+b - abs(a-b)
    
    from get_destripe_mask import get_destripe_mask
    destripe_mask = get_destripe_mask(helper, band, obsids)

    from image_combine import destripe_sky
    sky_data = destripe_sky(sky_data_, destripe_mask, subtract_bg=False)

    return sky_data

def make_combined_sky(hdus, frametypes=None):

    # from load_fits import get_hdus, get_combined_image
    # hdus = get_hdus(helper, band, obsids)

    from load_fits import get_combined_image

    if frametypes is None: # do A-B
        sky_data = get_combined_image(hdus) / len(hdus)
    else:
        a_and_b = dict()
        for frame, hdu in zip(frametypes, hdus):
            a_and_b.setdefault(frame.upper(), []).append(hdu)

        a = get_combined_image(a_and_b["A"]) / len(a_and_b["A"])
        b = get_combined_image(a_and_b["B"]) / len(a_and_b["B"])

        sky_data = a+b - abs(a-b)
    
    return sky_data

if 0:
    from get_destripe_mask import get_destripe_mask
    destripe_mask = get_destripe_mask(helper, band, obsids)

    from image_combine import destripe_sky
    sky_data = destripe_sky(sky_data_, destripe_mask, subtract_bg=False)

    return sky_data


def destripe_sky(data, destripe_mask, subtract_bg=True):
    """
    simple destripping. Suitable for sky.
    """

    from destriper import destriper
    from estimate_sky import estimate_background, get_interpolated_cubic

    if subtract_bg:
        xc, yc, v, std = estimate_background(data, destripe_mask,
                                             di=48, min_pixel=40)
        nx = ny = 2048
        ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)
        ZI3 = np.nan_to_num(ZI3)

        d = data - ZI3
    else:
        d= data

    mask = destripe_mask | ~np.isfinite(d)
    stripes = destriper.get_stripe_pattern64(d, mask=mask,
                                             concatenate=True,
                                             remove_vertical=False)

    return d - stripes
