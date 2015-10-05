import numpy as np

def get_variance_map0(a_minus_b, bias_mask2, pix_mask):
    #variance0 = a_minus_b
    #a_minus_b = a-b
    msk = bias_mask2 | pix_mask | ~np.isfinite(a_minus_b)

    from destriper import destriper
    variance0 = destriper.get_destriped(a_minus_b,
                                        msk,
                                        pattern=64,
                                        remove_vertical=False,
                                        hori=False)
    #variance0 = a_minus_b

    # stsci_median cannot be used due to too many array error.
    #ss = stsci_median([m1 for m1 in variance0],)
    dd1 = np.ma.array(variance0, mask=msk)
    ss = np.ma.median(dd1, axis=0)

    variance_ = variance0.copy()
    variance_[msk] = np.nan

    st = np.nanstd(variance_)
    st = np.nanstd(variance_[np.abs(variance_) < 3*st])

    variance_[np.abs(variance_-ss) > 3*st] = np.nan

    import scipy.ndimage as ni
    x_std = ni.median_filter(np.nanstd(variance_, axis=0), 11)

    variance_map0 = np.zeros_like(variance_) + x_std**2

    return variance_map0

def get_variance_map(a_plus_b,
                     a_minus_b,
                     bias_mask2=None, pix_mask=None, gain=None):
    """
    if bias_mask2 is None, consider a_minus_b as a variance_map0
    """
    if gain is None:
        raise ValueError("gain must be not None")

    if bias_mask2 is not None:
        variance_map0 = get_variance_map0(a_minus_b,
                                          bias_mask2, pix_mask)
    else:
        variance_map0 = a_minus_b

    # add poison noise in ADU
    variance_map = variance_map0 + np.abs(a_plus_b)/gain

    return variance_map


def get_variance_map2(a_plus_b, a_minus_b, bias_mask2, pix_mask, gain):
    #variance0 = a_minus_b
    #a_minus_b = a-b
    msk = bias_mask2 | pix_mask | ~np.isfinite(a_minus_b)

    from destriper import destriper
    variance0 = destriper.get_destriped(a_minus_b,
                                        msk,
                                        pattern=64,
                                        remove_vertical=False,
                                        hori=False)
    #variance0 = a_minus_b

    # stsci_median cannot be used due to too many array error.
    #ss = stsci_median([m1 for m1 in variance0],)
    dd1 = np.ma.array(variance0, mask=msk)
    ss = np.ma.median(dd1, axis=0)

    variance_ = variance0.copy()
    variance_[msk] = np.nan

    st = np.nanstd(variance_)
    st = np.nanstd(variance_[np.abs(variance_) < 3*st])

    variance_[np.abs(variance_-ss) > 3*st] = np.nan

    import scipy.ndimage as ni
    x_std = ni.median_filter(np.nanstd(variance_, axis=0), 11)

    variance_map0 = np.zeros_like(variance_) + x_std**2

    variance_map = variance_map0 + np.abs(a_plus_b)/gain # add poison noise in ADU
    return variance_map


if __name__ == "__main__":
    import libs.fits as pyfits
    a = pyfits.open("../indata/20140525/SDCH_20140525_0016.fits")[0].data
    b = pyfits.open("../indata/20140525/SDCH_20140525_0017.fits")[0].data

    flat_mask = pyfits.open("../calib/primary/20140525/FLAT_SDCH_20140525_0074.flat_mask.fits")[0].data > 0
    order_map2 = pyfits.open("../calib/primary/20140525/SKY_SDCH_20140525_0029.order_map_masked.fits")[0].data

    bias_mask2 = flat_mask & (order_map2 > 0)

    pix_mask0 = pyfits.open("../calib/primary/20140525/FLAT_SDCH_20140525_0074.flat_bpixed.fits")[0].data
    pix_mask = ~np.isfinite(pix_mask0)


    v = get_variance_map(a+b, a-b, bias_mask2, pix_mask, gain=2)
