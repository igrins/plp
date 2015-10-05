import numpy as np


def badpixel_mask(d, sigma_clip1=100, sigma_clip2=10,
                  medfilter_size=None):

    """
    msk1 : sigma_clip1
    """
    import scipy.ndimage as ni

    d_std = d.std()
    d_std2 = d[np.abs(d)<d_std * 3].std()
    msk1_ = d > d_std2 * sigma_clip1

    msk1 = ni.convolve(msk1_, [[0,1,0],[1,1,1],[0,1,0]])

    if medfilter_size is not None:
        d_med = ni.median_filter(d, size=medfilter_size)
        d = d - d_med

    msk2 = np.abs(d) > d_std2 * sigma_clip2

    msk = msk1 | msk2

    return msk

def estimate_normalization(d, lower_limit, bpix_mask):
    """
    This is similar mode.
    """
    med = np.median(d[d>lower_limit])

    bins = np.linspace(0.5*med, med*10, 50)
    ss=np.histogram(d[(~bpix_mask) & (d>lower_limit)], bins=bins)[0]
    idx_max = np.argmax(ss[:-1] - ss[1:])
    m = bins[idx_max+1]

    return m

def estimate_normalization_percentile(d, lower_limit, bpix_mask,
                                      percentile=99.):
    m1 = d > lower_limit
    dd = d[(~bpix_mask) & m1]
    m = np.percentile(dd, percentile)

    return m

if __name__ == "__main__":
    def get_file(i):
        import libs.fits as pyfits
        f = pyfits.open("../20140526/SDCH_20140526_%04d.fits" % i)
        f[0].data -= np.median(f[0].data, axis=1)
        return f

    flat_offs = [get_file(i)[0].data for i in range(182, 192)]
    d = np.median(flat_offs, axis=0)

    msk = badpixel_mask(d, sigma_clip1=100)
