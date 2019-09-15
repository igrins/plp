from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import numpy as np


def get_smoothed(d2, threshold=100):
    d2 = np.ma.array(d2, mask=d2 > threshold).filled(np.nan)
    kernel = Gaussian2DKernel(x_stddev=1)
    dg = convolve(d2, kernel)

    return dg


def get_smoothed_w_mask(d2, x_stddev=0.3, mask=None):
    if mask is not None:
        d2 = np.ma.array(d2, mask=mask).filled(np.nan)
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    dg = convolve(d2, kernel)

    return dg
