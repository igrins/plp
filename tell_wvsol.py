import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq
from functools import partial
from astropy.io import fits


def Poly(pars, middle, low, high, x):
    """
    Generates a polynomial with the given parameters
    for all of the x-values.
    x is assumed to be a np.ndarray!
     Not meant to be called directly by the user!
    """
    xgrid = (x - middle) / (high - low)  # re-scale
    return chebyshev.chebval(xgrid, pars)


    # ## -----------------------------------------------


def WavelengthErrorFunctionNew(pars, data, model, maxdiff=0.05):
    """
    Cost function for the new wavelength fitter.
    Not meant to be called directly by the user!
    """
    # xgrid = (data.x - np.median(data.x))/(data.x[-1] - data.x[0])
    # dx = chebyshev.chebval(xgrid, pars)
    dx = Poly(pars, np.median(data.x), min(data.x), max(data.x), data.x)
    penalty = np.sum(np.abs(dx[np.abs(dx) > maxdiff]))
    retval = (data.y / data.cont - model(data.x + dx)) + penalty
    return retval


    ### -----------------------------------------------


def FitWavelengthNew(data_original, telluric, fitorder=3, be_safe=True):
    """
    This is a vastly simplified version of FitWavelength.
    It takes the same inputs and returns the same thing,
    so is a drop-in replacement for the old FitWavelength.

    Instead of finding the lines, and generating a polynomial
    to apply to the axis as x --> f(x), it fits a polynomial
    to the delta-x. So, it fits the function for x --> x + f(x).
    This way, we can automatically penalize large deviations in
    the wavelength.
    """
    modelfcn = UnivariateSpline(telluric.x, telluric.y, s=0)
    pars = np.zeros(fitorder + 1)
    if be_safe:
        args = (data_original, modelfcn, 0.05)
    else:
        args = (data_original, modelfcn, 100)
    output = leastsq(self.WavelengthErrorFunctionNew, pars, args=args, full_output=True, xtol=1e-12, ftol=1e-12)
    pars = output[0]

    return partial(Poly, pars, np.median(data_original.x), min(data_original.x), max(data_original.x)), 0.0



def read_data(filename):
