import matplotlib.pyplot as plt

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq
from functools import partial
from astropy.io import fits
import readmultispec as multispec
from astropy import units as u
from scipy.signal import firwin, lfilter


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



def read_data(filename, debug=False):
    """
    This function reads in the given file. It uses Rick White's readmultispec script to get the wavelengths
    Output:
      a list of 2d numpy arrays holding the wavelengths in index 0, and the flux in index 1
      i.e. to get the wavelength and flux of order 10, do:
        wave, flux = orders[10][0], orders[10][1]
    """
    retdict = multispec.readmultispec(filename, quiet=not debug)

    # Check if wavelength units are in angstroms (common, but I like nm)
    hdulist = fits.open(filename)
    header = hdulist[0].header
    hdulist.close()
    wave_factor = 1.0  #factor to multiply wavelengths by to get them in nanometers
    for key in sorted(header.keys()):
        if "WAT1" in key:
            if "label=Wavelength" in header[key] and "units" in header[key]:
                waveunits = header[key].split("units=")[-1]
                if waveunits == "angstroms" or waveunits == "Angstroms":
                    #wave_factor = Units.nm/Units.angstrom
                    wave_factor = u.angstrom.to(u.nm)
                    if debug:
                        print "Wavelength units are Angstroms. Scaling wavelength by ", wave_factor

    # Compile all the orders
    numorders = retdict['flux'].shape[0]
    orders = []
    for i in range(numorders):
        wave = retdict['wavelen'][i] * wave_factor
        flux = retdict['flux'][i]
        orders.append(np.array((wave, flux)))
    return orders


def interpolate_telluric_model(filename):
    """
    This function reads in the telluric model and interpolates it for later use.
    """
    x, y = np.loadtxt(filename, unpack=True)
    return InterpolatedUnivariateSpline(x, y)


def remove_blaze(orders, telluric, N=200, freq=1e-2):
    """
    This function divides each order by the telluric model, and runs it through a high-pass filter to
    remove the blaze function. It is only approximate, so probably don't use for other things!
    #It also returns the original pixel indices for the eventual 2d fit...
    """
    half_window = N / 2
    filt = firwin(N, freq, window='hanning')
    new_orders = []
    original_pixels = []
    for i, o in enumerate(orders):
        idx = ~np.isnan(o[1])
        y = o[1][idx] / telluric(o[0][idx])

        # Extend the array to account for edge effects
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))

        # Filter the data
        filtered = lfilter(filt, 1.0, y)
        new_orders.append(np.array((o[0][idx][100:-100], (o[1][idx]/filtered[N:])[100:-100])))
        original_pixels.append(new_orders[-1].shape[0] + N + 100)
    return new_orders, original_pixels



if __name__ == '__main__':
    test_file = 'data/SDCH_20141014_0242.spec.fits'
    tell_file = 'data/TelluricModel.dat'

    orders = read_data(test_file, debug=True)
    tell_model = interpolate_telluric_model(tell_file)

    filtered_orders, original_pixels = remove_blaze(orders, tell_model)

    for order in filtered_orders:
        plt.plot(order[0], order[1], 'k-', alpha=0.4)
        plt.plot(order[0], tell_model(order[0]), 'r-', alpha=0.6)
    plt.show()