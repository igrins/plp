import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq
from functools import partial
from astropy.io import fits
import libs.readmultispec as multispec
from astropy import units as u
from astropy.modeling import models, fitting
from scipy.signal import firwin, lfilter, argrelmin
import libs.recipes as recipes
import os


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


def wavelength_errorfcn(pars, data, model, maxdiff=0.05):
    """
    Cost function for the new wavelength fitter.
    Not meant to be called directly by the user!
    """
    dx = Poly(pars, np.median(data[0]), min(data[0]), max(data[0]), data[0])
    penalty = np.sum(np.abs(dx[np.abs(dx) > maxdiff]))
    retval = (data[1] - model(data[0] + dx)) + penalty
    return retval


    ### -----------------------------------------------


def fit_wavelength(data_original, modelfcn, fitorder=3, be_safe=True):
    """
    This code fits $\Delta \lambda$ as a function of pixel to a polynomial.
    It returns a version of data_original with an updated wavelength axis.
    """
    pars = np.zeros(fitorder + 1)
    if be_safe:
        args = (data_original, modelfcn, 0.05)
    else:
        args = (data_original, modelfcn, 100)
    output = leastsq(wavelength_errorfcn, pars, args=args, full_output=True, xtol=1e-12, ftol=1e-12)
    pars = output[0]

    dx = Poly(pars, np.median(data_original[0]), min(data_original[0]), max(data_original[0]), data_original[0])
    data_original[0] += dx
    return data_original



def optimize_wavelength(data_original, modelfcn, fitorder=3, be_safe=True, N=3):
    """
    Runs fit_wavelength N times.
    """
    data = data_original.copy()
    for i in range(N):
        data = fit_wavelength(data.copy(), modelfcn, fitorder=fitorder, be_safe=be_safe)
    return data



def find_lines(spectrum, tol=0.99, linespacing = 5):
  """
  Function to find the spectral lines, given a model spectrum
  spectrum:        A numpy array with the model spectrum - MUST be normalized
  tol:             The line strength needed to count the line
                      (0 is a strong line, 1 is weak)
  linespacing:     The minimum spacing (in pixels) between two consecutive lines.
                      find_lines will choose the strongest line if there are
                      several too close.
  """
  # Run argrelmin
  lines = list(argrelmin(spectrum[1], order=linespacing)[0])

  #Check for lines that are too weak.
  for i in range(len(lines)-1, -1, -1):
    idx = lines[i]
    xval = spectrum[0][idx]
    yval = spectrum[1][idx]
    if yval > tol:
      lines.pop(i)

  return np.array(lines)


def fit_chip(original_orders, corrected_orders, pixels, order_nums, modelfcn, plot_file=None):
    """
    Fit the entire chip to a 2D surface.
    :param original_orders: The original orders, before blaze correction or anything (they have all pixels)
    :param corrected_orders: The wavelength-corrected orders. They are blaze-corrected and have a smaller size
    :param pixels: The correspondence between the two order arrays.
                   If pixels[10][50] = 245, then pixel 50 of order 10 in corrected_orders corresponds to pixel 245
                   or order 10 in original_orders.
    :param order_nums: The echelle order number corresponding to each order.
    :param modelfcn: The interpolated telluric model
    """
    # Make 3 big arrays for pixel, order number, and wavelength
    # We will only use the pixels that have a telluric line (from the model)
    pixel_list = []
    ordernum_list = []
    wavelength_list = []
    for p, o, c in zip(pixels, order_nums, corrected_orders):
        # Find the pixel locations of the telluric lines
        modelspec = np.array((c[0], modelfcn(c[0])))
        lines = find_lines(modelspec)

        # Save the original pixels, order numbers, and wavelengths for the fitter.
        pixel_list.append(p[lines])
        ordernum_list.append(np.ones(lines.size)*o)
        wavelength_list.append(c[0][lines])
    pixels = np.hstack(pixel_list).astype(float)
    ordernums = np.hstack(ordernum_list).astype(float)
    wavelengths = np.hstack(wavelength_list).astype(float)
    weights = np.ones(pixels.size).astype(float)

    from libs.ecfit import fit_2dspec
    p, m = fit_2dspec(pixels, ordernums, wavelengths*ordernums, x_degree=3, y_degree=4)

    pred = p(pixels, ordernums) / ordernums
    fig3d = plt.figure(2)
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter3D(pixels, ordernums, wavelengths - pred, 'ro')

    print('RMS Scatter = {} nm'.format(np.std(wavelengths - pred)))
    if plot_file is None:
        plt.show()
    else:
        plt.savefig(plot_file)
        plt.clf()

    # Assign the wavelength solution to each order
    new_orders = []
    for i, order in enumerate(original_orders):
        pixels = np.arange(order.shape[1]).astype(float)
        ord = order_nums[i] * np.ones(pixels.size).astype(float)
        wave = p(pixels, ord) / ord
        order[0] = wave
        new_orders.append(order)
    return new_orders


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
    wavefields = retdict['wavefields']
    apertures = []
    for i in range(numorders):
        wave = retdict['wavelen'][i] * wave_factor
        flux = retdict['flux'][i]
        orders.append(np.array((wave, flux)))
        order_number = int(wavefields[i][1])
        apertures.append(order_number)

    # Finally, get the echelle order number for each aperture from the wavefields
    wavefields = retdict['wavefields']

    return orders, apertures


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
        idx = (~np.isnan(o[1])) & (o[0] >= o[0][100]) & (o[0] <= o[0][-100])
        y = o[1][idx] / telluric(o[0][idx])

        # Extend the array to account for edge effects
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))

        # Filter the data
        filtered = lfilter(filt, 1.0, y)
        new_orders.append(np.array((o[0][idx], (o[1][idx]/filtered[N:]))))
        original_pixels.append(np.array([i for i, good in enumerate(idx) if good]))
    return new_orders, original_pixels


"""
def fit_by_order(test_file):
    #test_file = 'data/SDCK_20141014_0242.spec.fits'
    tell_file = 'data/TelluricModel.dat'

    orders, order_numbers = read_data(test_file, debug=False)
    tell_model = interpolate_telluric_model(tell_file)

    filtered_orders, original_pixels = remove_blaze(orders, tell_model)

    corrected_orders = []
    for order in filtered_orders:
        plt.plot(order[0], order[1], 'k-', alpha=0.4)
        plt.plot(order[0], tell_model(order[0]), 'r-', alpha=0.6)

        # Use the wavelength fit function to fit the wavelength.
        new_order = optimize_wavelength(order.copy(), tell_model, fitorder=4)
        plt.plot(new_order[0], new_order[1], 'g-', alpha=0.4)
        corrected_orders.append(new_order)

    plt.show()

    return orders, corrected_orders, original_pixels, order_numbers, tell_model
"""

def run(filename, plot_dir=None, tell_file='data/TelluricModel.dat'):
    #orders, corrected_orders, original_pixels, order_numbers, tell_model = fit_by_order(filename)

    orders, order_numbers = read_data(filename, debug=False)
    tell_model = interpolate_telluric_model(tell_file)

    print 'Roughly removing blaze function for {}'.format(filename)
    filtered_orders, original_pixels = remove_blaze(orders, tell_model)

    corrected_orders = []
    for i, order in enumerate(filtered_orders):
        plt.plot(order[0], order[1], 'k-', alpha=0.4)
        plt.plot(order[0], tell_model(order[0]), 'r-', alpha=0.6)

        # Use the wavelength fit function to fit the wavelength.
        print 'Optimizing wavelength for order {}/{}'.format(i+1, len(filtered_orders))
        new_order = optimize_wavelength(order.copy(), tell_model, fitorder=4)
        plt.plot(new_order[0], new_order[1], 'g-', alpha=0.4)
        corrected_orders.append(new_order)

    if plot_dir is None:
        plt.show()
    else:
        plt.savefig('{}{}-individual_fit.png'.format(plot_dir, filename.split('/')[-1]))
        plt.clf()

    original_orders = [o.copy() for o in orders]

    # Now, fit the entire chip to a surface
    if plot_dir is not None:
        plot_file = '{}{}-fullchip_fit.png'.format(plot_dir, filename.split('/')[-1])
    else:
        plot_file = None

    final_orders = fit_chip(original_orders, 
                            corrected_orders, 
                            original_pixels, 
                            order_numbers, 
                            tell_model,
                            plot_file=plot_file)

    for i, o in enumerate(final_orders):
        plt.plot(o[0], o[1], 'k-', alpha=0.4)
        plt.plot(orders[i][0], orders[i][1], 'r--', alpha=0.5)

    # Filter again just for plotting
    final_filtered, _ = remove_blaze(final_orders, tell_model)
    for order in final_filtered:
        plt.plot(order[0], order[1], 'k-', alpha=0.4)
        plt.plot(order[0], tell_model(order[0]), 'r-', alpha=0.6)
    plt.title('Final wavelength solution')
    
    if plot_dir is None:
        plt.show()
    else:
        plt.savefig('{}{}-final_fit.png'.format(plot_dir, filename.split('/')[-1]))
        plt.clf()

    # Output
    outfilename = filename.replace('spec.fits', 'wave.fits')
    wave_arr = np.array([o[0] for o in final_orders])
    hdulist = fits.PrimaryHDU(wave_arr)
    hdulist.writeto(outfilename, clobber=True)



def tell_wvsol(utdate, refdate="20140316", bands="HK",
               starting_obsids=None,
               config_file="recipe.config"):
 
    # Get the directory the data is in
    datadir = 'outdata/'+utdate+'/'

    # Make a new directory in the qa subdirectory for my plots
    if not os.path.exists(datadir + 'qa/tell_wvsol'):
        os.makedirs(datadir + 'qa/tell_wvsol')

    # Read the recipe log to get the A0V standards
    night = recipes.Recipes('recipe_logs/'+utdate+'.recipes') #Load up table for a night
    framenos = [] #Set up list to store numbers for first frame of each A0V standadr star
    for found_a0v in night.recipe_dict['A0V_AB']: #Loop through dictionary7
        framenos.append( '%.4d' % found_a0v[0][0]) #Append the first frame found for each A0V star in a night

    # Get the list of bands
    if bands=="HK":
        bandlist = ["H", "K"]
    elif bands=="H":
        bandlist = ["H"]
    elif bands=="K":
        bandlist==["K"]
    else:
        raise ValueError

    # Process each file separately
    for band in bandlist:
        for frame in framenos:
            if starting_obsids is not None and frame not in starting_obsids:
                continue
            run(datadir+'SDC'+band+'_'+utdate+'_'+frame+'.spec.fits', plot_dir=datadir+'qa/tell_wvsol/')
