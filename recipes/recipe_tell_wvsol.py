from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from numpy.polynomial import chebyshev
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import leastsq
from functools import partial
import libs.fits as fits
import libs.readmultispec as multispec
from astropy import units as u
from astropy.modeling import models, fitting
from scipy.signal import firwin, lfilter, argrelmin
import libs.recipes as recipes
from libs.ecfit import fit_2dspec
import os, sys


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
    if len(data[0]) < 100:
        return data
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

  return np.array(lines, dtype="i")


# pixels, order_nums, modelfcn = original_pixels, order_numbers, tell_model

def fit_chip(original_orders, corrected_orders, pixels, order_nums, modelfcn, ax3d=None):
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

    p, m = fit_2dspec(pixels, ordernums, wavelengths*ordernums, x_degree=4, y_degree=3)

    pred = p(pixels, ordernums) / ordernums

    # fig3d = plt.figure(2)
    # ax3d = fig3d.add_subplot(111, projection='3d')

    if ax3d is not None:
        ax3d.scatter3D(pixels[m], ordernums[m], wavelengths[m] - pred[m],
                       'ro')
        ax3d.set_xlabel('Pixel')
        ax3d.set_ylabel('Echelle order')
        ax3d.set_zlabel(r'$\lambda$ - fit(pixel)')

    print('RMS Scatter = {} nm'.format(np.std(wavelengths[m] - pred[m])))
    # if plot_file is None:
    #     plt.show()
    # else:
    #     plt.savefig(plot_file)
    #     plt.clf()

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

def plot_spec(ax, x, y, style='k-', alpha=0.5):
    l, = ax.plot(x, y, style, alpha=alpha)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Flux')
    return l


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
    if N is not None:
        half_window = N / 2
        filt = firwin(N, freq, window='hanning')
    else:
        filt = None
    new_orders = []
    original_pixels = []
    for i, o in enumerate(orders):
        idx = (np.isfinite(o[1])) & (o[0] >= o[0][100]) & (o[0] <= o[0][-100])
        if filt is not None:
            y = o[1][idx] / telluric(o[0][idx])

            # Extend the array to account for edge effects
            firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
            lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
            y = np.concatenate((firstvals, y, lastvals))

            # Filter the data
            filtered = lfilter(filt, 1.0, y)
            y = o[1][idx]/filtered[N:]
        else:
            y = o[1][idx]
        new_orders.append(np.array((o[0][idx], y)))
        original_pixels.append(np.array([i for i, good in enumerate(idx) if good]))
    return new_orders, original_pixels

tell_model_dict = {}
def get_tell_model(tell_file):
    import os
    tell_file = os.path.realpath(os.path.abspath(tell_file))

    tell_model = tell_model_dict.get(tell_file, None)

    if tell_file not in tell_model_dict:
        tell_model = interpolate_telluric_model(tell_file)
        tell_model_dict[tell_file] = tell_model

    return tell_model_dict[tell_file]


def run(filename, outfilename,
        plot_dir=None, tell_file='data/TelluricModel.dat',
        blaze_corrected=False):

    # prepare figures and axes
    if plot_dir is None: #interactive mode
        from matplotlib.pyplot import figure as Figure
    else:
        from matplotlib.figure import Figure

    fig1 = Figure(figsize=(12,6))
    ax1 = fig1.add_subplot(111)

    orders, order_numbers = read_data(filename, debug=False)

    tell_model = get_tell_model(tell_file)

    kwargs = {}
    if blaze_corrected:
        kwargs["N"] = None
    else:
        print 'Roughly removing blaze function for {}'.format(filename)


    filtered_orders, original_pixels = remove_blaze(orders, tell_model,
                                                    **kwargs)

    corrected_orders = []
    print 'Optimizing wavelength for order ',
    for o_n, order in zip(order_numbers, filtered_orders):

        l_orig = plot_spec(ax1, order[0], order[1],
                           'k-', alpha=0.4)
        l_model = plot_spec(ax1, order[0], tell_model(order[0]),
                            'r-', alpha=0.6)

        # Use the wavelength fit function to fit the wavelength.
        print ' {}'.format(o_n),
        sys.stdout.flush()

        new_order = optimize_wavelength(order, tell_model,
                                        fitorder=2)
        l_modified = plot_spec(ax1, new_order[0], new_order[1],
                               'g-', alpha=0.4)
        corrected_orders.append(new_order)

        # do not trt to plot if number valid points is less than 2
        if len(order[0]) < 2:
            continue

        ax1.legend([l_model, l_orig, l_modified],
                   ["Telluric Model",
                    "Original Spec.",
                    "Modified Spec."])

        if plot_dir is not None:
            ax1.set_title('Individual order fit : {}'.format(o_n))
            ax1.set_xlim(order[0][0], order[0][-1])
            ax1.set_ylim(-0.05, 1.15)

            figout = os.path.join(plot_dir, 'individual_order')
            postfix="_%03d" % (o_n,)
            from libs.qa_helper import fig_to_png
            fig_to_png(figout, fig1, postfix=postfix)

            # fig1.savefig('{}{}-individual_order{}.png'.format(plot_dir, filename.split('/')[-1], i+1))
            ax1.cla()

    print

    original_orders = [o.copy() for o in orders]

    # Now, fit the entire chip to a surface



    fig3d = Figure()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax3d = fig3d.add_subplot(111, projection='3d')

    final_orders = fit_chip(original_orders,
                            corrected_orders,
                            original_pixels,
                            order_numbers,
                            tell_model,
                            ax3d=ax3d)

    if plot_dir is not None:
        figout = os.path.join(plot_dir, 'fullchip_fit')
        from libs.qa_helper import fig_to_png
        fig_to_png(figout, fig3d)

    fig3 = Figure(figsize=(12,6))
    ax3 = fig3.add_subplot(111)

    # Filter again just for plotting
    final_filtered, _ = remove_blaze(final_orders, tell_model, **kwargs)
    for o_n, order in zip(order_numbers, final_filtered):
        l_final = plot_spec(ax3, order[0], order[1], 'k-', alpha=0.4)
        l_model = plot_spec(ax3, order[0], tell_model(order[0]),
                            'r-', alpha=0.6)

        if len(order[0]) < 2:
            continue

        ax3.legend([l_model, l_final],
                   ["Telluric Model",
                    "Final Spec."])


        if plot_dir is not None:
            ax3.set_title('Final wavelength solution : {}'.format(o_n))
            ax3.set_xlim(order[0][0], order[0][-1])
            ax3.set_ylim(-0.05, 1.15)

            figout = os.path.join(plot_dir, 'final_order')
            postfix="_%03d" % (o_n,)
            from libs.qa_helper import fig_to_png
            fig_to_png(figout, fig3, postfix=postfix)

            # fig1.savefig('{}{}-individual_order{}.png'.format(plot_dir, filename.split('/')[-1], i+1))
            ax3.cla()


    if plot_dir is None:
        ax3.set_title('Final wavelength solution')
        import matplotlib.pyplot as plt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.show()

    # Output
    wave_arr = np.array([o[0] for o in final_orders])
    hdulist = fits.PrimaryHDU(wave_arr)
    hdulist.writeto(outfilename, clobber=True)


def process_band(utdate, recipe_name, band, obsids, config,
                 interactive=True):

    # utdate, recipe_name, band, obsids, config = "20150525", "A0V", "H", [63, 64], "recipe.config"

    from libs.recipe_helper import RecipeHelper
    helper = RecipeHelper(config, utdate, recipe_name)
    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    desc = "SPEC_FITS_FLATTENED"
    blaze_corrected=True
    src_filename = caldb.query_item_path((band, master_obsid),
                                         desc)

    if not os.path.exists(src_filename):
        desc = "SPEC_FITS"
        blaze_corrected=False
        src_filename = caldb.query_item_path((band, master_obsid),
                                             desc)

    out_filename = caldb.query_item_path((band, master_obsid),
                                         "SPEC_FITS_WAVELENGTH")

    from libs.master_calib import get_ref_data_path
    tell_file = get_ref_data_path(helper.config, band,
                                  kind="TELL_WVLSOL_MODEL")

    if not interactive:
        tgt_basename = helper.get_basename(band, master_obsid)
        figout_dir = helper.igr_path.get_section_filename_base("QA_PATH",
                                                               "",
                                                               "tell_wvsol_"+tgt_basename)
        from libs.path_info import ensure_dir
        ensure_dir(figout_dir)
    else:
        figout_dir = None

    #print src_filename, out_filename, figout_dir, tell_file
    run(src_filename, out_filename,
        plot_dir=figout_dir, tell_file=tell_file,
        blaze_corrected=blaze_corrected)

# outfilename = "outdata/20150525/SDCK_20150525_0063.wave.fits"

#     run(filename, outfilename,
#         plot_dir=None, tell_file='data/TelluricModel.dat',
#         blaze_corrected=True)
#     pass


from libs.recipe_base import RecipeBase

class RecipeTellWvlsol(RecipeBase):

    def run_selected_bands_with_recipe(self, utdate, selected, bands):
        interactive = self.kwargs.get("interactive", True)

        for band in bands:
            for s in selected:
                recipe_name = s[0].strip()
                obsids = s[1]

                target_type = recipe_name.split("_")[0]

                if target_type not in ["A0V", "STELLAR", "EXTENDED"]:
                    print "Unsupported recipe : %s" % recipe_name
                    continue

                process_band(utdate, recipe_name, band, obsids, self.config,
                             interactive)
                #print (utdate, recipe_name, band, obsids, self.config)


def tell_wvsol(utdate, refdate=None, bands="HK",
               starting_obsids=None, interactive=False,
               recipe_name = "A0V*",
               config_file="recipe.config",
               ):

    recipe = RecipeTellWvlsol(interactive=interactive)
    recipe.set_recipe_name(recipe_name)
    recipe.process(utdate, bands,
                   starting_obsids, config_file)

def wvlsol_tell(utdate, refdate=None, bands="HK",
                starting_obsids=None, interactive=False,
                recipe_name = "A0V*",
                config_file="recipe.config",
                ):

    recipe = RecipeTellWvlsol(interactive=interactive)
    recipe.set_recipe_name(recipe_name)
    recipe.process(utdate, bands,
                   starting_obsids, config_file)


# if __name__ == "__main__":
#     utdate = "20150525"
#     bands="HK"

#     starting_obsids=None
#     interactive=False
#     recipe_name = "A0V-AB"
#     config_file="recipe.config"

#     recipe = RecipeTellWvlsol()
#     recipe.process(utdate, bands,
#                    starting_obsids, config_file)

# def tell_wvlsol(utdate, refdate="20140316", bands="HK",
#                 starting_obsids=None, interactive=False,
#                 recipe_name = "A0V",
#                 config_file="recipe.config",
#                 ):


#     from libs.igrins_config import IGRINSConfig
#     config = IGRINSConfig(config_file)

#     if not bands in ["H", "K", "HK"]:
#         raise ValueError("bands must be one of 'H', 'K' or 'HK'")

#     fn = config.get_value('RECIPE_LOG_PATH', utdate)
#     from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
#     recipe = Recipes(fn)

#     if starting_obsids is not None:
#         starting_obsids = map(int, starting_obsids.split(","))

#     # recipe_name = "ALL_RECIPES"
#     selected = recipe.select(recipe_name, starting_obsids)
#     if not selected:
#         print "no recipe of with matching arguments is found"

#     selected.sort()
#     for s in selected:
#         obsids = s[0]
#         frametypes = s[1]
#         recipe_name = s[2]["RECIPE"].strip()

#         target_type = recipe_name.split("_")[0]

#         if target_type not in ["A0V", "STELLAR", "EXTENDED"]:
#             print "Unsupported recipe : %s" % recipe_name
#             continue

#         for band in bands:
#             pass


# if __name__ == "__main__":
#     filename = "outdata/20150525/SDCK_20150525_0063.spec_flattened.fits"
#     outfilename = "outdata/20150525/SDCK_20150525_0063.wave.fits"

#     run(filename, outfilename,
#         plot_dir=None, tell_file='data/TelluricModel.dat',
#         blaze_corrected=True)
