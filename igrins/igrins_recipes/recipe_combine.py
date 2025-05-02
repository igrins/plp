import numpy as np
import copy
# import scipy.ndimage as ni

# from ..utils.image_combine import image_median

from .. import get_obsset_helper
from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..igrins_libs.cosmics import cosmicsimage

from ..pipeline.steps import Step

from ..procedures.readout_pattern_guard import remove_pattern_from_guard
from ..procedures.procedure_dark import (apply_rp_2nd_phase,
                                         apply_rp_3rd_phase)

from ..procedures.ro_pattern_fft import (get_amp_wise_rfft,
                                         make_model_from_rfft)
# from ..procedures.ro_pattern_fft import (get_amp_wise_rfft,
#                                          make_model_from_rfft)
from .gui_combine import setup_gui, factory_pattern_remove_n_smoothed

from ..procedures.sky_spec import get_exptime
from ..procedures.procedures_flexure_correction import estimate_flexure, estimate_flexure_short_exposures, check_telluric_shift

import astroscrappy
from scipy.ndimage import median_filter, binary_dilation

from ..procedures.destriper import destriper




def _get_combined_image(obsset):
    # Should not use median, Use sum.
    import_data_list = [hdu.data for hdu in obsset.get_hdus()]
    data_list = [] #Put the data list in a form that can be modified (not read only)

    for import_data_list_frame in import_data_list:
        data_list.append(np.array(import_data_list_frame.data))



    #New scheme for cosmic ray masking
    #Kyle Kaplan Dec 7, 2023, updated May 2, 2025 to add additional filter
    mask_cosmics = obsset.get_recipe_parameter("mask_cosmics")
    if mask_cosmics == True:
        # cosmics_sigmaclip = 15.0 #Set universal cosmic ray correction parameters
        # cosmics_sigfrac = 0.3
        # cosmcis_objlim = 5.0
        # satlevel = -1
        cosmics_sigmaclip = 1.7 #Set universal cosmic ray correction parameters
        cosmics_sigfrac = 13.0
        cosmcis_objlim = 4.0
        readnoise_multiplier = 2.25
        satlevel = -1
        h_band_gain = 2.5
        k_band_gain = 2.5
        h_band_read_noise = 5.2 #Note for fowler 16
        k_band_read_noise = 6.7
        n_frames = len(data_list) #Number of frames in nod
        cr_mask_count = np.zeros(np.shape(data_list[0][4:-4, 4:-4])) #Create an array to count how many times a pixel is masked for cosmics
        cr_masks = [] #Create a list of CR Masks
        date, band = obsset.get_resource_spec()
        if n_frames == 1: #Run only if n_frames is 1 (a single AB nod), this is rare but requires special treatment to interpolate over cosmics since we can't fill in masked cosmics with other frames
                data_without_overscan = data_list[0][4:-4, 4:-4] #Cut overscan
                if band == 'H':
                    cr_mask, cr_array = astroscrappy.detect_cosmics(data_without_overscan, gain=h_band_gain, readnoise=h_band_read_noise, sigclip = cosmics_sigmaclip, sigfrac = cosmics_sigfrac, objlim = cosmcis_objlim, niter=4, verbose=True, cleantype='medmask') # Build the object for H-band
                else: #if band == 'K'
                    cr_mask, cr_array  = astroscrappy.detect_cosmics(data_without_overscan, gain=k_band_gain, readnoise=k_band_read_noise, sigclip = cosmics_sigmaclip, sigfrac = cosmics_sigfrac, objlim = cosmcis_objlim, niter=4, verbose=True, cleantype='medmask') # Build the object for K-band
                data_list[0][4:-4, 4:-4] = cr_array #Interpolate over any cosmics found
        else: #If n_frames > 1, do the normal routine and use pixels frames without cosmics to fill in masked cosmics
            cleaned_data_list = []
            for i in range(n_frames):
                data_without_overscan = data_list[i][4:-4, 4:-4] #Cut overscan
                if band == 'H':
                    cr_mask_astroscrappy, cr_array  = astroscrappy.detect_cosmics(data_without_overscan, gain=h_band_gain, readnoise=h_band_read_noise, sigclip = cosmics_sigmaclip, sigfrac = cosmics_sigfrac, objlim = cosmcis_objlim, niter=4, verbose=True, cleantype='medmask') # Build the object for H-band
                else: #if band == 'K'
                    cr_mask_astroscrappy, cr_array  = astroscrappy.detect_cosmics(data_without_overscan, gain=k_band_gain, readnoise=k_band_read_noise, sigclip = cosmics_sigmaclip, sigfrac = cosmics_sigfrac, objlim = cosmcis_objlim, niter=4, verbose=True, cleantype='medmask') # Build the object for K-band            
                filtered_data_1 = cr_array - median_filter(cr_array, [7,1]) #Apply an additional mask for CRs and electronic noise that might have been missed by astroscrappy
                filtered_data_2 = filtered_data_1 - median_filter(filtered_data_1, [1,11])
                cr_mask_median_filter = (np.abs(filtered_data_2) > 40.0) & ( np.abs(filtered_data_2 /cr_array) > 0.6)
                cr_mask = binary_dilation(cr_mask_astroscrappy, iterations=1) | cr_mask_median_filter #Combine old and new masks
                cr_masks.append(cr_mask)
                cleaned_data_list.append(median_filter(cr_array, [3,3])) #Generate smoothed data to fill in pixels that were masked
                cr_mask_count[cr_mask] += 1 #Increment mask count array for later scaling the non-masked pixels
            masked_pixels_unlikely_to_be_cosmics = cr_mask_count == n_frames
            cr_mask_count[masked_pixels_unlikely_to_be_cosmics] = 0 #Zero out pixels in the CR mask count unlikely to be cosmics
            for i in range(n_frames): #Scale each frame to "fill" holes left by cosmic rays using data from other frames
                data_list[i][4:-4, 4:-4][cr_masks[i]] = 0. #Zero out cosmic rays found
                data_list[i][4:-4, 4:-4] *= (n_frames / (n_frames - cr_mask_count)) #Scale up pixels in frames without cosmics but where cosmics are in other frames
                data_list[i][4:-4, 4:-4][masked_pixels_unlikely_to_be_cosmics] = cleaned_data_list[i][masked_pixels_unlikely_to_be_cosmics] #Fill in bad pixels


    # correct nonlinearity
    from ..procedures.nonlinearity import get_nonlinearity_corrector
    corrector = get_nonlinearity_corrector(obsset)

    data_list = [corrector(d) for d in data_list]


    correct_flexure = obsset.get_recipe_parameter("correct_flexure")
    if correct_flexure == True:        
        exptime = get_exptime(obsset)
        # if exptime >= 20.0:
        data_list = estimate_flexure(obsset, data_list, exptime) #Estimate flexure and apply correction
        if len(data_list) > 1: #Testing detection
            check_telluric_shift(obsset, data_list)


    return np.sum(data_list, axis=0)


def remove_pattern(data_minus, mask=None, remove_level=1,
                   remove_amp_wise_var=True):

    d1 = remove_pattern_from_guard(data_minus)

    if remove_level == 2:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
    elif remove_level == 3:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
        d2 = apply_rp_3rd_phase(d2)
    else:
        d2 = d1

    if remove_amp_wise_var:
        c = get_amp_wise_rfft(d2)

        ii = select_k_to_remove(c)
        print(ii)
        # ii = [9, 6]

        new_shape = (32, 64, 2048)
        mm = np.zeros(new_shape)

        for i1 in ii:
            mm1 = make_model_from_rfft(c, slice(i1, i1+1))
            mm += mm1[:, np.newaxis, :]

        ddm = mm.reshape((-1, 2048))

        return d2 - ddm

    else:
        return d2


def select_k_to_remove(c, n=2):
    ca = np.abs(c)
    # k = np.median(ca, axis=0)[1:]  # do no include the 1st column
    k = np.percentile(ca, 95, axis=0)[1:]  # do no include the 1st column
    # print(k[:10])
    x = np.arange(1, 1 + len(k))
    msk = (x < 5) | (15 < x)  # only select k from 5:15

    # polyfit with 5:15 data
    p = np.polyfit(np.log10(x[msk]), np.log10(k[msk]), 2,
                   w=1./x[msk])
    # p = np.polyfit(np.log10(x[msk][:30]), np.log10(k[msk][:30]), 2,
    #                w=1./x[msk][:30])
    # print(p)

    # sigma from last 256 values
    ss = np.std(np.log10(k[-256:]))

    # model from p with 3 * ss
    y = 10.**(np.polyval(p, np.log10(x)))

    di = 5
    dly = np.log10(k/y)[di:15]

    # select first two values above 3 * ss
    ii = np.argsort(dly)
    yi = [di + i1 + 1for i1 in ii[::-1][:n] if dly[i1] > 3 * ss]

    return yi


def get_combined_images(obsset,
                        allow_no_b_frame=True):

    ab_mode = obsset.recipe_name.endswith("AB")

    obsset_a = obsset.get_subset("A", "ON")
    obsset_b = obsset.get_subset("B", "OFF")

    na, nb = len(obsset_a.obsids), len(obsset_b.obsids)

    # if ab_mode and (na != nb):
    #     raise RuntimeError("For AB nodding, number of A and B should match!")

    if na == 0:
        raise RuntimeError("No A Frame images are found")

    if nb == 0 and not allow_no_b_frame:
        raise RuntimeError("No B Frame images are found")

    if nb == 0:
        a_data = _get_combined_image(obsset_a)
        data_minus = a_data

    else:  # nb > 0
        # a_b != 1 for the cases when len(a) != len(b)
        a_b = float(na) / float(nb)



        a_data = _get_combined_image(obsset_a)
        b_data = _get_combined_image(obsset_b)

        exptime = get_exptime(obsset_a)
        # correct_flexure = obsset.get_recipe_parameter("correct_flexure")
        # if correct_flexure == True and exptime < 20.0: #Flexure correct short exposures
        #    a_data, b_data = estimate_flexure_short_exposures(obsset, a_data, b_data, exptime)


        data_minus = a_data - a_b * b_data

    if nb == 0:
        data_plus = a_data
    else:
        data_plus = (a_data + (a_b**2)*b_data)

    return data_minus, data_plus

from ..procedures.procedure_dark import get_per_amp_stat

def get_variances(data_minus, data_plus, gain):

    """
    Return two variances.
    1st is variance without poisson noise of source added. This was
    intended to be used by adding the noise from simulated spectra.
    2nd is the all variance.

    """

    # guards = data_minus[:, [0, 1, 2, 3, -4, -3, -2, -1]]
    guards = data_minus[:, [0, 1, 2, 3]]

    # FIXME IGRINS2 reference pixels may have pattern
    guards = destriper.get_destriped(guards, pattern=64, hori=False, remove_vertical=False)

    qq = get_per_amp_stat(guards)

    s = np.array(qq["stddev_lt_threshold"]) ** 2
    variance_per_amp = np.repeat(s, 64*2048).reshape((-1, 2048))

    variance = variance_per_amp + np.abs(data_plus)/gain

    return variance_per_amp, variance


def run_interactive(obsset,
                    data_minus_raw, data_plus, bias_mask,
                    remove_level, remove_amp_wise_var):
    import matplotlib.pyplot as plt
    # from astropy_smooth import get_smoothed
    # from functools import lru_cache

    get_im = factory_pattern_remove_n_smoothed(remove_pattern,
                                               data_minus_raw,
                                               bias_mask)

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    vmin, vmax = -30, 30
    # setup figure guis

    obsdate, band = obsset.get_resource_spec()
    obsid = obsset.master_obsid

    status = dict(to_save=False)

    def save(*kl, status=status):
        status["to_save"] = True
        plt.close(fig)
        # print("save")
        # pass

    ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

    # add callbacks
    d2 = get_im(1, False, False)
    im = ax.imshow(d2, origin="lower", interpolation="none")
    im.set_clim(vmin, vmax)

    box, get_params = setup_gui(im, vmin, vmax,
                                get_im, save)

    plt.show()
    params = get_params()
    params.update(status)

    return params

import scipy.ndimage as NI
from scipy.interpolate import LSQUnivariateSpline

def _remove_vertical_pattern(d2, bias_mask, hotpix):

    mask = NI.binary_dilation(bias_mask, iterations=2)

    k = np.ma.array(d2, mask=mask|hotpix).filled(np.nan)

    x = np.arange(2048)
    t = np.arange(128, 2048 - 128, 256)

    z = np.zeros((2048, 2048), dtype=float)

    for i in range(4, 2048-4):
        y = k[:, i]
        m = np.isfinite(y)
        m[[0, 1, 2, 3, -4, -3, -2, -1]] = False

        s = NI.median_filter(y[m], 15)
        q = 4*np.median(np.abs(y[m] - s))

        m[m] = (y[m] - s) < q

        spl = LSQUnivariateSpline(x[m], y[m], t,
                                  bbox=[0, 2047],
                                  k=3, ext=0, check_finite=False)
        z[:, i] = spl(x)

    return d2 - z

def make_combined_images(obsset, allow_no_b_frame=False,
                         remove_level=2,
                         remove_amp_wise_var=False,
                         remove_vertical_pattern=False,
                         interactive=False,
                         cache_only=False):

    if remove_level == "auto":
        remove_level = 2

    if remove_amp_wise_var == "auto":
        remove_amp_wise_var = False

    _ = get_combined_images(obsset,
                            allow_no_b_frame=allow_no_b_frame)
    data_minus_raw, data_plus = _

    bias_mask = obsset.load_resource_for("bias_mask")

    d_hotpix = obsset.load_resource_for("hotpix_mask")
    hotpix = d_hotpix > 0

    if interactive:
        params = run_interactive(obsset,
                                 data_minus_raw, data_plus, bias_mask,
                                 remove_level, remove_amp_wise_var)

        print("returned", params)
        if not params["to_save"]:
            print("canceled")
            return

        remove_level = params["remove_level"]
        remove_amp_wise_var = params["amp_wise"]

    d2_ = remove_pattern(data_minus_raw, mask=bias_mask,
                         remove_level=remove_level,
                         remove_amp_wise_var=remove_amp_wise_var)

    if remove_vertical_pattern:
        print("removing vertical pattern")
        ordermask = obsset.load_resource_for("ordermap")[0].data > 0
        # This is to remove the region for optical ghost in H
        _, band = obsset.get_resource_spec()
        # FIXME this is a workaround for IG2 Hㅁ band
        if band == "H":
            ordermask[40:112+40, 1925:] |= ordermask[:112, 1925:]

        d2 = _remove_vertical_pattern(d2_, ordermask, hotpix)
    else:
        d2 = d2_

    dp = remove_pattern(data_plus, remove_level=1,
                        remove_amp_wise_var=False)

    # FIXME This is a workaround for the reference pixel issue in IGRINS2.

    obsdate, band = obsset.get_resource_spec()

    if band == "H":
        dp[4:-4, 4:-4] -= np.median(dp[:512, 4:16])
    elif band == "K":
        dp[4:-4, 4:-4] -= np.median(dp[1600:, -16:-4])
    else:
        raise ValueError()


    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")
    # d2 = destriper.get_destriped(data_minus_raw, mask=destripe_mask, pattern=128, hori=True)
    # dp = data_plus
    d2 = destriper.get_destriped(d2, mask=destripe_mask, pattern=64, hori=True, remove_vertical=False)

    gain = float(obsset.rs.query_ref_value("GAIN"))

    variance_map0, variance_map = get_variances(d2, dp, gain)

    hdul = obsset.get_hdul_to_write(([], d2))

    obsset.store("combined_image1", data=hdul, cache_only=cache_only)

    hdul = obsset.get_hdul_to_write(([], variance_map0))
    obsset.store("combined_variance0", data=hdul, cache_only=cache_only)

    hdul = obsset.get_hdul_to_write(([], variance_map))
    obsset.store("combined_variance1", data=hdul, cache_only=cache_only)


steps = [Step("Make Combined Image", make_combined_images,
              allow_no_b_frame=False,
              interactive=False,
              remove_level="auto", remove_amp_wise_var="auto")]

