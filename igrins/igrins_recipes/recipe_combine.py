import numpy as np
# import scipy.ndimage as ni

# from ..utils.image_combine import image_median
# from ..igrins_libs.resource_helper_igrins import ResourceHelper

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


def _get_combined_image(obsset):
    # Should not use median, Use sum.
    data_list = [hdu.data for hdu in obsset.get_hdus()]
    date, band = obsset.get_resource_spec()
    
    correct_flexure = obsset.get_recipe_parameter("correct_flexure")
    if correct_flexure == True:
        
        exptime = get_exptime(obsset)
        if exptime >= 20.0:
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
                        allow_no_b_frame=False):

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
        correct_flexure = obsset.get_recipe_parameter("correct_flexure")
        if correct_flexure == True and exptime < 20.0: #Flexure correct short exposures
           a_data, b_data = estimate_flexure_short_exposures(obsset, a_data, b_data, exptime)


        data_minus = a_data - a_b * b_data

    if nb == 0:
        data_plus = a_data
    else:
        data_plus = (a_data + (a_b**2)*b_data)

    return data_minus, data_plus


def get_variances(data_minus, data_plus, gain):

    """
    Return two variances.
    1st is variance without poisson noise of source added. This was
    intended to be used by adding the noise from simulated spectra.
    2nd is the all variance.

    """
    from igrins.procedures.procedure_dark import get_per_amp_stat

    guards = data_minus[:, [0, 1, 2, 3, -4, -3, -2, -1]]

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


def make_combined_images(obsset, allow_no_b_frame=False,
                         remove_level=2,
                         remove_amp_wise_var=False,
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

    d2 = remove_pattern(data_minus_raw, mask=bias_mask,
                        remove_level=remove_level,
                        remove_amp_wise_var=remove_amp_wise_var)

    dp = remove_pattern(data_plus, remove_level=1,
                        remove_amp_wise_var=False)

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

