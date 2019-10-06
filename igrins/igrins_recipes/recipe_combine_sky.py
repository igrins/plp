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
from .gui_combine import (setup_gui_combine_sky,
                          factory_processed_n_smoothed)


# def _get_combined_image(obsset):
#     # Should not use median, Use sum.
#     data_list = [hdu.data for hdu in obsset.get_hdus()]

#     return np.sum(data_list, axis=0)


# def remove_pattern(data_minus, mask=None, remove_level=1,
#                    remove_amp_wise_var=True):

#     d1 = remove_pattern_from_guard(data_minus)

#     if remove_level == 2:
#         d2 = apply_rp_2nd_phase(d1, mask=mask)
#     elif remove_level == 3:
#         d2 = apply_rp_2nd_phase(d1, mask=mask)
#         d2 = apply_rp_3rd_phase(d2)
#     else:
#         d2 = d1

#     if remove_amp_wise_var:
#         c = get_amp_wise_rfft(d2)

#         ii = select_k_to_remove(c)
#         print(ii)
#         # ii = [9, 6]

#         new_shape = (32, 64, 2048)
#         mm = np.zeros(new_shape)

#         for i1 in ii:
#             mm1 = make_model_from_rfft(c, slice(i1, i1+1))
#             mm += mm1[:, np.newaxis, :]

#         ddm = mm.reshape((-1, 2048))

#         return d2 - ddm

#     else:
#         return d2


# def select_k_to_remove(c, n=2):
#     ca = np.abs(c)
#     # k = np.median(ca, axis=0)[1:]  # do no include the 1st column
#     k = np.percentile(ca, 95, axis=0)[1:]  # do no include the 1st column
#     # print(k[:10])
#     x = np.arange(1, 1 + len(k))
#     msk = (x < 5) | (15 < x)  # only select k from 5:15

#     # polyfit with 5:15 data
#     p = np.polyfit(np.log10(x[msk]), np.log10(k[msk]), 2,
#                    w=1./x[msk])
#     # p = np.polyfit(np.log10(x[msk][:30]), np.log10(k[msk][:30]), 2,
#     #                w=1./x[msk][:30])
#     # print(p)

#     # sigma from last 256 values
#     ss = np.std(np.log10(k[-256:]))

#     # model from p with 3 * ss
#     y = 10.**(np.polyval(p, np.log10(x)))

#     di = 5
#     dly = np.log10(k/y)[di:15]

#     # select first two values above 3 * ss
#     ii = np.argsort(dly)
#     yi = [di + i1 + 1for i1 in ii[::-1][:n] if dly[i1] > 3 * ss]

#     return yi


# def get_combined_images(obsset,
#                         allow_no_b_frame=False):

#     ab_mode = obsset.recipe_name.endswith("AB")

#     obsset_a = obsset.get_subset("A", "ON")
#     obsset_b = obsset.get_subset("B", "OFF")

#     na, nb = len(obsset_a.obsids), len(obsset_b.obsids)

#     if ab_mode and (na != nb):
#         raise RuntimeError("For AB nodding, number of A and B should match!")

#     if na == 0:
#         raise RuntimeError("No A Frame images are found")

#     if nb == 0 and not allow_no_b_frame:
#         raise RuntimeError("No B Frame images are found")

#     if nb == 0:
#         a_data = _get_combined_image(obsset_a)
#         data_minus = a_data

#     else:  # nb > 0
#         # a_b != 1 for the cases when len(a) != len(b)
#         a_b = float(na) / float(nb)

#         a_data = _get_combined_image(obsset_a)
#         b_data = _get_combined_image(obsset_b)

#         data_minus = a_data - a_b * b_data

#     if nb == 0:
#         data_plus = a_data
#     else:
#         data_plus = (a_data + (a_b**2)*b_data)

#     return data_minus, data_plus


# def get_variances(data_minus, data_plus, gain):

#     """
#     Return two variances.
#     1st is variance without poisson noise of source added. This was
#     intended to be used by adding the noise from simulated spectra.
#     2nd is the all variance.

#     """
#     from igrins.procedures.procedure_dark import get_per_amp_stat

#     guards = data_minus[:, [0, 1, 2, 3, -4, -3, -2, -1]]

#     qq = get_per_amp_stat(guards)

#     s = np.array(qq["stddev_lt_threshold"]) ** 2
#     variance_per_amp = np.repeat(s, 64*2048).reshape((-1, 2048))

#     variance = variance_per_amp + np.abs(data_plus)/gain

#     return variance_per_amp, variance


# def run_interactive(obsset,
#                     data_minus_raw, data_plus, bias_mask,
#                     remove_level, remove_amp_wise_var):
#     import matplotlib.pyplot as plt
#     # from astropy_smooth import get_smoothed
#     # from functools import lru_cache

#     get_im = factory_pattern_remove_n_smoothed(remove_pattern,
#                                                data_minus_raw,
#                                                bias_mask)

#     fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

#     vmin, vmax = -30, 30
#     # setup figure guis

#     obsdate, band = obsset.get_resource_spec()
#     obsid = obsset.master_obsid

#     status = dict(to_save=False)

#     def save(*kl, status=status):
#         status["to_save"] = True
#         plt.close(fig)
#         # print("save")
#         # pass

#     ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

#     # add callbacks
#     d2 = get_im(1, False, False)
#     im = ax.imshow(d2, origin="lower", interpolation="none")
#     im.set_clim(vmin, vmax)

#     box, get_params = setup_gui(im, vmin, vmax,
#                                 get_im, save)

#     plt.show()
#     params = get_params()
#     params.update(status)

#     return params

def run_interactive(obsset, params, _process, exptime=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    # vmin, vmax = -30, 30
    # setup figure guis

    obsdate, band = obsset.get_resource_spec()
    obsid = obsset.master_obsid

    status = dict(to_save=False)

    def save(*kl, status=status):
        status["to_save"] = True
        plt.close(fig)

    ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

    # add callbacks
    sky = _process(**params)

    im = ax.imshow(sky, origin="lower", interpolation="none")

    if exptime is None:
        vmin, vmax = -10, 30
    else:
        vmin, vmax = -1, 3  # for 30s exptime
        cor = max(int(exptime / 30.), 1)

        vmin, vmax = vmin * cor, vmax * cor

    im.set_clim(vmin, vmax)

    box, get_params = setup_gui_combine_sky(im, vmin, vmax,
                                            params,
                                            _process, save)

    plt.show()
    params = get_params()
    params.update(status)

    return params


def make_combined_images(obsset,
                         bg_subtraction_mode="sky",
                         remove_level="auto",
                         # remove_amp_wise_var=False,
                         interactive=False,
                         cache_only=False):

    from functools import lru_cache

    from ..procedures.readout_pattern_helper import (sub_bg_from_slice,
                                                     apply_rp_1st_phase)
    from ..procedures.procedures_flat import get_params
    from ..procedures.sky_spec import get_median_combined_image_n_exptime
    from .. import get_obsset_helper
    from ..procedures import destripe_dark_flatoff as dh

    obsdate, band = obsset.get_resource_spec()
    mode, bg_y_slice = get_params(band)

    # median-combined image
    raw_sky, exptime = get_median_combined_image_n_exptime(obsset)

    sky0 = remove_pattern_from_guard(raw_sky)
    sky1 = sub_bg_from_slice(sky0, bg_y_slice)

    helper = get_obsset_helper(obsset)
    destripe_mask = helper.get("destripe_mask")
    # badpix_mask = helper.get("badpix_mask")
    badpix_mask = obsset.load_resource_for("hotpix_mask")

    # This seem to work, but we may better refine the mask for the column-wise
    # background subtraction.
    if mode == 1:
        _d = apply_rp_1st_phase(sky1, destripe_mask)
    else:
        _d = sky1

    sky2 = sub_bg_from_slice(_d, bg_y_slice)

    if band == "H":
        from ..utils.sub_hotspot import subtract_hotspot
        cx, cy = 163, 586
        sky2 = subtract_hotspot(sky2, cx, cy, box_size=96)

    @lru_cache(maxsize=2)
    def _get_sky():
        return dh.model_bg(sky2, destripe_mask)

    def _process_sky(bg_subtraction_mode=bg_subtraction_mode,
                     remove_level=remove_level, fill_badpix_mask=True):

        if bg_subtraction_mode == "sky":
            bg_model = _get_sky()

            sky3 = apply_rp_1st_phase(sky2 - bg_model, destripe_mask)
        elif bg_subtraction_mode == "none":
            sky3 = sky2
        else:
            raise ValueError("Unknown bg-subtraction-mode : {}"
                             .format(bg_subtraction_mode))

        # print("remove level", remove_level)
        if remove_level == 2:
            # we may try even the 2nd phase?
            sky4 = apply_rp_2nd_phase(sky3, destripe_mask)
        else:
            sky4 = sky3

        if fill_badpix_mask:
            sky4 = np.ma.array(sky4, mask=badpix_mask).filled(np.nan)

        return sky4

    if remove_level == "auto":
        if bg_subtraction_mode == "sky":
            remove_level = 2
        else:
            remove_level = 1

    params = dict()
    params["bg_subtraction_mode"] = bg_subtraction_mode
    params["remove_level"] = remove_level

    if interactive:
        params = run_interactive(obsset, params, _process_sky,
                                 exptime=exptime)

        to_save = params.pop("to_save", False)
        if not to_save:
            print("canceled")
            return

    sky4 = _process_sky(fill_badpix_mask=False, **params)

    from astropy.io.fits import Card
    cards = []
    fits_cards = [Card(k, v) for (k, v, c) in cards]
    obsset.extend_cards(fits_cards)

    hdul = obsset.get_hdul_to_write(([], sky4))
    obsset.store("combined_sky", data=hdul)


steps = [Step("Make Combined Image", make_combined_images,
              interactive=False,
              bg_subtraction_mode="none",
              remove_level="auto")]


def main():
    from igrins import get_obsset
    band = "K"
    # config_file = "../../recipe.config"
    config_file = None

    obsset = get_obsset("20190318", band, "SKY",
                        obsids=range(10, 11),
                        frametypes=["-"],
                        config_file=config_file)

    make_combined_images(obsset, bg_subtraction_mode="none",
                         interactive=True)


if __name__ == '__main__':
    main()
