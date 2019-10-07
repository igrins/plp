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

from .. import get_obsset_helper


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


def generate_initial_sky(obsset, destripe_mask):
    from ..procedures.sky_spec import get_median_combined_image
    from ..procedures.readout_pattern_helper import (sub_bg_from_slice,
                                                     apply_rp_1st_phase)
    from ..procedures.procedures_flat import get_params

    obsdate, band = obsset.get_resource_spec()

    mode, bg_y_slice = get_params(band)

    # median-combined image
    raw_sky = get_median_combined_image(obsset)

    sky0 = remove_pattern_from_guard(raw_sky)
    sky1 = sub_bg_from_slice(sky0, bg_y_slice)

    # This seem to work, but we may better refine the mask for the column-wise
    # background subtraction.
    if mode == 1:
        # if destripe_mask is None:
        #     helper = get_obsset_helper(obsset)
        #     destripe_mask = helper.get("destripe_mask")

        _d = apply_rp_1st_phase(sky1, destripe_mask)
    else:
        _d = sky1

    sky2 = sub_bg_from_slice(_d, bg_y_slice)

    return sky2


def remove_hotspot(obsset, sky):
    _, band = obsset.get_resource_spec()

    if band == "H":
        from ..utils.sub_hotspot import subtract_hotspot
        cx, cy = 163, 586
        sky = subtract_hotspot(sky, cx, cy, box_size=96)

    return sky


# def subtract_hotspot(obsset, band):
#     pass

def get_post_slit_bg_model(obsset, sky, destripe_mask):
    from ..procedures import destripe_dark_flatoff as dh

    # helper = get_obsset_helper(obsset)
    # destripe_mask = helper.get("destripe_mask")

    return dh.model_bg(sky, destripe_mask)


def make_combined_images(obsset,
                         bg_subtraction_mode="sky",
                         remove_level="auto",
                         # remove_amp_wise_var=False,
                         interactive=False,
                         cache_only=False):

    from functools import lru_cache

    from ..procedures.readout_pattern_helper import (apply_rp_1st_phase)

    helper = get_obsset_helper(obsset)
    destripe_mask = helper.get("destripe_mask")
    # badpix_mask = helper.get("badpix_mask")
    badpix_mask = obsset.load_resource_for("hotpix_mask")

    # # This seem to work, but we may better refine the mask for the column-wise
    # # background subtraction.
    # if mode == 1:
    #     _d = apply_rp_1st_phase(sky1, destripe_mask)
    # else:
    #     _d = sky1

    # sky2 = sub_bg_from_slice(_d, bg_y_slice)

    sky2 = generate_initial_sky(obsset, destripe_mask)
    sky2 = remove_hotspot(obsset, sky2)

    @lru_cache(maxsize=2)
    def _get_sky():
        return get_post_slit_bg_model(obsset, sky2, destripe_mask)
        # return dh.model_bg(sky2, destripe_mask)

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
        from ..procedures.sky_spec import get_exptime
        exptime = get_exptime(obsset)

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


def main_notebook():
    from igrins import get_obsset
    band = "K"
    # config_file = "../../recipe.config"
    config_file = None

    obsset = get_obsset("20190318", band, "SKY",
                        obsids=range(10, 11),
                        frametypes=["-"],
                        config_file=config_file)

    helper = get_obsset_helper(obsset)
    destripe_mask = helper.get("destripe_mask")
    # badpix_mask = obsset.load_resource_for("hotpix_mask")

    sky2 = generate_initial_sky(obsset, destripe_mask)
    sky2 = remove_hotspot(obsset, sky2)

    bg = get_post_slit_bg_model(obsset, sky2, destripe_mask)


def main_recipe():
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
    main_notebook()
