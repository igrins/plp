import pandas as pd

from ..pipeline.steps import Step

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures.sky_spec import _get_combined_image, _destripe_sky
from ..procedures.target_spec import get_variance_map

from ..procedures.target_spec import extract_extended_spec
from ..procedures.ref_lines_db import SkyLinesDB, HitranSkyLinesDB

from ..procedures.process_identify_multiline import identify_lines_from_spec

from ..procedures.process_derive_wvlsol import fit_wvlsol, _convert2wvlsol


def make_combined_image_sky_old(obsset):
    # print("RECIPENAME", obsset.recipe_name)
    do_ab = obsset.recipe_name.endswith("AB")

    if do_ab:
        obsset_a = obsset.get_subset("A")
        obsset_b = obsset.get_subset("B")

        a = _get_combined_image(obsset_a)
        b = _get_combined_image(obsset_b)

        sky_data = a + b - abs(a - b)
        sky_plus = 2 * (a + b)  # assume that variance is the twice of a + b
    else:
        sky_data = _get_combined_image(obsset)
        sky_plus = sky_data

    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    sky_data = _destripe_sky(sky_data, destripe_mask, subtract_bg=False)

    variance_map0, variance_map = get_variance_map(obsset,
                                                   sky_data, sky_plus)

    hdul = obsset.get_hdul_to_write(([], sky_data),
                                    ([], variance_map0),
                                    ([], variance_map))
    obsset.store("combined_image1", data=hdul)


# from igrins.procedures.sky_spec import get_combined_image, _destripe_sky

from igrins.procedures.sky_spec import _make_combined_image_sky

# def get_exptime(obsset):
#     if obsset.recipe_entry is not None and "exptime" in obsset.recipe_entry:
#         exptime = obsset.recipe_entry["exptime"]
#     else:
#         exptime = float(obsset.get_hdus()[0].header["exptime"])

#     return exptime


# def _sky_subtract_bg(obsset, sky_image,
#                      bg_subtraction_mode="flat"):

#     sky_exptime = get_exptime(obsset)

#     if bg_usbtraction_mode == "flat":

#         bg_hdu = obsset.load_resource_sci_hdu_for(("flat_off",
#                                                    DESCS["FLAT_OFF_BG"]))
#         bg_exptime = float(bg_hdu.header["exptime"])
#     else:
#         raise ValueError("unknown bg_subtraction_mode: {}".
#                          format(bg_subtraction_mode))

#     sky_image2 = sky_image - bg_hdu.data / bg_exptime * sky_exptime

#     # subtract pattern noise

#     helper = ResourceHelper(obsset)
#     destripe_mask = helper.get("destripe_mask")

#     import igrins.procedures.readout_pattern as rp

#     pipe = [
#         rp.PatternP64ColWise,
#         rp.PatternAmpP2,
#         rp.PatternRowWiseBias
#     ]

#     destriped_sky = rp.apply(sky_image2, pipe, mask=destripe_mask)

#     return destriped_sky


# def _make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
#     sky_image = get_combined_image(obsset)

#     final_sky = _sky_subtract_bg(obsset, sky_image,
#                                  bg_subtraction_mode=bg_subtraction_mode)

#     return final_sky


def make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
    final_sky = _make_combined_image_sky(obsset, bg_subtraction_mode)

    # assume data_plus is equal to sky_data
    variance_map0, variance_map = get_variance_map(obsset,
                                                   final_sky, final_sky)

    hdul = obsset.get_hdul_to_write(([], final_sky),
                                    ([], variance_map0),
                                    ([], variance_map))
    obsset.store("combined_image1", data=hdul)


def estimate_slit_profile(obsset):
    from ..procedures.slit_profile import estimate_slit_profile_uniform
    estimate_slit_profile_uniform(obsset, do_ab=False)


def set_basename_postfix(obsset):
    # This only applies for the output name
    obsset.set_basename_postfix(basename_postfix="_sky")


def identify_sky_lines(obsset):

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]
    wvlsol = wvlsol_v0["wvl_sol"]

    #
    # from collections import namedtuple
    # Spec = namedtuple("Spec", ["s_map", "wvl_map"])

    # ref_lines_db = SkyLinesDB(config=obsset.get_config())
    ref_lines_db = SkyLinesDB(obsset.rs.master_ref_loader)

    if obsset.rs.get_resource_spec()[1] == "K":
        ref_lines_db_hitrans = HitranSkyLinesDB(obsset.rs.master_ref_loader)
    else:
        ref_lines_db_hitrans = None

    hdu = obsset.load_fits_sci_hdu("SPEC_FITS", postfix="_sky")

    fitted_pixels = identify_lines_from_spec(orders, hdu.data, wvlsol,
                                             ref_lines_db,
                                             ref_lines_db_hitrans)

    # storing multi-index seems broken. Enforce reindexing.
    _d = fitted_pixels.reset_index().to_dict(orient="split")
    obsset.store("SKY_IDENTIFIED_JSON", _d)


def derive_wvlsol(obsset):

    d = obsset.load("SKY_IDENTIFIED_JSON", postfix="_sky")
    df = pd.DataFrame(**d)

    p, fit_results = fit_wvlsol(df)

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    wvl_sol = _convert2wvlsol(p, orders)
    d = dict(orders=orders,
             wvl_sol=wvl_sol)

    obsset.store("SKY_WVLSOL_JSON", d)

    fit_results["orders"] = orders
    obsset.store("SKY_WVLSOL_FIT_RESULT_JSON",
                 fit_results)


steps = [Step("Set basename_postfix", set_basename_postfix),
         Step("Make Combined Sky", make_combined_image_sky),
         Step("Estimate slit profile (uniform)", estimate_slit_profile),
         Step("Extract spectra (for extendeded)",
              extract_extended_spec),
         Step("Identify sky lines",
              identify_sky_lines),
         Step("Derive wvlsol", derive_wvlsol),
         # Step("Generate Rectified 2d-spec", store_2dspec),
]
