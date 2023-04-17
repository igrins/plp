import json
import pandas as pd

from ..pipeline.steps import Step

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures.sky_spec import _get_combined_image, _destripe_sky
from ..procedures.target_spec import get_variance_map

from ..procedures.target_spec import extract_extended_spec1
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
import numpy as np
import warnings

from ..procedures.sky_spec import _sky_subtract_bg_list, _get_combined_image, _destripe_sky

from ..procedures.destripe_helper import sub_p64_from_guard, sub_bg64_from_guard
from .make_header_table import get_header_tables_as_hdu

# Initial skys : only use guard for readout pattern removal. No bg subtraction.
# In phase II, we combine these initial skys to make pattern-minimized sky,
# and subtract it to re-analyze the readout pattern.

def _make_sky_ab_pair(obsset):

    of_list = list(zip(obsset.obsids, obsset.frametypes))
    obsid_to_load = set()
    obsid_pairs = []

    for (ao, af), (bo, bf) in zip(of_list[:-1], of_list[1:]):
        if af+bf in ["AB", "BA"]:
            # print(af + bf)
            obsid_to_load.update([ao, bo])
            obsid_pairs.append([ao, bo])

    obsid_list = sorted(obsid_to_load)
    hdu_list = obsset.get_hdus(obsid_list)

    data_list_ = [hdu.data for hdu in hdu_list]

    data_list_ = [sub_bg64_from_guard(sub_p64_from_guard(d))
                  for d in data_list_]

    data_list_ = [d - np.nanmedian(d) for d in data_list_]
    data_dict = dict(zip(obsid_list, data_list_))

    sky_list = []
    for ao, bo in obsid_pairs:
        a, b = data_dict[ao], data_dict[bo]
        sky = .5 * (a+b - abs(a-b))

        sky_list.append(sky)

    aux_columns = dict(OBSID=obsid_list,
                       # OBSDATE=[obsdate] * len(obsid_list),
                       FRAMETYPE=obsset.frametypes)

    tbl_hdu = get_header_tables_as_hdu(hdu_list,
                                       aux_columns=aux_columns)

    mode_dict = dict(mode="ab-pair", pairs=obsid_pairs)

    return sky_list, tbl_hdu, mode_dict


def _make_sky_off_only(obsset_orig, frmtype="B"):

    if frmtype:
        obsset = obsset_orig.get_subset(frmtype)
    else:
        obsset = obsset_orig

    hdu_list = obsset.get_hdus()
    data_list_ = [hdu.data for hdu in hdu_list]

    data_list = [sub_bg64_from_guard(sub_p64_from_guard(d))
                 for d in data_list_]

    obsid_list = list(map(int, obsset.get_obsids()))
    aux_columns = dict(OBSID=obsid_list,
                       # OBSDATE=[obsdate] * len(obsid_list),
                       FRAMETYPE=obsset.frametypes)

    tbl_hdu = get_header_tables_as_hdu(hdu_list,
                                       aux_columns=aux_columns)

    mode_dict = dict(mode="off-only", pairs=[(b,) for b in obsid_list])
    return data_list, tbl_hdu, mode_dict


def _make_sky_list(obsset, mode=None):
    if mode is None:
        if obsset.recipe_name.endswith("_AB"):
            mode = "ab-pair"
        else:
            mode = "off-only"

    assert mode in ["ab-pair", "off-only"]

    if mode == "off-only":
        data_list, tbl_hdu, mode_dict = _make_sky_off_only(obsset, frmtype="")
    else:
        data_list, tbl_hdu, mode_dict = _make_sky_ab_pair(obsset)

    return data_list, tbl_hdu, mode_dict


def _get_sky_images_deprecated(obsset):

    if obsset.recipe_name.endswith("AB"):  # do A-B
        obsset_a = obsset.get_subset("A")
        obsset_b = obsset.get_subset("B")

        a = _get_combined_image(obsset_a)
        b = _get_combined_image(obsset_b)

        a = sub_bg64_from_guard(sub_p64_from_guard(a))
        b = sub_bg64_from_guard(sub_p64_from_guard(b))

        a = a - np.nanmedian(a)
        b = b - np.nanmedian(b)

        # sky_data = .5 * abs(a-b)
        sky_data = .5 * (a+b - abs(a-b))
        combine_mode = "median_sky"
        combine_par = json.dumps(dict(A=obsset_a.obsids,
                                      B=obsset_b.obsids))

def make_initial_sky_cube(obsset):

    # hdu_list = obsset.get_hdus()
    # data_list = [hdu.data for hdu in hdu_list]

    # obsid_list = map(int, obsset.get_obsids())
    # aux_columns = dict(OBSID=obsid_list,
    #                    # OBSDATE=[obsdate] * len(obsid_list),
    #                    FRAMETYPE=obsset.frametypes)

    # tbl_hdu = get_header_tables_as_hdu(hdu_list,
    #                                    aux_columns=aux_columns)

    # # data_list = [sub_bg64_from_guard(sub_p64_from_guard(d)) for d in data_list]

    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    #     skys = _sky_subtract_bg_list(obsset, data_list,
    #                                  bg_subtraction_mode=bg_subtraction_mode,
    #                                  destripe_mode=destripe_mode)

    skys, tbl_hdu, mode_dict = _make_sky_list(obsset)

    final_sky = np.array(skys)
    # final_sky = np.array(data_list)
    hdul = obsset.get_hdul_to_write(([("SKY_MODE", json.dumps(mode_dict))],
                                     final_sky))
    hdul.append(tbl_hdu)
    obsset.store("combined_image1", data=hdul)


def make_sky_cube(obsset, bg_subtraction_mode=None, destripe_mode="guard"):

    hdu_list = obsset.get_hdus()
    data_list = [hdu.data for hdu in hdu_list]

    obsid_list = map(int, obsset.get_obsids())
    aux_columns = dict(OBSID=obsid_list,
                       # OBSDATE=[obsdate] * len(obsid_list),
                       FRAMETYPE=obsset.frametypes)

    tbl_hdu = get_header_tables_as_hdu(hdu_list,
                                       aux_columns=aux_columns)

    # data_list = [sub_bg64_from_guard(sub_p64_from_guard(d)) for d in data_list]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        skys = _sky_subtract_bg_list(obsset, data_list,
                                     bg_subtraction_mode=bg_subtraction_mode,
                                     destripe_mode=destripe_mode)

    final_sky = np.array(skys)
    # final_sky = np.array(data_list)
    hdul = obsset.get_hdul_to_write(([], final_sky))
    hdul.append(tbl_hdu)
    obsset.store("combined_image1", data=hdul)


if False:

    sky_image, cards = get_combined_image(obsset)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        final_sky = _sky_subtract_bg(obsset, sky_image,
                                     bg_subtraction_mode=bg_subtraction_mode)

    final_sky, cards = _make_combined_image_sky(obsset,
                                                bg_subtraction_mode)
    ncombine = dict((k, v) for (k, v, c) in cards)["NCOMBINE"]
    scaled_final_sky = ncombine * final_sky

    # assume data_plus is equal to sky_data
    variance_map0, variance_map = get_variance_map(obsset,
                                                   scaled_final_sky,
                                                   scaled_final_sky)
    from astropy.io.fits import Card
    fits_cards = [Card(k, v) for (k, v, c) in cards]
    obsset.extend_cards(fits_cards)

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


def extract_extended_spec(obsset, lacosmic_thresh=0.):

    # refactored from recipe_extract.ProcessABBABand.process

    from ..utils.load_fits import get_science_hdus
    postfix = obsset.basename_postfix
    hdul = get_science_hdus(obsset.load("COMBINED_IMAGE1",
                                        postfix=postfix))
    cube = hdul[0].data

    variance_shape = cube.shape[-2:]
    variance_map = np.ones(variance_shape, dtype=cube.dtype)
    variance_map0 = variance_map

    ss_list = []
    for data in cube:
        _ = extract_extended_spec1(obsset, data,
                                   variance_map, variance_map0,
                                   lacosmic_thresh=lacosmic_thresh)

        s_list, v_list, cr_mask, aux_images = _

        ss = np.array(s_list)

        ss_list.append(ss)

    basename_postfix = obsset.basename_postfix

    from ..procedures.target_spec import get_wvl_header_data
    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    hdul = obsset.get_hdul_to_write(([], np.array(ss_list)))
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify('fix')

    obsset.store("SPEC_FITS", hdul,
                 postfix=basename_postfix)


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
         Step("Make Sky Cube", make_initial_sky_cube),
         # Step("Estimate slit profile (uniform)", estimate_slit_profile),
         # Step("Extract spectra (for extendeded)",
         #      extract_extended_spec, lacosmic_thresh=0),

         # Step("Identify sky lines",
         #      identify_sky_lines),
         # Step("Derive wvlsol", derive_wvlsol),
         # Step("Generate Rectified 2d-spec", store_2dspec),
]
