# This is to use new framework. Let's use this to measure flexure
# between emission spectra (e.g., sky, UNe, etc.)

#import os
import numpy as np
import pandas as pd

def save_qa(obsset):


    df = obsset.load_data_frame("SKY_FITTED_PIXELS_JSON", orient="split")

    msk_ = df["slit_center"] == 0.5
    dfm_ = df[msk_]

    msk = np.isfinite(dfm_["pixels"])
    dfm = dfm_[msk]

    lines_map = dict((o, (_["pixels"].values, _["wavelength"].values))
                     for o, _ in dfm.groupby("order"))

    from matplotlib.figure import Figure
    from igrins.libs.ecfit import check_fit

    d = obsset.load_item("SKY_WVLSOL_JSON")

    orders = d["orders"]

    fit_results = obsset.load_item("SKY_WVLSOL_FIT_RESULT_JSON")

    # fit_results = dict(xyz=[xl[msk], yl[msk], zl[msk]],
    #                    fit_params=fit_params,
    #                    fitted_model=poly_2d)

    # xl, yl, zl = get_ordered_line_data(reidentified_lines_map)

    xl, yl, zlo = fit_results["xyz"]
    xl, yl, zlo = [np.array(_) for _ in [xl, yl, zlo]]
    zl = zlo

    m = np.array(fit_results["fitted_mask"])

    lines_map_filtered = dict((o, (_["pixels"].values,
                                   _["wavelength"].values))
                              for o, _ in dfm[m].groupby("order"))

    modeul_name, class_name, serialized = fit_results["fitted_model"]
    from igrins.libs.astropy_poly_helper import deserialize_poly_model
   
    p = deserialize_poly_model(modeul_name, class_name, serialized)

    if 1:
        fig1 = Figure(figsize=(12, 7))
        check_fit(fig1, xl, yl, zl, p,
                  orders,
                  lines_map)
        fig1.tight_layout()

        fig2 = Figure(figsize=(12, 7))
        check_fit(fig2, xl[m], yl[m], zl[m], p,
                  orders,
                  lines_map_filtered)
        fig2.tight_layout()

    from igrins.libs.qa_helper import figlist_to_pngs
    dest_dir = obsset.query_item_path("qa_sky_fit2d_dir",
                                      subdir="sky_fit2d")
    figlist_to_pngs(dest_dir, [fig1, fig2])

    # sky_basename = helper.get_basename(band, obsids[0])
    # sky_figs = helper.get_section_filename_base("QA_PATH",
    #                                             "oh_fit2d",
    #                                             "oh_fit2d_"+sky_basename)
    # figlist_to_pngs(sky_figs, [fig1, fig2])


def save_distortion_db(obsset):

    db = obsset.load_db("distortion")
    db.update(obsset.band, obsset.basename)


def save_wvlsol_db(obsset):

    db = obsset.load_db("wvlsol")
    db.update(obsset.band, obsset.basename)


    # if 1:
    #     thar_db.update(band, thar_basename)


# 20151003 : Below is an attemp to modularize the recipes, which has
# not finished. Initial solution part is done, but the distortion part
# is not.




def save_ordermap_slitposmap(obsset):

    from aperture_helper import get_simple_aperture_from_obsset

    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]

    ap = get_simple_aperture_from_obsset(obsset,
                                         orders=orders)

    order_map = ap.make_order_map()
    slitpos_map = ap.make_slitpos_map()
    order_map2 = ap.make_order_map(mask_top_bottom=True)

    obsset.store_image("ordermap_fits", order_map)
    obsset.store_image("slitposmap_fits", slitpos_map)
    obsset.store_image("ordermap_masked_fits", order_map2)


def save_wavelength_map(obsset):

    fit_results = obsset.load_item("SKY_WVLSOL_FIT_RESULT_JSON")

    from igrins.libs.astropy_poly_helper import deserialize_poly_model

    module_name, klass_name, serialized = fit_results["fitted_model"]
    poly_2d = deserialize_poly_model(module_name, klass_name, serialized)

    order_map = obsset.load_item("ordermap_fits")[0].data
    # slitpos_map = caldb.load_item_from(basename, "slitposmap_fits")

    offset_map = obsset.load_item("slitoffset_fits")[0].data

    msk = order_map > 0

    _, pixels = np.indices(msk.shape)
    orders = order_map[msk]
    wvl = poly_2d(pixels[msk] - offset_map[msk], orders) / orders

    wvlmap = np.empty(msk.shape, dtype=float)
    wvlmap.fill(np.nan)

    wvlmap[msk] = wvl

    obsset.store_image("WAVELENGTHMAP_FITS", wvlmap)



from igrins.libs.recipe_helper import RecipeHelper

from process_wvlsol_v0 import extract_spectra_multi
from process_wvlsol_v0 import make_combined_image


def process_band(utdate, recipe_name, band, obsids, frametypes,
                 aux_infos, config_name):

    from igrins import get_caldb, get_obsset
    caldb = get_caldb(config_name, utdate)
    obsset = get_obsset(caldb, recipe_name, band, obsids, frametypes)

    # STEP 1 :
    ## make combined image

    make_combined_image(obsset)

    # Step 2

    extract_spectra_multi(obsset)

    from process_identify_multiline import identify_multiline

    identify_multiline(obsset)

    from process_wvlsol_volume_fit import volume_fit, generate_slitoffsetmap

    volume_fit(obsset)

    save_distortion_db(obsset)

    save_ordermap_slitposmap(obsset)

    generate_slitoffsetmap(obsset)

    from process_derive_wvlsol import derive_wvlsol
    derive_wvlsol(obsset)

    save_wvlsol_db(obsset)

    save_wavelength_map(obsset)

    from process_save_wat_header import save_wat_header
    save_wat_header(obsset)

    # save_wavelength_map(helper, band, obsids)
    # #fit_wvl_sol(helper, band, obsids)

    save_qa(obsset)

    # some of the fugures are missing.
    # save_figures()



from igrins.libs.recipe_factory import new_recipe_class, new_recipe_func

_recipe_class_wvlsol_sky = new_recipe_class("RecipeWvlsolSky",
                                            "SKY", process_band)

wvlsol_sky = new_recipe_func("wvlsol_sky",
                              _recipe_class_wvlsol_sky)

sky_wvlsol = new_recipe_func("sky_wvlsol",
                              _recipe_class_wvlsol_sky)

__all__ = wvlsol_sky, sky_wvlsol




# if 0:

#     # Step 3:

#     identify_lines(helper, band, obsids)

#     get_1d_wvlsol(helper, band, obsids)

#     save_1d_wvlsol(extractor,
#                    orders_w_solutions, wvl_sol, p)

#     save_qa(extractor, orders_w_solutions,
#             reidentified_lines_map, p, m)


#     save_figures(helper, band, obsids)

#     save_db(helper, band, obsids)



if __name__ == "__main__":

    utdate = "20140709"
    obsids = [62, 63]

    utdate = "20140525"
    obsids = [29]

    utdate = "20150525"
    obsids = [52]


    recipe_name = "SKY"


    # utdate = "20150525"
    # obsids = [32]

    # recipe_name = "THAR"

    band = "K"

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    process_band(utdate, recipe_name, band, obsids, config_name)
