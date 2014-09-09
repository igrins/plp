import os
#import numpy as np


from libs.path_info import IGRINSPath, IGRINSFiles
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

from libs.recipe_base import RecipeBase

class RecipeThAr(RecipeBase):
    RECIPE_NAME = "THAR"

    def run_selected_bands(self, utdate, selected, bands):
        for s in selected:
            obsids = s[0]
            print obsids
            # frametypes = s[1]

            for band in bands:
                process_thar_band(utdate, self.refdate, band, obsids,
                                  self.config)

def thar(utdate, bands="HK",
         starting_obsids=None, config_file="recipe.config"):

    RecipeThAr()(utdate, bands,
                 starting_obsids, config_file)



# def thar(utdate, refdate="20140316", bands="HK",
#          starting_obsids=None):

#     if not bands in ["H", "K", "HK"]:
#         raise ValueError("bands must be one of 'H', 'K' or 'HK'")

#     fn = "%s.recipes" % utdate
#     from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
#     recipe = Recipes(fn)

#     if starting_obsids is not None:
#         starting_obsids = map(int, starting_obsids.split(","))

#     selected = recipe.select("THAR", starting_obsids)

#     for s in selected:
#         obsids = s[0]

#         for band in bands:
#             process_thar_band(utdate, refdate, band, obsids)


def process_thar_band(utdate, refdate, band, obsids, config):

    from libs.products import ProductDB, PipelineStorage

    igr_path = IGRINSPath(config, utdate)

    igr_storage = PipelineStorage(igr_path)

    thar_filenames = igr_path.get_filenames(band, obsids)

    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]


    thar_master_obsid = obsids[0]

    flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "flat_on.db",
                                                        )
    flaton_db = ProductDB(flaton_db_name)

    flaton_basename = flaton_db.query(band, thar_master_obsid)


    from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC

    desc_list = [FLATCENTROID_SOL_JSON_DESC]
    products = igr_storage.load(desc_list,
                                mastername=flaton_basename)

    aperture_solution_products = products[FLATCENTROID_SOL_JSON_DESC]

    # igrins_orders = {}
    # igrins_orders["H"] = range(99, 122)
    # igrins_orders["K"] = range(72, 92)

    if 1:
        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]

        orders = range(len(bottomup_solutions))

        ap =  Apertures(orders, bottomup_solutions)

    if 1:
        from libs.process_thar import ThAr

        thar = ThAr(thar_filenames)

        thar_products = thar.process_thar(ap)

    if 1: # match order
        from libs.process_thar import match_order_thar
        from libs.master_calib import load_thar_ref_data

        #ref_date = "20140316"

        thar_ref_data = load_thar_ref_data(refdate, band)

        new_orders = match_order_thar(thar_products, thar_ref_data)

        print thar_ref_data["orders"]
        print  new_orders

        ap =  Apertures(new_orders, bottomup_solutions)

        from libs.storage_descriptions import ONED_SPEC_JSON_DESC
        thar_products[ONED_SPEC_JSON_DESC]["orders"] = new_orders


    if 1:

        hdu = pyfits.open(thar_filenames[0])[0]
        igr_storage.store(thar_products,
                          mastername=thar_filenames[0],
                          masterhdu=hdu)

    if 1:
        # measure shift of thar lines from reference spectra

        # load spec

        from libs.process_thar import reidentify_ThAr_lines
        thar_reidentified_products = reidentify_ThAr_lines(thar_products,
                                                           thar_ref_data)

        igr_storage.store(thar_reidentified_products,
                          mastername=thar_filenames[0],
                          masterhdu=hdu)

    if 1:

        from libs.process_thar import (load_echelogram,
                                       align_echellogram_thar,
                                       check_thar_transorm,
                                       get_wavelength_solutions)

        ref_date = thar_ref_data["ref_date"]
        echel = load_echelogram(ref_date, band)

        thar_aligned_echell_products = \
             align_echellogram_thar(thar_reidentified_products,
                                    echel, band, ap)

        # We do not save this product yet.
        # igr_storage.store(thar_aligned_echell_products,
        #                   mastername=thar_filenames[0],
        #                   masterhdu=hdu)



        fig_list = check_thar_transorm(thar_products,
                                       thar_aligned_echell_products)

        from libs.qa_helper import figlist_to_pngs
        thar_figs = igr_path.get_section_filename_base("QA_PATH",
                                                       "thar",
                                                       "thar_"+thar_basename)
        figlist_to_pngs(thar_figs, fig_list)

        thar_wvl_sol = get_wavelength_solutions(thar_aligned_echell_products,
                                                echel)

        igr_storage.store(thar_wvl_sol,
                          mastername=thar_filenames[0],
                          masterhdu=hdu)

    if 1: # make amp and order falt

        from libs.storage_descriptions import ONED_SPEC_JSON_DESC

        orders = thar_products[ONED_SPEC_JSON_DESC]["orders"]
        order_map = ap.make_order_map()
        #slitpos_map = ap.make_slitpos_map()


        # load flat on products
        #flat_on_params_name = flaton_path.get_secondary_path("flat_on_params")

        #flaton_products = PipelineProducts.load(flat_on_params_name)
        from libs.storage_descriptions import (FLAT_NORMED_DESC,
                                               FLAT_MASK_DESC)

        flaton_products = igr_storage.load([FLAT_NORMED_DESC, FLAT_MASK_DESC],
                                           flaton_basename)

        from libs.process_flat import make_order_flat, check_order_flat
        order_flat_products = make_order_flat(flaton_products,
                                              orders, order_map)

        #fn = thar_path.get_secondary_path("orderflat")
        #order_flat_products.save(fn, masterhdu=hdu)

        igr_storage.store(order_flat_products,
                          mastername=flaton_basename,
                          masterhdu=hdu)

    if 1:
        fig_list = check_order_flat(order_flat_products)

        from libs.qa_helper import figlist_to_pngs
        orderflat_figs = igr_path.get_section_filename_base("QA_PATH",
                                                            "orderflat",
                                                            "orderflat_"+thar_basename)
        figlist_to_pngs(orderflat_figs, fig_list)

    if 1:
        from libs.products import ProductDB
        thar_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                          "thar.db",
                                                          )
        thar_db = ProductDB(thar_db_name)
        # os.path.join(igr_path.secondary_calib_path,
        #                                  "thar.db"))
        thar_db.update(band, thar_basename)


if __name__ == "__main__":
    import sys

    utdate = sys.argv[1]
    bands = "HK"
    starting_obsids = None

    if len(sys.argv) >= 3:
        bands = sys.argv[2]

    if len(sys.argv) >= 4:
        starting_obsids = sys.argv[3]

    thar(utdate, refdate="20140316", bands=bands,
         starting_obsids=starting_obsids)


# if __name__ == "__main__":

#     from libs.recipes import load_recipe_list, make_recipe_dict
#     from libs.products import PipelineProducts, ProductPath, ProductDB

#     if 0:
#         utdate = "20140316"
#         # log_today = dict(flat_off=range(2, 4),
#         #                  flat_on=range(4, 7),
#         #                  thar=range(1, 2))
#     elif 1:
#         utdate = "20140525"
#         # log_today = dict(flat_off=range(64, 74),
#         #                  flat_on=range(74, 84),
#         #                  thar=range(3, 8),
#         #                  sky=[29])

#     band = "K"

#     fn = "%s.recipes" % utdate
#     recipe_list = load_recipe_list(fn)
#     recipe_dict = make_recipe_dict(recipe_list)

#     # igrins_log = IGRINSLog(igr_path, log_today)

#     obsids = recipe_dict["THAR"][0][0]
