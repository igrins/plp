import os
#import numpy as np


from libs.path_info import IGRINSPath, IGRINSFiles
import libs.fits as pyfits

from libs.products import PipelineProducts, PipelineImageBase
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


from libs.products import ProductDB

from libs.recipe_helper import RecipeHelper


def get_bottom_up_solution(helper, band, thar_master_obsid):
    """
    aperture that only requires flat product. No order information.
    """

    flaton_db_name = helper.igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "flat_on.db",
                                                        )
    flaton_db = ProductDB(flaton_db_name)

    flaton_basename = flaton_db.query(band, thar_master_obsid)


    from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC

    desc_list = [FLATCENTROID_SOL_JSON_DESC]
    products = helper.igr_storage.load(desc_list,
                                       mastername=flaton_basename)

    aperture_solution_products = products[FLATCENTROID_SOL_JSON_DESC]

    # igrins_orders = {}
    # igrins_orders["H"] = range(99, 122)
    # igrins_orders["K"] = range(72, 92)

    if 1:
        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]

        return bottomup_solutions

def get_simple_aperture(helper, band, thar_master_obsid, orders=None):

    bottomup_solutions = get_bottom_up_solution(helper, band, thar_master_obsid)

    if orders is None:
        orders = range(len(bottomup_solutions))

    ap =  Apertures(orders, bottomup_solutions)

    return ap

def get_thar_products(helper, band, obsids):

    thar_filenames = helper.get_filenames(band, obsids)
    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
    thar_master_obsid = obsids[0]

    ap = get_simple_aperture(helper, band, thar_master_obsid)

    if 1:
        from libs.process_thar import ThAr

        thar_filenames = helper.get_filenames(band, obsids)
        thar = ThAr(thar_filenames)

        thar_products = thar.process_thar(ap)

    return thar_products

def get_orders_matching_ref_spec(helper, band, obsids, thar_products):
    if 1:
        from libs.process_thar import match_order_thar
        from libs.master_calib import load_thar_ref_data

        thar_ref_data = load_thar_ref_data(helper.refdate, band)

        new_orders = match_order_thar(thar_products, thar_ref_data)

        print thar_ref_data["orders"]
        print  new_orders

    if 1:

        from libs.storage_descriptions import ONED_SPEC_JSON_DESC
        thar_products[ONED_SPEC_JSON_DESC]["orders"] = new_orders

        thar_filenames = helper.get_filenames(band, obsids)

        hdu = pyfits.open(thar_filenames[0])[0]
        helper.igr_storage.store(thar_products,
                                 mastername=thar_filenames[0],
                                 masterhdu=hdu)

    return new_orders



def get_orders_matching_ref_spec2(helper, band, obsids, thar_products):
    if 1:
        from libs.process_thar import match_order_thar
        from libs.master_calib import load_thar_ref_data

        thar_ref_data = load_thar_ref_data(helper.refdate, band)

        new_orders = match_order_thar(thar_products, thar_ref_data)

        print thar_ref_data["orders"]
        print  new_orders

    if 1:

        from libs.storage_descriptions import ONED_SPEC_JSON_DESC
        thar_products[ONED_SPEC_JSON_DESC]["orders"] = new_orders

        thar_filenames = helper.get_filenames(band, obsids)

        hdu = pyfits.open(thar_filenames[0])[0]
        helper.igr_storage.store(thar_products,
                                 mastername=thar_filenames[0],
                                 masterhdu=hdu)

    return new_orders




def identify_lines(helper, band, obsids, thar_products):
    thar_filenames = helper.get_filenames(band, obsids)

    if 1:
        from libs.master_calib import load_thar_ref_data

        #ref_date = "20140316"

        thar_ref_data = load_thar_ref_data(helper.refdate, band)

        from libs.process_thar import reidentify_ThAr_lines
        thar_reidentified_products = reidentify_ThAr_lines(thar_products,
                                                           thar_ref_data)

        hdu = pyfits.open(thar_filenames[0])[0]
        helper.igr_storage.store(thar_reidentified_products,
                                 mastername=thar_filenames[0],
                                 masterhdu=hdu)

        return thar_reidentified_products


def find_initial_wvlsol(helper, band, obsids,
                        thar_products,
                        thar_reidentified_products,
                        new_orders):
    if 1:

        from libs.process_thar import (load_echelogram,
                                       align_echellogram_thar,
                                       check_thar_transorm,
                                       get_wavelength_solutions)

        from libs.master_calib import load_thar_ref_data

        #ref_date = "20140316"

        thar_ref_data = load_thar_ref_data(helper.refdate, band)

        ref_date = thar_ref_data["ref_date"]
        echel = load_echelogram(ref_date, band)

        thar_master_obsid = obsids[0]
        ap = get_simple_aperture(helper, band, thar_master_obsid,
                                 orders=new_orders)

        thar_aligned_echell_products = \
             align_echellogram_thar(thar_reidentified_products,
                                    echel, band, ap)

        # We do not save this product yet.
        # igr_storage.store(thar_aligned_echell_products,
        #                   mastername=thar_filenames[0],
        #                   masterhdu=hdu)


    # Make figures

    thar_filenames = helper.get_filenames(band, obsids)
    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
    thar_master_obsid = obsids[0]

    if 1:

        fig_list = check_thar_transorm(thar_products,
                                       thar_aligned_echell_products)

        from libs.qa_helper import figlist_to_pngs
        igr_path = helper.igr_path
        thar_figs = igr_path.get_section_filename_base("QA_PATH",
                                                       "thar",
                                                       "thar_"+thar_basename)
        figlist_to_pngs(thar_figs, fig_list)

        thar_wvl_sol = get_wavelength_solutions(thar_aligned_echell_products,
                                                echel,
                                                new_orders)

        hdu = pyfits.open(thar_filenames[0])[0]
        helper.igr_storage.store(thar_wvl_sol,
                                 mastername=thar_filenames[0],
                                 masterhdu=hdu)

def save_figures(helper, band, obsids, thar_products, new_orders):
    thar_filenames = helper.get_filenames(band, obsids)
    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
    thar_master_obsid = obsids[0]

    if 1: # make amp and order falt

        ap = get_simple_aperture(helper, band, thar_master_obsid,
                                 orders=new_orders)

        from libs.storage_descriptions import ONED_SPEC_JSON_DESC

        orders = thar_products[ONED_SPEC_JSON_DESC]["orders"]
        order_map = ap.make_order_map()
        #slitpos_map = ap.make_slitpos_map()


        # load flat on products
        #flat_on_params_name = flaton_path.get_secondary_path("flat_on_params")

        #flaton_products = PipelineProducts.load(flat_on_params_name)
        from libs.storage_descriptions import (FLAT_NORMED_DESC,
                                               FLAT_MASK_DESC)

        flaton_db_name = helper.igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                                   "flat_on.db",
                                                                   )
        flaton_db = ProductDB(flaton_db_name)

        flaton_basename = flaton_db.query(band, thar_master_obsid)

        flaton_products = helper.igr_storage.load([FLAT_NORMED_DESC,
                                                   FLAT_MASK_DESC],
                                                  flaton_basename)

        from libs.process_flat import make_order_flat, check_order_flat
        order_flat_products = make_order_flat(flaton_products,
                                              orders, order_map)

        #fn = thar_path.get_secondary_path("orderflat")
        #order_flat_products.save(fn, masterhdu=hdu)

        hdu = pyfits.open(thar_filenames[0])[0]
        helper.igr_storage.store(order_flat_products,
                                 mastername=flaton_basename,
                                 masterhdu=hdu)

        flat_mask = helper.igr_storage.load1(FLAT_MASK_DESC,
                                             flaton_basename)
        order_map2 = ap.make_order_map(mask_top_bottom=True)
        bias_mask = flat_mask.data & (order_map2 > 0)

        pp = PipelineProducts("")
        from libs.storage_descriptions import BIAS_MASK_DESC
        pp.add(BIAS_MASK_DESC,
               PipelineImageBase([], bias_mask))

        helper.igr_storage.store(pp,
                                 mastername=flaton_basename,
                                 masterhdu=hdu)

    if 1:
        fig_list = check_order_flat(order_flat_products)

        from libs.qa_helper import figlist_to_pngs
        orderflat_figs = helper.igr_path.get_section_filename_base("QA_PATH",
                                                                   "orderflat",
                                                                   "orderflat_"+thar_basename)
        figlist_to_pngs(orderflat_figs, fig_list)


def save_db(helper, band, obsids):
    thar_filenames = helper.get_filenames(band, obsids)
    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]

    if 1:
        from libs.products import ProductDB
        thar_db_name = helper.igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                                 "thar.db",
                                                                 )
        thar_db = ProductDB(thar_db_name)
        # os.path.join(igr_path.secondary_calib_path,
        #                                  "thar.db"))
        thar_db.update(band, thar_basename)


def process_thar_band(utdate, refdate, band, obsids, config):

    helper = RecipeHelper(utdate, refdate, config)




    # STEP 1 :
    ## load simple-aperture (no order info; depends on
    ## aperture trace from Flat)

    # Step 2
    ## extract 1-d spectra from ThAr


    thar_products = get_thar_products(helper, band, obsids)

    # Step 3:
    ## compare to reference ThAr data to figure out orders of each strip
    ##  -  simple correlation w/ clipping

    new_orders = get_orders_matching_ref_spec(helper, band, obsids, thar_products)

    # Step 4:


    # step 5:
    ##  - For each strip, measure x-displacement from the reference
    ##    spec. Fit the displacement as a function of orders.
    ##  - Using the estimated displacement, identify lines from the spectra.
    thar_reidentified_products = identify_lines(helper, band, obsids,
                                                thar_products)

    # Step 6:

    ## load the reference echellogram, and find the transform using
    ## the identified lines.

    find_initial_wvlsol(helper, band, obsids,
                        thar_products,
                        thar_reidentified_products,
                        new_orders)


    # Step 8:

    ## make order_map and auxilary files.

    save_figures(helper, band, obsids, thar_products, new_orders)

    save_db(helper, band, obsids)


def main():
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


if __name__ == "__main__":
    utdate = "20150525"
    band = "H"
    obsids = [52]
    refdate = "20140316"

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig("recipe.config")


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
