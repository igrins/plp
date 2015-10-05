import os
#import numpy as np


import libs.fits as pyfits

from libs.products import PipelineProducts, PipelineImageBase
#from libs.apertures import Apertures

from libs.recipe_base import RecipeBase

import numpy as np

# class RecipeThAr(RecipeBase):
#     RECIPE_NAME = "THAR"

#     def run_selected_bands(self, utdate, selected, bands):
#         for s in selected:
#             obsids = s[0]
#             print obsids
#             # frametypes = s[1]

#             for band in bands:
#                 process_thar_band(utdate, self.refdate, band, obsids,
#                                   self.config)

# def thar(utdate, bands="HK",
#          starting_obsids=None, config_file="recipe.config"):

#     RecipeThAr()(utdate, bands,
#                  starting_obsids, config_file)



# # def thar(utdate, refdate="20140316", bands="HK",
# #          starting_obsids=None):

# #     if not bands in ["H", "K", "HK"]:
# #         raise ValueError("bands must be one of 'H', 'K' or 'HK'")

# #     fn = "%s.recipes" % utdate
# #     from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
# #     recipe = Recipes(fn)

# #     if starting_obsids is not None:
# #         starting_obsids = map(int, starting_obsids.split(","))

# #     selected = recipe.select("THAR", starting_obsids)

# #     for s in selected:
# #         obsids = s[0]

# #         for band in bands:
# #             process_thar_band(utdate, refdate, band, obsids)


from libs.products import ProductDB

from libs.recipe_helper import RecipeHelper


def make_combined_image(helper, band, obsids, mode=None):

    from libs.image_combine import make_combined_image_thar

    caldb = helper.get_caldb()

    d = make_combined_image_thar(helper, band, obsids)

    master_obsid = obsids[0]
    caldb.store_image((band, master_obsid),
                      item_type="combined_image", data=d)

from aperture_helper import get_simple_aperture

def extract_spectra(helper, band, obsids):

    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    data = caldb.load_image((band, master_obsid),
                            item_type="combined_image")

    ap = get_simple_aperture(helper, band, obsids)

    s = ap.extract_spectra_simple(data)

    caldb.store_dict((band, master_obsid),
                     item_type="ONED_SPEC_JSON",
                     data=dict(orders=ap.orders,
                               specs=s,
                               aperture_basename=ap.basename
                               ))


# def get_thar_products_deprecated(helper, band, obsids):

#     thar_master_obsid = obsids[0]

#     ap = get_simple_aperture(helper, band, thar_master_obsid)


#     if 1:
#         from libs.process_thar import ThAr

#         thar_filenames = helper.get_filenames(band, obsids)
#         thar = ThAr(thar_filenames)

#         thar_products = thar.process_thar(ap)

#     return thar_products

def _get_ref_spec_name(helper):
    if helper.recipe_name in ["SKY"]:
        ref_spec_key = "SKY_REFSPEC_JSON"
        ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"
    elif helper.recipe_name in ["THAR"]:
        ref_spec_key = "THAR_REFSPEC_JSON"
        ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"
    else:
        raise ValueError("Recipe name of '%s' is unsupported." % helper.recipe_name)

    return ref_spec_key, ref_identified_lines_key

def identify_orders(helper, band, obsids):

    ref_spec_key, _ = _get_ref_spec_name(helper)
    from libs.master_calib import load_ref_data
    ref_spectra = load_ref_data(helper.config, band,
                                kind=ref_spec_key)

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    src_spectra = caldb.load_item_from((band, master_obsid),
                                       "ONED_SPEC_JSON")

    from libs.process_thar import match_order
    new_orders = match_order(src_spectra, ref_spectra)

    print  new_orders

    src_spectra["orders"] = new_orders
    caldb.store_dict((band, master_obsid),
                     item_type="ONED_SPEC_JSON",
                     data=src_spectra
                     )

    aperture_basename = src_spectra["aperture_basename"]
    caldb.store_dict(aperture_basename,
                     item_type="FLATCENTROID_ORDERS_JSON",
                     data=dict(orders=new_orders,
                               aperture_basename=aperture_basename))



# def get_orders_matching_ref_spec2_deprecated(helper, band, obsids, thar_products):
#     if 1:
#         from libs.process_thar import match_order_thar
#         from libs.master_calib import load_thar_ref_data

#         thar_ref_data = load_thar_ref_data(helper.refdate, band)

#         new_orders = match_order_thar(thar_products, thar_ref_data)

#         print thar_ref_data["orders"]
#         print  new_orders

#     if 1:

#         from libs.storage_descriptions import ONED_SPEC_JSON_DESC
#         thar_products[ONED_SPEC_JSON_DESC]["orders"] = new_orders

#         thar_filenames = helper.get_filenames(band, obsids)

#         hdu = pyfits.open(thar_filenames[0])[0]
#         helper.igr_storage.store(thar_products,
#                                  mastername=thar_filenames[0],
#                                  masterhdu=hdu)

#     return new_orders




def identify_lines(helper, band, obsids):

    from libs.master_calib import load_ref_data

    ref_spec_key, ref_identified_lines_key = _get_ref_spec_name(helper)

    ref_spec = load_ref_data(helper.config, band,
                             kind=ref_spec_key)

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    tgt_spec_path = caldb.query_item_path((band, master_obsid),
                                          "ONED_SPEC_JSON")
    tgt_spec = caldb.load_item_from_path(tgt_spec_path)

    from libs.process_thar import get_offset_treanform_between_2spec
    intersected_orders, d = get_offset_treanform_between_2spec(ref_spec,
                                                               tgt_spec)


    #REF_TYPE="OH"
    #fn = "../%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band,
    #                                             helper.refdate)
    l = load_ref_data(helper.config, band,
                      kind=ref_identified_lines_key)
    #l = json.load(open(fn))
    #ref_spectra = load_ref_data(helper.config, band, kind="SKY_REFSPEC_JSON")

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    from libs.identified_lines import IdentifiedLines

    identified_lines_ref = IdentifiedLines(l)
    ref_map = identified_lines_ref.get_dict()

    identified_lines_tgt = IdentifiedLines(l)
    identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                     pixpos_list=[], orders=[],
                                     spec_path=tgt_spec_path))

    from libs.line_identify_simple import match_lines1_pix

    for o, s in zip(tgt_spec["orders"], tgt_spec["specs"]):
        if (o not in ref_map) or (o not in offsetfunc_map):
            wvl, indices, pixpos = [], [], []
        else:
            pixpos, indices, wvl = ref_map[o]
            pixpos = np.array(pixpos)
            msk = (pixpos >= 0)

            ref_pix_list = offsetfunc_map[o](pixpos[msk])
            pix_list, dist = match_lines1_pix(np.array(s), ref_pix_list)

            pix_list[dist>1] = -1
            pixpos[msk] = pix_list

        identified_lines_tgt.append_order_info(o, wvl, indices, pixpos)

    #REF_TYPE = "OH"
    #fn = "%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band, helper.utdate)
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    caldb.store_dict((band, master_obsid),
                     "IDENTIFIED_LINES_JSON",
                     identified_lines_tgt.data)

    #print d

    # aperture_basename = src_spectra["aperture_basename"]
    # orders = caldb.load_item_from(aperture_basename,
    #                               "FLATCENTROID_ORDERS_JSON")

    # caldb.load_resource_for((band, master_obsid),
    #                         resource_type="orders",
    #                         )


# if 0:
#     if 1:

#         from libs.process_thar import reidentify_ThAr_lines
#         thar_reidentified_products = reidentify_ThAr_lines(thar_products,
#                                                            thar_ref_data)

#         hdu = pyfits.open(thar_filenames[0])[0]
#         helper.igr_storage.store(thar_reidentified_products,
#                                  mastername=thar_filenames[0],
#                                  masterhdu=hdu)

#         return thar_reidentified_products


def test_identify_lines(helper, band, obsids):


    from libs.master_calib import load_ref_data
    ref_spec = load_ref_data(helper.config, band,
                             kind="SKY_REFSPEC_JSON")

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    tgt_spec = caldb.load_item_from((band, master_obsid),
                                    "ONED_SPEC_JSON")

    from libs.process_thar import get_offset_treanform_between_2spec
    d = get_offset_treanform_between_2spec(ref_spec, tgt_spec)

    print d



# def find_initial_wvlsol(helper, band, obsids,
#                         thar_products,
#                         thar_reidentified_products,
#                         new_orders):
#     if 1:

#         from libs.process_thar import (load_echelogram,
#                                        align_echellogram_thar,
#                                        check_thar_transorm,
#                                        get_wavelength_solutions)

#         from libs.master_calib import load_thar_ref_data

#         #ref_date = "20140316"

#         thar_ref_data = load_thar_ref_data(helper.refdate, band)

#         ref_date = thar_ref_data["ref_date"]
#         echel = load_echelogram(ref_date, band)

#         thar_master_obsid = obsids[0]
#         ap = get_simple_aperture(helper, band, thar_master_obsid,
#                                  orders=new_orders)

#         thar_aligned_echell_products = \
#              align_echellogram_thar(thar_reidentified_products,
#                                     echel, band, ap)

#         # We do not save this product yet.
#         # igr_storage.store(thar_aligned_echell_products,
#         #                   mastername=thar_filenames[0],
#         #                   masterhdu=hdu)


#     # Make figures

#     thar_filenames = helper.get_filenames(band, obsids)
#     thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
#     thar_master_obsid = obsids[0]

#     if 1:

#         fig_list = check_thar_transorm(thar_products,
#                                        thar_aligned_echell_products)

#         from libs.qa_helper import figlist_to_pngs
#         igr_path = helper.igr_path
#         thar_figs = igr_path.get_section_filename_base("QA_PATH",
#                                                        "thar",
#                                                        "thar_"+thar_basename)
#         figlist_to_pngs(thar_figs, fig_list)

#         thar_wvl_sol = get_wavelength_solutions(thar_aligned_echell_products,
#                                                 echel,
#                                                 new_orders)

#         hdu = pyfits.open(thar_filenames[0])[0]
#         helper.igr_storage.store(thar_wvl_sol,
#                                  mastername=thar_filenames[0],
#                                  masterhdu=hdu)

def save_figures(helper, band, obsids):

    ### THIS NEEDS TO BE REFACTORED!

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    orders = caldb.load_resource_for((band, master_obsid), "orders")["orders"]

    thar_filenames = helper.get_filenames(band, obsids)
    thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
    thar_master_obsid = obsids[0]

    if 1: # make amp and order falt

        ap = get_simple_aperture(helper, band, obsids,
                                 orders=orders)

        # from libs.storage_descriptions import ONED_SPEC_JSON_DESC

        #orders = thar_products[ONED_SPEC_JSON_DESC]["orders"]
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


def process_band(utdate, recipe_name, band, obsids, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    # STEP 1 :
    ## make combined image

    make_combined_image(helper, band, obsids, mode=None)

    # Step 2

    ## load simple-aperture (no order info; depends on

    extract_spectra(helper, band, obsids)
    ## aperture trace from Flat)

    ## extract 1-d spectra from ThAr


    # Step 3:
    ## compare to reference ThAr data to figure out orders of each strip
    ##  -  simple correlation w/ clipping

    identify_orders(helper, band, obsids)

    # Step 4:
    ##  - For each strip, measure x-displacement from the reference
    ##    spec. Fit the displacement as a function of orders.
    ##  - Using the estimated displacement, identify lines from the spectra.
    identify_lines(helper, band, obsids)


    # Step 6:

    ## load the reference echellogram, and find the transform using
    ## the identified lines.

    from find_affine_transform import find_affine_transform
    find_affine_transform(helper, band, obsids)

    from libs.transform_wvlsol import transform_wavelength_solutions
    transform_wavelength_solutions(helper, band, obsids)

    # Step 8:

    ## make order_map and auxilary files.

    save_figures(helper, band, obsids)

    save_db(helper, band, obsids)

# if 0:



#     # step 5:

#     # Step 6:

#     ## load the reference echellogram, and find the transform using
#     ## the identified lines.

#     find_initial_wvlsol(helper, band, obsids,
#                         thar_products,
#                         thar_reidentified_products,
#                         new_orders)


#     # Step 8:

#     ## make order_map and auxilary files.

#     save_figures(helper, band, obsids, thar_products, new_orders)

#     save_db(helper, band, obsids)


# def main():
#     import sys

#     utdate = sys.argv[1]
#     bands = "HK"
#     starting_obsids = None

#     if len(sys.argv) >= 3:
#         bands = sys.argv[2]

#     if len(sys.argv) >= 4:
#         starting_obsids = sys.argv[3]

#     thar(utdate, refdate="20140316", bands=bands,
#          starting_obsids=starting_obsids)


if __name__ == "__main__":

    utdate = "20140709"
    obsids = [62, 63]

    utdate = "20140525"
    obsids = [29]

    utdate = "20150525"
    obsids = [52]


    recipe_name = "SKY"


    utdate = "20150525"
    obsids = [32]

    recipe_name = "THAR"

    band = "K"

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    process_band(utdate, recipe_name, band, obsids, config_name)


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
