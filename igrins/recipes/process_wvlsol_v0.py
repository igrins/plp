""" WVLSOL_V0
"""

import os
import numpy as np
# import numpy as np

from igrins.libs.products import PipelineProducts, PipelineImageBase
# from igrins.libs.apertures import Apertures


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
# #     from igrins.libs.recipes import Recipes #load_recipe_list, make_recipe_dict
# #     recipe = Recipes(fn)

# #     if starting_obsids is not None:
# #         starting_obsids = map(int, starting_obsids.split(","))

# #     selected = recipe.select("THAR", starting_obsids)

# #     for s in selected:
# #         obsids = s[0]

# #         for band in bands:
# #             process_thar_band(utdate, refdate, band, obsids)


from igrins.libs.products import ProductDB

from igrins.libs.recipe_helper import RecipeHelper


def make_combined_image(obsset):

    from igrins.libs.image_combine import make_combined_sky

    # caldb = helper.get_caldb()

    if obsset.recipe_name.upper() in ["SKY"]:
        frame_types = None
    else:
        frame_types = obsset.frametypes

    hdus = obsset.get_hdu_list()
    d = make_combined_sky(hdus, frame_types)

    destripe_mask = obsset.get("destripe_mask")

    from igrins.libs.image_combine import destripe_sky
    sky_data = destripe_sky(d, destripe_mask, subtract_bg=False)

    obsset.store_image(item_type="combined_sky", data=sky_data)

    return sky_data


def extract_spectra(obsset):
    "extract spectra"

    from aperture_helper import get_simple_aperture_from_obsset

    # caldb = helper.get_caldb()
    # master_obsid = obsids[0]

    data = obsset.load_image(item_type="combined_sky")

    aperture = get_simple_aperture_from_obsset(obsset)

    specs = aperture.extract_spectra_simple(data)

    obsset.store_dict(item_type="ONED_SPEC_JSON",
                      data=dict(orders=aperture.orders,
                                specs=specs,
                                aperture_basename=aperture.basename))


def _get_slices(n_slice_one_direction):
    """
    given number of slices per direction, return slices for the
    center, up and down positions.
    """
    n_slice = n_slice_one_direction*2 + 1
    i_center = n_slice_one_direction
    slit_slice = np.linspace(0., 1., n_slice+1)

    slice_center = (slit_slice[i_center], slit_slice[i_center+1])

    slice_up = [(slit_slice[i_center+i], slit_slice[i_center+i+1])
                for i in range(1, n_slice_one_direction+1)]

    slice_down = [(slit_slice[i_center-i-1], slit_slice[i_center-i])
                  for i in range(n_slice_one_direction)]

    return slice_center, slice_up, slice_down


from collections import namedtuple
SimpleHDU = namedtuple('SimpleHDU', ['header', 'data'])


def extract_spectra_multi(obsset):

    n_slice_one_direction = 2
    slice_center, slice_up, slice_down = _get_slices(n_slice_one_direction)

    from aperture_helper import get_simple_aperture_from_obsset

    data = obsset.load_image(item_type="combined_sky")

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]

    ap = get_simple_aperture_from_obsset(obsset, orders=orders)

    def make_hdu(s_up, s_down, data):
        h = [("NSLIT", n_slice_one_direction*2 + 1),
             ("FSLIT_DN", s_down),
             ("FSLIT_UP", s_up),
             ("FSLIT_CN", 0.5 * (s_up+s_down)),
             ("NORDER", len(ap.orders)),
             ("B_ORDER", ap.orders[0]),
             ("E_ORDER", ap.orders[-1]), ]

        return SimpleHDU(h, np.array(data))

    hdu_list = []

    s_center = ap.extract_spectra_v2(data,
                                     slice_center[0], slice_center[1])
    hdu_list.append(make_hdu(slice_center[0], slice_center[1], s_center))

    #s_up, s_down = [], []

    for s1, s2 in slice_up:
        s = ap.extract_spectra_v2(data, s1, s2)
        hdu_list.append(make_hdu(s1, s2, s))
        #s_up.append(s)

    for s1, s2 in slice_down:
        s = ap.extract_spectra_v2(data, s1, s2)
        hdu_list.append(make_hdu(s1, s2, s))
        #s_down.append(s)

    obsset.store_multi_images(item_type="MULTI_SPEC_FITS",
                              hdu_list=hdu_list)


# def get_thar_products_deprecated(helper, band, obsids):

#     thar_master_obsid = obsids[0]

#     ap = get_simple_aperture(helper, band, thar_master_obsid)


#     if 1:
#         from igrins.libs.process_thar import ThAr

#         thar_filenames = helper.get_filenames(band, obsids)
#         thar = ThAr(thar_filenames)

#         thar_products = thar.process_thar(ap)

#     return thar_products

def _get_ref_spec_name(recipe_name):
    if (recipe_name in ["SKY"]) or recipe_name.endswith("_AB"):
        ref_spec_key = "SKY_REFSPEC_JSON"
        ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"
    elif recipe_name in ["THAR"]:
        ref_spec_key = "THAR_REFSPEC_JSON"
        ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"
    else:
        raise ValueError("Recipe name of '%s' is unsupported." % recipe_name)

    return ref_spec_key, ref_identified_lines_key

def identify_orders(obsset):

    ref_spec_key, _ = _get_ref_spec_name(obsset.recipe_name)
    # from igrins.libs.master_calib import load_ref_data
    #ref_spectra = load_ref_data(helper.config, band,

    ref_spec_path, ref_spectra = obsset.fetch_ref_data(kind=ref_spec_key)

    src_spectra = obsset.load_item("ONED_SPEC_JSON")

    from igrins.libs.process_thar import match_order
    new_orders = match_order(src_spectra, ref_spectra)

    print  new_orders

    src_spectra["orders"] = new_orders
    obsset.store_dict(item_type="ONED_SPEC_JSON",
                      data=src_spectra)

    aperture_basename = src_spectra["aperture_basename"]
    obsset.store_dict(item_type="ORDERS_JSON",
                      data=dict(orders=new_orders,
                                aperture_basename=aperture_basename,
                                ref_spec_path=ref_spec_path))



# def get_orders_matching_ref_spec2_deprecated(helper, band, obsids, thar_products):
#     if 1:
#         from igrins.libs.process_thar import match_order_thar
#         from igrins.libs.master_calib import load_thar_ref_data

#         thar_ref_data = load_thar_ref_data(helper.refdate, band)

#         new_orders = match_order_thar(thar_products, thar_ref_data)

#         print thar_ref_data["orders"]
#         print  new_orders

#     if 1:

#         from igrins.libs.storage_descriptions import ONED_SPEC_JSON_DESC
#         thar_products[ONED_SPEC_JSON_DESC]["orders"] = new_orders

#         thar_filenames = helper.get_filenames(band, obsids)

#         hdu = pyfits.open(thar_filenames[0])[0]
#         helper.igr_storage.store(thar_products,
#                                  mastername=thar_filenames[0],
#                                  masterhdu=hdu)

#     return new_orders




def identify_lines(obsset):

    ref_spec_key, ref_identified_lines_key = _get_ref_spec_name(obsset.recipe_name)

    _, ref_spec = obsset.fetch_ref_data(kind=ref_spec_key)

    tgt_spec_path = obsset.query_item_path("ONED_SPEC_JSON")
    tgt_spec = obsset.load_item("ONED_SPEC_JSON")
    #tgt_spec = obsset.load_item("ONED_SPEC_JSON")

    from igrins.libs.process_thar import get_offset_treanform_between_2spec
    intersected_orders, d = get_offset_treanform_between_2spec(ref_spec,
                                                               tgt_spec)


    #REF_TYPE="OH"
    #fn = "../%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band,
    #                                             helper.refdate)
    _, l = obsset.fetch_ref_data(kind=ref_identified_lines_key)
    #l = json.load(open(fn))
    #ref_spectra = load_ref_data(helper.config, band, kind="SKY_REFSPEC_JSON")

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    from igrins.libs.identified_lines import IdentifiedLines

    identified_lines_ref = IdentifiedLines(l)
    ref_map = identified_lines_ref.get_dict()

    identified_lines_tgt = IdentifiedLines(l)
    identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                     pixpos_list=[], orders=[],
                                     spec_path=tgt_spec_path))

    from igrins.libs.line_identify_simple import match_lines1_pix

    for o, s in zip(tgt_spec["orders"], tgt_spec["specs"]):
        if (o not in ref_map) or (o not in offsetfunc_map):
            wvl, indices, pixpos = [], [], []
        else:
            pixpos, indices, wvl = ref_map[o]
            pixpos = np.array(pixpos)
            msk = (pixpos >= 0)

            ref_pix_list = offsetfunc_map[o](pixpos[msk])
            pix_list, dist = match_lines1_pix(np.array(s), ref_pix_list)

            pix_list[dist > 1] = -1
            pixpos[msk] = pix_list

        identified_lines_tgt.append_order_info(o, wvl, indices, pixpos)

    #REF_TYPE = "OH"
    #fn = "%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band, helper.utdate)
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    obsset.store_dict("IDENTIFIED_LINES_JSON",
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

#         from igrins.libs.process_thar import reidentify_ThAr_lines
#         thar_reidentified_products = reidentify_ThAr_lines(thar_products,
#                                                            thar_ref_data)

#         hdu = pyfits.open(thar_filenames[0])[0]
#         helper.igr_storage.store(thar_reidentified_products,
#                                  mastername=thar_filenames[0],
#                                  masterhdu=hdu)

#         return thar_reidentified_products


def test_identify_lines(helper, band, obsids):


    from igrins.libs.master_calib import load_ref_data
    ref_spec = load_ref_data(helper.config, band,
                             kind="SKY_REFSPEC_JSON")

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    tgt_spec = caldb.load_item_from((band, master_obsid),
                                    "ONED_SPEC_JSON")

    from igrins.libs.process_thar import get_offset_treanform_between_2spec
    d = get_offset_treanform_between_2spec(ref_spec, tgt_spec)

    print d

# def find_initial_wvlsol(helper, band, obsids,
#                         thar_products,
#                         thar_reidentified_products,
#                         new_orders):
#     if 1:

#         from igrins.libs.process_thar import (load_echelogram,
#                                        align_echellogram_thar,
#                                        check_thar_transorm,
#                                        get_wavelength_solutions)

#         from igrins.libs.master_calib import load_thar_ref_data

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

#         from igrins.libs.qa_helper import figlist_to_pngs
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
#

def save_figures(obsset):

    ### THIS NEEDS TO BE REFACTORED!

    orders = obsset.load_item("ORDERS_JSON")["orders"]

    # thar_filenames = helper.get_filenames(band, obsids)
    # thar_basename = os.path.splitext(os.path.basename(thar_filenames[0]))[0]
    # thar_master_obsid = obsids[0]

    if 1: # make amp and order falt

        from aperture_helper import get_simple_aperture_from_obsset

        ap = get_simple_aperture_from_obsset(obsset, orders=orders)

        # from igrins.libs.storage_descriptions import ONED_SPEC_JSON_DESC

        #orders = thar_products[ONED_SPEC_JSON_DESC]["orders"]
        order_map = ap.make_order_map()
        #slitpos_map = ap.make_slitpos_map()


        # load flat on products
        #flat_on_params_name = flaton_path.get_secondary_path("flat_on_params")

        #flaton_products = PipelineProducts.load(flat_on_params_name)
        # from igrins.libs.storage_descriptions import (FLAT_NORMED_DESC,
        #                                        FLAT_MASK_DESC)

        # flaton_db_name = helper.get_section_filename_base("PRIMARY_CALIB_PATH",
        #                                                   "flat_on.db")
        # flaton_db = ProductDB(flaton_db_name)

        # flaton_basename = flaton_db.query(band, thar_master_obsid)

        # flaton_products = helper.load([FLAT_NORMED_DESC,
        #                                FLAT_MASK_DESC],
        #                               flaton_basename)

        # flat_on_basename = obsset.caldb.db_query_basename("flat_on",
        #                                                   obsset.basename)
        flat_normed = obsset.load_resource_for(("flat_on", "flat_normed"),
                                               get_science_hdu=True).data
        flat_mask = obsset.load_resource_for(("flat_on", "flat_mask"),
                                               get_science_hdu=True).data > 0

        from igrins.libs.process_flat import make_order_flat, check_order_flat
        order_flat_im, order_flat_json = make_order_flat(flat_normed, 
                                                         flat_mask,
                                                         orders, order_map)

        obsset.store_image("order_flat_im", order_flat_im)
        obsset.store_dict("order_flat_json", order_flat_json)


        # from igrins.libs.load_fits import load_fits_data
        # hdu = load_fits_data(thar_filenames[0])
        # # hdu = pyfits.open(thar_filenames[0])[0]
        
        # helper.store(order_flat_products,
        #              mastername=flaton_basename,
        #              masterhdu=hdu)

        # flat_mask = helper.load1(FLAT_MASK_DESC,
        #                          flaton_basename)
        order_map2 = ap.make_order_map(mask_top_bottom=True)
        bias_mask = flat_mask & (order_map2 > 0)

        obsset.store_image("bias_mask", bias_mask)
        
        # pp = PipelineProducts("")
        # from igrins.libs.storage_descriptions import BIAS_MASK_DESC
        # pp.add(BIAS_MASK_DESC,
        #        PipelineImageBase([], bias_mask))

        # helper.store(pp, mastername=flaton_basename, masterhdu=hdu)

    if 1:
        fig_list = check_order_flat(order_flat_json)

        from igrins.libs.qa_helper import figlist_to_pngs
        dest_dir = obsset.query_item_path("qa_orderflat_dir",
                                          subdir="orderflat")
        figlist_to_pngs(dest_dir, fig_list)


def save_db(obsset):

    # save db
    db = obsset.load_db("register")
    db.update(obsset.band, obsset.basename)


def process_band(utdate, recipe_name, band,
                 obsids, frametypes, aux_infos,
                 config_name, **kwargs):

    if not kwargs.pop("do_ab") and recipe_name.upper().endswith("_AB"):
        return

    from igrins import get_caldb, get_obsset
    caldb = get_caldb(config_name, utdate)
    obsset = get_obsset(caldb, recipe_name, band, obsids, frametypes)

    # STEP 1 :
    # make combined image

    if recipe_name.upper() in ["SKY"]:
        pass
    elif recipe_name.upper().endswith("_AB"):
        pass
    elif recipe_name.upper() in ["THAR"]:
        pass
    else:
        msg = ("recipe_name {} not supported "
               "for this recipe").format(recipe_name)
        raise ValueError(msg)

    make_combined_image(obsset)

    # Step 2

    # load simple-aperture (no order info; depends on

    extract_spectra(obsset)

    ## aperture trace from Flat)

    ## extract 1-d spectra from ThAr

    # Step 3:
    ## compare to reference ThAr data to figure out orders of each strip
    ##  -  simple correlation w/ clipping

    identify_orders(obsset)

    # Step 4:
    ##  - For each strip, measure x-displacement from the reference
    ##    spec. Fit the displacement as a function of orders.
    ##  - Using the estimated displacement, identify lines from the spectra.
    identify_lines(obsset)

    # Step 6:

    ## load the reference echellogram, and find the transform using
    ## the identified lines.

    from find_affine_transform import find_affine_transform
    find_affine_transform(obsset)

    from igrins.libs.transform_wvlsol import transform_wavelength_solutions
    transform_wavelength_solutions(obsset)

    # Step 8:

    ## make order_map and auxilary files.

    save_figures(obsset)

    save_db(obsset)



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

#     from igrins.libs.recipes import load_recipe_list, make_recipe_dict
#     from igrins.libs.products import PipelineProducts, ProductPath, ProductDB

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
