import os
#import numpy as np

from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath
import libs.fits as pyfits

from libs.recipe_base import RecipeBase

class RecipeFlat(RecipeBase):
    RECIPE_NAME = "FLAT"

    def run_selected_bands(self, utdate, selected, bands):
        for s in selected:
            obsids = s[0]
            frametypes = s[1]

            obsids_off = [obsid for obsid, frametype \
                          in zip(obsids, frametypes) if frametype == "OFF"]
            obsids_on = [obsid for obsid, frametype \
                         in zip(obsids, frametypes) if frametype == "ON"]

            for band in bands:
                process_flat_band(utdate, self.refdate, band,
                                  obsids_off, obsids_on,
                                  self.config)

# the end point for recipe needs to be a function so that Argh works.
# TODO : See if we can avoid duplicating the function signatures.

def flat(utdate, bands="HK",
         starting_obsids=None, config_file="recipe.config"):

    RecipeFlat()(utdate, bands,
                 starting_obsids, config_file)


def process_flat_band(utdate, refdate, band, obsids_off, obsids_on,
                      config):
    from libs.products import PipelineStorage

    igr_path = IGRINSPath(config, utdate)

    igr_storage = PipelineStorage(igr_path)


    flat_off_filenames = igr_path.get_filenames(band, obsids_off)
    flat_on_filenames = igr_path.get_filenames(band, obsids_on)

    if 1: # process flat off

        flat_offs_hdu_list = [pyfits.open(fn_)[0] for fn_ in flat_off_filenames]
        flat_offs = [hdu.data for hdu in flat_offs_hdu_list]


        flat = FlatOff(flat_offs)
        flatoff_products = flat.make_flatoff_hotpixmap(sigma_clip1=100,
                                                       sigma_clip2=5)

        igr_storage.store(flatoff_products,
                          mastername=flat_off_filenames[0],
                          masterhdu=flat_offs_hdu_list[0])



    if 1: # flat on

        from libs.storage_descriptions import (FLAT_OFF_DESC,
                                               HOTPIX_MASK_DESC,
                                               FLATOFF_JSON_DESC)

        desc_list = [FLAT_OFF_DESC, HOTPIX_MASK_DESC, FLATOFF_JSON_DESC]
        flatoff_products = igr_storage.load(desc_list,
                                            mastername=flat_off_filenames[0])

        flat_on_hdu_list = [pyfits.open(fn_)[0] for fn_ in flat_on_filenames]
        flat_ons = [hdu.data for hdu in flat_on_hdu_list]


        from libs.master_calib import get_master_calib_abspath
        fn = get_master_calib_abspath("deadpix_mask_%s_%s.fits" % (refdate,
                                                                   band))
        deadpix_mask_old = pyfits.open(fn)[0].data.astype(bool)

        flat_on = FlatOn(flat_ons)
        flaton_products = flat_on.make_flaton_deadpixmap(flatoff_products,
                                                         deadpix_mask_old=deadpix_mask_old)

        igr_storage.store(flaton_products,
                          mastername=flat_on_filenames[0],
                          masterhdu=flat_on_hdu_list[0])



    if 1: # now trace the orders

        from libs.process_flat import trace_orders

        trace_products = trace_orders(flaton_products)

        hdu = pyfits.open(flat_on_filenames[0])[0]

        igr_storage.store(trace_products,
                          mastername=flat_on_filenames[0],
                          masterhdu=flat_on_hdu_list[0])


        from libs.process_flat import trace_solutions
        trace_solution_products, trace_solution_products_plot = \
                                 trace_solutions(trace_products)


    if 1:
        trace_solution_products.keys()
        from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC

        myproduct = trace_solution_products[FLATCENTROID_SOL_JSON_DESC]
        bottomup_solutions = myproduct["bottom_up_solutions"]

        orders = range(len(bottomup_solutions))

        from libs.apertures import Apertures
        ap =  Apertures(orders, bottomup_solutions)

        from libs.storage_descriptions import FLAT_MASK_DESC
        flat_mask = igr_storage.load1(FLAT_MASK_DESC,
                                      flat_on_filenames[0])
        order_map2 = ap.make_order_map(mask_top_bottom=True)
        bias_mask = flat_mask.data & (order_map2 > 0)

        from libs.products import PipelineImageBase, PipelineProducts
        pp = PipelineProducts("")
        from libs.storage_descriptions import BIAS_MASK_DESC
        pp.add(BIAS_MASK_DESC,
               PipelineImageBase([], bias_mask))

        flaton_basename = flat_on_filenames[0]
        igr_storage.store(pp,
                          mastername=flaton_basename,
                          masterhdu=hdu)


    # plot qa figures.

    if 1:
        from libs.process_flat import check_trace_order
        from matplotlib.figure import Figure
        fig1 = Figure(figsize=[9, 4])
        check_trace_order(trace_products, fig1)

    if 1:
        from libs.process_flat import plot_trace_solutions
        fig2, fig3 = plot_trace_solutions(flaton_products,
                                          trace_solution_products,
                                          trace_solution_products_plot,
                                          )

    flatoff_basename = os.path.splitext(os.path.basename(flat_off_filenames[0]))[0]
    flaton_basename = os.path.splitext(os.path.basename(flat_on_filenames[0]))[0]

    if 1:
        from libs.qa_helper import figlist_to_pngs
        aperture_figs = igr_path.get_section_filename_base("QA_PATH",
                                                           "aperture_"+flaton_basename,
                                                           "aperture_"+flaton_basename)

        figlist_to_pngs(aperture_figs, [fig1, fig2, fig3])



    if 1: # now trace the orders

        #del trace_solution_products["bottom_up_solutions"]
        igr_storage.store(trace_solution_products,
                          mastername=flat_on_filenames[0],
                          masterhdu=flat_on_hdu_list[0])



    # save db
    if 1:
        from libs.products import ProductDB
        flatoff_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                             "flat_off.db",
                                                             )
        flatoff_db = ProductDB(flatoff_db_name)
        #dbname = os.path.splitext(os.path.basename(flat_off_filenames[0]))[0]
        flatoff_db.update(band, flatoff_basename)


        flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                             "flat_on.db",
                                                             )
        flaton_db = ProductDB(flaton_db_name)
        flaton_db.update(band, flaton_basename)



if __name__ == "__main__":
    import sys

    utdate = sys.argv[1]
    bands = "HK"
    starting_obsids = None

    if len(sys.argv) >= 3:
        bands = sys.argv[2]

    if len(sys.argv) >= 4:
        starting_obsids = sys.argv[3]

    flat(utdate, refdate="20140316", bands=bands,
         starting_obsids=starting_obsids)
