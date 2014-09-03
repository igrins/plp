import os
#import numpy as np

from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath
import astropy.io.fits as pyfits

REFDATE = "20140316"


class RecipeBase(object):
    """ The derived mus define RECIPE_NAME attribute and must implement
        run_selected_bands method.
    """

    def _validate_bands(self, bands):
        if not bands in ["H", "K", "HK"]:
            raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    def get_recipe_name(self, utdate):
        return "%s.recipes" % utdate

    def get_recipes(self, utdate):
        fn = self.get_recipe_name(utdate)
        from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
        return Recipes(fn)

    def parse_starting_obsids(self, starting_obsids):
        if starting_obsids is not None:
            starting_obsids = map(int, starting_obsids.split(","))
            return starting_obsids
        else:
            return None

    def __call__(self, utdate, bands="HK",
                 starting_obsids=None, config_file="recipe.config"):

        from libs.igrins_config import IGRINSConfig
        self.config = IGRINSConfig(config_file)

        self.refdate = self.config.get_value('REFDATE', utdate)

        self._validate_bands(bands)

        recipes = self.get_recipes(utdate)

        starting_obsids_parsed = self.parse_starting_obsids(starting_obsids)

        selected = recipes.select(self.RECIPE_NAME, starting_obsids_parsed)

        self.run_selected_bands(utdate, selected, bands)


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

        from libs.process_flat import FLAT_OFF_DESC, HOTPIX_MASK_DESC, FLATOFF_JSON_DESC
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
        trace_solution_products = trace_solutions(trace_products)


        igr_storage.store(trace_solution_products,
                          mastername=flat_on_filenames[0],
                          masterhdu=flat_on_hdu_list[0])


    # plot qa figures.

    if 1:
        from libs.process_flat import check_trace_order
        from matplotlib.figure import Figure
        fig1 = Figure()
        check_trace_order(trace_products, fig1)

    if 1:
        from libs.process_flat import plot_trace_solutions
        fig2, fig3 = plot_trace_solutions(flaton_products,
                                          trace_solution_products)

    if 1:
        from libs.qa_helper import figlist_to_pngs
        aperture_figs = igr_path.get_section_filename_base("QA_PATH",
                                                           "aperture",
                                                           "aperture_dir")

        figlist_to_pngs(aperture_figs, [fig1, fig2, fig3])

    # save db
    if 1:
        from libs.products import ProductDB
        flatoff_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                             "flat_off.db",
                                                             )
        flatoff_db = ProductDB(flatoff_db_name)
        flatoff_db.update(band, os.path.basename(flat_off_filenames[0]))


        flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                             "flat_on.db",
                                                             )
        flaton_db = ProductDB(flaton_db_name)
        flaton_db.update(band, os.path.basename(flat_on_filenames[0]))



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
