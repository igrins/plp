import os
#import numpy as np


from libs.path_info import IGRINSPath, IGRINSFiles
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

if __name__ == "__main__":

    from libs.recipes import load_recipe_list, make_recipe_dict
    from libs.products import PipelineProducts, ProductPath, ProductDB

    if 0:
        utdate = "20140316"
        # log_today = dict(flat_off=range(2, 4),
        #                  flat_on=range(4, 7),
        #                  thar=range(1, 2))
    elif 1:
        utdate = "20140525"
        # log_today = dict(flat_off=range(64, 74),
        #                  flat_on=range(74, 84),
        #                  thar=range(3, 8),
        #                  sky=[29])

    band = "K"

    igr_path = IGRINSPath(utdate)

    igrins_files = IGRINSFiles(igr_path)

    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    recipe_dict = make_recipe_dict(recipe_list)

    # igrins_log = IGRINSLog(igr_path, log_today)

    obsids = recipe_dict["THAR"][0][0]

    thar_filenames = igrins_files.get_filenames(band, obsids)

    thar_path = ProductPath(igr_path, thar_filenames[0])
    thar_master_obsid = obsids[0]

    flatoff_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "flat_off.db"))
    flaton_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                       "flat_on.db"))

    basename = flaton_db.query(band, thar_master_obsid)
    flaton_path = ProductPath(igr_path, basename)
    aperture_solutions_name = flaton_path.get_secondary_path("aperture_solutions")


    aperture_solution_products = PipelineProducts.load(aperture_solutions_name)

    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 92)

    if 1:
        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]

        orders = range(len(bottomup_solutions))

        ap =  Apertures(orders, bottomup_solutions)


    if 1:
        # thar_names = [igrins_log.get_filename(band, fn) for fn \
        #               in igrins_log.log["thar"]]
        from libs.process_thar import ThAr

        thar = ThAr(thar_filenames)

        thar_products = thar.process_thar(ap)


    if 1: # match order
        from libs.process_thar import match_order_thar
        from libs.master_calib import load_thar_ref_data

        ref_date = "20140316"

        thar_ref_data = load_thar_ref_data(ref_date, band)

        new_orders = match_order_thar(thar_products, thar_ref_data)

        print thar_ref_data["orders"]
        print  new_orders

        ap =  Apertures(new_orders, bottomup_solutions)

        thar_products["orders"] = new_orders

        # order_map = ap.make_order_map()
        # slitpos_map = ap.make_slitpos_map()


    if 1:

        fn = thar.get_product_name(igr_path)
        hdu = pyfits.open(thar_filenames[0])[0]
        thar_products.save(fn, masterhdu=hdu)

    if 1:
        # measure shift of thar lines from reference spectra

        # load spec

        from libs.process_thar import reidentify_ThAr_lines
        thar_reidentified_products = reidentify_ThAr_lines(thar_products,
                                                           thar_ref_data)

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

        # bpix_mask = r["flat_bpix_mask"]
        #from matplotlib.transforms import Affine2D

        # load echellogram data

        # now fit is done.

        fig_list = check_thar_transorm(thar_products,
                                       thar_aligned_echell_products)

        from libs.qa_helper import figlist_to_pngs
        thar_figs = thar_path.get_secondary_path("thar",
                                                 "thar_dir")
        figlist_to_pngs(thar_figs, fig_list)

        thar_wvl_sol = get_wavelength_solutions(thar_aligned_echell_products,
                                                echel)

        wvlsol_name = thar_path.get_secondary_path("wvlsol_v0")
        thar_wvl_sol.save(wvlsol_name, masterhdu=hdu)

    if 1: # make amp and order falt

        orders = thar_products["orders"]
        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()


        # load flat on products
        flat_on_params_name = flaton_path.get_secondary_path("flat_on_params")

        flaton_products = PipelineProducts.load(flat_on_params_name)

        from libs.process_flat import make_order_flat, check_order_flat
        order_flat_products = make_order_flat(flaton_products,
                                              orders, order_map)

        fn = thar_path.get_secondary_path("orderflat")
        order_flat_products.save(fn, masterhdu=hdu)

        fig_list = check_order_flat(order_flat_products)

        from libs.qa_helper import figlist_to_pngs
        fn = thar_path.get_secondary_path("orderflat", "orderflat_dir")
        figlist_to_pngs(fn, fig_list)

    if 1:
        from libs.products import ProductDB
        import os
        thar_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                         "thar.db"))
        thar_db.update(band, thar_path.basename)
