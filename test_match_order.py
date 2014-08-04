import os
#import numpy as np

from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSLog
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

if __name__ == "__main__":

    utdate = "20140316"
    igr_path = IGRINSPath(utdate)

    log_20140316 = dict(flat_off=range(2, 4),
                        flat_on=range(4, 7),
                        thar=range(1, 2))


    igrins_log = IGRINSLog(igr_path, log_20140316)

    band = "K"


    flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
    flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
    aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


    aperture_solution_products = PipelineProducts.load(aperture_solutions_name)

    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 92)

    if 1:
        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]

        orders = range(len(bottomup_solutions))

        ap =  Apertures(orders, bottomup_solutions)


    if 1:
        thar_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["thar"]]
        from libs.process_thar import ThAr

        thar = ThAr(thar_names)

        thar_products = thar.process_thar(ap)


    if 1: # match order
        from libs.process_thar import load_thar_ref_data, match_order_thar
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
        hdu = pyfits.open(thar_names[0])[0]
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
        figlist_to_pngs(fn, fig_list)

        thar_wvl_sol = get_wavelength_solutions(thar_aligned_echell_products,
                                                echel)
        thar_wvl_sol.save(fn+".wvlsol", masterhdu=hdu)

    if 1: # make amp and order falt

        orders = thar_products["orders"]
        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()


        # load flat on products
        flat_on_filenames = [igrins_log.get_filename(band, i) for i \
                             in igrins_log.log["flat_on"]]
        flat_on_name_ = flat_on_filenames[0]
        flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".flat_on_params"
        flat_on_name = igr_path.get_secondary_calib_filename(flat_on_name_)

        flaton_products = PipelineProducts.load(flat_on_name)

        from libs.process_flat import make_order_flat, check_order_flat
        order_flat_products = make_order_flat(flaton_products,
                                              orders, order_map)

        fn = thar.get_product_name(igr_path)+".orderflat"
        order_flat_products.save(fn, masterhdu=hdu)

        fig_list = check_order_flat(order_flat_products)

        fn = thar.get_product_name(igr_path)+".orderflat"
        from libs.qa_helper import figlist_to_pngs
        figlist_to_pngs(fn, fig_list)
