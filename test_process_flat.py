import os
#import numpy as np

import libs.process_flat
reload(libs.process_flat)

from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSLog
import astropy.io.fits as pyfits

from libs.products import PipelineProducts

if __name__ == "__main__":

    utdate = "20140316"
    igr_path = IGRINSPath(utdate)

    log_20140316 = dict(flat_off=range(2, 4),
                        flat_on=range(4, 7),
                        thar=range(1, 2))


    igrins_log = IGRINSLog(igr_path, log_20140316)

    band = "K"


    flatoff_products = None
    flaton_products = None

    # INPUT
    flat_off_filenames = [igrins_log.get_filename(band, i) for i \
                          in igrins_log.log["flat_off"]]


    flat_off_name_ = flat_off_filenames[0]
    flat_off_name_ = os.path.splitext(flat_off_name_)[0] + ".flat_off_params"
    flat_off_name = igr_path.get_secondary_calib_filename(flat_off_name_)


    #flat_offs = [destriper.get_destriped(hdu.data) for hdu in hdu_list]

    if 0:
        flat_offs_hdu_list = [pyfits.open(fn)[0] for fn in flat_off_filenames]
        flat_offs = [hdu.data for hdu in flat_offs_hdu_list]


        flat = FlatOff(flat_offs)
        flatoff_products = flat.make_flatoff_bpixmap(sigma_clip1=100,
                                                     sigma_clip2=5)

        flatoff_products.save(mastername=flat_off_name,
                              masterhdu=flat_offs_hdu_list[0])


    # INPUT
    flat_on_filenames = [igrins_log.get_filename(band, i) for i \
                         in igrins_log.log["flat_on"]]

    if flatoff_products is None:
        flatoff_products = PipelineProducts.load(flat_off_name)


    flat_on_name_ = flat_on_filenames[0]
    flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".flat_on_params"
    flat_on_name = igr_path.get_secondary_calib_filename(flat_on_name_)

    if 0:

        flat_on_hdu_list = [pyfits.open(fn)[0] for fn in flat_on_filenames]
        flat_ons = [hdu.data for hdu in flat_on_hdu_list]


        flat_on = FlatOn(flat_ons)
        flaton_products = flat_on.make_flaton_deadpixmap(flatoff_products)


        flaton_products.save(mastername=flat_on_name,
                             masterhdu=flat_on_hdu_list[0])


    if flaton_products is None:
        flaton_products = PipelineProducts.load(flat_on_name)

    flat_on_name_ = flat_on_filenames[0]
    flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_centroids"
    aperture_centroids_name = igr_path.get_secondary_calib_filename(flat_on_name_)


    flat_on_name_ = flat_on_filenames[0]
    flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
    aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


    if 1:
        from libs.process_flat import trace_orders

        trace_products = trace_orders(flaton_products)

        hdu = pyfits.open(flat_on_filenames[0])[0]
        trace_products.save(mastername=aperture_centroids_name,
                            masterhdu=hdu)


        from libs.process_flat import trace_solutions
        trace_solution_products = trace_solutions(trace_products)
        trace_solution_products.save(mastername=aperture_solutions_name,
                                     masterhdu=hdu)


    if 0:
        from libs.process_flat import check_trace_order
        fig = figure()
        check_trace_order(trace_products, fig)

    if 0:
        flat_normed = flaton_products["flat_normed"]
        bottom_up_centroids = trace_solution_products["bottom_up_centroids"]
        bottom_up_solutions_ = trace_solution_products["bottom_up_solutions"]
        bottom_up_solutions = []
        for b, d in bottom_up_solutions_:
            import numpy.polynomial as P
            assert b[0] == "poly"
            assert d[0] == "poly"
            bp = P.Polynomial(b[1])
            dp = P.Polynomial(d[1])
            bottom_up_solutions.append((bp, dp))

        from libs.trace_flat import plot_solutions
        plot_solutions(flat_normed,
                       bottom_up_centroids,
                       bottom_up_solutions)
