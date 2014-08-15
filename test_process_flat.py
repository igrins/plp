import os
#import numpy as np

from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
import astropy.io.fits as pyfits


if __name__ == "__main__":

    from libs.recipes import load_recipe_list, make_recipe_dict
    from libs.products import PipelineProducts, ProductPath

    if 0:
        utdate = "20140316"
        # log_today = dict(flat_off=range(2, 4),
        #                  flat_on=range(4, 7),
        #                  thar=range(1, 2))
    elif 1:
        utdate = "20140711"
        # log_today = dict(flat_off=range(64, 74),
        #                  flat_on=range(74, 84),
        #                  thar=range(3, 8),
        #                  sky=[29])

    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    recipe_dict = make_recipe_dict(recipe_list)

    igr_path = IGRINSPath(utdate)

    igrins_files = IGRINSFiles(igr_path)
    #igrins_log = IGRINSLog(igr_path, log_today)

    band = "H"


    flatoff_products = None
    flaton_products = None

    # INPUT
    obsids = recipe_dict["FLAT"][0][0]
    frametypes = recipe_dict["FLAT"][0][1]

    obsids_off = [obsid for obsid, frametype in zip(obsids, frametypes) if frametype == "OFF"]
    obsids_on = [obsid for obsid, frametype in zip(obsids, frametypes) if frametype == "ON"]

    flat_off_filenames = igrins_files.get_filenames(band, obsids_off)

    flatoff_path = ProductPath(igr_path, flat_off_filenames[0])
    #flat_off_name_ = flat_off_filenames[0]
    #flat_off_name_ = os.path.splitext(flat_off_name_)[0] + ".flat_off_params"
    flat_off_name = flatoff_path.get_secondary_path("flat_off_params")


    #flat_offs = [destriper.get_destriped(hdu.data) for hdu in hdu_list]

    if 1:
        flat_offs_hdu_list = [pyfits.open(fn_)[0] for fn_ in flat_off_filenames]
        flat_offs = [hdu.data for hdu in flat_offs_hdu_list]


        flat = FlatOff(flat_offs)
        flatoff_products = flat.make_flatoff_hotpixmap(sigma_clip1=100,
                                                       sigma_clip2=5)

        flatoff_products.save(mastername=flat_off_name,
                              masterhdu=flat_offs_hdu_list[0])


    # INPUT

    flat_on_filenames = igrins_files.get_filenames(band, obsids_on)

    # flat_on_filenames = [igrins_log.get_filename(band, i) for i \
    #                      in igrins_log.log["flat_on"]]

    if flatoff_products is None:
        flatoff_products = PipelineProducts.load(flat_off_name)


    flaton_path = ProductPath(igr_path, flat_on_filenames[0])
    flat_on_name = flaton_path.get_secondary_path("flat_on_params")
    # flat_on_name_ = flat_on_filenames[0]
    # flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".flat_on_params"
    # flat_on_name = igr_path.get_secondary_calib_filename(flat_on_name_)

    if 1:

        flat_on_hdu_list = [pyfits.open(fn_)[0] for fn_ in flat_on_filenames]
        flat_ons = [hdu.data for hdu in flat_on_hdu_list]


        flat_on = FlatOn(flat_ons)
        flaton_products = flat_on.make_flaton_deadpixmap(flatoff_products)


        flaton_products.save(mastername=flat_on_name,
                             masterhdu=flat_on_hdu_list[0])


    if flaton_products is None:
        flaton_products = PipelineProducts.load(flat_on_name)

    flat_on_name_ = flat_on_filenames[0]
    flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_centroids"

    aperture_centroids_name = flaton_path.get_secondary_path("aperture_centroids")
    # aperture_centroids_name = igr_path.get_secondary_calib_filename(flat_on_name_)


    # flat_on_name_ = flat_on_filenames[0]
    # flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
    aperture_solutions_name = flaton_path.get_secondary_path("aperture_solutions")
    #igr_path.get_secondary_calib_filename(flat_on_name_)


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


    if 1:
        from libs.process_flat import check_trace_order
        from matplotlib.figure import Figure
        fig1 = Figure()
        check_trace_order(trace_products, fig1)

    if 1:
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
        fig2, fig3 = plot_solutions(flat_normed,
                                    bottom_up_centroids,
                                    bottom_up_solutions)

    if 1:
        from libs.qa_helper import figlist_to_pngs
        aperture_figs = flaton_path.get_secondary_path("aperture",
                                                       "aperture_dir")

        figlist_to_pngs(aperture_figs, [fig1, fig2, fig3])

    if 1:
        from libs.products import ProductDB
        import os
        flatoff_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                            "flat_off.db"))
        flatoff_db.update(band, flatoff_path.basename)
        flaton_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                           "flat_on.db"))
        flaton_db.update(band, flaton_path.basename)
