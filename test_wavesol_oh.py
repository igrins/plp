import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
#import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

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

    band = "H"
    igr_path = IGRINSPath(utdate)

    igrins_files = IGRINSFiles(igr_path)

    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    recipe_dict = make_recipe_dict(recipe_list)

    # igrins_log = IGRINSLog(igr_path, log_today)

    obsids = recipe_dict["SKY"][0][0]

    sky_filenames = igrins_files.get_filenames(band, obsids)

    sky_path = ProductPath(igr_path, sky_filenames[0])
    sky_master_obsid = obsids[0]

    flatoff_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "flat_off.db"))
    flaton_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                       "flat_on.db"))
    thar_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                     "thar.db"))

    if 1: # load aperture product
        basename = flaton_db.query(band, sky_master_obsid)
        flaton_path = ProductPath(igr_path, basename)
        aperture_solutions_name = flaton_path.get_secondary_path("aperture_solutions")


        aperture_solution_products = PipelineProducts.load(aperture_solutions_name)


        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        basename = thar_db.query(band, sky_master_obsid)

        thar_path = ProductPath(igr_path, basename)
        thar_spec = thar_path.get_secondary_path("median_spectra")


        thar_spec_products = PipelineProducts.load(thar_spec)

        ap =  Apertures(thar_spec_products["orders"], bottomup_solutions)


    if 1: #

        from libs.process_thar import get_1d_median_specs
        raw_spec_product = get_1d_median_specs(sky_filenames, ap)


        # sky_master_fn_ = os.path.splitext(os.path.basename(sky_names[0]))[0]
        # sky_master_fn = igr_path.get_secondary_calib_filename(sky_master_fn_)

        fn = sky_path.get_secondary_path("raw_spec")
        import astropy.io.fits as pyfits
        raw_spec_product.save(fn,
                              masterhdu=pyfits.open(sky_filenames[0])[0])


        from libs.master_calib import load_sky_ref_data

        ref_date = "20140316"

        sky_ref_data = load_sky_ref_data(ref_date, band)


    if 1: # initial wavelength solution

        # this need to be fixed
        # thar_db.query(sky_master_obsid)
        # json_name_ = "SDC%s_%s_0003.median_spectra.wvlsol" % (band,
        #                                                      igrins_log.date)

        json_name = thar_path.get_secondary_path("wvlsol_v0")
        #json_name = igr_path.get_secondary_calib_filename(json_name_)
        thar_wvl_sol = PipelineProducts.load(json_name)




    if 1:
        # Now we fit with gaussian profile for matched positions.

        ohline_indices = sky_ref_data["ohline_indices"]
        ohlines_db = sky_ref_data["ohlines_db"]

        wvl_solutions = thar_wvl_sol["wvl_sol"]
        orders_w_solutions = thar_wvl_sol["orders"]
        _ = dict(zip(raw_spec_product["orders"],
                     raw_spec_product["specs"]))
        s_list = [_[o]for o in orders_w_solutions]


        from libs.reidentify_ohlines import fit_ohlines
        ref_pixel_list, reidentified_lines = \
                        fit_ohlines(ohlines_db, ohline_indices,
                                    orders_w_solutions,
                                    wvl_solutions, s_list)


        # from scipy.interpolate import interp1d
        # from reidentify import reidentify_lines_all

        x = np.arange(2048)


        # line_indices_list = [ref_ohline_indices[str(o)] for o in igrins_orders[band]]




        ###### not fit identified lines

        from libs.ecfit import get_ordered_line_data, fit_2dspec, check_fit

        # d_x_wvl = {}
        # for order, z in echel.zdata.items():
        #     xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
        #     x_T = xy_T[:,0]
        #     d_x_wvl[order]=(x_T, z.wvl)

        reidentified_lines_map = dict(zip(orders_w_solutions,
                                          reidentified_lines))

        if band == "K":
            json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
            r = json.load(open(json_name))
            for i, s in r.items():
                ss = reidentified_lines_map[int(i)]
                ss0 = np.concatenate([ss[0], s["pixel"]])
                ss1 = np.concatenate([ss[1], s["wavelength"]])
                reidentified_lines_map[int(i)] = (ss0, ss1)

        xl, yl, zl = get_ordered_line_data(reidentified_lines_map)
        # xl : pixel
        # yl : order
        # zl : wvl * order

        x_domain = [0, 2047]
        y_domain = [orders_w_solutions[0]-2, orders_w_solutions[-1]+2]
        x_degree, y_degree = 4, 3
        #x_degree, y_degree = 3, 2
        p, m = fit_2dspec(xl, yl, zl, x_degree=x_degree, y_degree=y_degree,
                          x_domain=x_domain, y_domain=y_domain)


        # derive wavelengths.
        xx = np.arange(2048)
        wvl_sol = []
        for o in orders_w_solutions:
            oo = np.empty_like(xx)
            oo.fill(o)
            wvl = p(xx, oo) / o
            wvl_sol.append(list(wvl))

        oh_sol_products = PipelineProducts("Wavelength solution based on ohlines",
                                           orders=orders_w_solutions,
                                           wvl_sol=wvl_sol)

        fn = sky_path.get_secondary_path("wvlsol_v1")
        oh_sol_products.save(fn)

        if 1: # save as WAT fits header
            xx = np.arange(0, 2048)
            xx_plus1 = np.arange(1, 2048+1)

            from astropy.modeling import models, fitting

            # naive solution to convert 2d to 1d chebyshev
            p1d_list = []
            for o in orders_w_solutions:
                oo = np.empty_like(xx)
                oo.fill(o)
                wvl = p(xx, oo) / o * 1.e4 # um to angstrom

                p_init1d = models.Chebyshev1D(domain=[1, 2048],
                                              degree=p.x_degree)
                fit_p1d = fitting.LinearLSQFitter()
                p1d = fit_p1d(p_init1d, xx_plus1, wvl)
                p1d_list.append(p1d)

            from libs.iraf_helper import get_wat_spec
            wat_list = get_wat_spec(orders_w_solutions, p1d_list)

        cards = [pyfits.Card.fromstring(l.strip()) \
                 for l in open("echell_2dspec.header")]

        wat = "wtype=multispec " + " ".join(wat_list)
        char_per_line = 68
        num_line, remainder = divmod(len(wat), char_per_line)
        for i in range(num_line):
            k = "WAT2_%03d" % (i+1,)
            v = wat[char_per_line*i:char_per_line*(i+1)]
            #print k, v
            c = pyfits.Card(k, v)
            cards.append(c)
        if remainder > 0:
            i = num_line
            k = "WAT2_%03d" % (i,)
            v = wat[char_per_line*i:]
            #print k, v
            c = pyfits.Card(k, v)
            cards.append(c)

        if 1:
            header = pyfits.Header(cards)
            hdu = pyfits.PrimaryHDU(header=header,
                                    data=np.array([]).reshape((0,0)))
            fn = sky_path.get_secondary_path("wvlsol_v1.fits")
            hdu.writeto(fn, clobber=True)

        if 0:
            # plot all spectra
            for w, s in zip(wvl_sol, s_list):
                plot(w, s)


        # filter out the line indices not well fit by the surface


        keys = reidentified_lines_map.keys()
        di_list = [len(reidentified_lines_map[k_][0]) for k_ in keys]

        endi_list = np.add.accumulate(di_list)

        filter_mask = [m[endi-di:endi] for di, endi in zip(di_list, endi_list)]
        #from itertools import compress
        # _ = [list(compress(indices, mm)) for indices, mm \
        #      in zip(line_indices_list, filter_mask)]
        # line_indices_list_filtered = _

        reidentified_lines_ = [reidentified_lines_map[k_] for k_ in keys]
        _ = [(v_[0][mm], v_[1][mm]) for v_, mm \
             in zip(reidentified_lines_, filter_mask)]

        reidentified_lines_map_filtered = dict(zip(orders_w_solutions, _))


        if 1:
            from matplotlib.figure import Figure

            fig1 = Figure(figsize=(12, 7))
            check_fit(fig1, xl, yl, zl, p,
                      orders_w_solutions,
                      reidentified_lines_map)
            fig1.tight_layout()

            fig2 = Figure(figsize=(12, 7))
            check_fit(fig2, xl[m], yl[m], zl[m], p,
                      orders_w_solutions,
                      reidentified_lines_map_filtered)
            fig2.tight_layout()

        from libs.qa_helper import figlist_to_pngs
        fn = sky_path.get_secondary_path("oh_fit2d", "oh_fit2d_dir")
        figlist_to_pngs(fn, [fig1, fig2])

    if 1:
        from libs.products import ProductDB
        import os
        sky_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "sky.db"))
        sky_db.update(band, sky_path.basename)
