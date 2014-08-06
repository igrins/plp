import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSLog
#import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

if __name__ == "__main__":

    utdate = "20140316"
    igr_path = IGRINSPath(utdate)

    log_20140316 = dict(flat_off=range(2, 4),
                        flat_on=range(4, 7),
                        thar=range(1, 2),
                        sky=[25])


    igrins_log = IGRINSLog(igr_path, log_20140316)

    band = "H"


    if 1: # load aperture product
        flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
        flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
        aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


        aperture_solution_products = PipelineProducts.load(aperture_solutions_name)

        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        from libs.process_thar import ThAr

        thar_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["thar"]]
        thar = ThAr(thar_names)

        fn = thar.get_product_name(igr_path)

        thar_products = PipelineProducts.load(fn)

        ap =  Apertures(thar_products["orders"], bottomup_solutions)


    if 1: #

        sky_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["sky"]]
        from libs.process_thar import get_1d_median_specs
        raw_spec_product = get_1d_median_specs(sky_names, ap)


        ref_utdate = "20140316"


        from libs.master_calib import load_sky_ref_data

        ref_date = "20140316"

        sky_ref_data = load_sky_ref_data(ref_date, band)


    if 1: # initial wavelength solution

        # this need to be fixed
        json_name_ = "SDC%s_%s_0001.median_spectra.wvlsol" % (band,
                                                             igrins_log.date)

        json_name = igr_path.get_secondary_calib_filename(json_name_)
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

        fn = thar.get_product_name(igr_path)+".oh_wvlsol"
        oh_sol_products.save(fn)

        if 0:
            # plot all spectra
            for w, s in zip(wvl_sol, s_list):
                plot(w, s)


        # filter out the line indices not well fit by the surface


        keys = reidentified_lines_map.keys()
        di_list = [len(reidentified_lines_map[k][0]) for k in keys]

        endi_list = np.add.accumulate(di_list)

        filter_mask = [m[endi-di:endi] for di, endi in zip(di_list, endi_list)]
        #from itertools import compress
        # _ = [list(compress(indices, mm)) for indices, mm \
        #      in zip(line_indices_list, filter_mask)]
        # line_indices_list_filtered = _

        reidentified_lines_ = [reidentified_lines_map[k] for k in keys]
        _ = [(v[0][mm], v[1][mm]) for v, mm \
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

        fn = thar.get_product_name(igr_path)+".oh_fit2d"
        from libs.qa_helper import figlist_to_pngs
        figlist_to_pngs(fn, [fig1, fig2])
