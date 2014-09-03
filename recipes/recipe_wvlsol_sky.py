import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
#import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

from libs.recipe_base import RecipeBase

class RecipeWvlsolSky(RecipeBase):
    RECIPE_NAME = "SKY"

    def run_selected_bands(self, utdate, selected, bands):
        for s in selected:
            obsids = s[0]
            print obsids
            # frametypes = s[1]

            for band in bands:
                process_wvlsol_band(utdate, self.refdate, band, obsids,
                                    self.config)

def wvlsol_sky(utdate, bands="HK",
               starting_obsids=None, config_file="recipe.config"):

    RecipeWvlsolSky()(utdate, bands,
                      starting_obsids, config_file)

# def wvlsol_sky(utdate, refdate="20140316", bands="HK",
#                starting_obsids=None):

#     if not bands in ["H", "K", "HK"]:
#         raise ValueError("bands must be one of 'H', 'K' or 'HK'")

#     fn = "%s.recipes" % utdate
#     from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
#     recipe = Recipes(fn)

#     if starting_obsids is not None:
#         starting_obsids = map(int, starting_obsids.split(","))

#     selected = recipe.select("SKY", starting_obsids)

#     for s in selected:
#         obsids = s[0]

#         for band in bands:
#             process_wvlsol_band(utdate, refdate, band, obsids)


def load_aperture(igr_storage, band, master_obsid, flaton_db, thar_db):
    from libs.process_flat import FLATCENTROID_SOL_JSON_DESC

    flaton_basename = flaton_db.query(band, master_obsid)
    thar_basename = thar_db.query(band, master_obsid)

    aperture_solution_products = igr_storage.load([FLATCENTROID_SOL_JSON_DESC], flaton_basename)


    bottomup_solutions = aperture_solution_products[FLATCENTROID_SOL_JSON_DESC]["bottom_up_solutions"]


    thar_basename = thar_db.query(band, master_obsid)

    # thar_path = ProductPath(igr_path, basename)
    from libs.process_thar import ONED_SPEC_JSON
    thar_spec_products = igr_storage.load([ONED_SPEC_JSON],
                                          thar_basename)

    ap =  Apertures(thar_spec_products[ONED_SPEC_JSON]["orders"],
                    bottomup_solutions)

    return ap


def process_wvlsol_band(utdate, refdate, band, obsids, config):

    from libs.products import ProductDB, PipelineStorage

    igr_path = IGRINSPath(config, utdate)

    igr_storage = PipelineStorage(igr_path)

    sky_filenames = igr_path.get_filenames(band, obsids)


    master_obsid = obsids[0]


    flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "flat_on.db",
                                                        )
    flaton_db = ProductDB(flaton_db_name)

    #flaton_basename = flaton_db.query(band, master_obsid)

    thar_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "thar.db",
                                                        )
    thar_db = ProductDB(thar_db_name)

    #thar_basename = thar_db.query(band, master_obsid)




    # flaton_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
    #                                    "flat_on.db"))
    # thar_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
    #                                  "thar.db"))

    ap = load_aperture(igr_storage, band, master_obsid,
                       flaton_db, thar_db)


    if 1: #

        from libs.process_thar import get_1d_median_specs
        raw_spec_product = get_1d_median_specs(sky_filenames, ap)


        # sky_master_fn_ = os.path.splitext(os.path.basename(sky_names[0]))[0]
        # sky_master_fn = igr_path.get_secondary_calib_filename(sky_master_fn_)

        import astropy.io.fits as pyfits
        masterhdu = pyfits.open(sky_filenames[0])[0]

        igr_storage.store(raw_spec_product,
                          mastername=sky_filenames[0],
                          masterhdu=masterhdu)

        # fn = sky_path.get_secondary_path("raw_spec")
        # raw_spec_product.save(fn,
        #                       masterhdu=masterhdu)


        from libs.master_calib import load_sky_ref_data

        # ref_date = "20140316"

        refdate = config.get_value("REFDATE", utdate)
        sky_ref_data = load_sky_ref_data(refdate, band)


    if 1: # initial wavelength solution

        # this need to be fixed
        # thar_db.query(sky_master_obsid)
        # json_name_ = "SDC%s_%s_0003.median_spectra.wvlsol" % (band,
        #                                                      igrins_log.date)

        from libs.process_thar import THAR_WVLSOL_JSON
        thar_basename = thar_db.query(band, master_obsid)
        thar_wvl_sol = igr_storage.load([THAR_WVLSOL_JSON],
                                        thar_basename)[THAR_WVLSOL_JSON]
        #print thar_wvl_sol.keys()
        #["wvl_sol"]

        #json_name = thar_path.get_secondary_path("wvlsol_v0")
        #json_name = igr_path.get_secondary_calib_filename(json_name_)
        #thar_wvl_sol = PipelineProducts.load(json_name)




    if 1:
        # Now we fit with gaussian profile for matched positions.

        ohline_indices = sky_ref_data["ohline_indices"]
        ohlines_db = sky_ref_data["ohlines_db"]

        wvl_solutions = thar_wvl_sol["wvl_sol"]

        if 0: # it would be better to iteratively refit the solution
            fn = sky_path.get_secondary_path("wvlsol_v1")
            p = PipelineProducts.load(fn)
            wvl_solutionv = p["wvl_sol"]

        orders_w_solutions_ = thar_wvl_sol["orders"]
        from libs.process_thar import ONED_SPEC_JSON
        orders_w_solutions = [o for o in orders_w_solutions_ if o in raw_spec_product[ONED_SPEC_JSON]["orders"]]
        _ = dict(zip(raw_spec_product[ONED_SPEC_JSON]["orders"],
                     raw_spec_product[ONED_SPEC_JSON]["specs"]))
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
            import libs.master_calib as master_calib
            fn = "hitran_bootstrap_K_%s.json" % refdate
            bootstrap_name = master_calib.get_master_calib_abspath(fn)
            import json
            bootstrap = json.load(open(bootstrap_name))

            import libs.hitran as hitran
            r, ref_pixel_list = hitran.reidentify(wvl_solutions, s_list, bootstrap)
            # json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
            # r = json.load(open(json_name))
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

        oh_sol_products = PipelineProducts("Wavelength solution based on ohlines")
        #from libs.process_thar import ONED_SPEC_JSON
        from libs.products import PipelineDict
        SKY_WVLSOL_JSON = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.json")
        oh_sol_products.add(SKY_WVLSOL_JSON,
                            PipelineDict(orders=orders_w_solutions,
                                         wvl_sol=wvl_sol))

    if 1:

        if 1: # save as WAT fits header
            xx = np.arange(0, 2048)
            xx_plus1 = np.arange(1, 2048+1)

            from astropy.modeling import models, fitting

            # We convert 2d chebyshev solution to a seriese of 1d
            # chebyshev.  For now, use naive (and inefficient)
            # approach of refitting the solution with 1d. Should be
            # reimplemented.

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

        from libs.iraf_helper import get_wat_spec, default_header_str
        wat_list = get_wat_spec(orders_w_solutions, p1d_list)

        # cards = [pyfits.Card.fromstring(l.strip()) \
        #          for l in open("echell_2dspec.header")]
        cards = [pyfits.Card.fromstring(l.strip()) \
                 for l in default_header_str]

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
            i = num_line+1
            k = "WAT2_%03d" % (i,)
            v = wat[char_per_line*i:]
            #print k, v
            c = pyfits.Card(k, v)
            cards.append(c)

        if 1:
            # save fits with empty header

            header = pyfits.Header(cards)
            hdu = pyfits.PrimaryHDU(header=header,
                                    data=np.array([]).reshape((0,0)))

            SKY_WVLSOL_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.fits")
            from libs.products import PipelineImage
            oh_sol_products.add(SKY_WVLSOL_FITS_DESC,
                                PipelineImage([],
                                              np.array([]).reshape((0,0))))

            igr_storage.store(oh_sol_products,
                              mastername=sky_filenames[0],
                              masterhdu=hdu)

            #fn = sky_path.get_secondary_path("wvlsol_v1.fits")
            #hdu.writeto(fn, clobber=True)

        if 0:
            # plot all spectra
            for w, s in zip(wvl_sol, s_list):
                plot(w, s)


    if 1:
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
        sky_figs = igr_path.get_section_filename_base("QA_PATH",
                                                       "oh_fit2d",
                                                       "oh_fit2d_dir")
        figlist_to_pngs(sky_figs, [fig1, fig2])

    if 1:
        from libs.products import ProductDB
        sky_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                          "sky.db",
                                                          )

        sky_db = ProductDB(sky_db_name)
        basename = os.path.splitext(os.path.basename(sky_filenames[0]))[0]
        sky_db.update(band, basename)



if __name__ == "__main__":
    import sys

    utdate = sys.argv[1]
    bands = "HK"
    starting_obsids = None

    if len(sys.argv) >= 3:
        bands = sys.argv[2]

    if len(sys.argv) >= 4:
        starting_obsids = sys.argv[3]

    wvlsol_sky(utdate, refdate="20140316", bands=bands,
               starting_obsids=starting_obsids)


# if __name__ == "__main__":

#     from libs.recipes import load_recipe_list, make_recipe_dict
#     from libs.products import PipelineProducts, ProductPath, ProductDB

#     if 0:
#         utdate = "20140316"
#         # log_today = dict(flat_off=range(2, 4),
#         #                  flat_on=range(4, 7),
#         #                  thar=range(1, 2))
#     elif 1:
#         utdate = "20140713"
#         # log_today = dict(flat_off=range(64, 74),
#         #                  flat_on=range(74, 84),
#         #                  thar=range(3, 8),
#         #                  sky=[29])

#     band = "H"
#     igr_path = IGRINSPath(utdate)

#     igrins_files = IGRINSFiles(igr_path)

#     fn = "%s.recipes" % utdate
#     recipe_list = load_recipe_list(fn)
#     recipe_dict = make_recipe_dict(recipe_list)

#     # igrins_log = IGRINSLog(igr_path, log_today)

#     obsids = recipe_dict["SKY"][0][0]
