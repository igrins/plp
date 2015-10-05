import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn

from libs.path_info import IGRINSPath
#import libs.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

from libs.recipe_base import RecipeBase

class RecipeSkyWvlsol(RecipeBase):
    #RECIPE_NAME = "SKY_WVLSOL"
    RECIPE_NAME = "SKY"

    def run_selected_bands(self, utdate, selected, bands):
        for s in selected:
            obsids = s[0]
            print obsids
            # frametypes = s[1]

            p = ProcessSkyBand(utdate, self.refdate,
                               self.config, debug_output=False)

            for band in bands:
                p.process(band, obsids)
                #process_wvlsol_band(utdate, self.refdate, band, obsids,
                #                    self.config)

                p.process_distortion_sky_band(band, obsids)
                # process_distortion_sky_band(utdate, self.refdate,
                #                             band, obsids,
                #                             self.config)

def sky_wvlsol(utdate, bands="HK",
               starting_obsids=None, config_file="recipe.config"):

    RecipeSkyWvlsol()(utdate, bands,
                      starting_obsids, config_file)

def wvlsol_sky(utdate, bands="HK",
               starting_obsids=None, config_file="recipe.config"):

    RecipeSkyWvlsol()(utdate, bands,
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

def get_orders_for_flat(extractor, band):

    igr_storage = extractor.igr_storage

    thar_basename = extractor.basenames["thar"]

    from libs.storage_descriptions import ONED_SPEC_JSON_DESC
    orders = igr_storage.load1(ONED_SPEC_JSON_DESC,
                                          thar_basename)["orders"]

    return orders

def get_bottomup_solutions(extractor, band):
    from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC

    igr_storage = extractor.igr_storage
    #flaton_basename = extractor.db["flat_on"].query(band, master_obsid)
    flaton_basename = extractor.basenames["flat_on"]

    aperture_solution_products = igr_storage.load([FLATCENTROID_SOL_JSON_DESC], flaton_basename)

    bottomup_solutions = aperture_solution_products[FLATCENTROID_SOL_JSON_DESC]["bottom_up_solutions"]

    return bottomup_solutions

def load_aperture(extractor, band, master_obsid):
    """
    for orders that flats are extracted.
    """
    orders = get_orders_for_flat(extractor, band)
    bottomup_solutions = get_bottomup_solutions(extractor, band)

    ap =  Apertures(orders, bottomup_solutions)

    return ap


def load_aperture_wvlsol(extractor, band):
    """
    for orders that wvlsols are derived.
    """
    #orders, orders_w_solutions):

    orders = get_orders_for_flat(extractor, band)
    bottomup_solutions = get_bottomup_solutions(extractor, band)

    _o_s = dict(zip(orders, bottomup_solutions))

    orders_w_solutions = extractor.orders_w_solutions
    ap =  Apertures(orders_w_solutions,
                    [_o_s[o] for o in orders_w_solutions])

    # _o_s = dict(zip(orders, bottomup_solutions))
    # ap =  Apertures(orders,
    #                 [_o_s[o] for o in orders])
    # # ap =  Apertures(orders,
    # #                 bottomup_solutions)

    return ap




from libs.products import ProductDB, PipelineStorage

class ProcessSkyBand(object):
    def __init__(self, utdate, refdate, config,
                 debug_output=False,
                 ):
        """
        cr_rejection_thresh : pixels that deviate significantly from the profile are excluded.
        """
        self.utdate = utdate
        self.refdate = refdate
        self.config = config

        self.igr_path = IGRINSPath(config, utdate)

        self.igr_storage = PipelineStorage(self.igr_path)

        self.debug_output = debug_output

    def get_sky_spectra(self, extractor, ap, band, master_obsid):
        from libs.process_thar import get_1d_median_specs
        sky_filenames = extractor.obj_filenames
        raw_spec_product = get_1d_median_specs(sky_filenames, ap)


        # sky_master_fn_ = os.path.splitext(os.path.basename(sky_names[0]))[0]
        # sky_master_fn = igr_path.get_secondary_calib_filename(sky_master_fn_)

        import libs.fits as pyfits
        masterhdu = pyfits.open(sky_filenames[0])[0]

        igr_storage = extractor.igr_storage
        igr_storage.store(raw_spec_product,
                          mastername=sky_filenames[0],
                          masterhdu=masterhdu)

        # fn = sky_path.get_secondary_path("raw_spec")
        # raw_spec_product.save(fn,
        #                       masterhdu=masterhdu)




        # initial wavelength solution

        # this need to be fixed
        # thar_db.query(sky_master_obsid)
        # json_name_ = "SDC%s_%s_0003.median_spectra.wvlsol" % (band,
        #                                                      igrins_log.date)

        #from libs.storage_descriptions import THAR_WVLSOL_JSON_DESC
        from libs.storage_descriptions import WVLSOL_V0_JSON_DESC
        thar_basename = extractor.db["thar"].query(band, master_obsid)
        thar_wvl_sol = igr_storage.load([WVLSOL_V0_JSON_DESC],
                                        thar_basename)[WVLSOL_V0_JSON_DESC]
        #print thar_wvl_sol.keys()
        #["wvl_sol"]

        #json_name = thar_path.get_secondary_path("wvlsol_v0")
        #json_name = igr_path.get_secondary_calib_filename(json_name_)
        #thar_wvl_sol = PipelineProducts.load(json_name)


        if 0: # it would be better to iteratively refit the solution
            fn = sky_path.get_secondary_path("wvlsol_v1")
            p = PipelineProducts.load(fn)
            wvl_solutionv = p["wvl_sol"]

        orders_w_solutions_ = thar_wvl_sol["orders"]
        from libs.storage_descriptions import ONED_SPEC_JSON_DESC
        orders_w_solutions = [o for o in orders_w_solutions_ if o in raw_spec_product[ONED_SPEC_JSON_DESC]["orders"]]
        _ = dict(zip(raw_spec_product[ONED_SPEC_JSON_DESC]["orders"],
                     raw_spec_product[ONED_SPEC_JSON_DESC]["specs"]))
        s_list = [_[o]for o in orders_w_solutions]

        wvl_solutions = thar_wvl_sol["wvl_sol"]

        return orders_w_solutions, wvl_solutions, s_list

    def process(self, band, obsids):

        from recipe_extract_base import RecipeExtractPR

        extractor = RecipeExtractPR(self.utdate, band,
                                    obsids,
                                    self.config,
                                    load_a0v_db=False)

        self.save_db(extractor, band)

        master_obsid = obsids[0]
        ap = load_aperture(extractor, band, master_obsid)

        _ = self.get_sky_spectra(extractor, ap, band, master_obsid)
        orders_w_solutions, wvl_solutions, s_list = _

        fitter = SkyFitter(self.refdate)
        _ = fitter.fit(band, orders_w_solutions, wvl_solutions, s_list)

        # _ = fit_oh_spectra(self.refdate, band,
        #                    orders_w_solutions,
        #                    wvl_solutions, s_list,
        #                    )


        orders_w_solutions, wvl_sol, reidentified_lines_map, p, m = _

        # save
        #self.save_oh_sol_products(orders_w_solutions, wvl_sol)

        self.save_wavelength_sol(extractor,
                                 orders_w_solutions, wvl_sol, p)

        self.save_qa(extractor, orders_w_solutions,
                     reidentified_lines_map, p, m)



    def save_wavelength_sol(self, extractor,
                            orders_w_solutions, wvl_sol, p):

        oh_sol_products = PipelineProducts("Wavelength solution based on ohlines")
        #from libs.process_thar import ONED_SPEC_JSON
        from libs.products import PipelineDict
        from libs.storage_descriptions import SKY_WVLSOL_JSON_DESC
        oh_sol_products.add(SKY_WVLSOL_JSON_DESC,
                            PipelineDict(orders=orders_w_solutions,
                                         wvl_sol=wvl_sol))

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
        import libs.fits as pyfits
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
            i = num_line
            k = "WAT2_%03d" % (i+1,)
            v = wat[char_per_line*i:]
            #print k, v
            c = pyfits.Card(k, v)
            cards.append(c)

        if 1:
            # save fits with empty header

            header = pyfits.Header(cards)
            hdu = pyfits.PrimaryHDU(header=header,
                                    data=np.array([]).reshape((0,0)))

            from libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
            from libs.products import PipelineImageBase
            oh_sol_products.add(SKY_WVLSOL_FITS_DESC,
                                PipelineImageBase([],
                                              np.array(wvl_sol)))

            igr_storage = extractor.igr_storage
            sky_filenames = extractor.obj_filenames

            igr_storage.store(oh_sol_products,
                              mastername=sky_filenames[0],
                              masterhdu=hdu)

            #fn = sky_path.get_secondary_path("wvlsol_v1.fits")
            #hdu.writeto(fn, clobber=True)

        if 0:
            # plot all spectra
            for w, s in zip(wvl_sol, s_list):
                plot(w, s)


    def save_qa(self, extractor, orders_w_solutions,
                reidentified_lines_map, p, m):
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
            from libs.ecfit import get_ordered_line_data, check_fit

            xl, yl, zl = get_ordered_line_data(reidentified_lines_map)

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
        igr_path = extractor.igr_path
        sky_basename = extractor.tgt_basename
        sky_figs = igr_path.get_section_filename_base("QA_PATH",
                                                      "oh_fit2d",
                                                      "oh_fit2d_"+sky_basename)
        figlist_to_pngs(sky_figs, [fig1, fig2])


    def save_db(self, extractor, band):
        # from libs.products import ProductDB
        # igr_path = extractor.pr.igr_path
        # sky_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
        #                                                   "sky.db",
        #                                                   )

        # sky_db = ProductDB(sky_db_name)

        sky_db = extractor.db["sky"]
        sky_db.update(band, extractor.tgt_basename)


        # thar_db = ProductDB(thar_db_name)
        # # os.path.join(igr_path.secondary_calib_path,
        # #                                  "thar.db"))
        # thar_db.update(band, basename)



    def load_oh_reference_data(self, band):
        from libs.master_calib import load_sky_ref_data

        #ref_utdate = self.config.get_value("REFDATE", self.utdate)
        refdate = self.refdate

        sky_ref_data = load_sky_ref_data(refdate, band)


        ohlines_db = sky_ref_data["ohlines_db"]
        ref_ohline_indices = sky_ref_data["ohline_indices"]

        return ohlines_db, ref_ohline_indices

    def get_slices(self, extractor, ap, n_slice_one_direction):
        n_slice = n_slice_one_direction*2 + 1
        i_center = n_slice_one_direction
        slit_slice = np.linspace(0., 1., n_slice+1)

        slice_center = (slit_slice[i_center], slit_slice[i_center+1])
        slice_up = [(slit_slice[i_center+i], slit_slice[i_center+i+1]) \
                    for i in range(1, n_slice_one_direction+1)]
        slice_down = [(slit_slice[i_center-i-1], slit_slice[i_center-i]) \
                      for i in range(n_slice_one_direction)]

        igr_storage = extractor.igr_storage
        sky_basename = extractor.tgt_basename
        from libs.storage_descriptions import COMBINED_IMAGE_DESC
        raw_spec_products = igr_storage.load([COMBINED_IMAGE_DESC],
                                             sky_basename)
        d = raw_spec_products[COMBINED_IMAGE_DESC][0].data
        s_center = ap.extract_spectra_v2(d, slice_center[0], slice_center[1])

        s_up, s_down = [], []
        for s1, s2 in slice_up:
            s = ap.extract_spectra_v2(d, s1, s2)
            s_up.append(s)
        for s1, s2 in slice_down:
            s = ap.extract_spectra_v2(d, s1, s2)
            s_down.append(s)

        return s_center, s_up, s_down

    def get_refit_centroid_func(self, extractor, band,
                                s_center,
                                ohlines_db, ref_ohline_indices):
        # now fit

        #ohline_indices = [ref_ohline_indices[o] for o in orders_w_solutions]


        if 0:
            def test_order(oi):
                ax=subplot(111)
                ax.plot(wvl_solutions[oi], s_center[oi])
                #ax.plot(wvl_solutions[oi], raw_spec_products["specs"][oi])
                o = orders[oi]
                line_indices = ref_ohline_indices[o]
                for li in line_indices:
                    um = np.take(ohlines_db.um, li)
                    intensity = np.take(ohlines_db.intensity, li)
                    ax.vlines(um, ymin=0, ymax=-intensity)


        from libs.reidentify_ohlines import fit_ohlines, fit_ohlines_pixel

        def get_reidentified_lines_OH(orders_w_solutions,
                                      wvl_solutions, s_center):
            ref_pixel_list, reidentified_lines = \
                            fit_ohlines(ohlines_db, ref_ohline_indices,
                                        orders_w_solutions,
                                        wvl_solutions, s_center)

            reidentified_lines_map = dict(zip(orders_w_solutions,
                                              reidentified_lines))
            return reidentified_lines_map, ref_pixel_list


        orders_w_solutions = extractor.orders_w_solutions
        wvl_solutions = extractor.wvl_solutions

        reidentified_lines_map, ref_pixel_list_oh = \
                                get_reidentified_lines_OH(orders_w_solutions,
                                                          wvl_solutions,
                                                          s_center)

        if band == "H":

            def refit_centroid(s_center,
                               ref_pixel_list=ref_pixel_list_oh):
                centroids = fit_ohlines_pixel(s_center,
                                              ref_pixel_list)
                return centroids

        else: # band K

            import libs.master_calib as master_calib
            fn = "hitran_bootstrap_K_%s.json" % self.refdate
            bootstrap_name = master_calib.get_master_calib_abspath(fn)
            import json
            bootstrap = json.load(open(bootstrap_name))

            import libs.hitran as hitran
            r, ref_pixel_dict_hitrans = hitran.reidentify(orders_w_solutions,
                                                          wvl_solutions,
                                                          s_center,
                                                          bootstrap)
            # for i, s in r.items():
            #     ss = reidentified_lines_map[int(i)]
            #     ss0 = np.concatenate([ss[0], s["pixel"]])
            #     ss1 = np.concatenate([ss[1], s["wavelength"]])
            #     reidentified_lines_map[int(i)] = (ss0, ss1)

            #reidentified_lines_map, ref_pixel_list

            def refit_centroid(s_center,
                               ref_pixel_list=ref_pixel_list_oh,
                               ref_pixel_dict_hitrans=ref_pixel_dict_hitrans):
                centroids_oh = fit_ohlines_pixel(s_center,
                                                 ref_pixel_list)

                s_dict = dict(zip(orders_w_solutions, s_center))
                centroids_dict_hitrans = hitran.fit_hitrans_pixel(s_dict,
                                                                  ref_pixel_dict_hitrans)
                centroids = []
                for o, c_oh in zip(orders_w_solutions, centroids_oh):
                    if o in centroids_dict_hitrans:
                        c = np.concatenate([c_oh,
                                            centroids_dict_hitrans[o]["pixel"]])
                        centroids.append(c)
                    else:
                        centroids.append(c_oh)

                return centroids

        # reidentified_lines_map = get_reidentified_lines(orders_w_solutions,
        #                                                 wvl_solutions,
        #                                                 s_center)

        return refit_centroid

    def fit_slices(self, refit_centroid, s_up, s_down, fitted_centroid_center):

        d_shift_up = []
        for s in s_up:
            # TODO: ref_pixel_list_filtered need to be updated with recent fit.
            fitted_centroid = refit_centroid(s)
            # fitted_centroid = fit_ohlines_pixel(s,
            #                                     ref_pixel_list)
            d_shift = [b-a for a, b in zip(fitted_centroid_center,
                                           fitted_centroid)]
            d_shift_up.append(d_shift)

        d_shift_down = []
        for s in s_down:
            # TODO: ref_pixel_list_filtered need to be updated with recent fit.
            fitted_centroid = refit_centroid(s)
            # fitted_centroid = fit_ohlines_pixel(s,
            #                                     ref_pixel_list)
            #fitted_centroid_center,
            d_shift = [b-a for a, b in zip(fitted_centroid_center,
                                           fitted_centroid)]
            d_shift_down.append(d_shift)

        return d_shift_up, d_shift_down

    def get_pm_list(self, extractor, fitted_centroid_center,
                    d_shift_up, d_shift_down):
        # now fit
        orders = extractor.orders_w_solutions

        x_domain = [0, 2048]
        y_domain = [orders[0]-2, orders[-1]+2]


        xl = np.concatenate(fitted_centroid_center)

        yl_ = [o + np.zeros_like(x_) for o, x_ in zip(orders,
                                                      fitted_centroid_center)]
        yl = np.concatenate(yl_)

        from libs.ecfit import fit_2dspec, check_fit_simple

        zl_list = [np.concatenate(d_) for d_ \
                   in d_shift_down[::-1] + d_shift_up]

        pm_list = []
        for zl in zl_list:
            p, m = fit_2dspec(xl, yl, zl,
                              x_degree=1, y_degree=1,
                              x_domain=x_domain, y_domain=y_domain)
            pm_list.append((p,m))

        zz_std_list = []
        for zl, (p, m)  in zip(zl_list, pm_list):
            z_m = p(xl[m], yl[m])
            zz = z_m - zl[m]
            zz_std_list.append(zz.std())

        return xl, yl, zl_list, pm_list, zz_std_list


    def save_check_images(self, extractor, xl, yl, zl_list, pm_list):

        from libs.ecfit import check_fit_simple

        fig_list = []
        from matplotlib.figure import Figure

        orders = extractor.orders_w_solutions
        for zl, (p, m)  in zip(zl_list, pm_list):
            fig = Figure()
            check_fit_simple(fig, xl[m], yl[m], zl[m], p, orders)
            fig_list.append(fig)

        igr_path = extractor.igr_path
        from libs.qa_helper import figlist_to_pngs
        sky_basename = extractor.tgt_basename
        sky_figs = igr_path.get_section_filename_base("QA_PATH",
                                                      "oh_distortion",
                                                      "oh_distortion_"+sky_basename)
        print fig_list
        figlist_to_pngs(sky_figs, fig_list)

        return fig_list


    def convert_to_slitoffset_map(self, extractor,
                                  pm_list, n_slice_one_direction,
                                  slit_slice):

        xi = np.linspace(0, 2048, 128+1)
        from astropy.modeling import fitting
        from astropy.modeling.polynomial import Chebyshev2D
        x_domain = [0, 2048]
        y_domain = [0., 1.]

        p2_list = []

        orders = extractor.orders_w_solutions

        for o in orders:
            oi = np.zeros_like(xi) + o
            shift_list = []
            for p,m in pm_list[:n_slice_one_direction]:
                shift_list.append(p(xi, oi))

            shift_list.append(np.zeros_like(xi))

            for p,m in pm_list[n_slice_one_direction:]:
                shift_list.append(p(xi, oi))


            p_init = Chebyshev2D(x_degree=1, y_degree=2,
                                 x_domain=x_domain, y_domain=y_domain)
            f = fitting.LinearLSQFitter()

            yi = 0.5*(slit_slice[:-1] + slit_slice[1:])
            xl, yl = np.meshgrid(xi, yi)
            zl = np.array(shift_list)
            p = f(p_init, xl, yl, zl)

            p2_list.append(p)

        return p2_list

    def store_wavelength_outputs(self, extractor, p2_list, ap):

        orders = extractor.orders_w_solutions
        wvl_solutions = extractor.wvl_solutions

        p2_dict = dict(zip(orders, p2_list))


        # save order_map, etc

        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()
        order_map2 = ap.make_order_map(mask_top_bottom=True)

        slitoffset_map = np.empty_like(slitpos_map)
        slitoffset_map.fill(np.nan)

        wavelength_map = np.empty_like(slitpos_map)
        wavelength_map.fill(np.nan)

        from scipy.interpolate import interp1d
        for o, wvl in zip(ap.orders, wvl_solutions):
            xi = np.arange(0, 2048)
            xl, yl = np.meshgrid(xi, xi)
            msk = order_map == o

            xl_msk = xl[msk]
            slitoffset_map_msk = p2_dict[o](xl_msk, slitpos_map[msk])
            slitoffset_map[msk] = slitoffset_map_msk

            wvl_interp1d = interp1d(xi, wvl, bounds_error=False)
            wavelength_map[msk] = wvl_interp1d(xl_msk - slitoffset_map_msk)


        from libs.storage_descriptions import (ORDERMAP_FITS_DESC,
                                               SLITPOSMAP_FITS_DESC,
                                               SLITOFFSET_FITS_DESC,
                                               WAVELENGTHMAP_FITS_DESC,
                                               ORDERMAP_MASKED_FITS_DESC)
        from libs.products import PipelineImageBase, PipelineProducts
        products = PipelineProducts("Distortion map")

        for desc, im in [(ORDERMAP_FITS_DESC, order_map),
                         (SLITPOSMAP_FITS_DESC, slitpos_map),
                         (SLITOFFSET_FITS_DESC, slitoffset_map),
                         (WAVELENGTHMAP_FITS_DESC,wavelength_map),
                         (ORDERMAP_MASKED_FITS_DESC, order_map2)]:
            products.add(desc,
                         PipelineImageBase([], im))

        igr_storage = extractor.igr_storage
        igr_storage.store(products,
                          mastername=extractor.obj_filenames[0],
                          masterhdu=None)



        if 0:
            # test
            x = np.arange(2048, dtype="d")
            oi = 10
            o = orders[oi]

            yi = 0.5*(slit_slice[:-1] + slit_slice[1:])

            ax1 = subplot(211)
            s1 = s_up[-1][oi]
            s2 = s_down[-1][oi]

            ax1.plot(x, s1)
            ax1.plot(x, s2)

            ax2 = subplot(212, sharex=ax1, sharey=ax1)
            dx1 = p2_dict[o](x, yi[-1]+np.zeros_like(x))
            ax2.plot(x-dx1, s1)

            dx2 = p2_dict[o](x, yi[0]+np.zeros_like(x))
            ax2.plot(x-dx2, s2)

    def process_distortion_sky_band(self, band, obsids):

        from recipe_extract_base import RecipeExtractPR

        extractor = RecipeExtractPR(self.utdate, band,
                                    obsids,
                                    self.config)

        ap = load_aperture_wvlsol(extractor, band)

        ohlines_db, ref_ohline_indices = self.load_oh_reference_data(band)


        n_slice_one_direction = 2
        n_slice = n_slice_one_direction*2 + 1
        slit_slice = np.linspace(0., 1., n_slice+1)

        s_center, s_up, s_down = self.get_slices(extractor, ap,
                                                 n_slice_one_direction)


        refit_centroid = self.get_refit_centroid_func(extractor, band,
                                                      s_center,
                                                      ohlines_db, ref_ohline_indices)

        # TODO: we should not need this, instead recycle from preivious step.
        fitted_centroid_center = refit_centroid(s_center)
        # fitted_centroid_center = fit_ohlines_pixel(s_center,
        #                                            ref_pixel_list)

        d_shift_up, d_shift_down = self.fit_slices(refit_centroid,
                                                   s_up, s_down,
                                                   fitted_centroid_center)


        _ = self.get_pm_list(extractor,
                             fitted_centroid_center,
                             d_shift_up, d_shift_down)

        xl, yl, zl_list, pm_list, zz_std_list = _

        self.save_check_images(extractor, xl, yl,
                               zl_list, pm_list)


        p2_list = self.convert_to_slitoffset_map(extractor,
                                                 pm_list,
                                                 n_slice_one_direction,
                                                 slit_slice)


        self.store_wavelength_outputs(extractor, p2_list, ap)



class WavelengthFitter(object):
    pass

class SkyFitter(WavelengthFitter):
    def get_refdata(self, band):
        from libs.master_calib import load_sky_ref_data

        if band not in self._refdata:
            sky_refdata = load_sky_ref_data(self.refdate, band)
            self._refdata[band] = sky_refdata

        return self._refdata[band]

    def __init__(self, refdate):
        self.refdate = refdate
        self._refdata = {}

    def fit_individual_orders(self, band,
                              orders_w_solutions, wvl_solutions, s_list):

        sky_ref_data = self.get_refdata(band)
        ohline_indices = sky_ref_data["ohline_indices"]
        ohlines_db = sky_ref_data["ohlines_db"]

        from libs.reidentify_ohlines import fit_ohlines
        ref_pixel_list, reidentified_lines = \
                        fit_ohlines(ohlines_db, ohline_indices,
                                    orders_w_solutions,
                                    wvl_solutions, s_list)

        return ref_pixel_list, reidentified_lines

    def update_K(self, reidentified_lines_map,
                 orders_w_solutions,
                 wvl_solutions, s_list):
        import libs.master_calib as master_calib
        fn = "hitran_bootstrap_K_%s.json" % self.refdate
        bootstrap_name = master_calib.get_master_calib_abspath(fn)
        import json
        bootstrap = json.load(open(bootstrap_name))

        import libs.hitran as hitran
        r, ref_pixel_list = hitran.reidentify(orders_w_solutions,
                                              wvl_solutions, s_list,
                                              bootstrap)
        # json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
        # r = json.load(open(json_name))
        for i, s in r.items():
            ss = reidentified_lines_map[int(i)]
            ss0 = np.concatenate([ss[0], s["pixel"]])
            ss1 = np.concatenate([ss[1], s["wavelength"]])
            reidentified_lines_map[int(i)] = (ss0, ss1)

        return reidentified_lines_map

    def convert2wvlsol(self, p, orders_w_solutions):

        # derive wavelengths.
        xx = np.arange(2048)
        wvl_sol = []
        for o in orders_w_solutions:
            oo = np.empty_like(xx)
            oo.fill(o)
            wvl = p(xx, oo) / o
            wvl_sol.append(list(wvl))

        return wvl_sol

    def fit(self, band, orders_w_solutions, wvl_solutions, s_list):
        ref_pixel_list, reidentified_lines = \
                            self.fit_individual_orders(band,
                                                       orders_w_solutions,
                                                       wvl_solutions, s_list)

        from libs.ecfit import get_ordered_line_data, fit_2dspec

        reidentified_lines_map = dict(zip(orders_w_solutions,
                                          reidentified_lines))

        if band == "K":
            self.update_K(reidentified_lines_map,
                          orders_w_solutions,
                          wvl_solutions, s_list)

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

        wvl_sol = self.convert2wvlsol(p, orders_w_solutions)

        return orders_w_solutions, wvl_sol, reidentified_lines_map, p, m


def fit_oh_spectra(refdate, band,
                   orders_w_solutions,
                   wvl_solutions, s_list):

    fitter = SkyFitter(self.refdate)
    _ = fitter.fit(band, orders_w_solutions, wvl_solutions, s_list)

    orders_w_solutions, wvl_sol, reidentified_lines_map, p, m = _

    return orders_w_solutions, wvl_sol, reidentified_lines_map, p, m

# def fit_oh_spectra(refdate, band,
#                    orders_w_solutions,
#                    wvl_solutions, s_list):
#     if 1:
#         from libs.master_calib import load_sky_ref_data

#         # ref_date = "20140316"

#         #refdate = config.get_value("REFDATE", utdate)
#         sky_ref_data = load_sky_ref_data(refdate, band)

#     if 1:
#         # Now we fit with gaussian profile for matched positions.


#         # from scipy.interpolate import interp1d
#         # from reidentify import reidentify_lines_all

#         x = np.arange(2048)


#         # line_indices_list = [ref_ohline_indices[str(o)] for o in igrins_orders[band]]




#         ###### not fit identified lines

#         from libs.ecfit import get_ordered_line_data, fit_2dspec

#         # d_x_wvl = {}
#         # for order, z in echel.zdata.items():
#         #     xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
#         #     x_T = xy_T[:,0]
#         #     d_x_wvl[order]=(x_T, z.wvl)

#         reidentified_lines_map = dict(zip(orders_w_solutions,
#                                           reidentified_lines))

#         if band == "K":
#             import libs.master_calib as master_calib
#             fn = "hitran_bootstrap_K_%s.json" % refdate
#             bootstrap_name = master_calib.get_master_calib_abspath(fn)
#             import json
#             bootstrap = json.load(open(bootstrap_name))

#             import libs.hitran as hitran
#             r, ref_pixel_list = hitran.reidentify(wvl_solutions, s_list, bootstrap)
#             # json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
#             # r = json.load(open(json_name))
#             for i, s in r.items():
#                 ss = reidentified_lines_map[int(i)]
#                 ss0 = np.concatenate([ss[0], s["pixel"]])
#                 ss1 = np.concatenate([ss[1], s["wavelength"]])
#                 reidentified_lines_map[int(i)] = (ss0, ss1)

#         xl, yl, zl = get_ordered_line_data(reidentified_lines_map)
#         # xl : pixel
#         # yl : order
#         # zl : wvl * order

#         x_domain = [0, 2047]
#         y_domain = [orders_w_solutions[0]-2, orders_w_solutions[-1]+2]
#         x_degree, y_degree = 4, 3
#         #x_degree, y_degree = 3, 2
#         p, m = fit_2dspec(xl, yl, zl, x_degree=x_degree, y_degree=y_degree,
#                           x_domain=x_domain, y_domain=y_domain)


#         # derive wavelengths.
#         xx = np.arange(2048)
#         wvl_sol = []
#         for o in orders_w_solutions:
#             oo = np.empty_like(xx)
#             oo.fill(o)
#             wvl = p(xx, oo) / o
#             wvl_sol.append(list(wvl))


#         return orders_w_solutions, wvl_sol, reidentified_lines_map, p, m







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
