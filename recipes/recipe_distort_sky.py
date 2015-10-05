import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
#import libs.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

from libs.recipe_base import RecipeBase

class RecipeDistortionSky(RecipeBase):
    RECIPE_NAME = "SKY"

    def run_selected_bands(self, utdate, selected, bands):
        for s in selected:
            obsids = s[0]
            print obsids
            # frametypes = s[1]

            for band in bands:
                process_distortion_sky_band(utdate, self.refdate, band, obsids,
                                            self.config)
                # process_wvlsol_band(utdate, self.refdate, band, obsids,
                #                     self.config)

def distortion_sky(utdate, bands="HK",
                   starting_obsids=None, config_file="recipe.config"):

    RecipeDistortionSky()(utdate, bands,
                          starting_obsids, config_file)

# def distortion_sky(utdate, refdate="20140316", bands="HK",
#                    starting_obsids=None):

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
#             process_distortion_sky_band(utdate, refdate, band, obsids)


def load_aperture(igr_storage, band, master_obsid, flaton_db,
                  orders, orders_w_solutions):
    from libs.process_flat import FLATCENTROID_SOL_JSON_DESC

    flaton_basename = flaton_db.query(band, master_obsid)

    aperture_solution_products = igr_storage.load([FLATCENTROID_SOL_JSON_DESC], flaton_basename)


    bottomup_solutions = aperture_solution_products[FLATCENTROID_SOL_JSON_DESC]["bottom_up_solutions"]


    _o_s = dict(zip(orders, bottomup_solutions))
    ap =  Apertures(orders_w_solutions,
                    [_o_s[o] for o in orders_w_solutions])

    # _o_s = dict(zip(orders, bottomup_solutions))
    # ap =  Apertures(orders,
    #                 [_o_s[o] for o in orders])
    # # ap =  Apertures(orders,
    # #                 bottomup_solutions)

    return ap



def process_distortion_sky_band(utdate, refdate, band, obsids, config):

    from libs.products import ProductDB, PipelineStorage

    igr_path = IGRINSPath(config, utdate)

    igr_storage = PipelineStorage(igr_path)

    sky_filenames = igr_path.get_filenames(band, obsids)


    sky_basename = os.path.splitext(os.path.basename(sky_filenames[0]))[0]

    master_obsid = obsids[0]


    flaton_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "flat_on.db",
                                                        )
    flaton_db = ProductDB(flaton_db_name)

    thar_db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "thar.db",
                                                        )
    thar_db = ProductDB(thar_db_name)



    from libs.process_thar import COMBINED_IMAGE_DESC, ONED_SPEC_JSON
    raw_spec_products = igr_storage.load([COMBINED_IMAGE_DESC, ONED_SPEC_JSON],
                                         sky_basename)

    # raw_spec_products = PipelineProducts.load(sky_path.get_secondary_path("raw_spec"))

    SKY_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.json")

    wvlsol_products = igr_storage.load([SKY_WVLSOL_JSON_DESC],
                                       sky_basename)[SKY_WVLSOL_JSON_DESC]

    orders_w_solutions = wvlsol_products["orders"]
    wvl_solutions = wvlsol_products["wvl_sol"]

    ap = load_aperture(igr_storage, band, master_obsid,
                       flaton_db,
                       raw_spec_products[ONED_SPEC_JSON]["orders"],
                       orders_w_solutions)
    #orders_w_solutions = ap.orders


    if 1: # load reference data
        from libs.master_calib import load_sky_ref_data

        ref_utdate = config.get_value("REFDATE", utdate)

        sky_ref_data = load_sky_ref_data(ref_utdate, band)


        ohlines_db = sky_ref_data["ohlines_db"]
        ref_ohline_indices = sky_ref_data["ohline_indices"]


        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = wvlsol_products["wvl_sol"]


    if 0:
        raw_spec_products = PipelineProducts.load(sky_path.get_secondary_path("raw_spec"))



        basename = flaton_db.query(band, sky_master_obsid)
        flaton_path = ProductPath(igr_path, basename)

        aperture_solution_products = PipelineProducts.load(flaton_path.get_secondary_path("aperture_solutions"))

        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        basename = thar_db.query(band, sky_master_obsid)
        thar_path = ProductPath(igr_path, basename)
        fn = thar_path.get_secondary_path("median_spectra")
        # thar_products = PipelineProducts.load(fn)

        # basename = sky_db.query(band, sky_master_obsid)
        # sky_path = ProductPath(igr_path, basename)
        fn = sky_path.get_secondary_path("wvlsol_v1")
        wvlsol_products = PipelineProducts.load(fn)

        if 1:
            from libs.master_calib import load_sky_ref_data

            ref_utdate = "20140316"

            sky_ref_data = load_sky_ref_data(ref_utdate, band)


            ohlines_db = sky_ref_data["ohlines_db"]
            ref_ohline_indices = sky_ref_data["ohline_indices"]


            orders_w_solutions = wvlsol_products["orders"]
            wvl_solutions = wvlsol_products["wvl_sol"]

        if 1: # make aperture

            _o_s = dict(zip(raw_spec_products["orders"], bottomup_solutions))
            ap =  Apertures(orders_w_solutions,
                            [_o_s[o] for o in orders_w_solutions])


    if 1:

        n_slice_one_direction = 2
        n_slice = n_slice_one_direction*2 + 1
        i_center = n_slice_one_direction
        slit_slice = np.linspace(0., 1., n_slice+1)

        slice_center = (slit_slice[i_center], slit_slice[i_center+1])
        slice_up = [(slit_slice[i_center+i], slit_slice[i_center+i+1]) \
                    for i in range(1, n_slice_one_direction+1)]
        slice_down = [(slit_slice[i_center-i-1], slit_slice[i_center-i]) \
                      for i in range(n_slice_one_direction)]

        d = raw_spec_products[COMBINED_IMAGE_DESC].data
        s_center = ap.extract_spectra_v2(d, slice_center[0], slice_center[1])

        s_up, s_down = [], []
        for s1, s2 in slice_up:
            s = ap.extract_spectra_v2(d, s1, s2)
            s_up.append(s)
        for s1, s2 in slice_down:
            s = ap.extract_spectra_v2(d, s1, s2)
            s_down.append(s)


    if 1:
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

        if band == "H":
            reidentified_lines_map, ref_pixel_list_oh = \
                       get_reidentified_lines_OH(orders_w_solutions,
                                                 wvl_solutions,
                                                 s_center)

            def refit_centroid(s_center,
                               ref_pixel_list=ref_pixel_list_oh):
                centroids = fit_ohlines_pixel(s_center,
                                              ref_pixel_list)
                return centroids

        else: # band K
            reidentified_lines_map, ref_pixel_list_oh = \
                       get_reidentified_lines_OH(orders_w_solutions,
                                                 wvl_solutions,
                                                 s_center)

            import libs.master_calib as master_calib
            fn = "hitran_bootstrap_K_%s.json" % ref_utdate
            bootstrap_name = master_calib.get_master_calib_abspath(fn)
            import json
            bootstrap = json.load(open(bootstrap_name))

            import libs.hitran as hitran
            r, ref_pixel_dict_hitrans = hitran.reidentify(wvl_solutions,
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


    if 1:
        # TODO: we should not need this, instead recycle from preivious step.
        fitted_centroid_center = refit_centroid(s_center)
        # fitted_centroid_center = fit_ohlines_pixel(s_center,
        #                                            ref_pixel_list)

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


    if 1:
        # now fit
        orders = orders_w_solutions

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

        fig_list = []
        from matplotlib.figure import Figure
        for zl, (p, m)  in zip(zl_list, pm_list):
            fig = Figure()
            check_fit_simple(fig, xl[m], yl[m], zl[m], p, orders)
            fig_list.append(fig)


    if 1:
        xi = np.linspace(0, 2048, 128+1)
        from astropy.modeling import fitting
        from astropy.modeling.polynomial import Chebyshev2D
        x_domain = [0, 2048]
        y_domain = [0., 1.]

        p2_list = []
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

    if 1:
        p2_dict = dict(zip(orders, p2_list))

        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()

        slitoffset_map = np.empty_like(slitpos_map)
        slitoffset_map.fill(np.nan)
        for o in ap.orders:
            xi = np.arange(0, 2048)
            xl, yl = np.meshgrid(xi, xi)
            msk = order_map == o
            slitoffset_map[msk] = p2_dict[o](xl[msk], slitpos_map[msk])

        import libs.fits as pyfits
        #fn = sky_path.get_secondary_path("slitoffset_map.fits")
        #pyfits.PrimaryHDU(data=slitoffset_map).writeto(fn, clobber=True)

        SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitoffset_map.fits")
        from libs.products import PipelineImageBase, PipelineProducts
        distortion_products = PipelineProducts("Distortion map")
        distortion_products.add(SLITOFFSET_FITS_DESC,
                                PipelineImageBase([],
                                              slitoffset_map))

        igr_storage.store(distortion_products,
                          mastername=sky_filenames[0],
                          masterhdu=None)


        from libs.qa_helper import figlist_to_pngs
        sky_figs = igr_path.get_section_filename_base("QA_PATH",
                                                      "oh_distortion",
                                                      "oh_distortion_dir")
        print fig_list
        figlist_to_pngs(sky_figs, fig_list)


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


# if 1:

#     obsids = recipe_dict["SKY"][0][0]

if __name__ == "__main__":
    import sys

    utdate = sys.argv[1]
    bands = "HK"
    starting_obsids = None

    if len(sys.argv) >= 3:
        bands = sys.argv[2]

    if len(sys.argv) >= 4:
        starting_obsids = sys.argv[3]

    distortion_sky(utdate, refdate="20140316", bands=bands,
                   starting_obsids=starting_obsids)
