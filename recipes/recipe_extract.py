import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts


def a0v_ab(utdate, refdate="20140316", bands="HK",
           starting_obsids=None, interactive=False,
           config_file="recipe.config"):
    recipe = "A0V_AB"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids, interactive=interactive,
             config_file=config_file)

def stellar_ab(utdate, refdate="20140316", bands="HK",
             starting_obsids=None,
               config_file="recipe.config"):
    recipe = "STELLAR_AB"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids,
             config_file=config_file)

def extended_ab(utdate, refdate="20140316", bands="HK",
                starting_obsids=None,
                config_file="recipe.config"):
    recipe = "EXTENDED_AB"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids,
             config_file=config_file)

def extended_onoff(utdate, refdate="20140316", bands="HK",
                   starting_obsids=None,
                   config_file="recipe.config"):
    recipe = "EXTENDED_ONOFF"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids,
             config_file=config_file)



def abba_all(recipe_name, utdate, refdate="20140316", bands="HK",
             starting_obsids=None, interactive=False,
             config_file="recipe.config"):

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    if not bands in ["H", "K", "HK"]:
        raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    fn = "%s.recipes" % utdate
    from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
    recipe = Recipes(fn)

    if starting_obsids is not None:
        starting_obsids = map(int, starting_obsids.split(","))

    selected = recipe.select(recipe_name, starting_obsids)
    if not selected:
        print "no recipe of with matching arguments is found"

    for s in selected:
        obsids = s[0]
        frametypes = s[1]

        for band in bands:
            process_abba_band(recipe_name, utdate, refdate, band,
                              obsids, frametypes, config,
                              do_interactive_figure=interactive)


def process_abba_band(recipe, utdate, refdate, band, obsids, frametypes,
                      config,
                      do_interactive_figure=False):

    from libs.products import ProductPath, ProductDB, PipelineStorage

    if recipe == "A0V_AB":

        DO_STD = True
        FIX_TELLURIC=False

    elif recipe == "STELLAR_AB":

        DO_STD = False
        FIX_TELLURIC=True

    elif recipe == "EXTENDED_AB":

        DO_STD = False
        FIX_TELLURIC=True

    elif recipe == "EXTENDED_ONOFF":

        DO_STD = False
        FIX_TELLURIC=True


    if 1:

        igr_path = IGRINSPath(config, utdate)

        igr_storage = PipelineStorage(igr_path)

        obj_filenames = igr_path.get_filenames(band, obsids)

        master_obsid = obsids[0]

        tgt_basename = os.path.splitext(os.path.basename(obj_filenames[0]))[0]

        db = {}
        basenames = {}

        db_types = ["flat_off", "flat_on", "thar", "sky", "a0v"]

        for db_type in db_types:

            db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "%s.db" % db_type,
                                                        )
            db[db_type] = ProductDB(db_name)


        # to get basenames
        db_types = ["flat_off", "flat_on", "thar", "sky"]
        if FIX_TELLURIC:
            db_types.append("a0v")

        for db_type in db_types:
            basenames[db_type] = db[db_type].query(band, master_obsid)



    if 1: # make aperture
        SKY_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.json")

        sky_basename = db["sky"].query(band, master_obsid)
        wvlsol_products = igr_storage.load([SKY_WVLSOL_JSON_DESC],
                                           sky_basename)[SKY_WVLSOL_JSON_DESC]

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = wvlsol_products["wvl_sol"]

        from libs.process_thar import COMBINED_IMAGE_DESC, ONED_SPEC_JSON
        raw_spec_products = igr_storage.load([COMBINED_IMAGE_DESC, ONED_SPEC_JSON],
                                             sky_basename)

        from recipe_wvlsol_sky import load_aperture2

        ap = load_aperture2(igr_storage, band, master_obsid,
                            db["flat_on"],
                            raw_spec_products[ONED_SPEC_JSON]["orders"],
                            orders_w_solutions)

        # load_aperture2(igr_storage, band, master_obsid, flaton_db,
        #                orders, orders_w_solutions)
        # _o_s = dict(zip(raw_spec_products["orders"], bottomup_solutions))
        # ap =  Apertures(orders_w_solutions,
        #                 [_o_s[o] for o in orders_w_solutions])


        # This should be saved somewhere and loaded, instead of making it every time.
        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()
        order_map2 = ap.make_order_map(mask_top_bottom=True)

    # telluric
    if FIX_TELLURIC:
        A0V_basename = db["a0v"].query(band, master_obsid)

        SPEC_FITS_FLATTENED_DESC = ("OUTDATA_PATH", "",
                                    ".spec_flattened.fits")
        telluric_cor_ = igr_storage.load([SPEC_FITS_FLATTENED_DESC],
                                         A0V_basename)[SPEC_FITS_FLATTENED_DESC]

        #A0V_path = ProductPath(igr_path, A0V_basename)
        #fn = A0V_path.get_secondary_path("spec_flattened.fits")
        telluric_cor = list(telluric_cor_.data)
        # fn = A0V_path.get_secondary_path("spec.fits")
        # telluric_cor = list(pyfits.open(fn)[0].data)
        #print fn


    if 1:

        from libs.process_flat import (HOTPIX_MASK_DESC,
                                       DEADPIX_MASK_DESC,
                                       ORDER_FLAT_IM_DESC,
                                       ORDER_FLAT_JSON_DESC,
                                       FLAT_NORMED_DESC,
                                       FLAT_MASK_DESC)

        hotpix_mask = igr_storage.load([HOTPIX_MASK_DESC],
                                       basenames["flat_off"])[HOTPIX_MASK_DESC]

        deadpix_mask = igr_storage.load([DEADPIX_MASK_DESC],
                                        basenames["flat_on"])[DEADPIX_MASK_DESC]

        pix_mask  = hotpix_mask.data | deadpix_mask.data



        # aperture_solution_products = PipelineProducts.load(aperture_solutions_name)


        orderflat_ = igr_storage.load([ORDER_FLAT_IM_DESC],
                                     basenames["flat_on"])[ORDER_FLAT_IM_DESC]


        orderflat = orderflat_.data
        orderflat[pix_mask] = np.nan

        orderflat_json = igr_storage.load([ORDER_FLAT_JSON_DESC],
                                          basenames["flat_on"])[ORDER_FLAT_JSON_DESC]
        order_flat_meanspec = np.array(orderflat_json["mean_order_specs"])

        # flat_normed = igr_storage.load([FLAT_NORMED_DESC],
        #                                basenames["flat_on"])[FLAT_NORMED_DESC]

        flat_mask = igr_storage.load([FLAT_MASK_DESC],
                                     basenames["flat_on"])[FLAT_MASK_DESC]
        bias_mask = flat_mask.data & (order_map2 > 0)

        SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitoffset_map.fits")
        prod_ = igr_storage.load([SLITOFFSET_FITS_DESC],
                                 basenames["sky"])[SLITOFFSET_FITS_DESC]
        #fn = sky_path.get_secondary_path("slitoffset_map.fits")
        slitoffset_map = prod_.data

    if 1:

        abba_names = obj_filenames

        def filter_abba_names(abba_names, frametypes, frametype):
            return [an for an, ft in zip(abba_names, frametypes) if ft == frametype]


        a_name_list = filter_abba_names(abba_names, frametypes, "A")
        b_name_list = filter_abba_names(abba_names, frametypes, "B")

        if recipe in ["A0V_AB", "STELLAR_AB"]:
            IF_POINT_SOURCE = True
        elif recipe in ["EXTENDED_AB", "EXTENDED_ONOFF"]:
            IF_POINT_SOURCE = False
        else:
            print "Unknown recipe : %s" % recipe

        if 1:
            #ab_names = ab_names_list[0]

            master_hdu = pyfits.open(a_name_list[0])[0]

            a_list = [pyfits.open(name)[0].data \
                      for name in a_name_list]
            b_list = [pyfits.open(name)[0].data \
                      for name in b_name_list]


            # we may need to detrip

            # first define extract profile (gaussian).


            # dx = 100

            if IF_POINT_SOURCE: # if point source
                # for point sources, variance estimation becomes wrong
                # if lenth of two is different,
                assert len(a_list) == len(b_list)

            # a_b != 1 for the cases when len(a) != len(b)
            a_b = float(len(a_list)) / len(b_list)

            a_data = np.sum(a_list, axis=0)
            b_data = np.sum(b_list, axis=0)

            data_minus = a_data - a_b*b_data
            #data_minus0 = data_minus

            from libs.destriper import destriper
            if 1:

                data_minus = destriper.get_destriped(data_minus,
                                                     ~np.isfinite(data_minus),
                                                     pattern=64)

            data_minus_flattened = data_minus / orderflat
            data_minus_flattened[~flat_mask.data] = np.nan
            #data_minus_flattened[order_flat_meanspec<0.1*order_flat_meanspec.max()] = np.nan


            # for variance, we need a square of a_b
            data_plus = (a_data + (a_b**2)*b_data)

            import scipy.ndimage as ni
            bias_mask2 = ni.binary_dilation(bias_mask)

            from libs import instrument_parameters
            gain =  instrument_parameters.gain[band]

            # random noise
            variance0 = data_minus

            variance_ = variance0.copy()
            variance_[bias_mask2] = np.nan
            variance_[pix_mask] = np.nan

            mm = np.ma.array(variance0, mask=~np.isfinite(variance0))
            ss = np.ma.median(mm, axis=0)
            variance_ = variance_ - ss

            for i in range(5):
                st = np.nanstd(variance_, axis=0)
                variance_[np.abs(variance_) > 3*st] = np.nan
                #st = np.nanstd(variance_, axis=0)

            variance = destriper.get_destriped(variance0,
                                                ~np.isfinite(variance_),
                                               pattern=64)

            variance_ = variance.copy()
            variance_[bias_mask2] = np.nan
            variance_[pix_mask] = np.nan

            st = np.nanstd(variance_)
            st = np.nanstd(variance_[np.abs(variance_) < 3*st])

            variance_[np.abs(variance_-ss) > 3*st] = np.nan

            x_std = ni.median_filter(np.nanstd(variance_, axis=0), 11)

            variance_map0 = np.zeros_like(variance) + x_std**2



            variance_map = variance_map0 + np.abs(data_plus)/gain # add poison noise in ADU
            # we ignore effect of flattening

            # now estimate lsf


            # estimate lsf
            ordermap_bpixed = order_map.copy()
            ordermap_bpixed[pix_mask] = 0
            ordermap_bpixed[~np.isfinite(orderflat)] = 0
        #


        if IF_POINT_SOURCE: # if point source

            x1, x2 = 800, 1200
            bins, lsf_list = ap.extract_lsf(ordermap_bpixed, slitpos_map,
                                            data_minus_flattened,
                                            x1, x2, bins=None)


            hh0 = np.sum(lsf_list, axis=0)
            peak1, peak2 = max(hh0), -min(hh0)
            lsf_x = 0.5*(bins[1:]+bins[:-1])
            lsf_y = hh0/(peak1+peak2)

            from scipy.interpolate import UnivariateSpline
            lsf_ = UnivariateSpline(lsf_x, lsf_y, k=3, s=0,
                                    bbox=[0, 1])
            roots = list(lsf_.roots())
            #assert(len(roots) == 1)
            integ_list = []
            from itertools import izip, cycle
            for ss, int_r1, int_r2 in izip(cycle([1, -1]),
                                                  [0] + roots,
                                                  roots + [1]):
                #print ss, int_r1, int_r2
                integ_list.append(lsf_.integral(int_r1, int_r2))
            integ = np.abs(np.sum(integ_list))

            def lsf(o, x, slitpos):
                return lsf_(slitpos) / integ

            # make weight map
            profile_map = ap.make_profile_map(order_map, slitpos_map, lsf)

            # extract spec

            s_list, v_list = ap.extract_stellar(ordermap_bpixed,
                                                profile_map,
                                                variance_map,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map)

            # make synth_spec : profile * spectra
            synth_map = ap.make_synth_map(order_map, slitpos_map,
                                          profile_map, s_list,
                                          slitoffset_map=slitoffset_map)

            sig_map = (data_minus_flattened - synth_map)**2/variance_map
            ## mark sig_map > 100 as cosmicay. The threshold need to be fixed.


            # reextract with new variance map and CR is rejected
            variance_map_r = variance_map0 + np.abs(synth_map)/gain
            variance_map2 = np.max([variance_map, variance_map_r], axis=0)
            variance_map2[np.abs(sig_map) > 100] = np.nan

            # extract spec

            s_list, v_list = ap.extract_stellar(ordermap_bpixed, profile_map,
                                                variance_map2,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map)

            # s_list_r, v_list = ap.extract_stellar(ordermap_bpixed, profile_map,
            #                                       variance_map2,
            #                                       data_minus_flattened,
            #                                       slitoffset_map=None)


        else: # if extended source
            from scipy.interpolate import UnivariateSpline
            if recipe in ["EXTENDED_AB", "EXTENDED_ABBA"]:
                delta = 0.01
                lsf_ = UnivariateSpline([0, 0.5-delta, 0.5+delta, 1],
                                        [1., 1., -1., -1.],
                                        k=1, s=0,
                                        bbox=[0, 1])
            else:
                lsf_ = UnivariateSpline([0, 1], [1., 1.],
                                        k=1, s=0,
                                        bbox=[0, 1])

            def lsf(o, x, slitpos):
                return lsf_(slitpos)

            profile_map = ap.make_profile_map(order_map, slitpos_map, lsf)

            # we need to update the variance map by rejecting
            # exteneded sources, but it is not clear how we do this
            # for extended source.
            variance_map2 = variance_map
            s_list, v_list = ap.extract_stellar(ordermap_bpixed,
                                                profile_map,
                                                variance_map2,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map
                                                )


            # # make synth_spec : profile * spectra
            # synth_map = ap.make_synth_map(order_map, slitpos_map,
            #                               lsf, s_list,
            #                               slitoffset_map=slitoffset_map)

        if 1: # save the product
            from libs.process_thar import COMBINED_IMAGE_DESC, ONED_SPEC_JSON
            VARIANCE_MAP_DESC = ("OUTDATA_PATH", "", ".variance_map.fits")
            from libs.products import PipelineImage

            r = PipelineProducts("1d specs")

            r.add(COMBINED_IMAGE_DESC, PipelineImage([],
                                                     data_minus_flattened))
            r.add(VARIANCE_MAP_DESC, PipelineImage([],
                                                   variance_map2))

            # r.add(VARIANCE_MAP_DESC, PipelineImage([],
            #                                        variance_map.data))

            igr_storage.store(r,
                              mastername=obj_filenames[0],
                              masterhdu=None)

            #                      combined_image_raw=data_minus0,
            #                      combined_image=data_minus_flattened,
            #                      variance_map=variance_map2
            #                      )
            # VARIANCE_MAP_DESC

            # SPECPARAM_MAP_DESC = ("OUTDATA_PATH", "", ".spec_params.json")
            # r.add(SPECPARAM_MAP_DESC,
            #       PipelineDict())


            # fn = obj_path.get_secondary_path("spec_params")
            # r.save(fn, masterhdu=master_hdu)


        from libs.process_flat import ORDER_FLAT_JSON_DESC
        prod = igr_storage.load([ORDER_FLAT_JSON_DESC],
                                basenames["flat_on"])[ORDER_FLAT_JSON_DESC]

        new_orders = prod["orders"]
        # fitted_response = orderflat_products["fitted_responses"]
        i1i2_list = prod["i1i2_list"]

        fig_list = []
        if 1:


            if do_interactive_figure:
                from matplotlib.pyplot import figure as Figure
            else:
                from matplotlib.figure import Figure
            fig1 = Figure(figsize=(12,6))
            fig_list.append(fig1)

            fig1b = Figure(figsize=(12,6))
            fig_list.append(fig1b)

            ax1 = fig1.add_subplot(111)
            ax2 = fig1b.add_subplot(111)
            #from libs.stddev_filter import window_stdev


            for o, s in zip(ap.orders, s_list):

                o_new_ind = np.searchsorted(new_orders, o)
                m = order_flat_meanspec[o_new_ind]
                s[m < 0.1] = np.nan


            if FIX_TELLURIC:
                s_list_cor = []
                for s, t in zip(s_list, telluric_cor):

                    s_list_cor.append(s/t)
            else:
                s_list_cor = s_list

            wvl_list_html, s_list_html, sn_list_html = [], [], []

            for o, wvl, s, v in zip(ap.orders, wvl_solutions,
                                    s_list, v_list):

                o_new_ind = np.searchsorted(new_orders, o)
                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                #res = fitted_response[o_new_ind]
                wvl, s = np.array(wvl), np.array(s)
                mmm = np.isfinite(s[sl])
                ax1.plot(wvl[sl][mmm], s[sl][mmm], "0.5")


                dw = np.gradient(wvl)
                pixel_per_res_element = (wvl/40000.)/dw
                #print pixel_per_res_element[1024]
                # len(pixel_per_res_element) = 2047. But we ignore it.
                sn = (s/v**.5)*(pixel_per_res_element**.5)
                ax2.plot(wvl[sl], sn[sl])

                wvl_list_html.append(wvl[sl])
                s_list_html.append(s[sl])
                sn_list_html.append(sn[sl])
                #s_std = window_stdev(s, 25)
                #ax1.plot(wvl[sl], ni.median_filter(s[sl]/s_std[sl], 10), "g-")

            if FIX_TELLURIC:

                for o, wvl, s, v in zip(ap.orders, wvl_solutions,
                                        s_list_cor, v_list):

                    o_new_ind = np.searchsorted(new_orders, o)
                    i1, i2 = i1i2_list[o_new_ind]
                    sl = slice(i1, i2)
                    wvl, s = np.array(wvl), np.array(s)
                    mmm = np.isfinite(s[sl])
                    ax1.plot(wvl[sl][mmm], s[sl][mmm], "0.8", zorder=0.5)



            ymax = 1.1*max(s_list[12][sl])
            ax1.set_ylim(0, ymax)
            pixel_per_res_element = 3.7
            ymax = 1.2*(s_list[12][1024]/v_list[12][1024]**.5*pixel_per_res_element**.5)
            ax2.set_ylim(0, ymax)
            ax2.set_ylabel("S/N per Res. Element")


        if IF_POINT_SOURCE: # if point source, try simple telluric factor for A0V
            # new_orders = orderflat_products["orders"]
            # # fitted_response = orderflat_products["fitted_responses"]
            # i1i2_list = orderflat_products["i1i2_list"]

            fig2 = Figure(figsize=(12,8))
            fig_list.append(fig2)

            ax1 = fig2.add_subplot(211)
            for o, wvl, s in zip(ap.orders, wvl_solutions, s_list):
                o_new_ind = np.searchsorted(new_orders, o)
                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                # res = fitted_response[o_new_ind]
                #ax1.plot(wvl[sl], (s/res)[sl])

            from libs.master_calib import get_master_calib_abspath
            fn = get_master_calib_abspath("A0V/vegallpr25.50000resam5")
            d = np.genfromtxt(fn)

            wvl, flux, cont = (d[:,i] for i in [0, 1, 2])
            ax2 = fig2.add_subplot(212, sharex=ax1)
            wvl = wvl/1000.
            if band == "H":
                mask_wvl1, mask_wvl2 = 1.450, 1.850
            else:
                mask_wvl1, mask_wvl2 = 1.850, 2.550

            mask_H = (mask_wvl1 < wvl) & (wvl < mask_wvl2)

            fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
            telluric = pyfits.open(fn)[1].data
            telluric_lam = telluric["lam"]
            tel_mask_H = (mask_wvl1 < telluric_lam) & (telluric_lam < mask_wvl2)
            #plot(telluric_lam[tel_mask_H], telluric["trans"][tel_mask_H])
            from scipy.interpolate import interp1d, UnivariateSpline
            spl = UnivariateSpline(telluric_lam[tel_mask_H],
                                   telluric["trans"][tel_mask_H],
                                   k=1,s=0)

            spl = interp1d(telluric_lam[tel_mask_H],
                           telluric["trans"][tel_mask_H],
                           bounds_error=False
                           )

            #plot(telluric_lam[tel_mask_H],
            #     spl(telluric_lam[tel_mask_H]))
            #www = np.linspace(1.4, 1.8, 1000)
            #plot(www,
            #     spl(www))
            trans = spl(wvl[mask_H])
            ax1.plot(wvl[mask_H], flux[mask_H]/cont[mask_H]*trans,
                     color="0.5", zorder=0.5)
            #ax2.plot(wvl[mask_H], flux[mask_H]/cont[mask_H]*trans)


            trans_m = ni.maximum_filter(trans, 128)
            trans_mg = ni.gaussian_filter(trans_m, 32)

            zzz0 = flux[mask_H]/cont[mask_H]
            zzz = zzz0*trans
            mmm = trans/trans_mg > 0.95
            zzz[~mmm] = np.nan
            wvl_zzz = wvl[mask_H]
            #ax2.plot(, zzz)

            #ax2 = subplot(212)
            if DO_STD:
                telluric_cor = []

            for o, wvl, s in zip(ap.orders, wvl_solutions, s_list):
                o_new_ind = np.searchsorted(new_orders, o)

                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                wvl1, wvl2 = wvl[i1], wvl[i2]
                #wvl1, wvl2 = wvl[0], wvl[-1]
                z_m = (wvl1 < wvl_zzz) & (wvl_zzz < wvl2)

                ss = interp1d(wvl, s)

                s_interped = ss(wvl_zzz[z_m])

                xxx, yyy = wvl_zzz[z_m], s_interped/zzz[z_m]

                from astropy.modeling import models, fitting
                p_init = models.Chebyshev1D(domain=[xxx[0], xxx[-1]],
                                            degree=6)
                fit_p = fitting.LinearLSQFitter()
                x_m = np.isfinite(yyy)
                p = fit_p(p_init, xxx[x_m], yyy[x_m])
                #ax2.plot(xxx, yyy)
                #ax2.plot(xxx, p(xxx))

                res_ = p(wvl)

                z_interp = interp1d(xxx, zzz0[z_m], bounds_error=False)
                A0V = z_interp(wvl)
                res_[res_<0.3*res_.max()] = np.nan
                ax1.plot(wvl[sl], (s/res_)[sl])
                ax2.plot(wvl[sl], (s/res_/A0V)[sl])

                if DO_STD:
                    telluric_cor.append((s/res_)/A0V)

            ax1.axhline(1, color="0.5")
            ax2.axhline(1, color="0.5")

        # save html

    if 1:
        SKY_WVLSOL_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.fits")
        fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
                                  basenames["sky"])

        # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
        f = pyfits.open(fn)

        d = np.array(s_list)
        #d[~np.isfinite(d)] = 0.
        f[0].data = d.astype("f32")

        SPEC_FITS_DESC = ("OUTDATA_PATH", "", ".spec.fits")
        fout = igr_storage.get_path(SPEC_FITS_DESC,
                                    tgt_basename)

        #fout = obj_path.get_secondary_path("spec.fits")
        f.writeto(fout, clobber=True)


        if DO_STD:
            d = np.array(telluric_cor)
            d[~np.isfinite(d)] = 0.
            f[0].data = d.astype("f32")

            SPEC_FITS_FLATTENED_DESC = ("OUTDATA_PATH", "",
                                        ".spec_flattened.fits")
            fout = igr_storage.get_path(SPEC_FITS_FLATTENED_DESC,
                                        tgt_basename)

            f.writeto(fout, clobber=True)

            db["a0v"].update(band, tgt_basename)



if 0:
    import matplotlib.pyplot as plt
    fig1 = plt.figure(2)
    ax = fig1.axes[0]
    for lam in [2.22112, 2.22328, 2.22740, 2.23106]:
        ax.axvline(lam)
    ax.set_title(objname)
    ax.set_xlim(2.211, 2.239)
    ax.set_ylim(0.69, 1.09)


if 0:


            def check_lsf(bins, lsf_list):
                from matplotlib.figure import Figure
                fig = Figure()
                ax = fig.add_subplot(111)

                xx = 0.5*(bins[1:]+bins[:-1])
                for hh0 in lsf_list:
                    peak1, peak2 = max(hh0), -min(hh0)
                    ax.plot(xx, hh0/(peak1+peak2),
                            "-", color="0.5")

                hh0 = np.sum(lsf_list, axis=0)
                peak1, peak2 = max(hh0), -min(hh0)
                ax.plot(0.5*(bins[1:]+bins[:-1]), hh0/(peak1+peak2),
                        "-", color="k")

                return fig


            def fit_lsf(bins, lsf_list):

                xx = 0.5*(bins[1:]+bins[:-1])
                hh0 = np.sum(lsf_list, axis=0)
                peak_ind1, peak_ind2 = np.argmax(hh0), np.argmin(hh0)
                # peak1, peak2 =



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

#     band = "K"
#     igr_path = IGRINSPath(utdate)

#     igrins_files = IGRINSFiles(igr_path)

#     fn = "%s.recipes" % utdate
#     recipe_list = load_recipe_list(fn)
#     recipe_dict = make_recipe_dict(recipe_list)

#     # igrins_log = IGRINSLog(igr_path, log_today)



#     if 0:
#         recipe = "A0V_AB"

#         DO_STD = True
#         FIX_TELLURIC=False

#     if 0:
#         recipe = "STELLAR_AB"

#         DO_STD = False
#         FIX_TELLURIC=True

#     if 1:
#         recipe = "EXTENDED_AB"

#         DO_STD = False
#         FIX_TELLURIC=True

#     if 0:
#         recipe = "EXTENDED_ONOFF"

#         DO_STD = False
#         FIX_TELLURIC=True


#     abba_list = recipe_dict[recipe]

#     do_interactive_figure=False

# if 1:
#     #abba  = abba_list[7] # GSS 30
#     #abba  = abba_list[11] # GSS 32
#     #abba  = abba_list[14] # Serpens 2
#     abba  = abba_list[6] # Serpens 15
#     do_interactive_figure=True

# for abba in abba_list:

#     objname = abba[-1][0]
#     print objname
#     obsids = abba[0]
#     frametypes = abba[1]


if __name__ == "__main__":
    import sys

    utdate = sys.argv[1]
    bands = "HK"
    starting_obsids = None

    if len(sys.argv) >= 3:
        bands = sys.argv[2]

    if len(sys.argv) >= 4:
        starting_obsids = sys.argv[3]

    a0v_ab(utdate, refdate="20140316", bands=bands,
           starting_obsids=starting_obsids)
