import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSLog, IGRINSFiles
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

def save_for_html(dir, name, band, orders, wvl_sol, s_list1, s_list2):
    from libs.path_info import ensure_dir
    ensure_dir(dir)

    import pandas as pd
    df_list = []
    for o, wvl, s in zip(orders, wvl_sol, s_list1):
        df = pd.DataFrame({'order%03d'%o: s},
                          index=wvl)
        df_list.append(df)
    df1 = df_list[0].join(df_list[1:], how="outer")

    df_list = []
    for o, wvl, s in zip(orders, wvl_sol, s_list2):
        df = pd.DataFrame({'order%03d'%o: s},
                          index=wvl)
        df_list.append(df)
    df2 = df_list[0].join(df_list[1:], how="outer")

    # mm = (df0.index[0] < wvl_thar) & (wvl_thar < df0.index[-1])
    # ss = s_thar[mm]
    # ww = wvl_thar[mm]
    # ww3 = np.empty((len(ww)*3,))
    # ss3 = np.empty((len(ww)*3,))
    # ww3[::3] = ww
    # ww3[1::3] = ww
    # ww3[2::3] = ww
    # ss3[::3] = ss
    # ss3[1::3] = 0
    # ss3[2::3] = np.nan

    # df_m = pd.DataFrame({'model': -ss3},
    #                     index=ww3)

    # model_output = "model_spec_%s_%s.csv.html" % (object_name, band)

    igrins_spec_output1 = "igrins_spec_%s_%s_fig1.csv.html" % (name, band)
    igrins_spec_output2 = "igrins_spec_%s_%s_fig2.csv.html" % (name, band)

    # if object_name == "ur" and uranium_model_urne: # special case for Ur-Ne model
    #     import astropy.io.fits as pyfits
    #     # Ur-Neon : limited H-band only
    #     ur_w = pyfits.open("wvl_literatures/une_wave.fits")[0].data/1.e4
    #     ur_s = pyfits.open("wvl_literatures/une_spec.fits")[0].data

    #     df_m = pd.DataFrame({'model': -ur_s},
    #                         index=ur_w)

    #     model_output = "model_spec_urne_%s_%s.csv.html" % (object_name, band)



    df1.to_csv(os.path.join(dir, igrins_spec_output1))
    df2.to_csv(os.path.join(dir, igrins_spec_output2))
    #df_m.to_csv(model_output)

    wvlminmax_list = []
    for o, wvl in zip(orders, wvl_sol):
        wvlminmax_list.append([min(wvl), max(wvl)])

    f = open(os.path.join(dir, "igrins_spec_%s_%s.js"%(name, band)),"w")
    f.write('name="%s : %s";\n' % (name,band))
    f.write("wvl_ranges=")
    f.write(str(wvlminmax_list))
    f.write(";\n")
    f.write("order_minmax=[%d,%d];\n" % (orders[0], orders[-1]))
    # if object_name == "thar":
    #     if band == "K":
    #         f.write("value_max1=70.;\n")
    #         f.write("value_max2=150.;\n")
    #     else:
    #         f.write("value_max1=300.;\n")
    #         f.write("value_max2=600.;\n")
    # elif object_name == "sky":
    #     if band == "K":
    #         f.write("value_max1=70.;\n")
    #         f.write("value_max2=150.;\n")
    #     else:
    #         f.write("value_max1=90.;\n")
    #         f.write("value_max2=250.;\n")
    # elif object_name == "ur":
    #     if band == "K":
    #         f.write("value_max1=70.;\n")
    #         f.write("value_max2=150.;\n")
    #     else:
    #         f.write("value_max1=1500.;\n")
    #         f.write("value_max2=1500.;\n")

    f.write('first_filename = "%s";\n' % igrins_spec_output1)
    f.write('second_filename = "%s";\n' % igrins_spec_output2)

    f.close()



if __name__ == "__main__":

    from libs.recipes import load_recipe_list, make_recipe_dict
    from libs.products import PipelineProducts, ProductPath, ProductDB


    if 0:
        utdate = "20140316"
        # log_today = dict(flat_off=range(2, 4),
        #                  flat_on=range(4, 7),
        #                  thar=range(1, 2))
    elif 1:
        utdate = "20140707"
        # log_today = dict(flat_off=range(64, 74),
        #                  flat_on=range(74, 84),
        #                  thar=range(3, 8),
        #                  sky=[29],
        #                  HIP94620=[50,51,52,53],
        #                  HIP99742=[54, 55, 56, 57],
        #                  PCyg=[58,59,60, 61],
        #                  J1833=[42,43,44,45],
        #                  TWHya=[8, 9],
        #                  GammaOph=[38,39,40,41],
        #                  V889=[46,47,48,49],
        #                  G11=[32, 33],
        #                  PN_M2_43=[34, 35],
        #                  SagARing=[30,31],
        #                  WL16=[24,25,26,27])

    band = "H"
    igr_path = IGRINSPath(utdate)

    igrins_files = IGRINSFiles(igr_path)

    fn = "%s.recipes" % utdate
    recipe_list = load_recipe_list(fn)
    recipe_dict = make_recipe_dict(recipe_list)

    # igrins_log = IGRINSLog(igr_path, log_today)



    if 0:
        recipe = "A0V_AB"

        DO_STD = True
        FIX_TELLURIC=False

    if 0:
        recipe = "STELLAR_AB"

        DO_STD = False
        FIX_TELLURIC=True

    if 0:
        recipe = "EXTENDED_AB"

        DO_STD = False
        FIX_TELLURIC=True

    if 1:
        recipe = "EXTENDED_ONOFF"

        DO_STD = False
        FIX_TELLURIC=True


    abba_list = recipe_dict[recipe]

    do_interactive_figure=False

# if 1:
#     abba  = abba_list[1]

for abba in abba_list:

    #i = 0
    objname = abba[-1][0]
    obsids = abba[0]
    frametypes = abba[1]


    print objname


    if 1:

        obj_filenames = igrins_files.get_filenames(band, obsids)
        obj_path = ProductPath(igr_path, obj_filenames[0])
        obj_master_obsid = obsids[0]

        flatoff_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                            "flat_off.db"))
        flaton_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                           "flat_on.db"))
        thar_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                           "thar.db"))
        sky_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "sky.db"))

        A0V_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                        "A0V.db"))



        basename = sky_db.query(band, obj_master_obsid)
        sky_path = ProductPath(igr_path, basename)
        raw_spec_products = PipelineProducts.load(sky_path.get_secondary_path("raw_spec"))

        basename = flaton_db.query(band, obj_master_obsid)
        flaton_path = ProductPath(igr_path, basename)

        aperture_solution_products = PipelineProducts.load(flaton_path.get_secondary_path("aperture_solutions"))

        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        # thar

        basename = thar_db.query(band, obj_master_obsid)
        thar_path = ProductPath(igr_path, basename)
        fn = thar_path.get_secondary_path("median_spectra")
        thar_products = PipelineProducts.load(fn)

        fn = thar_path.get_secondary_path("orderflat")
        orderflat_products = PipelineProducts.load(fn)

        # sky

        basename = sky_db.query(band, obj_master_obsid)
        sky_path = ProductPath(igr_path, basename)
        fn = sky_path.get_secondary_path("wvlsol_v1")
        wvlsol_products = PipelineProducts.load(fn)

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = wvlsol_products["wvl_sol"]

        # flat on & off

        basename = flatoff_db.query(band, obj_master_obsid)
        flatoff_path = ProductPath(igr_path, basename)
        fn = flatoff_path.get_secondary_path("flat_off_params")
        flatoff_products = PipelineProducts.load(fn)

        basename = flaton_db.query(band, obj_master_obsid)
        flaton_path = ProductPath(igr_path, basename)
        fn = flaton_path.get_secondary_path("flat_on_params")
        flaton_products = PipelineProducts.load(fn)


    # telluric
    if FIX_TELLURIC:
        basename = A0V_db.query(band, obj_master_obsid)
        A0V_path = ProductPath(igr_path, basename)
        fn = A0V_path.get_secondary_path("spec_flattened.fits")
        telluric_cor = list(pyfits.open(fn)[0].data)
        print fn

    # if 1:

    #     sky_names = [igrins_log.get_filename(band, fn) for fn \
    #                   in igrins_log.log["sky"]]

    #     sky_master_fn_ = os.path.splitext(os.path.basename(sky_names[0]))[0]
    #     sky_master_fn = igr_path.get_secondary_calib_filename(sky_master_fn_)

    #     raw_spec_products = PipelineProducts.load(sky_master_fn+".raw_spec")


    # if 1: # load aperture product
    #     flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
    #     flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
    #     aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


    #     aperture_solution_products = PipelineProducts.load(aperture_solutions_name)

    #     bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        # from libs.process_thar import ThAr

        # thar_names = [igrins_log.get_filename(band, fn) for fn \
        #               in igrins_log.log["thar"]]
        # thar = ThAr(thar_names)

        # fn = thar.get_product_name(igr_path)

        # thar_products = PipelineProducts.load(fn)


    # if 1:
    #     from libs.master_calib import load_sky_ref_data

    #     ref_utdate = "20140316"

    #     sky_ref_data = load_sky_ref_data(ref_utdate, band)


    #     ohlines_db = sky_ref_data["ohlines_db"]
    #     ref_ohline_indices = sky_ref_data["ohline_indices"]

    if 1: # make aperture

        _o_s = dict(zip(raw_spec_products["orders"], bottomup_solutions))
        ap =  Apertures(orders_w_solutions,
                        [_o_s[o] for o in orders_w_solutions])


        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()
        order_map2 = ap.make_order_map(mask_top_bottom=True)


### real

    if 1:

        # # INPUT
        # flat_off_filenames = [igrins_log.get_filename(band, i) for i \
        #                       in igrins_log.log["flat_off"]]


        # flat_off_name_ = flat_off_filenames[0]
        # flat_off_name_ = os.path.splitext(flat_off_name_)[0] + ".flat_off_params"
        # flat_off_name = igr_path.get_secondary_calib_filename(flat_off_name_)
        # flatoff_products = PipelineProducts.load(flat_off_name)


        # # load flat on products
        # flat_on_filenames = [igrins_log.get_filename(band, i) for i \
        #                      in igrins_log.log["flat_on"]]
        # flat_on_name_ = flat_on_filenames[0]
        # flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".flat_on_params"
        # flat_on_name = igr_path.get_secondary_calib_filename(flat_on_name_)

        # flaton_products = PipelineProducts.load(flat_on_name)

        pix_mask  = flatoff_products["hotpix_mask"] | flaton_products["deadpix_mask"]



        #
        # flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
        # flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
        # aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


        # aperture_solution_products = PipelineProducts.load(aperture_solutions_name)


        # fn = thar.get_product_name(igr_path)+".orderflat"
        # orderflat_products = PipelineProducts.load(fn)
        orderflat = orderflat_products['order_flat']
        orderflat[pix_mask] = np.nan

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


            dx = 100

            if IF_POINT_SOURCE: # if point source
                # for point sources, variance estimation becomes wrong
                # if lenth of two is different,
                assert len(a_list) == len(b_list)

            # a_b != 1 for the cases when len(a) != len(b)
            a_b = float(len(a_list)) / len(b_list)

            a_data = np.sum(a_list, axis=0)
            b_data = np.sum(b_list, axis=0)

            data_minus = a_data - a_b*b_data
            data_minus0 = data_minus

            from libs.destriper import destriper
            if 1:

                data_minus = destriper.get_destriped(data_minus,
                                                     ~np.isfinite(data_minus),
                                                     pattern=64)

            data_minus_flattened = data_minus / orderflat

            # for variance, we need a square of a_b
            data_plus = (a_data + (a_b**2)*b_data)

            bias_mask = flaton_products["flat_mask"] & (order_map2 > 0)
            import scipy.ndimage as ni
            bias_mask2 = ni.binary_dilation(bias_mask)

            import instrument_parameters
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

        #
        import astropy.io.fits as pyfits
        fn = sky_path.get_secondary_path("slitoffset_map.fits")
        slitoffset_map = pyfits.open(fn)[0].data


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
            r = PipelineProducts("1d specs",
                                 combined_image_raw=data_minus0,
                                 combined_image=data_minus_flattened,
                                 variance_map=variance_map2
                                 )


            fn = obj_path.get_secondary_path("spec_params")
            r.save(fn, masterhdu=master_hdu)

        fig_list = []
        if 1:
            new_orders = orderflat_products["orders"]
            fitted_response = orderflat_products["fitted_responses"]
            i1i2_list = orderflat_products["i1i2_list"]

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
                ax1.plot(wvl[sl][mmm], s[sl][mmm])


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
            new_orders = orderflat_products["orders"]
            fitted_response = orderflat_products["fitted_responses"]
            i1i2_list = orderflat_products["i1i2_list"]

            fig2 = Figure(figsize=(12,8))
            fig_list.append(fig2)

            ax1 = fig2.add_subplot(211)
            for o, wvl, s in zip(ap.orders, wvl_solutions, s_list):
                o_new_ind = np.searchsorted(new_orders, o)
                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                res = fitted_response[o_new_ind]
                #ax1.plot(wvl[sl], (s/res)[sl])

            from libs.master_calib import get_master_calib_abspath
            fn = get_master_calib_abspath("A0V/vegallpr25.50000resam5")
            d = np.genfromtxt(fn)

            wvl, flux, cont = (d[:,i] for i in [0, 1, 2])
            ax2 = fig2.add_subplot(212, sharex=ax1)
            wvl = wvl/1000.
            mask_H = (1.450 < wvl) & (wvl < 1.850)

            fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
            telluric = pyfits.open(fn)[1].data
            telluric_lam = telluric["lam"]
            tel_mask_H = (1.450 < telluric_lam) & (telluric_lam < 1.850)
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

        if 1:
            figout = obj_path.get_secondary_path("spec", "spec_dir")
            from libs.qa_helper import figlist_to_pngs
            figlist_to_pngs(figout, fig_list)
            #from libs.qa_helper import figlist_to_json
            #figlist_to_json(figout, fig_list)
            dirname = "html_%s" % utdate
            objroot = "%04d" % (obsids[0],)
            save_for_html(dirname, objroot, band, ap.orders,
                          wvl_list_html, s_list_html, sn_list_html)

    if 1:
        fn = sky_path.get_secondary_path("wvlsol_v1.fits")
        f = pyfits.open(fn)

        d = np.array(s_list)
        d[~np.isfinite(d)] = 0.
        f[0].data = d.astype("f32")
        fout = obj_path.get_secondary_path("spec.fits")
        f.writeto(fout, clobber=True)


        if DO_STD:
            d = np.array(telluric_cor)
            d[~np.isfinite(d)] = 0.
            f[0].data = d.astype("f32")
            fout = obj_path.get_secondary_path("spec_flattened.fits")
            f.writeto(fout, clobber=True)

            from libs.products import ProductDB
            import os
            A0V_db = ProductDB(os.path.join(igr_path.secondary_calib_path,
                                             "A0V.db"))
            A0V_db.update(band, obj_path.basename)



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
