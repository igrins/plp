import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSFiles
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts


def a0v_ab(utdate, refdate="20140316", bands="HK",
           starting_obsids=None,
           interactive=False,
           config_file="recipe.config"):
    recipe = "A0V_AB"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids, interactive=interactive,
             config_file=config_file)

def stellar_ab(utdate, refdate="20140316", bands="HK",
               starting_obsids=None,
               interactive=False,
               config_file="recipe.config"):
    recipe = "STELLAR_AB"
    abba_all(recipe, utdate, refdate=refdate, bands=bands,
             starting_obsids=starting_obsids,
             interactive=interactive,
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


def plot_spec(utdate, refdate="20140316", bands="HK",
              starting_obsids=None, interactive=False,
              recipe_name = "ALL_RECIPES",
              config_file="recipe.config",
              threshold_a0v=0.1):

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    if not bands in ["H", "K", "HK"]:
        raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    fn = "%s.recipes" % utdate
    from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
    recipe = Recipes(fn)

    if starting_obsids is not None:
        starting_obsids = map(int, starting_obsids.split(","))

    # recipe_name = "ALL_RECIPES"
    selected = recipe.select(recipe_name, starting_obsids)
    if not selected:
        print "no recipe of with matching arguments is found"

    for s in selected:
        obsids = s[0]
        frametypes = s[1]
        recipe_name = s[2]["RECIPE"]
        print recipe_name
        for band in bands:
            process_abba_band(recipe_name, utdate, refdate, band,
                              obsids, frametypes, config,
                              do_interactive_figure=interactive,
                              threshold_a0v=threshold_a0v)


def process_abba_band(recipe, utdate, refdate, band, obsids, frametypes,
                      config,
                      do_interactive_figure=False,
                      threshold_a0v=0.1):

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
        wvl_solutions = map(np.array, wvlsol_products["wvl_sol"])



    # prepare i1i2_list
    from libs.process_flat import ORDER_FLAT_JSON_DESC
    prod = igr_storage.load([ORDER_FLAT_JSON_DESC],
                            basenames["flat_on"])[ORDER_FLAT_JSON_DESC]

    new_orders = prod["orders"]
    i1i2_list = prod["i1i2_list"]


    order_indices = []

    for o in orders_w_solutions:
        o_new_ind = np.searchsorted(new_orders, o)
        order_indices.append(o_new_ind)



    if 1: # load target spectrum
        SPEC_FITS_DESC = ("OUTDATA_PATH", "",
                          ".spec.fits")
        tgt_spec_ = igr_storage.load([SPEC_FITS_DESC],
                                     tgt_basename)[SPEC_FITS_DESC]
        tgt_spec = list(tgt_spec_.data)

        SN_FITS_DESC = ("OUTDATA_PATH", "",
                        ".sn.fits")
        tgt_sn_ = igr_storage.load([SN_FITS_DESC],
                                   tgt_basename)[SN_FITS_DESC]
        tgt_sn = list(tgt_sn_.data)


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


        SPEC_FITS_DESC = ("OUTDATA_PATH", "",
                                    ".spec.fits")
        a0v_spec_ = igr_storage.load([SPEC_FITS_DESC],
                                         A0V_basename)[SPEC_FITS_DESC]

        #A0V_path = ProductPath(igr_path, A0V_basename)
        #fn = A0V_path.get_secondary_path("spec_flattened.fits")
        a0v_spec = list(a0v_spec_.data)

        # fn = A0V_path.get_secondary_path("spec.fits")
        # telluric_cor = list(pyfits.open(fn)[0].data)
        #print fn



        fig_list = []
        if 1:


            if do_interactive_figure:
                from matplotlib.pyplot import figure as Figure
            else:
                from matplotlib.figure import Figure
            fig1 = Figure(figsize=(12,6))
            fig_list.append(fig1)

            fig2 = Figure(figsize=(12,6))
            fig_list.append(fig2)

            ax1a = fig1.add_subplot(211)
            ax1b = fig1.add_subplot(212, sharex=ax1a)

            ax2a = fig2.add_subplot(211)
            ax2b = fig2.add_subplot(212, sharex=ax2a)
            #from libs.stddev_filter import window_stdev

            for wvl, s, sn in zip(wvl_solutions, tgt_spec, tgt_sn):
                ax1a.plot(wvl, s)
                ax1b.plot(wvl, sn)




            wvl_list_html, s_list_html, sn_list_html = [], [], []

            for o_index, wvl, s, sn in zip(order_indices,
                                          wvl_solutions,
                                          tgt_spec, tgt_sn):

                i1, i2 = i1i2_list[o_index]
                sl = slice(i1, i2)
                #res = fitted_response[o_new_ind]
                #wvl, s = np.array(wvl), np.array(s)
                mmm = np.isfinite(s[sl])

                wvl_list_html.append(wvl[sl])
                s_list_html.append(s[sl])
                sn_list_html.append(sn[sl])
                #s_std = window_stdev(s, 25)
                #ax1.plot(wvl[sl], ni.median_filter(s[sl]/s_std[sl], 10), "g-")



            if FIX_TELLURIC:
                tgt_spec_cor = []
                #for s, t in zip(s_list, telluric_cor):
                for s, t in zip(tgt_spec, a0v_spec):

                    st = s/t
                    st[t<np.median(t)*threshold_a0v] = np.nan
                    tgt_spec_cor.append(st)
            else:
                tgt_spec_cor = tgt_spec

            # wvl_list_html, s_list_html, sn_list_html = [], [], []


            if FIX_TELLURIC:

                for wvl, s, t in zip(wvl_solutions,
                                     tgt_spec_cor,
                                     telluric_cor):

                    ax2a.plot(wvl, t, "0.8", zorder=0.5)
                    ax2b.plot(wvl, s, zorder=0.5)



            # ymax = 1.1*np.nanmax(tgt_spec[12])
            # ax1.set_ylim(0, ymax)

            # pixel_per_res_element = 3.7
            # ymax = 1.2*(s_list[12][1024]/v_list[12][1024]**.5*pixel_per_res_element**.5)
            # ax2.set_ylim(0, ymax)
            ax1b.set_ylabel("S/N per Res. Element")


        if 0: # IF_POINT_SOURCE: # if point source, try simple telluric factor for A0V
            # new_orders = orderflat_products["orders"]
            # # fitted_response = orderflat_products["fitted_responses"]
            # i1i2_list = orderflat_products["i1i2_list"]

            fig2 = Figure(figsize=(12,8))
            fig_list.append(fig2)

            ax1 = fig2.add_subplot(211)
            for o_index, wvl, s in zip(order_indices,
                                       wvl_solutions, s_list):
                i1, i2 = i1i2_list[o_index]
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

            trans = spl(wvl[mask_H])
            ax1.plot(wvl[mask_H], flux[mask_H]/cont[mask_H]*trans,
                     color="0.5", zorder=0.5)


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

            for o_index, wvl, s in zip(order_indices, wvl_solutions, s_list):

                i1, i2 = i1i2_list[o_index]
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

    # save figures
    if 1:
        figout = igr_path.get_section_filename_base("QA_PATH",
                                                    "spec_"+tgt_basename,
                                                    "spec_dir")
        #figout = obj_path.get_secondary_path("spec", "spec_dir")
        from libs.qa_helper import figlist_to_pngs
        figlist_to_pngs(figout, fig_list)

    # save html
    if 1:
        #from libs.qa_helper import figlist_to_json
        #figlist_to_json(figout, fig_list)
        dirname = config.get_value('HTML_PATH', utdate)
        objroot = "%04d" % (master_obsid,)
        save_for_html(dirname, objroot, band, orders_w_solutions,
                      wvl_list_html, s_list_html, sn_list_html)


    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()

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


import pandas as pd

def save_for_html(dir, name, band, orders, wvl_sol, s_list1, s_list2):
    from libs.path_info import ensure_dir
    ensure_dir(dir)

    wvl_sol = [w.byteswap().newbyteorder() for w in wvl_sol]
    s_list1 = [s1.byteswap().newbyteorder() for s1 in s_list1]
    s_list2 = [s2.byteswap().newbyteorder() for s2 in s_list2]

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

    igrins_spec_output1 = "igrins_spec_%s_%s_fig1.csv.html" % (name, band)
    igrins_spec_output2 = "igrins_spec_%s_%s_fig2.csv.html" % (name, band)


    df1.to_csv(os.path.join(dir, igrins_spec_output1))
    df2.to_csv(os.path.join(dir, igrins_spec_output2))

    wvlminmax_list = []
    for o, wvl in zip(orders, wvl_sol):
        wvlminmax_list.append([min(wvl), max(wvl)])

    f = open(os.path.join(dir, "igrins_spec_%s_%s.js"%(name, band)),"w")
    f.write('name="%s : %s";\n' % (name,band))
    f.write("wvl_ranges=")
    f.write(str(wvlminmax_list))
    f.write(";\n")
    f.write("order_minmax=[%d,%d];\n" % (orders[0], orders[-1]))

    f.write('first_filename = "%s";\n' % igrins_spec_output1)
    f.write('second_filename = "%s";\n' % igrins_spec_output2)

    f.close()

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
