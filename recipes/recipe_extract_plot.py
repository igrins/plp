import os
import numpy as np

from libs.path_info import IGRINSPath


def plot_spec(utdate, refdate="20140316", bands="HK",
              starting_obsids=None, interactive=False,
              recipe_name = "ALL_RECIPES",
              config_file="recipe.config",
              threshold_a0v=0.2,
              multiply_model_a0v=False,
              html_output=False):

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    if not bands in ["H", "K", "HK"]:
        raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    fn = config.get_value('RECIPE_LOG_PATH', utdate)
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
        recipe_name = s[2]["RECIPE"].strip()
        objname = s[2]["OBJNAME"].strip()

        if recipe_name not in ["A0V_AB", "STELLAR_AB",
                               "A0V_ONOFF", "STELLAR_ONOFF",
                               "EXTENDED_AB", "EXTENDED_ONOFF"]:
            continue

        for band in bands:
            process_abba_band(recipe_name, utdate, refdate, band,
                              obsids, frametypes, config,
                              do_interactive_figure=interactive,
                              threshold_a0v=threshold_a0v,
                              objname=objname,
                              multiply_model_a0v=multiply_model_a0v,
                              html_output=html_output)


def process_abba_band(recipe, utdate, refdate, band, obsids, frametypes,
                      config,
                      do_interactive_figure=False,
                      threshold_a0v=0.1,
                      objname="",
                      multiply_model_a0v=False,
                      html_output=False):

    from libs.products import ProductDB, PipelineStorage

    if recipe == "A0V_AB":

        FIX_TELLURIC=False

    elif recipe == "A0V_ONOFF":

        FIX_TELLURIC=False

    elif recipe == "STELLAR_AB":

        FIX_TELLURIC=True

    elif recipe == "STELLAR_ONOFF":

        FIX_TELLURIC=True

    elif recipe == "EXTENDED_AB":

        FIX_TELLURIC=True

    elif recipe == "EXTENDED_ONOFF":

        FIX_TELLURIC=True

    else:
        raise ValueError("Unsupported Recipe : %s" % recipe)

    if 1:

        igr_path = IGRINSPath(config, utdate)

        igr_storage = PipelineStorage(igr_path)

        obj_filenames = igr_path.get_filenames(band, obsids)

        master_obsid = obsids[0]

        tgt_basename = os.path.splitext(os.path.basename(obj_filenames[0]))[0]

        db = {}
        basenames = {}

        db_types = ["flat_off", "flat_on", "thar", "sky"]

        for db_type in db_types:

            db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                        "%s.db" % db_type,
                                                        )
            db[db_type] = ProductDB(db_name)

        # db on output path
        db_types = ["a0v"]

        for db_type in db_types:

            db_name = igr_path.get_section_filename_base("OUTDATA_PATH",
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
        from libs.storage_descriptions import SKY_WVLSOL_JSON_DESC

        sky_basename = db["sky"].query(band, master_obsid)
        wvlsol_products = igr_storage.load([SKY_WVLSOL_JSON_DESC],
                                           sky_basename)[SKY_WVLSOL_JSON_DESC]

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = map(np.array, wvlsol_products["wvl_sol"])



    # prepare i1i2_list
    from libs.storage_descriptions import ORDER_FLAT_JSON_DESC
    prod = igr_storage.load([ORDER_FLAT_JSON_DESC],
                            basenames["flat_on"])[ORDER_FLAT_JSON_DESC]

    new_orders = prod["orders"]
    i1i2_list_ = prod["i1i2_list"]


    order_indices = []

    for o in orders_w_solutions:
        o_new_ind = np.searchsorted(new_orders, o)
        order_indices.append(o_new_ind)

    i1i2_list = get_fixed_i1i2_list(order_indices, i1i2_list_)


    from libs.storage_descriptions import (SPEC_FITS_DESC,
                                           SN_FITS_DESC)

    if 1: # load target spectrum
        tgt_spec_ = igr_storage.load([SPEC_FITS_DESC],
                                     tgt_basename)[SPEC_FITS_DESC]
        tgt_spec = list(tgt_spec_.data)

        tgt_sn_ = igr_storage.load([SN_FITS_DESC],
                                   tgt_basename)[SN_FITS_DESC]
        tgt_sn = list(tgt_sn_.data)

    fig_list = []

    # telluric
    if 1: #FIX_TELLURIC:
        A0V_basename = db["a0v"].query(band, master_obsid)

        from libs.storage_descriptions import SPEC_FITS_FLATTENED_DESC
        telluric_cor_ = igr_storage.load([SPEC_FITS_FLATTENED_DESC],
                                         A0V_basename)[SPEC_FITS_FLATTENED_DESC]

        #A0V_path = ProductPath(igr_path, A0V_basename)
        #fn = A0V_path.get_secondary_path("spec_flattened.fits")
        telluric_cor = list(telluric_cor_.data)


        a0v_spec_ = igr_storage.load([SPEC_FITS_DESC],
                                     A0V_basename)[SPEC_FITS_DESC]

        a0v_spec = list(a0v_spec_.data)


        if 1:


            if do_interactive_figure:
                from matplotlib.pyplot import figure as Figure
            else:
                from matplotlib.figure import Figure
            fig1 = Figure(figsize=(12,6))
            fig_list.append(fig1)

            ax1a = fig1.add_subplot(211)
            ax1b = fig1.add_subplot(212, sharex=ax1a)

            for wvl, s, sn in zip(wvl_solutions, tgt_spec, tgt_sn):
                #s[s<0] = np.nan
                #sn[sn<0] = np.nan

                ax1a.plot(wvl, s)
                ax1b.plot(wvl, sn)

            ax1a.set_ylabel("Counts [DN]")
            ax1b.set_ylabel("S/N per Res. Element")
            ax1b.set_xlabel("Wavelength [um]")

            ax1a.set_title(objname)


        if FIX_TELLURIC:

            fig2 = Figure(figsize=(12,6))
            fig_list.append(fig2)

            ax2a = fig2.add_subplot(211)
            ax2b = fig2.add_subplot(212, sharex=ax2a)

            #from libs.stddev_filter import window_stdev



            tgt_spec_cor = []
            #for s, t in zip(s_list, telluric_cor):
            for s, t, t2 in zip(tgt_spec, a0v_spec, telluric_cor):

                st = s/t
                #print np.percentile(t[np.isfinite(t)], 95), threshold_a0v
                t0 = np.percentile(t[np.isfinite(t)], 95)*threshold_a0v
                st[t<t0] = np.nan

                st[t2 < threshold_a0v] = np.nan

                tgt_spec_cor.append(st)


            if multiply_model_a0v:
                # multiply by A0V model
                from libs.a0v_spec import A0VSpec
                a0v_model = A0VSpec()

                a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                                           flatten=True,
                                                           smooth_pixel=32)
                for wvl, s in zip(wvl_solutions,
                                  tgt_spec_cor):

                    aa = a0v_interp1d(wvl)
                    s *= aa


            for wvl, s, t in zip(wvl_solutions,
                                 tgt_spec_cor,
                                 telluric_cor):

                ax2a.plot(wvl, t, "0.8", zorder=0.5)
                ax2b.plot(wvl, s, zorder=0.5)


            s_max_list = []
            s_min_list = []
            for s in tgt_spec_cor[3:-3]:
                s_max_list.append(np.nanmax(s))
                s_min_list.append(np.nanmin(s))
            s_max = np.max(s_max_list)
            s_min = np.min(s_min_list)
            ds_pad = 0.05 * (s_max - s_min)

            ax2a.set_ylabel("A0V flattened")
            ax2a.set_ylim(-0.05, 1.1)
            ax2b.set_ylabel("Target / A0V")
            ax2b.set_xlabel("Wavelength [um]")

            ax2b.set_ylim(s_min-ds_pad, s_max+ds_pad)
            ax2a.set_title(objname)



    # save figures
    if fig_list:
        for fig in fig_list:
            fig.tight_layout()

        figout = igr_path.get_section_filename_base("QA_PATH",
                                                    "spec_"+tgt_basename,
                                                    "spec_"+tgt_basename)
        #figout = obj_path.get_secondary_path("spec", "spec_dir")
        from libs.qa_helper import figlist_to_pngs
        figlist_to_pngs(figout, fig_list)

    # save html

    if html_output:
        dirname = config.get_value('HTML_PATH', utdate)
        objroot = "%04d" % (master_obsid,)
        html_save(utdate, dirname, objroot, band,
                  orders_w_solutions, wvl_solutions,
                  tgt_spec, tgt_sn, i1i2_list)

        if FIX_TELLURIC:
            objroot = "%04dA0V" % (master_obsid,)
            html_save(utdate, dirname, objroot, band,
                      orders_w_solutions, wvl_solutions,
                      telluric_cor, tgt_spec_cor, i1i2_list,
                      spec_js_name="jj_a0v.js")


    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()


def get_fixed_i1i2_list(order_indices, i1i2_list):
    i1i2_list2 = []
    for o_index in order_indices:

        i1i2_list2.append(i1i2_list[o_index])
    return i1i2_list2


def html_save(utdate, dirname, objroot, band,
              orders_w_solutions, wvl_solutions,
              tgt_spec, tgt_sn, i1i2_list,
              spec_js_name="jj.js"):

        wvl_list_html, s_list_html, sn_list_html = [], [], []

        for wvl, s, sn, (i1, i2) in zip(wvl_solutions,
                                        tgt_spec, tgt_sn,
                                        i1i2_list):

            sl = slice(i1, i2)

            wvl_list_html.append(wvl[sl])
            s_list_html.append(s[sl])
            sn_list_html.append(sn[sl])


        save_for_html(dirname, objroot, band,
                      orders_w_solutions,
                      wvl_list_html, s_list_html, sn_list_html)

        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('jinja_templates'))
        spec_template = env.get_template('spec.html')

        master_root = "igrins_spec_%s_%s" % (objroot, band)
        jsname = master_root + ".js"
        ss = spec_template.render(utdate=utdate,
                                  jsname=jsname,
                                  spec_js_name=spec_js_name)
        htmlname = master_root + ".html"
        open(os.path.join(dirname, htmlname), "w").write(ss)





def save_for_html(dir, name, band, orders, wvl_sol, s_list1, s_list2):
    import pandas as pd
    from libs.path_info import ensure_dir
    ensure_dir(dir)

    # Pandas requires the byte order of data (from fits) needs to be
    # converted to native byte order of the computer this script is
    # running.
    wvl_sol = [w.byteswap().newbyteorder() for w in wvl_sol]
    s_list1 = [s1.byteswap().newbyteorder() for s1 in s_list1]
    s_list2 = [s2.byteswap().newbyteorder() for s2 in s_list2]

    df_even_odd = {}
    for o, wvl, s in zip(orders, wvl_sol, s_list1):
        oo = ["even", "odd"][o % 2]
        dn = 'order_%s'%oo
        df = pd.DataFrame({dn: s},
                          index=wvl)
        df[dn][wvl[0]] = "NaN"
        df[dn][wvl[-1]] = "NaN"

        df_even_odd.setdefault(oo, []).append(df)

    df_list = [pd.concat(v).fillna("NaN") for v in df_even_odd.values()]
    df1 = df_list[0].join(df_list[1:], how="outer")

    #df_list = []
    df_even_odd = {}
    for o, wvl, s in zip(orders, wvl_sol, s_list2):
        oo = ["even", "odd"][o % 2]
        dn = 'order_%s'%oo
        df = pd.DataFrame({dn: s},
                          index=wvl)

        df[dn][wvl[0]] = "NaN"
        df[dn][wvl[-1]] = "NaN"

        df_even_odd.setdefault(oo, []).append(df)

        #df_list.append(df)
    df_list = [pd.concat(v).fillna("NaN") for v in df_even_odd.values()]
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
