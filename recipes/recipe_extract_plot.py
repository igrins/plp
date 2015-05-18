import os
import numpy as np

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

    selected.sort()
    for s in selected:
        obsids = s[0]
        frametypes = s[1]
        recipe_name = s[2]["RECIPE"].strip()
        objname = s[2]["OBJNAME"].strip()

        target_type = recipe_name.split("_")[0]

        if target_type not in ["A0V", "STELLAR", "EXTENDED"]:
            print "Unsupported recipe : %s" % recipe_name
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

    target_type, nodding_type = recipe.split("_")

    if target_type in ["A0V"]:
        FIX_TELLURIC=False
    elif target_type in ["STELLAR", "EXTENDED"]:
        FIX_TELLURIC=True
    else:
        raise ValueError("Unknown recipe : %s" % recipe)


    from recipe_extract_base import RecipeExtractPR
    extractor = RecipeExtractPR(utdate, band,
                                obsids, config)

    master_obsid = extractor.pr.master_obsid
    igr_path = extractor.pr.igr_path


    tgt = extractor.get_oned_spec_helper()

    orders_w_solutions = extractor.orders_w_solutions

    # if wavelengths list are sorted in increasing order of
    # wavelength, recerse the order list
    if tgt.um[0][0] < tgt.um[-1][0]:
        orders_w_solutions = orders_w_solutions[::-1]

    if FIX_TELLURIC:

        A0V_basename = extractor.basenames["a0v"]
        a0v = extractor.get_oned_spec_helper(A0V_basename)


        tgt_spec_cor = get_tgt_spec_cor(tgt, a0v,
                                        threshold_a0v,
                                        multiply_model_a0v)

    # prepare i1i2_list
    i1i2_list = get_i1i2_list(extractor,
                              orders_w_solutions)




    fig_list = []


    if 1:

        if do_interactive_figure:
            from matplotlib.pyplot import figure as Figure
        else:
            from matplotlib.figure import Figure
        fig1 = Figure(figsize=(12,6))
        fig_list.append(fig1)

        ax1a = fig1.add_subplot(211)
        ax1b = fig1.add_subplot(212, sharex=ax1a)

        for wvl, s, sn in zip(tgt.um,
                              tgt.spec, tgt.sn):
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

        for wvl, s, t in zip(tgt.um,
                             tgt_spec_cor,
                             a0v.flattened):

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

        tgt_basename = extractor.pr.tgt_basename
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
                  orders_w_solutions, tgt.um,
                  tgt.spec, tgt.sn, i1i2_list)

        if FIX_TELLURIC:
            objroot = "%04dA0V" % (master_obsid,)
            html_save(utdate, dirname, objroot, band,
                      orders_w_solutions, tgt.um,
                      a0v.flattened, tgt_spec_cor, i1i2_list,
                      spec_js_name="jj_a0v.js")


    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()


def get_fixed_i1i2_list(order_indices, i1i2_list):
    i1i2_list2 = []
    for o_index in order_indices:

        i1i2_list2.append(i1i2_list[o_index])
    return i1i2_list2


def get_i1i2_list(extractor, orders_w_solutions):
    from libs.storage_descriptions import ORDER_FLAT_JSON_DESC
    prod = extractor.igr_storage.load1(ORDER_FLAT_JSON_DESC,
                                       extractor.basenames["flat_on"])

    new_orders = prod["orders"]
    i1i2_list_ = prod["i1i2_list"]

    order_indices = []

    for o in orders_w_solutions:
        o_new_ind = np.searchsorted(new_orders, o)
        order_indices.append(o_new_ind)

    i1i2_list = get_fixed_i1i2_list(order_indices, i1i2_list_)
    return i1i2_list

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


def get_tgt_spec_cor(tgt, a0v, threshold_a0v, multiply_model_a0v):
    tgt_spec_cor = []
    #for s, t in zip(s_list, telluric_cor):
    for s, t, t2 in zip(tgt.spec,
                        a0v.spec,
                        a0v.flattened):

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
        for wvl, s in zip(tgt.um,
                          tgt_spec_cor):

            aa = a0v_interp1d(wvl)
            s *= aa


    return tgt_spec_cor
