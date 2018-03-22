import pandas as pd

from ..pipeline.steps import Step

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper


def set_basename_postfix(obsset):
    # This only applies for the output name
    obsset.set_basename_postfix(basename_postfix="_sky")


def _plot_source_spec(fig, tgt, objname=""):

    ax1a = fig.add_subplot(211)
    ax1b = fig.add_subplot(212, sharex=ax1a)

    for wvl, s, sn in zip(tgt.um,
                          tgt.spec, tgt.sn):
        #s[s<0] = np.nan
        #sn[sn<0] = np.nan

        ax1a.plot(wvl, s)
        ax1b.plot(wvl, sn)

    ax1a.set_ylabel("Counts [DN]")
    ax1b.set_ylabel("S/N per Res. Element")
    ax1b.set_xlabel("Wavelength [um]")

    if objname:
        ax1a.set_title(objname)


def _plot_div_a0v_spec(fig, tgt, obsset):
    # FIXME: This is simple copy from old version.

    if True:

        if (a0v_obsid is None) or (a0v_obsid == "1"):
            A0V_basename = extractor.basenames["a0v"]
        else:
            A0V_basename = "SDC%s_%s_%04d" % (band, utdate, int(a0v_obsid))
            print(A0V_basename)

        a0v = extractor.get_oned_spec_helper(A0V_basename,
                                             basename_postfix=basename_postfix)

        tgt_spec_cor = get_tgt_spec_cor(tgt, a0v,
                                        threshold_a0v,
                                        multiply_model_a0v,
                                        config)

    ax2a = fig2.add_subplot(211)
    ax2b = fig2.add_subplot(212, sharex=ax2a)

    #from ..libs.stddev_filter import window_stdev

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


def _save_to_pngs():
    # FIXME: This is copy from old version. Need to modify it.
    # tgt_basename = extractor.pr.tgt_basename
    tgt_basename = mastername

    dirname = "spec_"+tgt_basename
    basename_postfix_s = basename_postfix if basename_postfix is not None else ""
    filename_prefix = "spec_" + tgt_basename + basename_postfix_s
    figout = igr_path.get_section_filename_base("QA_PATH",
                                                filename_prefix,
                                                dirname)
    #figout = obj_path.get_secondary_path("spec", "spec_dir")
    from ..libs.qa_helper import figlist_to_pngs
    figlist_to_pngs(figout, fig_list)


def _save_to_html():
    i1i2_list = get_i1i2_list(extractor,
                              orders_w_solutions)

    if basename_postfix is not None:
        igr_log.warn("For now, no html output is generated if basename-postfix option is used")
    else:
        dirname = config.get_value('HTML_PATH', utdate)
        from ..libs.path_info import get_zeropadded_groupname

        objroot = get_zeropadded_groupname(groupname)
        html_save(utdate, dirname, objroot, band,
                  orders_w_solutions, tgt.um,
                  tgt.spec, tgt.sn, i1i2_list)

        if FIX_TELLURIC:
            objroot = get_zeropadded_groupname(groupname)+"A0V"
            html_save(utdate, dirname, objroot, band,
                      orders_w_solutions, tgt.um,
                      a0v.flattened, tgt_spec_cor, i1i2_list,
                      spec_js_name="jj_a0v.js")


def plot_spec(obsset, interactive=False):
    recipe = obsset.recipe_name
    target_type, nodding_type = recipe.split("_")

    if target_type in ["A0V"]:
        FIX_TELLURIC = False
    elif target_type in ["STELLAR", "EXTENDED"]:
        FIX_TELLURIC = True
    else:
        raise ValueError("Unknown recipe : %s" % recipe)

    from ..igrins_libs.oned_spec_helper import OnedSpecHelper
    tgt = OnedSpecHelper(obsset)

    do_interactive_figure = interactive

    if do_interactive_figure:
        from matplotlib.pyplot import figure as Figure
    else:
        from matplotlib.figure import Figure

    fig_list = []

    fig1 = Figure(figsize=(12, 6))
    fig_list.append(fig1)

    _plot_source_spec(fig1, tgt)

    # if FIX_TELLURIC:
    #     fig1 = Figure(figsize=(12, 6))
    #     fig_list.append(fig1)

    #     _plot_div_a0v_spec(fig1, tgt, obsset)

    if fig_list:
        for fig in fig_list:
            fig.tight_layout()

    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()


steps = [Step("Set basename_postfix", set_basename_postfix),
         Step("Plot spec", plot_spec, interactive=False),
]
