import numpy as np
import pandas as pd

from ..pipeline.steps import Step, ArghFactoryWithShort

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper
from ..igrins_libs.oned_spec_helper import OnedSpecHelper


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


from ..igrins_libs.a0v_obsid import get_group2, get_a0v_obsid

def get_tgt_spec_cor(obsset, tgt, a0v, threshold_a0v, multiply_model_a0v):
    tgt_spec_cor = []
    #for s, t in zip(s_list, telluric_cor):
    for s, t, t2 in zip(tgt.spec,
                        a0v.spec,
                        a0v.flattened):

        st = s/t
        msk = np.isfinite(t)
        if np.any(msk):
            #print np.percentile(t[np.isfinite(t)], 95), threshold_a0v
            t0 = np.percentile(t[msk], 95)*threshold_a0v
            st[t<t0] = np.nan

            st[t2 < threshold_a0v] = np.nan

        tgt_spec_cor.append(st)


    if multiply_model_a0v:
        # multiply by A0V model
        d = obsset.rs.load_ref_data("VEGA_SPEC")
        from ..procedures.a0v_spec import A0VSpec
        a0v_model = A0VSpec(d)

        a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                                   flatten=True,
                                                   smooth_pixel=32)
        for wvl, s in zip(tgt.um,
                          tgt_spec_cor):

            aa = a0v_interp1d(wvl)
            s *= aa


    return tgt_spec_cor


def _plot_div_a0v_spec(fig, tgt, obsset, a0v="GROUP2", a0v_obsid=None,
                       threshold_a0v=0.1,
                       objname="",
                       multiply_model_a0v=False,
                       html_output=False,
                       a0v_basename_postfix=""):
    # FIXME: This is simple copy from old version.

    a0v_obsid = get_a0v_obsid(obsset, a0v, a0v_obsid)
    if a0v_obsid is None:
        a0v_obsid_ = obsset.query_resource_basename("a0v")
        a0v_obsid = obsset.rs.parse_basename(a0v_obsid_)

    a0v_obsset = type(obsset)(obsset.rs, "A0V_AB", [a0v_obsid], ["A"],
                              basename_postfix=a0v_basename_postfix)

    a0v = OnedSpecHelper(a0v_obsset)

    # if True:

    #     if (a0v_obsid is None) or (a0v_obsid == "1"):
    #         A0V_basename = extractor.basenames["a0v"]
    #     else:
    #         A0V_basename = "SDC%s_%s_%04d" % (band, utdate, int(a0v_obsid))
    #         print(A0V_basename)

    #     a0v = extractor.get_oned_spec_helper(A0V_basename,
    #                                          basename_postfix=basename_postfix)
    # config = obsset.rs.config

    tgt_spec_cor = get_tgt_spec_cor(obsset, tgt, a0v,
                                    threshold_a0v,
                                    multiply_model_a0v)

    ax2a = fig.add_subplot(211)
    ax2b = fig.add_subplot(212, sharex=ax2a)

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


def plot_spec(obsset, interactive=False,
              multiply_model_a0v=False):
    recipe = obsset.recipe_name
    target_type, nodding_type = recipe.split("_")

    if target_type in ["A0V"]:
        FIX_TELLURIC = False
    elif target_type in ["STELLAR", "EXTENDED"]:
        FIX_TELLURIC = True
    else:
        raise ValueError("Unknown recipe : %s" % recipe)

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

    if FIX_TELLURIC:
        fig1 = Figure(figsize=(12, 6))
        fig_list.append(fig1)

        _plot_div_a0v_spec(fig1, tgt, obsset,
                           multiply_model_a0v=multiply_model_a0v)

    if fig_list:
        for fig in fig_list:
            fig.tight_layout()

    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()


steps = [Step("Set basename_postfix", set_basename_postfix),
         Step("Plot spec", plot_spec,
              interactive=ArghFactoryWithShort(False),
              multiply_model_a0v=ArghFactoryWithShort(False)),
]
