import numpy as np
import itertools
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredText
from matplotlib.offsetbox import AnchoredText

from .. import DESCS


def _imshow_flat_deriv(ax, flat_deriv):
    im = ax.imshow(flat_deriv, origin="lower", interpolation="none",
                   cmap="RdBu")
    im.set_clim(-0.05, 0.05)

    return im


def _draw_traces(ax, trace_dict):
    bottom_lines, up_lines = [], []

    for l in trace_dict["bottom_centroids"]:
        _l, = ax.plot(l[0], l[1], "r-")
        bottom_lines.append(_l)
    for l in trace_dict["up_centroids"]:
        _l, = ax.plot(l[0], l[1], "b-")
        up_lines.append(_l)

    return bottom_lines, up_lines


def check_trace_order(flat_deriv, trace_dict, fig,
                      rowcol=(1, 3),
                      rect=111,
                      title_fontsize=None):
    from mpl_toolkits.axes_grid1 import ImageGrid

    #from axes_grid import ImageGrid
    #d = trace_products["flat_deriv"]

    # from storage_descriptions import (FLAT_DERIV_DESC,
    #                                   FLATCENTROIDS_JSON_DESC)

    # d = trace_products[FLAT_DERIV_DESC].data
    # trace_dict = trace_products[FLATCENTROIDS_JSON_DESC]

    grid = ImageGrid(fig, rect, rowcol, share_all=True)
    _imshow_flat_deriv(grid[0], flat_deriv)

    # ax = grid[0]

    # im = ax.imshow(flat_deriv, origin="lower", interpolation="none",
    #                cmap="RdBu")
    # im.set_clim(-0.05, 0.05)

    _draw_traces(grid[1], trace_dict)

    _imshow_flat_deriv(grid[2], flat_deriv)
    _draw_traces(grid[2], trace_dict)

    # ax = grid[1]
    # for l in trace_dict["bottom_centroids"]:
    #     ax.plot(l[0], l[1], "r-")
    # for l in trace_dict["up_centroids"]:
    #     ax.plot(l[0], l[1], "b-")

    # ax = grid[2]
    # im = ax.imshow(flat_deriv, origin="lower", interpolation="none",
    #                cmap="RdBu")
    # im.set_clim(-0.05, 0.05)
    # for l in trace_dict["bottom_centroids"]:
    #     ax.plot(l[0], l[1], "r-")
    # for l in trace_dict["up_centroids"]:
    #     ax.plot(l[0], l[1], "b-")

    ax = grid[0]
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)

    if title_fontsize is None:
        return

    for ax, title in zip(grid, 
                         ["Derivative Image", "Traced Boundary", "Together"]):

        at = AnchoredText(title,
                          prop=dict(size=title_fontsize), frameon=True,
                          loc=2,
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)


def _imshow_flat(ax, obsset_on):
    flat = obsset_on.load("flat_normed")[0].data

    vmin, vmax = np.percentile(flat, [5, 95])
    im = ax.imshow(flat, origin="lower", interpolation="none", cmap="gray")
    im.set_clim(vmin, vmax)
    # , cmap="gray_r")
    return im


def _draw_modeled(ax, obsset_on):

    flat = obsset_on.load("flat_normed")[0].data

    flatcentroid_sol_json = obsset_on.load("flatcentroid_sol_json")
    bottom_up_solutions_ = flatcentroid_sol_json["bottom_up_solutions"]
    from .polynomials import nested_convert_to_poly
    bottom_up_solutions = nested_convert_to_poly(bottom_up_solutions_)

    x_indices = np.arange(flat.shape[1])

    ll = []
    colors = itertools.cycle("rg")
    for bottom_sol, up_sol in bottom_up_solutions:
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        c = next(colors)
        l1, = ax.plot(x_indices, y_bottom, "-", color=c)
        l2, = ax.plot(x_indices, y_up, "-", color=c)

        ll.extend([l1, l2])

    return ll


def plot_solutions1(fig1, flat,
                    bottom_up_solutions):

    x_indices = np.arange(flat.shape[1])

    ax = fig1.add_subplot(111)
    ax.imshow(flat, origin="lower")  #, cmap="gray_r")
    # ax.set_autoscale_on(False)

    colors = itertools.cycle("rg")
    for bottom_sol, up_sol in bottom_up_solutions:
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        c = next(colors)
        ax.plot(x_indices, y_bottom, "-", color=c)
        ax.plot(x_indices, y_up, "-", color=c)

    return fig1


def plot_solutions2(fig2, cent_bottomup_list,
                    bottom_up_solutions):

    # x_indices = np.arange(flat.shape[1])

    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212)

    for bottom_sol, bottom_cent in \
        zip(bottom_up_solutions[0], cent_bottomup_list[0]):

        x = np.array(bottom_cent[0], dtype="f")
        y = np.array(bottom_cent[1], dtype="f")
        ax21.plot(x, y - bottom_sol(x))

    for up_sol, up_cent in \
        zip(bottom_up_solutions[1], cent_bottomup_list[1]):

        x = np.array(up_cent[0], dtype="f")
        y = np.array(up_cent[1], dtype="f")
        ax22.plot(x, y - up_sol(x))

    ax21.set_xlim(0, 2048)
    ax22.set_xlim(0, 2048)

    ax21.set_ylim(-3, 3)
    ax22.set_ylim(-3, 3)

    return fig2


def plot_trace_solutions(fig1, fig2, flat_normed,
                         flatcentroid_sol_json,
                         # trace_solution_products,
                         # trace_solution_products_plot
                         ):

    # from storage_descriptions import (FLAT_NORMED_DESC,
    #                                   FLATCENTROID_SOL_JSON_DESC)

    # flat_normed = flaton_products[FLAT_NORMED_DESC].data
    # _d = trace_solution_products[FLATCENTROID_SOL_JSON_DESC]

    bottom_up_solutions_ = flatcentroid_sol_json["bottom_up_solutions"]

    from .polynomials import nested_convert_to_poly
    bottom_up_solutions = nested_convert_to_poly(bottom_up_solutions_)

    # for b, d in bottom_up_solutions_:
    #     import numpy.polynomial as P
    #     assert b[0] == "poly"
    #     assert d[0] == "poly"
    #     bp = P.Polynomial(b[1])
    #     dp = P.Polynomial(d[1])
    #     bottom_up_solutions.append((bp, dp))

    # from .qa_flat import plot_solutions1, plot_solutions2
    fig1 = plot_solutions1(fig1, flat_normed,
                           bottom_up_solutions)

    # _d = trace_solution_products_plot[FLATCENTROID_SOL_JSON_DESC]

    bottom_up_solutions_qa_ = flatcentroid_sol_json["bottom_up_solutions_qa"]
    bottom_up_solutions_qa = nested_convert_to_poly(bottom_up_solutions_qa_)

    fig2 = plot_solutions2(fig2, flatcentroid_sol_json["bottom_up_centroids"],
                           bottom_up_solutions_qa)

    return fig1, fig2


def set_visible_all(l, b):
    for l1 in l:
        l1.set_visible(b)


def run_interactive(obsset, params):
    # , params, _process, exptime=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), num=1, clear=True)

    obsdate, band = obsset.get_resource_spec()
    obsid = obsset.master_obsid

    # status = dict(to_save=False)

    # def save(*kl, status=status):
    #     status["to_save"] = True
    #     plt.close(fig)

    obsset_on = obsset.get_subset("ON")
    flat_deriv = obsset_on.load("flat_deriv")[0].data
    trace_dict = obsset_on.load("flatcentroids_json")

    im_deriv = _imshow_flat_deriv(ax, flat_deriv)
    l1, l2 = _draw_traces(ax, trace_dict)
    line_detected = list(l1) + list(l2)

    im_flat = _imshow_flat(ax, obsset_on)
    line_modeled = _draw_modeled(ax, obsset_on)

    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)

    def set_visibility(w, kl, user_params):

        image_kind = user_params["image_kind"]
        line_kind = user_params["line_kind"]

        lines = dict(detected=line_detected, modeled=line_modeled)
        images = dict(deriv=im_deriv, flat=im_flat)

        set_visible_all(lines.pop(line_kind, []), True)
        for l in lines.values():
            set_visible_all(l, False)

        im = images.pop(image_kind, None)
        if im is not None:
            im.set_visible(True)

        for im in images.values():
            if im is not None:
                im.set_visible(False)

    from ..utils.gui_box_helper import setup_basic_gui
    # from ..utils.gui_box_helper import WidgetSet, Input, Radio, Check, Button
    from ..utils.gui_box_helper import Radio
    widgets = [
        Radio("image_kind",
              ["none", "flat", "deriv"], ["none", "flat", "deriv"],
              value=params["image_kind"],
              on_trigger=set_visibility),
        Radio("line_kind",
              ["none", "detected", "modeled"], ["none", "detected", "modeled"],
              value=params["line_kind"],
              on_trigger=set_visibility),
    ]

    box, ws = setup_basic_gui(ax, params, widgets)
    set_visibility(None, None, params)

    ax.set_title("{}-{:04d} [{}]".format(obsdate, obsid, band))

    plt.show()

    return ws.status


def main():
    from igrins import get_obsset
    band = "H"
    config_file = "recipe.config"
    obsset = get_obsset("20190318", band, "FLAT",
                        obsids=range(1011, 1031),
                        frametypes=["OFF"]*10 + ["ON"]*10,
                        config_file=config_file)

    # from igrins import DESCS

    params = dict(image_kind="flat", line_kind="modeled")

    interactive = True
    if interactive:

        params = run_interactive(obsset, params)
        # , params, _process_sky,
        #                          exptime=exptime)

        to_save = params.pop("to_save", False)
        if not to_save:
            print("canceled")
            return


if __name__ == '__main__':
    main()
