
def check_trace_order(flat_deriv, trace_dict, fig, rect=111):
    from mpl_toolkits.axes_grid1 import ImageGrid
    #from axes_grid import ImageGrid
    #d = trace_products["flat_deriv"]

    # from storage_descriptions import (FLAT_DERIV_DESC,
    #                                   FLATCENTROIDS_JSON_DESC)

    # d = trace_products[FLAT_DERIV_DESC].data
    # trace_dict = trace_products[FLATCENTROIDS_JSON_DESC]

    grid = ImageGrid(fig, rect, (1, 3), share_all=True)
    ax = grid[0]
    im = ax.imshow(flat_deriv, origin="lower", interpolation="none",
                   cmap="RdBu")
    im.set_clim(-0.05, 0.05)
    ax = grid[1]
    for l in trace_dict["bottom_centroids"]:
        ax.plot(l[0], l[1], "r-")
    for l in trace_dict["up_centroids"]:
        ax.plot(l[0], l[1], "b-")

    ax = grid[2]
    im = ax.imshow(flat_deriv, origin="lower", interpolation="none",
                   cmap="RdBu")
    im.set_clim(-0.05, 0.05)
    for l in trace_dict["bottom_centroids"]:
        ax.plot(l[0], l[1], "r-")
    for l in trace_dict["up_centroids"]:
        ax.plot(l[0], l[1], "b-")
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)


def plot_trace_solutions(flat_normed,
                         flatcentroid_sol_json,
                         # trace_solution_products,
                         # trace_solution_products_plot
                         ):

    # from storage_descriptions import (FLAT_NORMED_DESC,
    #                                   FLATCENTROID_SOL_JSON_DESC)

    # flat_normed = flaton_products[FLAT_NORMED_DESC].data
    # _d = trace_solution_products[FLATCENTROID_SOL_JSON_DESC]

    bottom_up_solutions_ = flatcentroid_sol_json["bottom_up_solutions"]

    from polynomials import nested_convert_to_poly
    bottom_up_solutions = nested_convert_to_poly(bottom_up_solutions_)

    # for b, d in bottom_up_solutions_:
    #     import numpy.polynomial as P
    #     assert b[0] == "poly"
    #     assert d[0] == "poly"
    #     bp = P.Polynomial(b[1])
    #     dp = P.Polynomial(d[1])
    #     bottom_up_solutions.append((bp, dp))

    from trace_flat import plot_solutions1, plot_solutions2
    fig2 = plot_solutions1(flat_normed,
                           bottom_up_solutions)

    # _d = trace_solution_products_plot[FLATCENTROID_SOL_JSON_DESC]

    bottom_up_solutions_qa_ = flatcentroid_sol_json["bottom_up_solutions_qa"]
    bottom_up_solutions_qa = nested_convert_to_poly(bottom_up_solutions_qa_)

    fig3 = plot_solutions2(flatcentroid_sol_json["bottom_up_centroids"],
                           bottom_up_solutions_qa)

    return fig2, fig3
