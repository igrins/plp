import numpy as np
import scipy.ndimage as ni

from stsci_helper import stsci_median
import badpixel as bp
from destriper import destriper
from products import PipelineProducts

from igrins_detector import IGRINSDetector

class FlatOff(object):
    def __init__(self, offdata_list):
        self.data_list = offdata_list

    def make_flatoff_hotpixmap(self, sigma_clip1=100, sigma_clip2=10,
                               medfilter_size=None,
                               destripe=True):

        flat_off = stsci_median(self.data_list)

        if destripe:
            flat_offs = destriper.get_destriped(flat_off)

        hotpix_mask = bp.badpixel_mask(flat_offs,
                                       sigma_clip1=sigma_clip1,
                                       sigma_clip2=sigma_clip2,
                                       medfilter_size=medfilter_size)

        bg_std = flat_offs[~hotpix_mask].std()

        r = PipelineProducts("flat off products",
                             flat_off=flat_offs,
                             hotpix_mask=hotpix_mask,
                             bg_std=bg_std)
        return r

from trace_flat import get_flat_normalization, get_flat_mask

class FlatOn(object):
    def __init__(self, ondata_list):
        self.data_list = ondata_list

    def make_flaton_deadpixmap(self, flatoff_product=None,
                               deadpix_mask_old=None,
                               flat_mask_sigma=5.,
                               deadpix_thresh=0.6,
                               smooth_size=9):

        # load flat off data
        flat_off = flatoff_product["flat_off"]
        bg_std = flatoff_product["bg_std"]
        hotpix_mask = flatoff_product["hotpix_mask"]

        flat_on = stsci_median(self.data_list)
        flat_on_off = flat_on - flat_off

        # normalize it
        norm_factor = get_flat_normalization(flat_on_off,
                                             bg_std, hotpix_mask)

        flat_normed = flat_on_off / norm_factor
        bg_std_norm = bg_std/norm_factor

        # mask out bpix
        flat_bpixed = flat_normed.astype("d", copy=True)
        flat_bpixed[hotpix_mask] = np.nan

        flat_mask = get_flat_mask(flat_bpixed, bg_std_norm,
                                  sigma=flat_mask_sigma)


        # get dead pixel mask
        flat_smoothed = ni.median_filter(flat_normed,
                                         [smooth_size, smooth_size])
        #flat_smoothed[order_map==0] = np.nan
        flat_ratio = flat_normed/flat_smoothed

        refpixel_mask = np.ones(flat_mask.shape, bool)
        # mask out outer boundaries
        refpixel_mask[4:-4,4:-4] = False

        deadpix_mask = (flat_ratio<deadpix_thresh) & flat_mask & (~refpixel_mask)

        if deadpix_mask_old is not None:
            deadpix_mask = deadpix_mask | deadpix_mask_old

        flat_bpixed[deadpix_mask] = np.nan


        r = PipelineProducts("flat on products",
                             flat_normed=flat_normed,
                             flat_bpixed=flat_bpixed,
                             bg_std_normed=bg_std_norm,
                             flat_mask=flat_mask,
                             deadpix_mask=deadpix_mask)
        return r


from trace_flat import (get_y_derivativemap,
                        identify_horizontal_line,
                        trace_centroids_chevyshev)

def trace_orders(flaton_products):

    flat_normed=flaton_products["flat_normed"]
    flat_bpixed=flaton_products["flat_bpixed"]
    bg_std_normed=flaton_products["bg_std_normed"]
    flat_mask=flaton_products["flat_mask"]
    #deadpix_mask=deadpix_mask)

    flat_deriv_ = get_y_derivativemap(flat_normed, flat_bpixed,
                                      bg_std_normed,
                                      max_sep_order=150, pad=10,
                                      flat_mask=flat_mask)

    flat_deriv, flat_deriv_pos_msk, flat_deriv_neg_msk = \
                flat_deriv_["data"], flat_deriv_["pos_mask"], flat_deriv_["neg_mask"]


    ny, nx = flat_deriv.shape
    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=10,
                                                bg_std=bg_std_normed)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=10,
                                            bg_std=bg_std_normed)


    r = PipelineProducts("flat trace centroids",
                         flat_deriv=flat_deriv,
                         bottom_centroids=cent_bottom_list,
                         up_centroids=cent_up_list)

    return r


def check_trace_order(trace_products, fig, rect=111):
    from mpl_toolkits.axes_grid1 import ImageGrid
    d = trace_products["flat_deriv"]
    grid = ImageGrid(fig, rect, (1, 3), share_all=True)
    ax = grid[0]
    im = ax.imshow(d, origin="lower", interpolation="none",
                   cmap="RdBu")
    im.set_clim(-0.05, 0.05)
    ax = grid[1]
    for l in trace_products["bottom_centroids"]:
        ax.plot(l[0], l[1], "r-")
    for l in trace_products["up_centroids"]:
        ax.plot(l[0], l[1], "b-")

    ax = grid[2]
    im = ax.imshow(d, origin="lower", interpolation="none",
                   cmap="RdBu")
    im.set_clim(-0.05, 0.05)
    for l in trace_products["bottom_centroids"]:
        ax.plot(l[0], l[1], "r-")
    for l in trace_products["up_centroids"]:
        ax.plot(l[0], l[1], "b-")
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)


def trace_solutions(trace_products):

    bottom_centroids = trace_products["bottom_centroids"]
    up_centroids = trace_products["up_centroids"]

    nx = IGRINSDetector.nx

    _ = trace_centroids_chevyshev(bottom_centroids,
                                  up_centroids,
                                  domain=[0, nx],
                                  ref_x=nx/2)

    bottom_up_solutions, bottom_up_centroids = _

    from numpy.polynomial import Polynomial
    bottom_up_solutions_as_list = []
    for b, d in bottom_up_solutions:

        bb, dd = b.convert(kind=Polynomial), d.convert(kind=Polynomial)
        bb_ = ("poly", bb.coef)
        dd_ = ("poly", dd.coef)
        bottom_up_solutions_as_list.append((bb_, dd_))

    r = PipelineProducts("order trace solutions",
                         orders=[],
                         bottom_up_centroids=bottom_up_centroids,
                         bottom_up_solutions=bottom_up_solutions_as_list)


    return r



def make_order_flat(flaton_products, orders, order_map):


    flat_normed  = flaton_products["flat_normed"]
    flat_mask = flaton_products["flat_mask"]

    import scipy.ndimage as ni
    slices = ni.find_objects(order_map)

    mean_order_specs = []
    mask_list = []
    for o in orders:
        sl = (slices[o-1][0], slice(0, 2048))
        d_sl = flat_normed[sl].copy()
        d_sl[order_map[sl] != o] = np.nan

        f_sl = flat_mask[sl].copy()
        f_sl[order_map[sl] != o] = np.nan
        ff = np.nanmean(f_sl, axis=0)
        mask_list.append(ff)

        mmm = order_map[sl] == o
        ss = [np.nanmean(d_sl[2:-2][:,i][mmm[:,i][2:-2]]) \
              for i in range(2048)]
        mean_order_specs.append(ss)


    from trace_flat import (get_smoothed_order_spec,
                            get_order_boundary_indices,
                            get_order_flat1d)

    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]
    i1i2_list = [get_order_boundary_indices(s, s0) \
                 for s, s0 in zip(mean_order_specs, s_list)]
    p_list = [get_order_flat1d(s, i1, i2) for s, (i1, i2) \
              in zip(s_list, i1i2_list)]

    # make flat
    x = np.arange(len(s))
    flat_im = np.empty(flat_normed.shape, "d")
    flat_im.fill(np.nan)

    fitted_responses = []

    for o, p in zip(orders, p_list):
        sl = (slices[o-1][0], slice(0, 2048))
        d_sl = flat_normed[sl].copy()
        msk = (order_map[sl] == o)
        d_sl[~msk] = np.nan

        px = p(x)
        flat_im[sl][msk] = (d_sl / px)[msk]
        fitted_responses.append(px)

    r = PipelineProducts("order flat",
                         orders=orders,
                         order_flat=flat_im,
                         fitted_responses=fitted_responses,
                         i1i2_list=i1i2_list,
                         mean_order_specs=mean_order_specs)

    return r


def check_order_flat(order_flat_products):

    from trace_flat import (prepare_order_trace_plot,
                            check_order_trace1, check_order_trace2)

    mean_order_specs = order_flat_products["mean_order_specs"]

    from trace_flat import (get_smoothed_order_spec,
                            get_order_boundary_indices,
                            get_order_flat1d)

    # these are duplicated from make_order_flat
    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]
    i1i2_list = [get_order_boundary_indices(s, s0) \
                 for s, s0 in zip(mean_order_specs, s_list)]
    p_list = [get_order_flat1d(s, i1, i2) for s, (i1, i2) \
              in zip(s_list, i1i2_list)]

    fig_list, ax_list = prepare_order_trace_plot(s_list)
    x = np.arange(2048)
    for s, i1i2, ax in zip(mean_order_specs, i1i2_list, ax_list):
        check_order_trace1(ax, x, s, i1i2)

    for s, p, ax in zip(mean_order_specs, p_list, ax_list):
        check_order_trace2(ax, x, p)

    return fig_list


    # if 0:
    #     hdu_list = igrins_log.get_cal_hdus(band, "HD3417")
    #     hd_list = [destriper.get_destriped(hdu.data) for hdu in hdu_list]
    #     from stsci_helper import stsci_median
    #     hd_spec = stsci_median(hd_list)







if 0:
    if 1: # chevyshev
        _ = trace_centroids_chevyshev(cent_bottom_list,
                                      cent_up_list,
                                      domain=[0, 2048],
                                      ref_x=nx/2)

    if 0:
        order = 5
        func_fitter = get_line_fiiter(order)

        _ = trace_centroids(cent_bottom_list,
                            cent_up_list,
                            func_fitter=func_fitter,
                            ref_x=nx/2)

    bottom_up_solutions, bottom_up_centroids = _

    if 0:
        plot_sollutions(flat_normed,
                        bottom_up_centroids,
                        bottom_up_solutions)

    r = PipelineProducts(flat_deriv=flat_deriv,
                         bottom_up_centroids=bottom_up_centroids,
                         bottom_up_solutions=bottom_up_solutions)


    return_object["flat_deriv"] = flat_deriv
    #return_object["flat_deriv_pos_mask"] = flat_deriv_pos_msk
    #return_object["flat_deriv_neg_mask"] = flat_deriv_neg_msk


    return_object["cent_up_list"] = cent_up_list
    return_object["cent_bottom_list"] = cent_bottom_list
    return_object["bottomup_solutions"] = bottom_up_solutions
    return_object["bottomup_centroids"] = centroid_bottom_up_list





def process_flat(ondata_list, offdata_list):


    return_object = {}



    return_object["flat_on_off"] = flat_on_off
    return_object["flat_norm_factor"] = norm_factor
    return_object["flat_normed"] = flat_norm
    return_object["flat_bpix_mask"] = bpix_mask
    return_object["bg_std"] = bg_std
    return_object["bg_std_normed"] = bg_std_norm
    return_object["flat_mask"] = flat_mask


    flat_deriv_ = get_y_derivativemap(flat_norm, flat_bpix,
                                      bg_std_norm,
                                      max_sep_order=150, pad=50,
                                      flat_mask=flat_mask)

    flat_deriv, flat_deriv_pos_msk, flat_deriv_neg_msk = flat_deriv_["data"], flat_deriv_["pos_mask"], flat_deriv_["neg_mask"]


    return_object["flat_deriv"] = flat_deriv
    return_object["flat_deriv_pos_mask"] = flat_deriv_pos_msk
    return_object["flat_deriv_neg_mask"] = flat_deriv_neg_msk

    ny, nx = flat_deriv.shape
    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=50,
                                                bg_std=bg_std_norm)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=50,
                                            bg_std=bg_std_norm)


    if 1: # chevyshev
        _ = trace_centroids_chevyshev(cent_bottom_list,
                                      cent_up_list,
                                      domain=[0, 2048],
                                      ref_x=nx/2)

    if 0:
        order = 5
        func_fitter = get_line_fiiter(order)

        _ = trace_centroids(cent_bottom_list,
                            cent_up_list,
                            func_fitter=func_fitter,
                            ref_x=nx/2)

    bottom_up_solutions, centroid_bottom_up_list = _

    if 0:
        plot_solutions(flat_norm,
                       centroid_bottom_up_list,
                       bottom_up_solutions)

    return_object["cent_up_list"] = cent_up_list
    return_object["cent_bottom_list"] = cent_bottom_list
    return_object["bottomup_solutions"] = bottom_up_solutions
    return_object["bottomup_centroids"] = centroid_bottom_up_list

    return return_object
