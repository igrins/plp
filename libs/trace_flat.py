# import Libs.manual as m
# reload(m)
# import Libs.ap_tracing as m2
# reload(m2)
import os
import numpy as np

import libs.fits as pyfits
import scipy.ndimage as ni

import badpixel as bp

import itertools

def get_flat_normalization(flat_on_off, bg_std, bpix_mask):

    lower_limit = bg_std*10

    flat_norm = bp.estimate_normalization_percentile(flat_on_off,
                                                     lower_limit, bpix_mask,
                                                     percentile=99.)
    # alternative normalization value
    # norm1 = bp.estimate_normalization(d, lower_limit, bpix_mask)

    return flat_norm

def get_flat_mask(flat_bpix, bg_std_norm, sigma=3):
    """
    flat_bpix : bpix-masked normarlized flat
    bg_std_norm : bg stddev of flat_bpix
    """
    # now we try to build a reasonable mask
    # start with a simple thresholded mask

    flat_mask = (flat_bpix > sigma * bg_std_norm)

    # remove isolated dots by doing binary erosion
    m_opening = ni.binary_opening(flat_mask, iterations=2)
    # try to extend the mask with dilation
    m_dilation = ni.binary_dilation(m_opening, iterations=5)

    return m_dilation

def mask_median_clip(y_ma, median_size=5, clip=1):
    """
    Subtract a median-ed singal from the original.
    Then, return a mask with out-of-sigma values clipped.
    """
    from scipy.stats.mstats import trima
    from scipy.signal import medfilt
    y_filtered = y_ma - medfilt(y_ma, median_size)
    y_trimmed = trima(y_filtered, (-clip, clip))
    return y_trimmed.mask


# standard deviation filter
# from stddev_filter import window_stdev

# def mask_median_sima(y_ma, median_size=5, sima=1):
#     """
#     Subtract a median-ed singal from the original.
#     Then, return a mask with out-of-sigma values clipped.
#     """
#     from scipy.stats.mstats import trimmed_stde, trima
#     from scipy.signal import medfilt
#     y_filtered = y_ma - medfilt(y_ma, median_size)
#     y_std = ni.uniform_filter(np.abs(y_filtered.filled(0.)),
#                               median_size*2,
#                               mode='constant')**.5

#     window_stdev
#     y_trimed = trima(y_filtered, (-clip, clip))
#     pass


def find_nearest_object(mmp, im_labeled, slice_map, i, labels_center_column):
    """
    mmp : mask
    im_labeled : label
    i : object to be connected
    labels_center_column : known objects
    """
    thre = 40 # threshold # of pixels (in y-direction) to detect adjacent object
    steps = [5, 10, 20, 40, 80]

    sl_y, sl_x = slice_map[i]

    # right side
    ss = im_labeled[:,sl_x.stop-3:sl_x.stop].max(axis=1)
    ss_msk = ni.maximum_filter1d(ss == i, thre)

    if sl_x.stop < 2048/2.:
        sl_x0 = sl_x.stop
        sl_x_pos = [sl_x.stop + s for s in steps]
    else:
        sl_x0 = sl_x.start
        sl_x_pos = [sl_x.start - s for s in steps]

    for pos in sl_x_pos:
        ss1 = im_labeled[:,pos]
        detected_ob = set(np.unique(ss1[ss_msk])) - set([0])
        for ob_id in detected_ob:
            if ob_id in labels_center_column:
                sl = slice_map[ob_id][1]
                if sl.start < sl_x0 < sl.stop:
                    continue
                else:
                    return ob_id

def identify_horizontal_line(d_deriv, mmp, pad=20, bg_std=None):
    """
    d_deriv : derivative (along-y) image
    mmp : mask
    order : polyfit order
    pad : padding around the boundary will be ignored
    bg_std : if given, derivative smaller than bg_std will be suppressed.
             This will affect faint signal near the chip boundary

    Masks will be derived from mmp, and peak values of d_deriv will be
    fitted with polynomical of given order.
    """
    if 0:

        pad=50,
        bg_std=bg_std_norm
        d_deriv=-flat_deriv
        mmp=flat_deriv_neg_msk
        d_deriv=flat_deriv
        mmp=flat_deriv_pos_msk

    ny, nx = d_deriv.shape


    # We first identify objects
    im_labeled, label_max = ni.label(mmp)
    label_indx = np.arange(1, label_max+1, dtype="i")
    objects_found = ni.find_objects(im_labeled)

    from itertools import izip, compress

    slice_map = dict(izip(label_indx, objects_found))

    if 0:
        # make labeld image with small objects delted.
        s = ni.measurements.sum(mmp, labels=im_labeled,
                                index=label_indx)
        tiny_traces = s < 10. # 0.1 * s.max()
        mmp2 = im_labeled.copy()
        for label_num in compress(label_indx, tiny_traces):
            sl = slice_map[label_num]
            mmp_sl = mmp2[sl]
            mmp_sl[mmp_sl == label_num] = 0


    # We only traces solutions that are detected in the centeral colmn

    # label numbers along the central column

    # from itertools import groupby
    # labels_center_column = [i for i, _ in groupby(im_labeled[:,nx/2]) if i>0]

    thre_dx = 30
    labels_ = set(np.unique(im_labeled[:,nx/2-thre_dx:nx/2+thre_dx])) - set([0])
    labels_center_column = sorted(list(labels_))

    # remove objects with small area
    s = ni.measurements.sum(mmp, labels=im_labeled,
                            index=labels_center_column)

    labels_center_column = np.array(labels_center_column)[s > 0.1 * s.max()]

    # try to stitch undetected object to center ones.
    undetected_labels_ = [i for i in range(1, label_max+1) \
                         if i not in labels_center_column]
    s2 = ni.measurements.sum(mmp, labels=im_labeled,
                             index=undetected_labels_)

    undetected_labels = np.array(undetected_labels_)[s2 > 0.1 * s.max()]


    slice_map_update_required = False

    for i in undetected_labels:
        ob_id = find_nearest_object(mmp, im_labeled,
                                    slice_map, i, labels_center_column)
        if ob_id:
            print i, ob_id
            im_labeled[im_labeled == i] = ob_id
            slice_map_update_required = True

    if slice_map_update_required:
        objects_found = ni.find_objects(im_labeled)
        slice_map = dict(izip(label_indx, objects_found))

    # im_labeled is now updated

    y_indices = np.arange(ny)
    x_indices = np.arange(nx)

    centroid_list = []
    for indx in labels_center_column:

        sl = slice_map[indx]

        y_indices1 = y_indices[sl[0]]
        x_indices1 = x_indices[sl[1]]

        # mask for line to trace
        feature_msk = im_labeled[sl]==indx

        # nan for outer region.
        feature = d_deriv[sl].copy()
        feature[~feature_msk] = np.nan

        # measure centroid
        yc = np.nansum(y_indices1[:, np.newaxis] * feature, axis=0)
        ys = np.nansum(feature, axis=0)
        yn = np.sum(np.isfinite(feature), axis=0)

        yy = yc/ys

        msk = mask_median_clip(yy) | ~np.isfinite(yy)

        # we also clip wose derivative is smaller than bg_std
        # This suprress the lowest order of K band
        if bg_std is not None:
            msk = msk | (ys/yn < bg_std)

        # mask out columns with # of valid pixel is too many
        msk = msk | (yn > 4)


        centroid_list.append((x_indices1,
                              np.ma.array(yy, mask=msk)))

    return centroid_list

def get_line_fiiter(order):

    def _f(x, y, order=order):
        if not hasattr(y, "mask"):
            msk = np.isfinite(y)
            y = np.ma.array(y, mask=~msk)

        x1 = np.ma.array(x, mask=y.mask).compressed()
        y1 = y.compressed()

        p = np.polyfit(x1, y1, order)

        return np.poly1d(p)

    return _f

def trace_centroids_chevyshev(centroid_bottom_list,
                              centroid_up_list,
                              domain, ref_x=None):
    from trace_aperture import trace_aperture_chebyshev

    if ref_x is None:
        ref_x = 0.5 * (domain[0] + domain[-1])

    _ = trace_aperture_chebyshev(centroid_bottom_list,
                                 domain=domain)
    sol_bottom_list, sol_bottom_list_full = _

    _ = trace_aperture_chebyshev(centroid_up_list,
                                 domain=domain)
    sol_up_list, sol_up_list_full = _

    yc_down_list = [s(ref_x) for s in sol_bottom_list_full] # lower-boundary list
    #y0 = sol_bottom_list[1](ref_x) # lower-boundary of the bottom solution
    #y0 = yc_down_list[1]
    yc_up_list = [s(ref_x) for s in sol_up_list_full] # upper-boundary list

    # yc_down_list[1] should be the 1st down-boundary that is not
    # outside the detector

    indx_down_bottom = np.searchsorted(yc_down_list, yc_up_list[1])
    indx_up_top = np.searchsorted(yc_up_list, yc_down_list[-2],
                                  side="right")


    # indx_up_bottom = np.searchsorted(yc_up_list, yc_down_list[1])
    # indx_down_top = np.searchsorted(yc_down_list, yc_up_list[-2],
    #                                 side="right")

    #print zip(yc_down_list[1:-1], yc_up_list[indx:])
    print indx_down_bottom, indx_up_top
    print yc_down_list
    print yc_up_list
    print zip(yc_down_list[indx_down_bottom-1:-1],
              yc_up_list[1:indx_up_top+1])

    sol_bottom_up_list_full = zip(sol_bottom_list_full[indx_down_bottom-1:-1],
                                  sol_up_list_full[1:indx_up_top+1])

    sol_bottom_up_list = sol_bottom_list, sol_up_list
    centroid_bottom_up_list = centroid_bottom_list, centroid_up_list
    #centroid_bottom_up_list = []

    return sol_bottom_up_list_full, sol_bottom_up_list, centroid_bottom_up_list


def trace_centroids(centroid_bottom_list,
                    centroid_up_list,
                    func_fitter,
                    ref_x):
    if 0:
        centroid_bottom_list = cent_bottom_list
        centroid_up_list = cent_up_list
        func_fitter=func_fitter
        ref_x=nx/2

    sol_bottom_list = [func_fitter(cenx, ceny) \
                       for cenx, ceny in centroid_bottom_list]

    sol_up_list = [func_fitter(cenx, ceny) \
                   for cenx, ceny in centroid_up_list]

    y0 = sol_bottom_list[0](ref_x) # lower-boundary of the bottom solution
    yc_list = [s(ref_x) for s in sol_up_list] # upper-boundary list

    indx = np.searchsorted(yc_list, y0)

    sol_bottom_up_list = zip(sol_bottom_list, sol_up_list[indx:])

    centroid_bottom_up_list = zip(centroid_bottom_list,
                                  centroid_up_list[indx:])

    return sol_bottom_up_list, centroid_bottom_up_list


def get_y_derivativemap(flat, flat_bpix, bg_std_norm,
                        max_sep_order=150, pad=50,
                        med_filter_size=(7, 7),
                        flat_mask=None):

    """
    flat
    flat_bpix : bpix'ed flat
    """

    # 1d-derivatives along y-axis : 1st attempt
    # im_deriv = ni.gaussian_filter1d(flat, 1, order=1, axis=0)

    # 1d-derivatives along y-axis : 2nd attempt. Median filter first.

    flat_deriv_bpix = ni.gaussian_filter1d(flat_bpix, 1,
                                           order=1, axis=0)

    # We also make a median-filtered one. This one will be used to make masks.
    flat_medianed = ni.median_filter(flat,
                                     size=med_filter_size)

    flat_deriv = ni.gaussian_filter1d(flat_medianed, 1,
                                      order=1, axis=0)

    # min/max filter

    flat_max = ni.maximum_filter1d(flat_deriv, size=max_sep_order, axis=0)
    flat_min = ni.minimum_filter1d(flat_deriv, size=max_sep_order, axis=0)

    # mask for aperture boundray
    if pad is None:
        sl=slice()
    else:
        sl=slice(pad, -pad)

    flat_deriv_masked = np.zeros_like(flat_deriv)
    flat_deriv_masked[sl,sl] = flat_deriv[sl, sl]

    if flat_mask is not None:
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5) & flat_mask
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5) & flat_mask
    else:
        flat_deriv_pos_msk = (flat_deriv_masked > flat_max * 0.5)
        flat_deriv_neg_msk = (flat_deriv_masked < flat_min * 0.5)

    return dict(data=flat_deriv, #_bpix,
                pos_mask=flat_deriv_pos_msk,
                neg_mask=flat_deriv_neg_msk,
                )

def get_aperture_solutions(flat_deriv,
                           flat_deriv_pos_msk, flat_deriv_neg_msk,
                           func_fitter,
                           bg_std=None):

    ny, nx = flat_deriv.shape
    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=50,
                                                bg_std=bg_std)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=50,
                                            bg_std=bg_std)

    sol_bottom_up_list = trace_centroids(cent_bottom_list,
                                         cent_up_list,
                                         func_fitter=func_fitter,
                                         ref_x=nx/2)

    return sol_bottom_up_list


def plot_solutions(flat,
                   cent_bottomup_list,
                   bottom_up_solutions):


    x_indices = np.arange(flat.shape[1])

    from matplotlib.figure import Figure
    fig1 = Figure(figsize=(7,7))

    ax = fig1.add_subplot(111)
    ax.imshow(flat, origin="lower") #, cmap="gray_r")
    #ax.set_autoscale_on(False)

    fig2 = Figure()
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212)

    next_color = itertools.cycle("rg").next
    for bottom_sol, up_sol in bottom_up_solutions:
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        c = next_color()
        ax.plot(x_indices, y_bottom, "-", color=c)
        ax.plot(x_indices, y_up, "-", color=c)


    next_color = itertools.cycle("rg").next
    for (bottom_sol, up_sol), (bottom_cent, up_cent) in \
        zip(bottom_up_solutions, cent_bottomup_list):
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        c = next_color()
        ax.plot(x_indices, y_bottom, "-", color=c)
        ax.plot(x_indices, y_up, "-", color=c)

        ax21.plot(bottom_cent[0],
                  bottom_cent[1] - bottom_sol(bottom_cent[0]))
        ax22.plot(up_cent[0],
                  up_cent[1] - up_sol(up_cent[0]))

    ax21.set_xlim(0, 2048)
    ax22.set_xlim(0, 2048)

    ax21.set_ylim(-3, 3)
    ax22.set_ylim(-3, 3)

    return fig1, fig2



def plot_solutions1(flat,
                    bottom_up_solutions):


    x_indices = np.arange(flat.shape[1])

    from matplotlib.figure import Figure
    fig1 = Figure(figsize=(7,7))

    ax = fig1.add_subplot(111)
    ax.imshow(flat, origin="lower") #, cmap="gray_r")
    #ax.set_autoscale_on(False)

    next_color = itertools.cycle("rg").next
    for bottom_sol, up_sol in bottom_up_solutions:
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        c = next_color()
        ax.plot(x_indices, y_bottom, "-", color=c)
        ax.plot(x_indices, y_up, "-", color=c)


    return fig1


def plot_solutions2(cent_bottomup_list,
                    bottom_up_solutions):


    #x_indices = np.arange(flat.shape[1])

    from matplotlib.figure import Figure
    fig2 = Figure()
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212)

    for bottom_sol, bottom_cent in \
        zip(bottom_up_solutions[0], cent_bottomup_list[0]):

        ax21.plot(bottom_cent[0],
                  bottom_cent[1] - bottom_sol(bottom_cent[0]))

    for up_sol, up_cent in \
        zip(bottom_up_solutions[1], cent_bottomup_list[1]):

        ax22.plot(up_cent[0],
                  up_cent[1] - up_sol(up_cent[0]))

    ax21.set_xlim(0, 2048)
    ax22.set_xlim(0, 2048)

    ax21.set_ylim(-3, 3)
    ax22.set_ylim(-3, 3)

    return fig2



def process_flat(ondata_list, offdata_list):

    from stsci_helper import stsci_median

    return_object = {}

    flat_on = stsci_median(ondata_list)
    flat_off = stsci_median(offdata_list)

    bpix_mask = bp.badpixel_mask(flat_off, sigma_clip1=100)

    bg_std = flat_off[~bpix_mask].std()

    flat_on_off = flat_on - flat_off

    norm_factor = get_flat_normalization(flat_on_off,
                                         bg_std, bpix_mask)

    flat_norm = flat_on_off / norm_factor
    bg_std_norm = bg_std/norm_factor


    flat_bpix = flat_norm.astype("d", copy=True)
    flat_bpix[bpix_mask] = np.nan

    flat_mask = get_flat_mask(flat_bpix, bg_std_norm, sigma=5)


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


def make_order_map(im_shape, bottom_up_solutions, orders=None,
                   mask_top_bottom=True):

    #next_color = itertools.cycle("rg").next
    ny, nx = im_shape
    x_indices = np.arange(nx)

    YY, XX = np.indices(im_shape)
    m = np.zeros(im_shape, dtype="i")

    if orders is None:
        orders = [1] * len(bottom_up_solutions)

    for (bottom_sol, up_sol), o in zip(bottom_up_solutions, orders):
        y_bottom = bottom_sol(x_indices)
        y_up = up_sol(x_indices)

        m += o * ((y_bottom-1 < YY) & (YY < y_up+1))
        #m += 1 * (y_bottom-1 < YY) - 0.5 * (YY < y_up+1)

    if mask_top_bottom:
        m += (YY < bottom_up_solutions[0][0](x_indices)) # bottom of bottom
        m += (YY > bottom_up_solutions[-1][-1](x_indices)) # bottom of bottom

    return m

def get_mask_bg_pattern(flat_mask, bottom_up_solutions):
    im_shape = flat_mask.shape
    order_msk = make_order_map(im_shape, bottom_up_solutions)
    mask_to_estimate_bg_pattern = (flat_mask & order_msk)

    return mask_to_estimate_bg_pattern


def subtract_bg_pattern(d, bottomup_solutions, flat_mask, bpix_mask):
    msk = get_mask_bg_pattern(flat_mask, bottom_up_solutions)
    #d = d.copy()
    d_ma = np.ma.array(d, mask=msk|bpix_mask)
    s_vert = np.ma.median(d_ma, axis=0)
    d_ma_sv = d_ma - s_vert
    s_horiz = np.ma.median(d_ma_sv, axis=1)
    return d - s_vert - s_horiz[:,np.newaxis]






# if 0:
#     x = np.arange(len(s))
#     for s, ax in zip(mean_order_spec[:12], grid):
#         pass
#         #p = fit_p(p_init, x[300:-300], s[300:-300])
#         #p = fit_p(p_init, x[210:1890], s[210:1890])

def get_finite_boundary_indices(s1):
    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    #k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]

    #k1, k2 = np.nonzero(s1>0.)[0][[0, -1]]
    nonzero_indices = np.nonzero(s1>0.)[0] #[[0, -1]]

   # # return meaningless indices if non-zero spectra is too short
   #  if len(nonzero_indices) < 5:
   #      return 4, 4

    k1, k2 = nonzero_indices[[0, -1]]
    k1 = max(k1, 4)
    k2 = min(k2, 2047-4)
    return k1, k2
#s = s1[k1:k2+1]

def get_order_boundary_indices(s1, s0=None):
    #x = np.arange(len(s))

    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    #k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]
    nonzero_indices = np.nonzero(s1>0.05)[0] #[[0, -1]]

   # return meaningless indices if non-zero spectra is too short
    if len(nonzero_indices) < 5:
        return 4, 4

    k1, k2 = nonzero_indices[[0, -1]]
    k1 = max(k1, 4)
    k2 = min(k2, 2047-4)
    s = s1[k1:k2+1]

    if s0 is None:
        s0 = get_smoothed_order_spec(s)
    else:
        s0 = s0[k1:k2+1]


    mm = s > max(s) * 0.05
    dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

    # mask out absorption feature
    smooth_size=20
    #s_s0 = s-s0
    #s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

    #mmm = s_s0 > -3.*s_s0_std


    s1 = ni.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
    #x1 = x[dd1:dd2]

    #s1r = s1 # ni.median_filter(s1, 100)

    s1_std = s1.std()
    s1_std = s1[np.abs(s1)<2.*s1_std].std()

    s1[np.abs(s1) < 2.*s1_std] = np.nan

    indx_center = int(len(s1)*.5)

    left_half = s1[:indx_center]
    if np.any(np.isfinite(left_half)):
        i1 = np.nanargmax(left_half)
        a_ = np.where(~np.isfinite(left_half[i1:]))[0]
        if len(a_):
            i1r = a_[0]
        else:
            i1r = 0
        i1 = dd1+i1+i1r #+smooth_size
    else:
        i1 = dd1

    right_half = s1[indx_center:]
    if np.any(np.isfinite(right_half)):
        i2 = np.nanargmin(right_half)
        a_ = np.where(~np.isfinite(right_half[:i2]))[0]

        if len(a_):
            i2r = a_[-1]
        else:
            i2r = i2
        i2 = dd1+indx_center+i2r
    else:
        i2 = dd2

    return k1+i1, k1+i2


def get_order_flat1d(s, i1=None, i2=None):

    s = np.array(s)
    k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
    s1 = s[k1:k2+1]


    if i1 is None:
        i1 = 0
    else:
        i1 -= k1

    if i2 is None:
        i2 = len(s1)
    else:
        i2 -= k1

    x = np.arange(len(s1))

    if 0:

        from astropy.modeling import models, fitting
        p_init = models.Chebyshev1D(degree=6, window=[0, 2047])
        fit_p = fitting.LinearLSQFitter()
        p = fit_p(p_init, x[i1:i2][mmm[i1:i2]], s[i1:i2][mmm[i1:i2]])

    if 1:
        # t= np.linspace(x[i1]+10, x[i2-1]-10, 10)
        # p = LSQUnivariateSpline(x[i1:i2],
        #                         s[i1:i2],
        #                         t, bbox=[0, 2047])

        # t= np.concatenate([[x[1],x[i1-5],x[i1],x[i1+5]],
        #                    np.linspace(x[i1]+10, x[i2-1]-10, 10),
        #                    [x[i2-5], x[i2], x[i2+5],x[-2]]])

        t_list = []
        if i1 > 10:
            t_list.append([x[1],x[i1]])
        else:
            t_list.append([x[1]])

        t_list.append(np.linspace(x[i1]+10, x[i2-1]-10, 10))
        if i2 < len(s) - 10:
            t_list.append([x[i2], x[-2]])
        else:
            t_list.append([x[-2]])

        t= np.concatenate(t_list)

        # s0 = ni.median_filter(s, 40)
        from scipy.interpolate import LSQUnivariateSpline
        p = LSQUnivariateSpline(x,
                                s1,
                                t, bbox=[0, len(s1)-1])

        def p0(x, k1=k1, k2=k2, p=p):
            msk = (k1 <= x) & (x <= k2)
            r = np.empty(len(x), dtype="d")
            r.fill(np.nan)
            r[msk] = p(x[msk])
            return r

    return p0


def get_smoothed_order_spec(s):
    s = np.array(s)
    k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
    s1 = s[k1:k2+1]

    s0 = np.empty_like(s)
    s0.fill(np.nan)
    s0[k1:k2+1] = ni.median_filter(s1, 40)
    return s0

def check_order_trace1(ax, x, s, i1i2):
    x = np.arange(len(s))
    ax.plot(x, s)
    i1, i2 = i1i2
    ax.plot(np.array(x)[[i1, i2]], np.array(s)[[i1,i2]], "o")

def check_order_trace2(ax, x, p):
    ax.plot(x, p(x))

def prepare_order_trace_plot(s_list, row_col=(3, 2)):

    from matplotlib.figure import Figure
    #from mpl_toolkits.axes_grid1 import Grid
    from axes_grid_patched import Grid

    row, col = row_col

    n_ax = len(s_list)
    n_f, n_remain = divmod(n_ax, row*col)
    if n_remain:
        n_ax_list = [row*col]*n_f + [n_remain]
    else:
        n_ax_list = [row*col]*n_f


    i_ax = 0

    fig_list = []
    ax_list = []
    for n_ax in n_ax_list:
        fig = Figure()
        fig_list.append(fig)

        grid = Grid(fig, 111, (row, col), ngrids=n_ax,
                    share_x=True)

        sl = slice(i_ax, i_ax+n_ax)
        for s, ax in zip(s_list[sl], grid):
            ax_list.append(ax)

        i_ax += n_ax

    return fig_list, ax_list



def get_order_trace_old(s):
    x = np.arange(len(s))

    s = np.array(s)
    mm = s > max(s) * 0.05
    dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

    # mask out absorption feature
    smooth_size=20
    s0 = ni.median_filter(s, 40)
    s_s0 = s-s0
    s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

    mmm = s_s0 > -3.*s_s0_std


    s1 = ni.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
    #x1 = x[dd1:dd2]

    s1r = s1

    s1_std = s1r.std()
    s1_std = s1r[np.abs(s1r)<2.*s1_std].std()

    s1r[np.abs(s1r) < 2.*s1_std] = np.nan

    if np.any(np.isfinite(s1r[:1024])):
        i1 = np.nanargmax(s1r[:1024])
        i1r = np.where(~np.isfinite(s1r[:1024][i1:]))[0][0]
        i1 = dd1+i1+i1r #+smooth_size
    else:
        i1 = dd1
    if np.any(np.isfinite(s1r[1024:])):
        i2 = np.nanargmin(s1r[1024:])
        i2r = np.where(~np.isfinite(s1r[1024:][:i2]))[0][-1]
        i2 = dd1+1024+i2r
    else:
        i2 = dd2

    if 0:
        p_init = models.Chebyshev1D(degree=6, window=[0, 2047])
        fit_p = fitting.LinearLSQFitter()
        p = fit_p(p_init, x[i1:i2][mmm[i1:i2]], s[i1:i2][mmm[i1:i2]])

    if 1:
        # t= np.linspace(x[i1]+10, x[i2-1]-10, 10)
        # p = LSQUnivariateSpline(x[i1:i2],
        #                         s[i1:i2],
        #                         t, bbox=[0, 2047])

        # t= np.concatenate([[x[1],x[i1-5],x[i1],x[i1+5]],
        #                    np.linspace(x[i1]+10, x[i2-1]-10, 10),
        #                    [x[i2-5], x[i2], x[i2+5],x[-2]]])

        t= np.concatenate([[x[1],x[i1]],
                           np.linspace(x[i1]+10, x[i2-1]-10, 10),
                           [x[i2], x[-2]]])

        p = LSQUnivariateSpline(x,
                                s0,
                                t, bbox=[0, 2047])

    return p


if 0:
    #p = chebfit(x[300:-300], s[300:-300],

    #plot(s1)

    print i1, i2
    ax.plot(x, s)
    ax.plot(x, p(x))
    ax.plot(np.array(x)[[i1, i2]], np.array(s)[[i1,i2]], "o")





if 0:
    # filename="test_references/20140316_H_data/SDCH_20140316-1_FLAT_G1.fits"
    # f_on="test_references/20140316_H_data/SDCH_20140316-1_FLAT_G1_ON.fits"
    # f_off="test_references/20140316_H_data/SDCH_20140316-1_FLAT_G1_OFF.fits"

    from test_execute import CDisplayJJ

    import Tkinter
    root = Tkinter.Tk()
    cdisplay = CDisplayJJ(root, path="")

    workdir = "20140316_H_data"
    logname="IGRINS_DT_Log_20140316-1_H.txt"

    # workdir = "20140316_K_data"
    # logname="IGRINS_DT_Log_20140316-1_K.txt"

    cdisplay.init_test(workdir, logname)

    off_list = [s.split()[0] for s in cdisplay.item_list["FLAT"] if "OFF" in s]
    on_list = [s.split()[0] for s in cdisplay.item_list["FLAT"] if "ON" in s]


    r = test_badpixel(on_list, off_list)

    fn = cdisplay.item_list["ARC"][0].split()[0]
    f = pyfits.open(os.path.join(workdir, fn+".fits"))

    d_pattern_sub = subtract_bg_pattern(f[0].data,
                                        r["bottomup_solutions"],
                                        r["flat_mask"],
                                        r["flat_bpix_mask"])


if 0:



    for sol in [sol1]:
        for indx, (_, _, p) in zip(labels, sol): # if 1
            ym = np.polyval(p, x_indices)
            ax.plot(x_indices, ym, "k-")

    ax = subplot(212)
    for indx, (x_indices1, yy, p) in zip(labels, polyfit_solutions): # if 1
        ym = np.polyval(p, x_indices)
        x_indices1 = x_indices[slice_map[indx][1]]
        ax.plot(x_indices1, yy - ym[slice_map[indx][1]])





    if False:
        mmp_center = ni.center_of_mass(mmp, labels=im_labeled,
                                       index=labels)

        fmt = "text(%5.1f, %5.1f) # text={%d}"
        reg = "\n".join(fmt % (c[1], c[0], i) for i, c in enumerate(mmp_center))
        ds9.set("regions", reg)



def mymain():
    from pipeline_jjlee import IGRINSLog, Destriper

    if 1:
        log_20140316 = dict(flat_off=range(2, 4),
                            flat_on=range(4, 7),
                            thar=range(1, 2),
                            HD3417=[15, 16])


        igrins_log = IGRINSLog("20140316", log_20140316)
    else:
        log_20140525 = dict(flat_off=range(64, 74),
                            flat_on=range(74, 84),
                            thar=range(3, 8))

        igrins_log = IGRINSLog("20140525", log_20140525)

    band = "H"

    import numpy as np
    destriper = Destriper()

    hdu_list = igrins_log.get_cal_hdus(band, "flat_off")
    flat_offs = [destriper.get_destriped(hdu.data) for hdu in hdu_list]

    hdu_list = igrins_log.get_cal_hdus(band, "flat_on")
    flat_ons = [hdu.data for hdu in hdu_list]

    #flat_off = np.median(flat_offs, axis=0)

    #import trace_flat
    #reload(trace_flat)
    #process_flat =  trace_flat.process_flat

    r = process_flat(ondata_list=flat_ons, offdata_list=flat_offs)


    flat_normed = r["flat_normed"].copy()
    flat_normed[r["flat_bpix_mask"]] = np.nan

    starting_order = 5
    orders = range(starting_order,
                   len(r["bottomup_solutions"]) + starting_order)
    order_map = make_order_map(flat_normed.shape,
                               r["bottomup_solutions"],
                               orders=orders,
                               mask_top_bottom=False)

    # get dead pixel mask
    flat_smoothed = ni.median_filter(flat_normed, [1, 9])
    flat_smoothed[order_map==0] = np.nan
    flat_ratio = flat_normed/flat_smoothed
    flat_mask = r["flat_mask"]

    refpixel_mask = np.ones(flat_mask.shape, bool)
    refpixel_mask[4:-4,4:-4] = False

    dead_pixel_mask = (flat_ratio<0.3) & flat_mask & (~refpixel_mask)

    flat_normed[dead_pixel_mask] = np.nan

    import scipy.ndimage as ni
    slices = ni.find_objects(order_map)

    mean_order_spec = []
    mask_list = []
    for o in orders:
        sl = slices[o-1]
        d_sl = flat_normed[sl].copy()
        d_sl[order_map[sl] != o] = np.nan

        f_sl = flat_mask[sl].copy()
        f_sl[order_map[sl] != o] = np.nan
        ff = np.nanmean(f_sl, axis=0)
        mask_list.append(ff)

        mmm = order_map[sl] == o
        ss = [np.nanmean(d_sl[3:-3][:,i][mmm[:,i][3:-3]]) for i in range(2048)]
        mean_order_spec.append(ss)


    s_list = [get_smoothed_order_spec(s) for s in mean_order_spec]
    i1i2_list = [get_order_boundary_indices(s, s0) \
                 for s, s0 in zip(mean_order_spec, s_list)]
    p_list = [get_order_flat1d(s, i1, i2) for s, (i1, i2) \
              in zip(s_list, i1i2_list)]

    fig_list, ax_list = prepare_order_trace_plot(s_list)
    x = np.arange(2048)
    for s, i1i2, ax in zip(mean_order_spec, i1i2_list, ax_list):
        check_order_trace1(ax, x, s, i1i2)

    for s, p, ax in zip(mean_order_spec, p_list, ax_list):
        check_order_trace2(ax, x, p)


    # make flat
    x = np.arange(len(s))
    flat_im = np.empty(flat_normed.shape, "d")
    flat_im.fill(np.nan)

    for o, p in zip(orders, p_list):
        sl = slices[o-1]
        d_sl = flat_normed[sl].copy()
        msk = (order_map[sl] == o)
        d_sl[~msk] = np.nan

        flat_im[sl][msk] = (d_sl / p(x))[msk]

    if 0:
        hdu_list = igrins_log.get_cal_hdus(band, "HD3417")
        hd_list = [destriper.get_destriped(hdu.data) for hdu in hdu_list]
        from stsci_helper import stsci_median
        hd_spec = stsci_median(hd_list)


if __name__ == "__main__":
    #test_tracing_H()
    #test_tracing_K()
    pass
