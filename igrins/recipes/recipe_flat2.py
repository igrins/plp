from __future__ import print_function

from collections import namedtuple

import numpy as np
import scipy.ndimage as ni

from astropy.io.fits import Card

from ..libs.stsci_helper import stsci_median

from igrins import DESCS


def combine_flat_off(obsset, destripe=True):
    # destripe=True):

    obsset_off = obsset.get_subset("OFF")
    cards = []

    data_list = [hdu.data for hdu in obsset_off.get_hdus()]

    flat_off = stsci_median(data_list)

    if destripe:
        from ..libs.destriper import destriper
        flat_off = destriper.get_destriped(flat_off)

        cards.append(Card("HISTORY",
                          "IGR: image destriped."))

    hdul = obsset_off.get_hdul_to_write((cards, flat_off))
    obsset_off.store(DESCS["FLAT_OFF"], hdul, cache_only=True)


def make_hotpix_mask(obsset,
                     sigma_clip1=100, sigma_clip2=10,
                     medfilter_size=None):

    # caldb = helper.get_caldb()
    # master_obsid = obsset.obsids[0]

    obsset_off = obsset.get_subset("OFF")

    flat_off_hdu = obsset_off.load_fits_sci_hdu(DESCS["FLAT_OFF"])
    flat_off = flat_off_hdu.data

    from ..libs import badpixel as bp
    hotpix_mask = bp.badpixel_mask(flat_off,
                                   sigma_clip1=sigma_clip1,
                                   sigma_clip2=sigma_clip2,
                                   medfilter_size=medfilter_size)

    bg_std = flat_off[~hotpix_mask].std()

    flat_off_cards = [Card("BG_STD", bg_std, "IGR: stddev of combined flat")]

    # caldb = helper.get_caldb()

    # hdul = obsset.get_hdul_to_write(([], hotpix_mask))
    obsset_off.store(DESCS["HOTPIX_MASK"], hotpix_mask, item_type="mask")

    # save fits with updated header
    flat_off_hdu.header.update(flat_off_cards)
    obsset_off.store(DESCS["FLAT_OFF"], flat_off_hdu)


def combine_flat_on(obsset):
    # destripe=True):

    obsset_on = obsset.get_subset("ON")

    data_list = [hdu.data for hdu in obsset_on.get_hdus()]

    flat_on = stsci_median(data_list)

    flat_std = np.std(data_list, axis=0)

    hdu_list = [([], flat_on),
                ([], flat_std)]

    hdul = obsset_on.get_hdul_to_write(*hdu_list)
    obsset_on.store(DESCS["FLAT_ON"], hdul)


DeadpixMaskResult = namedtuple("DeadpixMaskResult", ["flat_normed",
                                                     "flat_bpixed",
                                                     "flat_mask",
                                                     "deadpix_mask",
                                                     "flat_info"])


def _make_deadpix_mask(flat_on, flat_std, hotpix_mask,
                       deadpix_mask_old=None,
                       deadpix_thresh=0.6, smooth_size=9):

    # normalize it
    from ..libs.trace_flat import (get_flat_normalization,
                                   get_flat_mask_auto,
                                   estimate_bg_mean_std)

    bg_mean, bg_fwhm = estimate_bg_mean_std(flat_on)
    norm_factor = get_flat_normalization(flat_on,
                                         bg_fwhm, hotpix_mask)

    flat_normed = flat_on / norm_factor
    flat_std_normed = ni.median_filter(flat_std / norm_factor,
                                       size=(3, 3))
    bg_fwhm_norm = bg_fwhm/norm_factor

    # mask out bpix
    flat_bpixed = flat_normed.astype("d")
    # by default, astype returns new array.

    flat_bpixed[hotpix_mask] = np.nan

    flat_mask = get_flat_mask_auto(flat_bpixed)
    # flat_mask = get_flat_mask(flat_bpixed, bg_std_norm,
    #                           sigma=flat_mask_sigma)

    # get dead pixel mask
    flat_smoothed = ni.median_filter(flat_normed,
                                     [smooth_size, smooth_size])
    # flat_smoothed[order_map==0] = np.nan
    flat_ratio = flat_normed/flat_smoothed
    flat_std_mask = (flat_smoothed - flat_normed) > 5*flat_std_normed

    refpixel_mask = np.ones(flat_mask.shape, bool)
    # mask out outer boundaries
    refpixel_mask[4:-4, 4:-4] = False

    deadpix_mask = ((flat_ratio < deadpix_thresh) &
                    flat_std_mask & flat_mask & (~refpixel_mask))

    if deadpix_mask_old is not None:
        deadpix_mask = deadpix_mask | deadpix_mask_old

    flat_bpixed[deadpix_mask] = np.nan

    r = DeadpixMaskResult(flat_normed=flat_normed,
                          flat_bpixed=flat_bpixed,
                          flat_mask=flat_mask,
                          deadpix_mask=deadpix_mask,
                          flat_info=dict(bg_fwhm_norm=bg_fwhm_norm))

    return r


def make_deadpix_mask(obsset,  # helper, band, obsids,
                      # flat_mask_sigma=5.,
                      deadpix_thresh=0.6,
                      smooth_size=9):

    obsset_on = obsset.get_subset("ON")
    obsset_off = obsset.get_subset("OFF")

    # we are using flat on images without subtracting off images.
    flat_on_hdus = obsset_on.load(DESCS["FLAT_ON"])

    flat_on = flat_on_hdus[0].data
    flat_std = flat_on_hdus[1].data

    hotpix_mask = obsset_off.load(DESCS["HOTPIX_MASK"])

    f = obsset.load_ref_data(kind="DEFAULT_DEADPIX_MASK")

    deadpix_mask_old = f[0].data.astype(bool)

    # main routine
    r = _make_deadpix_mask(flat_on, flat_std, hotpix_mask,
                           deadpix_mask_old,
                           deadpix_thresh=deadpix_thresh,
                           smooth_size=smooth_size)

    obsset_on.store(DESCS["FLAT_NORMED"],
                    obsset_on.get_hdul_to_write(([], r.flat_normed)))

    obsset_on.store(DESCS["FLAT_BPIXED"],
                    obsset_on.get_hdul_to_write(([], r.flat_bpixed)))

    obsset_on.store(DESCS["FLAT_MASK"], r.flat_mask)

    obsset_on.store(DESCS["DEADPIX_MASK"], r.deadpix_mask)

    obsset_on.store(DESCS["FLATON_JSON"], r.flat_info)


def identify_order_boundaries(obsset):

    from ..libs.trace_flat import get_y_derivativemap

    obsset_on = obsset.get_subset("ON")

    flat_normed = obsset_on.load_fits_sci_hdu(DESCS["FLAT_NORMED"]).data
    flat_bpixed = obsset_on.load_fits_sci_hdu(DESCS["FLAT_BPIXED"]).data

    flat_mask = obsset_on.load(DESCS["FLAT_MASK"])

    flaton_info = obsset_on.load(DESCS["FLATON_JSON"])
    bg_fwhm_normed = flaton_info["bg_fwhm_norm"]

    flat_deriv_ = get_y_derivativemap(flat_normed, flat_bpixed,
                                      bg_fwhm_normed,
                                      max_sep_order=150, pad=10,
                                      flat_mask=flat_mask)

    flat_deriv = flat_deriv_["data"]
    flat_deriv_pos_msk = flat_deriv_["pos_mask"]
    flat_deriv_neg_msk = flat_deriv_["neg_mask"]

    hdu_list = [([], flat_deriv),
                ([], flat_deriv_pos_msk),
                ([], flat_deriv_neg_msk)]

    hdul = obsset_on.get_hdul_to_write(*hdu_list)
    obsset_on.store(DESCS["FLAT_DERIV"], hdul)


def _check_boundary_orders(cent_list, nx=2048):

    c_list = []
    for xc, yc in cent_list:
        p = np.polyfit(xc[~yc.mask], yc.data[~yc.mask], 2)
        c_list.append(np.polyval(p, nx/2.))

    indexes = np.argsort(c_list)

    return [cent_list[i] for i in indexes]


def trace_order_boundaries(obsset):

    from ..libs.trace_flat import identify_horizontal_line

    obsset_on = obsset.get_subset("ON")

    hdu_list = obsset_on.load(DESCS["flat_deriv"])

    flat_deriv = hdu_list[0].data
    flat_deriv_pos_msk = hdu_list[1].data > 0
    flat_deriv_neg_msk = hdu_list[2].data > 0

    flaton_info = obsset_on.load(DESCS["flaton_json"])
    bg_fwhm_normed = flaton_info["bg_fwhm_norm"]

    ny, nx = flat_deriv.shape

    cent_bottom_list = identify_horizontal_line(flat_deriv,
                                                flat_deriv_pos_msk,
                                                pad=10,
                                                bg_std=bg_fwhm_normed)

    cent_bottom_list = _check_boundary_orders(cent_bottom_list, nx=nx)

    cent_up_list = identify_horizontal_line(-flat_deriv,
                                            flat_deriv_neg_msk,
                                            pad=10,
                                            bg_std=bg_fwhm_normed)

    cent_up_list = _check_boundary_orders(cent_up_list, nx=nx)

    obsset_on.store(DESCS["FLATCENTROIDS_JSON"],
                    dict(bottom_centroids=cent_bottom_list,
                         up_centroids=cent_up_list))


def stitch_up_traces(obsset):
    # from igrins.libs.process_flat import trace_solutions
    # trace_solution_products, trace_solution_products_plot = \
    #                          trace_solutions(trace_products)

    from ..libs.igrins_detector import IGRINSDetector
    nx = IGRINSDetector.nx

    from ..libs.trace_flat import trace_centroids_chevyshev

    obsset_on = obsset.get_subset("ON")

    centroids_dict = obsset_on.load(DESCS["flatcentroids_json"])

    bottom_centroids = centroids_dict["bottom_centroids"]
    up_centroids = centroids_dict["up_centroids"]

    _ = trace_centroids_chevyshev(bottom_centroids,
                                  up_centroids,
                                  domain=[0, nx],
                                  ref_x=nx/2)

    bottom_up_solutions_full, bottom_up_solutions, bottom_up_centroids = _

    assert len(bottom_up_solutions_full) != 0

    from numpy.polynomial import Polynomial

    bottom_up_solutions_as_list = []

    for b, d in bottom_up_solutions_full:

        bb, dd = b.convert(kind=Polynomial), d.convert(kind=Polynomial)
        bb_ = ("poly", bb.coef)
        dd_ = ("poly", dd.coef)
        bottom_up_solutions_as_list.append((bb_, dd_))

    def jsonize_cheb(l):
        return [(repr(l1), l1.coef, l1.domain, l1.window) for l1 in l]

    r = dict(orders=[],
             bottom_up_solutions=bottom_up_solutions_as_list,
             bottom_up_centroids=bottom_up_centroids,
             bottom_up_solutions_qa=[jsonize_cheb(bottom_up_solutions[0]),
                                     jsonize_cheb(bottom_up_solutions[1])])

    obsset_on.store(DESCS["flatcentroid_sol_json"], r)


def make_bias_mask(obsset):

    obsset_on = obsset.get_subset("ON")

    flatcentroid_info = obsset_on.load(DESCS["flatcentroid_sol_json"])

    bottomup_solutions = flatcentroid_info["bottom_up_solutions"]

    orders = list(range(len(bottomup_solutions)))

    from ..libs.apertures import Apertures
    ap = Apertures(orders, bottomup_solutions)

    order_map2 = ap.make_order_map(mask_top_bottom=True)

    # from igrins.libs.storage_descriptions import FLAT_MASK_DESC
    # flat_mask = igr_storage.load1(FLAT_MASK_DESC,
    #                               flat_on_filenames[0])

    flat_mask = obsset_on.load(DESCS["flat_mask"])
    bias_mask = flat_mask & (order_map2 > 0)

    obsset_on.store(DESCS["bias_mask"], bias_mask)


def update_db(obsset):

    obsset_off = obsset.get_subset("OFF")
    obsset_on = obsset.get_subset("ON")

    obsset_off.add_to_db("flat_off")
    obsset_on.add_to_db("flat_on")

####


def store_qa(obsset_on, obsset_off):

    # Prepare figures.

    from matplotlib.figure import Figure
    from ..libs.process_flat import plot_trace_solutions
    from ..libs.flat_qa import check_trace_order

    fig1 = Figure(figsize=[9, 4])

    flat_deriv = obsset_on.load_image("flat_deriv")
    trace_dict = obsset_on.load_item("flatcentroids_json")

    check_trace_order(flat_deriv, trace_dict, fig1)

    flat_normed = obsset_on.load_image("flat_normed")
    flatcentroid_sol_json = obsset_on.load_item("flatcentroid_sol_json")

    fig2, fig3 = plot_trace_solutions(flat_normed,
                                      flatcentroid_sol_json)

    # Now save them

    from ..libs.qa_helper import figlist_to_pngs
    # get_filename = helper.get_section_filename_base
    dest_dir = obsset_on.query_item_path("qa_flat_aperture_dir",
                                         subdir="aperture")
    # aperture_figs = get_filename("QA_PATH",
    #                              "aperture_"+flaton_basename,
    #                              "aperture_"+flaton_basename)

    figlist_to_pngs(dest_dir, [fig1, fig2, fig3])

    # if 1: # now trace the orders

    #     #del trace_solution_products["bottom_up_solutions"]
    #     igr_storage.store(trace_solution_products,
    #                       mastername=flat_on_filenames[0],
    #                       masterhdu=flat_on_hdu_list[0])

###


from ..pipeline.steps import Step


steps = [Step("Combine Flat-Off", combine_flat_off),
         Step("Hotpix Mask", make_hotpix_mask,
              sigma_clip1=100, sigma_clip2=5),
         Step("Combine Flat-On", combine_flat_on),
         Step("Deadpix Mask", make_deadpix_mask,
              deadpix_thresh=0.6, smooth_size=9),
         Step("Identify Order Boundary", identify_order_boundaries),
         Step("Trace Order Boundary", trace_order_boundaries),
         Step("Stitch Up Traces", stitch_up_traces),
         Step("Bias Mask", make_bias_mask),
         Step("Update DB", update_db),
]


if __name__ == "__main__":
    pass
