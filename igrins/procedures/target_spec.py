import numpy as np
import scipy.ndimage as ni

from ..utils.image_combine import image_median
from ..igrins_libs.resource_helper_igrins import ResourceHelper


def _get_int_from_config(obsset, kind, default):
    v = obsset.rs.query_ref_value_from_section("EXTRACTION",
                                               kind,
                                               default=default)
    if v is not None:
        v = int(v)

    return v


def setup_extraction_parameters(obsset, order_range="-1,-1",
                                height_2dspec=0):

    _order_range_s = order_range
    try:
        order_start, order_end = map(int, _order_range_s.split(","))
    except Exception:
        msg = "Failed to parse order range: {}".format(_order_range_s)
        raise ValueError(msg)

    order_start = _get_int_from_config(obsset, "ORDER_START", order_start)
    order_end = _get_int_from_config(obsset, "ORDER_END", order_end)

    height_2dspec = _get_int_from_config(obsset, "HEIGHT_2DSPEC",
                                         height_2dspec)

    obsset.set_recipe_parameters(order_start=order_start,
                                 order_end=order_end,
                                 height_2dspec=height_2dspec)


def _get_combined_image(obsset):
    # Should not use median, Use sum.
    data_list = [hdu.data for hdu in obsset.get_hdus()]

    return np.sum(data_list, axis=0)
# def _get_combined_image(obsset):
#     data_list = [hdu.data for hdu in obsset.get_hdus()]
#     return image_median(data_list)


def get_destriped(obsset,
                  data_minus,
                  destripe_pattern=64,
                  use_destripe_mask=None,
                  sub_horizontal_median=True,
                  remove_vertical=False):

    from .destriper import destriper

    if use_destripe_mask:
        helper = ResourceHelper(obsset)
        _destripe_mask = helper.get("destripe_mask")

        destrip_mask = ~np.isfinite(data_minus) | _destripe_mask
    else:
        destrip_mask = None

    data_minus_d = destriper.get_destriped(data_minus,
                                           destrip_mask,
                                           pattern=destripe_pattern,
                                           hori=sub_horizontal_median,
                                           remove_vertical=remove_vertical)

    return data_minus_d


def get_variance_map(obsset, data_minus, data_plus):
    helper = ResourceHelper(obsset)
    _destripe_mask = helper.get("destripe_mask")

    bias_mask2 = ni.binary_dilation(_destripe_mask)

    from .variance_map import (get_variance_map,
                               get_variance_map0)

    _pix_mask = helper.get("badpix_mask")
    variance_map0 = get_variance_map0(data_minus,
                                      bias_mask2, _pix_mask)

    _gain = obsset.rs.query_ref_value("GAIN")
    variance_map = get_variance_map(data_plus, variance_map0,
                                    gain=float(_gain))

    return variance_map0, variance_map


def get_variance_map_deprecated(obsset, data_minus, data_plus):
    helper = ResourceHelper(obsset)
    _destripe_mask = helper.get("destripe_mask")

    bias_mask2 = ni.binary_dilation(_destripe_mask)

    from .variance_map import (get_variance_map,
                               get_variance_map0)

    _pix_mask = helper.get("badpix_mask")
    variance_map0 = get_variance_map0(data_minus,
                                      bias_mask2, _pix_mask)

    _gain = obsset.rs.query_ref_value("GAIN")
    variance_map = get_variance_map(data_plus, variance_map0,
                                    gain=float(_gain))

    # variance_map0 : variance without poisson noise of source + sky
    # This is used to estimate model variance where poisson noise is
    # added from the simulated spectra.
    # variance : variance with poisson noise.
    return variance_map0, variance_map


# def make_combined_images(obsset,
#                          destripe_pattern=64,
#                          use_destripe_mask=True,
#                          sub_horizontal_median=True,
#                          allow_no_b_frame=False):

#     ab_mode = obsset.recipe_name.endswith("AB")

#     obsset_a = obsset.get_subset("A", "ON")
#     obsset_b = obsset.get_subset("B", "OFF")

#     na, nb = len(obsset_a.obsids), len(obsset_b.obsids)
#     if ab_mode and (na != nb):
#         raise RuntimeError("For AB nodding, number of A and B should match!")

#     if na == 0:
#         raise RuntimeError("No A Frame images are found")

#     if nb == 0 and not allow_no_b_frame:
#         raise RuntimeError("No B Frame images are found")

#     if nb == 0:
#         a_data = _get_combined_image(obsset_a)
#         data_minus = a_data

#     else:  # nb > 0
#         # a_b != 1 for the cases when len(a) != len(b)
#         a_b = float(na) / float(nb)

#         a_data = _get_combined_image(obsset_a)
#         b_data = _get_combined_image(obsset_b)

#         data_minus = a_data - a_b * b_data

#     if destripe_pattern is not None:

#         data_minus = get_destriped(obsset,
#                                    data_minus,
#                                    destripe_pattern=destripe_pattern,
#                                    use_destripe_mask=use_destripe_mask,
#                                    sub_horizontal_median=sub_horizontal_median)

#     if nb == 0:
#         data_plus = a_data
#     else:
#         data_plus = (a_data + (a_b**2)*b_data)

#     variance_map0, variance_map = get_variance_map(obsset,
#                                                    data_minus, data_plus)

#     # hdul = obsset.get_hdul_to_write(([], data_minus))
#     # obsset.store("combined_image1", data=hdul, cache_only=True)

#     hdul = obsset.get_hdul_to_write(([], variance_map0))
#     obsset.store("combined_variance0", data=hdul, cache_only=True)

#     hdul = obsset.get_hdul_to_write(([], variance_map))
#     obsset.store("combined_variance1", data=hdul, cache_only=True)


from ..igrins_recipes.recipe_combine import (make_combined_images
                                             as _make_combined_images)


def make_combined_images(obsset,
                         allow_no_b_frame=False,
                         force_image_combine=False):

    try:
        obsset.load("combined_image1")
        combined_image_exists = True
    except Exception:
        combined_image_exists = False
        pass

    if combined_image_exists and not force_image_combine:
        print("skipped")
        return

    _make_combined_images(obsset, allow_no_b_frame=allow_no_b_frame,
                          cache_only=True)


def subtract_interorder_background(obsset, di=24, min_pixel=40):

    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1").data

    helper = ResourceHelper(obsset)
    sky_mask = helper.get("sky_mask")

    from .estimate_sky import (estimate_background,
                               get_interpolated_cubic)

    xc, yc, v, std = estimate_background(data_minus, sky_mask,
                                         di=di, min_pixel=min_pixel)

    nx = ny = 2048
    ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)

    hdul = obsset.get_hdul_to_write(([], ZI3))
    obsset.store("interorder_background", data=hdul, cache_only=True)

    hdul = obsset.get_hdul_to_write(([], ZI3))
    obsset.store("combined_image1", data=hdul, cache_only=True)


def estimate_slit_profile(obsset,
                          x1=800, x2=2048-800,
                          do_ab=True, slit_profile_mode="1d",
                          frac_slit=None):

    if slit_profile_mode == "1d":
        from .slit_profile import estimate_slit_profile_1d
        estimate_slit_profile_1d(obsset, x1=x1, x2=x2, do_ab=do_ab,
                                 frac_slit=frac_slit)
    elif slit_profile_mode == "uniform":
        from .slit_profile import estimate_slit_profile_uniform
        estimate_slit_profile_uniform(obsset, do_ab=do_ab,
                                      frac_slit=frac_slit)
    else:
        msg = ("Unknwon mode ({}) in slit_profile estimation"
               .format(slit_profile_mode))
        raise ValueError(msg)


def get_wvl_header_data(obsset, wavelength_increasing_order=False):
    # from ..libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
    # fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
    #                           extractor.basenames["wvlsol"])
    # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
    # f = pyfits.open(fn)

    hdu = obsset.load_resource_sci_hdu_for("wvlsol_fits")
    if wavelength_increasing_order:
        from ..utils import iraf_helper
        header = iraf_helper.invert_order(hdu.header)

        def convert_data(d):
            return d[::-1]
    else:
        header = hdu.header

        def convert_data(d):
            return d

    return header.copy(), hdu.data, convert_data


def store_1dspec(obsset, v_list, s_list, sn_list=None):

    basename_postfix = obsset.basename_postfix

    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    d = np.array(v_list)
    v_data = convert_data(d.astype("float32"))

    hdul = obsset.get_hdul_to_write(([], v_data))
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify(option="silentfix")

    obsset.store("VARIANCE_FITS", hdul,
                 postfix=basename_postfix)

    if sn_list is not None:
        d = np.array(sn_list)
        sn_data = convert_data(d.astype("float32"))

        hdul = obsset.get_hdul_to_write(([], sn_data))
        wvl_header.update(hdul[0].header)
        hdul[0].header = wvl_header
        obsset.store("SN_FITS", hdul,
                     postfix=basename_postfix)

    d = np.array(s_list)
    s_data = convert_data(d.astype("float32"))

    hdul = obsset.get_hdul_to_write(([], s_data),
                                    ([], convert_data(wvl_data)))
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify(option="silentfix")

    obsset.store("SPEC_FITS", hdul,
                 postfix=basename_postfix)


def store_2dspec(obsset,
                 conserve_flux=True):

    basename_postfix = obsset.basename_postfix

    height_2dspec = obsset.get_recipe_parameter("height_2dspec")

    from .shifted_images import ShiftedImages
    hdul = obsset.load("WVLCOR_IMAGE", postfix=basename_postfix)
    shifted = ShiftedImages.from_hdul(hdul)

    data_shft = shifted.image
    variance_map_shft = shifted.variance

    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    # wvl_header, wvl_data, convert_data = \
    #             self.get_wvl_header_data(igr_storage,
    #                                      extractor)

    # from ..libs.load_fits import open_fits
    # f_obj = open_fits(extractor.obj_filenames[0])
    # f_obj[0].header.extend(wvl_header)

    # # tgt_basename = extractor.pr.tgt_basename
    # tgt_basename = mastername

    # from ..libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC
    # cent = igr_storage.load1(FLATCENTROID_SOL_JSON_DESC,
    #                          extractor.basenames["flat_on"])
    # fn = ("calib/primary/20140525/"
    #       "FLAT_SDCK_20140525_0074.centroid_solutions.json")
    # #cent = json.load(open(fn))
    # _bottom_up_solutions = cent["bottom_up_solutions"]
    # old_orders = extractor.get_old_orders()
    # _o_s = dict(zip(old_orders, _bottom_up_solutions))
    # new_bottom_up_solutions = [_o_s[o] for o in \
    #                            extractor.orders_w_solutions]

    bottom_up_solutions_ = obsset.load_resource_for("aperture_definition")
    bottom_up_solutions = bottom_up_solutions_["bottom_up_solutions"]

    helper = ResourceHelper(obsset)
    ordermap_bpixed = helper.get("ordermap_bpixed")

    from .correct_distortion import get_rectified_2dspec
    _ = get_rectified_2dspec(data_shft,
                             ordermap_bpixed,
                             bottom_up_solutions,
                             conserve_flux=conserve_flux,
                             height=height_2dspec)
    d0_shft_list, msk_shft_list = _

    with np.errstate(invalid="ignore"):
        d = np.array(d0_shft_list) / np.array(msk_shft_list)

    hdul = obsset.get_hdul_to_write(([], convert_data(d.astype("float32"))))
    # wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header

    obsset.store("SPEC2D_FITS", hdul, postfix=basename_postfix)

    # OUTPUT VAR2D, added by Kyle Kaplan Feb 25, 2015 to get variance map
    # outputted as a datacube
    _ = get_rectified_2dspec(variance_map_shft,
                             ordermap_bpixed,
                             bottom_up_solutions,
                             conserve_flux=conserve_flux,
                             height=height_2dspec)
    d0_shft_list, msk_shft_list = _

    with np.errstate(invalid="ignore"):
        d = np.array(d0_shft_list) / np.array(msk_shft_list)

    hdul = obsset.get_hdul_to_write(([], convert_data(d.astype("float32"))))
    # wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header

    obsset.store("VAR2D_FITS", hdul, postfix=basename_postfix)


def extract_stellar_spec(obsset, extraction_mode="optimal",
                         conserve_2d_flux=True, calculate_sn=True):

    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    postfix = obsset.basename_postfix
    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
                                          postfix=postfix).data

    orderflat = helper.get("orderflat")
    data_minus_flattened = data_minus / orderflat

    variance_map = obsset.load_fits_sci_hdu("combined_variance1",
                                            postfix=postfix).data
    variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
                                             postfix=postfix).data

    slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    # from .slit_profile import get_profile_func
    # profile = get_profile_func(obsset)

    gain = float(obsset.rs.query_ref_value("gain"))

    profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
                                           postfix=postfix).data

    from .spec_extract_w_profile import extract_spec_using_profile
    _ = extract_spec_using_profile(ap, profile_map,
                                   variance_map,
                                   variance_map0,
                                   data_minus_flattened,
                                   orderflat,
                                   ordermap, ordermap_bpixed,
                                   slitpos_map,
                                   slitoffset_map,
                                   gain,
                                   extraction_mode=extraction_mode,
                                   debug=False)

    s_list, v_list, cr_mask, aux_images = _

    if calculate_sn:
        # calculate S/N per resolution
        wvl_solutions = helper.get("wvl_solutions")

        sn_list = []
        for wvl, s, v in zip(wvl_solutions,
                             s_list, v_list):

            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw
            # print pixel_per_res_element[1024]
            # len(pixel_per_res_element) = 2047. But we ignore it.

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)
    else:
        sn_list = None

    store_1dspec(obsset, v_list, s_list, sn_list=sn_list)

    hdul = obsset.get_hdul_to_write(([], data_minus),
                                    ([], aux_images["synth_map"]))
    obsset.store("DEBUG_IMAGE", hdul)

    shifted = aux_images["shifted"]

    _hdul = shifted.to_hdul()
    hdul = obsset.get_hdul_to_write(*_hdul)
    obsset.store("WVLCOR_IMAGE", hdul)

    # store_2dspec(obsset,
    #              shifted.image,
    #              shifted.variance,
    #              ordermap_bpixed,
    #              cr_mask=cr_mask,
    #              conserve_flux=conserve_2d_flux,
    #              height_2dspec=height_2dspec)


def extract_stellar_spec_pp(obsset, extraction_mode="optimal", height_2dspec=0,
                            conserve_2d_flux=True, calculate_sn=True):
    """
    This function reads in "WVLCOR_IMAGE" and use extract_spec_from_shifted
    for spec-extraction.

    c.f., extract_stellar_spec work on the combined images and
    do wvl-cor by itself.

    """
    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    postfix = obsset.basename_postfix
    # data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
    #                                       postfix=postfix).data

    # orderflat = helper.get("orderflat")
    # data_minus_flattened = data_minus / orderflat

    # variance_map = obsset.load_fits_sci_hdu("combined_variance1",
    #                                         postfix=postfix).data
    # variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
    #                                          postfix=postfix).data

    # slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    # slitpos_map = helper.get("slitposmap")

    # # from .slit_profile import get_profile_func
    # # profile = get_profile_func(obsset)

    # gain = float(obsset.rs.query_ref_value("gain"))

    # profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
    #                                        postfix=postfix).data

    from .shifted_images import ShiftedImages
    hdul = obsset.load("WVLCOR_IMAGE", postfix=postfix)
    shifted = ShiftedImages.from_hdul(hdul)

    from .spec_extract_from_shifted import extract_spec_from_shifted
    _ = extract_spec_from_shifted(ap,
                                  ordermap, ordermap_bpixed,
                                  shifted,
                                  extraction_mode=extraction_mode,
                                  debug=False)

    s_list, v_list = _

    if calculate_sn:
        # calculate S/N per resolution
        wvl_solutions = helper.get("wvl_solutions")

        sn_list = []
        for wvl, s, v in zip(wvl_solutions,
                             s_list, v_list):

            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw
            # print pixel_per_res_element[1024]
            # len(pixel_per_res_element) = 2047. But we ignore it.

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)
    else:
        sn_list = None

    store_1dspec(obsset, v_list, s_list, sn_list=sn_list)

    # hdul = obsset.get_hdul_to_write(([], data_minus),
    #                                 ([], aux_images["synth_map"]))
    # obsset.store("DEBUG_IMAGE", hdul)

    # shifted = aux_images["shifted"]

    # _hdul = shifted.to_hdul()
    # hdul = obsset.get_hdul_to_write(*_hdul)
    # obsset.store("WVLCOR_IMAGE", hdul)


def extract_extended_spec1(obsset, data,
                           variance_map, variance_map0,
                           lacosmic_thresh=0.):

    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    orderflat = helper.get("orderflat")

    data_minus = data
    data_minus_flattened = data_minus / orderflat

    slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    wvl_solutions = helper.get("wvl_solutions")

    # from .slit_profile import get_profile_func
    # profile = get_profile_func(obsset)

    gain = float(obsset.rs.query_ref_value("gain"))

    postfix = obsset.basename_postfix
    profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
                                           postfix=postfix).data

    from .spec_extract_w_profile import extract_spec_uniform
    _ = extract_spec_uniform(ap, profile_map,
                             variance_map,
                             variance_map0,
                             data_minus_flattened,
                             data_minus, orderflat,  #
                             ordermap, ordermap_bpixed,
                             slitpos_map,
                             slitoffset_map,
                             gain,
                             lacosmic_thresh=lacosmic_thresh,
                             debug=False)

    s_list, v_list, cr_mask, aux_images = _

    return s_list, v_list, cr_mask, aux_images


def extract_extended_spec(obsset, lacosmic_thresh=0.):

    # refactored from recipe_extract.ProcessABBABand.process

    from ..utils.load_fits import get_science_hdus
    postfix = obsset.basename_postfix
    hdul = get_science_hdus(obsset.load("COMBINED_IMAGE1",
                                        postfix=postfix))
    data = hdul[0].data

    if len(hdul) == 3:
        variance_map = hdul[1].data
        variance_map0 = hdul[2].data
    else:
        variance_map = obsset.load_fits_sci_hdu("combined_variance1",
                                                postfix=postfix).data
        variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
                                                 postfix=postfix).data

    _ = extract_extended_spec1(obsset, data,
                               variance_map, variance_map0,
                               lacosmic_thresh=lacosmic_thresh)

    s_list, v_list, cr_mask, aux_images = _

    if 1:
        # calculate S/N per resolution
        helper = ResourceHelper(obsset)
        wvl_solutions = helper.get("wvl_solutions")

        sn_list = []
        for wvl, s, v in zip(wvl_solutions,
                             s_list, v_list):

            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw
            # print pixel_per_res_element[1024]
            # len(pixel_per_res_element) = 2047. But we ignore it.
            sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)

    store_1dspec(obsset, v_list, s_list, sn_list=sn_list)

    shifted = aux_images["shifted"]

    _hdul = shifted.to_hdul()
    hdul = obsset.get_hdul_to_write(*_hdul)
    obsset.store("WVLCOR_IMAGE", hdul, postfix=obsset.basename_postfix)

    # store_2dspec(obsset,
    #              shifted.image,
    #              shifted.variance_map,
    #              ordermap_bpixed,
    #              cr_mask=cr_mask,
    #              conserve_flux=conserve_2d_flux,
    #              height_2dspec=height_2dspec)


def _derive_data_for_slit_profile(ap, data_minus_flattened,
                                  spec1d, ordermap):

    def expand_1dspec_to_2dspec(s1d, o2d, min_order=None):
        mmm = (o2d > 0) & (o2d < 999)
        xi = np.indices(mmm.shape)[-1]
        if min_order is None:
            min_order = o2d[mmm].min()
        indx = (o2d[mmm]-min_order)*2048 + xi[mmm]

        s2 = np.empty(mmm.shape, dtype=float)
        s2.fill(np.nan)
        s2[mmm] = np.take(s1d, indx)

        return s2

    # ordermap = extractor.ordermap_bpixed (?)

    # correction factors for aperture width
    ds0 = np.array([ap(o, ap.xi, 1.) - ap(o, ap.xi, 0.) for o in ap.orders])
    ds = ds0 / 50.  # 50 is just a typical width.

    # try to estimate threshold to mask the spectra
    s_max = np.nanpercentile(spec1d / ds0, 90)  # mean counts per pixel
    s_cut = 0.03 * s_max  # 3 % of s_max

    ss_cut = s_cut * ds0
    with np.errstate(invalid="ignore"):
        msk = spec1d < ss_cut
    ss = np.ma.array(spec1d, mask=msk).filled(np.nan)

    s2d = expand_1dspec_to_2dspec(ss/ds, ordermap)

    ods = data_minus_flattened/s2d

    return ods


def _get_slit_profile_options(slit_profile_options):
    slit_profile_options = slit_profile_options.copy()
    n_comp = slit_profile_options.pop("n_comp", None)
    stddev_list = slit_profile_options.pop("stddev_list", None)
    if slit_profile_options:
        msgs = ["unrecognized options: %s"
                % slit_profile_options,
                "\n",
                "Available options are: n_comp, stddev_list"]

        raise ValueError("".join(msgs))

    return n_comp, stddev_list


def _estimate_slit_profile_glist(ap, ods,
                                 ordermap_bpixed, slitpos_map,
                                 x1=800, x2=2048-800,
                                 do_ab=True):
    """
    return a profile function. This has a signature of

    def profile(y_slit_pos):
        return profile_value

    """

    n_comp, stddev_list = _get_slit_profile_options({})

    # omap, slitpos = extractor.ordermap_bpixed, extractor.slitpos_map

    msk1 = np.isfinite(ods)  # & bias_mask

    x_min, x_max = x1, x2
    y_min, y_max = 128, 2048-128

    xx = np.array([ap(o, 1024, .5) for o in ap.orders])
    i1, i2 = np.searchsorted(xx, [y_min, y_max])
    o1, o2 = ap.orders[i1], ap.orders[i2]
    msk2 = (o1 < ordermap_bpixed) & (ordermap_bpixed < o2)

    msk2[:, :x_min] = False
    msk2[:, x_max:] = False

    msk = msk1 & msk2  # & (slitpos < 0.5)

    ods_mskd = ods[msk]
    s_mskd = slitpos_map[msk]

    from .slit_profile_model import derive_multi_gaussian_slit_profile
    g_list0 = derive_multi_gaussian_slit_profile(s_mskd, ods_mskd,
                                                 n_comp=n_comp,
                                                 stddev_list=stddev_list)

    return g_list0


def _run_order_main(args):
    o, x, y, s, g_list0, logi = args
    print(o)

    xmsk = (800 < x) & (x < 2048-800)

    # check if there is enough pixels to derive new slit profile
    if len(s[xmsk]) > 8000:  # FIXME : ?? not sure if this was what I meant?
        from .slit_profile_model import (
            derive_multi_gaussian_slit_profile
        )

        g_list = derive_multi_gaussian_slit_profile(
            y[xmsk], s[xmsk],
            n_comp=4, stddev_list=[0.02, 0.04, 0.1, 0.2]
        )
    else:
        g_list = g_list0

    if len(x) < 1000:
        # print "skipping"
        # def _f(order, xpixel, slitpos):
        #     return g_list(slitpos)

        # func_dict[o] = g_list0
        return None

    if 0:

        from . import slit_profile_model as slit_profile_model
        debug_func = slit_profile_model.get_debug_func()
        debug_func(g_list, g_list, y, s)

    from ..libs.slit_profile_2d_model import get_varying_conv_gaussian_model
    Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
    vcg = Varying_Conv_Gaussian_Model()

    if 0:
        def _vcg(y):
            centers = np.zeros_like(y) + 1024
            return vcg(centers, y)

        debug_func(_vcg, _vcg, y, s)

    from astropy.modeling import fitting

    fitter = fitting.LevMarLSQFitter()
    t = fitter(vcg, x, y, s, maxiter=100000)

    # func_dict[o] = t

    # print "saveing figure"

    if logi is not None:
        logi.submit("raw_data_scatter",
                    (x, y, s))

        logi.submit("profile_sub_scatter",
                    (x, y, s-vcg(x, y)),
                    label="const. model")

        logi.submit("profile_sub_scatter",
                    (x, y, s-t(x, y)),
                    label="varying model")

    # return t
    return g_list.parameters, t.parameters


def _estimate_slit_profile_gauss_2d(ap, ods, g_list0,
                                    # spec1d,
                                    ordermap_bpixed, slitpos_map,
                                    x1=800, x2=2048-800,
                                    do_ab=True,
                                    n_process=1):
    _, xx = np.indices(ods.shape)

    msk1 = np.isfinite(ods)  # & np.isfinite(ode) & bias_mask

    from ..libs.slit_profile_2d_model import Logger
    logger = Logger("test.pdf")

    # def oo(x_pixel, slitpos):
    #     return g_list0(slitpos)

    # def _run_order(o):
    #     print o

    # if 0:
    #     msk2 = ordermap_bpixed == o

    #     msk = msk1 & msk2 # & (slitpos < 0.5)

    #     x = xx[msk]
    #     y = slitpos[msk]
    #     s = ods[msk]

    #     xmsk = (800 < x) & (x < 2048-800)

    #     # check if there is enough pixels to derive new slit profile
    #     if len(s[xmsk]) > 8000:
    #         from igrins.libs.slit_profile_model import (
    #                  derive_multi_gaussian_slit_profile
    #         )
    #         g_list = derive_multi_gaussian_slit_profile(y[xmsk], s[xmsk])
    #     else:
    #         g_list = g_list0

    #     if len(x) < 1000:
    #         # print "skipping"
    #         # def _f(order, xpixel, slitpos):
    #         #     return g_list(slitpos)

    #         # func_dict[o] = g_list0
    #         return oo

    #     if 0:

    #         import igrins.libs.slit_profile_model as slit_profile_model
    #         debug_func = slit_profile_model.get_debug_func()
    #         debug_func(g_list, g_list, y, s)

    #     from igrins.libs.slit_profile_2d_model import
    #          get_varying_conv_gaussian_model
    #     Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
    #     vcg = Varying_Conv_Gaussian_Model()

    #     if 0:
    #         def _vcg(y):
    #             centers = np.zeros_like(y) + 1024
    #             return vcg(centers, y)

    #         debug_func(_vcg, _vcg, y, s)

    #     from astropy.modeling import fitting

    #     fitter = fitting.LevMarLSQFitter()
    #     t = fitter(vcg, x, y, s, maxiter=100000)

    #     # func_dict[o] = t

    #     # print "saveing figure"
    #     logi = logger.open("slit_profile_2d_conv_gaussian",
    #                        ("basename", o))

    #     logi.submit("raw_data_scatter",
    #                 (x, y, s))

    #     logi.submit("profile_sub_scatter",
    #                 (x, y, s-vcg(x, y)),
    #                 label="const. model")

    #     logi.submit("profile_sub_scatter",
    #                 (x, y, s-t(x, y)),
    #                 label="varying model")

    #     return t

    print("deriving 2d slit profiles")

    args = []
    for o in ap.orders:
        msk2 = ordermap_bpixed == o

        msk = msk1 & msk2  # & (slitpos < 0.5)

        x = xx[msk]
        y = slitpos_map[msk]
        s = ods[msk]

        logi = logger.open("slit_profile_2d_conv_gaussian",
                           ("basename", o))

        args.append((o, x, y, s, g_list0, logi))

    # n_process = 4
    if n_process > 1:
        from multiprocessing import Pool
        # from multiprocessing.pool import ThreadPool as Pool
        p = Pool(n_process)
        _ = p.map(_run_order_main, args)
    else:
        _ = []
        for a in args:
            r = _run_order_main(a)
            _.append(r)

    print("done")
    # func_dict.update((k, v) )

    from ..libs.slit_profile_2d_model import get_varying_conv_gaussian_model

    func_dict = {}
    for o, v in zip(ap.orders, _):
        if v is None:
            continue

        g_list_parameters, vcg_parameters = v
        g_list = type(g_list0)()
        g_list.parameters = g_list_parameters

        Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
        vcg = Varying_Conv_Gaussian_Model()
        vcg.parameters = vcg_parameters

        func_dict[o] = vcg

    for key in sorted(logger.instances.keys()):
        logger.finalize(key)

    logger.pdf.close()

    # for o in ap.orders:
    #     print o

    #     logi.close()

    def profile(order, x_pixel, slitpos):
        # print "profile", order, func_dict.keys()
        def oo(x_pixel, slitpos):
            return g_list0(slitpos)

        return func_dict.get(order, oo)(x_pixel, slitpos)

    return profile


def update_slit_profile(obsset, slit_profile_mode="gauss2d", frac_slit=None):

    # now try to derive the n-gaussian profile
    # print("updating profile using the multi gauss fit")
    assert slit_profile_mode in ["gauss", "gauss2d"]

    s_list = list(obsset.load_fits_sci_hdu("SPEC_FITS").data)

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1").data
    orderflat = helper.get("orderflat")
    data_minus_flattened = data_minus / orderflat

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    ods = _derive_data_for_slit_profile(ap, data_minus_flattened,
                                        s_list, ordermap=ordermap)

    glist = _estimate_slit_profile_glist(ap, ods,
                                         ordermap_bpixed, slitpos_map,
                                         x1=800, x2=2048-800,
                                         do_ab=True)

    if slit_profile_mode == "gauss2d":
        profile = _estimate_slit_profile_gauss_2d(ap, ods, glist,
                                                  ordermap_bpixed, slitpos_map,
                                                  x1=800, x2=2048-800,
                                                  do_ab=True)
    elif slit_profile_mode == "gauss":
        def profile(order, x_pixel, slitpos):
            return glist(slitpos)
    else:
        msg = "unexpected slit_profile_mode: %s" % slit_profile_mode
        raise ValueError(msg)

    # profile_map = extractor.make_profile_map(ap,
    #                                          profile,
    #                                          frac_slit=self.frac_slit)

    from .slit_profile import make_slitprofile_map
    profile_map = make_slitprofile_map(ap, profile,
                                       ordermap, slitpos_map,
                                       frac_slit=frac_slit)

    hdul = obsset.get_hdul_to_write(([], profile_map))
    obsset.store("slitprofile_fits", hdul, cache_only=True)


# if 0:
#                 # _ = self._extract_spec_using_profile(extractor,
#                 #                                      ap, profile_map,
#                 #                                      variance_map,
#                 #                                      variance_map0,
#                 #                                      data_minus_flattened,
#                 #                                      ordermap,
#                 #                                      slitpos_map,
#                 #                                      slitoffset_map,
#                 #                                      debug=False)

#                 # s_list, v_list, cr_mask, aux_images = _


#                 sig_map = aux_images["sig_map"]
#                 synth_map = aux_images["synth_map"]
#                 shifted = aux_images["shifted"]

#                 if 0: # save aux files
#                     synth_map = ap.make_synth_map(ordermap, slitpos_map,
#                                                   profile_map, s_list,
#                                                   slitoffset_map=slitoffset_map
#                                                   )

#                     shifted = extractor.get_shifted_all(ap, profile_map,
#                                                         variance_map,
#                                                         synth_map,
#                                                         slitoffset_map,
#                                                         debug=False)
