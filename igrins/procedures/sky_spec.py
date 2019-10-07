import json
import numpy as np
import warnings

from .. import DESCS

from ..utils.image_combine import image_median
from ..igrins_libs.resource_helper_igrins import ResourceHelper
# from ..libs.image_combine import destripe_sky


from .aperture_helper import get_simple_aperture_from_obsset

from .destripe_helper import sub_p64_from_guard, sub_bg64_from_guard


def _get_combined_image(obsset):
    data_list = [hdu.data for hdu in obsset.get_hdus()]
    return image_median(data_list)


def get_median_combined_image(obsset):
    im = _get_combined_image(obsset)

    return im


def get_median_combined_image_n_exptime(obsset):
    im = _get_combined_image(obsset)

    exptime = get_exptime(obsset)

    return im, exptime


def get_combined_image(obsset):

    if obsset.recipe_name.endswith("AB"):  # do A-B
        obsset_a = obsset.get_subset("A")
        obsset_b = obsset.get_subset("B")

        a = _get_combined_image(obsset_a)
        b = _get_combined_image(obsset_b)

        a = sub_bg64_from_guard(sub_p64_from_guard(a))
        b = sub_bg64_from_guard(sub_p64_from_guard(b))

        a = a - np.nanmedian(a)
        b = b - np.nanmedian(b)

        # sky_data = .5 * abs(a-b)
        sky_data = .5 * (a+b - abs(a-b))
        combine_mode = "median_sky"
        combine_par = json.dumps(dict(A=obsset_a.obsids,
                                      B=obsset_b.obsids))

    else:
        # TODO: readout pattern need to be subtracted
        sky_data = _get_combined_image(obsset)
        combine_mode = "median"
        combine_par = json.dumps(obsset.obsids)

    ncombine = len(obsset.obsids)
    cards = [("NCOMBINE", ncombine, "number of images combined"),
             ("COMBMODE", combine_mode, "how images are combined"),
             ("COMB_PAR", combine_par,
              "parameters for image combine")
    ]

    return sky_data, cards


def _destripe_sky(data, destripe_mask, subtract_bg=True):
    """
    simple destripping. Suitable for sky.
    """

    from .destriper import destriper
    from .estimate_sky import estimate_background, get_interpolated_cubic

    if hasattr(subtract_bg, "shape"):
        d = data - subtract_bg

    elif subtract_bg:
        xc, yc, v, std = estimate_background(data, destripe_mask,
                                             di=48, min_pixel=40)
        nx = ny = 2048
        ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)
        ZI3 = np.nan_to_num(ZI3)

        d = data - ZI3
    else:
        d = data

    mask = destripe_mask | ~np.isfinite(d)
    stripes = destriper.get_stripe_pattern64(d, mask=mask,
                                             concatenate=True,
                                             remove_vertical=False)

    return d - stripes


def make_combined_image_sky_old(obsset):
    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    sky_data = get_combined_image(obsset)

    sky_data = _destripe_sky(sky_data, destripe_mask, subtract_bg=False)

    hdul = obsset.get_hdul_to_write(([], sky_data))
    obsset.store("combined_sky", data=hdul)


def get_exptime(obsset):
    if obsset.recipe_entry is not None and "exptime" in obsset.recipe_entry:
        exptime = obsset.recipe_entry["exptime"]
    else:
        exptime = float(obsset.get_hdus()[0].header["exptime"])

    return exptime


def _sky_subtract_bg_list(obsset, sky_image_list,
                          destripe_mode="guard",
                          bg_subtraction_mode=None):

    sky_exptime = get_exptime(obsset)

    if bg_subtraction_mode is None:
        bg_data_norm = 0.

    elif bg_subtraction_mode == "flat":

        bg_hdu = obsset.load_resource_sci_hdu_for(("flat_off",
                                                   DESCS["FLAT_OFF_BG"]))
        bg_exptime = float(bg_hdu.header["exptime"])
        bg_data_norm = bg_hdu.data / bg_exptime

    else:
        raise ValueError("unknown bg_subtraction_mode: {}".
                         format(bg_subtraction_mode))


    # subtract pattern noise

    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    import igrins.procedures.readout_pattern as rp
    from .destripe_helper import sub_p64_from_guard, sub_bg64_from_guard

    pipe = [
        rp.PatternP64ColWise,
        rp.PatternAmpP2,
        rp.PatternRowWiseBias
    ]

    destriped_sky_list = []

    for sky_image in sky_image_list:
        sky_image2 = sky_image - bg_data_norm * sky_exptime
        # destriped_sky = rp.apply(sky_image, pipe, mask=destripe_mask)

        if destripe_mode == "guard":
            # destriped_sky = rp.sub_guard_column(sky_image2)
            destriped_sky = sub_bg64_from_guard(sub_p64_from_guard(sky_image2))
        elif destripe_mode is not None:
            destriped_sky = rp.apply(sky_image2, pipe, mask=destripe_mask)
        else:
            destriped_sky = sky_image2

        destriped_sky_list.append(destriped_sky)

    return destriped_sky_list


def _sky_subtract_bg(obsset, sky_image,
                     bg_subtraction_mode="flat"):

    sky_exptime = get_exptime(obsset)

    if bg_subtraction_mode == "flat":

        bg_hdu = obsset.load_resource_sci_hdu_for(("flat_off",
                                                   DESCS["FLAT_OFF_BG"]))
        bg = bg_hdu.data
        bg_exptime = float(bg_hdu.header["exptime"])
    elif bg_subtraction_mode == "no":
        bg = np.zeros_like(sky_image)
        bg_exptime = 1.
    else:
        raise ValueError("unknown bg_subtraction_mode: {}".
                         format(bg_subtraction_mode))

    sky_image2 = sky_image - bg / bg_exptime * sky_exptime

    # subtract pattern noise

    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    import igrins.procedures.readout_pattern as rp

    pipe = [
        rp.PatternP64ColWise,
        rp.PatternAmpP2,
        rp.PatternRowWiseBias
    ]

    destriped_sky = rp.apply(sky_image2, pipe, mask=destripe_mask)

    return destriped_sky


def _make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
    sky_image, cards = get_combined_image(obsset)

    if bg_subtraction_mode is None:
        return sky_image, cards

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        final_sky = _sky_subtract_bg(obsset, sky_image,
                                     bg_subtraction_mode=bg_subtraction_mode)

        return final_sky, cards


def extract_spectra(obsset):
    "extract spectra"

    # caldb = helper.get_caldb()
    # master_obsid = obsids[0]

    data = obsset.load_fits_sci_hdu(DESCS["combined_sky"]).data
    data = np.nan_to_num(data)

    aperture = get_simple_aperture_from_obsset(obsset)

    specs = aperture.extract_spectra_simple(data)
    obsset.store(DESCS["oned_spec_json"],
                 data=dict(orders=aperture.orders,
                           specs=specs,
                           aperture_basename=aperture.basename))


def _get_slices(n_slice_one_direction):
    """
    given number of slices per direction, return slices for the
    center, up and down positions.
    """
    n_slice = n_slice_one_direction*2 + 1
    i_center = n_slice_one_direction
    slit_slice = np.linspace(0., 1., n_slice+1)

    slice_center = (slit_slice[i_center], slit_slice[i_center+1])

    slice_up = [(slit_slice[i_center+i], slit_slice[i_center+i+1])
                for i in range(1, n_slice_one_direction+1)]

    slice_down = [(slit_slice[i_center-i-1], slit_slice[i_center-i])
                  for i in range(n_slice_one_direction)]

    return slice_center, slice_up, slice_down


def extract_spectra_multi(obsset):

    n_slice_one_direction = 2
    slice_center, slice_up, slice_down = _get_slices(n_slice_one_direction)

    data = obsset.load_fits_sci_hdu(DESCS["combined_sky"]).data

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]

    ap = get_simple_aperture_from_obsset(obsset, orders=orders)

    def make_hdu(s_up, s_down, data):
        h = [("NSLIT", n_slice_one_direction*2 + 1),
             ("FSLIT_DN", s_down),
             ("FSLIT_UP", s_up),
             ("FSLIT_CN", 0.5 * (s_up+s_down)),
             ("NORDER", len(ap.orders)),
             ("B_ORDER", ap.orders[0]),
             ("E_ORDER", ap.orders[-1]), ]

        return (h, np.array(data))

    hdu_list = []

    s_center = ap.extract_spectra_v2(data,
                                     slice_center[0], slice_center[1])

    hdu_list.append(make_hdu(slice_center[0], slice_center[1], s_center))

    for s1, s2 in slice_up:
        s = ap.extract_spectra_v2(data, s1, s2)
        hdu_list.append(make_hdu(s1, s2, s))

    for s1, s2 in slice_down:
        s = ap.extract_spectra_v2(data, s1, s2)
        hdu_list.append(make_hdu(s1, s2, s))

    hdul = obsset.get_hdul_to_write(*hdu_list)

    obsset.store("MULTI_SPEC_FITS", hdul)
