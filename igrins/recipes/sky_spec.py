import numpy as np

from .. import DESCS

from ..libs.stsci_helper import stsci_median
from ..libs.resource_helper_igrins import ResourceHelper
from ..libs.image_combine import destripe_sky


from .aperture_helper import get_simple_aperture_from_obsset


def _get_combined_image(obsset):
    data_list = [hdu.data for hdu in obsset.get_hdus()]
    return stsci_median(data_list)


def make_combined_image_sky(obsset):

    if obsset.recipe_name.endswith("AB"): # do A-B
        obsset_a = obsset.get_subset("A")
        obsset_b = obsset.get_subset("B")

        a = _get_combined_image(obsset_a)
        b = _get_combined_image(obsset_b)

        sky_data = a+b - abs(a-b)
    else:
        sky_data = _get_combined_image(obsset)

    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    sky_data = destripe_sky(sky_data, destripe_mask, subtract_bg=False)

    hdul = obsset.get_hdul_to_write(([], sky_data))
    obsset.store("combined_sky", data=hdul)


def extract_spectra(obsset):
    "extract spectra"

    # caldb = helper.get_caldb()
    # master_obsid = obsids[0]

    data = obsset.load_fits_sci_hdu(DESCS["combined_sky"]).data

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
