import numpy as np
import pandas as pd


from .aperture_helper import get_simple_aperture_from_obsset

from ..libs.astropy_poly_helper import deserialize_poly_model

from .nd_poly import NdPolyNamed


def make_ordermap_slitposmap(obsset):

    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]

    ap = get_simple_aperture_from_obsset(obsset,
                                         orders=orders)

    order_map = ap.make_order_map()
    slitpos_map = ap.make_slitpos_map()
    order_map2 = ap.make_order_map(mask_top_bottom=True)

    hdul = obsset.get_hdul_to_write(([], order_map))
    obsset.store("ordermap_fits", hdul)
    hdul = obsset.get_hdul_to_write(([], slitpos_map))
    obsset.store("slitposmap_fits", hdul)
    hdul = obsset.get_hdul_to_write(([], order_map2))
    obsset.store("ordermap_masked_fits", hdul)



def make_slitoffsetmap(obsset):

    ordermap_hdu = obsset.load_fits_sci_hdu("ordermap_fits")

    slitposmap_hdu = obsset.load_fits_sci_hdu("slitposmap_fits")

    yy, xx = np.indices(ordermap_hdu.data.shape)

    msk = np.isfinite(ordermap_hdu.data) & (ordermap_hdu.data > 0)
    pixels, orders, slit_pos = (xx[msk], ordermap_hdu.data[msk],
                                slitposmap_hdu.data[msk])

    names = ["pixel", "order", "slit"]

    d = obsset.load("VOLUMEFIT_COEFFS_JSON")
    in_df = pd.DataFrame(**d)

    # pixel, order, slit : saved as float, needt obe int.
    for n in names:
        in_df[n] = in_df[n].astype("i")

    in_df = in_df.set_index(names)
    poly, coeffs = NdPolyNamed.from_pandas(in_df)

    cc0 = slit_pos - 0.5
    values = dict(zip(names, [pixels, orders, cc0]))
    offsets = poly.multiply(values, coeffs) # * cc0

    offset_map = np.empty(ordermap_hdu.data.shape, dtype=np.float64)
    offset_map.fill(np.nan)
    offset_map[msk] = offsets * cc0 # dd["offsets"]

    hdul = obsset.get_hdul_to_write(([], offset_map))
    obsset.store("slitoffset_fits", hdul)


def make_wavelength_map(obsset):

    fit_results = obsset.load("SKY_WVLSOL_FIT_RESULT_JSON")

    module_name, klass_name, serialized = fit_results["fitted_model"]
    poly_2d = deserialize_poly_model(module_name, klass_name, serialized)

    order_map = obsset.load_fits_sci_hdu("ordermap_fits").data
    # slitpos_map = caldb.load_item_from(basename, "slitposmap_fits")

    offset_map = obsset.load_fits_sci_hdu("slitoffset_fits").data

    msk = order_map > 0

    _, pixels = np.indices(msk.shape)
    orders = order_map[msk]
    wvl = poly_2d(pixels[msk] - offset_map[msk], orders) / orders

    wvlmap = np.empty(msk.shape, dtype=float)
    wvlmap.fill(np.nan)

    wvlmap[msk] = wvl

    hdul = obsset.get_hdul_to_write(([], wvlmap))
    obsset.store("WAVELENGTHMAP_FITS", hdul)
