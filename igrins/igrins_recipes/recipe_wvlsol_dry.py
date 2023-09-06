from __future__ import print_function

# from ..procedures.sky_spec import (_make_combined_image_sky,
#                                    extract_spectra_multi)
# from ..procedures.process_identify_multiline import identify_multiline
# from ..procedures.process_wvlsol_volume_fit import volume_fit


from ..procedures.generate_wvlsol_maps import (make_ordermap_slitposmap,
                                               make_slitoffsetmap,
                                               make_wavelength_map)

# from ..procedures.process_derive_wvlsol import derive_wvlsol

# # update_distortion_db : see below
# # update_wvlsol_db : see below

from ..procedures.process_save_wat_header import save_wat_header

from ..pipeline.steps import Step
from .recipe_wvlsol import update_distortion_db, update_wvlsol_db
from ..igrins_libs.resource_helper_igrins import ResourceHelper

def make_fake_wvlsol(obsset):
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    wvl_sol = [list(range(o*2048, (o+1)*2048)) for o in orders]

    d = dict(orders=orders,
             wvl_sol=wvl_sol)

    obsset.store("SKY_WVLSOL_JSON", d)

import numpy as np

def make_fake_slitoffsetmap(obsset):

    ordermap_hdu = obsset.load_fits_sci_hdu("ordermap_fits")
    shape = ordermap_hdu.data.shape

    offset_map = np.zeros(shape=shape)
    hdul = obsset.get_hdul_to_write(([], offset_map))
    obsset.store("slitoffset_fits", hdul)

def make_fake_wavelength_map(obsset):

    order_map = obsset.load_fits_sci_hdu("ordermap_fits").data
    msk = order_map > 0

    _, pixels = np.indices(msk.shape)
    orders = order_map[msk]
    wvl = orders * 2048 + pixels[msk]

    wvlmap = np.empty(msk.shape, dtype=float)
    wvlmap.fill(np.nan)

    wvlmap[msk] = wvl

    hdul = obsset.get_hdul_to_write(([], wvlmap))
    obsset.store("WAVELENGTHMAP_FITS", hdul)

def save_fake_wat_header(obsset):

    d = obsset.load("SKY_WVLSOL_JSON")

    wvl_sol = d["wvl_sol"]

    hdul = obsset.get_hdul_to_write(([], np.array(wvl_sol)))
    obsset.store("SKY_WVLSOL_FITS", hdul)

steps = [Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         Step("Make fake Slitoffset map", make_fake_slitoffsetmap),
         Step("Update distortion db", update_distortion_db),
         Step("Make fake wvlsol", make_fake_wvlsol),
         Step("Update wvlsol db", update_wvlsol_db),
         Step("Make fake wvlmap", make_fake_wavelength_map),
         Step("Save fake WAT header", save_fake_wat_header),
]

