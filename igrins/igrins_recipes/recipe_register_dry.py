from ..pipeline.steps import Step

from ..procedures.sky_spec import extract_spectra

from ..procedures.procedures_register import (identify_orders,
                                              identify_lines,
                                              find_affine_transform,
                                              transform_wavelength_solutions,
                                              save_orderflat,
                                              update_db)

# from .recipe_register import make_combined_image_sky
from .. import DESCS
from ..procedures.aperture_helper import get_simple_aperture_from_obsset

def make_fake_order(obsset):
    # aperture_basename = src_spectra["aperture_basename"]
    aperture = get_simple_aperture_from_obsset(obsset)

    obsset.store(DESCS["ORDERS_JSON"],
                 data=dict(orders=aperture.orders))

                           # aperture_basename=aperture_basename,
                           # ref_spec_path=ref_spec_path))

from ..igrins_libs.resource_helper_igrins import ResourceHelper

def make_fake_wvlsol(obsset):
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    wvl_sol = [list(range(o*2048, (o+1)*2048)) for o in orders]

    obsset.store(DESCS["WVLSOL_V0_JSON"],
                 data=dict(orders=orders, wvl_sol=wvl_sol))


steps = [Step("Setup fake orders", make_fake_order),
         Step("Mkae fake wvlsol", make_fake_wvlsol),
         Step("Save Order-Flats, etc", save_orderflat),
         Step("Update DB", update_db),
]
