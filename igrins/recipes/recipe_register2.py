from __future__ import print_function

from collections import namedtuple

import numpy as np
import scipy.ndimage as ni

from astropy.io.fits import Card

from .. import DESCS
from ..libs.resource_helper_igrins import ResourceHelper
from ..libs.load_fits import get_first_science_hdu

from .sky_spec import make_combined_image_sky, extract_spectra


def _get_ref_spec_name(recipe_name):

    if recipe_name is None:
        recipe_name = self.recipe_name

    if (recipe_name in ["SKY"]) or recipe_name.endswith("_AB"):
        ref_spec_key = "SKY_REFSPEC_JSON"
        ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"

    elif recipe_name in ["THAR"]:
        ref_spec_key = "THAR_REFSPEC_JSON"
        ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"

    else:
        raise ValueError("Recipe name of '%s' is unsupported."
                         % recipe_name)

    return ref_spec_key, ref_identified_lines_key


def identify_orders(obsset):

    ref_spec_key, _ = _get_ref_spec_name(obsset.recipe_name)
    # from igrins.libs.master_calib import load_ref_data
    #ref_spectra = load_ref_data(helper.config, band,

    ref_spec_path, ref_spectra = obsset.rs.load_ref_data(ref_spec_key,
                                                         get_path=True)

    src_spectra = obsset.load(DESCS["ONED_SPEC_JSON"])

    from ..libs.process_thar import match_order
    new_orders = match_order(src_spectra, ref_spectra)

    print(new_orders)

    src_spectra["orders"] = new_orders
    obsset.store(DESCS["ONED_SPEC_JSON"],
                 data=src_spectra)

    aperture_basename = src_spectra["aperture_basename"]
    obsset.store(DESCS["ORDERS_JSON"],
                      data=dict(orders=new_orders,
                                aperture_basename=aperture_basename,
                                ref_spec_path=ref_spec_path))


def identify_lines(obsset):

    ref_spec_key, ref_identified_lines_key = _get_ref_spec_name(obsset.recipe_name)

    ref_spec = obsset.rs.load_ref_data(ref_spec_key)

    tgt_spec = obsset.load(DESCS["ONED_SPEC_JSON"])
    # tgt_spec_path = obsset.query_item_path("ONED_SPEC_JSON")
    #tgt_spec = obsset.load_item("ONED_SPEC_JSON")

    from ..libs.process_thar import get_offset_treanform_between_2spec
    intersected_orders, d = get_offset_treanform_between_2spec(ref_spec,
                                                               tgt_spec)


    #REF_TYPE="OH"
    #fn = "../%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band,
    #                                             helper.refdate)
    l = obsset.rs.load_ref_data(ref_identified_lines_key)
    #l = json.load(open(fn))
    #ref_spectra = load_ref_data(helper.config, band, kind="SKY_REFSPEC_JSON")

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    from ..libs.identified_lines import IdentifiedLines

    identified_lines_ref = IdentifiedLines(l)
    ref_map = identified_lines_ref.get_dict()

    identified_lines_tgt = IdentifiedLines(l)
    identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                     pixpos_list=[], orders=[],
                                     groupname=obsset.groupname))

    from ..libs.line_identify_simple import match_lines1_pix

    for o, s in zip(tgt_spec["orders"], tgt_spec["specs"]):
        if (o not in ref_map) or (o not in offsetfunc_map):
            wvl, indices, pixpos = [], [], []
        else:
            pixpos, indices, wvl = ref_map[o]
            pixpos = np.array(pixpos)
            msk = (pixpos >= 0)

            ref_pix_list = offsetfunc_map[o](pixpos[msk])
            pix_list, dist = match_lines1_pix(np.array(s), ref_pix_list)

            pix_list[dist > 1] = -1
            pixpos[msk] = pix_list

        identified_lines_tgt.append_order_info(o, wvl, indices, pixpos)

    #REF_TYPE = "OH"
    #fn = "%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band, helper.utdate)
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    obsset.store(DESCS["IDENTIFIED_LINES_JSON"],
                 identified_lines_tgt.data)



# def update_db(obsset):

#     obsset_off = obsset.get_subset("OFF")
#     obsset_on = obsset.get_subset("ON")

#     obsset_off.add_to_db("flat_off")
#     obsset_on.add_to_db("flat_on")


###

# def process_band(utdate, recipe_name, band,
#                  groupname,
#                  obsids, frametypes, aux_infos,
#                  config_name, **kwargs):

#     if recipe_name.upper() != "SKY_AB":
#         if recipe_name.upper().endswith("_AB") and not kwargs.pop("do_ab"):
#             logger.info("ignoring {}:{}".format(recipe_name, groupname))
#             return

#     from .. import get_caldb, get_obsset
#     caldb = get_caldb(config_name, utdate)
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)

#     # STEP 1 :
#     # make combined image

#     if recipe_name.upper() in ["SKY"]:
#         pass
#     elif recipe_name.upper().endswith("_AB"):
#         pass
#     elif recipe_name.upper() in ["THAR"]:
#         pass
#     else:
#         msg = ("recipe_name {} not supported "
#                "for this recipe").format(recipe_name)
#         raise ValueError(msg)

#     if recipe_name.upper() in ["THAR"]:
#         make_combined_image_thar(obsset)
#     else:
#         make_combined_image_sky(obsset)

#     # Step 2

#     # load simple-aperture (no order info; depends on

#     extract_spectra(obsset)

#     ## aperture trace from Flat)

#     ## extract 1-d spectra from ThAr

#     # Step 3:
#     ## compare to reference ThAr data to figure out orders of each strip
#     ##  -  simple correlation w/ clipping

#     identify_orders(obsset)

#     # Step 4:
#     ##  - For each strip, measure x-displacement from the reference
#     ##    spec. Fit the displacement as a function of orders.
#     ##  - Using the estimated displacement, identify lines from the spectra.
#     identify_lines(obsset)

#     # Step 6:

#     ## load the reference echellogram, and find the transform using
#     ## the identified lines.

#     from .find_affine_transform import find_affine_transform
#     find_affine_transform(obsset)

#     from ..libs.transform_wvlsol import transform_wavelength_solutions
#     transform_wavelength_solutions(obsset)

#     # Step 8:

#     ## make order_map and auxilary files.

#     save_orderflat(obsset)

#     # save figures

#     save_figures(obsset)

#     save_db(obsset)

from .find_affine_transform import find_affine_transform

from ..libs.transform_wvlsol import transform_wavelength_solutions

def save_orderflat(obsset):

    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    from .aperture_helper import get_simple_aperture_from_obsset

    ap = get_simple_aperture_from_obsset(obsset, orders=orders)

    order_map = ap.make_order_map()

    hdul = obsset.load_resource_for("flat_normed")
    flat_normed = get_first_science_hdu(hdul).data

    flat_mask = obsset.load_resource_for("flat_mask")

    from ..libs.process_flat import make_order_flat
    order_flat_im, order_flat_json = make_order_flat(flat_normed,
                                                     flat_mask,
                                                     orders, order_map)


    hdul = obsset.get_hdul_to_write(([], order_flat_im))
    obsset.store(DESCS["order_flat_im"], hdul)

    obsset.store(DESCS["order_flat_json"], order_flat_json)


    order_map2 = ap.make_order_map(mask_top_bottom=True)
    bias_mask = flat_mask & (order_map2 > 0)

    obsset.store(DESCS["bias_mask"], bias_mask)


def update_db(obsset):

    # save db
    db = obsset.add_to_db("register")


from ..pipeline.steps import Step


steps = [Step("Make Combined Sky", make_combined_image_sky),
         Step("Extract Simple 1d Spectra", extract_spectra),
         Step("Identify Orders", identify_orders),
         Step("Identify Lines", identify_lines),
         Step("Find Affine Transform", find_affine_transform),
         Step("Derive transformed Wvl. Solution", transform_wavelength_solutions),
         Step("Save Order-Flats, etc", save_orderflat),
         Step("Update DB", update_db),
]


if __name__ == "__main__":
    pass
