# reference
# aperture
# wavelength solution

# target
# aperture
# identified lines

import numpy as np

from ..libs.recipe_helper import RecipeHelper

from .aperture_helper import get_simple_aperture_from_obsset

from .. import DESCS

def find_affine_transform(obsset):

    # As register.db has not been written yet, we cannot use
    # obsset.get("orders")
    orders = obsset.load(DESCS["ORDERS_JSON"])["orders"]

    ap = get_simple_aperture_from_obsset(obsset, orders)

    lines_data = obsset.load(DESCS["IDENTIFIED_LINES_JSON"])

    from ..libs.identified_lines import IdentifiedLines
    identified_lines_tgt = IdentifiedLines.load(lines_data)

    xy_list_tgt = identified_lines_tgt.get_xy_list_from_pixlist(ap)

    from ..libs.echellogram import Echellogram

    from ..libs.master_calib import load_ref_data
    echellogram_data = obsset.load_ref_data(kind="ECHELLOGRAM_JSON")

    echellogram = Echellogram.from_dict(echellogram_data)

    xy_list_ref = identified_lines_tgt.get_xy_list_from_wvllist(echellogram)

    assert len(xy_list_tgt) == len(xy_list_ref)

    from ..libs.align_echellogram_thar import fit_affine_clip
    affine_tr, mm = fit_affine_clip(np.array(xy_list_ref),
                                    np.array(xy_list_tgt))

    d = dict(xy1f=xy_list_ref, xy2f=xy_list_tgt,
             affine_tr_matrix=affine_tr.get_matrix(),
             affine_tr_mask=mm)

    obsset.store(DESCS["ALIGNING_MATRIX_JSON"],
                 data=d)


def main(utdate, band, obsids, config_name):
    helper = RecipeHelper(config_name, utdate)
    find_affine_transform(helper, band, obsids)

if __name__ == "__main__":
    utdate = "20150525"
    band = "H"
    obsids = [52]
    master_obsid = obsids[0]

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    main(utdate, band, obsids, config_name)
