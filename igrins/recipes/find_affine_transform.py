# reference
# aperture
# wavelength solution

# target
# aperture
# identified lines

import numpy as np

from igrins.libs.recipe_helper import RecipeHelper

from aperture_helper import get_simple_aperture


#def find_affine_transform(utdate, band, obsids, config_name):
def find_affine_transform(helper, band, obsids):

    master_obsid = obsids[0]

    caldb = helper.get_caldb()
    orders = caldb.load_resource_for((band, master_obsid), "orders")["orders"]

    ap = get_simple_aperture(helper, band, obsids, orders=orders)

    item_path = caldb.query_item_path((band, master_obsid),
                                      "IDENTIFIED_LINES_JSON")

    from igrins.libs.identified_lines import IdentifiedLines
    identified_lines_tgt = IdentifiedLines.load(item_path)

    xy_list_tgt = identified_lines_tgt.get_xy_list_from_pixlist(ap)

    from igrins.libs.echellogram import Echellogram


    from igrins.libs.master_calib import load_ref_data
    echellogram_data = load_ref_data(helper.config, band,
                                     kind="ECHELLOGRAM_JSON")

    echellogram = Echellogram.from_dict(echellogram_data)

    xy_list_ref = identified_lines_tgt.get_xy_list_from_wvllist(echellogram)

    assert len(xy_list_tgt) == len(xy_list_ref)

    from igrins.libs.align_echellogram_thar import fit_affine_clip
    affine_tr, mm = fit_affine_clip(np.array(xy_list_ref),
                                    np.array(xy_list_tgt))

    from igrins.libs.products import PipelineDict
    d = PipelineDict(xy1f=xy_list_ref, xy2f=xy_list_tgt,
                     affine_tr_matrix=affine_tr.get_matrix(),
                     affine_tr_mask=mm)

    caldb.store_dict((band, master_obsid),
                     item_type="ALIGNING_MATRIX_JSON",
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
