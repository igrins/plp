from __future__ import print_function

from ..pipeline.steps import Step
from .. import DESCS
import os
from ..procedures import wvlsol_tell as wvlsol_tell

def do_wvlsol_tell(obsset):
    from igrins import get_obsset
    # obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))
    obsset = get_obsset("20220303", "H", "A0V", obsids=range(93, 105))

    # spec = obsset.load(DESCS["SPEC_FITS_FLATTENED"])
    src_filename = obsset.locate(DESCS["SPEC_FITS_FLATTENED"])

    # f = obsset.rs.load_ref_data(kind="TELL_WVLSOL_MODEL")
    item_desc = DESCS["SPEC_FITS_WAVELENGTH"]
    basename = obsset.groupname
    postfix = ""
    # , DESCS[]

    section, fnout = obsset.rs.get_section_n_fn(basename, item_desc, postfix)

    tell_file = obsset.rs.query_ref_data_path(kind="TELL_WVLSOL_MODEL")
    figout_dir = None

    blaze_corrected = True

    out_filename = os.path.join(os.path.dirname(src_filename), fnout)

    wvlsol_tell.run(src_filename, out_filename,
        plot_dir=figout_dir, tell_file=tell_file,
        blaze_corrected=blaze_corrected)


# def process_band(utdate, recipe_name, band, 
#                  groupname, obsids, config,
#                  interactive=True):

#     # utdate, recipe_name, band, obsids, config = "20150525", "A0V", "H", [63, 64], "recipe.config"

#     from igrins.libs.recipe_helper import RecipeHelper
#     helper = RecipeHelper(config, utdate, recipe_name)
#     caldb = helper.get_caldb()

#     master_obsid = obsids[0]
#     desc = "SPEC_FITS_FLATTENED"
#     blaze_corrected=True
#     src_filename = caldb.query_item_path((band, groupname),
#                                          desc)

#     if not os.path.exists(src_filename):
#         desc = "SPEC_FITS"
#         blaze_corrected=False
#         src_filename = caldb.query_item_path((band, groupname),
#                                              desc)

#     out_filename = caldb.query_item_path((band, groupname),
#                                          "SPEC_FITS_WAVELENGTH")

#     from igrins.libs.master_calib import get_ref_data_path
#     tell_file = get_ref_data_path(helper.config, band,
#                                   kind="TELL_WVLSOL_MODEL")

#     if not interactive:
#         tgt_basename = helper.get_basename(band, groupname)
#         figout_dir = helper._igr_path.get_section_filename_base("QA_PATH",
#                                                                "",
#                                                                "tell_wvsol_"+tgt_basename)
#         from igrins.libs.path_info import ensure_dir
#         ensure_dir(figout_dir)
#     else:
#         figout_dir = None

#     #print src_filename, out_filename, figout_dir, tell_file
#     run(src_filename, out_filename,
#         plot_dir=figout_dir, tell_file=tell_file,
#         blaze_corrected=blaze_corrected)

#     #             process_band(utdate, recipe_name, band, 
#     #                          groupname, obsids, self.config,
#     #                          interactive)

# def wvlsol_tell(utdate, refdate=None, bands="HK",
#                 starting_obsids=None, 
#                 groups=None,
#                 interactive=False,
#                 recipe_name = "A0V*",
#                 config_file="recipe.config",
#                 ):

#     recipe = RecipeTellWvlsol(interactive=interactive)
#     recipe.set_recipe_name(recipe_name)
#     recipe.process(utdate, bands,
#                    starting_obsids, groups, 
#                    config_file=config_file)

steps = [Step("Refine wvlsol w/ tellurics", do_wvlsol_tell),
]


if __name__ == "__main__":
    pass
