import pandas as pd

from ..pipeline.steps import Step

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures.sky_spec import _get_combined_image, _destripe_sky
from ..procedures.target_spec import get_variance_map

from ..procedures.target_spec import extract_extended_spec
from ..procedures.ref_lines_db import SkyLinesDB, HitranSkyLinesDB

from ..procedures.process_identify_multiline import identify_lines_from_spec

from .recipe_extract_sky import make_combined_image_sky as make_combined_image_arc


def estimate_slit_profile(obsset):
    from ..procedures.slit_profile import estimate_slit_profile_uniform
    estimate_slit_profile_uniform(obsset, do_ab=False)


def set_basename_postfix(obsset, basename_postfix="_arc"):
    # This only applies for the output name
    obsset.set_basename_postfix(basename_postfix=basename_postfix)



steps = [Step("Set basename_postfix", set_basename_postfix,
              basename_postfix="_arc"),
         Step("Make Combined Image", make_combined_image_arc),
         Step("Estimate slit profile (uniform)", estimate_slit_profile),
         Step("Extract spectra (for extendeded)",
              extract_extended_spec),
         # Step("Generate Rectified 2d-spec", store_2dspec),
]
