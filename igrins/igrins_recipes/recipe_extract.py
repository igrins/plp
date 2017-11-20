"""
"""

from __future__ import print_function

from ..procedures.target_spec import (make_combined_images,
                                      estimate_slit_profile,
                                      extract_stellar_spec,
                                      extract_extended_spec,
                                      store_2dspec)

# # from .target_spec import subtract_interorder_background
# # from .target_spec import xshift_images
# from .target_spec import estimate_slit_profile
# from .target_spec import extract_stellar_spec
# from .target_spec import extract_extended_spec
# # from .target_spec import update_slit_profile  # This needs furthe fix
# from .target_spec import store_2dspec
from ..procedures.a0v_flatten import flatten_a0v

# def update_distortion_db(obsset):

#     db = obsset.add_to_db("distortion")


# def update_wvlsol_db(obsset):

#     db = obsset.add_to_db("wvlsol")

from ..pipeline.steps import Step


steps_stellar = [Step("Make Combined Images", make_combined_images),
                 Step("Estimate slit profile", estimate_slit_profile,
                      slit_profile_mode="1d"),
                 # Step("Extract spectra (for extendeded)",
                 #      extract_extended_spec),
                 Step("Extract spectra (for stellar)",
                      extract_stellar_spec,
                      extraction_mode="optimal"),
                 Step("Generate Rectified 2d-spec", store_2dspec),
]


steps_a0v = steps_stellar + [Step("Flatten A0V", flatten_a0v),
]


steps_extended = [Step("Make Combined Images", make_combined_images),
                  Step("Estimate slit profile", estimate_slit_profile,
                       slit_profile_mode="uniform"),
                  Step("Extract spectra (for extendeded)",
                       extract_extended_spec,
                       extraction_mode="simple"),
                  # Step("Extract spectra (for stellar)",
                  #      extract_stellar_spec),
                  Step("Generate Rectified 2d-spec", store_2dspec),
]


if __name__ == "__main__":
    pass
