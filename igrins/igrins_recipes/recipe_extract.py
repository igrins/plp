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


def set_basename_postfix(obsset, basename_postfix):
    # This only applies for the output name
    obsset.set_basename_postfix(basename_postfix)


def _get_do_ab_from_recipe_name(obsset):
    recipe = obsset.recipe_name
    if recipe.endswith("AB"):
        do_ab = True
    elif recipe.endswith("ONOFF"):
        do_ab = False
    else:
        msg = "recipe name is not supported: {}".format(recipe)
        raise ValueError(msg)

    return do_ab


def estimate_slit_profile_stellar(obsset,
                                  x1=800, x2=2048-800,
                                  do_ab="recipe",
                                  slit_profile_mode="1d"):

    do_ab = _get_do_ab_from_recipe_name(obsset)
    estimate_slit_profile(obsset,
                          x1=800, x2=2048-800,
                          do_ab=do_ab, slit_profile_mode=slit_profile_mode)


def estimate_slit_profile_extended(obsset,
                                   x1=800, x2=2048-800,
                                   do_ab="recipe",
                                   slit_profile_mode="uniform"):

    do_ab = _get_do_ab_from_recipe_name(obsset)
    estimate_slit_profile(obsset,
                          x1=800, x2=2048-800,
                          do_ab=do_ab,
                          slit_profile_mode=slit_profile_mode)


steps_stellar = [Step("Set baename-postfix", set_basename_postfix,
                      basename_postfix=""),
                 Step("Make Combined Images", make_combined_images,
                      allow_no_b_frame=False),
                 Step("Estimate slit profile (stellar)",
                      estimate_slit_profile_stellar,
                      slit_profile_mode="1d"),
                 # Step("Extract spectra (for extendeded)",
                 #      extract_extended_spec),
                 Step("Extract spectra (for stellar)",
                      extract_stellar_spec,
                      extraction_mode="optimal"),
                 Step("Generate Rectified 2d-spec", store_2dspec),
]


def update_db(obsset):

    obsset.add_to_db("a0v")

steps_a0v = steps_stellar + [Step("Flatten A0V", flatten_a0v),
                             Step("Update db", update_db)
]


steps_extended = [Step("Set baename-postfix", set_basename_postfix,
                       basename_postfix=None),
                  Step("Make Combined Images", make_combined_images,
                       allow_no_b_frame=False),
                  Step("Estimate slit profile (extended)",
                       estimate_slit_profile_extended,
                       slit_profile_mode="uniform"),
                  Step("Extract spectra (for extendeded)",
                       extract_extended_spec,
                       lacosmic_thresh=0.,
                       # extraction_mode="simple",
                  ),
                  # Step("Extract spectra (for stellar)",
                  #      extract_stellar_spec),
                  Step("Generate Rectified 2d-spec", store_2dspec),
]


if __name__ == "__main__":
    pass
