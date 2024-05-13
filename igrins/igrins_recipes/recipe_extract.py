"""
"""

from __future__ import print_function

from ..procedures.target_spec import (setup_extraction_parameters,
                                      make_combined_images,
                                      estimate_slit_profile,
                                      extract_stellar_spec,
                                      extract_stellar_spec_pp,
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


def _convert_frac_slit(s: str | None) -> list[tuple]:
    frac_slit_list = []
    if s is None:
        return frac_slit_list

    ss = s.split(",")
    if len(ss) == 2 and all(":" not in s1 for s1 in ss):
        frac_slit = (float(ss[0]), float(ss[1]))
        frac_slit_list.append(frac_slit)
    else:
        for s1 in ss:
            frac_slit = tuple(map(float, s1.split(":")))
            frac_slit_list.append(frac_slit)

    return frac_slit_list

def test_convert_frac_slit():

    assert _convert_frac_slit("0.1,0.9") == [(0.1, 0.9)]
    assert _convert_frac_slit("0.1:0.9") == [(0.1, 0.9)]
    assert _convert_frac_slit("0.1:0.3,0.5:0.8") == [(0.1, 0.3), (0.5, 0.8)]


def estimate_slit_profile_stellar(obsset,
                                  x1=800, x2=2048-800,
                                  slit_profile_mode="1d",
                                  frac_slit=None):

    command_name = obsset.runner_config["command_name"]
    if "onoff" in command_name:
        do_ab = False
    else:
        do_ab = True

    frac_slit_list = _convert_frac_slit(frac_slit)

    estimate_slit_profile(obsset,
                          x1=x1, x2=x2,
                          do_ab=do_ab, slit_profile_mode=slit_profile_mode,
                          frac_slit_list=frac_slit_list)


def estimate_slit_profile_extended(obsset,
                                   x1=800, x2=2048-800,
                                   frac_slit=None):

    command_name = obsset.runner_config["command_name"]
    if "onoff" in command_name:
        do_ab = False
    else:
        do_ab = True

    estimate_slit_profile(obsset,
                          x1=800, x2=2048-800,
                          do_ab=do_ab,
                          slit_profile_mode="uniform",
                          frac_slit=frac_slit)


_steps_default = [
    Step("Setup extraction parameters",
         setup_extraction_parameters,
         height_2dspec=0,
         order_range="",
         correct_flexure=False,
         mask_cosmics=False,
         ),
    Step("Set basename-postfix", set_basename_postfix,
         basename_postfix=""),
]


_steps_stellar = [
    Step("Make Combined Images", make_combined_images,
         force_image_combine=False,
         pattern_remove_level="auto"),
    Step("Estimate slit profile (stellar)",
         estimate_slit_profile_stellar,
         slit_profile_mode="1d",
         frac_slit=None), # frac_slit can be "0.3,0.9" or "0.1:0.4,0.6:0.9"
    # Step("Extract spectra (for extendeded)",
    #      extract_extended_spec),
    Step("Extract spectra (for stellar)",
         extract_stellar_spec,
         extraction_mode="optimal",
         pixel_per_res_element=None,
         ),
    Step("Generate Rectified 2d-spec", store_2dspec),
]

steps_stellar = _steps_default + _steps_stellar

_steps_stellar_pp = [
    Step("Extract spectra (PP for stellar)",
         extract_stellar_spec_pp,
         extraction_mode="optimal"),
]


steps_stellar_pp = _steps_default + _steps_stellar_pp


def update_db(obsset):

    obsset.add_to_db("a0v")


steps_a0v = steps_stellar + [Step("Flatten A0V", flatten_a0v),
                             Step("Update db", update_db)
]


_steps_extended = [
    Step("Make Combined Images", make_combined_images,
         force_image_combine=False,
         pattern_remove_level="auto",
         allow_no_b_frame=True),
    Step("Estimate slit profile (extended)", estimate_slit_profile_extended,
         frac_slit=None),
    Step("Extract spectra (for extendeded)",
         extract_extended_spec,
         pixel_per_res_element=None,
         # extraction_mode="simple",
    ),
    # Step("Extract spectra (for stellar)",
    #      extract_stellar_spec),
    Step("Generate Rectified 2d-spec", store_2dspec)
]


steps_extended = _steps_default + _steps_extended

# _steps_extended_dry = [
#     Step("Make Combined Images", make_combined_images,
#          force_image_combine=False,
#          pattern_remove_level=0,
#          allow_no_b_frame=False),
#     Step("Estimate slit profile (extended)", estimate_slit_profile,
#          slit_profile_mode="uniform"),
#     Step("Extract spectra (for extendeded)",
#          extract_extended_spec,
#          calculate_sn=False,
#          # extraction_mode="simple",
#     ),
#     Step("Generate Rectified 2d-spec", store_2dspec)
# ]

# steps_extended_dry = _steps_default + _steps_extended_dry

if __name__ == "__main__":
    pass
