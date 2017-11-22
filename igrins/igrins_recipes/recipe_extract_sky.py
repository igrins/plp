from ..pipeline.steps import Step

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures.sky_spec import _get_combined_image, _destripe_sky
from ..procedures.target_spec import get_variance_map

from ..procedures.target_spec import (extract_extended_spec,
                                      store_2dspec)


def make_combined_image_sky(obsset):
    # print("RECIPENAME", obsset.recipe_name)
    do_ab = obsset.recipe_name.endswith("AB")

    if do_ab:
        obsset_a = obsset.get_subset("A")
        obsset_b = obsset.get_subset("B")

        a = _get_combined_image(obsset_a)
        b = _get_combined_image(obsset_b)

        sky_data = a + b - abs(a - b)
        sky_plus = 2 * (a + b)  # assume that variance is the twice of a + b
    else:
        sky_data = _get_combined_image(obsset)
        sky_plus = sky_data

    helper = ResourceHelper(obsset)
    destripe_mask = helper.get("destripe_mask")

    sky_data = _destripe_sky(sky_data, destripe_mask, subtract_bg=False)

    variance_map0, variance_map = get_variance_map(obsset,
                                                   sky_data, sky_plus)

    hdul = obsset.get_hdul_to_write(([], sky_data),
                                    ([], variance_map0),
                                    ([], variance_map))
    obsset.store("combined_image1", data=hdul)


def estimate_slit_profile(obsset):
    from ..procedures.slit_profile import estimate_slit_profile_uniform
    estimate_slit_profile_uniform(obsset, do_ab=False)


def set_basename_postfix(obsset):
    obsset.set_basename_postfix(basename_postfix="_sky")


steps = [Step("Set basename_postfix", set_basename_postfix),
         Step("Make Combined Sky", make_combined_image_sky),
         Step("Estimate slit profile", estimate_slit_profile),
         Step("Extract spectra (for extendeded)",
              extract_extended_spec),
         Step("Generate Rectified 2d-spec", store_2dspec),
]

