"""
"""

from __future__ import print_function

from .target_spec import make_combined_images
# from .target_spec import subtract_interorder_background
# from .target_spec import xshift_images
from .target_spec import estimate_slit_profile
from .target_spec import extract_stellar_spec
from .target_spec import extract_extended_spec
# from .target_spec import update_slit_profile  # This needs furthe fix
from .target_spec import store_2dspec
from .a0v_flatten import flatten_a0v

# def update_distortion_db(obsset):

#     db = obsset.add_to_db("distortion")


# def update_wvlsol_db(obsset):

#     db = obsset.add_to_db("wvlsol")

from ..driver import Step


steps = [Step("Make Combined Images", make_combined_images),
         # Step("Subtract Interorder Background",
         # subtract_interorder_background),
         Step("Estimate slit profile", estimate_slit_profile,
              mode="1d"),
         # Step("X-shift images", xshift_images),
         Step("Extract spectra (for stellar)", extract_stellar_spec,
              extraction_mode="simple"),
         Step("Generate Rectified 2d-spec", store_2dspec),
         Step("Flatten A0V", flatten_a0v),
         # Step("Identify lines in multi-slit", identify_multiline),
         # Step("Fit wvlsol volume", volume_fit),
         # Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         # Step("Make Slitoffset map", make_slitoffsetmap),
         # Step("Derive wvlsol", derive_wvlsol),
         # Step("Update distortion db", update_distortion_db),
         # Step("Update wvlsol db", update_wvlsol_db),
         # Step("Make wvlmap", make_wavelength_map),
         # Step("Save WAT header", save_wat_header),
]

# extended
steps2 = [Step("Make Combined Images", make_combined_images),
         Step("Estimate slit profile", estimate_slit_profile,
              mode="uniform"),
         Step("Extract spectra (for extendeded)", extract_extended_spec),
         Step("Generate Rectified 2d-spec", store_2dspec),
         # Step("Update slit profile", update_slit_profile),
         # Step("Identify lines in multi-slit", identify_multiline),
         # Step("Fit wvlsol volume", volume_fit),
         # Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         # Step("Make Slitoffset map", make_slitoffsetmap),
         # Step("Derive wvlsol", derive_wvlsol),
         # Step("Update distortion db", update_distortion_db),
         # Step("Update wvlsol db", update_wvlsol_db),
         # Step("Make wvlmap", make_wavelength_map),
         # Step("Save WAT header", save_wat_header),
]


if __name__ == "__main__":
    pass
