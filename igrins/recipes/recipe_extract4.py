
#import os
from __future__ import print_function


from .target_spec import make_combined_images
from .target_spec import subtract_interorder_background
from .target_spec import estimate_slit_profile
from .target_spec import extract_stellar_spec
from .target_spec import update_slit_profile  # This needs furthe fix

# def update_distortion_db(obsset):

#     db = obsset.add_to_db("distortion")


# def update_wvlsol_db(obsset):

#     db = obsset.add_to_db("wvlsol")


from ..driver import Step


steps = [Step("Make Combined Images", make_combined_images),
         # Step("Subtract Interorder Background", subtract_interorder_background),
         Step("Estimate slit profile", estimate_slit_profile, mode="1d"),
         Step("Extract spectra", extract_stellar_spec),
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
