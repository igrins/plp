# This is to use new framework. Let's use this to measure flexure
# between emission spectra (e.g., sky, UNe, etc.)

#import os
from __future__ import print_function


from .sky_spec import make_combined_image_sky, extract_spectra_multi
from .process_identify_multiline import identify_multiline
from .process_wvlsol_volume_fit import volume_fit


from .generate_wvlsol_maps import make_ordermap_slitposmap
from .generate_wvlsol_maps import make_slitoffsetmap

from .process_derive_wvlsol import derive_wvlsol

# update_distortion_db : see below
# update_wvlsol_db : see below

from .generate_wvlsol_maps import make_wavelength_map
from .process_save_wat_header import save_wat_header


def update_distortion_db(obsset):

    db = obsset.add_to_db("distortion")


def update_wvlsol_db(obsset):

    db = obsset.add_to_db("wvlsol")

from ..driver import Step


steps = [Step("Make Combined Sky", make_combined_image_sky),
         Step("Extract spectra-multi", extract_spectra_multi),
         Step("Identify lines in multi-slit", identify_multiline),
         Step("Fit wvlsol volume", volume_fit),
         Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         Step("Make Slitoffset map", make_slitoffsetmap),
         Step("Derive wvlsol", derive_wvlsol),
         Step("Update distortion db", update_distortion_db),
         Step("Update wvlsol db", update_wvlsol_db),
         Step("Make wvlmap", make_wavelength_map),
         Step("Save WAT header", save_wat_header),
]


if __name__ == "__main__":
    pass
