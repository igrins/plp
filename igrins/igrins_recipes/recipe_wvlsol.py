from __future__ import print_function

from ..procedures.sky_spec import (_make_combined_image_sky,
                                   extract_spectra_multi)
from ..procedures.process_identify_multiline import identify_multiline
from ..procedures.process_wvlsol_volume_fit import volume_fit


from ..procedures.generate_wvlsol_maps import (make_ordermap_slitposmap,
                                               make_slitoffsetmap,
                                               make_wavelength_map)

from ..procedures.process_derive_wvlsol import derive_wvlsol

# update_distortion_db : see below
# update_wvlsol_db : see below

from ..procedures.process_save_wat_header import save_wat_header

from ..pipeline.steps import Step


def make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
    final_sky = _make_combined_image_sky(obsset, bg_subtraction_mode)

    hdul = obsset.get_hdul_to_write(([], final_sky))
    obsset.store("combined_image", data=hdul)


def update_distortion_db(obsset):

    obsset.add_to_db("distortion")


def update_wvlsol_db(obsset):

    obsset.add_to_db("wvlsol")


steps = [Step("Make Combined Sky", make_combined_image_sky),
         Step("Extract spectra-multi", extract_spectra_multi),
         Step("Identify lines in multi-slit", identify_multiline),
         Step("Derive Distortion Solution", volume_fit),
         Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         Step("Make Slitoffset map", make_slitoffsetmap),
         Step("Update distortion db", update_distortion_db),
         Step("Derive wvlsol", derive_wvlsol),
         Step("Update wvlsol db", update_wvlsol_db),
         Step("Make wvlmap", make_wavelength_map),
         Step("Save WAT header", save_wat_header),
]


if __name__ == "__main__":
    pass
