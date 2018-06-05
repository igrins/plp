from ..pipeline.steps import Step

from ..procedures.sky_spec import (_make_combined_image_sky,
                                   extract_spectra)

from ..procedures.procedures_register import (identify_orders,
                                              identify_lines,
                                              find_affine_transform,
                                              transform_wavelength_solutions,
                                              save_orderflat,
                                              update_db)

def make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
    final_sky = _make_combined_image_sky(obsset, bg_subtraction_mode)

    hdul = obsset.get_hdul_to_write(([], final_sky))
    obsset.store("combined_image", data=hdul)


steps = [Step("Make Combined Sky", make_combined_image_sky),
         Step("Extract Simple 1d Spectra", extract_spectra),
         Step("Identify Orders", identify_orders),
         Step("Identify Lines", identify_lines),
         Step("Find Affine Transform", find_affine_transform),
         Step("Derive transformed Wvl. Solution",
              transform_wavelength_solutions),
         Step("Save Order-Flats, etc", save_orderflat),
         Step("Update DB", update_db),
]
