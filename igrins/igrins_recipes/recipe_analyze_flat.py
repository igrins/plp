import numpy as np

from ..pipeline.steps import Step

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures import analyze_flat as _analyze_flat


def analyze_flat(obsset, search_width=7):
    obsset = obsset.get_subset("ON")
    helper = ResourceHelper(obsset)
    ap = helper.get("aperture")
    orange = ap.orders

    imderiv, ol = _analyze_flat.get(obsset, ap, search_width)

    with np.errstate(invalid="ignore"):

        msk = _analyze_flat.get_response_mask(obsset)
        ps = _analyze_flat.get_peak_array(imderiv, ol, orange)
        fwhm = _analyze_flat.get_fwhm(imderiv, ol, orange, search_width, ps)
        ys = _analyze_flat.get_ys_array(imderiv, ol, orange)

    orders = " ".join([str(o) for o in orange])

    hdul = obsset.get_hdul_to_write(([("EXTNAME", "MASK"),
                                      ("ORDERS", orders)], msk),
                                    ([("EXTNAME", "PEAK-BOTTOM")],
                                     ps["bottom"]),
                                    ([("EXTNAME", "PEAK-TOP")],
                                     ps["top"]),
                                    ([("EXTNAME", "FWHM-BOTTOM")],
                                     fwhm["bottom"]),
                                    ([("EXTNAME", "FWHM-TOP")],
                                     fwhm["top"]),
                                    ([("EXTNAME", "MOM1-BOTTOM")],
                                     ys["1st_moment"]["bottom"]),
                                    ([("EXTNAME", "MOM1-TOP")],
                                     ys["1st_moment"]["bottom"]),
                                    ([("EXTNAME", "MOM2-BOTTOM")],
                                     ys["2nd_moment"]["bottom"]),
                                    ([("EXTNAME", "MOM2-TOP")],
                                     ys["2nd_moment"]["top"]),
                                   )

    obsset.store("FLAT_MOMENTS_FITS", data=hdul)


steps = [Step("Calculate Moments", analyze_flat, search_width=7),
]
