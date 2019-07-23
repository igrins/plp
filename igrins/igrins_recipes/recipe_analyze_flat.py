from ..pipeline.steps import Step

from ..igrins_libs.resource_helper_igrins import ResourceHelper

from ..procedures import analyze_flat as _analyze_flat


def analyze_flat(obsset, search_width=7):
    obsset = obsset.get_subset("ON")
    helper = ResourceHelper(obsset)
    ap = helper.get("aperture")
    orange = ap.orders

    imderiv, ol = _analyze_flat.get(obsset, ap, search_width)

    ys = _analyze_flat.get_ys_array(imderiv, ol, orange)
    ps = _analyze_flat.get_peak_array(imderiv, ol, orange)

    msk = _analyze_flat.get_response_mask(obsset)
    orders = " ".join([str(o) for o in orange])

    hdul = obsset.get_hdul_to_write(([("EXTNAME", "MASK"),
                                      ("ORDERS", orders)], msk),
                                    ([("EXTNAME", "PEAK-BOTTOM")],
                                     ps["bottom"]),
                                    ([("EXTNAME", "PEAK-TOP")],
                                     ps["top"]),
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
