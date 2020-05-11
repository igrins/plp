import numpy as np

from ..pipeline.steps import Step
# from ..pipeline.steps import Step, ArghFactoryWithShort

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.oned_spec_helper import OnedSpecHelper

# from igrins.libs.recipe_base import filter_a0v
from ..igrins_libs.a0v_obsid import get_a0v_obsid


def set_basename_postfix(obsset):
    pass


def get_a0v_thresh_masks(a0v, threshold_a0v):
    # thresh_masks = []

    thresh_masks = np.ones(a0v.spec.shape, dtype="b")

    for t, t2, m in zip(a0v.spec,
                        a0v.flattened,
                        thresh_masks):

        msk = np.isfinite(t)
        if np.any(msk) and threshold_a0v is not None:
            t0 = np.percentile(t[msk], 95)*threshold_a0v
            m[:] = (t < t0) | (t2 < threshold_a0v)

    return thresh_masks


def get_interpolated_vega_spec(obsset, um):
    """
    um: list of wevalength array
    """
    from ..procedures.a0v_spec import A0VSpec
    d = obsset.rs.load_ref_data("VEGA_SPEC")
    a0v_model = A0VSpec(d)

    a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                               flatten=True,
                                               smooth_pixel=32)

    aa = [a0v_interp1d(wvl)[0] for wvl in zip(um)]

    return np.array(aa)


# def get_tgt_spec_cor(obsset, tgt, a0v,
#                      threshold_a0v=None):

#     thresh_masks = get_a0v_thresh_masks(a0v, threshold_a0v)

#     aa = get_vega_spec(obsset, tgt.um)

#     tgt_spec_cor = [s/t*a for s, t, a in zip(tgt.spec, a0v.spec, aa)]

#     return tgt_spec_cor, thresh_masks, aa


def _make_spec_a0v_hdu_list(obsset, wvl, spec, a0v_spec, vega,
                            a0v_fitted_continuum,
                            thresh_masks,
                            header_updates=None):

    primary_header_cards = [("EXTNAME", "SPEC_DIVIDE_A0V")]
    if header_updates is not None:
        primary_header_cards.extend(header_updates)

    _hdul = [
        (primary_header_cards, spec/a0v_spec*vega),
        ([("EXTNAME", "WAVELENGTH")], wvl),
        ([("EXTNAME", "TGT_SPEC")], spec),
        ([("EXTNAME", "A0V_SPEC")], a0v_spec),
        ([("EXTNAME", "VEGA_SPEC")], vega),
        ([("EXTNAME", "SPEC_DIVIDE_CONT")], spec/a0v_fitted_continuum*vega),
        ([("EXTNAME", "MASK")], thresh_masks.astype("i"))
    ]

    # _hdul[0].verify(option="fix")

    hdul = obsset.get_hdul_to_write(*_hdul, convention="gemini")

    return hdul


def divide_a0v(obsset,
               a0v=None,
               a0v_obsid=None,
               basename_postfix=None,
               # outname_postfix=None,
               a0v_basename_postfix="",
               threshold_a0v=0.1):

    tgt = OnedSpecHelper(obsset, basename_postfix=basename_postfix)

    a0v_obsid = get_a0v_obsid(obsset, a0v, a0v_obsid)
    if a0v_obsid is None:
        a0v_obsid_ = obsset.query_resource_basename("a0v")
        a0v_obsid = obsset.rs.parse_basename(a0v_obsid_)

    a0v_obsset = type(obsset)(obsset.rs, "A0V_AB", [a0v_obsid], ["A"],
                              basename_postfix=a0v_basename_postfix)

    a0v = OnedSpecHelper(a0v_obsset, basename_postfix=a0v_basename_postfix)

    # tgt_spec_cor, thresh_masks, aa = get_tgt_spec_cor(obsset, tgt, a0v,
    #                                                   threshold_a0v)

    vega_spec = get_interpolated_vega_spec(obsset, tgt.um)

    a0v_fitted_continuum = a0v.flattened_hdu_list["FITTED_CONTINUUM"].data

    thresh_masks = get_a0v_thresh_masks(a0v, threshold_a0v)

    hdul = _make_spec_a0v_hdu_list(obsset,
                                   tgt.um,
                                   tgt.spec, a0v.spec, vega_spec,
                                   a0v_fitted_continuum,
                                   thresh_masks,
                                   header_updates=None)

    obsset.store("SPEC_A0V_FITS", hdul, postfix=basename_postfix)

# Step("Set basename_postfix", set_basename_postfix),

steps = [Step("Divide w/ A0V", divide_a0v)]
