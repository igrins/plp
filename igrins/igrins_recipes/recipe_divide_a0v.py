import numpy as np

from ..pipeline.steps import Step, ArghFactoryWithShort

# from ..pipeline.steps import Step, ArghFactoryWithShort

# from ..procedures.sky_spec import make_combined_image_sky

# from ..utils.image_combine import image_median

from ..igrins_libs.oned_spec_helper import OnedSpecHelper

# from igrins.libs.recipe_base import filter_a0v
from ..igrins_libs.a0v_obsid import get_a0v_obsid


def set_basename_postfix(obsset, basename_postfix):
    pass
    #obsset.set_basename_postfix(basename_postfix)


def get_a0v_thresh_masks(a0v, threshold_a0v):
    # thresh_masks = []

    thresh_masks = np.ones(a0v.spec.shape, dtype=bool)

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
                                               #flatten=True,
                                               flatten=False,
                                               smooth_pixel=32)

    aa = [a0v_interp1d(wvl)[0] for wvl in zip(um)]

    return np.array(aa)


# def get_tgt_spec_cor(obsset, tgt, a0v,
#                      threshold_a0v=None):

#     thresh_masks = get_a0v_thresh_masks(a0v, threshold_a0v)

#     aa = get_vega_spec(obsset, tgt.um)

#     tgt_spec_cor = [s/t*a for s, t, a in zip(tgt.spec, a0v.spec, aa)]

#     return tgt_spec_cor, thresh_masks, aa

from ..procedures.order_mismatch_cor import get_order_match_corr

def _make_spec_a0v_hdu_list(obsset, wvl, spec, variance, a0v_spec, a0v_variance, vega,
                            a0v_fitted_continuum,
                            thresh_masks,
                            header_updates=None,
                            force_order_match=True):

    #primary_header_cards = [("EXTNAME", "SPEC_DIVIDE_A0V")]
    primary_header_cards = [("EXTNAME", "SCI"), ("EXTVER", 1), ("EXTDESC", "SPEC_DIVIDE_A0V")]
    if header_updates is not None:
        primary_header_cards.extend(header_updates)

    spec_divided_by_a0v_variance = vega**2 * (spec/a0v_spec)**2 * ( (variance/(spec**2)) + (a0v_variance/(a0v_spec**2)) ) #Compute variance for spec/a0v * vega using error propogation
    spec_divided_by_cont_variance = (a0v_spec / a0v_fitted_continuum)**2 * spec_divided_by_a0v_variance #Scale to get the variance for spec /flattened a0v  * vega


    if force_order_match:
        from ..igrins_libs.resource_helper_igrins import ResourceHelper
        helper = ResourceHelper(obsset)
        orders = helper.get("orders")

        _variance = (spec/a0v_spec)**2 * ( (variance/(spec**2)) + (a0v_variance/(a0v_spec**2)) ) #Compute variance for spec/a0v * vega using error propogation

        mask, corr = get_order_match_corr(orders, wvl,
                                          spec / a0v_spec, _variance)
        thresh_masks[mask] = True
    else:
        corr = 1

    _hdul = [
        (primary_header_cards, spec/a0v_spec*vega/corr),
        # ([("EXTNAME", "SPEC_DIVIDE_A0V_VARIANCE")], spec_divided_by_a0v_variance), #Old EXTNAME
        # ([("EXTNAME", "WAVELENGTH")], wvl),
        # ([("EXTNAME", "TGT_SPEC")], spec),
        # ([("EXTNAME", "TGT_SPEC_VARIANCE")], variance),
        # ([("EXTNAME", "A0V_SPEC")], a0v_spec),
        # ([("EXTNAME", "A0V_SPEC_VARIANCE")], a0v_variance),
        # ([("EXTNAME", "VEGA_SPEC")], vega),
        # ([("EXTNAME", "SPEC_DIVIDE_CONT")], spec/a0v_fitted_continuum*vega),
        # ([("EXTNAME", "SPEC_DIVIDE_CONT_VARIANCE")], spec_divided_by_cont_variance),
        # ([("EXTNAME", "MASK")], thresh_masks.astype("i"))
        ([("EXTNAME", "VAR"), ("EXTVER", 1), ("EXTDESC", "SPEC_DIVIDE_A0V_VARIANCE")], spec_divided_by_a0v_variance), #Changes made to EXTNAME for extenstion headers in spec_a0v.fits files for ingestion into Gemini Archive, moved old keyword value to new keyword EXTDESC
        ([("EXTNAME", "SCI"), ("EXTVER", 2), ("EXTDESC", "WAVELENGTH")], wvl),
        ([("EXTNAME", "SCI"), ("EXTVER", 3), ("EXTDESC", "TGT_SPEC")], spec),
        ([("EXTNAME", "VAR"), ("EXTVER", 3), ("EXTDESC", "TGT_SPEC_VARIANCE")], variance),
        ([("EXTNAME", "SCI"), ("EXTVER", 4), ("EXTDESC", "A0V_SPEC")], a0v_spec),
        ([("EXTNAME", "VAR"), ("EXTVER", 4), ("EXTDESC", "A0V_SPEC_VARIANCE")], a0v_variance),
        ([("EXTNAME", "SCI"), ("EXTVER", 5), ("EXTDESC", "VEGA_SPEC")], vega),
        ([("EXTNAME", "SCI"), ("EXTVER", 6), ("EXTDESC", "SPEC_DIVIDE_CONT")], spec/a0v_fitted_continuum*vega),
        ([("EXTNAME", "VAR"), ("EXTVER", 6), ("EXTDESC", "SPEC_DIVIDE_CONT_VARIANCE")], spec_divided_by_cont_variance),
        ([("EXTNAME", "DQ"), ("EXTVER", 1), ("EXTDESC", "MASK")], thresh_masks.astype("i"))
    ]



    if force_order_match:
        #_hdul.append(([("EXTNAME", "ORDER_MATCH_COR")], corr))
        _hdul.append(([("EXTNAME", "SCI"), ("EXTVER", 7), ("EXTDESC", "ORDER_MATCH_COR")], corr))

    # _hdul[0].verify(option="fix")

    hdul = obsset.get_hdul_to_write(*_hdul, convention="gemini")


    return hdul

def get_a0v(obsset, a0v="GROUP2", a0v_obsid=None,
            basename_postfix=None):

    a0v_obsid = get_a0v_obsid(obsset, a0v, a0v_obsid)
    if a0v_obsid is None:
        a0v_obsid_ = obsset.query_resource_basename("a0v")
        a0v_obsid = obsset.rs.parse_basename(a0v_obsid_)

    a0v_obsset = type(obsset)(obsset.rs, "A0V_AB", [a0v_obsid], ["A"],
                              basename_postfix=basename_postfix)

    a0v = OnedSpecHelper(a0v_obsset, basename_postfix=basename_postfix)

    return a0v

def get_divide_a0v_hdul(obsset,
                        a0v='GROUP2',
                        a0v_obsid=None,
                        basename_postfix=None,
                        # outname_postfix=None,
                        #a0v_basename_postfix="",
                        no_order_match=False,
                        threshold_a0v=0.1):

    tgt = OnedSpecHelper(obsset, basename_postfix=basename_postfix)
    a0v = get_a0v(obsset, a0v, a0v_obsid, basename_postfix)


    vega_spec = get_interpolated_vega_spec(obsset, tgt.um)

    a0v_fitted_continuum = a0v.flattened_hdu_list["FITTED_CONTINUUM"].data

    thresh_masks = get_a0v_thresh_masks(a0v, threshold_a0v)

    #breakpoint()

    hdul = _make_spec_a0v_hdu_list(obsset,
                                   tgt.um,
                                   tgt.spec,
                                   tgt.variance,
                                   a0v.spec, a0v.variance, vega_spec,
                                   a0v_fitted_continuum,
                                   thresh_masks,
                                   force_order_match=not no_order_match,
                                   header_updates=None)



    if a0v_obsid == None: #Error catch, fill in a0v_obsid if it is None so that it properly saves in 
        a0v_obsid = a0v.obsset.obsids[0]

    #Pass headers from .spec.fits and .variance.fits files to the various extensions, and make sure the OBSIDs are passed
    #hdul["TGT_SPEC"].header.update(tgt._spec_hdu_list[0].header)
    #hdul["TGT_SPEC"].header["OBSID"] = str(obsset.obsids[0])
    #hdul["A0V_SPEC"].header.update(a0v._spec_hdu_list[0].header)
    #hdul["A0V_SPEC"].header["OBSID"] = a0v_obsid
    hdul[4].header.update(tgt._spec_hdu_list[0].header)
    hdul[4].header["OBSID"] = str(obsset.obsids[0])
    hdul[6].header.update(a0v._spec_hdu_list[0].header)
    hdul[6].header["OBSID"] = a0v_obsid

    #Delete keywords from headers carried over from the primary headers of single fits files that are not fits standard for an extension in a multi-extension fits file
    del hdul[4].header['SIMPLE']
    del hdul[4].header['EXTEND']
    del hdul[6].header['SIMPLE']
    del hdul[6].header['EXTEND']

    return hdul


def divide_a0v(obsset,
               a0v='GROUP2',
               a0v_obsid=None,
               basename_postfix=None,
               # outname_postfix=None,
               #a0v_basename_postfix="",
               no_order_match=False,
               threshold_a0v=0.1):

    hdul = get_divide_a0v_hdul(obsset,
                               a0v=a0v,
                               a0v_obsid=a0v_obsid,
                               basename_postfix=basename_postfix,
                               no_order_match=no_order_match,
                               threshold_a0v=threshold_a0v)

    obsset.store("SPEC_A0V_FITS", hdul, postfix=basename_postfix)


steps = [
        #Step("Set basename-postfix", set_basename_postfix,
        # basename_postfix=""),
        Step("Divide w/ A0V", divide_a0v,
             basename_postfix="",
             no_order_match=ArghFactoryWithShort(False)),
        ]
