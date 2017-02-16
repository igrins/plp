import astropy.io.fits as pyfits
import numpy as np

from libs.a0v_spec import A0V


def generate_a0v_divided(helper, band, obsids, a0v_obsid=None,
                         basename_postfix=None):

    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    tgt_spec_hdulist = caldb.load_item_from((band, master_obsid),
                                            "SPEC_FITS",
                                            basename_postfix=basename_postfix)
    spec = tgt_spec_hdulist[0].data
    wvl = tgt_spec_hdulist[1].data

    print master_obsid, a0v_obsid
    if a0v_obsid is None:
        a0v_basename = caldb.db_query_basename("a0v", band, master_obsid)
    else:
        a0v_basename = helper.get_basename(band, a0v_obsid)

    a0v_spec_hdulist = caldb.load_item_from(a0v_basename,
                                            "SPEC_FITS",
                                            basename_postfix=basename_postfix)

    a0v_spec = a0v_spec_hdulist[0].data

    a0v_spec_flattened_hdulist = caldb.load_item_from(a0v_basename,
                                                      "SPEC_FITS_FLATTENED",
                                                      basename_postfix=basename_postfix)

    a0v_fitted_continuum = a0v_spec_flattened_hdulist["FITTED_CONTINUUM"].data

    a0v_interp1d = A0V.get_flux_interp1d(helper.config)
    vega = np.array([a0v_interp1d(w1) for w1 in wvl])

    master_hdu = tgt_spec_hdulist[0]

    basename = helper.get_basename(band, master_obsid)

    
    header_updates = [("hierarch IGR_A0V_BASENAME", a0v_basename),
                      ("hierarch IGR_BASENAME_POSTFIX", basename_postfix),]

    store_tgt_divide_a0v(caldb, basename,
                         master_hdu,
                         wvl, spec, a0v_spec, vega,
                         a0v_fitted_continuum,
                         header_updates=header_updates,
                         basename_postfix=basename_postfix)


def store_tgt_divide_a0v(caldb, basename,
                         master_hdu,
                         wvl, spec, a0v_spec, vega,
                         a0v_fitted_continuum,
                         header_updates=None,
                         basename_postfix=None):

    from libs.products import PipelineImage as Image

    primary_header_cards = [("EXTNAME", "SPEC_DIVIDE_A0V")]
    if header_updates is not None:
        primary_header_cards.extend(header_updates)

    image_list = [Image(primary_header_cards,
                        spec/a0v_spec*vega)]
    image_list.append(Image([("EXTNAME", "WAVELENGTH")],
                            wvl))
    image_list.append(Image([("EXTNAME", "TGT_SPEC")],
                            spec))
    image_list.append(Image([("EXTNAME", "A0V_SPEC")],
                            a0v_spec))
    image_list.append(Image([("EXTNAME", "VEGA_SPEC")],
                            vega))
    image_list.append(Image([("EXTNAME", "SPEC_DIVIDE_CONT")],
                            spec/a0v_fitted_continuum*vega))


    item_type = "SPEC_A0V_FITS"
    caldb.store_multi_image(basename, item_type, image_list,
                            basename_postfix=basename_postfix)



if 0:
    hdulist = pyfits.open("/home/jjlee/work/igrins/plp_jjlee/outdata/20140525/SDCK_20140525_0042.spec_a0v.fits")

    s = hdulist["SPEC_DIVIDE_A0V"].data
    wvl = hdulist["WAVELENGTH"].data
    tgt_s = hdulist["TGT_SPEC"].data
    a0v_s = hdulist["A0V_SPEC"].data
    vega_s = hdulist["VEGA_SPEC"].data

    clf()
    m = np.median(a0v_s)
    for w1, s1, a0v1 in zip(wvl, s, a0v_s):
        s1 = np.ma.array(s1, mask=a0v1<0.3*m).filled(np.nan)
        plot(w1, s1)

from libs.recipe_helper import RecipeHelper

def process_band(utdate, recipe_name, band, 
                 obsids, frame_types, aux_infos,
                 config_name, 
                 a0v_obsid=None,
                 basename_postfix=None):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    generate_a0v_divided(helper, band, obsids, a0v_obsid,
                         basename_postfix=basename_postfix)


if __name__ == "__main__":

    recipe_name = "divide_a0v"
    utdate = "20140525"
    obsids = [42]
    frame_types = [None]
    aux_infos = [None]

    band = "K"

    config_name = "../recipe.config"

    process_band(utdate, recipe_name, band, 
                 obsids, frame_types, aux_infos, 
                 config_name)
