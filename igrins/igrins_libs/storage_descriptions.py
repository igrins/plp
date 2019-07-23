
# desciptions for pipeline storage

# INPUT

RAWIMAGE_DESC = ("INDATA_PATH", "{basename}{postfix}.fits")

# DARK related

PAIR_SUBTRACTED_IMAGES_DESC = ("OUTDATA_PATH", "{basename}{postfix}.pair_subtracted.fits")

# FLAT_OFF related

FLAT_OFF_DESC = ("OUTDATA_PATH", "{basename}{postfix}.flat_off.fits")
FLAT_OFF_BG_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.flat_off_bg.fits")
HOTPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.hotpix_mask.fits")
FLATOFF_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.flat_off.json")

FLAT_MOMENTS_FITS_DESC = ("SECONDARY_CALIB_PATH",
                          "FLAT_{basename}{postfix}.moments.fits")

# FLAT_ON related

FLAT_ON_DESC = ("OUTDATA_PATH", "{basename}{postfix}.flat_on.fits")
FLAT_NORMED_DESC = ("OUTDATA_PATH", "{basename}{postfix}.flat_normed.fits")
FLAT_BPIXED_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.flat_bpixed.fits")
FLAT_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.flat_mask.fits")
DEADPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.deadpix_mask.fits")
FLATON_JSON_DESC = ("OUTDATA_PATH", "{basename}{postfix}.flat_on.json")

FLAT_DERIV_DESC = ("SECONDARY_CALIB_PATH", "FLAT_{basename}{postfix}.flat_deriv.fits")
FLATCENTROIDS_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.centroids.json")

FLATCENTROID_SOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.centroid_solutions.json")

BIAS_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_{basename}{postfix}.bias_mask.fits")

ORDER_FLAT_IM_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_{basename}{postfix}.fits")
ORDER_FLAT_JSON_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_{basename}{postfix}.json")


# image related descriptions

STACKED_DESC = ("OUTDATA_PATH", "{basename}{postfix}.stacked.fits")
COMBINED_SKY_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_sky.fits")
COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_image.fits")

# below 4 are cache only
COMBINED_IMAGE1_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_image1.fits")
COMBINED_VARIANCE1_DESC = ("OUTDATA_PATH",
                          "{basename}{postfix}.combined_variance1.fits")
COMBINED_VARIANCE0_DESC = ("OUTDATA_PATH",
                           "{basename}{postfix}.combined_variance0.fits")
INTERORDER_BACKGROUND_DESC = ("OUTDATA_PATH",
                              "{basename}{postfix}.interorder_background.fits")

SLITPROFILE_FITS_DESC = ("OUTDATA_PATH",
                        "{basename}{postfix}.slitprofile.fits")

COMBINED_IMAGE_A_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_image_a.fits")
COMBINED_IMAGE_B_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_image_b.fits")
WVLCOR_IMAGE_DESC = ("OUTDATA_PATH", "{basename}{postfix}.wvlcor_image.fits")
DEBUG_IMAGE_DESC = ("OUTDATA_PATH", "{basename}{postfix}.debug_image.fits")
ONED_SPEC_JSON_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.oned_spec.json")
MULTI_SPEC_FITS_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.multi_spec.fits")

# THAR related descriptions

#COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "{basename}{postfix}.combined_image.fits")
#ONED_SPEC_JSON_DESC = ("OUTDATA_PATH", "{basename}{postfix}.oned_spec.json")

ORDERS_JSON_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.orders.json")

IDENTIFIED_LINES_JSON_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.identified_lines.json")

THAR_REID_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_{basename}{postfix}.thar_reid.json")
THAR_ALIGNED_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_{basename}{postfix}.thar_aligned.json")
THAR_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_{basename}{postfix}.wvlsol_v0.json")

ALIGNING_MATRIX_JSON_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.aligning_matrix.json")
WVLSOL_V0_JSON_DESC = ("PRIMARY_CALIB_PATH", "{basename}{postfix}.wvlsol_v0.json")

# SKY

SKY_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.wvlsol_v1.json")
SKY_WVLSOL_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.wvlsol_v1.fits")

ORDERMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.order_map.fits")
ORDERMAP_MASKED_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.order_map_masked.fits")
SLITPOSMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.slitpos_map.fits")
SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.slitoffset_map.fits")
WAVELENGTHMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.wavelength_map.fits")

SKY_IDENTIFIED_JSON_DESC = ("OUTDATA_PATH", "{basename}{postfix}.sky_identified.json")
SKY_FITTED_PIXELS_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.fitted_pixels.json")

VOLUMEFIT_COEFFS_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.volumefit_coeffs.json")

SKY_WVLSOL_FIT_RESULT_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_{basename}{postfix}.wvlsol_fit_result.json")

# extract
SPEC_FITS_WAVELENGTH_DESC = ("OUTDATA_PATH", "{basename}{postfix}.wave.fits")
SPEC_FITS_FLATTENED_DESC = ("OUTDATA_PATH", "{basename}{postfix}.spec_flattened.fits")
SPEC_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.spec.fits")
VARIANCE_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.variance.fits")
SN_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.sn.fits")

VARIANCE_MAP_DESC = ("OUTDATA_PATH", "{basename}{postfix}.variance_map.fits")

SLIT_PROFILE_JSON_DESC = ("OUTDATA_PATH", "{basename}{postfix}.slit_profile.json")

SPEC2D_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.spec2d.fits")

SPEC_A0V_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.spec_a0v.fits")

#Added by Kyle Kaplan on Feb 25, 2015
#Save variance map as straitened out 2D datacube like spec2d.fits
VAR2D_FITS_DESC = ("OUTDATA_PATH", "{basename}{postfix}.var2d.fits")


# QA

QA_FLAT_APERTURE_DIR_DESC = ("QA_PATH", "aperture_{basename}{postfix}")
QA_ORDERFLAT_DIR_DESC = ("QA_PATH", "orderflat_{basename}{postfix}")
QA_SKY_FIT2D_DIR_DESC = ("QA_PATH", "sky_fit2d_{basename}{postfix}")


#####

DB_Specs = dict(flat_on=("PRIMARY_CALIB_PATH", "flat_on.db"),
                flat_off=("PRIMARY_CALIB_PATH", "flat_off.db"),
                register=("PRIMARY_CALIB_PATH", "register.db"),
                distortion=("PRIMARY_CALIB_PATH", "distortion.db"),
                wvlsol=("PRIMARY_CALIB_PATH", "wvlsol.db"),
                a0v=("OUTDATA_PATH", "a0v.db"),
                )


class UpperDict(dict):
    def __getitem__(self, k):
        return dict.__getitem__(self, k.upper())

def load_descriptions():
    storage_descriptions = globals()
    desc_list = [n for n in storage_descriptions if n.endswith("_DESC")]
    desc_dict = UpperDict((n[:-5].upper(),
                           storage_descriptions[n]) for n in desc_list)

    return desc_dict


_resource_definitions = dict(
    flat_normed=("flat_on", "FLAT_NORMED"),
    flat_mask=("flat_on", "FLAT_MASK"),
    #
    aperture_definition=("flat_on", "FLATCENTROID_SOL_JSON"),
    deadpix_mask=("flat_on", "DEADPIX_MASK"),
    bias_mask=("flat_on", "BIAS_MASK"),
    #
    hotpix_mask=("flat_off", "HOTPIX_MASK"),
    flat_off=("flat_off", "FLAT_OFF"),
    #
    orders=("register", "ORDERS_JSON"),
    wvlsol_v0=("register", "WVLSOL_V0_JSON"),
    order_flat=("register", "ORDER_FLAT_IM"),
    order_flat_json=("register", "ORDER_FLAT_JSON"),
    #
    ordermap=("distortion", "ORDERMAP_FITS"),
    slitposmap=("distortion", "SLITPOSMAP_FITS"),
    slitoffsetmap=("distortion", "SLITOFFSET_FITS"),
    #
    wvlsol=("wvlsol", "SKY_WVLSOL_JSON"),
    wvlsol_fits=("wvlsol", "SKY_WVLSOL_FITS"),
)


def load_resource_def():
    resource_dict = {}
    desc_names = globals()

    for k, (db_name, desc_prefix) in _resource_definitions.items():
        e1 = desc_names.get(desc_prefix + "_DESC")
        resource_dict[k] = (db_name, e1)

    return resource_dict
