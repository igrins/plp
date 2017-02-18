
# desciptions for pipeline storage

# FLAT_OFF related

FLAT_OFF_DESC = ("OUTDATA_PATH", "", ".flat_off.fits")
HOTPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".hotpix_mask.fits")
FLATOFF_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_off.json")


# FLAT_ON related

FLAT_ON_DESC = ("OUTDATA_PATH", "", ".flat_on.fits")
FLAT_NORMED_DESC = ("OUTDATA_PATH", "", ".flat_normed.fits")
FLAT_BPIXED_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_bpixed.fits")
FLAT_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_mask.fits")
DEADPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".deadpix_mask.fits")
FLATON_JSON_DESC = ("OUTDATA_PATH", "", ".flat_on.json")

FLAT_DERIV_DESC = ("SECONDARY_CALIB_PATH", "FLAT_", ".flat_deriv.fits")
FLATCENTROIDS_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".centroids.json")

FLATCENTROID_SOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".centroid_solutions.json")

BIAS_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".bias_mask.fits")

ORDER_FLAT_IM_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_", ".fits")
ORDER_FLAT_JSON_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_", ".json")


# image related descriptions

COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "", ".combined_image.fits")
COMBINED_IMAGE_A_DESC = ("OUTDATA_PATH", "", ".combined_image_a.fits")
COMBINED_IMAGE_B_DESC = ("OUTDATA_PATH", "", ".combined_image_b.fits")
WVLCOR_IMAGE_DESC = ("OUTDATA_PATH", "", ".wvlcor_image.fits")
ONED_SPEC_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".oned_spec.json")
MULTI_SPEC_FITS_DESC = ("PRIMARY_CALIB_PATH", "", ".multi_spec.fits")

# THAR related descriptions

#COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "", ".combined_image.fits")
#ONED_SPEC_JSON_DESC = ("OUTDATA_PATH", "", ".oned_spec.json")

ORDERS_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".orders.json")

IDENTIFIED_LINES_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".identified_lines.json")

THAR_REID_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".thar_reid.json")
THAR_ALIGNED_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".thar_aligned.json")
THAR_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".wvlsol_v0.json")

ALIGNING_MATRIX_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".aligning_matrix.json")
WVLSOL_V0_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".wvlsol_v0.json")

# SKY

COMBINED_SKY_DESC = ("OUTDATA_PATH", "", ".combined_sky.fits")
SKY_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.json")
SKY_WVLSOL_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.fits")

ORDERMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".order_map.fits")
ORDERMAP_MASKED_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".order_map_masked.fits")
SLITPOSMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitpos_map.fits")
SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitoffset_map.fits")
WAVELENGTHMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wavelength_map.fits")

SKY_FITTED_PIXELS_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".fitted_pixels.json")

VOLUMEFIT_COEFFS_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".volumefit_coeffs.json")

SKY_WVLSOL_FIT_RESULT_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_",
                                   ".wvlsol_fit_result.json")

# extract
SPEC_FITS_WAVELENGTH_DESC = ("OUTDATA_PATH", "",
                            ".wave.fits")
SPEC_FITS_FLATTENED_DESC = ("OUTDATA_PATH", "",
                            ".spec_flattened.fits")
SPEC_FITS_DESC = ("OUTDATA_PATH", "", ".spec.fits")
VARIANCE_FITS_DESC = ("OUTDATA_PATH", "", ".variance.fits")
SN_FITS_DESC = ("OUTDATA_PATH", "", ".sn.fits")

VARIANCE_MAP_DESC = ("OUTDATA_PATH", "", ".variance_map.fits")

SLIT_PROFILE_JSON_DESC = ("OUTDATA_PATH", "", ".slit_profile.json")

SPEC2D_FITS_DESC = ("OUTDATA_PATH", "", ".spec2d.fits")

SPEC_A0V_FITS_DESC = ("OUTDATA_PATH", "", ".spec_a0v.fits")

#Added by Kyle Kaplan on Feb 25, 2015
#Save variance map as straitened out 2D datacube like spec2d.fits
VAR2D_FITS_DESC = ("OUTDATA_PATH", "", ".var2d.fits")


# QA

QA_FLAT_APERTURE_DIR_DESC = ("QA_PATH", "aperture_", "")
QA_ORDERFLAT_DIR_DESC = ("QA_PATH", "orderflat_", "")


#####

DB_Specs = dict(flat_on=("PRIMARY_CALIB_PATH", "flat_on.db"),
                flat_off=("PRIMARY_CALIB_PATH", "flat_off.db"),
                register=("PRIMARY_CALIB_PATH", "register.db"),
                distortion=("PRIMARY_CALIB_PATH", "distortion.db"),
                wvlsol=("PRIMARY_CALIB_PATH", "wvlsol.db"),
                a0v=("OUTDATA_PATH", "a0v.db"),
                )


def load_descriptions():
    storage_descriptions = globals()
    desc_list = [n for n in storage_descriptions if n.endswith("_DESC")]
    desc_dict = dict((n[:-5].upper(),
                      storage_descriptions[n]) for n in desc_list)

    return desc_dict


_resource_definitions = dict(
    aperture_definition=("flat_on", "FLATCENTROID_SOL_JSON"),
    deadpix_mask=("flat_on", "DEADPIX_MASK"),
    bias_mask=("flat_on", "BIAS_MASK"),
    #
    hotpix_mask=("flat_off", "HOTPIX_MASK"),
    flat_off=("flat_off", "FLAT_OFF"),
    #
    orders=("register", "ORDERS_JSON"),
    wvlsol_v0=("register", "WVLSOL_V0_JSON"),
    #
    ordermap=("distortion", "ORDERMAP_FITS"),
    slitposmap=("distortion", "SLITPOSMAP_FITS"),
    #
    wvlsol=("wvlsol", "SKY_WVLSOL_JSON"),
)

def load_resource_def():
    resource_dict = {}
    desc_names = globals()

    for k, (db_name, desc_prefix) in _resource_definitions.iteritems():
        e1 = desc_names.get(desc_prefix + "_DESC")
        resource_dict[k] = (db_name, e1)

    return resource_dict

