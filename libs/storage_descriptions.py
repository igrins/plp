
# desciptions for pipeline storage

# FLAT_OFF related

FLAT_OFF_DESC = ("OUTDATA_PATH", "", ".flat_off.fits")
HOTPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".hotpix_mask.fits")
FLATOFF_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_off.json")


# FLAT_ON related

FLAT_NORMED_DESC = ("OUTDATA_PATH", "", ".flat_normed.fits")
FLAT_BPIXED_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_bpixed.fits")
FLAT_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".flat_mask.fits")
DEADPIX_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".deadpix_mask.fits")
FLATON_JSON_DESC = ("OUTDATA_PATH", "", ".flat_on.json")

FLAT_DERIV_DESC = ("SECONDARY_CALIB_PATH", "FLAT_", ".flat_deriv.fits")
FLATCENTROIDS_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".centroids.json")

FLATCENTROID_SOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".centroid_solutions.json")
FLATCENTROID_ORDERS_JSON_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".orders.json")

BIAS_MASK_DESC = ("PRIMARY_CALIB_PATH", "FLAT_", ".bias_mask.fits")

ORDER_FLAT_IM_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_", ".fits")
ORDER_FLAT_JSON_DESC = ("PRIMARY_CALIB_PATH", "ORDERFLAT_", ".json")


# image related descriptions

COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "", ".combined_image.fits")
COMBINED_IMAGE_A_DESC = ("OUTDATA_PATH", "", ".combined_image_a.fits")
COMBINED_IMAGE_B_DESC = ("OUTDATA_PATH", "", ".combined_image_b.fits")
WVLCOR_IMAGE_DESC = ("OUTDATA_PATH", "", ".wvlcor_image.fits")
ONED_SPEC_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".oned_spec.json")

# THAR related descriptions

#COMBINED_IMAGE_DESC = ("OUTDATA_PATH", "", ".combined_image.fits")
#ONED_SPEC_JSON_DESC = ("OUTDATA_PATH", "", ".oned_spec.json")

IDENTIFIED_LINES_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".identified_lines.json")

THAR_REID_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".thar_reid.json")
THAR_ALIGNED_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".thar_aligned.json")
THAR_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "THAR_", ".wvlsol_v0.json")

ALIGNING_MATRIX_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".aligning_matrix.json")
WVLSOL_V0_JSON_DESC = ("PRIMARY_CALIB_PATH", "", ".wvlsol_v0.json")

# SKY

SKY_WVLSOL_JSON_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.json")
SKY_WVLSOL_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wvlsol_v1.fits")

ORDERMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".order_map.fits")
ORDERMAP_MASKED_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".order_map_masked.fits")
SLITPOSMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitpos_map.fits")
SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitoffset_map.fits")
WAVELENGTHMAP_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".wavelength_map.fits")

# extract
SPEC_FITS_FLATTENED_DESC = ("OUTDATA_PATH", "",
                            ".spec_flattened.fits")
SPEC_FITS_DESC = ("OUTDATA_PATH", "", ".spec.fits")
VARIANCE_FITS_DESC = ("OUTDATA_PATH", "", ".variance.fits")
SN_FITS_DESC = ("OUTDATA_PATH", "", ".sn.fits")

VARIANCE_MAP_DESC = ("OUTDATA_PATH", "", ".variance_map.fits")

SLIT_PROFILE_JSON_DESC = ("OUTDATA_PATH", "", ".slit_profile.json")

SPEC2D_FITS_DESC = ("OUTDATA_PATH", "", ".spec2d.fits")
