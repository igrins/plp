import numpy as np


def _get_a0v_interp1d(obsset):
    vega_data = obsset.rs.load_ref_data("VEGA_SPEC")
    from .a0v_spec import A0VSpec
    a0v_spec = A0VSpec(vega_data)
    a0v_interp1d = a0v_spec.get_flux_interp1d(1.3, 2.5,
                                              flatten=False,
                                              smooth_pixel=32)
    return a0v_interp1d


def flatten_a0v(obsset, fill_nan=None):  # refactor of get_a0v_flattened
    "This is the main function to do flattening"

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)

    wvl_solutions = helper.get("wvl_solutions")  # extractor.wvl_solutionsw
    tel_interp1d_f = get_tel_interp1d_f(obsset, wvl_solutions)

    a0v_interp1d = _get_a0v_interp1d(obsset)
    # from ..libs.a0v_spec import A0V
    # a0v_interp1d = A0V.get_flux_interp1d(self.config)
    orderflat_json = obsset.load_resource_for("order_flat_json")
    orderflat_response = orderflat_json["fitted_responses"]

    s_list = obsset.load_fits_sci_hdu("SPEC_FITS",
                                      postfix=obsset.basename_postfix).data

    from .a0v_flatten_telluric import get_a0v_flattened
    data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                                  wvl_solutions, s_list, orderflat_response)

    if fill_nan is not None:
        flattened_s = data_list[0][1]
        flattened_s[~np.isfinite(flattened_s)] = fill_nan

    store_a0v_results(obsset, data_list)


def store_a0v_results(obsset, a0v_flattened_data):

    from .target_spec import get_wvl_header_data
    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    image_list = []
    image_list.append(([("EXTNAME", "SPEC_FLATTENED")],
                       convert_data(a0v_flattened_data[0][1])))

    for ext_name, data in a0v_flattened_data[1:]:
        _ = ([("EXTNAME", ext_name.upper())], convert_data(data))
        image_list.append(_)

    hdul = obsset.get_hdul_to_write(*image_list)
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify(option="silentfix")

    obsset.store("spec_fits_flattened", hdul)


# if 0:

#     # orderflat_response = extractor.orderflat_json["fitted_responses"]

#     # tgt_basename = extractor.pr.tgt_basename
#     # igr_path = extractor.igr_path
#     # figout = igr_path.get_section_filename_base("QA_PATH",
#     #                                             "flattened_"+tgt_basename) + ".pdf"

#     from igrins.libs.a0v_flatten import get_a0v_flattened
#     data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
#                                     wvl, s_list, orderflat_response,
#                                     figout=figout)

#     if self.fill_nan is not None:
#         flattened_s = data_list[0][1]
#         flattened_s[~np.isfinite(flattened_s)] = self.fill_nan

#     return data_list


def get_tel_interp1d_f(obsset, wvl_solutions):

    # from ..libs.master_calib import get_master_calib_abspath
    # fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
    # self.telluric = pyfits.open(fn)[1].data

    # telfit_outname = "telluric/transmission-795.20-288.30-41.9-45.0-368.50-3.90-1.80-1.40.%s" % extractor.band
    # telfit_outname_npy = telfit_outname+".npy"
    telfit_outname_npy = obsset.rs.query_ref_data_path("TELFIT_MODEL")
    # telfit_outname_npy = obsset.rs.query_ref_data_path("VEGA_SPEC")
    from ..igrins_libs.logger import debug

    debug("loading TELFIT_MODEL: {}".format(telfit_outname_npy))

    # if 0:
    #     dd = np.genfromtxt(telfit_outname)
    #     np.save(open(telfit_outname_npy, "w"), dd[::10])

    from .a0v_flatten_telluric import TelluricTransmission
    # _fn = get_master_calib_abspath(telfit_outname_npy)
    tel_trans = TelluricTransmission(telfit_outname_npy)

    wvl_solutions = np.array(wvl_solutions)

    w_min = wvl_solutions.min()*0.9
    w_max = wvl_solutions.max()*1.1

    def tel_interp1d_f(gw=None):
        return tel_trans.get_telluric_trans_interp1d(w_min, w_max, gw)

    return tel_interp1d_f
