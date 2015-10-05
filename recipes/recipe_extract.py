import os
import numpy as np

from libs.path_info import IGRINSPath
import libs.fits as pyfits

from libs.products import PipelineProducts

import argh

def extractor_factory(recipe_name):
    @argh.arg("-s", "--starting-obsids", default=None)
    @argh.arg("-c", "--config-file", default="recipe.config")
    @argh.arg("--wavelength-increasing-order")
    @argh.arg("--fill-nan")
    @argh.arg("--lacosmics-thresh")
    def extract(utdate, refdate="20140316", bands="HK",
                starting_obsids=None,
                config_file="recipe.config",
                frac_slit=None,
                cr_rejection_thresh=30,
                debug_output=False,
                wavelength_increasing_order=False,
                fill_nan=None,
                lacosmics_thresh=0,
                subtract_interorder_background=False,
                ):
        abba_all(recipe_name, utdate, refdate=refdate, bands=bands,
                 starting_obsids=starting_obsids,
                 config_file=config_file,
                 frac_slit=frac_slit,
                 cr_rejection_thresh=cr_rejection_thresh,
                 debug_output=debug_output,
                 wavelength_increasing_order=wavelength_increasing_order,
                 fill_nan=fill_nan,
                 lacosmics_thresh=lacosmics_thresh,
                 subtract_interorder_background=subtract_interorder_background,
                 )

    extract.__name__ = recipe_name.lower()
    return extract

a0v_ab = extractor_factory("A0V_AB")
a0v_onoff = extractor_factory("A0V_ONOFF")
stellar_ab = extractor_factory("STELLAR_AB")
stellar_onoff = extractor_factory("STELLAR_ONOFF")
extended_ab = extractor_factory("EXTENDED_AB")
extended_onoff = extractor_factory("EXTENDED_ONOFF")



def abba_all(recipe_name, utdate, refdate="20140316", bands="HK",
             starting_obsids=None, interactive=False,
             config_file="recipe.config",
             frac_slit=None,
             cr_rejection_thresh=100.,
             debug_output=False,
             wavelength_increasing_order=False,
             fill_nan=None,
             lacosmics_thresh=0,
             subtract_interorder_background=False,
             ):

    from libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    if not bands in ["H", "K", "HK"]:
        raise ValueError("bands must be one of 'H', 'K' or 'HK'")

    fn = config.get_value('RECIPE_LOG_PATH', utdate)
    from libs.recipes import Recipes #load_recipe_list, make_recipe_dict
    recipe = Recipes(fn)

    if starting_obsids is not None:
        starting_obsids = map(int, starting_obsids.split(","))

    selected = recipe.select(recipe_name, starting_obsids)
    if not selected:
        print "no recipe of with matching arguments is found"

    if frac_slit is not None:
        frac_slit = map(float, frac_slit.split(","))
        if len(frac_slit) !=2:
            raise ValueError("frac_slit must be two floats separated by comma")

    kwargs = dict(frac_slit=frac_slit,
                  cr_rejection_thresh=cr_rejection_thresh,
                  debug_output=debug_output,
                  wavelength_increasing_order=wavelength_increasing_order,
                  subtract_interorder_background=subtract_interorder_background,
                  fill_nan=fill_nan)

    process_abba_band = ProcessABBABand(utdate, refdate,
                                        config,
                                        **kwargs).process

    if len(selected) == 0:
        print "No entry with given recipe is found : %s" % recipe_name

    for s in selected:
        obsids = s[0]
        frametypes = s[1]

        for band in bands:
            process_abba_band(recipe_name, band,
                              obsids, frametypes,
                              #do_interactive_figure=interactive
                              )

from libs.products import ProductDB, PipelineStorage

class ProcessABBABand(object):
    def __init__(self, utdate, refdate, config,
                 frac_slit=None,
                 cr_rejection_thresh=100,
                 debug_output=False,
                 wavelength_increasing_order=False,
                 fill_nan=None,
                 lacosmics_thresh=0,
                 subtract_interorder_background=False):
        """
        cr_rejection_thresh : pixels that deviate significantly from the profile are excluded.
        """
        self.utdate = utdate
        self.refdate = refdate
        self.config = config

        self.igr_path = IGRINSPath(config, utdate)

        self.igr_storage = PipelineStorage(self.igr_path)

        self.frac_slit = frac_slit
        self.cr_rejection_thresh = cr_rejection_thresh
        self.lacosmics_thresh = lacosmics_thresh
        self.debug_output = debug_output

        self.fill_nan=fill_nan

        self.wavelength_increasing_order = wavelength_increasing_order

        self.subtract_interorder_background = subtract_interorder_background

    def process(self, recipe, band, obsids, frametypes):

        igr_storage = self.igr_storage


        if "_" in recipe:
            target_type, nodding_type = recipe.split("_")
        else:
            raise ValueError("Unknown recipe : %s" % recipe)

        if target_type in ["A0V", "STELLAR"]:
            IF_POINT_SOURCE = True
        elif target_type in ["EXTENDED"]:
            IF_POINT_SOURCE = False
        else:
            raise ValueError("Unknown recipe : %s" % recipe)

        if nodding_type not in ["AB", "ONOFF"]:
            raise ValueError("Unknown recipe : %s" % recipe)

        DO_STD = (target_type == "A0V")
        DO_AB = (nodding_type == "AB")



        from recipe_extract_base import RecipeExtractBase

        extractor = RecipeExtractBase(self.utdate, band,
                                      obsids, frametypes,
                                      self.config,
                                      ab_mode=DO_AB)

        _ = extractor.get_data_variance(destripe_pattern=64,
                                        use_destripe_mask=True,
                                        sub_horizontal_median=True)

        data_minus, variance_map, variance_map0 = _

        if self.subtract_interorder_background:
            print "### doing sky subtraction"
            data_minus_sky = extractor.estimate_interorder_background(\
                data_minus, extractor.sky_mask)

            data_minus -= data_minus_sky

        data_minus_flattened = data_minus / extractor.orderflat

        ordermap_bpixed = extractor.ordermap_bpixed

        ap = extractor.get_aperture()

        ordermap = extractor.ordermap
        slitpos_map = extractor.slitpos_map


        slitoffset_map = extractor.slitoffset_map
        #slitoffset_map_extract = "none"
        slitoffset_map_extract = slitoffset_map

        if 1:


            if IF_POINT_SOURCE: # if point source


                _ = extractor.extract_slit_profile(ap,
                                                   data_minus_flattened,
                                                   x1=800, x2=1200)
                bins, hh0, slit_profile_list = _

                if DO_AB:
                    profile_x, profile_y = extractor.get_norm_profile_ab(bins, hh0)
                else:
                    profile_x, profile_y = extractor.get_norm_profile(bins, hh0)


                self.store_profile(igr_storage,
                                   extractor.obj_filenames[0],
                                   ap.orders, slit_profile_list,
                                   profile_x, profile_y)

                if DO_AB:
                    profile = extractor.get_profile_func_ab(profile_x, profile_y)
                else:
                    profile = extractor.get_profile_func(profile_x, profile_y)

                # make weight map
                profile_map = extractor.make_profile_map(ap,
                                                         profile,
                                                         frac_slit=self.frac_slit)


                # extract spec

                shifted = extractor.get_shifted_all(ap,
                                                    profile_map,
                                                    variance_map,
                                                    data_minus_flattened,
                                                    slitoffset_map_extract,
                                                    debug=self.debug_output)

                # for khjeong
                weight_thresh = None
                remove_negative = False


                _ = extractor.extract_spec_stellar(ap, shifted,
                                                   weight_thresh,
                                                   remove_negative)


                s_list, v_list = _

                # if self.debug_output:
                #     self.save_debug_output()

                # make synth_spec : profile * spectra
                synth_map = extractor.make_synth_map(\
                    ap, profile_map, s_list,
                    ordermap=ordermap,
                    slitpos_map=slitpos_map,
                    slitoffset_map=slitoffset_map)


                # update variance map
                variance_map = extractor.get_updated_variance(\
                    variance_map, variance_map0, synth_map)

                # get cosmicray mask
                sig_map = np.abs(data_minus_flattened - synth_map)/variance_map**.5

                cr_mask = np.abs(sig_map) > self.cr_rejection_thresh

                if self.lacosmics_thresh > 0:
                    from libs.cosmics import cosmicsimage

                    cosmic_input = variance_map.copy()
                    cosmic_input[~np.isfinite(data_minus_flattened)] = np.nan
                    c = cosmicsimage(cosmic_input,
                                     readnoise=self.lacosmics_thresh)
                    c.run()
                    cr_mask_cosmics = c.getmask()

                    cr_mask = cr_mask | cr_mask_cosmics

                #variance_map[cr_mask] = np.nan


                # masking this out will affect the saved combined image.
                data_minus_flattened_orig = data_minus_flattened.copy()
                data_minus_flattened[cr_mask] = np.nan


                #profile_map_shft = profile_map

                #data_minus_flattened_shft = (data_minus_flattened)
                #variance_map_shft = (variance_map)



                # extract spec

                # extract spec


                shifted = extractor.get_shifted_all(ap, profile_map,
                                                    variance_map,
                                                    data_minus_flattened,
                                                    slitoffset_map,
                                                    debug=False)

                _ = extractor.extract_spec_stellar(ap, shifted,
                                                   weight_thresh,
                                                   remove_negative)

                s_list, v_list = _


                if 0: # save aux files
                    synth_map = ap.make_synth_map(ordermap, slitpos_map,
                                                  profile_map, s_list,
                                                  slitoffset_map=slitoffset_map
                                                  )


            else: # if extended source
                from scipy.interpolate import UnivariateSpline
                if DO_AB:
                    delta = 0.01
                    profile_ = UnivariateSpline([0, 0.5-delta, 0.5+delta, 1],
                                                [1., 1., -1., -1.],
                                                k=1, s=0,
                                                bbox=[0, 1])
                else:
                    profile_ = UnivariateSpline([0, 1], [1., 1.],
                                                k=1, s=0,
                                                bbox=[0, 1])

                def profile(o, x, slitpos):
                    return profile_(slitpos)

                profile_map = ap.make_profile_map(ordermap,
                                                  slitpos_map,
                                                  profile)


                if self.frac_slit is not None:
                    frac1, frac2 = min(self.frac_slit), max(self.frac_slit)
                    slitpos_msk = (slitpos_map < frac1) | (slitpos_map > frac2)
                    profile_map[slitpos_msk] = np.nan

                # we need to update the variance map by rejecting
                # cosmic rays, but it is not clear how we do this
                # for extended source.
                #variance_map2 = variance_map

                # detect CRs

                if self.lacosmics_thresh > 0:

                    from libs.cosmics import cosmicsimage

                    cosmic_input = variance_map.copy()
                    cosmic_input[~np.isfinite(data_minus_flattened)] = np.nan
                    c = cosmicsimage(cosmic_input,
                                     readnoise=self.lacosmics_thresh)
                    c.run()
                    cr_mask = c.getmask()

                else:
                    cr_mask = np.zeros(data_minus_flattened.shape,
                                       dtype=bool)

                #variance_map[cr_mask] = np.nan


                data_minus_flattened_orig = data_minus_flattened
                data_minus_flattened = data_minus_flattened_orig.copy()
                data_minus_flattened[cr_mask] = np.nan

                # extract spec

                shifted = extractor.get_shifted_all(ap,
                                                    profile_map,
                                                    variance_map,
                                                    data_minus_flattened,
                                                    slitoffset_map,
                                                    debug=self.debug_output)

                # for khjeong
                weight_thresh = None
                remove_negative = False

                _ = extractor.extract_spec_stellar(ap, shifted,
                                                   weight_thresh,
                                                   remove_negative)


                s_list, v_list = _

                # if self.debug_output:
                #     hdu_list = pyfits.HDUList()
                #     hdu_list.append(pyfits.PrimaryHDU(data=data_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=variance_map_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=profile_map_shft))
                #     hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
                #     #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
                #     #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
                #     hdu_list.writeto("test0.fits", clobber=True)





            if 1:
                # calculate S/N per resolution
                sn_list = []
                for wvl, s, v in zip(extractor.wvl_solutions,
                                     s_list, v_list):

                    dw = np.gradient(wvl)
                    pixel_per_res_element = (wvl/40000.)/dw
                    #print pixel_per_res_element[1024]
                    # len(pixel_per_res_element) = 2047. But we ignore it.
                    sn = (s/v**.5)*(pixel_per_res_element**.5)

                    sn_list.append(sn)



        mastername = extractor.obj_filenames[0]

        from libs.products import PipelineImage as Image
        if self.debug_output:
            image_list = [Image([("EXTNAME", "DATA_CORRECTED")],
                                data_minus_flattened),
                          Image([("EXTNAME", "DATA_UNCORRECTED")],
                                data_minus_flattened_orig),
                          Image([("EXTNAME", "VARIANCE_MAP")],
                                variance_map),
                          Image([("EXTNAME", "CR_MASK")],
                                cr_mask)]


            shifted_image_list = [Image([("EXTNAME", "DATA_CORRECTED")],
                                        shifted["data"]),
                                  Image([("EXTNAME", "VARIANCE_MAP")],
                                        shifted["variance_map"]),
                                  ]

            if IF_POINT_SOURCE: # if point source
                image_list.extend([Image([("EXTNAME", "SIG_MAP")],
                                         sig_map),
                                   Image([("EXTNAME", "SYNTH_MAP")],
                                         synth_map),
                                   ])

            if self.subtract_interorder_background:
                image_list.append(Image([("EXTNAME", "INTERORDER_BG")],
                                        data_minus_sky))

            self.store_processed_inputs(igr_storage, mastername,
                                        image_list,
                                        variance_map,
                                        shifted_image_list)



        self.store_1dspec(igr_storage,
                          extractor,
                          v_list, sn_list, s_list)

        if not IF_POINT_SOURCE: # if point source
            cr_mask = None

        self.store_2dspec(igr_storage,
                          extractor,
                          shifted["data"],
                          shifted["variance_map"],
                          ordermap_bpixed,
                          cr_mask=cr_mask)


        if DO_STD:

            wvl = self.refine_wavelength_solution(extractor.wvl_solutions)

            a0v_flattened_data = self.get_a0v_flattened(extractor, ap,
                                                        s_list,
                                                        wvl)

            a0v_flattened = a0v_flattened_data[0][1]  #"flattened_spec"]

            self.store_a0v_results(igr_storage, extractor,
                                   a0v_flattened_data)


    def refine_wavelength_solution(self, wvl_solutions):
        return wvl_solutions


    def store_profile(self, igr_storage, mastername,
                      orders, slit_profile_list,
                      profile_x, profile_y):
        ## save profile
        r = PipelineProducts("slit profile for point source")
        from libs.storage_descriptions import SLIT_PROFILE_JSON_DESC
        from libs.products import PipelineDict
        slit_profile_dict = PipelineDict(orders=orders,
                                         slit_profile_list=slit_profile_list,
                                         profile_x=profile_x,
                                         profile_y=profile_y)
        r.add(SLIT_PROFILE_JSON_DESC, slit_profile_dict)

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None)

    def save_debug_output(self, shifted):
        """
        Save debug output.
        This method currently does not work and need to be fixed.
        """
        # hdu_list = pyfits.HDUList()
        # hdu_list.append(pyfits.PrimaryHDU(data=data_minus_flattened))
        # hdu_list.append(pyfits.ImageHDU(data=variance_map))
        # hdu_list.append(pyfits.ImageHDU(data=profile_map))
        # hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
        # #hdu_list.writeto("test_input.fits", clobber=True)

        if 1:
            return

        hdu_list = pyfits.HDUList()
        hdu_list.append(pyfits.PrimaryHDU(data=data_minus))
        hdu_list.append(pyfits.ImageHDU(data=extractor.orderflat))
        hdu_list.append(pyfits.ImageHDU(data=profile_map))
        hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
        #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
        #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
        hdu_list.writeto("test0.fits", clobber=True)




        # s_list, v_list = ap.extract_stellar(ordermap_bpixed,
        #                                     profile_map_shft,
        #                                     variance_map_shft,
        #                                     data_minus_flattened_shft,
        #                                     #slitoffset_map=slitoffset_map,
        #                                     slitoffset_map=None,
        #                                     remove_negative=True
        #                                     )


    def store_processed_inputs(self, igr_storage,
                               mastername,
                               image_list,
                               variance_map,
                               shifted_image_list):

        from libs.storage_descriptions import (COMBINED_IMAGE_DESC,
                                               # COMBINED_IMAGE_A_DESC,
                                               # COMBINED_IMAGE_B_DESC,
                                               WVLCOR_IMAGE_DESC,
                                               #VARIANCE_MAP_DESC
                                               )
        from libs.products import PipelineImages #Base

        r = PipelineProducts("1d specs")

        #r.add(COMBINED_IMAGE_DESC, PipelineImageBase([], *image_list))
        r.add(COMBINED_IMAGE_DESC, PipelineImages(image_list))
        # r.add(COMBINED_IMAGE_A_DESC, PipelineImageBase([],
        #                                            a_data))
        # r.add(COMBINED_IMAGE_B_DESC, PipelineImageBase([],
        #                                            b_data))
        #r.add(VARIANCE_MAP_DESC, PipelineImageBase([],
        #                                       variance_map))

        # r.add(VARIANCE_MAP_DESC, PipelineImageBase([],
        #                                        variance_map.data))

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None)


        r = PipelineProducts("1d specs")

        r.add(WVLCOR_IMAGE_DESC, PipelineImages(shifted_image_list))

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=None)



    def get_wvl_header_data(self, igr_storage, extractor):
        from libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
        fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
                                  extractor.basenames["sky"])

        # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
        f = pyfits.open(fn)

        if self.wavelength_increasing_order:
            import libs.iraf_helper as iraf_helper
            header = iraf_helper.invert_order(f[0].header)
            convert_data = lambda d: d[::-1]
        else:
            header = f[0].header
            convert_data = lambda d: d

        return header, f[0].data, convert_data


    def store_1dspec(self, igr_storage,
                     extractor,
                     v_list, sn_list, s_list):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        f_obj = pyfits.open(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        tgt_basename = extractor.pr.tgt_basename

        from libs.storage_descriptions import (SPEC_FITS_DESC,
                                               VARIANCE_FITS_DESC,
                                               SN_FITS_DESC)



        d = np.array(v_list)
        f_obj[0].data = convert_data(d.astype("f32"))
        fout = igr_storage.get_path(VARIANCE_FITS_DESC,
                                    tgt_basename)

        f_obj.writeto(fout, clobber=True)

        d = np.array(sn_list)
        f_obj[0].data = convert_data(d.astype("f32"))
        fout = igr_storage.get_path(SN_FITS_DESC,
                                    tgt_basename)

        f_obj.writeto(fout, clobber=True)

        d = np.array(s_list)
        f_obj[0].data = convert_data(d.astype("f32"))

        fout = igr_storage.get_path(SPEC_FITS_DESC,
                                    tgt_basename)

        hdu_wvl = pyfits.ImageHDU(data=convert_data(wvl_data),
                                  header=wvl_header)
        f_obj.append(hdu_wvl)

        f_obj.writeto(fout, clobber=True)


    def store_2dspec(self, igr_storage,
                     extractor,
                     data_shft,
                     variance_map_shft,
                     ordermap_bpixed,
                     cr_mask=None):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        f_obj = pyfits.open(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        tgt_basename = extractor.pr.tgt_basename

        from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC
        cent = igr_storage.load1(FLATCENTROID_SOL_JSON_DESC,
                                 extractor.basenames["flat_on"])

        #cent = json.load(open("calib/primary/20140525/FLAT_SDCK_20140525_0074.centroid_solutions.json"))
        _bottom_up_solutions = cent["bottom_up_solutions"]
        old_orders = extractor.get_old_orders()
        _o_s = dict(zip(old_orders, _bottom_up_solutions))
        new_bottom_up_solutions = [_o_s[o] for o in \
                                   extractor.orders_w_solutions]

        from libs.correct_distortion import get_flattened_2dspec

        d0_shft_list, msk_shft_list = \
                      get_flattened_2dspec(data_shft,
                                           ordermap_bpixed,
                                           new_bottom_up_solutions)


        d = np.array(d0_shft_list) / np.array(msk_shft_list)
        f_obj[0].data = convert_data(d.astype("f32"))

        from libs.storage_descriptions import SPEC2D_FITS_DESC

        fout = igr_storage.get_path(SPEC2D_FITS_DESC,
                                    tgt_basename)

        hdu_wvl = pyfits.ImageHDU(data=convert_data(wvl_data),
                                  header=wvl_header)
        f_obj.append(hdu_wvl)

        f_obj.writeto(fout, clobber=True)

        #OUTPUT VAR2D, added by Kyle Kaplan Feb 25, 2015 to get variance map outputted as a datacube
        d0_shft_list, msk_shft_list = \
                      get_flattened_2dspec(variance_map_shft,
                                           ordermap_bpixed,
                                           new_bottom_up_solutions)
        d = np.array(d0_shft_list) / np.array(msk_shft_list)
        f_obj[0].data = d.astype("f32")
        from libs.storage_descriptions import VAR2D_FITS_DESC
        fout = igr_storage.get_path(VAR2D_FITS_DESC,
                                    tgt_basename)
        f_obj.writeto(fout, clobber=True)


    def get_tel_interp1d_f(self, extractor, wvl_solutions):

        from libs.master_calib import get_master_calib_abspath
        #fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
        #self.telluric = pyfits.open(fn)[1].data

        telfit_outname = "telluric/transmission-795.20-288.30-41.9-45.0-368.50-3.90-1.80-1.40.%s" % extractor.band
        telfit_outname_npy = telfit_outname+".npy"
        if 0:
            dd = np.genfromtxt(telfit_outname)
            np.save(open(telfit_outname_npy, "w"), dd[::10])

        from libs.a0v_flatten import TelluricTransmission
        _fn = get_master_calib_abspath(telfit_outname_npy)
        tel_trans = TelluricTransmission(_fn)

        wvl_solutions = np.array(wvl_solutions)

        w_min = wvl_solutions.min()*0.9
        w_max = wvl_solutions.max()*1.1
        def tel_interp1d_f(gw=None):
            return tel_trans.get_telluric_trans_interp1d(w_min, w_max, gw)

        return tel_interp1d_f

    def get_a0v_interp1d(self, extractor):

        from libs.a0v_spec import A0VSpec
        a0v_model = A0VSpec()
        a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                                   flatten=True,
                                                   smooth_pixel=32)
        return a0v_interp1d


    def get_a0v_flattened(self, extractor, ap,
                          s_list, wvl):

        tel_interp1d_f = self.get_tel_interp1d_f(extractor, wvl)
        a0v_interp1d = self.get_a0v_interp1d(extractor)


        orderflat_response = extractor.orderflat_json["fitted_responses"]

        tgt_basename = extractor.pr.tgt_basename
        igr_path = extractor.igr_path
        figout = igr_path.get_section_filename_base("QA_PATH",
                                                    "flattened_"+tgt_basename) + ".pdf"

        from libs.a0v_flatten import get_a0v_flattened
        data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                                      wvl, s_list, orderflat_response,
                                      figout=figout)

        if self.fill_nan is not None:
            flattened_s = data_list[0][1]
            flattened_s[~np.isfinite(flattened_s)] = self.fill_nan

        return data_list


    def get_a0v_flattened_deprecated(self, igr_storage, extractor, ap,
                          s_list):

        from libs.storage_descriptions import ORDER_FLAT_JSON_DESC
        prod = igr_storage.load1(ORDER_FLAT_JSON_DESC,
                                 extractor.basenames["flat_on"])

        new_orders = prod["orders"]
        # fitted_response = orderflat_products["fitted_responses"]
        i1i2_list_ = prod["i1i2_list"]

        #order_indices = []
        i1i2_list = []

        for o in ap.orders:
            o_new_ind = np.searchsorted(new_orders, o)
            #order_indices.append(o_new_ind)
            i1i2_list.append(i1i2_list_[o_new_ind])


        from libs.a0v_spec import (A0VSpec, TelluricTransmission,
                                   get_a0v, get_flattend)
        a0v_spec = A0VSpec()
        tel_trans = TelluricTransmission()

        wvl_limits = []
        for wvl_ in extractor.wvl_solutions:
            wvl_limits.extend([wvl_[0], wvl_[-1]])

        dwvl = abs(wvl_[0] - wvl_[-1])*0.2 # padding

        wvl1 = min(wvl_limits) - dwvl
        wvl2 = max(wvl_limits) + dwvl

        a0v_wvl, a0v_tel_trans, a0v_tel_trans_masked = get_a0v(a0v_spec, wvl1, wvl2, tel_trans)

        a0v_flattened = get_flattend(a0v_spec,
                                     a0v_wvl, a0v_tel_trans_masked,
                                     extractor.wvl_solutions, s_list,
                                     i1i2_list=i1i2_list)


        return a0v_flattened

    def store_a0v_results(self, igr_storage, extractor,
                          a0v_flattened_data):

        wvl_header, wvl_data, convert_data = \
                    self.get_wvl_header_data(igr_storage,
                                             extractor)


        f_obj = pyfits.open(extractor.obj_filenames[0])
        f_obj[0].header.extend(wvl_header)

        from libs.products import PipelineImage as Image
        image_list = [Image([("EXTNAME", "SPEC_FLATTENED")],
                            convert_data(a0v_flattened_data[0][1]))]
        if self.debug_output:
            for ext_name, data in a0v_flattened_data[1:]:
                image_list.append(Image([("EXTNAME", ext_name.upper())],
                                        convert_data(data)))


        from libs.products import PipelineImages #Base
        from libs.storage_descriptions import SPEC_FITS_FLATTENED_DESC

        r = PipelineProducts("flattened 1d specs")
        r.add(SPEC_FITS_FLATTENED_DESC, PipelineImages(image_list))

        mastername = extractor.obj_filenames[0]

        igr_storage.store(r,
                          mastername=mastername,
                          masterhdu=f_obj[0])

        tgt_basename = extractor.pr.tgt_basename
        extractor.db["a0v"].update(extractor.band, tgt_basename)
