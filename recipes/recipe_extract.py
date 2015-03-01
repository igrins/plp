import os
import numpy as np

from libs.path_info import IGRINSPath
import astropy.io.fits as pyfits

from libs.products import PipelineProducts

def extractor_factory(recipe_name):
    def extract(utdate, refdate="20140316", bands="HK",
                starting_obsids=None,
                config_file="recipe.config",
                frac_slit=None,
                cr_rejection_thresh=100,
                debug_output=False):
        abba_all(recipe_name, utdate, refdate=refdate, bands=bands,
                 starting_obsids=starting_obsids,
                 config_file=config_file,
                 frac_slit=frac_slit,
                 cr_rejection_thresh=cr_rejection_thresh,
                 debug_output=debug_output)

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
             debug_output=False):

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

    process_abba_band = ProcessABBABand(utdate, refdate,
                                        config,
                                        frac_slit=frac_slit,
                                        cr_rejection_thresh=cr_rejection_thresh,
                                        debug_output=debug_output).process

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
                 debug_output=False):
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
        self.debug_output = debug_output

    def process(self, recipe, band, obsids, frametypes):

        igr_path = self.igr_path
        igr_storage = self.igr_storage


        target_type, nodding_type = recipe.split("_")

        if target_type in ["A0V", "STELLAR"]:
            IF_POINT_SOURCE = True
        elif target_type in ["EXTENDED"]:
            IF_POINT_SOURCE = False
        else:
            raise ValueError("Unknown recipe : %s" % recipe)

        DO_STD = (target_type == "A0V")
        DO_AB = (nodding_type == "AB")

        if 1:

            obj_filenames = igr_path.get_filenames(band, obsids)

            master_obsid = obsids[0]

            tgt_basename = os.path.splitext(os.path.basename(obj_filenames[0]))[0]

            db = {}
            basenames = {}

            db_types = ["flat_off", "flat_on", "thar", "sky"]

            for db_type in db_types:

                db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                            "%s.db" % db_type,
                                                            )
                db[db_type] = ProductDB(db_name)


            # db on output path
            db_types = ["a0v"]

            for db_type in db_types:

                db_name = igr_path.get_section_filename_base("OUTDATA_PATH",
                                                            "%s.db" % db_type,
                                                            )
                db[db_type] = ProductDB(db_name)

            # to get basenames
            db_types = ["flat_off", "flat_on", "thar", "sky"]
            # if FIX_TELLURIC:
            #     db_types.append("a0v")

            for db_type in db_types:
                basenames[db_type] = db[db_type].query(band, master_obsid)



        if 1: # make aperture
            from libs.storage_descriptions import SKY_WVLSOL_JSON_DESC

            sky_basename = db["sky"].query(band, master_obsid)
            wvlsol_products = igr_storage.load([SKY_WVLSOL_JSON_DESC],
                                               sky_basename)[SKY_WVLSOL_JSON_DESC]

            orders_w_solutions = wvlsol_products["orders"]
            wvl_solutions = map(np.array, wvlsol_products["wvl_sol"])

            from libs.storage_descriptions import ONED_SPEC_JSON_DESC

            raw_spec_products = igr_storage.load([ONED_SPEC_JSON_DESC],
                                                 sky_basename)

            from recipe_wvlsol_sky import load_aperture2

            old_orders = raw_spec_products[ONED_SPEC_JSON_DESC]["orders"]
            ap = load_aperture2(igr_storage, band, master_obsid,
                                db["flat_on"],
                                old_orders,
                                orders_w_solutions)


            from libs.storage_descriptions import (ORDERMAP_FITS_DESC,
                                                   SLITPOSMAP_FITS_DESC)

            order_map = igr_storage.load1(ORDERMAP_FITS_DESC, sky_basename).data
            slitpos_map = igr_storage.load1(SLITPOSMAP_FITS_DESC, sky_basename).data
            #order_map = ap.make_order_map()
            #slitpos_map = ap.make_slitpos_map()
            order_map2 = ap.make_order_map(mask_top_bottom=True)

        if 1:

            from libs.storage_descriptions import (HOTPIX_MASK_DESC,
                                                   DEADPIX_MASK_DESC,
                                                   ORDER_FLAT_IM_DESC,
                                                   ORDER_FLAT_JSON_DESC,
                                                   FLAT_MASK_DESC)

            hotpix_mask = igr_storage.load([HOTPIX_MASK_DESC],
                                           basenames["flat_off"])[HOTPIX_MASK_DESC]

            deadpix_mask = igr_storage.load([DEADPIX_MASK_DESC],
                                            basenames["flat_on"])[DEADPIX_MASK_DESC]

            pix_mask  = hotpix_mask.data | deadpix_mask.data



            # aperture_solution_products = PipelineProducts.load(aperture_solutions_name)


            orderflat_ = igr_storage.load([ORDER_FLAT_IM_DESC],
                                         basenames["flat_on"])[ORDER_FLAT_IM_DESC]


            orderflat = orderflat_.data
            orderflat[pix_mask] = np.nan

            orderflat_json = igr_storage.load([ORDER_FLAT_JSON_DESC],
                                              basenames["flat_on"])[ORDER_FLAT_JSON_DESC]
            order_flat_meanspec = np.array(orderflat_json["mean_order_specs"])

            # flat_normed = igr_storage.load([FLAT_NORMED_DESC],
            #                                basenames["flat_on"])[FLAT_NORMED_DESC]

            flat_mask = igr_storage.load([FLAT_MASK_DESC],
                                         basenames["flat_on"])[FLAT_MASK_DESC]
            bias_mask = flat_mask.data & (order_map2 > 0)

            SLITOFFSET_FITS_DESC = ("PRIMARY_CALIB_PATH", "SKY_", ".slitoffset_map.fits")
            prod_ = igr_storage.load([SLITOFFSET_FITS_DESC],
                                     basenames["sky"])[SLITOFFSET_FITS_DESC]
            #fn = sky_path.get_secondary_path("slitoffset_map.fits")
            slitoffset_map = prod_.data

        if 1:

            abba_names = obj_filenames

            def filter_abba_names(abba_names, frametypes, frametype):
                return [an for an, ft in zip(abba_names, frametypes) if ft == frametype]


            a_name_list = filter_abba_names(abba_names, frametypes, "A")
            b_name_list = filter_abba_names(abba_names, frametypes, "B")

            if 1:
                #ab_names = ab_names_list[0]

                f_obj = pyfits.open(obj_filenames[0])


                a_list = [pyfits.open(name)[0].data \
                          for name in a_name_list]
                b_list = [pyfits.open(name)[0].data \
                          for name in b_name_list]


                # we may need to detrip

                # first define extract profile (gaussian).


                # dx = 100

                if DO_AB:
                    # for point sources, variance estimation becomes wrong
                    # if lenth of two is different,
                    if len(a_list) != len(b_list):
                        raise RuntimeError("For AB nodding, number of A and B should match!")

                # a_b != 1 for the cases when len(a) != len(b)
                a_b = float(len(a_list)) / len(b_list)

                a_data = np.sum(a_list, axis=0)
                b_data = np.sum(b_list, axis=0)

                from libs.destriper import destriper

                data_minus = a_data - a_b*b_data
                #data_minus0 = data_minus

                destrip_mask = ~np.isfinite(data_minus)|bias_mask

                data_minus = destriper.get_destriped(data_minus,
                                                     destrip_mask,
                                                     pattern=64,
                                                     hori=True)
                if self.debug_output:
                    a_data = destriper.get_destriped(a_data,
                                                     destrip_mask,
                                                     pattern=64)
                    b_data = destriper.get_destriped(b_data,
                                                     destrip_mask,
                                                     pattern=64)

                data_minus_flattened = data_minus / orderflat
                data_minus_flattened[~flat_mask.data] = np.nan
                #data_minus_flattened[order_flat_meanspec<0.1*order_flat_meanspec.max()] = np.nan


                # for variance, we need a square of a_b
                data_plus = (a_data + (a_b**2)*b_data)

                import scipy.ndimage as ni
                bias_mask2 = ni.binary_dilation(bias_mask)

                from libs import instrument_parameters
                gain =  instrument_parameters.gain[band]

                from libs.variance_map import get_variance_map
                variance_map = get_variance_map(data_plus, data_minus,
                                                bias_mask2, pix_mask, gain)


                # now estimate lsf


                # estimate lsf
                ordermap_bpixed = order_map.copy()
                ordermap_bpixed[pix_mask] = 0
                ordermap_bpixed[~np.isfinite(orderflat)] = 0
            #


            if IF_POINT_SOURCE: # if point source



                x1, x2 = 800, 1200
                bins, slit_profile_list = \
                      ap.extract_slit_profile(ordermap_bpixed,
                                              slitpos_map,
                                              data_minus_flattened,
                                              x1, x2, bins=None)


                hh0 = np.sum(slit_profile_list, axis=0)
                if DO_AB:
                    peak1, peak2 = max(hh0), -min(hh0)
                    profile_x = 0.5*(bins[1:]+bins[:-1])
                    profile_y = hh0/(peak1+peak2)
                else:
                    peak1 = max(hh0)
                    profile_x = 0.5*(bins[1:]+bins[:-1])
                    profile_y = hh0/peak1

                ## save profile
                r = PipelineProducts("slit profile for point source")
                from libs.storage_descriptions import SLIT_PROFILE_JSON_DESC
                from libs.products import PipelineDict
                slit_profile_dict = PipelineDict(orders=ap.orders,
                                                 slit_profile_list=slit_profile_list,
                                                 profile_x=profile_x,
                                                 profile_y=profile_y)
                r.add(SLIT_PROFILE_JSON_DESC, slit_profile_dict)

                igr_storage.store(r,
                                  mastername=obj_filenames[0],
                                  masterhdu=None)

                #

                from scipy.interpolate import UnivariateSpline
                profile_ = UnivariateSpline(profile_x, profile_y, k=3, s=0,
                                            bbox=[0, 1])

                if DO_AB:
                    roots = list(profile_.roots())
                    #assert(len(roots) == 1)
                    integ_list = []
                    from itertools import izip, cycle
                    for ss, int_r1, int_r2 in izip(cycle([1, -1]),
                                                   [0] + roots,
                                                   roots + [1]):
                        #print ss, int_r1, int_r2
                        integ_list.append(profile_.integral(int_r1, int_r2))
                    integ = np.abs(np.sum(integ_list))
                else:
                    integ = profile_.integral(0, 1)

                def profile(o, x, slitpos):
                    return profile_(slitpos) / integ

                # make weight map
                profile_map = ap.make_profile_map(order_map, slitpos_map,
                                                  profile)

                # try to select portion of the slit to extract

                if self.frac_slit is not None:
                    frac1, frac2 = min(self.frac_slit), max(self.frac_slit)
                    slitpos_msk = (slitpos_map < frac1) | (slitpos_map > frac2)
                    profile_map[slitpos_msk] = np.nan



                #profile_map_shft = profile_map

                #data_minus_flattened_shft = (data_minus_flattened)
                #variance_map_shft = (variance_map)


                # extract spec


                _ = ap.get_shifted_images(profile_map,
                                          variance_map,
                                          data_minus_flattened,
                                          slitoffset_map=slitoffset_map,
                                          debug=self.debug_output)


                # hdu_list = pyfits.HDUList()
                # hdu_list.append(pyfits.PrimaryHDU(data=data_minus_flattened))
                # hdu_list.append(pyfits.ImageHDU(data=variance_map))
                # hdu_list.append(pyfits.ImageHDU(data=profile_map))
                # hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
                # #hdu_list.writeto("test_input.fits", clobber=True)

                data_shft, variance_map_shft, profile_map_shft, msk1_shft = _

                _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
                                                    profile_map_shft,
                                                    variance_map_shft,
                                                    data_shft, msk1_shft,
                                                    remove_negative=True)
                s_list, v_list = _

                if self.debug_output:
                    hdu_list = pyfits.HDUList()
                    hdu_list.append(pyfits.PrimaryHDU(data=data_shft))
                    hdu_list.append(pyfits.ImageHDU(data=variance_map_shft))
                    hdu_list.append(pyfits.ImageHDU(data=profile_map_shft))
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

                ### save test1




                # make synth_spec : profile * spectra
                synth_map = ap.make_synth_map(order_map,
                                              slitpos_map,
                                              profile_map, s_list,
                                              slitoffset_map=slitoffset_map
                                              )

                sig_map = (data_minus_flattened - synth_map)**2/variance_map


                # reextract with new variance map and CR is rejected
                variance_map_r = variance_map + np.abs(synth_map)/gain
                variance_map = np.max([variance_map, variance_map_r], axis=0)

                cr_mask = np.abs(sig_map) > self.cr_rejection_thresh
                variance_map[cr_mask] = np.nan

                ## mark sig_map > 100 as cosmicay. The threshold need to be fixed.

                # masking this out will affect the saved combined image.
                data_minus_flattened_orig = data_minus_flattened.copy()
                data_minus_flattened[cr_mask] = np.nan


                #profile_map_shft = profile_map

                #data_minus_flattened_shft = (data_minus_flattened)
                #variance_map_shft = (variance_map)



                # extract spec

                # extract spec

                _ = ap.get_shifted_images(profile_map,
                                          variance_map,
                                          data_minus_flattened,
                                          slitoffset_map=slitoffset_map)

                data_shft, variance_map_shft, profile_map_shft, msk1_shft = _

                _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
                                                    profile_map_shft,
                                                    variance_map_shft,
                                                    data_shft, msk1_shft,
                                                    remove_negative=True)
                s_list, v_list = _


                if 0: # save aux files
                    synth_map = ap.make_synth_map(order_map, slitpos_map,
                                                  profile_map, s_list,
                                                  slitoffset_map=slitoffset_map
                                                  )


            else: # if extended source
                from scipy.interpolate import UnivariateSpline
                if recipe in ["EXTENDED_AB", "EXTENDED_ABBA"]:
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

                profile_map = ap.make_profile_map(order_map, slitpos_map,
                                                  profile)


                if self.frac_slit is not None:
                    frac1, frac2 = min(self.frac_slit), max(self.frac_slit)
                    slitpos_msk = (slitpos_map < frac1) | (slitpos_map > frac2)
                    profile_map[slitpos_msk] = np.nan

                # we need to update the variance map by rejecting
                # cosmic rays, but it is not clear how we do this
                # for extended source.
                #variance_map2 = variance_map


                # extract spec

                _ = ap.get_shifted_images(profile_map,
                                          variance_map,
                                          data_minus_flattened,
                                          slitoffset_map=slitoffset_map,
                                          debug=True)

                data_shft, variance_map_shft, profile_map_shft, msk1_shft = _

                if self.debug_output:
                    hdu_list = pyfits.HDUList()
                    hdu_list.append(pyfits.PrimaryHDU(data=data_shft))
                    hdu_list.append(pyfits.ImageHDU(data=variance_map_shft))
                    hdu_list.append(pyfits.ImageHDU(data=profile_map_shft))
                    hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
                    #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
                    #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
                    hdu_list.writeto("test0.fits", clobber=True)

                _ = ap.extract_extended_from_shifted(ordermap_bpixed,
                                                     profile_map_shft,
                                                     variance_map_shft,
                                                     data_shft, msk1_shft,
                                                     slitpos_map,
                                                     remove_negative=True)
                s_list, v_list = _



            if 1:
                # calculate S/N per resolution
                sn_list = []
                for wvl, s, v in zip(wvl_solutions,
                                     s_list, v_list):

                    dw = np.gradient(wvl)
                    pixel_per_res_element = (wvl/40000.)/dw
                    #print pixel_per_res_element[1024]
                    # len(pixel_per_res_element) = 2047. But we ignore it.
                    sn = (s/v**.5)*(pixel_per_res_element**.5)

                    sn_list.append(sn)



        if 1: # save the product
            from libs.storage_descriptions import (COMBINED_IMAGE_DESC,
                                                   COMBINED_IMAGE_A_DESC,
                                                   COMBINED_IMAGE_B_DESC,
                                                   VARIANCE_MAP_DESC)
            from libs.products import PipelineImage

            r = PipelineProducts("1d specs")

            image_list = [data_minus_flattened]
            if IF_POINT_SOURCE: # if point source
                image_list.extend([data_minus_flattened_orig,
                                   synth_map,
                                   sig_map,
                                   cr_mask,
                                   ])
            r.add(COMBINED_IMAGE_DESC, PipelineImage([], *image_list))
            r.add(COMBINED_IMAGE_A_DESC, PipelineImage([],
                                                       a_data))
            r.add(COMBINED_IMAGE_B_DESC, PipelineImage([],
                                                       b_data))
            r.add(VARIANCE_MAP_DESC, PipelineImage([],
                                                   variance_map))

            # r.add(VARIANCE_MAP_DESC, PipelineImage([],
            #                                        variance_map.data))

            igr_storage.store(r,
                              mastername=obj_filenames[0],
                              masterhdu=None)



        if 1: # save spectra, variance, sn
            from libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
            fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
                                      basenames["sky"])

            # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
            f = pyfits.open(fn)

            f_obj[0].header.extend(f[0].header)

            from libs.storage_descriptions import (SPEC_FITS_DESC,
                                                   VARIANCE_FITS_DESC,
                                                   SN_FITS_DESC)



            d = np.array(v_list)
            f_obj[0].data = d.astype("f32")
            fout = igr_storage.get_path(VARIANCE_FITS_DESC,
                                        tgt_basename)

            f_obj.writeto(fout, clobber=True)

            d = np.array(sn_list)
            f_obj[0].data = d.astype("f32")
            fout = igr_storage.get_path(SN_FITS_DESC,
                                        tgt_basename)

            f_obj.writeto(fout, clobber=True)

            d = np.array(s_list)
            f_obj[0].data = d.astype("f32")

            fout = igr_storage.get_path(SPEC_FITS_DESC,
                                        tgt_basename)

            hdu_wvl = pyfits.ImageHDU(data=f[0].data, header=f[0].header)
            f_obj.append(hdu_wvl)

            f_obj.writeto(fout, clobber=True)


            if not IF_POINT_SOURCE: # if extended source
                from libs.storage_descriptions import FLATCENTROID_SOL_JSON_DESC
                cent = igr_storage.load1(FLATCENTROID_SOL_JSON_DESC,
                                         basenames["flat_on"])

                #cent = json.load(open("calib/primary/20140525/FLAT_SDCK_20140525_0074.centroid_solutions.json"))
                _bottom_up_solutions = cent["bottom_up_solutions"]
                old_orders
                _o_s = dict(zip(old_orders, _bottom_up_solutions))
                new_bottom_up_solutions = [_o_s[o] for o in orders_w_solutions]

                from libs.correct_distortion import get_flattened_2dspec
                d0_shft_list, msk_shft_list = \
                              get_flattened_2dspec(data_shft,
                                                   ordermap_bpixed,
                                                   new_bottom_up_solutions)


                d = np.array(d0_shft_list) / np.array(msk_shft_list)
                f_obj[0].data = d.astype("f32")

                from libs.storage_descriptions import SPEC2D_FITS_DESC

                fout = igr_storage.get_path(SPEC2D_FITS_DESC,
                                            tgt_basename)

                f_obj.writeto(fout, clobber=True)



        if 1: #
            from libs.storage_descriptions import ORDER_FLAT_JSON_DESC
            prod = igr_storage.load([ORDER_FLAT_JSON_DESC],
                                    basenames["flat_on"])[ORDER_FLAT_JSON_DESC]

            new_orders = prod["orders"]
            # fitted_response = orderflat_products["fitted_responses"]
            i1i2_list_ = prod["i1i2_list"]

            #order_indices = []
            i1i2_list = []

            for o in ap.orders:
                o_new_ind = np.searchsorted(new_orders, o)
                #order_indices.append(o_new_ind)
                i1i2_list.append(i1i2_list_[o_new_ind])


            if DO_STD:
                from libs.a0v_spec import (A0VSpec, TelluricTransmission,
                                           get_a0v, get_flattend)
                a0v_spec = A0VSpec()
                tel_trans = TelluricTransmission()

                wvl_limits = []
                for wvl_ in wvl_solutions:
                    wvl_limits.extend([wvl_[0], wvl_[-1]])

                dwvl = abs(wvl_[0] - wvl_[-1])*0.2 # padding

                wvl1 = min(wvl_limits) - dwvl
                wvl2 = max(wvl_limits) + dwvl

                a0v_wvl, a0v_tel_trans, a0v_tel_trans_masked = get_a0v(a0v_spec, wvl1, wvl2, tel_trans)

                a0v_flattened = get_flattend(a0v_spec,
                                             a0v_wvl, a0v_tel_trans_masked,
                                             wvl_solutions, s_list,
                                             i1i2_list=i1i2_list)

                d = np.array(a0v_flattened)
                #d[~np.isfinite(d)] = 0.
                f[0].data = d.astype("f32")

                from libs.storage_descriptions import SPEC_FITS_FLATTENED_DESC
                fout = igr_storage.get_path(SPEC_FITS_FLATTENED_DESC,
                                            tgt_basename)

                f.writeto(fout, clobber=True)

                db["a0v"].update(band, tgt_basename)
