import numpy as np
import libs.fits as pyfits
import scipy.ndimage as ni

from libs.lazyprop import lazyprop

import os
from libs.products import ProductDB, PipelineStorage

class ProcessBase(object):
    def __init__(self, utdate, refdate, config):
        """
        cr_rejection_thresh : pixels that deviate significantly from the profile are excluded.
        """
        self.utdate = utdate
        self.refdate = refdate
        self.config = config

        self.igr_storage = PipelineStorage.from_utdate(utdate,
                                                       config=config)
        self.igr_path = self.igr_storage.igr_path

    def prepare(self, band, obsids, load_a0v_db=True):

        igr_path = self.igr_path

        self.obj_filenames = igr_path.get_filenames(band, obsids)

        self.master_obsid = obsids[0]

        self.tgt_basename = os.path.splitext(os.path.basename(self.obj_filenames[0]))[0]

        self.db = {}
        self.basenames = {}


        db_types_calib = ["flat_off", "flat_on", "thar", "sky"]

        for db_type in db_types_calib:

            db_name = igr_path.get_section_filename_base("PRIMARY_CALIB_PATH",
                                                         "%s.db" % db_type,
                                                         )
            self.db[db_type] = ProductDB(db_name)


        # db on output path
        db_types = ["a0v"]

        for db_type in db_types:

            db_name = igr_path.get_section_filename_base("OUTDATA_PATH",
                                                         "%s.db" % db_type,
                                                         )
            #if os.path.exists(db_name):
            self.db[db_type] = ProductDB(db_name)

        # to get basenames
        db_types = ["flat_off", "flat_on", "thar", "sky"]
        if load_a0v_db:
            db_types.append("a0v")

        for db_type in db_types:
            if db_type in self.db:
                self.basenames[db_type] = self.db[db_type].query(band,
                                                                 self.master_obsid)

def get_pr(utdate, config_file="recipe.config"):
    from libs.igrins_config import IGRINSConfig
    #from jj_recipe_base import ProcessBase
    config = IGRINSConfig(config_file)
    #refdate = config.get_value("REFDATE", None)
    refdate = config.get("MASTER_CAL", "REFDATE")
    pr = ProcessBase(utdate, refdate, config)

    return pr



class RecipeExtractPR(object):
    def load1(self, db_name, description):
        import libs.storage_descriptions
        master_obsid = self.pr.master_obsid
        basename = self.db[db_name].query(self.band, master_obsid)
        desc = getattr(libs.storage_descriptions,
                       description.upper() + "_DESC")
        product = self.igr_storage.load1(desc,
                                         basename)
        return product
    @property
    def igr_storage(self):
        return self.pr.igr_storage

    @property
    def igr_path(self):
        return self.pr.igr_path

    @property
    def tgt_basename(self):
        return self.pr.tgt_basename

    @property
    def basenames(self):
        return self.pr.basenames

    @property
    def obj_filenames(self):
        return self.pr.obj_filenames

    @property
    def db(self):
        return self.pr.db

    @property
    def gain(self):
        from libs import instrument_parameters
        gain =  instrument_parameters.gain[self.band]
        return gain

    @lazyprop
    def _orders_and_wvl_solutions(self):
        from libs.storage_descriptions import SKY_WVLSOL_JSON_DESC

        sky_basename = self.basenames["sky"]
        wvlsol_products = self.igr_storage.load1(SKY_WVLSOL_JSON_DESC,
                                                 sky_basename)

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = map(np.array, wvlsol_products["wvl_sol"])

        return orders_w_solutions, wvl_solutions

    @property
    def wvl_solutions(self):
        return self._orders_and_wvl_solutions[1]

    @property
    def orders_w_solutions(self):
        return self._orders_and_wvl_solutions[0]

    @lazyprop
    def orderflat(self):

        from libs.storage_descriptions import ORDER_FLAT_IM_DESC

        orderflat_ = self.igr_storage.load1(ORDER_FLAT_IM_DESC,
                                            self.basenames["flat_on"])

        orderflat = orderflat_.data
        orderflat[self.pix_mask] = np.nan

        return orderflat

    @lazyprop
    def orderflat_json(self):

        from libs.storage_descriptions import ORDER_FLAT_JSON_DESC

        orderflat = self.igr_storage.load1(ORDER_FLAT_JSON_DESC,
                                           self.basenames["flat_on"])

        return orderflat


    @lazyprop
    def ordermap(self):

        sky_basename = self.basenames["sky"]

        from libs.storage_descriptions import ORDERMAP_FITS_DESC

        ordermap = self.igr_storage.load1(ORDERMAP_FITS_DESC,
                                          sky_basename).data

        return ordermap

    @lazyprop
    def ordermap_bpixed(self):

        ordermap_bpixed = self.ordermap.copy()
        ordermap_bpixed[self.pix_mask] = 0
        ordermap_bpixed[~np.isfinite(self.orderflat)] = 0

        return ordermap_bpixed


    @lazyprop
    def pix_mask(self):
        from libs.storage_descriptions import (HOTPIX_MASK_DESC,
                                               DEADPIX_MASK_DESC)

        hotpix_mask = self.igr_storage.load1(HOTPIX_MASK_DESC,
                                             self.basenames["flat_off"])

        deadpix_mask = self.igr_storage.load1(DEADPIX_MASK_DESC,
                                              self.basenames["flat_on"])

        pix_mask  = hotpix_mask.data | deadpix_mask.data

        return pix_mask


    @lazyprop
    def slitpos_map(self):
        sky_basename = self.basenames["sky"]

        from libs.storage_descriptions import SLITPOSMAP_FITS_DESC

        slitpos_map = self.igr_storage.load1(SLITPOSMAP_FITS_DESC,
                                             sky_basename).data

        return slitpos_map

    @lazyprop
    def slitoffset_map(self):
        from libs.storage_descriptions import SLITOFFSET_FITS_DESC
        prod_ = self.igr_storage.load1(SLITOFFSET_FITS_DESC,
                                       self.basenames["sky"])

        slitoffset_map = prod_.data
        return slitoffset_map


    @lazyprop
    def destripe_mask(self):
        from libs.storage_descriptions import BIAS_MASK_DESC
        bias_mask = self.igr_storage.load1(BIAS_MASK_DESC,
                                           self.basenames["flat_on"]).data
        #bias_mask[-100:,:] = False
        bias_mask[self.pix_mask] = True
        bias_mask[:4] = True
        bias_mask[-4:] = True

        destripe_mask = bias_mask
        return destripe_mask


    def __init__(self, utdate, band, obsids,
                 config,
                 load_a0v_db=True):
        #self.pr = get_pr(utdate=utdate, config=config)

        #refdate = config.get_value("REFDATE", None)
        refdate = config.get("MASTER_CAL", "REFDATE")
        self.pr = ProcessBase(utdate, refdate, config)
        self.pr.prepare(band, obsids,
                        load_a0v_db=load_a0v_db) #[32], ["A"])
        self.band = band


    def get_oned_spec_helper(self, basename=None):
        igr_storage = self.igr_storage
        if basename is None:
            basename = self.pr.tgt_basename

        from libs.spec_helper import OnedSpecHelper
        oned_spec_helper = OnedSpecHelper(igr_storage,
                                          basename)
        return oned_spec_helper

class RecipeExtractBase(RecipeExtractPR):

    def __init__(self, utdate, band, obsids, frametypes,
                 config=None,
                 ab_mode=True, load_a0v_db=False):
        """
        ab_mode : True if nodding tye is 'ABBA' or its variation.
        """

        if config is None or isinstance(config, str):
            from libs.igrins_config import IGRINSConfig
            config = IGRINSConfig(config)

        RecipeExtractPR.__init__(self,
                                 utdate, band, obsids,
                                 config,
                                 load_a0v_db=load_a0v_db)

        self.frametypes = frametypes

        self.ab_mode = ab_mode

        #orderflat_json = igr_storage.load1(ORDER_FLAT_JSON_DESC,
        #                                   basenames["flat_on"])
        #fitted_response = orderflat_json["fitted_responses"]

    @lazyprop
    def shiftx(self):
        from libs.correct_distortion import ShiftX
        return ShiftX(self.slitoffset_map)


    @lazyprop
    def sky_mask(self):

        msk_ = (self.ordermap_bpixed > 0) | self.destripe_mask

        msk = msk_
        msk[self.pix_mask] = True

        msk[:4] = True
        msk[-4:] = True
        msk[:,:4] = True
        msk[:,-4:] = True

        return msk

    def estimate_interorder_background(self, data, msk, di=24, min_pixel=40):

        from libs.estimate_sky import (estimate_background,
                                       get_interpolated_cubic)

        xc, yc, v, std = estimate_background(data, msk,
                                             di=di, min_pixel=min_pixel)

        nx = ny = 2048
        ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)

        return ZI3

    def get_data_variance(self,
                          destripe_pattern=64,
                          use_destripe_mask=True,
                          sub_horizontal_median=True):

        abba_names = self.obj_filenames
        frametypes = self.frametypes

        def filter_abba_names(abba_names, frametypes, frametype):
            return [an for an, ft in zip(abba_names, frametypes) if ft == frametype]


        a_name_list = filter_abba_names(abba_names, frametypes, "A")
        b_name_list = filter_abba_names(abba_names, frametypes, "B")


        a_list = [pyfits.open(name)[0].data \
                  for name in a_name_list]
        b_list = [pyfits.open(name)[0].data \
                  for name in b_name_list]


        if self.ab_mode:
            # for point sources, variance estimation becomes wrong
            # if lenth of two is different,
            if len(a_list) != len(b_list):
                raise RuntimeError("For AB nodding, number of A and B should match!")

        # a_b != 1 for the cases when len(a) != len(b)
        a_b = float(len(a_list)) / len(b_list)

        a_data = np.sum(a_list, axis=0)
        b_data = np.sum(b_list, axis=0)


        data_minus = a_data - a_b*b_data
        #data_minus0 = data_minus

        if destripe_pattern is not None:
            data_minus = self.get_destriped(data_minus,
                                            destripe_pattern=destripe_pattern,
                                            use_destripe_mask=use_destripe_mask,
                                            sub_horizontal_median=sub_horizontal_median)

        # if use_destripe_mask:
        #     destrip_mask = ~np.isfinite(data_minus)|self.destripe_mask
        # else:
        #     destrip_mask = None

        # data_minus = destriper.get_destriped(data_minus,
        #                                      destrip_mask,
        #                                      pattern=destripe_pattern,
        #                                      hori=sub_horizontal_median)

        # remove sky

        # now estimate variance_map

        data_plus = (a_data + (a_b**2)*b_data)

        import scipy.ndimage as ni
        bias_mask2 = ni.binary_dilation(self.destripe_mask)

        from libs.variance_map import (get_variance_map,
                                       get_variance_map0)

        variance_map0 = get_variance_map0(data_minus,
                                          bias_mask2, self.pix_mask)

        variance_map = get_variance_map(data_plus, variance_map0,
                                        gain=self.gain)


        return data_minus, variance_map, variance_map0


    def get_destriped(self, data_minus,
                      destripe_pattern=64,
                      use_destripe_mask=True,
                      sub_horizontal_median=True,
                      remove_vertical=False):

        if use_destripe_mask:
            destrip_mask = ~np.isfinite(data_minus)|self.destripe_mask
        else:
            destrip_mask = None

        from libs.destriper import destriper
        data_minus_d = destriper.get_destriped(data_minus,
                                               destrip_mask,
                                               pattern=destripe_pattern,
                                               hori=sub_horizontal_median,
                                               remove_vertical=remove_vertical)

        return data_minus_d


    def get_data1(self, i, hori=True, vert=False):

        fn = self.obj_filenames[i]
        data = pyfits.open(fn)[0].data

        from libs.destriper import destriper
        destrip_mask = ~np.isfinite(data)|self.destripe_mask

        data = destriper.get_destriped(data,
                                       destrip_mask,
                                       pattern=64,
                                       hori=hori)

        if vert:
            #m = [np.median(row[4:-4].compressed()) for row in dd1]
            dd1 = np.ma.array(data, mask=destrip_mask)
            m = np.ma.median(dd1, axis=1)
            #m = [np.ma.median(d) for d in dd1]
            datam = data - m[:,np.newaxis]

            return datam
        else:
            return data

    def get_old_orders(self):
        from libs.storage_descriptions import ONED_SPEC_JSON_DESC

        sky_basename = self.basenames["sky"]

        raw_spec_products = self.igr_storage.load1(ONED_SPEC_JSON_DESC,
                                                   sky_basename)

        old_orders = raw_spec_products["orders"]

        return old_orders

    def get_aperture(self):

        from recipe_wvlsol_sky import load_aperture_wvlsol

        # ap = load_aperture2(self.igr_storage, self.band,
        #                     self.pr.master_obsid,
        #                     self.db["flat_on"],
        #                     old_orders,
        #                     orders_w_solutions)

        ap = load_aperture_wvlsol(self, self.band)

        return ap

    def extract_slit_profile(self, ap, data_minus_flattened,
                             x1=800, x2=1200):
        bins, slit_profile_list = \
              ap.extract_slit_profile(self.ordermap_bpixed,
                                      self.slitpos_map,
                                      data_minus_flattened,
                                      x1, x2, bins=None)


        hh0 = np.sum(slit_profile_list, axis=0)

        return bins, hh0, slit_profile_list

    def get_norm_profile_ab(self, bins, hh0):
        peak1, peak2 = max(hh0), -min(hh0)
        profile_x = 0.5*(bins[1:]+bins[:-1])
        profile_y = hh0/(peak1+peak2)

        return profile_x, profile_y

    def get_norm_profile(self, bins, hh0):
        peak1 = max(hh0)
        profile_x = 0.5*(bins[1:]+bins[:-1])
        profile_y = hh0/peak1

        return profile_x, profile_y

    def get_profile_func_ab(self, profile_x, profile_y):
        from scipy.interpolate import UnivariateSpline
        profile_ = UnivariateSpline(profile_x, profile_y, k=3, s=0,
                                    bbox=[0, 1])

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

        def profile(o, x, slitpos):
            return profile_(slitpos) / integ

        return profile

    def get_profile_func(self, profile_x, profile_y):

        from scipy.interpolate import UnivariateSpline
        profile_ = UnivariateSpline(profile_x, profile_y, k=3, s=0,
                                    bbox=[0, 1])
        integ = profile_.integral(0, 1)

        def profile(o, x, slitpos):
            return profile_(slitpos) / integ

        return profile

    def make_profile_map(self, ap, profile,
                         ordermap=None, slitpos_map=None,
                         frac_slit=None):
        if ordermap is None:
            ordermap = self.ordermap
        if slitpos_map is None:
            slitpos_map = self.slitpos_map

        profile_map = ap.make_profile_map(ordermap,
                                          slitpos_map,
                                          profile)

        # select portion of the slit to extract

        if frac_slit is not None:
            frac1, frac2 = min(frac_slit), max(frac_slit)
            slitpos_msk = (slitpos_map < frac1) | (slitpos_map > frac2)
            profile_map[slitpos_msk] = np.nan

        return profile_map

    def get_shifted_all(self, ap, profile_map, variance_map,
                        data_minus_flattened, slitoffset_map,
                        debug=False):

        if slitoffset_map == "none":
            msk1 = np.isfinite(data_minus_flattened) & np.isfinite(variance_map)
            shifted = dict(data=data_minus_flattened,
                           variance_map=variance_map,
                           profile_map=profile_map,
                           mask=msk1)
        else:
            _ = ap.get_shifted_images(profile_map,
                                      variance_map,
                                      data_minus_flattened,
                                      slitoffset_map=slitoffset_map,
                                      debug=debug)

            data_shft, variance_map_shft, profile_map_shft, msk1_shft = _

            shifted = dict(data=data_shft,
                           variance_map=variance_map_shft,
                           profile_map=profile_map_shft,
                           mask=msk1_shft)


        return shifted

    def extract_spec_stellar(self, ap, shifted,
                             weight_thresh, remove_negative):

        _ = ap.extract_stellar_from_shifted(self.ordermap_bpixed,
                                            shifted["profile_map"],
                                            shifted["variance_map"],
                                            shifted["data"],
                                            shifted["mask"],
                                            weight_thresh=weight_thresh,
                                            remove_negative=remove_negative)
        s_list, v_list = _

        return s_list, v_list

    def make_synth_map(self, ap, profile_map, s_list,
                       ordermap=None, slitpos_map=None,
                       slitoffset_map=None):

        if ordermap is None:
            ordermap = self.ordermap

        if slitpos_map is None:
            slitpos_map = self.slitpos_map

        if slitoffset_map == "none":
            "do not use slit offset map"
            slitoffset_map = None
        elif slitoffset_map is None:
            "use default slit offset map"
            slitoffset_map = self.slitoffset_map

        synth_map = ap.make_synth_map(ordermap,
                                      slitpos_map,
                                      profile_map, s_list,
                                      slitoffset_map=slitoffset_map
                                      )

        return synth_map

    def get_updated_variance(self, variance_map, variance_map0, synth_map):
        variance_map_r = variance_map0 + np.abs(synth_map)/self.gain
        variance_map = np.max([variance_map, variance_map_r], axis=0)

        return variance_map

    def iter_order(self, bad_mask=None):
        ordermap_bpixed = self.ordermap_bpixed
        orders = self.orders_w_solutions

        slices = ni.find_objects(ordermap_bpixed)
        if bad_mask is None:
            mask = np.zeros_like(ordermap_bpixed, dtype=bool)

        ny, nx = ordermap_bpixed.shape
        for o in orders:
            sl = slices[o-1][0], slice(None)
            order_mask = (ordermap_bpixed[sl] == o) & ~(mask[sl])

            yield sl, order_mask

    def get_shifted(self, data, divide_orderflat=True):
        data2 = data/self.orderflat
        msk = ~np.isfinite(data2)
        data2[msk] = 0.
        data_shifted = self.shiftx(data2)
        imask_shifted = self.shiftx(~msk)

        return data_shifted, imask_shifted

    def get_simple_spec(self, data_shifted, frac1, frac2):
        ss = []
        slitpos_map = self.slitpos_map
        for sl, order_mask in self.iter_order():

            slitpos_m = (frac1 < slitpos_map[sl])  & (slitpos_map[sl] < frac2)

            order_data_masked = np.ma.array(data_shifted[sl],
                                            mask=~(order_mask&slitpos_m))

            s1 = np.ma.median(order_data_masked, axis=0)
            ss.append(s1)

        return ss
