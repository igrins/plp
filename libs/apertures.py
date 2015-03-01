import numpy as np
import numpy.polynomial as P
import scipy.ndimage as ni

from stsci_helper import stsci_median

class ApCoeff(object):
    """
    Apcoeff from original PLP.
    """
    def __init__(self, bottom_solution, up_solution):
        self.bottom_solution = bottom_solution
        self.up_solution = up_solution

    def __call__(self, pixel, frac=0.5):
        pixel = np.array(pixel)
        pixel_y1 = self.bottom_solution(pixel)
        pixel_y2 = self.up_solution(pixel)
        pixel_y = pixel_y1 + frac*(pixel_y2-pixel_y1)
        return pixel_y


class Apertures(object):
    def __init__(self, orders, bottomup_solutions):
        self.orders = orders

        self.apcoeffs = {}
        for o, (bottom, up) in zip(orders, bottomup_solutions):
            if isinstance(bottom, (list, tuple)) and bottom[0] == "poly":
                bottom = P.Polynomial(bottom[1])
            if isinstance(up, (list, tuple)) and up[0] == "poly":
                up = P.Polynomial(up[1])

            self.apcoeffs[o] = ApCoeff(bottom, up)

        self.yi = np.arange(2048)
        self.xi = np.arange(2048)


    def __call__(self, order, pixels, frac=0.5):
        return self.apcoeffs[order](pixels, frac)

    def get_xy_list(self, pixels_list, nan_filter=None):
        """
        ohlines_list : dict of tuples of (pixel coord list)
        """

        xy2 = []
        for order_i, pixel in pixels_list.items():
            pixel_y = self.apcoeffs[order_i](pixel)
            xy2.extend(zip(pixel, pixel_y))

        if nan_filter is not None:
            xy2 = np.compress(nan_filter, xy2, axis=0)

        return xy2

    def make_order_map(self, frac1=0., frac2=1.,
                       mask_top_bottom=False):

        from itertools import izip

        xx, yy = self.xi, self.yi

        bottom_list = [self.apcoeffs[o](xx, frac1) for o in self.orders]
        top_list = [self.apcoeffs[o](xx, frac2) for o in self.orders]

        if mask_top_bottom is False:
            def _g(i1):
                order_map1 = np.zeros(len(xx), dtype="i")
                for order, bottom, top in izip(self.orders,
                                               bottom_list, top_list):
                    m_up = yy>bottom[i1]
                    m_down = yy<top[i1]
                    order_map1[m_up & m_down] = order

                return order_map1
        else:
            def _g(i1):
                order_map1 = np.zeros(len(xx), dtype="i")
                for order, bottom, top in izip(self.orders,
                                               bottom_list, top_list):
                    m_up = yy>bottom[i1]
                    m_down = yy<top[i1]
                    order_map1[m_up & m_down] = order

                order_map1[yy>top_list[-1][i1]] = 999
                order_map1[yy<bottom_list[0][i1]] = 999
                return order_map1

        order_map = np.hstack([_g(i1).reshape((-1,1)) for i1 in xx])

        return order_map


    def make_slitpos_map(self):
        from itertools import izip

        xx, yy = self.xi, self.yi

        bottom_list = [self.apcoeffs[o](xx, 0.) for o in self.orders]
        top_list = [self.apcoeffs[o](xx, 1.) for o in self.orders]

        def _g(i1):
            slitpos_map1 = np.empty(len(xx), dtype="d")
            slitpos_map1.fill(np.nan)
            for order, bottom, top in izip(self.orders,
                                           bottom_list, top_list):
                m_up = yy>bottom[i1]
                m_down = yy<top[i1]
                m_order = m_up & m_down
                slit_pos = (yy[m_order] - bottom[i1])/(top[i1] - bottom[i1])
                slitpos_map1[m_order] = slit_pos

            return slitpos_map1

        order_map = np.hstack([_g(i1).reshape((-1,1)) for i1 in xx])

        return order_map

    def make_order_map_old(self, frac1=0., frac2=1.):
        """
        This one is significantly slower than make_order_map.
        """
        yy, xx = np.indices((2048, 2048))
        #order_maps = []
        order_map = np.zeros_like(yy)
        for o in self.orders:
            #izip(count(1), self.apcoeffs):
            ap = self.apcoeffs[o]
            m_up = yy>ap(xx, frac1)
            m_down = yy<ap(xx, frac2)
            order_map[m_up & m_down] = o

        return order_map

    def extract_spectra_v2(self, data, f1=0., f2=1.):

        xx = np.arange(2048)

        s_list = []
        for o in self.orders:
            yy1 = self.apcoeffs[o](xx, frac=f1)
            yy2 = self.apcoeffs[o](xx, frac=f2)

            down = np.clip((yy1+0.5).astype("i"), 0, 2048)
            up = np.clip((yy2++0.5).astype("i"), 0, 2048)

            s = [np.median(data[down[i]:up[i],i]) for i in range(2048)]
            s_list.append(s)

        return s_list

    def extract_spectra_from_ordermap(self, data, order_map):
        slices = ni.find_objects(order_map)
        s_list = []
        for o in self.orders:
            sl = slices[o - 1]
            msk = (order_map[sl] != o)
            s = stsci_median(data[sl], badmasks=msk)

            s_list.append(s)

        return s_list


    def get_mask_bg_pattern(self, flat_mask):
        im_shape = flat_mask.shape
        order_msk = make_order_map(im_shape, bottom_up_solutions)
        mask_to_estimate_bg_pattern = (flat_mask & order_msk)

        return mask_to_estimate_bg_pattern


    def get_shifted_images(self, profile_map, variance_map,
                           data, slitoffset_map=None, debug=False):

        msk1 = np.isfinite(data) & np.isfinite(variance_map)

        # it would be easier if shift data before all this?
        if slitoffset_map is not None:
            from correct_distortion import ShiftX
            shiftx = ShiftX(slitoffset_map)

            data, variance_map = data.copy(), variance_map.copy()
            data[~msk1] = 0
            variance_map[~msk1] = 0

            msk10 = shiftx(msk1)
            if debug:
                import astropy.io.fits as pyfits
                hdu_list = pyfits.HDUList()
                #hdu_list.append(pyfits.PrimaryHDU(data=msk1.astype("i4")))
                hdu_list.append(pyfits.PrimaryHDU(data=np.array(msk1, dtype="i")))
                #hdu_list.append(pyfits.ImageHDU(data=np.array(msk1, dtype="i")))
                hdu_list.append(pyfits.ImageHDU(data=msk10))
                hdu_list.writeto("test_mask.fits", clobber=True)

            profile_map = profile_map*msk10
            data = shiftx(data)#/msk10
            variance_map = shiftx(variance_map)#/msk10#/msk10 #**2

            msk1 = (msk10 > 0) & (variance_map > 0) #& (msk10 > 0.2)

        return data, variance_map, profile_map, msk1

    def extract_stellar_from_shifted(self, ordermap_bpixed,
                                     profile_map, variance_map,
                                     data, msk1,
                                     #slitpos_map,
                                     weight_thresh=0.05,
                                     remove_negative=False):

        s_list = []
        v_list = []
        slices = ni.find_objects(ordermap_bpixed)

        SAVE_PROFILE = True
        if SAVE_PROFILE:
            import astropy.io.fits as pyfits
            hl = pyfits.HDUList()
            hl.append(pyfits.PrimaryHDU())

        for o in self.orders:
            sl = slices[o-1][0], slice(0, 2048)
            msk = (ordermap_bpixed[sl] == o) & msk1[sl]

            profile_map1 = profile_map[sl].copy()

            profile_map1[~msk] = 0.

            #profile_map1[np.abs(profile_map1) < 0.02] = np.nan

            profile_map1[~np.isfinite(profile_map1)] = 0.

            # normalize profile
            #profile_map1 /= np.abs(profile_map1).sum(axis=0)


            variance_map1 = variance_map[sl]
            map_weighted_spectra1 = (profile_map1*data[sl])/variance_map1
            map_weights1 = profile_map1**2/variance_map1

            map_weighted_spectra1[~msk] = 0.
            map_weights1[~msk] = 0.

            mmm = np.isfinite(map_weighted_spectra1) & np.isfinite(map_weights1) & np.isfinite(profile_map1)
            map_weighted_spectra1[~mmm] = 0.
            map_weights1[~mmm] = 0.
            profile_map1[~mmm] = 0.

            #profile_map1 = profile_map[sl].copy()
            #profile_map1[~msk] = 0.

            sum_weighted_spectra1 = map_weighted_spectra1.sum(axis=0)

            # mask out range where sum_weighted_spectra1 < 0
            if remove_negative:
               sum_weighted_spectra1[sum_weighted_spectra1<0] = np.nan

            sum_weights1 = map_weights1.sum(axis=0)
            sum_profile1 = np.abs(profile_map1).sum(axis=0)

            # weight_thresh = 0.01 safe enough?
            #thresh_msk = sum_profile1 < 0.1
            thresh_msk = ni.binary_dilation(sum_profile1 < 0.1, iterations=5)
            sum_weights1[thresh_msk] = np.nan

            #sum_variance1 = variance_map1.sum(axis=0)
            #sum_weights1[sum_variance1 < 0.1*np.nanmax(sum_variance1)] = np.nan

            s = sum_weighted_spectra1 / sum_weights1

            s_list.append(s)

            v = sum_profile1 / sum_weights1

            v_list.append(v)

            if SAVE_PROFILE:
                hl.append(pyfits.ImageHDU(np.array([sum_weighted_spectra1,
                                                    sum_weights1,
                                                    sum_profile1,
                                                    ])))



        if SAVE_PROFILE:
            hl.writeto("test_profile.fits", clobber=True)
        return s_list, v_list


    def extract_extended_from_shifted(self, ordermap_bpixed,
                                      profile_map, variance_map,
                                      data, msk1,
                                      weight_thresh=0.05,
                                      remove_negative=False):

        s_list = []
        v_list = []
        slices = ni.find_objects(ordermap_bpixed)

        SAVE_PROFILE = True
        if SAVE_PROFILE:
            import astropy.io.fits as pyfits
            hl = pyfits.HDUList()
            hl.append(pyfits.PrimaryHDU())

        for o in self.orders:
            sl = slices[o-1][0], slice(0, 2048)
            msk = (ordermap_bpixed[sl] == o) & msk1[sl]

            profile_map1 = profile_map[sl].copy()

            profile_map1[~msk] = 0.

            #profile_map1[np.abs(profile_map1) < 0.02] = np.nan

            profile_map1[~np.isfinite(profile_map1)] = 0.

            # normalize profile
            #profile_map1 /= np.abs(profile_map1).sum(axis=0)


            variance_map1 = variance_map[sl]
            map_weighted_spectra1 = (profile_map1*data[sl])/variance_map1
            map_weights1 = profile_map1**2/variance_map1

            map_weighted_spectra1[~msk] = 0.
            map_weights1[~msk] = 0.

            mmm = np.isfinite(map_weighted_spectra1) & np.isfinite(map_weights1) & np.isfinite(profile_map1)
            map_weighted_spectra1[~mmm] = 0.
            map_weights1[~mmm] = 0.
            profile_map1[~mmm] = 0.

            #profile_map1 = profile_map[sl].copy()
            #profile_map1[~msk] = 0.

            sum_weighted_spectra1 = map_weighted_spectra1.sum(axis=0)

            # mask out range where sum_weighted_spectra1 < 0
            if remove_negative:
               sum_weighted_spectra1[sum_weighted_spectra1<0] = np.nan

            sum_weights1 = map_weights1.sum(axis=0)
            sum_profile1 = np.abs(profile_map1).sum(axis=0)

            # weight_thresh = 0.01 safe enough?
            #thresh_msk = sum_profile1 < 0.1
            thresh_msk = ni.binary_dilation(sum_profile1 < 0.1, iterations=5)
            sum_weights1[thresh_msk] = np.nan

            #sum_variance1 = variance_map1.sum(axis=0)
            #sum_weights1[sum_variance1 < 0.1*np.nanmax(sum_variance1)] = np.nan

            s = sum_weighted_spectra1 / sum_weights1

            s_list.append(s)

            v = sum_profile1 / sum_weights1

            v_list.append(v)

            if SAVE_PROFILE:
                hl.append(pyfits.ImageHDU(np.array([sum_weighted_spectra1,
                                                    sum_weights1,
                                                    sum_profile1,
                                                    ])))



        if SAVE_PROFILE:
            hl.writeto("test_profile.fits", clobber=True)
        return s_list, v_list


    def extract_stellar(self, ordermap_bpixed, profile_map, variance_map,
                        data, slitoffset_map=None, weight_thresh=0.05,
                        remove_negative=False):

        _ = self.get_shifted_images(profile_map, variance_map,
                                    data, slitoffset_map=slitoffset_map)

        data, variance_map, profile_map, msk1 = _

        _ = self.extract_stellar_from_shifted(ordermap_bpixed,
                                              profile_map, variance_map,
                                              data, msk1,
                                              weight_thresh=weight_thresh,
                                              remove_negative=remove_negative)
        s_list, v_list = _

        return s_list, v_list


    def extract_slit_profile(self, order_map, slitpos_map, data,
                             x1, x2, bins=None):

        x1, x2 = int(x1), int(x2)

        slices = ni.find_objects(order_map)
        slit_profile_list = []
        if bins is None:
            bins = np.linspace(0., 1., 40)

        for o in self.orders:
            sl = slices[o-1][0], slice(x1, x2)
            msk = (order_map[sl] == o)

            #ss = slitpos_map[sl].copy()
            #ss[~msk] = np.nan

            d = data[sl][msk]
            finite_mask = np.isfinite(d)
            hh = np.histogram(slitpos_map[sl][msk][finite_mask],
                              weights=d[finite_mask], bins=bins,
                              )
            slit_profile_list.append(hh[0])

        return bins, slit_profile_list


    def extract_stellar_orig(self, ordermap_bpixed, profile_map, variance_map,
                        data, slitoffset_map=None, weight_thresh=0.05,
                        remove_negative=False):

        msk1 = np.isfinite(data) & np.isfinite(variance_map)

        # it would be easier if shift data before all this?
        if slitoffset_map is not None:
            from correct_distortion import ShiftX
            shiftx = ShiftX(slitoffset_map)

            data, variance_map = data.copy(), variance_map.copy()
            data[~msk1] = 0
            variance_map[~msk1] = 0

            msk10 = shiftx(msk1)

            profile_map = profile_map*msk10
            data = shiftx(data)#/msk10
            variance_map = shiftx(variance_map)#/msk10#/msk10 #**2

            msk1 = (msk10 > 0) & (variance_map > 0) #& (msk10 > 0.2)

        s_list = []
        v_list = []
        slices = ni.find_objects(ordermap_bpixed)
        for o in self.orders:
            sl = slices[o-1][0], slice(0, 2048)
            msk = (ordermap_bpixed[sl] == o) & msk1[sl]

            profile_map1 = profile_map[sl].copy()
            profile_map1[~msk] = 0.

            profile_map1[np.abs(profile_map1) < 0.02] = np.nan

            profile_map1[~np.isfinite(profile_map1)] = 0.


            # normalize profile
            #profile_map1 /= np.abs(profile_map1).sum(axis=0)


            variance_map1 = variance_map[sl]
            map_weighted_spectra1 = (profile_map1*data[sl])/variance_map1
            map_weights1 = profile_map1**2/variance_map1

            map_weighted_spectra1[~msk] = 0.
            map_weights1[~msk] = 0.

            #profile_map1 = profile_map[sl].copy()
            #profile_map1[~msk] = 0.

            sum_weighted_spectra1 = map_weighted_spectra1.sum(axis=0)

            # mask out range where sum_weighted_spectra1 < 0
            if remove_negative:
                sum_weighted_spectra1[sum_weighted_spectra1<0] = np.nan

            sum_weights1 = map_weights1.sum(axis=0)
            sum_profile1 = np.abs(profile_map1).sum(axis=0)

            # weight_thresh = 0.01 safe enough?
            sum_weights1[sum_weights1<np.nanmax(sum_weights1)*weight_thresh] = np.nan
            s = sum_weighted_spectra1 / sum_weights1

            s_list.append(s)

            v = sum_profile1 / sum_weights1

            v_list.append(v)

        return s_list, v_list


    def make_profile_map(self, order_map, slitpos_map, lsf,
                         slitoffset_map=None):
        """
        lsf : callable object which takes (o, x, slit_pos)

        o : order (integer)
        x : detector position in dispersion direction
        slit_pos : 0..1

        x and slit_pos can be array.
        """

        iy, ix = np.indices(slitpos_map.shape)

        if slitoffset_map is not None:
            ix = ix - slitoffset_map

        profile_map = np.empty(slitpos_map.shape, "d")
        profile_map.fill(np.nan)

        slices = ni.find_objects(order_map)
        for o in self.orders:
            sl = slices[o-1][0], slice(0, 2048)
            msk = (order_map[sl] == o)

            profile1 = np.zeros(profile_map[sl].shape, "d")
            profile1[msk] = lsf(o, ix[sl][msk], slitpos_map[sl][msk])
            profile_sum = np.abs(profile1).sum(axis=0)
            profile1 /= profile_sum

            profile_map[sl][msk] = profile1[msk]

        return profile_map

    def make_synth_map(self, order_map, slitpos_map,
                       profile_map, s_list,
                       slitoffset_map=None):
        """
        lsf : callable object which takes (o, x, slit_pos)

        o : order (integer)
        x : detector position in dispersion direction
        slit_pos : 0..1

        x and slit_pos can be array.

        s_list : list of specs
        """

        iy, ix = np.indices(slitpos_map.shape)

        if slitoffset_map is not None:
            ix = ix - slitoffset_map

        synth_map = np.empty(slitpos_map.shape, "d")
        synth_map.fill(np.nan)

        xx = np.arange(2048)

        slices = ni.find_objects(order_map)
        for o, s in zip(self.orders, s_list):
            sl = slices[o-1][0], slice(0, 2048)
            msk = (order_map[sl] == o)

            from scipy.interpolate import UnivariateSpline
            msk_s = np.isfinite(s)
            s_spline = UnivariateSpline(xx[msk_s], s[msk_s], k=3, s=0,
                                        bbox=[0, 2047])

            ixm = ix[sl][msk]
            #synth_map[sl][msk] = s_spline(ixm) * lsf(o, ixm, slitpos_map[sl][msk])
            synth_map[sl][msk] = s_spline(ixm) * profile_map[sl][msk]

        return synth_map


if 0:
    for s, wvl in zip(s_list, wvl_solutions):
        plot(wvl[100:-5], s[100:-5])



if __name__ == "__main__":
    from pipeline_jjlee import IGRINSLog

    log_20140316 = dict(flat_off=range(2, 4),
                        flat_on=range(4, 7),
                        thar=[1],
                        sky=[25],
                        HD3417=[15, 16],
                        ur=[993]) # 93 U-Ne (no ThAr)


    igrins_log_src = IGRINSLog("20140316", log_20140316)

    band = "H"

    import pickle
    r_src = pickle.load(open("flat_info_%s_%s.pickle" % (igrins_log_src.date, band)))


    #orders, order_map = make_order_map(r_src)
    orders = np.arange(len(r_src["bottomup_solutions"])) + 1

    ap = Apertures(orders, r_src["bottomup_solutions"])
    #orders, order_map = ap.make_order_map()
    slitpos_map = ap.make_slitpos_map()
    #orders, order_map = ap.make_order_map_old()

    if 0:
        data = r_src["flat_normed"]
        ap.extract_spectra_v2(data)

        ap.extract_spectra_from_ordermap(data, order_map)
