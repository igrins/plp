import libs.fits as pyfits
import numpy as np
import scipy.ndimage as ni

from libs.a0v_spec import A0VSpec

from scipy.interpolate import interp1d

def air2vac(x0):
    """
    x in angstrom
    """
    f = 10000.0
    x = x0*f
    n = 1.0 + 2.735182e-4 + 131.4182 / x ** 2 + 2.76249e8 / x ** 4
    print n
    return x0*n

class TelluricTransmission(object):
    def __init__(self, fn):
        #fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
        #self.telluric = pyfits.open(fn)[1].data
        assert fn.endswith("npy")

        self._data = np.load(fn)
        #self.dd = np.genfromtxt(fn)
        self.trans = self._data[:,1]
        self.wvl = air2vac(self._data[:,0]/1.e3)

    def get_telluric_trans(self, wvl1, wvl2):
        i1, i2 = np.searchsorted(self.wvl, [wvl1, wvl2])
        #mask = (wvl1 < self.wvl) & (self.wvl < wvl2)
        mask = slice(i1, i2)
        # spl = UnivariateSpline(telluric_lam[tel_mask_igr],
        #                        telluric["trans"][tel_mask_igr],
        #                        k=1,s=0)

        return self.wvl[mask], self.trans[mask]

    def get_telluric_trans_interp1d(self, wvl1, wvl2, gw=None):
        mask = (wvl1 < self.wvl) & (self.wvl < wvl2)
        # spl = UnivariateSpline(telluric_lam[tel_mask_igr],
        #                        telluric["trans"][tel_mask_igr],
        #                        k=1,s=0)

        if gw is not None:
            from scipy.ndimage import gaussian_filter1d
            trans = gaussian_filter1d(self.trans[mask], gw)
        else:
            trans = self.trans[mask]
        spl = interp1d(self.wvl[mask],
                       trans,
                       bounds_error=False
                       )

        return spl


# def get_weight_func():
#     #for testing purpose

#     rad1 = 35
#     rad2 = 50
#     factor12 = 0.


#     from astropy.modeling.functional_models import Moffat1D
#     weight_func = Moffat1D(gamma=15, alpha=2)

#     def weight_func2(dist):
#         return np.exp(-(dist/(2.*rad1))**2) + factor12*np.exp(-(dist/(2.*rad2))**2)

#     return weight_func2


class FlattenFailException(Exception):
    pass

class SpecFlattener(object):
    def __init__(self, tel_interp1d_f, a0v_interp1d):
        self.tel_interp1d_f = tel_interp1d_f
        self.a0v_interp1d = a0v_interp1d


    def get_norm_const(self, w1, s1_orig, of):
        return np.nanmax(s1_orig)

    def get_initial_mask(self, ratio, tt):
        """
        ratio : spectra divided by tel model.
        tt : tel model
        """
        thresh=0.5
        ratio[tt<thresh] = np.nan # blank out area with low tel-trans.

        # now we blank out area where derivative varies significantly.

        # derivative of ratio
        ratiom = ni.gaussian_filter1d(ratio, 3, order=1)

        # do median filtering to derive a baseline for derivative.

        # Because of nan values of original spec. doing it with
        # single value of filter size does not seem to work well.
        # Instead, we first filter with large filter size, then
        # with smaller one. Then replaces nan of former with
        # result of latter.

        dd = ni.median_filter(ratiom, 11)
        dd_small = ni.median_filter(ratiom, 5)
        dd_msk = ~np.isfinite(dd)
        dd[dd_msk] = dd_small[dd_msk]

        # subtract baseline from derivative.
        dd1 = (ratiom - dd)

        # mask out areas of high deriv.
        # The threshold is arbitrary but seem to work for most of orders.
        msk = np.abs(dd1) > 0.0005
        #msk = np.abs(dd1) > 0.0003# for H

        # do binary-closing for mask
        msk = ni.binary_closing(msk, iterations=1) | ~np.isfinite(dd1)

        dd1[msk] = np.nan

        return dd1, msk

    def get_sv_mask(self, ratio):
        # use sv_iter to derive continuum level from ratio
        from libs.smooth_continuum import sv_iter
        from libs.trace_flat import get_finite_boundary_indices

        # naive iterative approach.
        ratio = ratio.copy()
        for thresh in [0.05, 0.03, 0.01]: #, 0.005, 0.0001]:
            i1, i2 = get_finite_boundary_indices(ratio)
            sl2 = slice(i1, i2+1)
            f12 = sv_iter(ratio[sl2], winsize1=251, winsize2=151)
            #fl = s1_orig_n[sl2]/f12 # s1m : normalization factor
            # f12 : flattening curve
            # fl : flattened spectrum
            new_msk = np.abs(ratio[sl2]-f12) > thresh
            ratio[sl2][new_msk] = np.nan


        # # recover masked area due to underestimated continuum
        # ratio0 = s1_orig_n/tt # model corrected spec
        # mmm = ratio0[sl2] > f12
        # ratio[sl2][mmm] = ratio0[sl2][mmm]

        # for thresh in [0.01]:
        #     i1, i2 = get_finite_boundary_indices(ratio)
        #     sl2 = slice(i1, i2+1)
        #     f12 = sv_iter(ratio[sl2], winsize1=351, winsize2=261)
        #     fl = (s1_orig/s1m)[sl2]/f12 # s1m : normalization factor
        #     # f12 : flattening curve
        #     # fl : flattened spectrum

        #     new_msk = np.abs(ratio[sl2]-f12) > thresh
        #     ratio[sl2][new_msk] = np.nan

        msk = ~np.isfinite(ratio)

        return msk

    def get_of_interpolated(self, ratio, tt, of, msk,
                            weight_func=None):

        if weight_func is None:
            finite_frac = 1 - np.mean(msk) #float(np.sum((msk)))/len(msk)
            print finite_frac
            finite_frac = np.clip(finite_frac , 0.1, 0.8)
            rad1 = 15./ finite_frac
            #rad1 = 150
            rad2 = 160.
            frac = 0.5 * (1. - finite_frac)
            print "frac, rad1, A = %d, %3f, %3f" % (finite_frac, int(rad1), frac),
            def weight_func(dist, rad1=rad1, rad2=rad2):
                return np.exp(-(dist/(2.*rad1))**2) + frac*np.exp(-(dist/(2.*rad2))**2)


        of = np.array(of)
        of[of<0.01] = np.nan

        indices=np.arange(len(msk))[(~msk) & np.isfinite(of)]

        lll = []
        for i in np.arange(len(msk)):

            ii = np.searchsorted(indices, i)

            dd = 25
            lower_indices = indices[max(0, ii-dd):ii]
            upper_indices = indices[ii:min(len(msk), ii+dd)]

            lu_indices = np.concatenate([lower_indices, upper_indices])

            #lu_indices = indices
            dist = np.abs(lu_indices - i)
            #weight_dist = (np.exp(-(dist/(2.*rad1))**2) + factor12*np.exp(-(dist/(2.*rad2))**2))*tt[lu_indices]


            weight_dist = weight_func(dist)*tt[lu_indices]*of[lu_indices]

            vv = ratio[lu_indices] / of[lu_indices]
            vv0 = np.sum(vv*weight_dist)/np.sum(weight_dist)
            lll.append(vv0*of[i])

        return lll


    def get_s_a0v(self, w1, dw_opt):
        s_a0v = self.a0v_interp1d(w1+dw_opt)
        return s_a0v

    def get_tel_trans(self, w1, dw_opt, gw_opt):
        tel_interp1d_f = self.tel_interp1d_f
        tel_interp1d = tel_interp1d_f(gw_opt)

        tt = tel_interp1d(w1+dw_opt) # telluric model

        return tt


    def flatten(self, w1, s1_orig, of, s1_a0v, tel_trans):

        tt1 = tel_trans
        _ = self.flatten_base(w1, s1_orig, of, s1_a0v, tt1)
        msk_i, msk_sv, fitted_continuum = _

        return msk_i, msk_sv, fitted_continuum


    def flatten_base(self, w1, s1_orig, of,
                     s_a0v, tt):

        s1_orig_c = s1_orig / s_a0v
        s1m = self.get_norm_const(w1, s1_orig_c, of)

        s1_orig_cn = s1_orig_c/s1m # normalize

        ratio = s1_orig_cn/tt # model corrected spec

        dd1, msk_i = self.get_initial_mask(ratio, tt)
        if np.all(msk_i):
            raise FlattenFailException()

        ratio[:4] = np.nan
        ratio[-4:] = np.nan

        ratio_msk = np.ma.array(ratio, mask=msk_i).filled(np.nan)

        try:
            msk_sv = self.get_sv_mask(ratio_msk)
        except RuntimeError:
            msk_sv = msk_i
        except ValueError:
            msk_sv = msk_i

        lll = self.get_of_interpolated(ratio_msk, tt, of, msk_sv)

        # if False:
        #     # recover area where continuum is underestimated
        #     ratio0 = s1_orig_cn/tt # model corrected spec
        #     ratio0[tt < 0.5] = np.nan
        #     ratio0[s1_orig_cn < 0.1] = np.nan
        #     mmm = np.array(lll) < ratio0
        #     ratio[mmm] = ratio0[mmm]

        #     msk = ~np.isfinite(ratio)
        #     lll = self.get_of_interpolated(ratio, tt, of, msk)

        f12 = np.array(lll)

        fitted_continuum = f12*s1m*s_a0v

        return msk_i, msk_sv, fitted_continuum

    def plot_fitted(self, w1, s1_orig, s_a0v, tt,
                    msk_i, msk_sv, fitted_continuum,
                    ax1=None, ax2=None):

        fl = s1_orig / fitted_continuum # s1m : normalization factor
        ratio = s1_orig/tt # model corrected spec
        if ax1:
            # plot raw spec
            color1 = ax1._get_lines.color_cycle.next()
            ax1.plot(w1, s1_orig, alpha=0.3, color=color1)
            # plot mask
            ax1.plot(w1,
                     np.ma.array(ratio, mask=msk_i).filled(np.nan),
                     lw=5, alpha=0.2, color=color1)

            # show model corrected raw spec
            ax1.plot(w1,
                     np.ma.array(ratio, mask=msk_sv).filled(np.nan),
                     lw=3, alpha=0.5, color=color1)

            # show fitted continuum
            ax1.plot(w1, fitted_continuum/s_a0v, lw=1.5, alpha=0.3, color="0.3")
            ax1.plot(w1, fitted_continuum, color="k")
            #plot(w1, s1_new)

        if ax2:
            # plot telluric model
            ax2.plot(w1, tt, alpha=0.2, lw=3, color=color1)
            # plot flattened spec
            ax2.plot(w1,
                     np.ma.array(fl,mask=fitted_continuum<0.05).filled(np.nan),
                     color=color1)


    def flatten_deprecated(self, w1, s1_orig, dw_opt, gw_opt, try_small=True,
                           ax1=None, ax2=None):
        tel_interp1d_f = self.tel_interp1d_f

        s_a0v = self.a0v_interp1d(w1+dw_opt)
        s1_orig = s1_orig / s_a0v

        tel_interp1d = tel_interp1d_f(gw_opt)

        tt = tel_interp1d(w1+dw_opt) # telluric model
        if ax2:
            # plot telluric model
            ax2.plot(w1, tt, alpha=0.2, lw=3)

        s1m = np.nanmax(s1_orig)
        #s1m = 1.
        s1_orig_n = s1_orig/s1m # normalize

        # plot raw spec
        if ax1:
            ax1.plot(w1, s1_orig, alpha=0.3)

        ratio = s1_orig_n/tt # model corrected spec
        thresh=0.5
        ratio[tt<thresh] = np.nan # blank out area with low tel-trans.

        # now we blank out area where derivative varies significantly.

        # derivative of ratio
        ratiom = ni.gaussian_filter1d(ratio, 3, order=1)

        # do median filtering to derive a baseline for derivative.

        # Because of nan values of original spec. doing it with
        # single value of filter size does not seem to work well.
        # Instead, we first filter with large filter size, then
        # with smaller one. Then replaces nan of former with
        # result of latter.

        dd = ni.median_filter(ratiom, 11)
        dd_small = ni.median_filter(ratiom, 5)
        dd_msk = ~np.isfinite(dd)
        dd[dd_msk] = dd_small[dd_msk]

        # subtract baseline from derivative.
        dd1 = (ratiom - dd)

        # mask out areas of high deriv.
        # The threshold is arbitrary but seem to work for most of orders.
        msk = np.abs(dd1) > 0.0005
        #msk = np.abs(dd1) > 0.0003# for H

        # do binary-closing for mask
        msk = ni.binary_closing(msk, iterations=1) | ~np.isfinite(dd1)

        dd1[msk] = np.nan

        if ax1:
            # plot mask
            ax1.plot(w1, dd1)

        if ax2:
            # plot mask
            ax2.plot(w1, dd1)

        # now, mask out msk from original ratio.
        ratio[msk] = np.nan
        ratio[:4] = np.nan
        ratio[-4:] = np.nan

        # use sv_iter to derive continuum level from ratio
        from libs.smooth_continuum import sv_iter
        from libs.trace_flat import get_finite_boundary_indices

        # naive iterative approach.
        for thresh in [0.05, 0.03, 0.01]:
            i1, i2 = get_finite_boundary_indices(ratio)
            sl2 = slice(i1, i2+1)
            f12 = sv_iter(ratio[sl2], winsize1=251, winsize2=151)
            fl = (s1_orig/s1m)[sl2]/f12 # s1m : normalization factor
            # f12 : flattening curve
            # fl : flattened spectrum

            new_msk = np.abs(ratio[sl2]-f12) > thresh
            ratio[sl2][new_msk] = np.nan


        # recover masked area due to underestimated continuum
        ratio0 = s1_orig_n/tt # model corrected spec
        mmm = ratio0[sl2] > f12
        ratio[sl2][mmm] = ratio0[sl2][mmm]

        for thresh in [0.01]:
            i1, i2 = get_finite_boundary_indices(ratio)
            sl2 = slice(i1, i2+1)
            f12 = sv_iter(ratio[sl2], winsize1=351, winsize2=261)
            fl = (s1_orig/s1m)[sl2]/f12 # s1m : normalization factor
            # f12 : flattening curve
            # fl : flattened spectrum

            new_msk = np.abs(ratio[sl2]-f12) > thresh
            ratio[sl2][new_msk] = np.nan


        if try_small:
            i1, i2 = get_finite_boundary_indices(ratio)
            sl2 = slice(i1, i2+1)
            f12 = sv_iter(ratio[sl2], winsize1=151, winsize2=91)
            fl = (s1_orig/s1m)[sl2]/f12
            # f12 : flattening curve
            # fl : flattened spectrum

        new_msk = np.abs(ratio[sl2]-f12) > thresh
        ratio[sl2][new_msk] = np.nan

        if ax1:
            # show model corrected raw spec
            ax1.plot(w1, ratio*s1m, lw=3, color="0.8")

            # show fitted continuum
            ax1.plot(w1[sl2], f12*s1m)
            #plot(w1, s1_new)

        if ax2:
            # plot flattened spec
            ax2.plot(w1[sl2],
                     np.ma.array(fl,mask=f12<0.05).filled(np.nan))



        f12_0 = np.zeros_like(s1_orig)
        f12_0[sl2] = f12

        return s1m*f12_0, np.isfinite(ratio)


def plot_flattend_a0v(spec_flattener, w, s_orig, of_list, data_list,
                      fout=None):
        import matplotlib.pyplot as plt
        print "Now generating figures"
        fig = plt.figure(0)
        fig.clf()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        # data_list = [("flattened_spec", s_orig/continuum_array),
        #              ("wavelength", w),
        #              ("fitted_continuum", continuum_array),
        #              ("mask", mask_array),
        #              ("a0v_norm", a0v_array),
        #              ("model_teltrans", teltrans_array),
        #              ]

        s_a0v = data_list[4][1]
        tt_list = data_list[5][1]
        msk_list = data_list[3][1]
        cont_list = data_list[2][1]

        _interim_result = zip(w, s_orig, of_list, s_a0v, tt_list, msk_list, cont_list)

        for w1, s1_orig, of, s1_a0v, tt1, msk, cont in _interim_result:
            spec_flattener.plot_fitted(w1, s1_orig, s1_a0v, tt1,
                                       msk, msk, cont,
                                       ax1=ax1, ax2=ax2)

        ax2.set_ylim(-0.1, 1.2)
        ax2.axhline(1.0, color="0.8")
        ax2.set_xlabel(r"Wavelength [$\mu$m]")
        ax2.set_ylabel(r"Flattened Spectra")
        ax1.set_ylabel(r"Spectra w/o Blaze function correction")

        from matplotlib.backends.backend_pdf import PdfPages
        if fout is None:
            fout = 'multipage_pdf.pdf'
        with PdfPages(fout) as pdf:
            w = np.array(w)
            wmin, wmax = w.min(), w.max()
            ymax = np.nanmax(s_orig)
            ax1.set_xlim(wmin, wmax)
            ax1.set_ylim(-0.1*ymax, 1.1*ymax)
            pdf.savefig(figure=fig)

            for w1, s1_orig, of, s1_a0v, tt1, msk, cont in _interim_result:
                ax1.set_xlim(min(w1), max(w1))
                pdf.savefig(figure=fig)



# def get_a0v_flattened(self, extractor, ap,
#                       s_list, wvl):
def get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                      wvl, s_list, orderflat_response,
                      figout=None):

    # a0v_interp1d = self.get_a0v_interp1d(extractor)
    # tel_interp1d_f = self.get_tel_interp1d_f(extractor, wvl)


    # orderflat_response = extractor.orderflat_json["fitted_responses"]


    #from libs.a0v_flatten import SpecFlattener, FlattenFailException
    spec_flattener = SpecFlattener(tel_interp1d_f, a0v_interp1d)


    dw_opt=0
    gw_opt=3.

    ccc = []
    _interim_result = []


    print "flattening ...",
    for w1, s1_orig, of in zip(wvl, s_list, orderflat_response):

        print "(%5.3f~%5.3f)" % (w1[0], w1[-1]),
        s1_a0v = spec_flattener.get_s_a0v(w1, dw_opt)
        tt1 = spec_flattener.get_tel_trans(w1, dw_opt, gw_opt)

        try:
            _ = spec_flattener.flatten(w1, s1_orig,
                                       of, s1_a0v, tt1)
        except FlattenFailException:
            ccc.append((np.zeros_like(w1),
                        np.ones_like(w1, dtype=bool),
                        s1_a0v, tt1))

        else:
            _interim_result.append((w1, s1_orig, of, s1_a0v, tt1, _))
            msk_i, msk_sv, fitted_continuum = _
            ccc.append((fitted_continuum, msk_sv, s1_a0v, tt1))

    print " - Done."

    continuum_array = np.array([c for c, m, a, t in ccc])
    mask_array = np.array([m for c, m, a, t in ccc])
    a0v_array = np.array([a for c, m, a, t in ccc])
    teltrans_array = np.array([t for c, m, a, t in ccc])

    flattened_s = s_list/continuum_array
    # if self.fill_nan is not None:
    #     flattened_s[~np.isfinite(flattened_s)] = self.fill_nan

    data_list = [("flattened_spec", flattened_s),
                 ("wavelength", np.array(wvl)),
                 ("fitted_continuum", continuum_array),
                 ("mask", mask_array),
                 ("a0v_norm", a0v_array),
                 ("model_teltrans", teltrans_array),
                 ]


    if figout is not None:
        from libs.a0v_flatten import plot_flattend_a0v

        plot_flattend_a0v(spec_flattener, wvl, s_list, orderflat_response, data_list, fout=figout)

    return data_list



if __name__ == "__main__":


    band = "H"

    f = pyfits.open("outdata/20140525/SDC%s_20140525_0016.spec.fits" % band)
    # f_flattened = pyfits.open("outdata/20140525/SDC%s_20140525_0016.spec_flattened.fits" % band)
    w = f[1].data
    s_orig = f[0].data
    #s_flattened = f_flattened[0].data

    import json
    orderflat = json.load(open("calib/primary/20140525/ORDERFLAT_SDC%s_20140525_0074.json" % band))

    telfit_outname = "transmission-795.20-288.30-41.9-45.0-368.50-3.90-1.80-1.40.%s" % band
    telfit_outname_npy = telfit_outname+".npy"
    if 0:
        dd = np.genfromtxt(telfit_outname)
        np.save(open(telfit_outname_npy, "w"), dd[::10])

    tel_trans = TelluricTransmission(telfit_outname_npy)

    a0v_model = A0VSpec()
    a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                               flatten=True,
                                               smooth_pixel=32)
    # if 0:
    #     def _a0v_interp1d(x):
    #         return np.ones_like(x)
    #     a0v_interp1d = _a0v_interp1d

    w_min = w.min()*0.9
    w_max = w.max()*1.1
    def tel_interp1d_f(gw=None):
        return tel_trans.get_telluric_trans_interp1d(w_min, w_max, gw)


    of_list = orderflat["fitted_responses"]

    figout = "multi_page.pdf"
    data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                                  w, s_orig, of_list,
                                  figout=figout)
