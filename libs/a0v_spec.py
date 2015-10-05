import numpy as np

import libs.fits as pyfits
from scipy.interpolate import interp1d
from libs.master_calib import get_master_calib_abspath

import scipy.ndimage as ni

from astropy.modeling import models, fitting

class TelluricTransmission(object):
    def __init__(self):
        fn = get_master_calib_abspath("telluric/LBL_A15_s0_w050_R0060000_T.fits")
        self.telluric = pyfits.open(fn)[1].data
        self.trans = self.telluric["trans"]
        self.wvl = self.telluric["lam"]

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

        #trans = spl(wvl_a0v[mask_igr])



class A0VSpec(object):
    def __init__(self):
        from libs.master_calib import get_master_calib_abspath

        #fn = get_master_calib_abspath("A0V/vegallpr25.50000resam5")
        #d = np.genfromtxt(fn)

        fn = get_master_calib_abspath("A0V/vegallpr25.50000resam5.npy")
        d = np.load(fn)

        wvl, flux, cont = (d[:,i] for i in [0, 1, 2])
        wvl = wvl/1000.

        self.wvl = wvl

        self.flux = flux
        self.cont = cont

    def get_flux_interp1d(self, wvl1, wvl2, flatten=False, trans=None,
                          smooth_pixel=None):

        wvl1, wvl2 = min(wvl1, wvl2), max(wvl1, wvl2)
        dwvl = (wvl2 - wvl1) * 0.01

        wvl1, wvl2 = wvl1 - dwvl, wvl2 + dwvl

        mask = (wvl1 < self.wvl) & (self.wvl < wvl2)


        #flattened_flux = (self.flux/self.cont)

        if flatten:
            flux = self.flux / self.cont
        else:
            flux = self.flux

        if trans is not None:
            flux_masked = flux[mask] * trans(self.wvl[mask])
        else:
            flux_masked = flux[mask]

        if smooth_pixel is not None:
            flux_masked = ni.gaussian_filter(flux_masked, smooth_pixel)

        spl = interp1d(self.wvl[mask],
                       flux_masked,
                       bounds_error=False
                       )

        return spl


#a0v_wvl, a0v_tel_trans
def get_a0v(a0v_spec, wvl1, wvl2, tel_trans, flatten=True):

    a0v_interp1d = a0v_spec.get_flux_interp1d(wvl1, wvl2,
                                              flatten=flatten,
                                              smooth_pixel=32)

    tel_trans_interp1d = tel_trans.get_telluric_trans_interp1d(wvl1, wvl2)

    trans = tel_trans_interp1d(a0v_interp1d.x)

    trans_m = ni.maximum_filter(trans, 128)
    trans_mg = ni.gaussian_filter(trans_m, 32)

    a0v_wvl = a0v_interp1d.x
    a0v_tel_trans = a0v_interp1d.y*trans

    #plot(a0v_wvl, a0v_tel_trans)

    mmm = trans/trans_mg > 0.98
    a0v_tel_trans_masked = a0v_tel_trans.copy()
    a0v_tel_trans_masked[~mmm] = np.nan

    return a0v_wvl, a0v_tel_trans, a0v_tel_trans_masked

    # return a0v_wvl, a0v_tel_trans


def get_flattend(a0v_spec,
                 a0v_wvl, a0v_tel_trans_masked, wvl_solutions, s_list,
                 i1i2_list=None):

    a0v_flattened = []

    if i1i2_list is None:
        i1i2_list = [[0, -1]] * len(wvl_solutions)

    a0v_interp1d = a0v_spec.get_flux_interp1d(a0v_wvl[0], a0v_wvl[-1],
                                              flatten=True,
                                              smooth_pixel=32)

    for wvl, s, (i1, i2) in zip(wvl_solutions, s_list, i1i2_list):

        wvl1, wvl2 = wvl[i1], wvl[i2]
        z_m = (wvl1 < a0v_wvl) & (a0v_wvl < wvl2)

        ss = interp1d(wvl, s)

        x_s = a0v_wvl[z_m]
        if len(x_s):

            s_interped = ss(x_s)

            xxx, yyy = a0v_wvl[z_m], s_interped/a0v_tel_trans_masked[z_m]

            p_init = models.Chebyshev1D(domain=[wvl1, wvl2],
                                        degree=6)
            fit_p = fitting.LinearLSQFitter()
            x_m = np.isfinite(yyy)
            p = fit_p(p_init, xxx[x_m], yyy[x_m])

            res_ = p(wvl)

            s_f = s/res_

            # now divide by A0V
            a0v = a0v_interp1d(wvl)


            s_f = s_f/a0v

            z_m = (wvl1 < wvl) & (wvl < wvl2)
            s_f[~z_m] = np.nan

        else:
            s_f = s / np.nanmax(s)

        a0v_flattened.append(s_f)

    return a0v_flattened


if __name__ == "__main__":

    a0v_spec = A0VSpec()
    tel_trans = TelluricTransmission()

    if 1:

        import json
        wvlsol_products = json.load(open("calib/primary/20140525/SKY_SDCH_20140525_0029.wvlsol_v1.json"))

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = map(np.array, wvlsol_products["wvl_sol"])

        wvl_limits = []
        for wvl_ in wvl_solutions:
            wvl_limits.extend([wvl_[0], wvl_[-1]])

        dwvl = abs(wvl_[0] - wvl_[-1])*0.1 # padding

        wvl1 = min(wvl_limits) - dwvl
        wvl2 = max(wvl_limits) + dwvl

        of_prod = json.load(open("calib/primary/20140525/ORDERFLAT_SDCH_20140525_0074.json"))

        new_orders = of_prod["orders"]
        i1i2_list_ = of_prod["i1i2_list"]

        order_indices = []

        for o in orders_w_solutions:
            o_new_ind = np.searchsorted(new_orders, o)
            order_indices.append(o_new_ind)


        i1i2_list = []
        for o_index in order_indices:
            i1i2_list.append(i1i2_list_[o_index])

    a0v_wvl, a0v_tel_trans, a0v_tel_trans_masked = get_a0v(a0v_spec,
                                                           wvl1, wvl2,
                                                           tel_trans)

    s_list = list(pyfits.open("outdata/20140525/SDCH_20140525_0016.spec.fits")[0].data)

    order_flat_meanspec = np.array(of_prod["mean_order_specs"])

    # for s, v in zip(s_list, order_flat_meanspec):
    #    s[v<np.nanmax(v)*0.1] = np.nan

    a0v_flattened = get_flattend(a0v_spec,
                                 a0v_wvl, a0v_tel_trans_masked,
                                 wvl_solutions, s_list,
                                 i1i2_list=i1i2_list)

    for wvl, s2 in zip(wvl_solutions, a0v_flattened):
        plot(wvl, s2)

    plot(a0v_wvl, a0v_tel_trans)
