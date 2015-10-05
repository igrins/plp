
import numpy as np
import libs.fits as pyfits
import scipy.ndimage as ni


class Hitran(object):

    wvl_min = 2.29
    wvl_max = 2.49

    smooth_lambda = 0.0007 # um

    def __init__(self):
        pass


    @classmethod
    def get_median_filtered_spec(self, wvl, s):

        n = len(wvl)
        wvl_max, wvl_min = max(wvl), min(wvl)

        smooth_pix = int(self.smooth_lambda * n/ (wvl_max - wvl_min))

        s_m = ni.median_filter(s, smooth_pix)

        return s - s_m

    def load_cires(self):

        f = pyfits.open("crires/CR_GCAT_061130A_lines_hitran.fits")
        d = f[1].data

        wvl = d["Wavelength"]*1.e-3
        weight = d["Emission"]/.5e-11

        i1 = np.searchsorted(wvl, self.wvl_min)
        i2 = np.searchsorted(wvl, self.wvl_max)

        s1 = weight[i1:i2]
        wvl1 = wvl[i1:i2]

        s1_m = self.get_median_filtered_spec(wvl1, s1)

        return wvl1, s1_m



def fit_hitrans_wvl(s_list, wvl_list, ref_wvl_list):
    """
    dict
    """
    from scipy.interpolate import interp1d

    ref_pixel_list = {}
    ref_wvl_list_revised = {}
    for i, ref_wvl in ref_wvl_list.items():
        wvl = wvl_list[i]
        xx = np.arange(len(wvl))
        wvl2pix = interp1d(wvl, xx, bounds_error=False)
        ref_pixel = [wvl2pix(wvl1) for wvl1 in ref_wvl]
        #ref_pixel_list[i] = ref_pixel

        nan_filter = [np.all(np.isfinite(p)) for p in ref_pixel]

        # there could be cases when the ref lines fall out of bounds,
        # resulting nans.
        ref_wvl_list_revised[i] = [r for r, m in zip(ref_wvl, nan_filter) if m]
        # import operator
        # ref_wvl_list_revised[i] = reduce(operator.add,
        #                                  ref_wvl_list_revised_,
        #                                  [])

        ref_pixel_list[i] = [r for r, m in zip(ref_pixel, nan_filter) if m]

    igr_sol_list = fit_hitrans_pixel(s_list, ref_pixel_list)
    # pixel values in the igr_sol_list is the position of the first pixel if multiplet.

    keys = igr_sol_list.keys()

    for k in keys:
        s = ref_pixel_list[k]
        dpix = np.array([np.mean(s1-s1[0]) for s1 in s])
        igr_sol_list[k]["pixel"] = igr_sol_list[k]["pixel"] + dpix
        ref_wvl_list_revised[k] = map(np.mean, ref_wvl_list_revised[k])

    return ref_wvl_list_revised, igr_sol_list, ref_pixel_list


def fit_hitrans_pixel(s_dict, ref_pixel_dict):
    """
    s_list : dict
    ref_pixel_list : dict
    """
    igr_sol_list = {}

    from fit_gaussian import fit_gaussian_simple

    for i, ref_pixels in ref_pixel_dict.items():

        s_igr = s_dict[i]
        xx = np.arange(len(s_igr))

        sol_list = []
        for ll in ref_pixels:
            sol = fit_gaussian_simple(xx, s_igr, ll,
                                      sigma_init=5, do_plot=False)
            sigma_init = max(1, sol[0][1])
            sol = fit_gaussian_simple(xx, s_igr,
                                      sol[0][0]+ll-ll[0],
                                      sigma_init=sigma_init, do_plot=False)
            sol_list.append(sol)


        pixel = [sol_[0][0] for sol_ in sol_list]
        pixel_sigma = [sol_[0][1] for sol_ in sol_list]

        igr_sol_list[i] = dict(pixel=pixel,
                               pixel_sigma=pixel_sigma)

    return igr_sol_list


def bootstrap(utdate):
    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 94)


    hitran = Hitran()


    from hitran_igrins import order as hitrans_detected
    from hitran_igrins import ig_order as hitrans_ig_detected
    from fit_gaussian import fit_gaussian_simple

    band = "K"

    import json
    s_list = json.load(open("arc_spec_sky_%s_%s.json" % (band, utdate)))
    wvl_sol = json.load(open("ecfit/wvl_sol_ohlines_%s_%s.json" \
                             % (band, utdate)))


    # emasure line coordinates from hitran data

    hitran_wvl, hitran_s = hitran.load_cires()

    hitran_sol_list = []

    for i in range(5):
        # fit hitran data
        sol_list = []

        dwvl_list = []

        for ll in hitrans_detected[i]:
            sol = fit_gaussian_simple(hitran_wvl, hitran_s,
                                      ll,
                                      sigma_init=7e-5, do_plot=False)
            dwvl = np.array(ll)-ll[0]
            sol = fit_gaussian_simple(hitran_wvl, hitran_s,
                                      sol[0][0]+dwvl,
                                      sigma_init=7e-5, do_plot=False)
            dwvl_list.append(dwvl)
            sol_list.append(sol)


        wavelength = [sol_[0][0] for sol_ in sol_list]
        wavelength_grouped = [list(sol_[0][0]+dwvl_) for dwvl_, sol_ \
                              in zip(dwvl_list, sol_list)]
        wavelength_sigma = [sol_[0][1] for sol_ in sol_list]
        hitran_sol_list.append(dict(wavelength=wavelength,
                                    wavelength_grouped=wavelength_grouped,
                                    wavelength_sigma=wavelength_sigma))

    hitran_sol_dict = dict(zip(igrins_orders["K"], hitran_sol_list))

    # now get the pixel corrdinates from observed data.

    med_filter = hitran.get_median_filtered_spec
    s_list_m = [med_filter(wvl_sol[i], s_list[i]) for i in range(5)]
    s_list_dict = dict(zip(igrins_orders[band],
                           s_list_m))

    ref_pixel_dict = dict((igrins_orders[band][i], r) for i, r \
                          in hitrans_ig_detected.items())

    igr_sol = fit_hitrans_pixel(s_list_dict, ref_pixel_dict)


    # merge tow data
    sol_sol = {}

    for i in igr_sol:
        s0 = dict()
        s0.update(hitran_sol_dict[i])
        s0.update(igr_sol[i])
        sol_sol[i] = s0

    if 1:
        json.dump(sol_sol,
                  open("hitran_bootstrap_%s_%s.json" % (band, utdate), "w"))



def reidentify(orders_w_solutions, wvl_sol, s_list, bootstrap):
    """
    dict
    """
    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 94)

    hitran = Hitran()

    band = "K"

    # now get the pixel corrdinates from observed data.

    med_filter = hitran.get_median_filtered_spec
    bootstrapped_order = range(72, 94)[:5]
    s_list_dict = dict()
    for o in bootstrapped_order:
        if o in orders_w_solutions:
            i = orders_w_solutions.index(o)
            s = med_filter(wvl_sol[i], s_list[i])
            s_list_dict[o] = s

    # s_list_m = [med_filter(wvl_sol[i], s_list[i]) for i in range(5)]
    # s_list_dict = dict(zip(orders_w_solutions, #igrins_orders[band],
    #                        s_list_m))

    h_dict = dict((int(i), r) for i, r in bootstrap.items())

    ref_wvl_dict = dict((i, r["wavelength_grouped"]) for i, r \
                        in h_dict.items())

    wvl_sol_dict = dict(zip(orders_w_solutions, wvl_sol))

    ref_wvl_dict_v2, igr_sol, ref_pixel_list = fit_hitrans_wvl(s_list_dict,
                                                               wvl_sol_dict,
                                                               ref_wvl_dict)

    # merge tow data
    sol_sol = {}

    for i in igr_sol:
        s0 = dict()
        s0["wavelength"] = ref_wvl_dict_v2[i]
        #s0.update(h_dict[i])
        s0.update(igr_sol[i])
        sol_sol[i] = s0

    return sol_sol, ref_pixel_list

if __name__ == "__main__":

    # bootstrap
    utdate = "20140316"


    if 0:
        bootstrap(utdate)


    if 0:
        import json
        band = "K"
        utdate = "20140316"
        s_list = json.load(open("arc_spec_sky_%s_%s.json" % (band, utdate)))
        wvl_sol = json.load(open("ecfit/wvl_sol_ohlines_%s_%s.json" \
                             % (band, utdate)))

        bootstrap_name = "hitran_bootstrap_K_20140316.json"
        bootstrap = json.load(open(bootstrap_name))

        reidentify(wvl_sol, s_list, bootstrap)

    if 1:
        band = "K"
        utdate = "20140525"
        import json
        s_list = json.load(open("arc_spec_sky_%s_%s.json" % (band, utdate)))
        wvl_sol = json.load(open("wvl_sol_phase0_%s_%s.json" \
                                 % (band, utdate)))
        bootstrap_name = "hitran_bootstrap_K_20140316.json"
        bootstrap = json.load(open(bootstrap_name))

        sol_sol = reidentify(wvl_sol, s_list, bootstrap)

        from json_helper import json_dump
        if 1:
            json_dump(sol_sol,
                      open("hitran_reidentified_%s_%s.json" % (band, utdate),
                           "w"),
                      )


if 0:
        if 0:
            import matplotlib.pyplot as plt
            plt.clf()
            ax = plt.subplot(311)
            ax.plot(wvl_igr, s_igr_m)
            ax.plot(wvl1, ss)

            for ll, sol in zip(hitrans_detected[i], sol_list):
                xx = sol[0][0]+np.array(ll)-ll[0]
                ax.vlines(xx, ymin=0, ymax=sol[0][2])
                ax.hlines(sol[0][2]*0.5,
                          xmin=xx-sol[0][1], xmax=xx+sol[0][1])


        if 0:
            ax2 = plt.subplot(312, sharex=ax)
            ax2.plot(wvl_igr, s_list[i])
            ax2.plot(wvl_igr, s_list[i] - s1_igr_m)

            ax2.set_xlim(wvl_igr[0], wvl_igr[-1])


            ax3 = plt.subplot(313)
            ax3.plot(sss)
            for ll in hitrans_ig_detected[i]:
                ax3.vlines(ll, ymin=0, ymax=5)
if 0:
        import json
        json.dump(sol_sol,
                  open("hitran_K_20140316.json", "w"))
