import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn


from libs.path_info import IGRINSPath, IGRINSLog
import astropy.io.fits as pyfits

from libs.products import PipelineProducts
from libs.apertures import Apertures

#from libs.products import PipelineProducts

if __name__ == "__main__":

    if 0:
        utdate = "20140316"
        log_today = dict(flat_off=range(2, 4),
                         flat_on=range(4, 7),
                         thar=range(1, 2))
    elif 1:
        utdate = "20140525"
        log_today = dict(flat_off=range(64, 74),
                         flat_on=range(74, 84),
                         thar=range(3, 8),
                         sky=[29],
                         HIP94620=[50,51,52,53],
                         HIP99742=[54, 55, 56, 57],
                         PCyg=[58,59,60, 61],
                         J1833=[42,43,44,45],
                         TWHya=[8, 9, 10, 11],
                         G11=[32, 33],
                         SagARing=[30,31])

    igr_path = IGRINSPath(utdate)

    igrins_log = IGRINSLog(igr_path, log_today)

    band = "H"

if 1: # now extract from differnt slit positions to measure the distortion


    object_name = "sky"
    object_Name = "Sky"



    if 1:

        sky_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["sky"]]

        sky_master_fn_ = os.path.splitext(os.path.basename(sky_names[0]))[0]
        sky_master_fn = igr_path.get_secondary_calib_filename(sky_master_fn_)

        raw_spec_products = PipelineProducts.load(sky_master_fn+".raw_spec")


    if 1: # load aperture product
        flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
        flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
        aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


        aperture_solution_products = PipelineProducts.load(aperture_solutions_name)

        bottomup_solutions = aperture_solution_products["bottom_up_solutions"]


        from libs.process_thar import ThAr

        thar_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["thar"]]
        thar = ThAr(thar_names)

        fn = thar.get_product_name(igr_path)

        thar_products = PipelineProducts.load(fn)


    if 1:
        from libs.master_calib import load_sky_ref_data

        ref_utdate = "20140316"

        sky_ref_data = load_sky_ref_data(ref_utdate, band)


        ohlines_db = sky_ref_data["ohlines_db"]
        ref_ohline_indices = sky_ref_data["ohline_indices"]

        fn = thar.get_product_name(igr_path)+".oh_wvlsol"
        wvlsol_products = PipelineProducts.load(fn)

        orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = wvlsol_products["wvl_sol"]

    if 1: # make aperture

        _o_s = dict(zip(raw_spec_products["orders"], bottomup_solutions))
        ap =  Apertures(orders_w_solutions,
                        [_o_s[o] for o in orders_w_solutions])


        order_map = ap.make_order_map()
        slitpos_map = ap.make_slitpos_map()
        order_map2 = ap.make_order_map(mask_top_bottom=True)


### real

    if 1:

        # INPUT
        flat_off_filenames = [igrins_log.get_filename(band, i) for i \
                              in igrins_log.log["flat_off"]]


        flat_off_name_ = flat_off_filenames[0]
        flat_off_name_ = os.path.splitext(flat_off_name_)[0] + ".flat_off_params"
        flat_off_name = igr_path.get_secondary_calib_filename(flat_off_name_)
        flatoff_products = PipelineProducts.load(flat_off_name)


        # load flat on products
        flat_on_filenames = [igrins_log.get_filename(band, i) for i \
                             in igrins_log.log["flat_on"]]
        flat_on_name_ = flat_on_filenames[0]
        flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".flat_on_params"
        flat_on_name = igr_path.get_secondary_calib_filename(flat_on_name_)

        flaton_products = PipelineProducts.load(flat_on_name)

        pix_mask  = flatoff_products["bpix_mask"] | flaton_products["deadpix_mask"]



        #
        flat_on_name_ = igrins_log.get_filename(band, igrins_log.log["flat_on"][0])
        flat_on_name_ = os.path.splitext(flat_on_name_)[0] + ".aperture_solutions"
        aperture_solutions_name = igr_path.get_secondary_calib_filename(flat_on_name_)


        aperture_solution_products = PipelineProducts.load(aperture_solutions_name)


        fn = thar.get_product_name(igr_path)+".orderflat"
        orderflat_products = PipelineProducts.load(fn)
        orderflat = orderflat_products['order_flat']
        orderflat[pix_mask] = np.nan

    if 1:
        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["HIP94620"]]
        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["HIP99742"]]
        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["PCyg"]]
        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["J1833"]]

        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["G11"]]
        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log["SagARing"]]

        abba_names = [igrins_log.get_filename(band, fn) for fn \
                      in igrins_log.log[objname]]

        objname = "TWHya"
        if len(abba_names) == 4:
            IF_POINT_SOURCE = True
            ab_names_list = [abba_names[:2], abba_names[2:4][::-1]]
        elif len(abba_names) == 2:
            IF_POINT_SOURCE = False
            ab_names_list = [abba_names[:2]]


        if 1:
            #ab_names = ab_names_list[0]

            a_list = [pyfits.open(ab_names[0])[0].data \
                      for ab_names in ab_names_list]
            b_list = [pyfits.open(ab_names[1])[0].data \
                      for ab_names in ab_names_list]


            # we may need to detrip

            # first define extract profile (gaussian).


            dx = 100

            a_data = np.sum(a_list, axis=0)
            b_data = np.sum(b_list, axis=0)
            data_minus = a_data - b_data

            if 1:
                data_minus = destriper.get_destriped(data_minus,
                                                     ~np.isfinite(data_minus),
                                                     pattern=64)

            data_minus_flattened = data_minus / orderflat
            data_plus = (a_data + b_data)

            from libs.destriper import destriper
            bias_mask = flaton_products["flat_mask"] & (order_map2 > 0)
            import scipy.ndimage as ni
            bias_mask2 = ni.binary_dilation(bias_mask)

            gain = 5. # what is the gain??

            # random noise
            variance0 = data_minus

            variance_ = variance0.copy()
            variance_[bias_mask2] = np.nan
            variance_[pix_mask] = np.nan

            st = np.nanstd(variance_)
            st = np.nanstd(variance_[np.abs(variance_) < 3*st])

            variance_[np.abs(variance_) > 3*st] = np.nan

            variance = destriper.get_destriped(variance0,
                                                ~np.isfinite(variance_),
                                               pattern=64)

            variance_ = variance.copy()
            variance_[bias_mask2] = np.nan
            variance_[pix_mask] = np.nan

            st = np.nanstd(variance_)
            st = np.nanstd(variance_[np.abs(variance_) < 3*st])

            variance_[np.abs(variance_) > 3*st] = np.nan

            x_std = ni.median_filter(np.nanstd(variance_, axis=0), 11)

            variance_map0 = np.zeros_like(variance) + x_std**2



            variance_map = variance_map0 + np.abs(data_plus)/gain # add poison noise in ADU
            # we ignore effect of flattening

            # now estimate lsf


            # estimate lsf
            ordermap_bpixed = order_map.copy()
            ordermap_bpixed[pix_mask] = 0


        if IF_POINT_SOURCE: # if point source

            x1, x2 = 800, 1200
            bins, lsf_list = ap.extract_lsf(ordermap_bpixed, slitpos_map,
                                            data_minus_flattened,
                                            x1, x2, bins=None)


            hh0 = np.sum(lsf_list, axis=0)
            peak1, peak2 = max(hh0), -min(hh0)
            lsf_x = 0.5*(bins[1:]+bins[:-1])
            lsf_y = hh0/(peak1+peak2)

            from scipy.interpolate import UnivariateSpline
            lsf_ = UnivariateSpline(lsf_x, lsf_y, k=3, s=0,
                                    bbox=[0, 1])
            roots = list(lsf_.roots())
            #assert(len(roots) == 1)
            integ_list = []
            from itertools import izip, cycle
            for ss, int_r1, int_r2 in izip(cycle([1, -1]),
                                                  [0] + roots,
                                                  roots + [1]):
                #print ss, int_r1, int_r2
                integ_list.append(lsf_.integral(int_r1, int_r2))
            integ = np.abs(np.sum(integ_list))

            def lsf(o, x, slitpos):
                return lsf_(slitpos) / integ

            # make weight map
            profile_map = ap.make_profile_map(order_map, slitpos_map, lsf)

            #
            import astropy.io.fits as pyfits
            slitoffset_map = pyfits.open("t.fits")[0].data

            # extract spec

            s_list, v_list = ap.extract_stellar(ordermap_bpixed,
                                                profile_map,
                                                variance_map,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map)

            # make synth_spec : profile * spectra
            synth_map = ap.make_synth_map(order_map, slitpos_map,
                                          profile_map, s_list,
                                          slitoffset_map=slitoffset_map)

            sig_map = (data_minus_flattened - synth_map)**2/variance_map
            ## mark sig_map > 100 as cosmicay. The threshold need to be fixed.


            # reextract with new variance map and CR is rejected
            variance_map_r = variance_map0 + np.abs(synth_map)/gain
            variance_map2 = np.max([variance_map, variance_map_r], axis=0)
            variance_map2[np.abs(sig_map) > 100] = np.nan

            # extract spec

            s_list, v_list = ap.extract_stellar(ordermap_bpixed, profile_map,
                                                variance_map2,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map)

        else: # if extended source
            from scipy.interpolate import UnivariateSpline
            lsf_ = UnivariateSpline([0, 1], [1., 1.], k=1, s=0,
                                    bbox=[0, 1])

            def lsf(o, x, slitpos):
                return lsf_(slitpos)

            profile_map = ap.make_profile_map(order_map, slitpos_map, lsf)

            s_list, v_list = ap.extract_stellar(ordermap_bpixed,
                                                profile_map,
                                                variance_map,
                                                data_minus_flattened,
                                                slitoffset_map=slitoffset_map
                                                )


            # # make synth_spec : profile * spectra
            # synth_map = ap.make_synth_map(order_map, slitpos_map,
            #                               lsf, s_list,
            #                               slitoffset_map=slitoffset_map)

        if 1:
            new_orders = orderflat_products["orders"]
            fitted_response = orderflat_products["fitted_responses"]
            i1i2_list = orderflat_products["i1i2_list"]

            fig1 = plt.figure(1)
            ax1 = fig1.add_subplot(211)
            ax2 = fig1.add_subplot(212)
            #from libs.stddev_filter import window_stdev
            for o, wvl, s, v in zip(ap.orders, wvl_solutions,
                                    s_list, v_list):

                o_new_ind = np.searchsorted(new_orders, o)
                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                #res = fitted_response[o_new_ind]
                ax1.plot(wvl[sl], s[sl])

                ax2.plot(wvl[sl], s[sl]/v[sl]**.5)
                #s_std = window_stdev(s, 25)
                #ax1.plot(wvl[sl], ni.median_filter(s[sl]/s_std[sl], 10), "g-")



        if IF_POINT_SOURCE: # if point source, try simple telluric factor for A0V
            new_orders = orderflat_products["orders"]
            fitted_response = orderflat_products["fitted_responses"]
            i1i2_list = orderflat_products["i1i2_list"]

            fig2 = plt.figure(2)
            ax1 = fig2.add_subplot(211)
            for o, wvl, s in zip(ap.orders, wvl_solutions, s_list):
                o_new_ind = np.searchsorted(new_orders, o)
                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                res = fitted_response[o_new_ind]
                #ax1.plot(wvl[sl], (s/res)[sl])

            d = np.genfromtxt("A0V/vegallpr25.50000resam5")

            wvl, flux, cont = (d[:,i] for i in [0, 1, 2])
            ax2 = fig2.add_subplot(212, sharex=ax1)
            wvl = wvl/1000.
            mask_H = (1.450 < wvl) & (wvl < 1.850)

            telluric = pyfits.open("telluric/LBL_A15_s0_w050_R0060000_T.fits")[1].data
            telluric_lam = telluric["lam"]
            tel_mask_H = (1.450 < telluric_lam) & (telluric_lam < 1.850)
            #plot(telluric_lam[tel_mask_H], telluric["trans"][tel_mask_H])
            from scipy.interpolate import interp1d, UnivariateSpline
            spl = UnivariateSpline(telluric_lam[tel_mask_H],
                                   telluric["trans"][tel_mask_H],
                                   k=1,s=0)

            spl = interp1d(telluric_lam[tel_mask_H],
                           telluric["trans"][tel_mask_H],
                           bounds_error=False
                           )

            #plot(telluric_lam[tel_mask_H],
            #     spl(telluric_lam[tel_mask_H]))
            #www = np.linspace(1.4, 1.8, 1000)
            #plot(www,
            #     spl(www))
            trans = spl(wvl[mask_H])
            ax1.plot(wvl[mask_H], flux[mask_H]/cont[mask_H]*trans,
                     color="0.5", zorder=0.5)
            #ax2.plot(wvl[mask_H], flux[mask_H]/cont[mask_H]*trans)


            trans_m = ni.maximum_filter(trans, 128)
            trans_mg = ni.gaussian_filter(trans_m, 32)

            zzz0 = flux[mask_H]/cont[mask_H]
            zzz = zzz0*trans
            mmm = trans/trans_mg > 0.95
            zzz[~mmm] = np.nan
            wvl_zzz = wvl[mask_H]
            #ax2.plot(, zzz)

            #ax2 = subplot(212)
            for o, wvl, s in zip(ap.orders, wvl_solutions, s_list):
                o_new_ind = np.searchsorted(new_orders, o)

                i1, i2 = i1i2_list[o_new_ind]
                sl = slice(i1, i2)
                wvl1, wvl2 = wvl[i1], wvl[i2]
                #wvl1, wvl2 = wvl[0], wvl[-1]
                z_m = (wvl1 < wvl_zzz) & (wvl_zzz < wvl2)

                ss = interp1d(wvl, s)

                s_interped = ss(wvl_zzz[z_m])

                xxx, yyy = wvl_zzz[z_m], s_interped/zzz[z_m]

                from astropy.modeling import models, fitting
                p_init = models.Chebyshev1D(domain=[xxx[0], xxx[-1]],
                                            degree=6)
                fit_p = fitting.LinearLSQFitter()
                x_m = np.isfinite(yyy)
                p = fit_p(p_init, xxx[x_m], yyy[x_m])
                #ax2.plot(xxx, yyy)
                #ax2.plot(xxx, p(xxx))

                res_ = p(wvl[sl])

                z_interp = interp1d(xxx, zzz0[z_m], bounds_error=False)
                A0V = z_interp(wvl[sl])
                res_[res_<0.3*res_.max()] = np.nan
                ax1.plot(wvl[sl], (s[sl]/res_))
                ax2.plot(wvl[sl], (s[sl]/res_)/A0V)

            ax1.axhline(1, color="0.5")
            ax2.axhline(1, color="0.5")

                #res = fitted_response[o_new_ind]
                #ax1.plot(wvl[sl], s[sl])

            #oi = 10
if 0:


            def check_lsf(bins, lsf_list):
                from matplotlib.figure import Figure
                fig = Figure()
                ax = fig.add_subplot(111)

                xx = 0.5*(bins[1:]+bins[:-1])
                for hh0 in lsf_list:
                    peak1, peak2 = max(hh0), -min(hh0)
                    ax.plot(xx, hh0/(peak1+peak2),
                            "-", color="0.5")

                hh0 = np.sum(lsf_list, axis=0)
                peak1, peak2 = max(hh0), -min(hh0)
                ax.plot(0.5*(bins[1:]+bins[:-1]), hh0/(peak1+peak2),
                        "-", color="k")

                return fig


            def fit_lsf(bins, lsf_list):

                xx = 0.5*(bins[1:]+bins[:-1])
                hh0 = np.sum(lsf_list, axis=0)
                peak_ind1, peak_ind2 = np.argmax(hh0), np.argmin(hh0)
                # peak1, peak2 =
