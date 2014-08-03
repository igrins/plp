import os
import numpy as np

import astropy.io.fits as pyfits
from stsci_helper import stsci_median

from products import PipelineProducts

class ThAr(object):
    def __init__(self, thar_names):
        self.file_names = thar_names

    def process_thar(self, ap):
        r = _process_thar(self.file_names, ap)
        return r

    def get_product_name(self, igr_path):
        first_name_ = self.file_names[0]
        first_name = os.path.splitext(first_name_)[0] + ".median_spectra"
        return igr_path.get_secondary_calib_filename(first_name)


def _process_thar(thar_names, ap):
    hdu_list = [pyfits.open(fn)[0] for fn in thar_names]
    _data = stsci_median([hdu.data for hdu in hdu_list])

    from destriper import destriper
    data = destriper.get_destriped(_data)

    s = ap.extract_spectra_v2(data)

    r = PipelineProducts("1d median specs",
                         combined_image=data,
                         specs=s)

    return r

def get_master_calib_abspath(fn):
    import os
    return os.path.join("master_calib", fn)


def load_thar_ref_data(ref_date, band):
    import json
    # load spec

    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 92)

    ref_spec_file = "arc_spec_thar_%s_%s.json" % (band, ref_date)
    ref_id_file = "thar_identified_%s_%s.json" % (band, ref_date)

    s_list_ = json.load(open(get_master_calib_abspath(ref_spec_file)))
    s_list_src = [np.array(s) for s in s_list_]

    # reference line list : from previous run
    ref_lines_list = json.load(open(get_master_calib_abspath(ref_id_file)))

    r = dict(ref_date=ref_date,
             band=band,
             ref_spec_file=ref_spec_file,
             ref_id_file=ref_id_file,
             ref_lines_list=ref_lines_list,
             ref_s_list=s_list_src,
             orders=igrins_orders[band])

    return r

def match_order_thar(thar_products, thar_ref_data):
    import numpy as np

    orders_src = thar_ref_data["orders"]
    s_list_src = thar_ref_data["ref_s_list"]

    # load spec
    #s_list_ = json.load(open("arc_spec_thar_%s_%s.json" % (band, date)))
    s_list_ = thar_products["specs"]
    s_list_dst = [np.array(s) for s in s_list_]

    # match the orders of s_list_src & s_list_dst
    from libs.reidentify_thar_lines import match_orders
    delta_indx, orders_dst = match_orders(orders_src, s_list_src,
                                          s_list_dst)

    return orders_dst


def reidentify_ThAr_lines(thar_products, thar_ref_data):
    import numpy as np

    orders_src = thar_ref_data["orders"]
    s_list_src = thar_ref_data["ref_s_list"]

    # load spec
    #s_list_ = json.load(open("arc_spec_thar_%s_%s.json" % (band, date)))
    orders_dst = thar_products["orders"]
    s_list_ = thar_products["specs"]
    s_list_dst = [np.array(s) for s in s_list_]

    orders_intersection = set(orders_src).intersection(orders_dst)

    def filter_order(orders, s_list, orders_intersection):
        s_list_filtered = [s for o, s in zip(orders, s_list) if o in orders_intersection]
        return s_list_filtered

    s_list_src = filter_order(orders_src, s_list_src, orders_intersection)
    s_list_dst = filter_order(orders_dst, s_list_dst, orders_intersection)

    ref_lines_list = filter_order(orders_src,
                                  thar_ref_data["ref_lines_list"],
                                  orders_intersection)

    orders = sorted(orders_intersection)

    from libs.reidentify_thar_lines import get_offset_transform
    # get offset function from source spectra to target specta.
    sol_list_transform = get_offset_transform(s_list_src, s_list_dst)

    from libs.reidentify import reidentify_lines_all2

    #ref_lines_map = dict(zip(orders_src, ref_lines_list))



    #ref_lines_list_dst = [ref_lines_map[o] for o in orders_dst]
    reidentified_lines_with_id = reidentify_lines_all2(s_list_dst,
                                                       ref_lines_list,
                                                       sol_list_transform)

    r = PipelineProducts("initial reidentification of ThAr lines",
                         orders=orders,
                         match_list=reidentified_lines_with_id,
                         #ref_date=ref_date,
                         ref_spec_file=thar_ref_data["ref_spec_file"],
                         ref_id_file=thar_ref_data["ref_id_file"])

    return r

def load_echelogram(ref_date, band):
    from echellogram import Echellogram

    echel_name = get_master_calib_abspath("fitted_echellogram_sky_%s_%s.json" % (band, ref_date))
    echel = Echellogram.from_json_fitted_echellogram_sky(echel_name)

    return echel

def align_echellogram_thar(thar_reidentified_products, echel, band, ap):

    orders = thar_reidentified_products["orders"]

    th = np.genfromtxt("ThArlines.dat")
    wvl_thar = th[:,0]/1.e4
    #s_thar = np.clip(th[:,1], a_min=20, a_max=np.inf)

    # line_list : dict of (order, (pixel coord list, wavelengths))
    wvl_list = {}
    pixel_list = {}
    match_list = thar_reidentified_products["match_list"]
    for o, s in zip(orders, match_list):
        lineid_list = s[0] # [s1[0] for s1 in s]
        wvl = wvl_thar[lineid_list]
        wvl_list[o] = wvl
        x = [s1[0] for s1 in s[1]]
        pixel_list[o] = x

    xy1f, nan_mask = echel.get_xy_list_filtered(wvl_list)
    xy2f = ap.get_xy_list(pixel_list, nan_mask)

    from libs.align_echellogram_thar import fit_affine_clip
    affine_tr, mm = fit_affine_clip(xy1f, xy2f)

    r = PipelineProducts("ThAr aligned echellogram products",
                         xy1f=xy1f, xy2f=xy2f,
                         affine_tr=affine_tr,
                         affine_tr_mask=mm)

    return r



def plot_detected(ax1, im, x_det, y_det):

    ax1.imshow(im, origin="lower",
               cmap="gray_r",
               vmin=0, vmax=63)

    ax1.plot(x_det, y_det, "ro", ms=3)



def check_dx1(ax, x, y, dx, gi, mystd):

    grid_z2 = gi(x, y, dx)
    im = ax.imshow(grid_z2, origin="lower", aspect="auto",
                   extent=(gi.xi[0], gi.xi[-1], gi.yi[0], gi.yi[-1]),
                   interpolation="none")
    im.set_clim(-mystd, mystd)
    return im

def check_dx2(ax, x, y, dx):
    m1 = dx >= 0
    ax.scatter(x[m1], y[m1], dx[m1]*10, color="r")
    m2 = dx < 0
    ax.scatter(x[m2], y[m2], -dx[m2]*10, color="b")

# def check_dx(ax, ax2, im, x, y, dx, mystd):
#     m1 = dx >= 0
#     ax.scatter(x[m1], y[m1], dx[m1]*10, color="r")
#     m2 = dx < 0
#     ax.scatter(x[m2], y[m2], -dx[m2]*10, color="b")

#     # Rbf does not seem to work for some reason
#     from scipy.interpolate import griddata
#     YI, XI = np.mgrid[0:2048, 0:2048]
#     grid_z2 = griddata(np.array((x, y)).T, dx,
#                        (XI, YI), method='cubic')

#     im = ax2.imshow(grid_z2, origin="lower")
#     im.set_clim(-mystd, mystd)



def check_thar_transorm(thar_products, thar_echell_products):
    # to check the fit results.

    combined_im = thar_products["combined_image"]

    affine_tr = thar_echell_products["affine_tr"]
    affine_tr_mask = thar_echell_products["affine_tr_mask"]
    xy1f, xy2f = thar_echell_products["xy1f"], thar_echell_products["xy2f"]

    xy1f_tr = affine_tr.transform(xy1f) #[:,0], xy1f[:,1])


    dx_ = xy1f_tr[:,0] - xy2f[:,0]
    #dy_ = xy1f_tr[:,1] - xy2f[:,1]

    mystd = dx_[affine_tr_mask].std()
    mm = [np.abs(dx_) < 3. * mystd]
    dx = dx_[mm]
    x  = xy1f_tr[:,0][mm]
    y = xy1f_tr[:,1][mm]

    from matplotlib.figure import Figure

    if 0: #plot the coverage of previous echellogram
        import matplotlib.pyplot as plt
        im = igrins_log.get_cal_hdus(band, "thar")[0].data

        zemax_xy_list = get_wvl_range(zdata_band, affine_tr)
        fig1 = plt.figure(figsize=(8, 8))
        ax = fig1.add_subplot(111)
        plot_zemax_coverage(ax, im, zemax_xy_list, band)
        ax.set_xlim(0, 2048)
        ax.set_ylim(0, 2048)
        ax.set_xlabel("x-pixel")
        ax.set_ylabel("y-pixel")
        fig1.tight_layout()

    #check_dx(im, x, y, dx, 3*mystd)

    if 1:
        #orders_band = igrins_orders[band]
        xi = np.linspace(0, 2048, 256+1)
        yi = np.linspace(0, 2048, 256+1)
        # yi = np.linspace(orders_band[0]-1, orders_band[-1]+1,
        #                  len(orders_band)*10)
        from grid_interpolator import GridInterpolator
        gi = GridInterpolator(xi, yi)

        from mpl_toolkits.axes_grid1 import ImageGrid
        #fig2 = plt.figure(figsize=(14, 7))
        fig2 = Figure(figsize=(14, 7))
        grid = ImageGrid(fig2, 111, (1, 2), share_all=True)

        x_det, y_det = x, y
        plot_detected(grid[0], combined_im, x_det, y_det)

        ax2 = grid[1]
        im = check_dx1(ax2, x, y, dx, gi, mystd=2*mystd)
        from mpl_toolkits.decorator import colorbar
        cb = colorbar(im, ax=ax2, loc=1)
        cb.set_label(r"$\Delta\lambda$ [pixel]")
        check_dx2(ax2, x, y, dx)

        ax1 = grid[0]
        ax1.set_xlim(0, 2048)
        ax1.set_ylim(0, 2048)

        ax1.set_xlabel("x-pixel")
        ax1.set_ylabel("y-pixel")

        fig2.tight_layout()

    fig3 = Figure() #plt.figure()
    ax=fig3.add_subplot(111)
    #ax.hist(dx, bins=np.linspace(-10, 10, 50))
    ax.hist(dx, bins=np.linspace(-5*mystd, 5*mystd, 50))
    ax.set_xlabel(r"$\Delta\lambda$ [pixel]")
    ax.set_ylabel("counts")
    fig3.tight_layout()

    return [fig2, fig3]

def get_wavelength_solutions(thar_echellogram_products, echel):
    from ecfit.ecfit import get_ordered_line_data, fit_2dspec, check_fit

    affine_tr = thar_echellogram_products["affine_tr"]

    d_x_wvl = {}
    for order, z in echel.zdata.items():
        xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
        x_T = xy_T[:,0]
        d_x_wvl[order]=(x_T, z.wvl)

    xl, yl, zl = get_ordered_line_data(d_x_wvl)
    # xl : pixel
    # yl : order
    # zl : wvl * order

    x_domain = [0, 2047]
    orders_band = sorted(echel.zdata.keys())
    #orders = igrins_orders[band]
    y_domain = [orders_band[0]-2, orders_band[-1]+2]
    p, m = fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3,
                      x_domain=x_domain, y_domain=y_domain)

    if 0:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 7))
        check_fit(fig, xl, yl, zl, p, orders_band, d_x_wvl)
        fig.tight_layout()


    xx = np.arange(2048)
    wvl_sol = []
    for o in orders_band:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    if 0:
        json.dump(wvl_sol,
                  open("wvl_sol_phase0_%s_%s.json" % \
                       (band, igrins_log.date), "w"))

    r = PipelineProducts("wavelength solution from ThAr",
                         wvl_sol=wvl_sol)

    return r
