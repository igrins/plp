

import numpy as np
from scipy.interpolate import interp1d

from matplotlib.transforms import Affine2D


class StripBase(object):
    def __init__(self, band, order, wvl, x, y):
        self.order = order
        self.band = band
        self.x = x
        self.y = y
        self.wvl = wvl

        self.interp_x = interp1d(self.wvl, self.x, kind='cubic',
                                 bounds_error=False)
        self.interp_y = interp1d(self.wvl, self.y, kind='cubic',
                                 bounds_error=False)


def fit_affine(xy1, xy2):
    """
    affine transfrom from xy1 to xy2
    xy1 : list of (x, y)
    xy2 : list of (x, y)

    Simply using leastsquare
    """

    # xy1f_ = np.empty((3, len(xy1f)))
    # xy1f_[:2, :] = xy1f.T
    # xy1f_[2, :] = 1
    xy1f_ = np.empty((len(xy1), 3))
    xy1f_[:, :2] = xy1
    xy1f_[:, 2] = 1

    abcdef = np.linalg.lstsq(xy1f_, xy2)

    return np.ravel(abcdef[0])


def fit_affine_clip(xy1f, xy2f):
    sol = fit_affine(xy1f, xy2f)

    affine_tr = Affine2D.from_values(*sol)

    xy1f_tr = affine_tr.transform(xy1f) #[:,0], xy1f[:,1])

    # mask and refit
    dx_ = xy1f_tr[:,0] - xy2f[:,0]
    mystd = dx_.std()
    mm = [np.abs(dx_) < 3. * mystd]

    sol = fit_affine(xy1f[mm], xy2f[mm])

    affine_tr = Affine2D.from_values(*sol)

    return affine_tr, mm


def get(ohlines_list, zdata, apcoeffs):
    """
    ohlines_list : dict of tuples of (pixel coord list, wavelengths)
    """
    if 0:
        ohlines_list, zdata, apcoeffs = line_list, zdata_band, trace_solutions

    xy1, xy2 = [], []
    for order_i, (pixel, wvl) in ohlines_list.items():
        #print order_i
        if len(wvl) > 0 and order_i in zdata:
            zz = zdata[order_i]
            xy1.extend(zip(zz.interp_x(wvl), zz.interp_y(wvl)))
            pixel_y = apcoeffs[order_i](pixel)
            xy2.extend(zip(pixel, pixel_y))

    nan_filter = [np.all(np.isfinite([x, y])) for x, y in xy1]
    xy1f = np.compress(nan_filter, xy1, axis=0)
    xy2f = np.compress(nan_filter, xy2, axis=0)

    #print xy1f
    #xy1ftr = tr.transform(xy1f)

    return xy1f, xy2f


def plot_detected_feature_on_zemax(ax1, ax2, im,
                                   x_det, y_det,
                                   zemax_xy_list, band):
    """
    """
    ax1.imshow(im, origin="lower",
               cmap="gray_r",
               vmin=0, vmax=63)

    ax1.plot(x_det, y_det, "ro", ms=3)


    ax2.imshow(im, origin="lower",
               cmap="gray_r",
               vmin=0, vmax=63)

    for x, y in zemax_xy_list:
        ax2.plot(x, y)


def plot_zemax_coverage(ax2, im,
                        zemax_xy_list, band):
    """
    """
    ax2.imshow(im, origin="lower",
               cmap="gray_r",
               vmin=0, vmax=63)

    for x, y in zemax_xy_list:
        ax2.plot(x, y)


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

def check_dx(im, x, y, dx, mystd):
    ax = fig.add_subplot(121, aspect=1)
    m1 = dx >= 0
    ax.scatter(x[m1], y[m1], dx[m1]*10, color="r")
    m2 = dx < 0
    ax.scatter(x[m2], y[m2], -dx[m2]*10, color="b")

    # Rbf does not seem to work for some reason
    from scipy.interpolate import griddata
    YI, XI = np.mgrid[0:2048, 0:2048]
    grid_z2 = griddata(np.array((x, y)).T, dx,
                       (XI, YI), method='cubic')
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(grid_z2, origin="lower")
    im.set_clim(-mystd, mystd)

def get_wvl_solution(zdata_band, affine_tr):
    pixel2wvl_list= []
    for zz in zdata_band:
        #xy = zip(zz.interp_x(wvl), zz.interp_y(wvl))
        xy = zip(zz.x, zz.y)
        xy1f_tr = affine_tr.transform(xy)

        xf = xy1f_tr[:,0]
        pixel2wvl = interp1d(xf, zz.wvl, kind='cubic', bounds_error=False)
        pixel2wvl_list.append(pixel2wvl)
    return pixel2wvl_list


def get_wvl_range(zdata_band, affine_tr):
    xy_list= []
    for ii in sorted(zdata_band.keys()):
        zz = zdata_band[ii]
        #xy = zip(zz.interp_x(wvl), zz.interp_y(wvl))
        xy = zip(zz.x, zz.y)
        xy1f_tr = affine_tr.transform(xy)

        xf = xy1f_tr[:,0]
        yf = xy1f_tr[:,1]
        xy_list.append((xf, yf))
    return xy_list



def check_order(band):
    for order_i, o in enumerate(orders[band]):
        pixel, wvl = ohlines[band][order_i]
        zz = zdata[band][order_i]
        xy = zip(zz.interp_x(wvl), zz.interp_y(wvl))
        xy1f_tr = affine_tr.transform(xy)
        dx = pixel - xy1f_tr[:,0]
        plot(wvl, dx, "o-")




class GridInterpolator(object):
    def __init__(self, xi, yi, interpolator="mlab"):
        self.xi = xi
        self.yi = yi
        self.xx, self.yy = np.meshgrid(xi, yi)
        self._interpolator = interpolator

    def __call__(self, xl, yl, zl):
        if self._interpolator == "scipy":
            from scipy.interpolate import griddata
            x_sample = 256
            z_gridded = griddata(np.array([yl*x_sample, xl]).T,
                                 np.array(zl),
                                 (self.yy*x_sample, self.xx),
                                 method="linear")
        elif self._interpolator == "mlab":
            from matplotlib.mlab import griddata
            z_gridded = griddata(xl, yl, zl, self.xi, self.yi)

        return z_gridded

#####



# Try to match wavelength solution from unknown date to 20140316.
# Just use shift.

#from pipeline_jjlee import IGRINSLog, Destriper
import pickle
import scipy.ndimage as ni



if __name__ == "__main__":


    import json
    date = "20140525"
    band = "H"

    thar_init = json.load(open("thar_shifted_%s_%s.json" % (band, date)))

    log_20140525 = dict(flat_off=range(64, 74),
                        flat_on=range(74, 84),
                        thar=range(3, 8))


    igrins_log = IGRINSLog("20140525", log_20140525)

    # 1. find transform from old to new
    # 2. apply transform to old echellogram
    # 3. derive new solution

    # 1. find transform from old to new
    #   a. for each order
    #     i. convert line id to wavelength
    #     ii. find position from the old echellogram
    #     iii. fit the transform
    # 2. apply transform to old echellogram
    # 3. derive new solution

if 0:

    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 94)


    r = pickle.load(open("flat_info_%s_%s.pickle" % (igrins_log.date, band)))
    r2 = pickle.load(open("thar_%s_%s.pickle" % (igrins_log.date, band)))

    bpix_mask = r["flat_bpix_mask"]
    trace_sol = r["bottomup_solutions"]

    from apertures import Apertures
    apertures = Apertures(igrins_orders[band],
                          trace_sol)


    # load echellogram data
    from echellogram import Echellogram
    echel_name = "fitted_echellogram_sky_%s_%s.json" % (band, thar_init["ref_date"])
    echel = Echellogram.from_json_fitted_echellogram_sky(echel_name)


    th = np.genfromtxt("ThArlines.dat")
    wvl_thar = th[:,0]/1.e4
    s_thar = np.clip(th[:,1], a_min=20, a_max=np.inf)

    # line_list : dict of (order, (pixel coord list, wavelengths))
    wvl_list = {}
    pixel_list = {}
    for o, s in zip(igrins_orders[band], thar_init["match_list"]):
        lineid_list = s[0] # [s1[0] for s1 in s]
        wvl = wvl_thar[lineid_list]
        wvl_list[o] = wvl
        x = [s1[0] for s1 in s[1]]
        pixel_list[o] = x

    xy1f, nan_mask = echel.get_xy_list_filtered(wvl_list)
    xy2f = apertures.get_xy_list(pixel_list, nan_mask)

    affine_tr, mm = fit_affine_clip(xy1f, xy2f)

    # now fit is done.

if 0:
    # The rest is to check the fit results.

    xy1f_tr = affine_tr.transform(xy1f) #[:,0], xy1f[:,1])


    dx_ = xy1f_tr[:,0] - xy2f[:,0]
    dy_ = xy1f_tr[:,1] - xy2f[:,1]

    mystd = dx_[mm].std()
    mm = [np.abs(dx_) < 3. * mystd]
    dx = dx_[mm]
    x  = xy1f_tr[:,0][mm]
    y = xy1f_tr[:,1][mm]

    import matplotlib.pyplot as plt
    im = igrins_log.get_cal_hdus(band, "thar")[0].data
    if 1:
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
        orders_band = igrins_orders[band]
        xi = np.linspace(0, 2048, 256+1)
        yi = np.linspace(0, 2048, 256+1)
        # yi = np.linspace(orders_band[0]-1, orders_band[-1]+1,
        #                  len(orders_band)*10)
        gi = GridInterpolator(xi, yi)

        from mpl_toolkits.axes_grid1 import ImageGrid
        fig2 = plt.figure(figsize=(14, 7))
        grid = ImageGrid(fig2, 111, (1, 2), share_all=True)

        x_det, y_det = x, y
        plot_detected(grid[0], im, x_det, y_det)

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

    fig3 = plt.figure()
    ax=fig3.add_subplot(111)
    #ax.hist(dx, bins=np.linspace(-10, 10, 50))
    ax.hist(dx, bins=np.linspace(-5*mystd, 5*mystd, 50))
    ax.set_xlabel(r"$\Delta\lambda$ [pixel]")
    ax.set_ylabel("counts")
    fig3.tight_layout()

    if 0:
        postfix = "%s_%s" % (igrins_log.date, band)
        fig1.savefig("align_zemax_%s_fig1_coverage.png" % postfix)
        fig2.savefig("align_zemax_%s_fig2_fit.png" % postfix)
        fig3.savefig("align_zemax_%s_fig3_hist_dlambda.png" % postfix)




if 0:
    from ecfit.ecfit import get_ordered_line_data, fit_2dspec, check_fit

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
    orders_band = igrins_orders[band]
    #orders = igrins_orders[band]
    y_domain = [orders_band[0]-2, orders_band[-1]+2]
    p = fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3,
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

    if 1:
        json.dump(wvl_sol,
                  open("wvl_sol_phase0_%s_%s.json" % \
                       (band, igrins_log.date), "w"))
