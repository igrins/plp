from __future__ import print_function

import numpy as np
from .polyfitr import polyfitr
#reload(polyfitr)


# this maybe reformed to be more adaptive.

def trace_aperture_chebyshev(xy_list, domain=None):
    """
    a list of (x_array, y_array).

    y_array must be a masked array
    """
    import numpy.polynomial.chebyshev as cheb

    #for x, y in r["cent_bottom_list"]:
    # xy_list = r["cent_up_list"]
    # domain = [0, 2047]

    # if domain is None:
    #     xmax = max(max(x) for x, y in xy_list)
    #     xmin = min(min(x) for x, y in xy_list)
    #     domain = [xmin, xmax]

    if domain is None:
        domain = [0, 2047]

    # we first fit the all traces with 2d chebyshev polynomials
    x_list, o_list, y_list = [], [], []
    for o, (x, y) in enumerate(xy_list):
        if hasattr(y, "mask"):
            msk = ~y.mask & np.isfinite(y.data)
            y = y.data
        else:
            msk = np.isfinite(np.array(y, "d"))
        x1 = np.array(x)[msk]
        x_list.append(x1)
        o_list.append(np.zeros(len(x1))+o)
        y_list.append(np.array(y)[msk])

    n_o = len(xy_list)

    from astropy.modeling import models, fitting
    from astropy.modeling.polynomial import Chebyshev2D
    x_degree, y_degree = 4, 5
    p_init = Chebyshev2D(x_degree, y_degree,
                         x_domain=domain, y_domain=[0, n_o-1])
    fit_p = fitting.LinearLSQFitter()

    xxx, ooo, yyy = (np.concatenate(x_list),
                     np.concatenate(o_list),
                     np.concatenate(y_list))
    p = fit_p(p_init, xxx, ooo, yyy)

    if 0:
        ax1 = subplot(121)
        for o, xy in enumerate(xy_list):
            ax1.plot(x_list[o], y_list[o] - p(x_list[o], o+np.zeros_like(x_list[o])))

    for ii in range(3): # number of iteration
        mmm = np.abs(yyy - p(xxx, ooo)) < 1 # This need to be fixed with actual estimation of sigma.

        p = fit_p(p_init, xxx[mmm], ooo[mmm], yyy[mmm])

    if 0:
        ax2=subplot(122, sharey=ax1)
        for o, xy in enumerate(xy_list):
            ax2=plot(x_list[o], y_list[o] - p(x_list[o], o+np.zeros_like(x_list[o])))

    # Now we need to derive a 1d chebyshev for each order.  While
    # there should be an analitical way, here we refit the trace for
    # each order using the result of 2d fit.

    xx = np.arange(domain[0], domain[1])
    oo = np.zeros_like(xx)
    def _get_f(o0):
        y_m = p(xx, oo+o0)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        return f

    f_list = []
    ooo = [o[0] for o in o_list]
    #for x, o in zip(x_list, o_list):
    f_list = [_get_f(o0) for o0 in ooo]


    def _get_f_old(next_orders, y_thresh):
        oi = next_orders.pop(0)
        y_m = p(xx, oo+oi)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        if next_orders: # if not the last order
            if np.all(y_thresh(y_m)):
                print("all negative at ", oi)
                next_orders = next_orders[:1]

        return oi, f, next_orders

    def _get_f(next_orders, y_thresh):
        oi = next_orders.pop(0)
        y_m = p(xx, oo+oi)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        if np.all(y_thresh(y_m)):
            print("all negative at ", oi)
            next_orders = []

        return oi, f, next_orders

    # go down in order
    f_list_down = []
    o_list_down = []
    go_down_orders = [ooo[0]-_oi for _oi in range(1, 5)]
    while go_down_orders:
        oi, f, go_down_orders = _get_f(go_down_orders,
                                       y_thresh=lambda y_m: y_m < domain[0])
        f_list_down.append(f)
        o_list_down.append(oi)

    f_list_up = []
    o_list_up = []
    go_up_orders = [ooo[-1]+_oi for _oi in range(1, 5)]
    while go_up_orders:
        oi, f, go_up_orders = _get_f(go_up_orders,
                                     y_thresh=lambda y_m: y_m > domain[-1])
        f_list_up.append(f)
        o_list_up.append(oi)


    if 0:
        _get_f(next_orders)

        oi = go_down_orders.pop(0)
        y_m = p(xx, oo+ooo[0]-oi)
        f = cheb.Chebyshev.fit(xx, y_m, x_degree, domain=domain)
        if go_down_orders: # if not the last order
            if np.all(y_m < domain[0]):
                print("all negative at ", ooo[0]-oi)
                go_down_orders = [oi+1]
            else:
                f_list_down.append(f)
        else:
            f_list_down.append(f)

    print(o_list_down[::-1] + ooo + o_list_up)
    return f_list, f_list_down[::-1] + f_list + f_list_up


def trace_aperture_chebyshev_old(xy_list, domain=None):
    """
    a list of (x_array, y_array).

    y_array must be a masked array
    """
    import numpy.polynomial.chebyshev as cheb

    #for x, y in r["cent_bottom_list"]:
    # xy_list = r["cent_up_list"]
    # domain = [0, 2047]


    if domain is None:
        xmax = max(x.max() for x, y in xy_list)
        xmin = min(x.min() for x, y in xy_list)
        domain = [xmin, xmax]

    #xx = np.arange(domain[0], domain[1]+1)

    # fit with chebyshev polynomical with domain 0..2047
    f_list = []
    for x, y in xy_list:
        msk = ~y.mask & np.isfinite(y.data)
        x1, y1 = x[msk], y.data[msk]
        f = cheb.Chebyshev.fit(x1, y1, 4, domain=domain)
        f_list.append(f)

    # fit 3rd and 2nd order term to remove erroneous value
    o = np.arange(len(f_list))
    # fitting poly order = 1, sigma = 3
    y3_ = [f_.coef[3] for f_ in f_list]
    p3_ = polyfitr(o, y3_, 1, 3)
    p3 = np.polyval(p3_, o)


    # fitting poly order = 2, sigma = 3
    y2_ = [f_.coef[2] for f_ in f_list]
    p2_ = polyfitr(o, y2_, 2, 3)
    p2 = np.polyval(p2_, o)


    # now subtract the fitted 3rd and 2nd term from the original data,
    # and then refit with 1st order poly.
    f2_list = []
    for i, (x, y) in enumerate(xy_list):

        f = f_list[i]
        y_3 = p3[i]*f.basis(3, domain=domain)(x)
        y_2 = p2[i]*f.basis(2, domain=domain)(x)

        y = y - (y_2+y_3)

        msk = ~y.mask & np.isfinite(y.data)
        x1, y1 = x[msk], y.data[msk]

        # fit with 1st poly.
        f_ = cheb.Chebyshev.fit(x1, y1, 1,
                                domain=domain)

        # now, regenerate 3d poly.
        f = cheb.Chebyshev(np.concatenate([f_.coef, [p2[i], p3[i]]]),
                           domain=domain)
        f2_list.append(f)

    return f2_list

if 0:
    i = 2
    x, y = xy_list[i]
    plot(x, y-f_list[i](x))
    plot(x, y-f2_list[i](x))


def test_aperture(xy_list, f_list, domain):
    xx = np.arange(domain[0], domain[1]+1)
    for (x, y), f in zip(xy_list, f_list):
        figure()

        ax1 = subplot(121)
        ax1.plot(x, y-f(x))
        ax1.set_xlim(domain[0], domain[1])

        ax2 = subplot(122, sharey=ax1)
        ax2.hist((y - f(x))[np.isfinite(y)],
                 bins=np.linspace(-2, 2, 41), orientation="horizontal")

        ax2.set_ylim(-1, 1)

##

if 0: # testing code with B-spline

    from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline


    coeffs_list = []
    for x, y in r["cent_bottom_list"]:
        msk = ~y.mask & np.isfinite(y.data)
        x1, y1 = x[msk], y.data[msk]
        xmin = x1.min()
        xmax = x1.max()

        # i1 = max(np.searchsorted(t, xmin)-1, 0)
        # i2 = np.searchsorted(t, xmax)

        m = (xmin <= t) & (t <= xmax)
        m1 = ni.maximum_filter1d(m, 3)
        t1 = t[m1]

        spl = LSQUnivariateSpline(x1, y1,
                                  t1[1:-1], bbox = [t1[0], t1[-1]])

        spl = LSQUnivariateSpline(x1, y1,
                                  t2[1:-3], bbox = [t2[0], t2[-3]])

        coeffs = np.empty(len(t)+2, dtype="f")
        coeffs.fill(np.nan)

        if np.all(m1):
            coeffs[:] = spl.get_coeffs()
        else:
            coeffs[1:-1][m1] = spl.get_coeffs()[1:-1]

        coeffs_list.append([coeffs])
        print(len(spl.get_knots()), len(spl.get_coeffs()))
        plot(x1, y1)
        plot(xx, spl(xx))
    aa = np.concatenate(coeffs_list, axis=0)




    aaf = np.empty_like(aa)
    tt = np.arange(aa.shape[0])
    for ix in range(aa.shape[-1]):
        y_data = aa[:,ix]
        my = np.isfinite(y_data)
        p = np.polyfit(tt[my], y_data[my], 4)
        p = polyfitr.polyfitr(tt[my], y_data[my], 4, 2,
                              fev=100, w=None, diag=False, clip='both',
                              verbose=False, plotfit=False, plotall=False)

        pf = np.polyval(p, tt)
        aaf[:,ix] = pf



    knots = spl.get_knots()
    coeffs = spl.get_coeffs()
    def from_knots_coeffs(knots, coeffs, k=3):
        n = len(knots) + 2*k

        t = np.empty(n, dtype="d")
        t[:k] = knots[0]
        t[k:-k] = knots
        t[-k:] = knots[-1]

        c = np.zeros(n, dtype="d")
        c[:len(coeffs)] = coeffs

        return LSQUnivariateSpline._from_tck((t, c, k))

    k = 3
    t = np.linspace(0, 2048, 16 + 1) #[1:-1]
    spl_list = [from_knots_coeffs(t, c) for c in aaf]
    #spl_list = [from_knots_coeffs(t, c) for c in aa]


    for (x, y), spl in zip(r["cent_bottom_list"], spl_list):
        msk = ~y.mask & np.isfinite(y.data)
        x1, y1 = x[msk], y.data[msk]
        plot(x1, y1)
        plot(xx, spl(xx))
        plot(spl.get_knots())

    if 0:
        ax = subplot(111)
        for x, y in r["cent_bottom_list"]:
            ax.plot(x, y)
        for x, y in r["cent_up_list"]:
            ax.plot(x, y)

        plot_sollutions(r["flat_normed"],
                        r["bottomup_centroids"],
                        r["bottomup_solutions"])
