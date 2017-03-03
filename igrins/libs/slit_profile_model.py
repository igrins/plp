import numpy as np

from astropy.modeling import models, fitting

def get_multi_gaussian_model(mode, n=None, stddev_list=None):
    if mode == "plus":
        bounds = dict(amplitude=[0., np.inf],
                      stddev=[0.002, 0.4],
                      mean=[0.1, 0.4])
        amplitude, mean = 0.05, 0.25
    elif mode == "minus":
        bounds = dict(amplitude=[-np.inf, 0.],
                      stddev=[0.002, 0.4],
                      mean=[0.6, 0.9])
        amplitude, mean = -0.05, 0.75
    else:
        raise ValueError("invalid mode %s : should be plus or minus" % 
                         mode)

    fixed = dict()#stddev=True)
    kw = dict(bounds=bounds, fixed=fixed)

    if stddev_list is None:
        stddev_list = [0.02, 0.04, 0.1, 0.2]

    if n is None:
        n = len(stddev_list)
    elif n <= len(stddev_list):
        stddev_list = stddev_list[:n]
    else:
        raise ValueError("n(=%d) is incompatible with stddev_list (%s)" %
                         (n, stddev_list))

    m = []
    for stddev in stddev_list:
        g_ = models.Gaussian1D(amplitude=amplitude, mean=mean, 
                               stddev=stddev,
                               **kw)
        m.append(g_)

    # print m
    return np.sum(m)


def get_debug_func(axes=None):

    import matplotlib.pyplot as plt

    if axes is None:
        def _get_axes():
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            return (ax1, ax2)
    else:
        def _get_axes():
            return axes
        

    def _debug_func(m0, m1, y, s):

        ax1, ax2 = _get_axes()

        ax1.plot(y, s, ".")

        yi = np.linspace(0, 1, 100)

        ax1.plot(yi, m1(yi), lw=2, label="all")

        for n in m1.submodel_names:
            ax1.plot(yi, m1[n](yi), label=n)

        # ax1.legend(loc=4)

        sp2d = np.histogram2d(s - m1(y), y,
                              bins=[np.linspace(-0.04, 0.04, 1024),
                                    np.linspace(0, 1, 1024)])


        # ax2.plot(y, s - m1(y), ".")
        import scipy.ndimage as ni
        _d = ni.gaussian_filter(sp2d[0], sigma=2)
        ax2.imshow(_d, extent=[0, 1, -0.04, 0.04], 
                   aspect="auto", interpolation="none")

    return _debug_func


def multi_gaussian_fit_by_mode(mode, y, s, 
                               n_comp,
                               stddev_list=None,
                               debug_func=None):
    assert mode in ["plus", "minus"]

    fit_g = fitting.LevMarLSQFitter()
    
    if stddev_list is None:
        # initial fit with 1 component
        m = get_multi_gaussian_model(mode=mode, n=1)

        g = fit_g(m, y, s)

        stddev = g.stddev.value
        stddev_list = [stddev/2., stddev, stddev*2]

        fixed_params = [(i, k, getattr(g, k)) 
                        for k in ["amplitude", "stddev", "mean"]
                        for i in [1]]
    else:
        fixed_params = [(i, k, v) 
                        for k in ["stddev"]
                        for (i, v) in enumerate(stddev_list)]

    m = get_multi_gaussian_model(mode=mode, 
                                 n=n_comp, 
                                 stddev_list=stddev_list)

    for i, k, v in fixed_params:
        k1 = k + "_%d" % i

        setattr(m, k1, v)
        m.fixed[k1] = True

    m.amplitude_1.value *= .5

    g = fit_g(m, y, s)

    for i, k, v in fixed_params:
        k1 = k + "_%d" % i

        g.fixed[k1] = False

    g = fit_g(g, y, s)

    if debug_func is not None:
        debug_func(m, g, y, s)

    return g



def derive_multi_gaussian_slit_profile(s, y, n_comp, stddev_list,
                                       ye=None):

    g_plus = multi_gaussian_fit_by_mode("plus", s, y,
                                        n_comp=n_comp,
                                        stddev_list=stddev_list)

    g_minus = multi_gaussian_fit_by_mode("minus", s, y,
                                         n_comp=n_comp,
                                         stddev_list=stddev_list)

    g = g_plus+g_minus
    g.parameters = np.concatenate([g_plus.parameters, g_minus.parameters])

    return g


def test_plot():
    # from igrins_config import IGRINSConfig
    from recipe_helper import RecipeHelper

    config_file = "../recipe.config"
    utdate = 20151124

    # config = IGRINSConfig(config_file)
    helper = RecipeHelper(config_file, utdate)
    caldb = helper.get_caldb()

    basename = ("H", 77)
    hdus_star = caldb.load_item_from(basename, "spec_fits")

    hdus_combined = caldb.load_item_from(basename, "combined_image")

    omap = caldb.load_resource_for(basename, ("thar", "ordermap_masked_fits"))[0].data
    slitpos = caldb.load_resource_for(basename, ("sky", "slitposmap_fits"))[0].data

    from aperture_helper import load_aperture
    ap = load_aperture(caldb, basename)

    def expand_1dspec_to_2dspec(s1d, o2d, min_order=None):
        mmm = (o2d > 0) & (o2d < 999)
        xi = np.indices(mmm.shape)[-1]
        if min_order is None:
            min_order = o2d[mmm].min()
        indx = (o2d[mmm]-min_order)*2048 + xi[mmm]

        s2 = np.empty(mmm.shape, dtype=float)
        s2.fill(np.nan)
        s2[mmm] = np.take(s1d, indx)

        return s2


    ss = hdus_star[0].data

    ds = np.array([ap(o, ap.xi, 1.) - ap(o, ap.xi, 0.) for o in ap.orders])
    ds /= 50. # 50 is just a typical width.

    # omap[0].data

    s2d = expand_1dspec_to_2dspec(ss/ds, omap)

    ods = hdus_combined[0].data/s2d
    ode = hdus_combined["VARIANCE_MAP"].data**.5/s2d
    ode[ode<0] = np.nan

    msk1 = np.isfinite(ods) & np.isfinite(ode)

    # select only the central part

    # o1 = np.percentile(ap.orders, ap.orders[i1])
    # o2 = np.percentile(ap.orders, ap.orders[i2])

    xx = np.array([ap(o, 1024, .5) for o in ap.orders])
    i1, i2 = np.searchsorted(xx, [128, 2048-128])
    o1, o2 = ap.orders[i1], ap.orders[i2]
    #o1, o2 = 0, 999
    #o1, o2 = 79, 81
    #o1, o2 = 111, 113
    msk2 = (o1 < omap) & (omap < o2)

    msk2[:, :800] = False
    msk2[:, -800:] = False

    #msk2[:, :512] = False
    #msk2[:, -512:] = False

    msk = msk1 & msk2 # & (slitpos < 0.5)

    ods_mskd = ods[msk]
    s_mskd = slitpos[msk]

    g = derive_multi_gaussian_slit_profile(s_mskd, ods_mskd)

    debug_func = get_debug_func()
    debug_func(g, g, s_mskd, ods_mskd)


if 0:
    g_plus = multi_gaussian_fit_by_mode("plus", s_mskd, ods_mskd, 
                                        debug_func=debug_func)

    g_minus = multi_gaussian_fit_by_mode("minus", s_mskd, ods_mskd, 
                                         debug_func=debug_func)

    g = g_plus+g_minus
    g.parameters = np.concatenate([g_plus.parameters, g_minus.parameters])

    debug_func(g, g, s_mskd, ods_mskd)
    # g = multi_gaussian_fit
