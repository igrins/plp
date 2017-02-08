import numpy as np
from astropy.modeling import models # , fitting

def get_varying_conv_gaussian_model(g_list):

    # VaryingConvolutionGaussian
    def _f(x, y,
           a_0=1, 
           a_1=0, a_2=0, a_3=0,
           b_0=0., 
           b_1=0, b_2=0, b_3=0, # b_4=0, b_5=0,
           # c_0=0.25, 
           c_1=0, c_2=0, c_3=0):
        """
        b : shift
        c : sigma
        """
        from astropy.modeling.utils import poly_map_domain
        domain = [0, 2047]
        window = [-1, 1]
        #dw = window[1] - window[0]

        x = poly_map_domain(x, domain, window)
        # y = poly_map_domain(y, [0, 1], window)

        # a_0 = a_2 + 1 # so that it always 1 at the center
        a_coeffs = [a_0, a_1, a_2, a_3]
        a = models.Chebyshev1D.clenshaw(x, a_coeffs)

        #b_0 = b_2 # so that the curve passes through 0 at the center
        b_coeffs = [b_0, b_1, b_2, b_3]
        b = models.Chebyshev1D.clenshaw(x, b_coeffs)

        c_0 = c_2
        c_coeffs = [c_0, c_1, c_2, c_3]
        c = models.Chebyshev1D.clenshaw(x, c_coeffs)

        #z1, z2, z3, z4 = [poly_map_domain(z, [0, 1], window) for z in [z1, z2, z3, z4]]
        #w1, w2, w3, w4 = [w*dw for w in [w1, w2, w3, w4]]

        y_b = y - b
        shifted_y = [y_b - g1.mean.value for g1 in g_list]

        c2 = c**2
        broadened_sigma2 = [c2 + g1.stddev**2 for g1 in g_list]

        # argument for the exponetial function
        d_list = [-yy**2/s2/2. for (yy, s2) in zip(shifted_y, broadened_sigma2)]

        _ = np.sum([g1.amplitude*np.exp(d) for (g1, d) in zip(g_list, d_list)],
                   axis=0)

        return a * _

    VaryingConvolutionGaussian = models.custom_model(_f)

    return VaryingConvolutionGaussian

    # aa1 = get_amp(w1**2)
    # aa2 = get_amp(w2**2)
    # aa3 = get_amp(w3**2)
    # aa4 = get_amp(w4**2)

    # return a * (k1*aa1*np.exp(d1) + k2*aa2*np.exp(d2) + k3*aa3*np.exp(d3) + k4*aa4*np.exp(d4))


def test_plot():
    # from libs.igrins_config import IGRINSConfig
    from libs.recipe_helper import RecipeHelper

    config_file = "../recipe.config"
    utdate = 20151124

    # config = IGRINSConfig(config_file)
    helper = RecipeHelper(config_file, utdate)
    caldb = helper.get_caldb()

    basename = ("K", 77)
    hdus_star = caldb.load_item_from(basename, "spec_fits")

    hdus_combined = caldb.load_item_from(basename, "combined_image")

    omap = caldb.load_resource_for(basename, ("thar", "ordermap_masked_fits"))[0].data
    slitpos = caldb.load_resource_for(basename, ("sky", "slitposmap_fits"))[0].data


    bias_mask = caldb.load_resource_for(basename, ("flat_on", "bias_mask")).data > 0


    from libs.aperture_helper import load_aperture
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

    ds0 = np.array([ap(o, ap.xi, 1.) - ap(o, ap.xi, 0.) for o in ap.orders])
    ds = ds0 / 50. # 50 is just a typical width.

    s_max = np.nanpercentile(ss / ds0, 90) # mean counts per pixel
    s_cut = 0.03 * s_max # 3 % of s_max

    ss_cut = s_cut * ds0

    ss= np.ma.array(ss, mask=(ss < ss_cut)).filled(np.nan)

    # omap[0].data

    s2d = expand_1dspec_to_2dspec(ss/ds, omap)

    ods = hdus_combined[0].data/s2d
    ode = hdus_combined["VARIANCE_MAP"].data**.5/s2d
    ode[ode<0] = np.nan

    msk1 = np.isfinite(ods) # & np.isfinite(ode) & bias_mask

    # select only the central part

    # o1 = np.percentile(ap.orders, ap.orders[i1])
    # o2 = np.percentile(ap.orders, ap.orders[i2])

    # select central orders : orders of y between [128, -128]
    #                       : x between [800:-800]
    x_min, x_max = 800, 2048-800
    y_min, y_max = 128, 2048-128

    xx = np.array([ap(o, 1024, .5) for o in ap.orders])
    i1, i2 = np.searchsorted(xx, [y_min, y_max])
    o1, o2 = ap.orders[i1], ap.orders[i2]
    msk2 = (o1 < omap) & (omap < o2)

    msk2[:, :x_min] = False
    msk2[:, x_max:] = False


    msk = msk1 & msk2 # & (slitpos < 0.5)

    ods_mskd = ods[msk]
    s_mskd = slitpos[msk]

    from slit_profile_model import derive_multi_gaussian_slit_profile
    g_list0 = derive_multi_gaussian_slit_profile(s_mskd, ods_mskd)


    import scipy.ndimage as ni
    def extract_slit_profile(ap, order_map, slitpos_map, data,
                             x1, x2, bins=None):

        x1, x2 = int(x1), int(x2)

        slices = ni.find_objects(order_map)
        slit_profile_list = []
        if bins is None:
            bins = np.linspace(0., 1., 40)

        for o in ap.orders:
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


    if 0:
        bins = np.linspace(0, 1, 100)
        r1 = ap.extract_slit_profile(omap, slitpos, hdus_combined[0].data,
                                    x_min, x_max, bins=bins)

        # r = np.array(r1[1])/r0[1]
        s0 = np.array(r1[1])
        ss = np.sum(np.abs(s0), axis=1)

        clf()
        plot(np.nanmedian(r/ss[:, np.newaxis], axis=0), lw=8, color="0.6")
        plot((r/ss[:, np.newaxis])[2:-2].T)

    if 0:
        import slit_profile_model
        debug_func = slit_profile_model.get_debug_func()
        debug_func(g_list, g_list, s_mskd, ods_mskd)

if 0:

    _, xx = np.indices(ods.shape)


    msk1 = np.isfinite(ods) & np.isfinite(ode) & bias_mask

    logger = Logger("test.pdf")

    for o in ap.orders[1:]:
        print o
        msk2 = omap == o

        msk = msk1 & msk2 # & (slitpos < 0.5)

        x = xx[msk]
        y = slitpos[msk]
        s = ods[msk]

        xmsk = (800 < x) & (x < 2048-800)

        # check if there is enough pixels to derive new slit profile
        if len(s[xmsk]) > 8000:
            import slit_profile_model
            from slit_profile_model import derive_multi_gaussian_slit_profile

            g_list = derive_multi_gaussian_slit_profile(y[xmsk], s[xmsk])
        else:
            g_list = g_list0

        if 0:

            debug_func = slit_profile_model.get_debug_func()
            debug_func(g_list, g_list, y, s)
    
        Varying_Conv_Gaussian_Model = get_varying_conv_gaussian_model(g_list)
        vcg = Varying_Conv_Gaussian_Model()

        if 0:
            def _vcg(y):
                centers = np.zeros_like(y) + 1024
                return vcg(centers, y)

            debug_func(_vcg, _vcg, y, s)
    
        from astropy.modeling import fitting

        fitter = fitting.LevMarLSQFitter()
        t = fitter(vcg, x, y, s, maxiter=100000)


        print "saveing figure"
        logi = logger.open("slit_profile_2d_conv_gaussian",
                           (basename, o))

        logi.submit("raw_data_scatter",
                    (x, y, s))

        logi.submit("profile_sub_scatter",
                    (x, y, s-vcg(x, y)),
                    label="const. model")

        logi.submit("profile_sub_scatter",
                    (x, y, s-t(x, y)),
                    label="varying model")

        logi.close()

    logger.pdf.close()

if 0:

        figure()
        clf()

        sc2 = ax2.scatter(x, y, c=s-vcg(x, y))

        ax1 = subplot(311)
        sc1 = ax1.scatter(x, y, c=s)
        sc1.set_clim(-0.1, 0.1)

        # df = pd.read_pickle("test_slit.pickle")
        # x, y, s = df["x"], df["y"], df["s"]
        # mmm = np.isfinite(s) & (y < 0.5)
        # dfm = df[mmm]
        # x, y, s = dfm["x"], dfm["y"], dfm["s"]



        ax2 = subplot(312)
        sc2 = ax2.scatter(x, y, c=s-vcg(x, y))
        sc2.set_clim(-0.01, 0.01)

        ax3 = subplot(313)
        sc3 = ax3.scatter(x, y, c=s-t(x, y))
        sc3.set_clim(-0.01, 0.01)


class Logger():
    def __init__(self, fout):
        self.instances = dict()

        self.queue = dict()

        from matplotlib.backends.backend_pdf import PdfPages

        self.pdf = PdfPages(fout)
        # w = np.array(w)
        # wmin, wmax = w.min(), w.max()
        # ymax = np.nanmax(s_orig)
        # ax1.set_xlim(wmin, wmax)
        # ax1.set_ylim(-0.1*ymax, 1.1*ymax)
        # pdf.savefig(figure=fig)

        # for w1, s1_orig, of, s1_a0v, tt1, msk, cont in _interim_result:
        #     ax1.set_xlim(min(w1), max(w1))
        #     pdf.savefig(figure=fig)


    def open(self, context, key):
        logi = LoggerInstance(self, context, key)
        self.instances[key] = logi
        return logi

    def finalize(self, key):
        logi = self.instances.pop(key)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_pdf import FigureCanvasPdf

        fig = Figure(figsize=(6, 8))
        fig.suptitle("%s - order:%d" % (key))
        logi.render(fig)

        FigureCanvasPdf(fig)

        self.pdf.savefig(figure=fig)

class LoggerInstance():
    def __init__(self, parent, context, key):
        self.parent = parent
        self.context = context
        self.key = key
        self.log_list = []

    def submit(self, log_type, data, label=""):
        self.log_list.append((log_type, data, label))

    def close(self):
        self.parent.finalize(self.key)

    def render(self, fig):
        n = len(self.log_list)

        for i, l in enumerate(self.log_list):
            ax = fig.add_subplot(n, 1, i+1)

            log_type, data, label = l
            if log_type == "raw_data_scatter":
                x, y, s = data
                sc = ax.scatter(x, y, c=s, edgecolors="none")
                sc.set_clim(-0.1, 0.1)
                sc.set_rasterized(True)

            elif log_type == "profile_sub_scatter":
                x, y, s = data
                sc = ax.scatter(x, y, c=s, edgecolors="none")
                sc.set_clim(-0.01, 0.01)
                sc.set_rasterized(True)

            ax.set_xlim(0, 2048)
            ax.set_ylim(0, 1)
            
