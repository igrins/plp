import numpt as np
import scipy.ndimage as ni

def get_order_boundary_indices(s, s0=None):
    #x = np.arange(len(s))

    s = np.array(s)
    mm = s > max(s) * 0.05
    dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

    if s0 is None:
        s0 = get_smoothed_order_spec(s)

    # mask out absorption feature
    smooth_size=20
    #s_s0 = s-s0
    #s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

    #mmm = s_s0 > -3.*s_s0_std


    s1 = ni.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
    #x1 = x[dd1:dd2]

    s1r = s1

    s1_std = s1r.std()
    s1_std = s1r[np.abs(s1r)<2.*s1_std].std()

    s1r[np.abs(s1r) < 2.*s1_std] = np.nan

    if np.any(np.isfinite(s1r[:1024])):
        i1 = np.nanargmax(s1r[:1024])
        i1r = np.where(~np.isfinite(s1r[:1024][i1:]))[0][0]
        i1 = dd1+i1+i1r #+smooth_size
    else:
        i1 = dd1
    if np.any(np.isfinite(s1r[1024:])):
        i2 = np.nanargmin(s1r[1024:])
        i2r = np.where(~np.isfinite(s1r[1024:][:i2]))[0][-1]
        i2 = dd1+1024+i2r
    else:
        i2 = dd2

    return i1, i2


def get_order_flat1d(s, i1=None, i2=None):

    if i1 is None:
        i1 = 0
    if i2 is None:
        i2 = len(s)

    x = np.arange(len(s))

    if 0:

        from astropy.modeling import models, fitting
        p_init = models.Chebyshev1D(degree=6, window=[0, 2047])
        fit_p = fitting.LinearLSQFitter()
        p = fit_p(p_init, x[i1:i2][mmm[i1:i2]], s[i1:i2][mmm[i1:i2]])

    if 1:
        # t= np.linspace(x[i1]+10, x[i2-1]-10, 10)
        # p = LSQUnivariateSpline(x[i1:i2],
        #                         s[i1:i2],
        #                         t, bbox=[0, 2047])

        # t= np.concatenate([[x[1],x[i1-5],x[i1],x[i1+5]],
        #                    np.linspace(x[i1]+10, x[i2-1]-10, 10),
        #                    [x[i2-5], x[i2], x[i2+5],x[-2]]])

        t_list = []
        if i1 > 10:
            t_list.append([x[1],x[i1]])
        else:
            t_list.append([x[1]])
        t_list.append(np.linspace(x[i1]+10, x[i2-1]-10, 10))
        if i2 < len(s) - 10:
            t_list.append([x[i2], x[-2]])
        else:
            t_list.append([x[-2]])

        t= np.concatenate(t_list)

        # s0 = ni.median_filter(s, 40)
        from scipy.interpolate import LSQUnivariateSpline
        p = LSQUnivariateSpline(x,
                                s,
                                t, bbox=[0, len(s)-1])

    return p


def get_smoothed_order_spec(s):
    s0 = ni.median_filter(s, 40)
    return s0

def check_order_trace1(ax, x, s, i1i2):
    x = np.arange(len(s))
    ax.plot(x, s)
    i1, i2 = i1i2
    ax.plot(np.array(x)[[i1, i2]], np.array(s)[[i1,i2]], "o")

def check_order_trace2(ax, x, p):
    ax.plot(x, p(x))

def prepare_order_trace_plot(s_list, row_col=(3, 2)):

    import matplotlib.pyplot as plt

    row, col = row_col

    n_ax = len(s_list)
    n_f, n_remain = divmod(n_ax, row*col)
    if n_remain:
        n_ax_list = [row*col]*n_f + [n_remain]
    else:
        n_ax_list = [row*col]*n_f


    from mpl_toolkits.axes_grid1 import Grid
    i_ax = 0

    fig_list = []
    ax_list = []
    for n_ax in n_ax_list:
        fig = plt.figure()
        fig_list.append(fig)

        grid = Grid(fig, 111, (row, col), ngrids=n_ax,
                    share_x=True)

        sl = slice(i_ax, i_ax+n_ax)
        for s, ax in zip(s_list[sl], grid):
            ax_list.append(ax)

        i_ax += n_ax

    return fig_list, ax_list
