import numpy as np
import scipy.ndimage as ni
import astropy.io.fits as pyfits
from igrins.procedures import destripe_helper as dh


def _get_slices(ny, dy):
    n_dy = ny//dy
    dy_slices = [slice(iy*dy, (iy+1)*dy) for iy in range(n_dy)]
    return dy_slices

def get_ref(d):
    """
    Ext
    """
    dm = ni.median_filter(d, [1, 64])
    dd = d - dm

    ddm = ni.median_filter(dd, [64, 1])
    ddd = dd - ddm[:]

    return ddd


def get_individual_bg64(d, median_col=None):
    ny_slices = dh._get_ny_slice(2048, 64)
    rr = []
    for sl in ny_slices[:]:
        # sl = ny_slices[-3]
        r = np.median(d[sl][2:-2,:], axis=0)
        if median_col is not None:
            r = ni.median_filter(r, median_col)
        # plot(r)
        rr.append(r)

    return np.vstack(rr)

def sub_individual_bg64(d, median_col=None):
    g = get_individual_bg64(d, median_col=median_col)
    vv = np.vstack(np.swapaxes(np.broadcast_to(g,
                                               (64,) + g.shape), 0, 1))

    return d - vv

# if False:
#     q = np.concatenate(qq)
#     return q
def get_guard_column(d):
    k = dh.get_median_guard_column(d)
    k0 = dh.subtract_row64_median(k)
    ds, p = dh.get_pattern64_from_each_column(k0)

    return p

def sub_guard_column(d, mask=None):
    p = get_guard_column(d)
    vv = d - p[:, np.newaxis]
    # xx = np.median(vv[-512:], axis=0)
    return vv # - xx

def get_p64_slope(d0, mask=None):
    p64 = dh.stack64(d0, mask=mask)
    # p64 = p64_ - np.median(p64_, axis=1)[:, np.newaxis]

    # extract the slope component
    a0 = np.median(p64[:, :1024], axis=1)
    a1 = np.median(p64[:, 1024:], axis=1)
    v = (a1 + a0)/2.
    x = np.arange(2048)
    u = (a1 - a0)[:, np.newaxis]/1024.*(x-1024.)

    return v, u

def sub_p64_slope(d0, mask=None):
    v, u = get_p64_slope(d0, mask=mask)
    p = dh.concat(v[:, np.newaxis] + u, [1, -1], 16)

    return d0 - p

def get_p64_pattern(d1, mask=None):
    p64a = dh.stack64(d1, mask)
    p64a0 = np.median(p64a, axis=1)

    return p64a0 - np.median(p64a0)

def sub_p64_pattern(d1, mask=None):
    p64a0 = get_p64_pattern(d1, mask)
    k = dh.concat(p64a0, [1, -1], 16)

    return d1 - k[:, np.newaxis]

def get_p64_pattern_each(d1, mask=None):
    p64a = dh.stack64(d1, mask)

    return p64a

def sub_p64_pattern_each(d1, mask=None):
    p64a0 = get_p64_pattern_each(d1, mask)
    k = dh.concat(p64a0, [1, -1], 16)

    return d1 - k

# x = np.arange(2048)
# yy = []
# for r in p64:
#     p = np.polyfit(x[4:-4], r[4:-4], 1)
#     y0 = np.polyval([p[0], 0], x)
#     yy.append(y0)

def get_row64_median(d, mask=None, q=None):
    if q is None:
        f = np.nanmedian
    else:
        def f(d2):
            return np.nanpercentile(d2, q)

    d3 = np.ma.array(d, mask=mask).filled(np.nan)

    ny_slices = _get_slices(2048, 64)
    return np.concatenate([np.broadcast_to(f(d3[sl]), (len(d3[sl]), ))
                           for sl in ny_slices])


def factory_get_amp_bias(q=None):
    if q is None:
        f = np.nanmedian
    else:
        def f(d2):
            return np.nanpercentile(d2, q)

    slice_size = 64
    ny_slices = _get_slices(2048, slice_size)

    def _broadcast(s):
        return np.broadcast_to(s, (slice_size, ))

    def _get_amp_bias(d2, mask=None):
        d3 = np.ma.array(d2, mask=mask).filled(np.nan)

        s = np.array([f(d3[sl]) for sl in ny_slices])

        return s - np.median(s)

    return _get_amp_bias, _broadcast

    # return np.concatenate([np.broadcast_to(f(d3[sl]), (len(d3[sl]), ))
    #                        for sl in ny_slices])

    # def _get_amp_bias(d2, mask=None):
    #     if q is None:
    #         k = np.median(d2, axis=1)
    #     else:
    #         k = np.percentile(d2, q, axis=1)
    #     k0 = dh.get_row64_median(k)

    #     return k0

    # return _get_amp_bias

def factory_sub_amp_bias(q=None):
    _get_amp_bias, _broadcast = factory_get_amp_bias(q=None)

    def _sub_amp_bias(d2, mask=None):
        s = _get_amp_bias(d2, mask)
        k0 = np.concatenate([_broadcast(s1) for s1 in s])
        return d2 - k0[:, np.newaxis]

    return _sub_amp_bias


def factory_get_amp_bias_old(q=None):
    def _get_amp_bias(d2, mask=None):
        if q is None:
            k = np.median(d2, axis=1)
        else:
            k = np.percentile(d2, q, axis=1)
        k0 = dh.get_row64_median(k)

        return k0

    return _get_amp_bias

def factory_sub_amp_bias_old(q=None):
    _get_amp_bias = factory_get_amp_bias(q)

    def _sub_amp_bias(d2, mask=None):
        k0 = _get_amp_bias(d2, mask)
        return d2 - k0[:, np.newaxis]

    return _sub_amp_bias

def get_col_median_slow(d5, mask=None):
    k = get_col_median(d5, mask=mask)
    # k = np.ma.median(np.ma.array(d5, mask=mask), axis=0)
    k = ni.median_filter(k, 64)
    return k - np.median(k)

def sub_col_median_slow(d5, mask=None):
    k = get_col_median_slow(d5, mask=mask)

    return d5 - k

def get_col_median(d5, mask=None):
    if mask is not None:
        d6 = np.ma.array(d5, mask=mask).filled(np.nan)
    else:
        d6 = d5

    k = np.nanmedian(d6, axis=0)
    # k = ni.median_filter(k, 64)
    return k - np.nanmedian(k)

def sub_col_median(d5, mask=None):
    k = get_col_median(d5, mask=mask)

    return d5 - k

def get_row_median(d6, mask=None):
    if mask is not None:
        d6 = np.ma.array(d6, mask=mask).filled(np.nan)

    c = np.nanmedian(d6, axis=1)
    return c

def sub_row_median(d6, mask=None):
    c = get_row_median(d6, mask=mask)
    return d6 - c[:, np.newaxis]

def get_amp_p2(d, mask=None):
    d = np.ma.array(d, mask=mask).filled(np.nan)
    do = d.reshape(32, 32, 2, -1)
    av = np.nanmedian(do, axis=[1,2,3])
    dd = do - av.reshape(32, 1, 1, 1)
    p = np.nanmedian(dd * np.array([1, -1])[np.newaxis, np.newaxis, :, np.newaxis],
                     axis=[1, 2, 3])
    return av, p

def broadcast_amp_p2(av, p):
    k = p[:, np.newaxis] * np.array([1, -1])
    v = np.zeros((32, 32, 2, 1)) + k[:, np.newaxis, :, np.newaxis]
    avv = av.reshape(32, 1, 1, 1) + v
    return avv.reshape(2048, 1)

def sub_amp_p2(d, mask=None):
    av, p = get_amp_p2(d, mask)
    return d - broadcast_amp_p2(av, p)

def get_amp_bias_variation(d7, mask=None):
    g = get_individual_bg64(d7, median_col=64)

    return g

def sub_amp_bias_variation(d7, mask=None):
    return sub_individual_bg64(d7, 64)

def _apply(d, flist, mask=None, draw_hist_ax=None):
    for f in flist:
        d = f(d, mask=mask)
    return d

