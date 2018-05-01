from __future__ import division

import numpy as np
import pandas as pd

from igrins.procedures import destripe_helper as dh
from igrins.quicklook.calib import get_calibs as _get_calibs
import scipy.ndimage as ni


def get_stat_segment(dd):

    med = np.median(dd)
    dd = dd - med

    t_up = np.percentile(dd, 99.9)
    t_down = np.percentile(dd, 0.1)

    m = (t_down < dd) & (dd < t_up)
    std = dd[m].std()

    t_up = np.percentile(dd, 90)
    t_down = np.percentile(dd, 10)

    return dict(median=med, t_up_999=t_up, t_down_001=t_down, std=std)


def get_calibs(band, telescope):
    ap, ordermap = _get_calibs(band)

    slit_length_arcsec = {"Gemini South": 5.,
                          "DCT": 10.,
                          "HJST": 15.}.get(telescope, 0.)

    return ap, ordermap, slit_length_arcsec


def _rebin(x, y, bins):
    bx, _ = np.histogram(x, bins=bins, weights=y)
    bn, _ = np.histogram(x, bins=bins)

    return 0.5 * (bins[:-1] + bins[1:]), bx / bn


def _measure_height_width(xxm, yym, bg_percentile=50):

    bg = np.nanpercentile(yym, bg_percentile)

    yym = yym - bg
    yym[yym < 0] = 0.

    max_indx = np.argmax(yym)
    max_x = xxm[max_indx]
    height = yym[max_indx]

    # height_mask = yym > 0.5 * height
    # weighted_x = np.sum(xxm[height_mask] * yym[height_mask]) \
    #              / np.sum(yym[height_mask])

    bin_dx = xxm[1:] - xxm[:-1]
    bin_height = .5 * (yym[1:] + yym[:-1])
    fwhm = np.sum(bin_height * bin_dx) / height

    return bg, max_x, height, fwhm


# def do_ql_slit_profile(hdu, band, frametype):
def _get_smoothed_profiles(hdu, band, frametype, ap):
    # This is a second one that does not require an ordermap

    d1 = dh.sub_p64_guard(hdu.data)

    ny, nx = d1.shape

    order_map = ap.make_order_map()

    yi, xi = np.indices(hdu.data.shape)

    x1, x2 = 1024 - 128, 1024 + 128
    xmask = (x1 < xi) & (xi < x2)
    # lets do some sampling along the x-direction
    xmask[:, ::2] = False

    smoothed_profiles = []

    # for o in range(min_order + 2, max_order - 2):
    for o in ap.orders[5:-5]:
        # o = 112

        omask = order_map == o

        # omask = ap(o, ap.xi, 0.)
        msk = omask & xmask

        yc = ap(o, ap.xi, 0.5)
        yh = np.abs(ap(o, ap.xi, 0.) - ap(o, ap.xi, 1.))
        # yic = np.ma.array((yi - yc), mask=~msk).filled(np.nan)

        xx, yy = ((yi - yc) / yh)[msk], hdu.data[msk]
        indx = np.argsort(xx)

        slit_length_pixel = yh[int(len(yh)*0.5)]

        sl = slice(None, None, 2)
        xxm = xx[indx][sl]
        n = int(len(xxm) / 128.)
        yym = ni.median_filter(yy[indx][sl], n)

        r = dict(x=xxm, y=yym,
                 smoothing_length=n,
                 slit_length_pixel=slit_length_pixel)
                 # slit_length_arcsec=slit_length_arcsec)

        smoothed_profiles.append((o, r))

    return smoothed_profiles


def _get_sampled_per_order(smoothed_profiles):

    l = []

    for o, xy in smoothed_profiles:

        xxm, yym = xy["x"], xy["y"]
        n = xy["smoothing_length"]

        r = dict(order=o,
                 x=xxm[::n],
                 y=yym[::n],
                 slit_length_pixel=xy["slit_length_pixel"])

        l.append(r)

    return l


def _get_eq_width_per_order(sample_profiles, bg_percentile=50):

    l = []

    for xy in sample_profiles:
        o = xy["order"]

        xxm, yym = xy["x"], xy["y"]

        (bg, max_x, height,
         fwhm) = _measure_height_width(xxm, yym, bg_percentile)

        r = dict(order=o,
                 bg=bg,
                 height=height,
                 fwhm=fwhm,
                 slit_length_pixel=xy["slit_length_pixel"])

        l.append(r)

    return l


def _get_stacked(smoothed_profiles):

    bins = np.linspace(-0.5, 0.5, 128)

    ww = []
    for o, xy in smoothed_profiles:

        xxm, yym = xy["x"], xy["y"]
        _, w = _rebin(xxm, yym, bins)
        ww.append(w)

    ww = np.array(ww)
    ww0 = ww.sum(axis=0)

    bins0 = 0.5 * (bins[1:] + bins[:-1])

    r = dict(xx=bins0,  # * slit_length_arcsec,
             yy=ww0)

    return ww, r


def do_ql_slit_profile(hdu, band, frametype):
    ap, order_map, slit_length_arcsec = get_calibs(band, hdu.header["TELESCOP"])

    smoothed_profiles = _get_smoothed_profiles(hdu, band, frametype, ap)

    # stacked = dict(slit_length_arcsec=slit_length_arcsec)
    # per_order = dict(slit_length_arcsec=slit_length_arcsec)

    sampled_per_order = _get_sampled_per_order(smoothed_profiles)
    stacked_image, stacked = _get_stacked(smoothed_profiles)


    stacked_stat = {}
    per_order_stat = {}

    for bg_percentile in [25, 50]:

        percentile_key = "{}%".format(bg_percentile)

        _ = _get_eq_width_per_order(sampled_per_order,
                                    bg_percentile)

        per_order_stat[percentile_key] = _

        # r = _get_stacked(smoothed_profiles, bg_percentile)

        (bg, max_x, height,
         fwhm) = _measure_height_width(stacked["xx"],
                                                   stacked["yy"],
                                                   bg_percentile)
        r = dict(bg=bg,
                 max_x=max_x,  # * slit_length_arcsec,
                 height=height,  # * slit_length_arcsec,
                 fwhm=fwhm)  # * slit_length_arcsec)

        # for _k in ["xx", "max_x", "fwhm"]:
        #     r[_k] = r[_k] * slit_length_arcsec

        stacked_stat[percentile_key] = r

    jo_raw = dict(slit_length_arcsec=slit_length_arcsec,
                  smoothed_profiles=smoothed_profiles,
                  sampled_per_order=sampled_per_order,
                  stacked_image=stacked_image)

    jo = dict(slit_length_arcsec=slit_length_arcsec,
              stacked=stacked,
              per_order_stat=per_order_stat,
              stacked_stat=stacked_stat)

    expected_sn = get_expected_sn(jo)
    jo["expected_sn"] = expected_sn

    if True:  # print out information
        df = pd.DataFrame(jo["per_order_stat"]["25%"])

        counts = df["height"] * df["fwhm"] * df["slit_length_pixel"]
        df["Counts"] = counts
        df["SN_AB"] = 2.**0.5 * 3.5 * counts**.5
        df["SN_ABBA"] = 2. * 3.5 * counts**.5
        df["FWHM_ARCSEC"] = df["fwhm"] * slit_length_arcsec

        print(df.set_index("order")[["FWHM_ARCSEC", "Counts", "SN_AB", "SN_ABBA"]])

    return jo, jo_raw


import matplotlib as mpl
from itertools import cycle

def do_figure_stacked_profile(stacked, stacked_stat,
                              slit_length,
                              expected_sn,
                              fig=None, color_cycle=None):

    if fig is None:
        fig = mpl.figure.Figure(figsize=(4, 4))

    if color_cycle is None:
        color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    cc = cycle(color_cycle)

    ax1 = fig.add_subplot(111)

    ax1.plot(stacked["xx"] * slit_length, stacked["yy"], color="k")  # - stacked["bg"])

    for k, ss in stacked_stat.items():
        c = next(cc)
        ax1.axhline(ss["bg"], color=c, ls="--", alpha=0.5)
        ax1.errorbar([ss["max_x"]*slit_length], [.5*ss["height"] + ss["bg"]],
                     xerr=.5*ss["fwhm"]*slit_length,
                     yerr=.5*ss["height"], color=c)

    ax1.set_xlabel("slit length")
    ax1.set_ylabel("count / pixel")
    ax1.tick_params(labelleft=False)

    titles = []
    for quad, v in expected_sn.items():
        title1 = "{}: max({:d}) median({:d})".format(quad,
                                                     int(v["max"]/10.)*10,
                                                     int(v["median"]/10)*10)
        titles.append(title1)

    ax1.set_title("\n".join(titles))

    return fig


def get_expected_sn(jo):

    df = pd.DataFrame(jo["per_order_stat"]["25%"])
    counts = df["height"] * df["fwhm"] * df["slit_length_pixel"]

    expected_sn = 3.5 * counts**.5

    factor_ab = 2.**0.5
    factor_abba = 2.

    return dict(AB=dict(max=factor_ab * expected_sn.max(),
                        median=factor_ab * expected_sn.median()),
                ABBA=dict(max=factor_abba * expected_sn.max(),
                          median=factor_abba * expected_sn.median()))


def do_plot_per_order_stat(jo_raw, jo, fig=None):

    if fig is None:
        fig = mpl.figure.Figure(figsize=(4, 4))

    ax1 = fig.add_subplot(221)
    for v in jo_raw["sampled_per_order"]:
        ax1.plot(v["x"] * v["slit_length_pixel"], v["y"])

    ax2 = fig.add_subplot(222)
    ax2.imshow(jo_raw["stacked_image"], aspect="auto")

    df = pd.DataFrame(jo["per_order_stat"]["25%"])

    ax3 = fig.add_subplot(223)
    ax3.plot(df["order"], df["fwhm"] * jo["slit_length_arcsec"],
             "o")

    v = jo["stacked_stat"]["25%"]
    l1 = ax3.axhline(v["fwhm"] * jo["slit_length_arcsec"], ls=":")

    ax4 = fig.add_subplot(224)
    counts = df["height"] * df["fwhm"] * df["slit_length_pixel"]
    ax4.plot(df["order"], 2. * 3.5 * counts**.5, "o", label="ABBA")
    ax4.plot(df["order"], 2**0.5 * 3.5 * counts**.5, "o", label="AB")

    ax4.legend()


def do_figure_eq_width(stacked, stacked_stat,
                       slit_length, expected_sn,
                       fig=None, color_cycle=None):

    if fig is None:
        fig = mpl.Figure(figsize=(4, 4))

    if color_cycle is None:
        color_cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']

    cc = cycle(color_cycle)

    ax1 = fig.add_subplot(111)

    ax1.plot(stacked["xx"] * slit_length, stacked["yy"], color="k")  # - stacked["bg"])

    for k, ss in stacked_stat.items():
        c = next(cc)
        ax1.axhline(ss["bg"], color=c, ls="--", alpha=0.5)
        ax1.errorbar([ss["max_x"]*slit_length], [.5*ss["height"] + ss["bg"]],
                     xerr=.5*ss["fwhm"]*slit_length,
                     yerr=.5*ss["height"], color=c)

    ax1.set_xlabel("slit length")
    ax1.set_ylabel("count / pixel")
    ax1.tick_params(labelleft=False)

    return fig


def main():
    import astropy.io.fits as pyfits

    band = "K"
    # calib = Calib(band, 10.)

    # fn = "/media/igrins128/jjlee/igrins/20170904/SDCK_20170904_0025.fits"

    frametype = "ON"
    # fn = "/media/igrins128/jjlee/annex/igrins/20170414/SDCK_20170414_0014.fits.gz"
    fn = "/data/IGRINS_OBSDATA/20180406/SDC{}_20180406_0055.fits".format(band)
    f = pyfits.open(fn)

    jo, jo_raw = do_ql_slit_profile(f[0], band, frametype)

    # stacked = jo["stacked"]

    # for bg_percentile_k in stacked:
    # plot_flat_v2(jo)

    fig = plt.figure(1)
    fig.clf()
    do_figure_stacked_profile(jo["stacked"], jo["stacked_stat"],
                              jo["slit_length_arcsec"],
                              jo["expected_sn"],
                              fig=fig)

    fig = plt.figure(2)
    fig.clf()
    do_plot_per_order_stat(jo_raw, jo, fig=fig)


if __name__ == "__main__":
    main()
