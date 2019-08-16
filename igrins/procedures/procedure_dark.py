from __future__ import print_function

import os
import numpy as np
import pandas as pd

from ..procedures.readout_pattern import pipes, apply as apply_pipe
from ..procedures.readout_pattern_guard import remove_pattern_from_guard

from .ascii_plot import (asciiplot_per_amp, pad_with_axes,
                         pad_yaxis_label, pad_xaxis_label,
                         pad_title, to_string, markers)

from .. import DESCS


def apply_rp_2nd_phase(d, mask=None):
    if mask is None:
        mask = np.zeros(d.shape, dtype=bool)
    else:
        mask = mask.copy()

    mask[:4] = True
    mask[-4:] = True

    p = [pipes[k] for k in ['p64_1st_order',
                            'col_wise_bias_c64',
                            'amp_wise_bias_r2',
                            'col_wise_bias']]

    return apply_pipe(d, p, mask=mask)


def apply_rp_3rd_phase(d):
    p = [pipes[k] for k in ['p64_per_column',
                            'row_wise_bias',
                            'amp_wise_bias_c64']]

    return apply_pipe(d, p)


def make_guard_n_bg_subtracted_images(obsset):

    hdu_list = obsset.get_hdus()

    # n = len(hdu_list)
    cube0 = np.array([hdu.data
                     for hdu in hdu_list])

    cube = np.array([remove_pattern_from_guard(hdu.data)
                     for hdu in hdu_list])

    bg = np.median(cube, axis=0)

    cube1 = cube - bg

    bias_mask = obsset.load_resource_for("bias_mask")

    # cube20 = np.array([apply_rp_2nd_phase(d1) for d1 in cube1])
    cube2 = np.array([apply_rp_2nd_phase(d1, mask=bias_mask) for d1 in cube1])
    cube3 = np.array([apply_rp_3rd_phase(d1) for d1 in cube2])

    hdu_list = [([("EXTNAME", "DIRTY")], cube0),
                ([("EXTNAME", "GUARD-REMOVED")], cube1),
                ([("EXTNAME", "ESTIMATED-BG")], bg),
                ([("EXTNAME", "LEVEL2-REMOVED")], cube2),
                ([("EXTNAME", "BIAS-MASK")], bias_mask),
                ([("EXTNAME", "LEVEL3-REMOVED")], cube3)]
    hdul = obsset.get_hdul_to_write(*hdu_list)
    obsset.store(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"], hdul, cache_only=True)


def get_per_amp_stat(cube, namp=32, threshold=100):
    r = {}

    ds = cube.reshape((namp, -1))

    msk_100 = np.abs(ds) > threshold

    r["count_gt_threshold"] = np.sum(msk_100, axis=1)

    r["stddev_lt_threshold"] = [np.std(ds1[~msk1])
                                for ds1, msk1 in zip(ds, msk_100)]

    return r


def estimate_amp_wise_noise(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    dl = []

    obsids = obsset.get_obsids()

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data
        for obsid, c in zip(obsids, cube):
            qq = get_per_amp_stat(c)

            ka = dict(obsid=obsid, level=k)

            _ = [dict(amp=i,
                      stddev_lt_threshold=q1,
                      count_gt_threshold=q2, **ka)
                 for i, (q1, q2) in enumerate(zip(qq["stddev_lt_threshold"],
                                                  qq["count_gt_threshold"]))]

            dl.extend(_)

    obsset.store(DESCS["RO_PATTERN_SUB_STAT_JSON"],
                 dict(stat=dl), cache_only=False)


def analyze_amp_wise_fft(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    from .ro_pattern_fft import (get_amp_wise_rfft)

    hdul_amp = []

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data

        qq_amp = [get_amp_wise_rfft(c) for c in cube]
        hdul_amp.append(([("EXTNAME", k)], np.array([np.abs(qq_amp),
                                                     np.angle(qq_amp)])))

        # qq0_c64 = get_c64_wise_noise_spectrum(np.array(cube))

    hdul = obsset.get_hdul_to_write(*hdul_amp)
    obsset.store(DESCS["RO_PATTERN_AMP_WISE_FFT_IMAGES"],
                 hdul, cache_only=False)


def analyze_c64_wise_fft(obsset):

    hdul = obsset.load(DESCS["RO_PATTERN_SUB_CUBE_IMAGES"])

    from .ro_pattern_fft import (get_c64_wise_noise_spectrum)

    hdul_amp = []

    kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    for k in kl:
        cube = hdul[k].data

        # qq_amp = [get_c64_wise_noise_spectrum(c) for c in cube]
        qq_amp = get_c64_wise_noise_spectrum(cube)
        hdul_amp.append(([("EXTNAME", k)], np.array([np.abs(qq_amp),
                                                     np.angle(qq_amp)])))

        # qq0_c64 = get_c64_wise_noise_spectrum(np.array(cube))

    hdul = obsset.get_hdul_to_write(*hdul_amp)
    obsset.store(DESCS["RO_PATTERN_C64_WISE_FFT_IMAGES"],
                 hdul, cache_only=False)


def generate_white_noise_image(obsset):
    # obsset.load(DESCS["PAIR_SUBTRACTED_IMAGES"])
    pass


# def plot_ap_per_amp(v, v2=None, ymargin=0):
#     amp = list(range(1, 33))
#     fig = AFigure(shape=(70, 20), margins=(1, ymargin), mod_mode="value")
#     _ = fig.plot(amp, v, marker="_o")
#     fig.xlim(0, 32)
#     return fig


def make_ap_badpix_count(cnt):

    # m, nn = asciiplot_per_amp(v, height=8, mmin=0, mmax=8)
    m1, nn = asciiplot_per_amp(cnt, height=8, xfactor=1)

    ss10 = np.take([" ", markers["o"], "*"], m1)
    ss11, sl = pad_with_axes(ss10)
    ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    ss14, sl = pad_title(ss13, sl, "Badpixel count per amp")

    S = to_string(ss14)

    return S


def make_ap_v1_v2(v1, v2):

    mmin = min(min(v1), min(v2))
    mmax = max(max(v1), max(v2))
    # m, nn = asciiplot_per_amp(v, height=8, mmin=0, mmax=8)
    m1, nn = asciiplot_per_amp(v1, height=8, xfactor=1,
                               mmin=mmin, mmax=mmax)
    m2, nn = asciiplot_per_amp(v2, height=8, xfactor=1,
                               mmin=mmin, mmax=mmax)

    ss10 = np.take([" ", markers["o"], "*"], m1)
    ss11, sl = pad_with_axes(ss10)
    ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    ss14, sl = pad_title(ss13, sl, "noise per amp: Raw")

    ss20 = np.take([" ", markers["o"], "*"], m2)
    ss21, sl = pad_with_axes(ss20)
    ss23, sl = pad_xaxis_label(ss21, sl, "1", "32")
    ss24, sl = pad_title(ss23, sl, "Reduced: level2")
    # ss2, sl = pad_yaxis_label(ss1, sl, nn[0], nn[-1])

    # S = "\n".join(["".join(sl) for sl in [::-1]])
    S = to_string(np.hstack([ss14, ss24]))

    return S


def print_out_stat_summary(obsset):

    dl = obsset.load(DESCS["RO_PATTERN_SUB_STAT_JSON"])

    df = pd.DataFrame(dl["stat"])
    g = df.groupby(['level', 'amp']).mean()[["count_gt_threshold",
                                             "stddev_lt_threshold"]]

    l1 = g.loc['DIRTY']
    cnt = l1["count_gt_threshold"].values
    std_dirty = l1["stddev_lt_threshold"].values

    l2 = g.loc['LEVEL2-REMOVED']
    std_lvl2 = l2["stddev_lt_threshold"].values

    S = make_ap_badpix_count(cnt)
    print()
    print(S)

    S = make_ap_v1_v2(std_dirty, std_lvl2)
    print()
    print(S)


def test_asciiplot():
    from igrins import get_obsset, DESCS

    # obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))
    obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))

    dl = obsset.load(DESCS["RO_PATTERN_SUB_STAT_JSON"])

    df = pd.DataFrame(dl["stat"])
    ss = df.groupby(['level', 'amp']).mean()
    g = ss[["count_gt_threshold", "stddev_lt_threshold"]]

    l1 = g.loc['DIRTY']
    cnt = l1["count_gt_threshold"].values
    v1 = l1["stddev_lt_threshold"].values

    l2 = g.loc['LEVEL2-REMOVED']
    v2 = l2["stddev_lt_threshold"].values

    S = make_ap_badpix_count(cnt)
    print()
    print(S)

    S = make_ap_v1_v2(v1, v2)
    print()
    print(S)


def _get_hh(qq):

    xbins = np.arange(-0.5, 1025.5, 1.)
    ybins = np.arange(-0.5, 256, 2.)

    hh = np.zeros((len(ybins) - 1, len(xbins) - 1), dtype="d")
    xx = np.arange(1025)

    for i, ql in enumerate(qq):
        for ii, q in enumerate(ql):
            h_ = np.histogram2d(q, xx, bins=(ybins, xbins))
            hh += h_[0]

    return hh, xbins, ybins


def _get_qabs(obsset):
    hdul = obsset.load(DESCS["RO_PATTERN_AMP_WISE_FFT_IMAGES"])

    # qq = hdul["LEVEL3-REMOVED"].data
    qabs = dict((k, hdul[k].data[0]) for k in ["DIRTY", "GUARD-REMOVED",
                                               "LEVEL2-REMOVED",
                                               "LEVEL3-REMOVED"])

    return qabs


def _get_qabs_c64(obsset):
    hdul = obsset.load(DESCS["RO_PATTERN_C64_WISE_FFT_IMAGES"])

    # qq = hdul["LEVEL3-REMOVED"].data
    qabs = dict((k, hdul[k].data[0].swapaxes(1, 2))
                for k in ["DIRTY", "GUARD-REMOVED",
                          "LEVEL2-REMOVED",
                          "LEVEL3-REMOVED"])

    return qabs

def _plot_median_fft(fig, qabs):
    hh, xbins, ybins = _get_hh(qabs["DIRTY"])

    # ncomp = np.
    ax1 = fig.add_subplot(211)

    im = ax1.imshow(hh, aspect='auto', origin='lower', cmap='gist_gray_r',
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    cmax = im.get_clim()[-1]
    im.set_clim(0, cmax*0.5)
    g0 = np.median(qabs["DIRTY"], axis=(0, 1))
    g1 = np.median(qabs["GUARD-REMOVED"], axis=(0, 1))
    ax1.plot(g0, label="raw")
    ax1.plot(g1, label="guard-removed")
    ax1.set_ylim(0, 256)
    ax1.legend(loc=1)

    ax2 = fig.add_subplot(212, sharex=ax1)

    hh, xbins, ybins = _get_hh(qabs["LEVEL2-REMOVED"])
    im = ax2.imshow(hh, aspect='auto', origin='lower', cmap='gist_gray_r',
                    extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    cmax = im.get_clim()[-1]
    im.set_clim(0, cmax*0.5)
    g0 = np.median(qabs["LEVEL2-REMOVED"], axis=(0, 1))
    g1 = np.median(qabs["LEVEL3-REMOVED"], axis=(0, 1))
    ax2.plot(g0, label="level2-removed")
    ax2.plot(g1, label="level3-removed")
    ax2.set_ylim(0, 64)
    ax2.legend(loc=1)



from ..utils.mpl_helper import add_inner_title

def _plot_fft_y(fig, qq, title=None, obsids=None, vmax=256):
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1 import ImageGrid
    from itertools import repeat

    n = qq.shape[0]
    gs = GridSpec(2, 1, height_ratios=[5, 1])

    grid = ImageGrid(fig, gs[0], (n, 1), cbar_mode="single",
                     cbar_location="top",
                     aspect=False,
                     cbar_size="20%", cbar_pad=0.1)

    if obsids is None:
        obsids = repeat(None)

    for ax, q, obsid in zip(grid, qq, obsids):
        # ax.imshow(ni.gaussian_filter1d(np.abs(q), 1.5, axis=0),
        im = ax.imshow(np.abs(q),
                       origin="lower", aspect="auto",
                       interpolation="none",
                       vmin=10, vmax=vmax)
        if obsid is not None:
            add_inner_title(ax, "obsid={}".format(obsid), loc=1,
                            style="pe")

    grid.axes_llc.cax.colorbar(im)
    grid.axes_llc.set_ylabel("Amp num.")

    # line plot
    hh, xbins, ybins = _get_hh(qq)

    ax = fig.add_subplot(gs[1])

    im = ax.imshow(hh, aspect='auto', origin='lower',
                   cmap='gist_gray_r',
                   extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])

    cmax = im.get_clim()[-1]
    im.set_clim(0, cmax*0.5)
    g0 = np.median(qq, axis=(0, 1))
    ax.plot(g0)
    ax.set_ylim(0, vmax)
    ax.set_xlabel("y-wavenumber")

    if title is not None:
        fig.suptitle(title)


def plot_qa_amp_wise_fft(obsset, outtype="pdf"):
    from matplotlib.figure import Figure
    from ..quicklook.qa_helper import (save_figlist, check_outtype)
    from ..igrins_libs.path_info import get_zeropadded_groupname

    check_outtype(outtype)
    qabs = _get_qabs(obsset)
    obsids = obsset.get_obsids()

    figlist = []
    fig = Figure(figsize=(14, 8))
    _plot_median_fft(fig, qabs)
    figlist.append(fig)

    for k, vmax in [("DIRTY", 128), ("GUARD-REMOVED", 128),
                    ("LEVEL2-REMOVED", 64), ("LEVEL3-REMOVED", 64)]:
        qq = qabs[k]

        fig = Figure(figsize=(14, 8))

        title = dict(DIRTY="RAW").get(k, k)
        _plot_fft_y(fig, qq, title=title, obsids=obsids, vmax=vmax)

        figlist.append(fig)

    section, _outroot = DESCS["QA_DARK_DIR"], "qa_amp_wise_fft"
    obsdate, band = obsset.get_resource_spec()
    groupname = get_zeropadded_groupname(obsset.groupname)
    outroot = "SDC{}_{}_{}_{}".format(band, obsdate, groupname, _outroot)
    save_figlist(obsset, figlist, section, outroot, outtype)


def plot_qa_c64_wise_fft(obsset, outtype="pdf"):
    from matplotlib.figure import Figure
    from ..quicklook.qa_helper import (save_figlist, check_outtype)
    from ..igrins_libs.path_info import get_zeropadded_groupname

    check_outtype(outtype)
    qabs = _get_qabs_c64(obsset)
    obsids = obsset.get_obsids()

    figlist = []
    fig = Figure(figsize=(14, 8))
    _plot_median_fft(fig, qabs)
    figlist.append(fig)

    for k, vmax in [("DIRTY", 256), ("GUARD-REMOVED", 256),
                    ("LEVEL2-REMOVED", 64), ("LEVEL3-REMOVED", 64)]:
        qq = qabs[k]

        fig = Figure(figsize=(14, 8))

        title = dict(DIRTY="RAW").get(k, k)
        _plot_fft_y(fig, qq, title=title, obsids=obsids, vmax=vmax)

        figlist.append(fig)

    section, _outroot = DESCS["QA_DARK_DIR"], "qa_c64_wise_fft"
    obsdate, band = obsset.get_resource_spec()
    groupname = get_zeropadded_groupname(obsset.groupname)
    outroot = "SDC{}_{}_{}_{}".format(band, obsdate, groupname, _outroot)
    save_figlist(obsset, figlist, section, outroot, outtype)


def test_qa_amp_fft():
    from igrins import get_obsset, DESCS

    # obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))
    obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))

    plot_qa_amp_wise_fft(obsset, outtype="png")
    # plot_qa_amp_wise_fft(obsset, outname="test.pdf")

    # xx = np.arange(1025)
    # for i, ql in enumerate(qabs["DIRTY"]):
    #     for ii, q in enumerate(ql):
    #         h_ = np.histogram2d(q, xx, bins=(ybins, xbins))
    #         hh += h_[0]


def test_qa_c64_fft():
    from igrins import get_obsset, DESCS

    # obsset = get_obsset("20190116", "H", "DARK", obsids=range(1, 11))
    obsset = get_obsset("20190116", "K", "DARK", obsids=range(1, 11))

    plot_qa_amp_wise_fft(obsset, outtype="pdf")


def store_qa(obsset, qa_outtype):
    plot_qa_amp_wise_fft(obsset, qa_outtype)
    plot_qa_c64_wise_fft(obsset, qa_outtype)


if False:
    from matplotlib.figure import Figure

    from matplotlib.backends.backend_pdf import PdfPages

    obsids = range(10)

    with PdfPages('multipage_pdf.pdf') as pdf:

        for k, vmax in [("DIRTY", 128), ("GUARD-REMOVED", 128),
                        ("LEVEL2-REMOVED", 64), ("LEVEL3-REMOVED", 64)]:
            qq = qabs[k]

            fig = Figure(figsize=(14, 8))
            # fig = plt.figure(1, figsize=(14, 8))
            # fig.clf()

            _plot_fft_y(fig, qq, k, obsids=obsids, vmax=vmax)

            pdf.savefig(fig)


if False:
    amp, ang = hdul["LEVEL3-REMOVED"].data
    r = amp * np.exp(1j*ang)
    kk0 = make_model_from_rfft(r, slice(None, None))
    kk26 = make_model_from_rfft(r, slice(26, 27))



if __name__ == '__main__':
    test_qa_amp_fft()
