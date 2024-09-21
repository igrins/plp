import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin

i1i2_map_K = {70: [4, 4],
              71: [4, 1437],
              72: [4, 2043],
              73: [4, 2043],
              74: [4, 2043],
              75: [4, 2043],
              76: [4, 2043],
              77: [4, 1990],
              78: [45, 1975],
              79: [59, 1966],
              80: [63, 1961],
              81: [100, 1955],
              82: [94, 1935],
              83: [102, 1934],
              84: [119, 1920],
              85: [125, 1911],
              86: [133, 1907],
              87: [143, 1895],
              88: [150, 1891],
              89: [158, 1882],
              90: [165, 1875],
              91: [188, 1866],
              92: [215, 1818],
              93: [220, 1761],
              94: [1371, 1780],
              95: [4, 4]}

i1i2_map_H = {98: [42, 1680],
              99: [35, 1979],
              100: [26, 2043],
              101: [27, 2043],
              102: [30, 2043],
              103: [35, 2043],
              104: [38, 2043],
              105: [42, 2043],
              106: [48, 2043],
              107: [57, 2043],
              108: [64, 2043],
              109: [72, 2043],
              110: [85, 2043],
              111: [94, 2043],
              112: [105, 2043],
              113: [116, 2043],
              114: [127, 2043],
              115: [138, 2043],
              116: [149, 2043],
              117: [164, 2043],
              118: [180, 2043],
              119: [197, 2043],
              120: [210, 2043],
              121: [226, 2043],
              122: [244, 2043],
              123: [262, 2043],
              124: [1817, 2043]}

i1i2_map = {**i1i2_map_H, **i1i2_map_K}


def get_diff(cor_param, components, o, wvl, ss, ss_var):
    # FIXME the current implementation is quite inefficient.
    cor = np.empty(2048, dtype=float)
    cor.fill(np.nan)
    # qq = pca.components_
    # q = cor_param @ pca.components_ + 1
    q = cor_param @ components + 1
    cor = q
    # cor[sl] = q

    w1 = wvl[o]
    # s1 = hdu.data[o] / a0v[o] / cor
    s1 = ss[o] / cor
    var = ss_var[o]
    # var = np.ones_like(cor)

    # wbin = np.linspace(w1[0], w1[-1], 16)
    wbin = np.linspace(wvl[o+1][1024], wvl[o-1][1024], 2048//2)

    msk = np.isfinite(s1)
    h0, _ = np.histogram(w1[msk], bins=wbin, weights=s1[msk]/var[msk])
    h0w, _ = np.histogram(w1[msk], bins=wbin, weights=1/var[msk])
    h0v, _ = np.histogram(w1[msk], bins=wbin, weights=var[msk])

    # negative order
    o1 = o-1
    w1 = wvl[o1]
    s1 = ss[o1] / cor
    # s1 = hdu.data[o1] / a0v[o1] / cor
    var = ss_var[o1]

    msk = np.isfinite(s1)
    hn, _ = np.histogram(w1[msk], bins=wbin, weights=s1[msk]/var[msk])
    hnw, _ = np.histogram(w1[msk], bins=wbin, weights=1/var[msk])
    hnv, _ = np.histogram(w1[msk], bins=wbin, weights=var[msk])

    # positive order
    o1 = o+1
    w1 = wvl[o1]
    # s1 = hdu.data[o1] / a0v[o1] / cor
    s1 = ss[o1] / cor
    var = ss_var[o1]

    msk = np.isfinite(s1)
    hp, _ = np.histogram(w1[msk], bins=wbin, weights=s1[msk]/var[msk])
    hpw, _ = np.histogram(w1[msk], bins=wbin, weights=1/var[msk])
    hpv, _ = np.histogram(w1[msk], bins=wbin, weights=var[msk])

    hh = np.nanstd([h0/h0w, hn/hnw, hp/hpw], axis=0)
    # hh = (h0/h0w - hn/hnw)**2 + (h0/h0w - hp/hpw)**2
    # , axis=0)
    ww = np.nansum([h0v, hnv, hpv], axis=0)

    return hh, ww


def get_diff_scalar(cor_param, components, o, wvl, ss, ss_var):
    hh, vv = get_diff(cor_param, components, o, wvl, ss, ss_var)
    k = np.nansum((hh/vv)**2) # .sum() # + (cor_param**2).sum()*1.e-24
    return k


# FIXME the extrapolation part only works with a linear component.

# components_ = [c for c in np.loadtxt("test_pca.npy")]
components = [np.linspace(-0.2, 0.2, 2048)] # + components_
NCOMP = len(components)

def get_order_match_corr(orders, wvl, spec_corrected_wo_vega, spec_divide_a0v_variance):

    ss = spec_corrected_wo_vega
    ss_var = spec_divide_a0v_variance.copy()

    mask = np.zeros(ss.shape, dtype=bool)

    for o1, o in enumerate(orders):
        i1, i2 = i1i2_map.get(o, (None, None))
        if (i1, i2) != (None, None):
            mask[o1, :i1] = True
            mask[o1, i2:] = True

    ss[mask] = np.nan
    ss_var[mask] = np.nan

    cc = []
    # for K, 7 .. 25
    # orange = np.arange(7, 22) # This seems to work for both H and K.
    orange = np.arange(len(orders))[3:-3] # This seems to work for both H and K.
    for o in orange:
        cor_param0 = np.zeros(NCOMP, dtype=float)
        cor_param = fmin(get_diff_scalar, cor_param0, args=(components, o, wvl, ss, ss_var),
                         disp=False)
        cc.append(cor_param)

    cc = np.vstack(cc)

    kk = np.polyfit(orange, cc[:, 0], 1)
    cor = np.polyval(kk, orange)
    cc_extrapolated = np.zeros(len(orders), dtype="float")
    cc_extrapolated[orange] = cor
    cc_extrapolated[:orange[0]] = cor[0]
    cc_extrapolated[orange[-1]:] = cor[-1]

    cor_params = np.array(cc_extrapolated)
    corr = cor_params[:, np.newaxis] @ components + 1

    return mask, corr


def main():
    # def get_diff(cor_param, o, wbin, wvl, ss):
    # o = 14
    from pathlib import Path
    from astropy.io import fits
    import json

    band = "H"

    # root = Path("../wasp33")
    root = Path("../test_demo")

    obsdate = "20240718"
    obsid_sky = 165
    obsid = 268 # 10 Lac
    obsid_a0v = 272
    # obsid_a0v = 43

    fn = root / f"calib/primary/{obsdate}/ORDERFLAT_SDC{band}_{obsdate}_{obsid_sky:04d}.json"
    j = json.load(open(fn))
    r = j["fitted_responses"]
    orders = j["orders"]
    i1i2_list = j["i1i2_list"]

    fn = root / "outdata" / obsdate / f"SDC{band}_{obsdate}_{obsid:04d}.spec_a0v.fits"
    hdul = fits.open(fn)
    # hdu = hdul[0]
    hdu = hdul[1]# .data
    wvl = hdul[3].data

    a0v = hdul["VEGA_SPEC"].data

    # ss = hdu.data / a0v

    spec_corrected = hdu.data
    spec_corrected_wo_a0v = hdu.data / a0v

    spec_divide_a0v_variance = hdul["SPEC_DIVIDE_A0V_VARIANCE"].data.copy()

    corr = get_order_match_corr(orders, wvl, spec_corrected_wo_a0v, spec_divide_a0v_variance)

    fig, axs = plt.subplots(2, 1, num=5, clear=True, sharey=True, sharex=True)

    z0 = np.empty_like(hdu.data)
    z0.fill(np.nan)

    for o, o1, (i1, i2), cor1 in zip(orders, range(len(hdu.data)), i1i2_list, corr):
        # if o1 == 6:
        #     i2 = 2008

        i1, i2 = i1i2_map.get(o, [4, 4])
        sl1 = slice(i1, i2)
        axs[0].plot(wvl[o1], (hdu.data[o1]))
        axs[1].plot(wvl[o1][sl1], (hdu.data[o1])[sl1]/cor1[sl1])

        z0[o1][sl1] = (hdu.data[o1])[sl1]/cor1[sl1]

    # y0 = np.nanmedian(hdu.data[o])
    y0 = np.nanmedian(hdu.data)

    axs[1].set_ylim(0, y0*3)
    # axs[1].set_ylim(y0-0.1, y0+0.1)

if __name__ == '__main__':
    main()

