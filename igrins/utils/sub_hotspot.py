import numpy as np
from scipy.ndimage import median_filter


def get_spl(d1, s1):
    tx1 = [-0.5, 0.001, 1, 1.4, 2, 2.24, 2.83, 3, 3.6, 4.1, 5.]
    tx2 = [3.2, 3.6, 4, 4.3, 5] + list(range(6, 32)) + list(range(32, 64, 2)) + list(range(64, 128, 4))

    from scipy.interpolate import LSQUnivariateSpline
    spl1 = LSQUnivariateSpline(np.concatenate([[-2, -1], d1]),
                               np.concatenate([[0, 0.], s1]),
                               tx1, k=1)

    ii = np.searchsorted(d1, 3.1)

    spl2 = LSQUnivariateSpline(d1[ii:],
                               median_filter(s1[ii:], 3),
                               tx2, k=2)

    def spl(x):
        zz = np.empty_like(x)
        msk = x > 3.8
        zz[~msk] = spl1(x[~msk])
        zz[msk] = spl2(x[msk])

        return zz

    return spl


def interpolate_hotspot(d, distance_map):

    dd = distance_map

    # dd = ((xx - cx)**2 + (yy - cy)**2)**.5
    msk = dd < 129

    d1_ = dd[msk]
    indx = np.argsort(d1_)
    d1 = d1_[indx]

    s1 = d[msk][indx]
    spl = get_spl(d1, s1)

    return spl


def get_distance_map(d, cx, cy):
    shape = d.shape
    yy, xx = np.indices(shape)
    jj = (xx - cx) + (yy - cy)*1j

    # we need strictly increains numbers. we add small values to order them
    aa = np.angle(jj) / 1.e4
    dd = np.abs(jj) + aa

    return dd


def subtract_hotspot(d, cx, cy, box_size):
    assert box_size < 128

    dd = get_distance_map(d, cx, cy)

    spl = interpolate_hotspot(d, dd)

    zz = np.zeros_like(d)

    msk2 = dd < box_size

    zz[msk2] = spl(dd[msk2])

    return d - zz


def main():
    # def main():
    from igrins import get_obsset
    band = "H"
    # config_file = "../../recipe.config"
    config_file = None

    obsset = get_obsset("20190318", band, "SKY",
                        # obsids=range(10, 11),
                        obsids=range(1011, 1020),
                        frametypes=["-"],
                        config_file=config_file)

    hdul = obsset.load("FLAT_OFF")

    obsset_sky = get_obsset("20190318", band, "SKY",
                            # obsids=range(10, 11),
                            obsids=range(10, 11),
                            frametypes=["-"],
                            config_file=config_file)

    hdul_sky = obsset_sky.load("combined_sky")

    cx, cy = 163, 586
    dx = 128
    sl = (slice(cy - dx, cy + dx), slice(cx - dx, cx + dx))

    k = hdul[0].data
    s = hdul_sky[0].data

    import matplotlib.pyplot as plt
    fig, axlist = plt.subplots(2, 2, num=1, clear=True)

    sz = subtract_hotspot(s, cx, cy, box_size=96)
    kz = subtract_hotspot(k, cx, cy, box_size=96)

    im00 = axlist[0, 0].imshow(k[sl])
    im01 = axlist[0, 1].imshow(kz[sl])
    for im in [im00, im01]:
        im.set_clim(-3, 3)

    im10 = axlist[1, 0].imshow(s[sl])
    im11 = axlist[1, 1].imshow(sz[sl])
    for im in [im10, im11]:
        im.set_clim(-10, 10)

    plt.show()


if __name__ == '__main__':
    main()
