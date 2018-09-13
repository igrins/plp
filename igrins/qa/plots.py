from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from igrins.libs.zscale import zscale as calc_zscale
import scipy.ndimage as ni


def get_zscale(d):
    z1, z2 = calc_zscale(np.nan_to_num(d), bpmask=~np.isfinite(d))
    return z1, z2


def imshow(fig, d, zscale=False, **kwargs):
    d = np.asarray(d)

    ax = fig.add_subplot(111)

    default_kwargs = dict(interpolation="none", origin="lower")

    default_kwargs.update(kwargs)

    im = ax.imshow(d, **default_kwargs)

    fig.tight_layout()

    if zscale:
        z1, z2 = get_zscale(d)
        im.set_clim(z1, z2)

    return im


def imshow2(fig, d1, d2, zscale=False, **kwargs):

    default_kwargs = dict(interpolation="none", origin="lower")

    default_kwargs.update(kwargs)

    d1 = np.asarray(d1)
    d2 = np.asarray(d2)

    from mpl_toolkits.axes_grid1 import ImageGrid

    if default_kwargs.get("aspect", None) == "auto":
        aspect = False
    else:
        aspect = True

    grid = ImageGrid(fig, 111, (1, 2), share_all=True, aspect=False)
    ax1, ax2 = grid[0], grid[1]

    im1 = ax1.imshow(d1, **default_kwargs)
    im2 = ax2.imshow(d2, **default_kwargs)

    fig.tight_layout()

    if zscale:
        z1, z2 = get_zscale(d1)
        im1.set_clim(z1, z2)

        z1, z2 = get_zscale(d2)
        im2.set_clim(z1, z2)

    return im1, im2


def hist_mask(fig, mask):
    labels, nmax = ni.label(mask)
    pix_sum = ni.sum(mask, labels=labels, index=np.arange(1, nmax+1))

    ax = fig.add_subplot(121)
    ax.hist((pix_sum), bins=np.arange(0.5, 10.5))
    ax.set_title("A <= 9")

    ax2 = fig.add_subplot(122)
    ax2.hist((pix_sum), bins=np.linspace(9.5, pix_sum.max()+0.5, 20))
    ax2.set_title("A > 9")


def print_mask_summary(mask):
    labels, nmax = ni.label(mask)
    pix_sum = ni.sum(mask, labels=labels, index=np.arange(1, nmax+1))

    print("number of islands :" , len(pix_sum),
          "%5.2f%%" % (float(len(pix_sum))/len(mask.flat)*100.))
    print("total area :", np.max(pix_sum))
    print("number of large islands (A > 9 pix) :", np.sum(pix_sum > 9))
