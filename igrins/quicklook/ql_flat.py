from __future__ import division

import numpy as np

from igrins.procedures import destripe_helper as dh
from matplotlib.figure import Figure

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


def do_ql_flat(hdu, frametype):
    # This is a second one that does not require an ordermap

    if frametype == "OFF":
        d1 = dh.sub_p64_upper_quater(hdu.data)
    else:
        d1 = dh.sub_p64_guard(hdu.data)

    p64 = dh.get_pattern64_from_guard_column(hdu.data)

    ny, nx = d1.shape

    n_segment = 4
    ns = int(ny / n_segment)
    slices = [slice(i * ns,  (i + 1) * ns) for i in range(n_segment)]

    # yy = [256, 768, 1280, 1792]

    stats = [dict(y=(sl.start + sl.stop) / 2,
                  **get_stat_segment(d1[sl]))
             for sl in slices]

    profile = np.median(d1[:, 1024-128:1024+128], axis=1)
    profile[:4] = np.nan
    profile[-4:] = np.nan

    yy = [(sl.start + sl.stop) / 2 for sl in slices]
    t_up = [np.nanpercentile(profile[sl], 90) for sl in slices]
    t_down = [np.nanpercentile(profile[sl], 10) for sl in slices]

    stat_profile = [dict(y=y, t_up_90=t_up_90, t_down_10=t_down_10)
                    for (y, t_up_90, t_down_10) in zip(yy, t_up, t_down)]

    jo = dict(p64=p64,
              stats=stats,
              mean_profile=profile,
              stat_profile=stat_profile)

    return jo


def plot_flat(jo, fig=None):
    if fig is None:
        fig = Figure(figsize=(4, 4))

    ax = fig.add_subplot(111)
    ax.plot(jo["mean_profile"])
    import pandas as pd
    df = pd.DataFrame(jo["stat_profile"])
    ax.plot(df["y"], df["t_up_90"], "^")
    ax.plot(df["y"], df["t_down_10"], "v")

    return fig

if __name__ == "__main__":

    import astropy.io.fits as pyfits

    band = "K"
    # calib = Calib(band, 10.)

    # fn = "/media/igrins128/jjlee/igrins/20170904/SDCK_20170904_0025.fits"

    frametype = "ON"
    # fn = "/media/igrins128/jjlee/annex/igrins/20170414/SDCK_20170414_0014.fits.gz"
    fn = "/data/IGRINS_OBSDATA/20180406/SDCH_20180406_0042.fits"
    f = pyfits.open(fn)

    jo = do_ql_flat(f[0], frametype)

    fig = figure(1)
    fig.clf()
    plot_flat(jo, fig=fig)
