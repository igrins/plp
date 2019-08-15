import numpy as np


def get_per_amp_stat(cube, namp=32):
    r = {}

    ds = cube.reshape((namp, -1))

    msk_100 = np.abs(ds) > 100

    r["count_gt_100"] = np.sum(msk_100, axis=1)

    r["stddev_lt_100"] = [np.std(ds1[~msk1])
                          for ds1, msk1 in zip(ds, msk_100)]

    return r


def _get_amp_wise_rfft(d):
    dr = d.reshape((-1, 64, 2048))
    m = np.median(dr, axis=1)

    return np.fft.rfft(m, axis=1)


def get_amp_wise_noise_spectrum(cube):
    return _get_amp_wise_rfft(cube)


def _get_c64_wise_rfft(d):
    dr = d.reshape((-1, 2048, 32, 64))
    m = np.median(dr, axis=3)

    return np.fft.rfft(m, axis=1)


def get_c64_wise_noise_spectrum(cube):
    return _get_c64_wise_rfft(cube)


if False:
    from igrins.procedures.readout_pattern import pipes, apply as apply_pipes

    def remove_kk(d):
        kk = ["p64_per_column", "row_wise_bias"]
        d1 = apply_pipes(d, [pipes[k] for k in kk])

        return d1

    def get_amp_wise_real(d):
        dr = d.reshape((-1, 64, 2048))
        m = np.median(dr, axis=1)

        return m

    def get_amp_wise_stacked(d):
        dr = d.reshape((-1, 64, 2048))
        return dr

    def get_amp_wise_rfft(d):
        dr = d.reshape((-1, 64, 2048))
        m = np.median(dr, axis=1)

        return np.fft.rfft(m, axis=1)

    def make_model_from_rfft(q, kslice):
        orig_shape = q.shape
        qr = q.reshape((-1,) + orig_shape[-1:])
        q0 = np.zeros_like(qr)
        q0[:, kslice] = qr[:, kslice]

        return np.fft.irfft(q0, axis=-1).reshape(orig_shape[:-1] + (-1,))

    import astropy.io.fits as pyfits
    d1 = pyfits.open("indata/20190116/SDCH_20190116_0001.fits")[0].data
    f = pyfits.open("outdata/20190116/SDCH_20190116_0001.pair_subtracted.fits")

    cube0 = f["DIRTY"].data
    cube1 = f["GUARD-REMOVED"].data
    cube2 = f["LEVEL2-REMOVED"].data
    cube3 = f["LEVEL3-REMOVED"].data

    qq0 = [get_amp_wise_rfft(c) for c in cube0]
    qq1 = [get_amp_wise_rfft(c) for c in cube1]
    qq2 = [get_amp_wise_rfft(c) for c in cube2]
    qq3 = [get_amp_wise_rfft(c) for c in cube3]

    qq = qq3
    xbins = np.arange(-0.5, 1025.5, 1.)
    ybins = np.arange(-0.5, 256, 2.)
    hh = np.zeros((len(ybins) - 1, len(xbins) - 1), dtype="d")

    xx = np.arange(1025)
    for i, ql in enumerate(qq):
        for ii, q in enumerate(ql):
            h_ = np.histogram2d(np.abs(q), xx, bins=(ybins, xbins))
            hh += h_[0]

    clf()
    im = imshow(hh, aspect='auto', origin='lower', cmap='gist_gray_r',
                extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    cmax = im.get_clim()[-1]
    im.set_clim(0, cmax*0.5)
    g0 = np.median(np.abs(qq), axis=(0, 1))
    plot(g0, color="r")
    ylim(0, 256)

    # clf()
    # for i, ql in enumerate(qq0):
    #     for ii, q in enumerate(ql):
    #         # g = np.sum(np.abs(q[1:-1]), axis=0)
    #         #plot(g + i * 2000)
    #         plot(np.abs(q), color="0.8")

    # g0 = np.median(np.abs(qq0), axis=(0, 1))
    # plot(g0)

    ss3 = np.vstack([get_amp_wise_stacked(c) for c in cube3])
    mm3 = np.vstack([get_amp_wise_real(c) for c in cube3])

    m0 = make_model_from_rfft(qq3, slice(None, 32))
    m66 = make_model_from_rfft(qq3, slice(66, 67))


    s0 = np.median(cube2, axis=1)  # column-wise variation
    s1 = np.median(cube2, axis=2)  # row-wise variation

    n = len(cube2)
    v = s1.reshape((n, -1, 2, 64))
    v1 = np.hstack([v[:, :, 0, :], v[:, :, 1, ::-1]]).reshape((n, -1, 64))

    e = np.median(v1, axis=1)

    u = np.hstack([np.hstack([e, e[::-1]])]*16)


    sa0 = np.median(cube3[9], axis=0)  # column-wise variation
    sa1 = np.median(cube3[9], axis=1)  # row-wise variation

    k0 = np.fft.rfft(cube[0], axis=0)
    k1 = np.fft.rfft(cube[0], axis=1)


    v = s1.reshape((-1, 2, 64))
    v1 = np.hstack([v[:, 0, :], v[:, 1, ::-1]]).reshape((-1, 64))

    e = np.median(v1, axis=0)

    u = np.hstack([np.hstack([e, e[::-1]])]*16)

if False:
    # y-direction power spectrum, along x-direction
    import scipy.ndimage as ni

    qq0 = get_c64_wise_noise_spectrum(np.array(cube0))
    qq1 = get_c64_wise_noise_spectrum(np.array(cube1))
    qq2 = get_c64_wise_noise_spectrum(np.array(cube2))
    qq3 = get_c64_wise_noise_spectrum(np.array(cube3))

    n = 10
    fig, axlist = plt.subplots(n, 1, num=2,
                               sharey=True, sharex=True, clear=True)

    #for axl, ql in zip(axlist, [qq1, qq2, qq3]):

    axl, ql = axlist, qq0
    for ax, q in zip(axl, ql):
        # ax.imshow(ni.gaussian_filter1d(np.abs(q), 1.5, axis=0),
        ax.imshow(np.abs(q).T,
                  origin="lower", aspect="auto",
                  vmin=10, vmax=50)

    fig, axlist = plt.subplots(1, 1, num=4,
                               sharex=True, sharey=True, clear=True)
    ax = axlist
    for i, qq in enumerate([qq0, qq1, qq2, qq3]):
        kk = np.median(np.abs(qq), axis=[0, 2])
        # ax = axlist[i]
        ax.plot(kk + 200 * (3-i))
