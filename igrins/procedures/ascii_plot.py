import numpy as np


markers = { '-' : u'None' ,     # solid line style
            ',': u'\u2219',     # point marker
            '.': u'\u2218',     # pixel marker
            '.f': u'\u2218',    # pixel marker
            'o': u'\u25CB',     # circle marker
            'of': u'\u25CF',    # circle marker
            'v': u'\u25BD',     # triangle_down marker
            'vf': u'\u25BC',    # filler triangle_down marker
            '^': u'\u25B3',     # triangle_up marker
            '^f': u'\u25B2',    # filled triangle_up marker
            '<':  u'\u25C1',    # triangle_left marker
            '<f': u'\u25C0',    # filled triangle_left marker
            '>': u'\u25B7',     # triangle_right marker
            '>f': u'\u25B6',    # filled triangle_right marker
            's': u'\u25FD',     # square marker
            'sf': u'\u25FC',    # square marker
            '*': u'\u2606',     # star marker
            '*f': u'\u2605',    # star marker
            '+': u'\u271A',     # plus marker
            'x': u'\u274C',     # x marker
            'd':  u'\u25C7',    # diamond marker
            'df':  u'\u25C6'    # filled diamond marker
            }


def select_i(r):
    c = [1, 2, 3, 4, 5, 10]
    ii = np.array(c).searchsorted(10**r)

    return c[ii]


def asciiplot_per_amp(v, height=10, xfactor=2, mmin=None, mmax=None):
    # v = [2530, 200, 100, 300, 200]
    if mmax is None:
        mmax = np.max(v)
    if mmin is None:
        mmin = np.min(v)
    step = (mmax - mmin) / height
    k, r = divmod(np.log10(step), 1.)

    c = select_i(r) * 10**k
    nmax = np.ceil(mmax/c) * c
    nmin = np.floor(mmin/c) * c
    nn = np.arange(nmin, nmax+2*c, c)

    if k >= 0:
        ss = [str(s1) for s1 in nn.astype("i")]
    else:
        fmt = '{{:.{}f}}'.format(-int(k))
        ss = [fmt.format(s1) for s1 in nn]

    ii = (nn - .5*c).searchsorted(v) - 1

    zz = np.zeros((len(nn) - 1, 32*xfactor), dtype="i")
    ix = xfactor * np.arange(len(v))
    zz[np.array(ii), ix] += 1

    return zz, ss[:-1]


def pad_with_axes(ss):
    # tickSymbols = u'\u253C'  # "+"
    tickSymbols = u'\u2514'  # "L"
    x_axis_symbol = u'\u2500'  # u"\u23bc"  # "-"
    y_axis_symbol = u'\u2502'  # "|"

    ny, nx = ss.shape

    _ = np.vstack([[x_axis_symbol] * nx, ss])
    _ = np.hstack([np.array([tickSymbols] + [y_axis_symbol] * ny)[:, np.newaxis],
                   _])

    return _, (slice(1, ny+1), slice(1, nx+1))


def pad_title(ss, sl, title):

    sly, slx = sl

    sz = np.empty((ss.shape[1]), dtype="U1")
    sz.fill(" ")

    si, ei = np.arange(ss.shape[1])[slx][[0, -1]]

    sz[si:si+len(title)] = list(title)

    _ = np.vstack([ss, sz])
    return _, ()


def pad_yaxis_label(ss, sl, ymin, ymax):
    # arrow_at_bottom = u'\u25A5'
    arrow_at_bottom = u'\u21A5'
    arrow_at_top = u'\u21A7'

    sly, slx = sl

    nl = max(len(ymin), len(ymax))
    sz = np.empty((ss.shape[0], nl), dtype="U1")
    sz.fill(" ")

    si, ei = np.arange(ss.shape[0])[sly][[0, -1]]
    sz[si, -1] = arrow_at_bottom
    sz[ei, -1] = arrow_at_top

    sz[si+1, -len(ymin):] = list(ymin)
    sz[ei-1, -len(ymax):] = list(ymax)

    _ = np.hstack([sz, ss])
    return _, (sly, slice(slx.start + nl if slx.start else nl,
                          slx.stop + nl if slx.stop else None))


def pad_xaxis_label(ss, sl, xmin, xmax):
    # arrow_at_bottom = u'\u25A5'
    arrow_at_left = u'\u21A6'
    arrow_at_right = u'\u21A4'

    sly, slx = sl

    sz = np.empty((ss.shape[1]), dtype="U1")
    sz.fill(" ")

    si, ei = np.arange(ss.shape[1])[slx][[0, -1]]
    sz[si] = arrow_at_left
    sz[ei] = arrow_at_right

    sz[si + 1:si + len(xmin) + 1] = list(xmin)
    sz[ei-len(xmax):ei] = list(xmax)

    _ = np.vstack([sz, ss])

    nl = 1
    return _, (slice(sly.start + nl if sly.start else nl,
                     sly.stop + nl if sly.stop else None), slx)


def to_string(ss):
    S = "\n".join(["".join(sl) for sl in ss[::-1]])
    return S


def main():
    v1 = np.array([3.76732042, 3.40458679, 5.35214171, 3.47665145, 3.524175  ,
                   3.32952354, 3.2470633 , 3.35547774, 3.39414377, 4.09123256,
                   3.42220364, 3.63298829, 3.33341334, 3.43555818, 3.43419983,
                   3.69191523, 3.4959254 , 3.45680025, 3.50497289, 3.84336905,
                   3.39535074, 3.85151758, 3.48880289, 3.2847472 , 3.62306197,
                   3.40256624, 3.56090777, 3.39973562, 3.58725591, 3.56186614,
                   3.59632683, 3.52902713])

    m1, nn = asciiplot_per_amp(v1, height=8, xfactor=1,
                               mmin=mmin, mmax=mmax)

    ss10 = np.take([" ", markers["o"], "*"], m1)
    ss11, sl = pad_with_axes(ss10)
    ss12, sl = pad_yaxis_label(ss11, sl, nn[0], nn[-1])
    ss13, sl = pad_xaxis_label(ss12, sl, "1", "32")
    ss14, sl = pad_title(ss13, sl, "noise per amp: Raw")

    # S = "\n".join(["".join(sl) for sl in ss14[::-1]])
    S = to_string(ss14)

    print()
    print(S)

if __name__ == '__main__':
    main()
