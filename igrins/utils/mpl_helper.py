from matplotlib import rcParams


def add_inner_title(ax, title, loc, style="pe", **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    kkwargs = dict(pad=0., borderpad=0.5)
    if style == "pe":
        prop = dict(path_effects=[withStroke(foreground='w', linewidth=2)],
                    size=rcParams['legend.fontsize'])
        kkwargs["frameon"] = False
    elif style == "frame":
        prop = dict(bbox=dict(facecolor="white", edgecolor="none"))
        kkwargs = dict(pad=0.1, borderpad=1.)
    else:
        prop = {}

    at = AnchoredText(title, loc=loc, prop=prop,
                      # bbox=dictpad=0, borderpad=0.5,
                      **kkwargs, **kwargs)
    # if style == "box":
    #     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at
