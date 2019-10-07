from matplotlib import rcParams


def add_inner_title(ax, title, loc, style="pe",
        extra_prop=None, extra_kwargs=None):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    kwargs = dict(pad=0., borderpad=0.5)
    if style == "pe":
        prop = dict(path_effects=[withStroke(foreground='w', linewidth=2)],
                    size=rcParams['legend.fontsize'])
        kwargs["frameon"] = False
    elif style == "frame":
        prop = dict(bbox=dict(facecolor="white", edgecolor="none"))
        kwargs = dict(pad=0.1, borderpad=1.)
    else:
        prop = {}

    if extra_prop is not None:
        prop.update(extra_prop)

    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    at = AnchoredText(title, loc=loc, prop=prop,
                      # bbox=dictpad=0, borderpad=0.5,
                      **kwargs)
    # if style == "box":
    #     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    return at
