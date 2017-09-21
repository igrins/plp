def fig_to_png(rootname, fig, postfix=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if postfix is None:
        postfix = ""

    FigureCanvasAgg(fig)
    fig.savefig("%s%s.png" % (rootname, postfix))

def figlist_to_pngs(rootname, figlist, postfixes=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from itertools import count

    if postfixes is None:
        postfixes = ("fig%02d" % i for i in count(1))

    for postfix, fig in zip(postfixes, figlist):
        FigureCanvasAgg(fig)
        fig.savefig("%s_%s.png" % (rootname, postfix))
        #fig2.savefig("align_zemax_%s_fig2_fit.png" % postfix)
        #fig3.savefig("align_zemax_%s_fig3_hist_dlambda.png" % postfix)

def figlist_to_json(rootname, figlist, postfixes=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from itertools import count
    from mpld3 import save_json

    if postfixes is None:
        postfixes = ("fig%02d" % i for i in count(1))

    for postfix, fig in zip(postfixes, figlist):
        FigureCanvasAgg(fig)
        save_json(fig, "%s_%s.json" % (rootname, postfix))
