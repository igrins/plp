from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages

from six import BytesIO


outtype_available = ["pdf", "png"]


def figlist_to_pngs(figlist):
    sout_list = []
    for fig in figlist:
        FigureCanvasAgg(fig)
        s = BytesIO()
        fig.savefig(s, format="png")
        sout_list.append(s.getvalue())

    return sout_list


def figlist_to_pdf(figlist):
    s = BytesIO()
    with PdfPages(s) as pdf:
        for fig in figlist:
            pdf.savefig(fig)

    return s.getvalue()


def save_figlist_to_pdf(obsset, figlist, section, outroot):
    s = figlist_to_pdf(figlist)
    fn = "{}.pdf".format(outroot)
    obsset.store_under(section, fn, s)


def save_figlist_to_png(obsset, figlist, section, outroot):
    sl = figlist_to_pngs(figlist)

    for i, s in enumerate(sl):
        fn = "{}_fig{:02d}.png".format(outroot, i)
        obsset.store_under(section, fn, s)


_registered = dict(png=save_figlist_to_png,
                   pdf=save_figlist_to_pdf)


def check_outtype(outtype):
    if outtype not in _registered:
        outtype_available = list(_registered.keys())
        raise ValueError("outtype must be one of %s".format(outtype_available))


def save_figlist(obsset, figlist, section, outroot, outtype):
    check_outtype(outtype)
    _registered[outtype](obsset, figlist, section, outroot)
