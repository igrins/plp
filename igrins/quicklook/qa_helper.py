from matplotlib.backends.backend_agg import FigureCanvasAgg
from itertools import count

from six import StringIO

def figlist_to_pngs(figlist):
    sout_list = []
    for fig in figlist:
        FigureCanvasAgg(fig)
        s = StringIO()
        fig.savefig(s, format="png")
        sout_list.append(s.getvalue())

    return sout_list
