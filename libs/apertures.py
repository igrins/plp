import numpy as np
import numpy.polynomial as P
from stsci_helper import stsci_median
import scipy.ndimage as ni

class ApCoeff(object):
    """
    Apcoeff from original PLP.
    """
    def __init__(self, bottom_solution, up_solution):
        self.bottom_solution = bottom_solution
        self.up_solution = up_solution

    def __call__(self, pixel, frac=0.5):
        pixel = np.array(pixel)
        pixel_y1 = self.bottom_solution(pixel)
        pixel_y2 = self.up_solution(pixel)
        pixel_y = pixel_y1 + frac*(pixel_y2-pixel_y1)
        return pixel_y


class Apertures(object):
    def __init__(self, orders, bottomup_solutions):
        self.orders = orders

        self.apcoeffs = {}
        for o, (bottom, up) in zip(orders, bottomup_solutions):
            if isinstance(bottom, (list, tuple)) and bottom[0] == "poly":
                bottom = P.Polynomial(bottom[1])
            if isinstance(up, (list, tuple)) and up[0] == "poly":
                up = P.Polynomial(up[1])

            self.apcoeffs[o] = ApCoeff(bottom, up)

        self.yi = np.arange(2048)
        self.xi = np.arange(2048)


    def __call__(self, order, pixels, frac=0.5):
        return self.apcoeffs[order](pixels, frac)

    def get_xy_list(self, pixels_list, nan_filter=None):
        """
        ohlines_list : dict of tuples of (pixel coord list)
        """

        xy2 = []
        for order_i, pixel in pixels_list.items():
            pixel_y = self.apcoeffs[order_i](pixel)
            xy2.extend(zip(pixel, pixel_y))

        if nan_filter is not None:
            xy2 = np.compress(nan_filter, xy2, axis=0)

        return xy2

    def make_order_map(self, frac1=0., frac2=1.):
        from itertools import izip

        xx, yy = self.xi, self.yi

        bottom_list = [self.apcoeffs[o](xx, frac1) for o in self.orders]
        top_list = [self.apcoeffs[o](xx, frac2) for o in self.orders]

        def _g(i1):
            order_map1 = np.zeros(len(xx), dtype="i")
            for order, bottom, top in izip(self.orders,
                                           bottom_list, top_list):
                m_up = yy>bottom[i1]
                m_down = yy<top[i1]
                order_map1[m_up & m_down] = order

            return order_map1

        order_map = np.hstack([_g(i1).reshape((-1,1)) for i1 in xx])

        return order_map


    def make_slitpos_map(self):
        from itertools import izip

        xx, yy = self.xi, self.yi

        bottom_list = [self.apcoeffs[o](xx, 0.) for o in self.orders]
        top_list = [self.apcoeffs[o](xx, 1.) for o in self.orders]

        def _g(i1):
            slitpos_map1 = np.empty(len(xx), dtype="d")
            slitpos_map1.fill(np.nan)
            for order, bottom, top in izip(self.orders,
                                           bottom_list, top_list):
                m_up = yy>bottom[i1]
                m_down = yy<top[i1]
                m_order = m_up & m_down
                slit_pos = (yy[m_order] - bottom[i1])/(top[i1] - bottom[i1])
                slitpos_map1[m_order] = slit_pos

            return slitpos_map1

        order_map = np.hstack([_g(i1).reshape((-1,1)) for i1 in xx])

        return order_map

    def make_order_map_old(self, frac1=0., frac2=1.):
        """
        This one is significantly slower than make_order_map.
        """
        yy, xx = np.indices((2048, 2048))
        #order_maps = []
        order_map = np.zeros_like(yy)
        for o in self.orders:
            #izip(count(1), self.apcoeffs):
            ap = self.apcoeffs[o]
            m_up = yy>ap(xx, frac1)
            m_down = yy<ap(xx, frac2)
            order_map[m_up & m_down] = o

        return order_map

    def extract_spectra_v2(self, data, f1=0., f2=1.):

        xx = np.arange(2048)

        s_list = []
        for o in self.orders:
            yy1 = self.apcoeffs[o](xx, frac=f1)
            yy2 = self.apcoeffs[o](xx, frac=f2)

            down = np.clip((yy1+0.5).astype("i"), 0, 2048)
            up = np.clip((yy2++0.5).astype("i"), 0, 2048)

            s = [np.median(data[down[i]:up[i],i]) for i in range(2048)]
            s_list.append(s)

        return s_list

    def extract_spectra_from_ordermap(self, data, order_map):
        slices = ni.find_objects(order_map)
        s_list = []
        for o in self.orders:
            sl = slices[o - 1]
            msk = (order_map[sl] != o)
            s = stsci_median(data[sl], badmasks=msk)

            s_list.append(s)

        return s_list


if __name__ == "__main__":
    from pipeline_jjlee import IGRINSLog

    log_20140316 = dict(flat_off=range(2, 4),
                        flat_on=range(4, 7),
                        thar=[1],
                        sky=[25],
                        HD3417=[15, 16],
                        ur=[993]) # 93 U-Ne (no ThAr)


    igrins_log_src = IGRINSLog("20140316", log_20140316)

    band = "H"

    import pickle
    r_src = pickle.load(open("flat_info_%s_%s.pickle" % (igrins_log_src.date, band)))


    #orders, order_map = make_order_map(r_src)
    orders = np.arange(len(r_src["bottomup_solutions"])) + 1

    ap = Apertures(orders, r_src["bottomup_solutions"])
    #orders, order_map = ap.make_order_map()
    slitpos_map = ap.make_slitpos_map()
    #orders, order_map = ap.make_order_map_old()

    if 0:
        data = r_src["flat_normed"]
        ap.extract_spectra_v2(data)

        ap.extract_spectra_from_ordermap(data, order_map)
