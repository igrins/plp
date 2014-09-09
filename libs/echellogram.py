

import numpy as np
from scipy.interpolate import interp1d

class StripBase(object):
    def __init__(self, order, wvl, x, y):
        self.order = order
        self.x = x
        self.y = y
        self.wvl = wvl

        self.interp_x = interp1d(self.wvl, self.x, kind='cubic',
                                 bounds_error=False)
        self.interp_y = interp1d(self.wvl, self.y, kind='cubic',
                                 bounds_error=False)

class Echellogram(object):
    def __init__(self, orders, wvl_x_y_list):

        self.orders = orders
        self.zdata = {}
        for o, (wvl, x, y) in zip(orders, wvl_x_y_list):
            z = StripBase(o, wvl, x, y)
            self.zdata[o] = z

    @classmethod
    def from_json_fitted_echellogram_sky(cls, json_name):
        import json
        echel = json.load(open(json_name))

        wvl_x_y_list = []
        for wvl, y in zip(echel["wvl_sampled_list"],
                          echel["y_sampled_list"]):
            x = echel["x_sample"]
            wvl_x_y_list.append((wvl, x, y))

        obj = cls(echel["orders"], wvl_x_y_list)
        return obj

    def get_xy_list(self, lines_list):
        """
        get (x, y) list for given orders and wavelengths.

        lines_list : dict of wavelength list
        """
        zdata = self.zdata
        xy1 = []
        for order_i, wvl in lines_list.items():
            #print order_i
            if len(wvl) > 0 and order_i in zdata:
                zz = zdata[order_i]
                xy1.extend(zip(zz.interp_x(wvl), zz.interp_y(wvl)))

        return xy1


    def get_xy_list_filtered(self, lines_list):

        xy_list = self.get_xy_list(lines_list)
        nan_filter = [np.all(np.isfinite([x, y])) for x, y in xy_list]
        xy1f = np.compress(nan_filter, xy_list, axis=0)

        return xy1f, nan_filter


if __name__ == "__main__":
    echel_name = "fitted_echellogram_sky_H_20140316.json"
    echel = Echellogram.from_json_fitted_echellogram_sky(echel_name)
