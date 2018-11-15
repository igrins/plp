from __future__ import print_function

import numpy as np

# class _ResourceManagerInterface:
#     def __init__(self, rmi):
#         self.rmi = rmi

#     def load_resource_for(self, basename, itemname):
#         return self.rmi.load_resource_for(basename, itemname)


class ResourceHelper(object):

    _RESOURCE_MAP = {}

    def resource(name, resource_map=_RESOURCE_MAP):
        def wrapper(f):
            resource_map[name] = f
            return f

        return wrapper

    def get(self, name):
        return self._RESOURCE_MAP[name](self, self.obsset)

    def __init__(self, obsset):
        # for developing purpose
        self.obsset = obsset

    @resource("orders")
    def get_orders(self, obsset):
        return obsset.load_resource_for("orders", check_self=True)["orders"]

    @resource("badpix_mask")
    def get_badpix_mask(self, obsset):

        d_hotpix = obsset.load_resource_for("hotpix_mask")
        d_deadpix = obsset.load_resource_for("deadpix_mask")
        badpix_mask = d_hotpix | d_deadpix

        return badpix_mask

    @resource("destripe_mask")
    def get_destripe_mask(self, obsset):

        pix_mask = self.get("badpix_mask")

        d_bias = self.obsset.load_resource_for("bias_mask")

        mask = d_bias
        mask[pix_mask] = True
        mask[:4] = True
        mask[-4:] = True

        return mask

    @resource("orderflat")
    def get_orderflat(self, obsset):

        orderflat_ = obsset.load_resource_sci_hdu_for("order_flat")
        bias_mask = obsset.load_resource_for("bias_mask")
        badpix_mask = self.get("badpix_mask")

        orderflat = orderflat_.data.copy()
        orderflat[badpix_mask] = np.nan

        orderflat[~bias_mask] = 1.

        return orderflat

    @resource("ordermap")
    def get_ordermap(self, obsset):

        return obsset.load_resource_sci_hdu_for("ordermap").data

    @resource("ordermap_bpixed")
    def get_ordermap_bpixed(self, obsset):

        badpix_mask = self.get("badpix_mask")
        orderflat = self.get("orderflat")
        ordermap_bpixed = self.get("ordermap").copy()
        ordermap_bpixed[badpix_mask] = 0
        ordermap_bpixed[~np.isfinite(orderflat)] = 0

        return ordermap_bpixed

    @resource("slitposmap")
    def get_slitpos_map(self, obsset):

        return obsset.load_resource_sci_hdu_for("slitposmap").data

    @resource("slitoffsetmap")
    def get_slitoffset_map(self, obsset):

        return obsset.load_resource_sci_hdu_for("slitoffsetmap").data

    @resource("sky_mask")
    def get_sky_mask(self, obsset):

        _destripe_mask = self.get("destripe_mask")
        _ordermap_bpixed = self.get("ordermap_bpixed")
        _badpix_mask = self.get("badpix_mask")

        msk = (_ordermap_bpixed > 0) | _destripe_mask

        msk[_badpix_mask] = True

        msk[:4] = True
        msk[-4:] = True
        msk[:, :4] = True
        msk[:, -4:] = True

        return msk

    # wavelength
    @resource("wvl_solutions")
    def get_wvl_solutions(self, obsset):
        wvlsol_products = obsset.load_resource_for("wvlsol")

        # orders_w_solutions = wvlsol_products["orders"]
        wvl_solutions = [np.array(a) for a in wvlsol_products["wvl_sol"]]

        return wvl_solutions

    @resource("aperture")
    def get_aperture(self, obsset):
        from ..procedures.aperture_helper import get_aperture_from_obsset
        orders = self.get("orders")

        order_start = obsset.get_recipe_parameter("order_start")
        # if order_start < 0:
        #     order_start = orders[0]

        order_end = obsset.get_recipe_parameter("order_end")
        # if order_end < 0:
        #     order_end = orders[-1]

        ap = get_aperture_from_obsset(obsset, orders=orders)
        ap.set_order_minmax_to_extract(order_start, order_end)

        return ap


# if 0:
#     rm = ResourceManager()
#     print(rm._RESOURCE_MAP)
