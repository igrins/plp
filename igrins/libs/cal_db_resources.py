from __future__ import print_function

class _ResourceManagerInterface:
    def __init__(self, rmi):
        self.rmi = rmi

    def load_resource_for(self, basename, itemname):
        return self.rmi.load_resource_for(basename, itemname)



class ResourceManager(object):

    _RESOURCE_MAP = {}

    def resource(name, resource_map = _RESOURCE_MAP):
        def wrapper(f):
            resource_map[name] = f
            return f

        return wrapper

    def get(self, basename, name):
        return self._RESOURCE_MAP[name](self, basename)

    def __init__(self, rmi):
        # for developing purpose
        self.rmi = _ResourceManagerInterface(rmi)
        # self.rmi = rmi

    @resource("orders")
    def get_orders(self, basename):
        return self.rmi.load_resource_for(basename, "orders")["orders"]

    @resource("badpix_mask")
    def get_badpix_mask(self, basename):

        d_hotpix = self.rmi.load_resource_for(basename, "hotpix_mask")
        d_deadpix = self.rmi.load_resource_for(basename, "deadpix_mask")
        badpix_mask = d_hotpix.data | d_deadpix.data

        return badpix_mask


    @resource("destripe_mask")
    def get_destripe_mask(self, basename):

        pix_mask = self.get(basename, "badpix_mask")

        d_bias = self.rmi.load_resource_for(basename, "bias_mask")

        mask = d_bias.data
        mask[pix_mask] = True
        mask[:4] = True
        mask[-4:] = True

        return mask

if 0:
    rm = ResourceManager()
    print(rm._RESOURCE_MAP)
