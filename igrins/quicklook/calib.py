from igrins.igrins_libs.resource_manager import get_igrins_resource_manager
from igrins.igrins_libs.igrins_config import IGRINSConfig

def get_resource_manager(config_name, obsdate, band):

    config = IGRINSConfig(config_name)

    resource_spec = (obsdate, band)

    rm = get_igrins_resource_manager(config, resource_spec)

    return rm


def load_resource_for(resource_manager, resource_type):

    rm = resource_manager
    master_obsid = "0"
    resource_basename, item_desc = rm.query_resource_for(master_obsid,
                                                         resource_type)
    r = rm.load(resource_basename, item_desc)

    return r

def load_simple_aperture(resource_manager):

    centroid = load_resource_for(resource_manager, "aperture_definition")

    bottomup_solutions = centroid["bottom_up_solutions"]

    # orders = load_resource_for(resource_manager, "orders")["orders"]
    orders = list(range(len(bottomup_solutions)))

    from igrins.procedures.apertures import Apertures
    ap = Apertures(orders, bottomup_solutions)

    return ap

def get_calibs(band):
    import os
    config_name = os.path.join("./master_calib",
                               "calib_recipe.config")
    obsdate = "20180404"
    rm = get_resource_manager(config_name, obsdate, band)
    # r = load_resource_for(rm, "aperture_definition")

    ap = load_simple_aperture(rm)
    ordermap = load_resource_for(rm, "ordermap")[0].data
    # slitposmap = load_resource_for(rm, "slitposmap")[0].data

    return ap, ordermap  # , slitposmap


if __name__ == "__main__":
    ap, ordermap, slitposmap = get_calibs("K")
