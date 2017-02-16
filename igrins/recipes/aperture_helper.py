from igrins.libs.apertures import Apertures

def get_bottom_up_solution(helper, band, master_obsid):
    """
    aperture that only requires flat product. No order information.
    """

    caldb = helper.get_caldb()

    basename, item_desc = caldb.query_resource_for((band, master_obsid),
                                                   resource_type="aperture_definition")
    resource = caldb.load_item_from(basename, item_desc)

    bottomup_solutions = resource["bottom_up_solutions"]

    return basename, bottomup_solutions


def get_simple_aperture(helper, band, obsids, orders=None):

    master_obsid = obsids[0]

    basename, bottomup_solutions = get_bottom_up_solution(helper, band,
                                                          master_obsid)

    if orders is None:
        orders = range(len(bottomup_solutions))

    ap =  Apertures(orders, bottomup_solutions, basename=basename)

    return ap
