import matplotlib
import numpy as np

def transform_wavelength_solutions(helper, band, obsids):

    # load affine transform

    master_obsid = obsids[0]

    caldb = helper.get_caldb()
    orders = caldb.load_resource_for((band, master_obsid), "orders")["orders"]

    d = caldb.load_item_from((band, master_obsid),
                             "ALIGNING_MATRIX_JSON")

    affine_tr_matrix = d["affine_tr_matrix"]

    # load echellogram
    from master_calib import load_ref_data
    echellogram_data = load_ref_data(helper.config, band,
                                     kind="ECHELLOGRAM_JSON")

    from echellogram import Echellogram
    echellogram = Echellogram.from_dict(echellogram_data)


    wvl_sol = get_wavelength_solutions(affine_tr_matrix,
                                       echellogram.zdata,
                                       orders)

    caldb.store_dict((band, master_obsid),
                     item_type="WVLSOL_V0_JSON",
                     data=dict(orders=orders,
                               wvl_sol=wvl_sol))


    return wvl_sol

def get_wavelength_solutions_old(thar_echellogram_products, echelle,
                             new_orders):

    from storage_descriptions import THAR_ALIGNED_JSON_DESC

    affine_tr = thar_echellogram_products[THAR_ALIGNED_JSON_DESC]["affine_tr"]

    wvl_sol = get_wavelength_solutions2(affine_tr,
                                        echelle.zdata,
                                        new_orders)


    from storage_descriptions import THAR_WVLSOL_JSON_DESC

    r = PipelineProducts("wavelength solution from ThAr")
    r.add(THAR_WVLSOL_JSON_DESC,
          PipelineDict(orders=new_orders,
                       wvl_sol=wvl_sol))

    return r


def get_wavelength_solutions(affine_tr_matrix, zdata,
                             new_orders):
    """
    new_orders : output orders
    """
    from ecfit import get_ordered_line_data, fit_2dspec, check_fit

    affine_tr = matplotlib.transforms.Affine2D()
    affine_tr.set_matrix(affine_tr_matrix)

    d_x_wvl = {}
    for order, z in zdata.items():
        xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
        x_T = xy_T[:,0]
        d_x_wvl[order]=(x_T, z.wvl)

    xl, yl, zl = get_ordered_line_data(d_x_wvl)
    # xl : pixel
    # yl : order
    # zl : wvl * order

    x_domain = [0, 2047]
    orders_band = sorted(zdata.keys())
    #orders = igrins_orders[band]
    #y_domain = [orders_band[0]-2, orders_band[-1]+2]
    y_domain = [new_orders[0], new_orders[-1]]
    p, m = fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3,
                      x_domain=x_domain, y_domain=y_domain)

    if 0:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 7))
        check_fit(fig, xl, yl, zl, p, orders_band, d_x_wvl)
        fig.tight_layout()


    xx = np.arange(2048)
    wvl_sol = []
    for o in new_orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    if 0:
        json.dump(wvl_sol,
                  open("wvl_sol_phase0_%s_%s.json" % \
                       (band, igrins_log.date), "w"))

    return wvl_sol


def main(utdate, band, obsids, config_name):
    helper = RecipeHelper(config_name, utdate)
    wvl_sol = transform_wavelength_solutions(helper, band, obsids)

if __name__ == "__main__":
    utdate = "20150525"
    band = "H"
    obsids = [52]
    master_obsid = obsids[0]

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    main(utdate, band, obsids, config_name)
