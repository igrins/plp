import numpy as np
import pandas as pd

from oh_lines import OHLines
from scipy.interpolate import interp1d
from reidentify import reidentify_lines_all


def get_ref_wavelengths(ohlines, line_indices):

    ref_wvl = [ohlines.um[l] for l in line_indices]

    return ref_wvl


def flatten(l):
    return [r_ for r1 in l for r_ in r1]


def nested(line_list, colname):
    r = [row[colname].values for group_id, row
         in line_list.groupby(["group_id"])]

    return r


def get_group_flag_generator():
    import itertools
    p = itertools.chain([1], itertools.repeat(0))
    return p


def get_group_flags(l):
    return [f for r1 in l for r_, f in zip(r1, get_group_flag_generator())]


def get_ref_pixels(ref_wvl, wvlsol0, x=None):
    """
    Given the list of wavelengths tuples, return expected pixel
    positions from the initial wavelength solution of wvlsol0.

    """

    if x is None:
        x = np.arange(len(wvl))
    um2pixel = interp1d(wvl, x, bounds_error=False)

    ref_pixel = [um2pixel(w) for w in ref_wvl]

    # there could be cases when the ref lines fall out of bounds,
    # resulting nans.
    nan_filter = [np.all(np.isfinite(p)) for p in ref_pixel]
    valid_list = [[np.all(np.isfinite(p))]*len(p) for p in ref_pixel]

    group_flags = get_group_flags(ref_wvl)
    df = pd.DataFrame(dict(wavelength=flatten(ref_wvl),
                           valid=flatten(valid_list),
                           group_flag=group_flags,
                           group_id=np.add.accumulate(group_flags))
                       )

    ref_pixel_filtered = [r for r, m in zip(ref_pixel, nan_filter) if m]
    df2 = df.join(pd.DataFrame(dict(pixel_i=flatten(ref_pixel_filtered)),
                               index=df.index[flatten(valid_list)]))

    return df2


def get_ref_list1(ohlines, line_indices,
                  wvlsol0, x=None):
    ref_wvl = get_ref_wavelengths(ohlines, line_indices)
    line_list = get_ref_pixels(ref_wvl, wvlsol0, x=x)

    return line_list


def get_ref_list(ohlines, line_indices_list,
                 orders_w_solution, wvl_solutions, s_list):
    ref_wvl_list = []
    ref_pixel_list = []
    for o, wvl, s in  zip(orders_w_solution,
                          wvl_solutions, s_list):
        if o not in line_indices_list:
            ref_wvl_list.append([])
            ref_pixel_list.append([])
            continue
        line_indices = line_indices_list[o]
        x = np.arange(len(s))
        um2pixel = interp1d(wvl, x, bounds_error=False)

        ref_wvl = [ohlines.um[l] for l in line_indices]
        ref_pixel = [um2pixel(w) for w in ref_wvl]

        nan_filter = [np.all(np.isfinite(p)) for p in ref_pixel]

        # there could be cases when the ref lines fall out of bounds,
        # resulting nans.
        ref_wvl_list.append([r for r, m in zip(ref_wvl, nan_filter) if m])
        ref_pixel_list.append([r for r, m in zip(ref_pixel, nan_filter) if m])

    return ref_wvl_list, ref_pixel_list


def fit_ohlines_parameters(ohlines, line_indices_list,
                           orders_w_solution, wvl_solutions, s_list,
                           nan_for_bad_fits=True):

    ref_wvl_list, ref_pixel_list = \
            get_ref_list(ohlines, line_indices_list,
                         orders_w_solution, wvl_solutions,
                         s_list)

    fit_results = reidentify_lines_all(s_list, ref_pixel_list,
                                       sol_list_transform=None)


    # check fit status and replace with nan when fit is not successful.
    if nan_for_bad_fits:
        for r in fit_results:
            for r1 in r[0]:
                if r1[2] < 0:
                    r1[0][:] = np.nan

    return ref_wvl_list, ref_pixel_list, fit_results


def fit_ohlines(ohlines, line_indices_list,
                orders_w_solution, wvl_solutions, s_list):

    ref_wvl_list, ref_pixel_list, fit_results = \
                  fit_ohlines_parameters(ohlines, line_indices_list,
                                         orders_w_solution,
                                         wvl_solutions, s_list,
                                         nan_for_bad_fits=False)

    fitted_positions = retrieve_positions_from_fit(fit_results)

    reidentified_lines = []
    for ref_wvl, positions in zip(ref_wvl_list, fitted_positions):
        reidentified_lines.append((positions,
                                   np.array(map(np.mean, ref_wvl))))

    return ref_pixel_list, reidentified_lines


def fit_ohlines_pixel(s_list, ref_pixel_list):

    fit_results = reidentify_lines_all(s_list, ref_pixel_list,
                                       sol_list_transform=None)

    # extract centroids from the fit
    fitted_positions = []
    for results_, dpix_list in fit_results:

        positions = [sol_[0][0] + dpix for sol_, dpix in \
                     zip(results_, dpix_list)]

        # reidentified_lines.append((np.concatenate(fitted_positions),
        #                            np.concatenate(ref_wvl)))

        fitted_positions.append(np.array(map(np.mean, positions)))

    return fitted_positions


def retrieve_positions_from_fit(fit_results):

    # extract centroids from the fit
    fitted_positions = []
    for results_, dpix_list in fit_results:

        positions = [sol_[0][0] + dpix for sol_, dpix in
                     zip(results_, dpix_list)]

        # reidentified_lines.append((np.concatenate(fitted_positions),
        #                            np.concatenate(ref_wvl)))

        fitted_positions.append(np.array(map(np.mean, positions)))

    return fitted_positions



class Test:
    def __init__(self):

        from igrins_config import IGRINSConfig
        config = IGRINSConfig("../recipe.config")

        def get_refdata(band):
            from master_calib import load_sky_ref_data

            sky_refdata = load_sky_ref_data(config, band)

            return sky_refdata

        sky_ref_data = get_refdata("H")

        self.ohline_indices = sky_ref_data["ohline_indices"]
        self.ohlines_db = sky_ref_data["ohlines_db"]

        import json
        fn = "../calib/primary/20140525/SDCH_20140525_0003.wvlsol_v0.json"
        wvl_solution = json.load(open(fn))
        self.wvl_map = dict(zip(wvl_solution["orders"],
                                wvl_solution["wvl_sol"]))

        fn = "../calib/primary/20140525/SDCH_20140525_0029.oned_spec.json"
        s_list = json.load(open(fn))
        self.s_map = dict(zip(s_list["orders"],
                              s_list["specs"]))

    def test(self, o):

        #o = 115
        wvl = self.wvl_map[o]
        s = self.s_map[o]
        line_indices = self.ohline_indices[o]

        x = np.arange(len(wvl))

        ref_lines = get_ref_list1(self.ohlines_db, line_indices,
                                  wvl, x=x)

        ref_lines["order"] = o

        from reidentify import reidentify
        ref_pixel = nested(ref_lines, "pixel_i")

        res = reidentify(s, ref_pixel, x=x, sigma_init=1.5)

        fitted_lines = get_fitted_lines(ref_lines, res)


def get_fitted_lines(ref_lines, res):

    # calculate centroid pixel positions
    _p = ref_lines.groupby(["group_id"])["pixel_i"]
    dx_list = _p.mean() - _p.first()

    x_list0 = [params[0] for params, _, _ in res]
    x_list = x_list0 + dx_list

    # centroid wavelength
    wvl_list = ref_lines.groupby(["group_id"])["wavelength"].mean()

    fitted_lines = pd.DataFrame(dict(params=[p for p, _, _ in res],
                                     wavelength=wvl_list,
                                     pixel=x_list))

    fitted_lines["order"] = o

    return fitted_lines

    #line_list["pixle_i"]
    # wavelengths, pixels, centroid_wavelengths, centoid_pixels


#def test():
#if 1:



if 0:
    ref_wvl = nested(ref_lines, "wavelength")
    for w in ref_wvl:
        plot(w, np.zeros_like(w), "ro-")





if __name__ == "__main__":

    log_20140525 = dict(flat_off=range(64, 74),
                        flat_on=range(74, 84),
                        thar=range(3, 8),
                        sky=[29])

    from igrins_log import IGRINSLog
    igrins_log = IGRINSLog("20140525", log_20140525)

    ohlines = OHLines()

    igrins_orders = {}
    igrins_orders["H"] = range(99, 122)
    igrins_orders["K"] = range(72, 94)

    ref_utdate = "20140316"
    band = "K"

    import json
    json_name = "ref_ohlines_indices_%s.json" % (ref_utdate,)
    ref_ohline_indices_map = json.load(open(json_name))
    ref_ohline_indices = ref_ohline_indices_map[band]

    json_name = "wvl_sol_phase0_%s_%s.json" % (band, igrins_log.date)
    wvl_solutions = json.load(open(json_name))

    # load spec
    object_name = "sky"
    specname = "arc_spec_%s_%s_%s.json" % (object_name,
                                           band,
                                           igrins_log.date)

    import json
    s_list = json.load(open(specname))

    for wvl, s in zip(wvl_solutions, s_list):
        plot(wvl, s)


    from oh_lines import OHLines
    ohlines = OHLines()

    # from fit_gaussian import fit_gaussian_simple

    # Now we fit with gaussian profile for matched positions.

    from scipy.interpolate import interp1d
    from reidentify import reidentify_lines_all

    x = np.arange(2048)


    line_indices_list = [ref_ohline_indices[str(o)] for o in igrins_orders[band]]

    ref_pixel_list, reidentified_lines = \
                    fit_ohlines(ohlines, line_indices_list,
                                wvl_solutions, s_list)



    ######

    from ecfit.ecfit import get_ordered_line_data, fit_2dspec, check_fit

    # d_x_wvl = {}
    # for order, z in echel.zdata.items():
    #     xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
    #     x_T = xy_T[:,0]
    #     d_x_wvl[order]=(x_T, z.wvl)

    reidentified_lines_map = dict(zip(igrins_orders[band], reidentified_lines))

    if band == "K":
        json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
        r = json.load(open(json_name))
        for i, s in r.items():
            ss = reidentified_lines_map[int(i)]
            ss0 = np.concatenate([ss[0], s["pixel"]])
            ss1 = np.concatenate([ss[1], s["wavelength"]])
            reidentified_lines_map[int(i)] = (ss0, ss1)

    xl, yl, zl = get_ordered_line_data(reidentified_lines_map)
    # xl : pixel
    # yl : order
    # zl : wvl * order

    x_domain = [0, 2047]
    orders_band = igrins_orders[band]
    #orders = igrins_orders[band]
    y_domain = [orders_band[0]-2, orders_band[-1]+2]
    x_degree, y_degree = 4, 3
    #x_degree, y_degree = 3, 2
    p, m = fit_2dspec(xl, yl, zl, x_degree=x_degree, y_degree=y_degree,
                      x_domain=x_domain, y_domain=y_domain)

    # filter out the line indices not well fit by the surface

    keys = reidentified_lines_map.keys()
    di_list = [len(reidentified_lines_map[k][0]) for k in keys]

    endi_list = np.add.accumulate(di_list)

    filter_mask = [m[endi-di:endi] for di, endi in zip(di_list, endi_list)]
    #from itertools import compress
    # _ = [list(compress(indices, mm)) for indices, mm \
    #      in zip(line_indices_list, filter_mask)]
    # line_indices_list_filtered = _

    reidentified_lines_ = [reidentified_lines_map[k] for k in keys]
    _ = [(v[0][mm], v[1][mm]) for v, mm \
         in zip(reidentified_lines_, filter_mask)]

    reidentified_lines_map_filtered = dict(zip(igrins_orders[band], _))


    if 1:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 7))
        check_fit(fig, xl, yl, zl, p,
                  orders_band,
                  reidentified_lines_map)
        fig.tight_layout()

        fig = plt.figure(figsize=(12, 7))
        check_fit(fig, xl[m], yl[m], zl[m], p,
                  orders_band,
                  reidentified_lines_map_filtered)
        fig.tight_layout()
