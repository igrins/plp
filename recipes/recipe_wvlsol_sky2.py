# This is to use new framework. Let's use this to measure flexure
# between emission spectra (e.g., sky, UNe, etc.)

#import os
import numpy as np

#from libs.process_flat import FlatOff, FlatOn

#from libs.path_info import IGRINSPath
#import astropy.io.fits as pyfits

#from libs.products import PipelineProducts
#from libs.apertures import Apertures

#from libs.products import PipelineProducts

#from libs.recipe_base import RecipeBase


def save_wavelength_sol(helper, band, obsids,
                        orders_w_solutions, wvl_sol, p):

    caldb = helper.get_caldb()

    d = dict(orders=orders_w_solutions,
             wvl_sol=wvl_sol)
    master_obsid = obsids[0]
    caldb.store_dict((band, master_obsid), "SKY_WVLSOL_JSON", d)

    if 1:  # save as WAT fits header
        xx = np.arange(0, 2048)
        xx_plus1 = np.arange(1, 2048+1)

        from astropy.modeling import models, fitting

        # We convert 2d chebyshev solution to a seriese of 1d
        # chebyshev.  For now, use naive (and inefficient)
        # approach of refitting the solution with 1d. Should be
        # reimplemented.

        p1d_list = []
        for o in orders_w_solutions:
            oo = np.empty_like(xx)
            oo.fill(o)
            wvl = p(xx, oo) / o * 1.e4  # um to angstrom

            p_init1d = models.Chebyshev1D(domain=[1, 2048],
                                          degree=p.x_degree)
            fit_p1d = fitting.LinearLSQFitter()
            p1d = fit_p1d(p_init1d, xx_plus1, wvl)
            p1d_list.append(p1d)

    from libs.iraf_helper import get_wat_spec, default_header_str
    wat_list = get_wat_spec(orders_w_solutions, p1d_list)

    # cards = [pyfits.Card.fromstring(l.strip()) \
    #          for l in open("echell_2dspec.header")]
    import astropy.io.fits as pyfits
    cards = [pyfits.Card.fromstring(l.strip())
             for l in default_header_str]

    wat = "wtype=multispec " + " ".join(wat_list)
    char_per_line = 68
    num_line, remainder = divmod(len(wat), char_per_line)
    for i in range(num_line):
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:char_per_line*(i+1)]
        #print k, v
        c = pyfits.Card(k, v)
        cards.append(c)
    if remainder > 0:
        i = num_line
        k = "WAT2_%03d" % (i+1,)
        v = wat[char_per_line*i:]
        #print k, v
        c = pyfits.Card(k, v)
        cards.append(c)

    if 1:
        # save fits with empty header

        header = pyfits.Header(cards)
        hdu = pyfits.PrimaryHDU(header=header,
                                data=np.array([]).reshape((0,0)))

        caldb.store_image((band, master_obsid),
                          "SKY_WVLSOL_FITS", np.array(wvl_sol),
                          hdu=hdu)

    # if 0:
    #     # plot all spectra
    #     for w, s in zip(wvl_sol, s_list):
    #         plot(w, s)


def save_qa(helper, band, obsids,
            orders_w_solutions,
            reidentified_lines_map, p, m):
    # filter out the line indices not well fit by the surface

    keys = reidentified_lines_map.keys()
    di_list = [len(reidentified_lines_map[k_][0]) for k_ in keys]

    endi_list = np.add.accumulate(di_list)

    filter_mask = [m[endi-di:endi] for di, endi in zip(di_list, endi_list)]

    reidentified_lines_ = [reidentified_lines_map[k_] for k_ in keys]

    _ = [(v_[0][mm], v_[1][mm]) for v_, mm
         in zip(reidentified_lines_, filter_mask)]

    reidentified_lines_map_filtered = dict(zip(orders_w_solutions, _))

    if 1:
        from matplotlib.figure import Figure
        from libs.ecfit import get_ordered_line_data, check_fit

        xl, yl, zl = get_ordered_line_data(reidentified_lines_map)

        fig1 = Figure(figsize=(12, 7))
        check_fit(fig1, xl, yl, zl, p,
                  orders_w_solutions,
                  reidentified_lines_map)
        fig1.tight_layout()

        fig2 = Figure(figsize=(12, 7))
        check_fit(fig2, xl[m], yl[m], zl[m], p,
                  orders_w_solutions,
                  reidentified_lines_map_filtered)
        fig2.tight_layout()

    from libs.qa_helper import figlist_to_pngs
    igr_path = helper.igr_path
    sky_basename = helper.get_basename(band, obsids[0])
    sky_figs = igr_path.get_section_filename_base("QA_PATH",
                                                  "oh_fit2d",
                                                  "oh_fit2d_"+sky_basename)
    figlist_to_pngs(sky_figs, [fig1, fig2])


class WavelengthFitter(object):
    pass


class SkyFitter(WavelengthFitter):
    def get_refdata(self, band):
        from libs.master_calib import load_sky_ref_data

        if band not in self._refdata:
            sky_refdata = load_sky_ref_data(self.config, band)
            self._refdata[band] = sky_refdata

        return self._refdata[band]

    def __init__(self, config, refdate):
        self.config = config
        self.refdate = refdate
        self._refdata = {}

    def fit_individual_orders(self, band,
                              orders_w_solutions, wvl_solutions, s_list):

        sky_ref_data = self.get_refdata(band)
        ohline_indices = sky_ref_data["ohline_indices"]
        ohlines_db = sky_ref_data["ohlines_db"]

        from libs.reidentify_ohlines import fit_ohlines

        _ = fit_ohlines(ohlines_db, ohline_indices,
                        orders_w_solutions,
                        wvl_solutions, s_list)
        ref_pixel_list, reidentified_lines = _

        return ref_pixel_list, reidentified_lines

    def update_K(self, reidentified_lines_map,
                 orders_w_solutions,
                 wvl_solutions, s_list):
        # import libs.master_calib as master_calib
        # fn = "hitran_bootstrap_K_%s.json" % self.refdate
        # bootstrap_name = master_calib.get_master_calib_abspath(fn)
        # import json
        # bootstrap = json.load(open(bootstrap_name))

        from libs.master_calib import load_ref_data
        bootstrap = load_ref_data(self.config, band="K",
                                  kind="HITRAN_BOOTSTRAP_K")


        import libs.hitran as hitran
        r, ref_pixel_list = hitran.reidentify(orders_w_solutions,
                                              wvl_solutions, s_list,
                                              bootstrap)
        # json_name = "hitran_reidentified_K_%s.json" % igrins_log.date
        # r = json.load(open(json_name))
        for i, s in r.items():
            ss = reidentified_lines_map[int(i)]
            ss0 = np.concatenate([ss[0], s["pixel"]])
            ss1 = np.concatenate([ss[1], s["wavelength"]])
            reidentified_lines_map[int(i)] = (ss0, ss1)

        return reidentified_lines_map

    def convert2wvlsol(self, p, orders_w_solutions):

        # derive wavelengths.
        xx = np.arange(2048)
        wvl_sol = []
        for o in orders_w_solutions:
            oo = np.empty_like(xx)
            oo.fill(o)
            wvl = p(xx, oo) / o
            wvl_sol.append(list(wvl))

        return wvl_sol

    def fit(self, band, orders_w_solutions, wvl_solutions, s_list):
        ref_pixel_list, reidentified_lines = \
                            self.fit_individual_orders(band,
                                                       orders_w_solutions,
                                                       wvl_solutions, s_list)

        from libs.ecfit import get_ordered_line_data, fit_2dspec

        reidentified_lines_map = dict(zip(orders_w_solutions,
                                          reidentified_lines))

        if band == "K":
            self.update_K(reidentified_lines_map,
                          orders_w_solutions,
                          wvl_solutions, s_list)

        xl, yl, zl = get_ordered_line_data(reidentified_lines_map)
        # xl : pixel
        # yl : order
        # zl : wvl * order

        x_domain = [0, 2047]
        y_domain = [orders_w_solutions[0]-2, orders_w_solutions[-1]+2]
        x_degree, y_degree = 4, 3
        #x_degree, y_degree = 3, 2
        p, m = fit_2dspec(xl, yl, zl, x_degree=x_degree, y_degree=y_degree,
                          x_domain=x_domain, y_domain=y_domain)

        wvl_sol = self.convert2wvlsol(p, orders_w_solutions)

        return orders_w_solutions, wvl_sol, reidentified_lines_map, p, m


def fit_oh_spectra(refdate, band,
                   orders_w_solutions,
                   wvl_solutions, s_list):

    fitter = SkyFitter(refdate)
    _ = fitter.fit(band, orders_w_solutions, wvl_solutions, s_list)

    orders_w_solutions, wvl_sol, reidentified_lines_map, p, m = _

    return orders_w_solutions, wvl_sol, reidentified_lines_map, p, m



# 20151003 : Below is an attemp to modularize the recipes, which has
# not finished. Initial solution part is done, but the distortion part
# is not.

from libs.recipe_helper import RecipeHelper

from process_wvlsol_v0 import extract_spectra, extract_spectra_multi
from process_wvlsol_v0 import make_combined_image


def fit_wvl_sol(helper, band, obsids):

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    spec_json = caldb.load_item_from((band, master_obsid), "ONED_SPEC_JSON")

    wvlsol_json = caldb.load_item_from((band, master_obsid),
                                       "WVLSOL_V0_JSON")

    assert len(wvlsol_json["orders"]) == len(spec_json["orders"])
    orders = wvlsol_json["orders"]
    wvlsol = wvlsol_json["wvl_sol"]
    s = spec_json["specs"]

    fitter = SkyFitter(helper.config, helper.refdate)
    _ = fitter.fit(band, orders, wvlsol, s)
    orders_w_solutions, wvl_sol, reidentified_lines_map, p, m = _
    #_ = fitter.fit(band, orders, wvlsol["wvl_sol"], s["specs"])

    save_wavelength_sol(helper, band, obsids,
                        orders_w_solutions, wvl_sol, p)

    save_qa(helper, band, obsids,
            orders_w_solutions, reidentified_lines_map, p, m)


def fit_wvl_sol2(helper, band, obsids):

    caldb = helper.get_caldb()
    master_obsid = obsids[0]
    spec_json = caldb.load_item_from((band, master_obsid), "ONED_SPEC_JSON")

    wvlsol_json = caldb.load_item_from((band, master_obsid),
                                       "WVLSOL_V0_JSON")

    assert len(wvlsol_json["orders"]) == len(spec_json["orders"])
    orders = wvlsol_json["orders"]
    wvlsol = wvlsol_json["wvl_sol"]
    s = spec_json["specs"]

    # load reference data
    # find pixel positions from the given wavelength solution
    # reidentify lines and find new pixel positions


    fitter = SkyFitter(helper.config, helper.refdate)
    _ = fitter.fit(band, orders, wvlsol, s)
    orders_w_solutions, wvl_sol, reidentified_lines_map, p, m = _
    #_ = fitter.fit(band, orders, wvlsol["wvl_sol"], s["specs"])

    save_wavelength_sol(helper, band, obsids,
                        orders_w_solutions, wvl_sol, p)

    save_qa(helper, band, obsids,
            orders_w_solutions, reidentified_lines_map, p, m)


def process_band(utdate, recipe_name, band, obsids, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    # STEP 1 :
    ## make combined image

    make_combined_image(helper, band, obsids, mode=None)

    # Step 2

    ## load simple-aperture (no order info; depends on

    extract_spectra(helper, band, obsids)

    extract_spectra_multi(helper, band, obsids)

    fit_wvl_sol(helper, band, obsids)


if 0:

    # Step 3:

    identify_lines(helper, band, obsids)

    get_1d_wvlsol(helper, band, obsids)

    save_1d_wvlsol(extractor,
                   orders_w_solutions, wvl_sol, p)

    save_qa(extractor, orders_w_solutions,
            reidentified_lines_map, p, m)


    save_figures(helper, band, obsids)

    save_db(helper, band, obsids)



if __name__ == "__main__":

    utdate = "20140709"
    obsids = [62, 63]

    utdate = "20140525"
    obsids = [29]

    utdate = "20150525"
    obsids = [52]


    recipe_name = "SKY"


    # utdate = "20150525"
    # obsids = [32]

    # recipe_name = "THAR"

    band = "K"

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    process_band(utdate, recipe_name, band, obsids, config_name)
