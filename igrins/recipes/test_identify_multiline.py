import numpy as np
import pandas as pd

from igrins.libs.recipe_helper import RecipeHelper


def process_band(utdate, recipe_name, band, obsids, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    basename = (band, master_obsid)

    multi_spec = caldb.load_item_from((band, master_obsid),
                                      "multi_spec_fits")

    # just to retrieve order information
    wvlsol_v0 = caldb.load_resource_for(basename, "wvlsol_v0")
    orders = wvlsol_v0["orders"]
    wvlsol = wvlsol_v0["wvl_sol"]

    #
    from collections import namedtuple
    Spec = namedtuple("Spec", ["s_map", "wvl_map"])

    keys = []
    fitted_pixels_list = []

    from igrins.libs.ref_lines_db import SkyLinesDB, HitranSkyLinesDB

    ref_lines_db = SkyLinesDB(config=config_name)
    ref_lines_db_hitrans = HitranSkyLinesDB(config=config_name)

    for hdu in multi_spec:

        slit_center = hdu.header["FSLIT_CN"]

        spec = Spec(dict(zip(orders, hdu.data)),
                    dict(zip(orders, wvlsol)))

        fitted_pixels_list.append(ref_lines_db.identify(band, spec))
        keys.append((slit_center, "OH"))

        if band == "K":
            fitted_pixels_hitran = ref_lines_db_hitrans.identify(band, spec)
            fitted_pixels_list.append(fitted_pixels_hitran)
            keys.append((slit_center, "Hitran"))

    # concatenate collected list of fitted pixels.
    fitted_pixels_master = pd.concat(fitted_pixels_list,
                                     keys=keys,
                                     names=["slit_center", "kind"],
                                     axis=0)

    if 0:
        # storing multiindex seems broken. Enforce reindexing.
        fitted_pixels_master.reset_index().to_json("test.json",
                                                   orient="split")
        fitted_pixels_master.reset_index().to_csv("test.csv")

if 0:
    filename = "test.json"
    fitted_pixels_master0 = pd.read_json(filename, orient="split")
    filename = "test.csv"
    fitted_pixels_master0 = pd.read_csv(filename)
    index = ["slit_center", "kind", "order", "wavelength"]
    fitted_pixels_master = fitted_pixels_master0.set_index(index)

    for do, group in fitted_pixels_master.groupby(level=0):
        o = group.index.get_level_values("order")
        #do = group.index.get_level_values("order")
        x = group["pixels"]
        #print len(group)

        plot(x, o+do, ".")


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
