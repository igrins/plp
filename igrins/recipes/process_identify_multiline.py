import numpy as np
import pandas as pd

from igrins.libs.recipe_helper import RecipeHelper

def identify_multiline(obsset):

    multi_spec = obsset.load_item("multi_spec_fits")

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]
    wvlsol = wvlsol_v0["wvl_sol"]

    #
    from collections import namedtuple
    Spec = namedtuple("Spec", ["s_map", "wvl_map"])

    keys = []
    fitted_pixels_list = []

    from igrins.libs.ref_lines_db import SkyLinesDB, HitranSkyLinesDB

    ref_lines_db = SkyLinesDB(config=obsset.get_config())
    ref_lines_db_hitrans = HitranSkyLinesDB(config=obsset.get_config())

    for hdu in multi_spec:

        small_list = []
        small_keys = []

        slit_center = hdu.header["FSLIT_CN"]
        keys.append(slit_center)

        spec = Spec(dict(zip(orders, hdu.data)),
                    dict(zip(orders, wvlsol)))

        fitted_pixels_oh = ref_lines_db.identify(obsset.band, spec)
        small_list.append(fitted_pixels_oh)
        small_keys.append("OH")
        
        if obsset.band == "K":
            fitted_pixels_hitran = ref_lines_db_hitrans.identify(obsset.band,
                                                                 spec)
            small_list.append(fitted_pixels_hitran)
            small_keys.append("Hitran")

        fitted_pixels_ = pd.concat(small_list,
                                   keys=small_keys,
                                   names=["kind"],
                                   axis=0)

        fitted_pixels_list.append(fitted_pixels_)

    # concatenate collected list of fitted pixels.
    fitted_pixels_master = pd.concat(fitted_pixels_list,
                                     keys=keys,
                                     names=["slit_center"],
                                     axis=0)

    # storing multi-index seems broken. Enforce reindexing.
    _d = fitted_pixels_master.reset_index()
    obsset.store_data_frame("SKY_FITTED_PIXELS_JSON", _d, orient="split")


def process_band(utdate, recipe_name, band, obsids, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    identify_multiline(helper, band, obsids)


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

    # utdate = "20140709"
    # obsids = [62, 63]

    # utdate = "20140525"
    # obsids = [29]

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
