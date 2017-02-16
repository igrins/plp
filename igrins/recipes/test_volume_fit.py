import pandas as pd
import numpy as np



def test_fit():

    df = pd.read_json("test.json", orient="split")
    index_names = ["kind", "order", "wavelength"]
    df = df.set_index(index_names)[["slit_center", "pixels"]]

    dd = append_offset(df)

    names = ["pixel", "order", "slit"]
    orders = [3, 2, 1]

    # because the offset at slit center should be 0, we divide the
    # offset by slit_pos, and fit the data then multiply by slit_pos.
    
    cc0 = dd["slit_center"] - 0.5

    points0 = dict(zip(names, [dd["pixels0"],
                               dd["order"],
                               cc0]))
    scalar0 = dd["offsets"]

    
    msk = abs(cc0) > 0.

    points = dict(zip(names, [dd["pixels0"][msk],
                              dd["order"][msk],
                              cc0[msk]]))

    scalar = dd["offsets"][msk] / cc0[msk]


    poly, params = volume_poly_fit(points, scalar, orders, names)

    if 0:
        #values = dict(zip(names, [pixels, orders, slit_pos]))
        offsets_fitted = poly.multiply(points0, params[0])
        doffsets = scalar0 - offsets_fitted * cc0

        clf()
        scatter(dd["pixels0"], doffsets, c=cc0.values, cmap="gist_heat")

        # clf()
        # scatter(dd["pixels0"] + doffsets, dd["order"] + dd["slit_center"], color="g")
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"], color="r")


        # # test with fitted data
        # #input_points = np.zeros_like(offsets_fitted)
        # input_points = offsets_fitted
        # poly, params = volume_poly_fit(points,
        #                                input_points,
        #                                orders, names)

        # offsets_fitted = poly.multiply(points, params[0])
        # doffsets = input_points - offsets_fitted

        # clf()
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"] + doffsets, color="g")
        # scatter(dd["pixels0"], dd["order"] + dd["slit_center"], color="r")
        
    # save
    out_df = poly.to_pandas(coeffs=params[0])
    out_df = out_df.reset_index()

    d = out_df.to_dict(orient="split")
    import json
    json.dump(d, open("coeffs.json", "w"))

if 0:
    # read
    in_df = pd.read_json("coeffs.json", orient="split")
    in_df = in_df.set_index(["pixel", "order", "slit"])

    poly, coeffs = NdPolyNamed.from_pandas(in_df)


def process_band_make_offset_map(utdate, recipe_name, band,
                                 obsids, config_name):

    from libs.recipe_helper import RecipeHelper
    helper = RecipeHelper(config_name, utdate, recipe_name)

    caldb = helper.get_caldb()

    master_obsid = obsids[0]
    basename = (band, master_obsid)

    ordermap_fits = caldb.load_resource_for(basename,
                                            ("sky", "ordermap_fits"))

    slitposmap_fits = caldb.load_resource_for(basename,
                                              ("sky", "slitposmap_fits"))

    # slitoffset_fits = caldb.load_resource_for(basename,
    #                                           ("sky", "slitoffset_fits"))

    yy, xx = np.indices(ordermap_fits[0].data.shape)

    msk = np.isfinite(ordermap_fits[0].data) & (ordermap_fits[0].data > 0)
    pixels, orders, slit_pos = (xx[msk], ordermap_fits[0].data[msk],
                                slitposmap_fits[0].data[msk])

    # load coeffs
    # This needs to be fixed
    names = ["pixel", "order", "slit"]

    in_df = pd.read_json("coeffs.json", orient="split")
    in_df = in_df.set_index(names)

    poly, coeffs = NdPolyNamed.from_pandas(in_df)

    cc0 = slit_pos - 0.5
    values = dict(zip(names, [pixels, orders, cc0]))
    offsets = poly.multiply(values, coeffs) # * cc0

    offset_map = np.empty(ordermap_fits[0].data.shape, dtype=np.float64)
    offset_map.fill(np.nan)
    offset_map[msk] = offsets * cc0 # dd["offsets"]
    

    if 0:
        slitoffset_fits = caldb.load_resource_for(basename,
                                                  ("sky", "slitoffset_fits"))
        offset_map_orig = slitoffset_fits[0].data
        


if 0:
    xy = np.indices([60, 2048]) / np.array([60., 1.])[:, np.newaxis, np.newaxis]

    p93, params93 = p.freeze("order", 93, params)
    offset93 = p93.multiply(dict(pixel=xy[1], slit=xy[0]), params93)

    p72, params72 = p.freeze("order", 72, params)
    offset72 = p93.multiply(dict(pixel=xy[1], slit=xy[0]), params72)


if 0:
    offsets_fit = p.multiply(values, params)

    #plot(values["slit"], offsets, "x")
    plot(values["slit"], offsets_fit - offsets, "o")


if __name__ == "__main__":
    utdate = "20150525"
    obsids = [52]

    recipe_name = "SKY"

    band = "K"

    #helper = RecipeHelper("../recipe.config", utdate)
    config_name = "../recipe.config"

    process_band_make_offset_map(utdate, recipe_name, band,
                                 obsids, config_name)
