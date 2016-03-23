import pandas as pd
import numpy as np

from numpy.linalg import lstsq

from nd_poly import NdPolyNamed


def get_center(key_list):
    key_list = sorted(key_list)
    n = len(key_list)
    assert divmod(n, 2)[1] == 1
    center_key = key_list[divmod(n, 2)[0]]
    return center_key


def append_offset(df):
    """
    input should be indexed with multiple values of 'slit_center'.
    Columns of 'pixel0' and 'offsets' will be appended and returned.
    """

    grouped = df.groupby("slit_center")

    slit_center0 = get_center(grouped.groups.keys())
    rename_dict = {'pixels': 'pixels0'}
    center = grouped.get_group(slit_center0).rename(columns=rename_dict)

    pp = df.join(center["pixels0"])

    pp["offsets"] = pp["pixels"] - pp["pixels0"]
    pp_masked = pp[np.isfinite(pp["offsets"])]

    df_offset = pp_masked.reset_index()

    return df_offset


def volume_poly_fit(points, scalar, orders, names):

    p = NdPolyNamed(orders, names)  # order 2 for all dimension.

    v = p.get_array(points)
    v = np.array(v)

    # errors are not properly handled for now.
    s = lstsq(v.T, scalar)

    return p, s


def test_fit():
    df = pd.read_json("test.json", orient="split")
    index_names = ["kind", "order", "wavelength"]
    df = df.set_index(index_names)[["slit_center", "pixels"]]

    dd = append_offset(df)

    names = ["pixel", "order", "slit"]
    orders = [2, 2, 2]

    points = dict(zip(names, [dd["pixels0"], dd["order"], dd["slit_center"]]))
    scalar = dd["offsets"]

    poly, params = volume_poly_fit(points, scalar, orders, names)

    # save
    out_df = poly.to_pandas(coeffs=params[0])
    out_df = out_df.reset_index()

    out_df.to_json("coeffs.json", orient="split")

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

    msk = np.isfinite(ordermap_fits[0].data)
    pixels, orders, slit_pos = (xx[msk], ordermap_fits[0].data[msk],
                                slitposmap_fits[0].data[msk])

    # load coeffs
    # This needs to be fixed
    names = ["pixel", "order", "slit"]

    in_df = pd.read_json("coeffs.json", orient="split")
    in_df = in_df.set_index(names)

    poly, coeffs = NdPolyNamed.from_pandas(in_df)

    values = dict(zip(names, [pixels, orders, slit_pos]))
    offsets = poly.multiply(values, coeffs)

    offset_map = np.empty(ordermap_fits[0].data.shape, dtype=np.float64)
    offset_map.fill(np.nan)
    offset_map[msk] = offsets  # dd["offsets"]




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
