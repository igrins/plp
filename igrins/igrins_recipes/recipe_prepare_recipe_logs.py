import os
import pandas as pd
from ..igrins_libs import dt_logs

from ..igrins_libs.logger import logger
from ..external import argh


_default_recipe_name = dict(flat="FLAT", std="A0V_AB", tar="STELLAR_AB",
                            dark="DARK",
                            arc_thar="ARC", arc_une="ARC")


def get_recipe_name(objtype):

    recipe = _default_recipe_name.get(objtype.lower(), "DEFAULT")

    return recipe


def make_recipe_logs(obsdate, l, populate_group1=False,
                     keep_original=False):

    d3 = l.fillna("-")

    # If the OBJTYPE is flat, we replace its name ot "FLAT ON/OFF" so
    # that it can be assembled in a single recipe even tough their
    # names are different.
    msk = d3["OBJTYPE"].str.lower() == "flat"
    d3.loc[msk, "OBJNAME"] = "FLAT ON/OFF"

    # For arcs, which has OBJTYPE=ARC, FRAMETYPE=ThAr|Une, we replace OBJTYPE
    # with OBJTYPE_FRAMETYPE.
    msk = d3["OBJTYPE"].str.lower() == "arc"
    d3.loc[msk, "OBJTYPE"] = "ARC_" + d3.loc[msk, "FRAMETYPE"].str.upper()

    groupby_keys = ["OBJNAME", "OBJTYPE", "GROUP1", "GROUP2", "EXPTIME"]

    recipe_logs = []
    #~/work/kmtnet for k, group in d3.groupby(groupby_keys, sort=False):
    from ..utils.groupby import groupby
    for k, group in groupby(d3, groupby_keys):
        objname, objtype, group1, group2, exptime = k
        recipe_name = get_recipe_name(objtype)

        obsids = " ".join(group["OBSID"].apply(str))
        frametypes = " ".join(group["FRAMETYPE"])

        if populate_group1:
            group1 = group["OBSID"].iloc[0]

        v = dict(OBJNAME=objname, OBJTYPE=objtype,
                 GROUP1=group1, GROUP2=group2,
                 EXPTIME=exptime,
                 RECIPE=recipe_name,
                 OBSIDS=obsids,
                 FRAMETYPES=frametypes)

        if keep_original:
            v["_rows"] = group

        recipe_logs.append(v)

    headers = groupby_keys + ["RECIPE", "OBSIDS", "FRAMETYPES"]
    if keep_original:
        headers.append("_rows")

    df_recipe_logs = pd.DataFrame(recipe_logs,
                                  columns=headers)

    return df_recipe_logs


def write_to_file(df_recipe_logs, fn_out):

    from six import StringIO
    fout = StringIO()

    headers = df_recipe_logs.keys()
    fout.write(", ".join(headers) + "\n")
    fout.write("# Avaiable recipes : FLAT, SKY, A0V_AB, A0V_ONOFF, "
               "STELLAR_AB, STELLAR_ONOFF, EXTENDED_AB, EXTENDED_ONOFF\n")

    df_recipe_logs.to_csv(fout, index=False, header=False)

    open(fn_out, "w").write(fout.getvalue())


@argh.arg("-n", "--no-tmp-file", default=False)
def prepare_recipe_logs(obsdate, config_file="recipe.config",
                        populate_group1=False, no_tmp_file=False,
                        overwrite=None):

    from ..igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    _fn0 = config.get_value('INDATA_PATH', obsdate)
    fn0 = os.path.join(config.root_dir, _fn0)

    if not os.path.exists(fn0):
        raise RuntimeError("directory {} does not exist.".format(fn0))

    l = dt_logs.load_from_dir(obsdate, fn0)

    dt_logs.update_file_links(obsdate, l, bands="HK")

    df_recipe_logs = make_recipe_logs(obsdate, l,
                                      populate_group1=populate_group1)

    recipe_log_name = config.get_value('RECIPE_LOG_PATH', obsdate)
    if no_tmp_file:
        _fn_out = recipe_log_name
    else:
        _fn_out = recipe_log_name + ".tmp"

    fn_out = os.path.join(config.root_dir, _fn_out)

    if overwrite is None:
        if no_tmp_file:
            overwrite = False
        else:
            overwrite = True

    if not overwrite and os.path.exists(fn_out):
        raise RuntimeError("output file exists and "
                           "overwrite is not set: {}".format(fn_out))

    write_to_file(df_recipe_logs, fn_out)

    # df_recipe_logs.to_html("test2.html")

    if not no_tmp_file:
        print("".join(["A draft version of the recipe log is ",
                       "written to '{}'.\n".format(fn_out),
                       "Make an adjusment and ",
                       "rename it to '{}'.\n".format(recipe_log_name)]))
    else:
        print("".join(["A draft version of the recipe log is ",
                       "written to '{}'.\n".format(fn_out)]))


def test2():
    import os
    rootdir = "/home/jjlee/annex/igrins/"
    obsdate = "20170315"
    dirname = os.path.join(rootdir, obsdate)
    d3 = dt_logs.load_from_dir(obsdate, dirname)
    # d3.to_json("test.json", orient="records")
    df_recipe_logs = make_recipe_logs(obsdate, d3)

    return df_recipe_logs


if False:
    df_recipe_logs = test2()

    write_to_file(df_recipe_logs, "test.csv")

if __name__ == "__main__":
    test2()

