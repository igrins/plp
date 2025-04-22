from __future__ import print_function

import os
import numpy as np
import pandas as pd
from ..igrins_libs import dt_logs

from ..igrins_libs.logger import logger
from ..external import argh


_default_recipe_name = dict(flat="FLAT", std="A0V_AB", tar="STELLAR_AB",
                            sky='SKY',
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

        if objtype.lower() == 'sky' or objname=='Blank sky' or objname=='SKY 300s':
            objtype = 'TAR'
            recipe_name = 'SKY'

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
    fout.write("# Available recipes : FLAT, SKY, A0V_AB, A0V_ONOFF, "
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

    dt_logs.update_file_links(config, obsdate, l, bands="HK")

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


def _fmt_exptime(row, max_n, max_len):
    # The tabulate module seems to strip the string. So it does not work okay
    etime = float(row["exptime"])
    if etime < 2.:
        etime = "{:.1f}".format(etime)
    else:
        etime = str(int(etime))

    fmt = "{{:>{}}} x {{:>{}}}".format(max_n, max_len)

    return fmt.format(etime, len(row["obsids"]))


def fmt_exptime(row):
    etime = float(row["exptime"])
    if etime < 2.:
        etime = "{:.1f}".format(etime)
    else:
        etime = str(int(etime))

    return "{} x {}".format(etime, len(row["obsids"]))


import re

def get_replaced_pattern(s):
    """
    replace patterns like 'ABAABA' or 'ABBAABBA' to '(ABA)x2' or '(ABBA)x2'
    """
    p = re.compile(r'(((.)(.)(\4)?\3){2,})')
    ps = re.compile(r'\s+')

    s1 = s
    m = p.search(s1)

    while m:
        st, c, _, _, _ = m.groups()
        n = len(st) // len(c)
        r = " ({})x{} ".format(c, n)
        s1, _ = p.subn(r, s1, 1)
        m = p.search(s1)

    s1 = ps.sub(" ", s1)
    # s1 = p.sub(" ", s1)
    return s1.strip()


def get_replaced_repeated(s):
    """
    replace patterns like 'OOOO' (or longer) to '0x4'
    """
    p = re.compile(r'((.)\2{3,})')
    ps = re.compile(r'\s+')

    s1 = s
    m = p.search(s1)
    while m:
        st, c = m.groups()
        r = " {}x{} ".format(c, len(st))
        s1 = p.sub(r, s1, 1)
        m = p.search(s1)

    s1 = ps.sub(" ", s1)
    return s1.strip()


def fmt_frames(frames):
    frames = [dict(on="O", off="S").get(f.lower(), f) for f in frames]

    return get_replaced_pattern(get_replaced_repeated("".join(frames)))


def fmt_obsids(obsids):
    oo = []
    old_o = np.nan

    for o in obsids:
        o = int(o)
        if o - old_o == 1:
            oo[-1].append(o)
        else:
            oo.append([o])
        old_o = o

    for o1 in oo:
        o1[1:-1] = []

    return ",".join(["-".join(map(str, o1)) for o1 in oo])


def tabulated_recipe_logs(obsdate, config):

    fn = os.path.join(config.root_dir,
                      config.get_value('RECIPE_LOG_PATH', obsdate))

    from ..igrins_libs.recipes import RecipeLog
    recipes = RecipeLog(obsdate, fn)

    recipes["obsid-fmt"] = recipes["obsids"].apply(fmt_obsids)
    recipes["frame-fmt"] = recipes["frametypes"].apply(fmt_frames)

    # # length of the exptime string
    # max_n = int(np.log10(max(recipes["exptime"]))) + 1
    # max_len = int(np.log10(max(recipes["obsids"].apply(len)))) + 1

    recipes["exptime-fmt"] = recipes.apply(fmt_exptime, axis=1)

    from ..external.tabulate import tabulate
    r = tabulate(recipes[["group1", "objname", "recipe",
                          "exptime-fmt", "obsid-fmt", "frame-fmt"]],
                 headers=["GID", "Obj. Name", "Recip",
                          "ExpTime", "ObsIDs", "Frames"],
                 tablefmt='psql', showindex=False)

    return r


def show_recipe_logs(obsdate, config_file="recipe.config"):

    from ..igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file)

    r = tabulated_recipe_logs(obsdate, config)
    print(r)


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

