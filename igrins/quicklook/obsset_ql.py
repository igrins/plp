import numpy as np
import pandas as pd
import hashlib
import json

from collections import OrderedDict

from ..storage_interface.db_file import load_key, save_key

from ..pipeline.driver import get_obsset as _get_obsset

from ..pipeline.argh_helper import argh, arg, wrap_multi

from ..igrins_libs.logger import info

from .ql_slit_profile import plot_stacked_profile, plot_per_order_stat
from .ql_flat import plot_flat


def _hash(recipe, band, groupid, basename_postfix, params):
    d = dict(recipe=recipe, band=band, groupid=groupid,
             basename_postfix=basename_postfix,
             params=params)

    h = hashlib.new("sha1")
    h.update(json.dumps(d, sort_keys=True).encode("utf8"))

    return h.hexdigest(), d


class IndexDB(object):
    def __init__(self, storage):
        self.storage = storage
        # self.storage = self.storage.new_sectioned_storage("OUTDATA")

    def check_hexdigest(self, recipe, band, groupid, basename_postfix, params):
        dbname = "index"
        sectionname = recipe

        k = "{}/{:04d}".format(band, groupid)
        v = load_key(self.storage,
                     dbname, sectionname, k)
        if v is None:
            return False

        hexdigest_old, value_old = v
        hexdigest_new, value_new = _hash(recipe, band, groupid,
                                         basename_postfix, params)

        return hexdigest_old == hexdigest_new

    def save_hexdigest(self, recipe, band, groupid, basename_postfix, params):
        dbname = "index"
        sectionname = recipe

        hexdigest, d = _hash(recipe, band, groupid, basename_postfix, params)
        k = "{}/{:04d}".format(band, groupid)
        # k = (groupid, basename_postfix)
        save_key(self.storage, dbname, sectionname, k, [hexdigest, d])

    def save_dtlog(self, band, obsid, param):
        dbname = "index"
        sectionname = "dtlog"

        k = "{}/{:04d}".format(band, obsid)

        save_key(self.storage, dbname, sectionname, k, param)


def get_obsset(obsdate, recipe_name, band,
               obsids, frametypes,
               groupname=None, recipe_entry=None,
               config_file=None, saved_context_name=None,
               basename_postfix=""):

    obsset = _get_obsset(obsdate, recipe_name, band,
                         obsids, frametypes,
                         groupname=groupname, recipe_entry=recipe_entry,
                         config_file=config_file,
                         saved_context_name=saved_context_name,
                         basename_postfix=basename_postfix)
    return obsset


driver_args = [arg("-b", "--bands", default="HK"),
               arg("-o", "--obsids", default=None),
               arg("-t", "--objtypes", default=None),
               arg("-f", "--frametypes", default=None),
               arg("-c", "--config-file", default=None),
               arg("-v", "--verbose", default=0),
               arg("-ns", "--no-skip", default=False),
               # arg("--resume-from-context-file", default=None),
               # arg("--save-context-on-exception", default=False),
               arg("-d", "--debug", default=False)]


def _get_obsid_obstype_frametype_list(config, obsdate,
                                      obsids, objtypes, frametypes):

    from ..igrins_libs import dt_logs

    if None not in [obsids, objtypes, frametypes]:
        return zip(obsids, objtypes, frametypes)

    fn0 = config.get_value('INDATA_PATH', obsdate)
    df = dt_logs.load_from_dir(obsdate, fn0)

    keys = ["OBSID", "OBJNAME", "FRAMETYPE", "OBJTYPE", "EXPTIME", "ROTPA"]
    m = df[keys].set_index("OBSID").to_dict(orient="index")

    if obsids is None:
        if (objtypes is not None) or (frametypes is not None):
            raise ValueError("objtypes and frametypes should not be None when obsids is None")

        obsids = sorted(m.keys())
        # obsids.sort()

    if objtypes is None:
        objtypes = [m[o]["OBJTYPE"] for o in obsids]

    if frametypes is None:
        frametypes = [m[o]["FRAMETYPE"] for o in obsids]

    dt_rows = [m[o] for o in obsids]

    return zip(obsids, objtypes, frametypes, dt_rows)


def do_ql_flat(obsset):
    from ..quicklook import ql_flat

    hdus = obsset.get_hdus()
    jo_raw_list = []
    jo_list = []
    for hdu, oi, ft in zip(hdus, obsset.obsids, obsset.frametypes):
        jo = ql_flat.do_ql_flat(hdus[0], ft)
        jo_list.append((oi, jo))
        jo_raw_list.append((oi, dict()))

    return jo_list, jo_raw_list


def do_ql_std(obsset, band):
    from ..quicklook import ql_slit_profile

    hdus = obsset.get_hdus()
    jo_list = []
    jo_raw_list = []
    for hdu, oi, ft in zip(hdus, obsset.obsids, obsset.frametypes):
        jo, jo_raw = ql_slit_profile.do_ql_slit_profile(hdus[0], band, ft)
        jo_list.append((oi, jo))
        jo_raw_list.append((oi, jo_raw))

    return jo_list, jo_raw_list


do_ql_tar = do_ql_std


def save_jo_list(obsset, jo_list, jo_raw_list):
    item_desc = ("QL_PATH", "{basename}{postfix}.quicklook.json")
    for oi, jo in jo_list:
        obsset.rs.store(str(oi), item_desc, jo)

    item_desc = ("QL_PATH", "{basename}{postfix}.quicklook_raw.json")
    for oi, jo in jo_raw_list:
        obsset.rs.store(str(oi), item_desc, jo)


def save_fig_list(obsset, oi, fig_list):
    from qa_helper import figlist_to_pngs

    pngs = figlist_to_pngs(fig_list)
    for i, png in enumerate(pngs):
        item_desc = ("QL_PATH",
                     "{basename}{postfix}i"
                     + ".fig{:02d}.png".format(i))
        obsset.rs.store(str(oi), item_desc, png)


def oi_ot_ft_generator(recipe_name,
                       obsdate, obsids=None, objtypes=None, frametypes=None,
                       bands="HK", **kwargs):

    import os
    from ..igrins_libs.igrins_config import IGRINSConfig

    config_file = kwargs.pop("config_file", None)
    if config_file is not None:
        config = IGRINSConfig(config_file)
    else:
        config = IGRINSConfig("recipe.config")

    fn0 = config.get_value('INDATA_PATH', obsdate)

    if not os.path.exists(fn0):
        raise RuntimeError("directory {} does not exist.".format(fn0))

    if isinstance(obsids, str):
        obsids = [int(_) for _ in obsids.split(",")]

    oi_ot_ft_list = _get_obsid_obstype_frametype_list(config, obsdate,
                                                      obsids, objtypes,
                                                      frametypes)

    no_skip = kwargs.pop("no_skip", False)

    for b in bands:
        for oi, ot, ft, dt_row in oi_ot_ft_list:
            obsset = get_obsset(obsdate, "quicklook", b,
                                obsids=[oi], frametypes=[ft],
                                config_file=config_file)
            storage = obsset.rs.storage.new_sectioned_storage("OUTDATA_PATH")
            index_db = IndexDB(storage)

            index_db.save_dtlog(b, oi, dt_row)

            if (not no_skip and
                index_db.check_hexdigest(recipe_name, b, oi, "",
                                         dict(obstype=ot, frametype=ft))):
                info("{band}/{obsid:04d} - skipping. already processed"
                     .format(band=b, obsid=oi, objtype=ot))
                continue

            stat = (yield b, oi, ot, ft, dt_row, obsset)

            print("send:", stat)
            if stat:
                index_db.save_hexdigest(recipe_name, b, oi, "",
                                        dict(obstype=ot, frametype=ft))


def quicklook_decorator(recipe_name):
    def _decorated(fun):
        def _f(obsdate, obsids=None, objtypes=None, frametypes=None,
               bands="HK", **kwargs):
            cgen = oi_ot_ft_generator(recipe_name, obsdate,
                                      obsids, objtypes,
                                      frametypes, bands, **kwargs)
            stat = None
            while True:
                try:
                    _ = cgen.send(stat)
                except StopIteration:
                    break

                (b, oi, ot, ft, dt_row, obsset) = _
                print("# entering", _)
                fun(b, oi, ot, ft, dt_row, obsset)

                stat = True
        return _f
    return _decorated


@quicklook_decorator("quicklook")
def quicklook_func(b, oi, ot, ft, dt_row, obsset):

    if ot == "FLAT":
        jo_list, jo_raw_list = do_ql_flat(obsset)
        # print(len(jo_list), jo_list[0][1]["stat_profile"])
        # df = pd.DataFrame(jo_list[0][1]["stat_profile"])
        save_jo_list(obsset, jo_list, jo_raw_list)

        jo = jo_list[0][1]
        fig1 = plot_flat(jo)

        save_fig_list(obsset, oi, [fig1])

    elif ot in ["STD"]:
        info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
             .format(band=b, obsid=oi, objtype=ot))

        jo_list, jo_raw_list = do_ql_std(obsset, b)
        # df = pd.DataFrame(jo_list[0][1]["stat_profile"])
        # print(df[["y", "t_down_10", "t_up_90"]])
        save_jo_list(obsset, jo_list, jo_raw_list)

        jo = jo_list[0][1]
        jo_raw = jo_raw_list[0][1]

        fig1 = plot_stacked_profile(jo)
        fig2 = plot_per_order_stat(jo_raw, jo)

        save_fig_list(obsset, oi, [fig1, fig2])

    elif ot in ["TAR"]:
        info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
             .format(band=b, obsid=oi, objtype=ot))

        jo_list, jo_raw_list = do_ql_std(obsset, b)
        # df = pd.DataFrame(jo_list[0][1]["stat_profile"])
        # print(df[["y", "t_down_10", "t_up_90"]])
        save_jo_list(obsset, jo_list, jo_raw_list)

        jo = jo_list[0][1]
        jo_raw = jo_raw_list[0][1]

        fig1 = plot_stacked_profile(jo)
        fig2 = plot_per_order_stat(jo_raw, jo)

        save_fig_list(obsset, oi, [fig1, fig2])

    else:
        info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
             .format(band=b, obsid=oi, objtype=ot))


def get_guard_column_pattern(d):
    from igrins.procedures.readout_pattern import pipes
    pipenames_dark1 = ['amp_wise_bias_r2', 'p64_0th_order']

    guards = d[:, [0, 1, 2, 3, -4, -3, -2, -1]]

    pp = OrderedDict()
    for k in pipenames_dark1:
        p = pipes[k]
        _ = p.get(guards)
        guards = guards - p.broadcast(guards, _)
        pp[k] = _

    guards = guards - np.median(guards)

    return guards, pp


def get_column_percentile(guards, percentiles=None):
    if percentiles is None:
        percentiles = [10, 90]
    # guards = d[:, [0, 1, 2, 3, -4, -3, -2, -1]]
    r = OrderedDict(zip(percentiles, np.percentile(guards, percentiles)))
    r["std"] = np.std(guards[(r[10] < guards) & (guards < r[90])])
    return r


@quicklook_decorator("noise_guard")
def noise_guard_func(b, oi, ot, ft, dt_row, obsset):
    if True:
        hdus = obsset.get_hdus()
        assert len(hdus) == 1
        d = hdus[0].data
        guard, pp = get_guard_column_pattern(d)
        percent = get_column_percentile(guard)

        item_desc = ("QL_PATH", "{basename}{postfix}.noise_guard.json")
        obsset.rs.store(str(oi), item_desc, dict(percentiles=percent,
                                                 pattern_noise=pp))

    else:
        info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
             .format(band=b, obsid=oi, objtype=ot))


def quicklook_func_deprecated(obsdate, obsids=None, objtypes=None, frametypes=None,
                              bands="HK", **kwargs):
    import os
    from ..igrins_libs.igrins_config import IGRINSConfig

    config_file = kwargs.pop("config_file", None)
    if config_file is not None:
        config = IGRINSConfig(config_file)
    else:
        config = IGRINSConfig("recipe.config")

    fn0 = config.get_value('INDATA_PATH', obsdate)

    if not os.path.exists(fn0):
        raise RuntimeError("directory {} does not exist.".format(fn0))

    if isinstance(obsids, str):
        obsids = [int(_) for _ in obsids.split(",")]

    oi_ot_ft_list = _get_obsid_obstype_frametype_list(config, obsdate,
                                                      obsids, objtypes,
                                                      frametypes)

    no_skip = kwargs.pop("no_skip", False)

    for b in bands:
        for oi, ot, ft, dt_row in oi_ot_ft_list:
            obsset = get_obsset(obsdate, "quicklook", b,
                                obsids=[oi], frametypes=[ft],
                                config_file=config_file)
            storage = obsset.rs.storage.new_sectioned_storage("OUTDATA_PATH")
            index_db = IndexDB(storage)

            index_db.save_dtlog(b, oi, dt_row)

            if (not no_skip and
                index_db.check_hexdigest("quicklook", b, oi, "",
                                         dict(obstype=ot, frametype=ft))):
                info("{band}/{obsid:04d} - skipping. already processed"
                     .format(band=b, obsid=oi, objtype=ot))
                continue

            if ot == "FLAT":
                jo_list, jo_raw_list = do_ql_flat(obsset)
                # print(len(jo_list), jo_list[0][1]["stat_profile"])
                df = pd.DataFrame(jo_list[0][1]["stat_profile"])
                print(df[["y", "t_down_10", "t_up_90"]])
                save_jo_list(obsset, jo_list, jo_raw_list)

                jo = jo_list[0][1]
                fig1 = plot_flat(jo)

                save_fig_list(obsset, oi, [fig1])

            elif ot in ["STD"]:
                info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
                     .format(band=b, obsid=oi, objtype=ot))

                jo_list, jo_raw_list = do_ql_std(obsset, b)
                # df = pd.DataFrame(jo_list[0][1]["stat_profile"])
                # print(df[["y", "t_down_10", "t_up_90"]])
                save_jo_list(obsset, jo_list, jo_raw_list)

                jo = jo_list[0][1]
                jo_raw = jo_raw_list[0][1]

                fig1 = plot_stacked_profile(jo)
                fig2 = plot_per_order_stat(jo_raw, jo)

                save_fig_list(obsset, oi, [fig1, fig2])

            elif ot in ["TAR"]:
                info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
                     .format(band=b, obsid=oi, objtype=ot))

                jo_list, jo_raw_list = do_ql_std(obsset, b)
                # df = pd.DataFrame(jo_list[0][1]["stat_profile"])
                # print(df[["y", "t_down_10", "t_up_90"]])
                save_jo_list(obsset, jo_list, jo_raw_list)

                jo = jo_list[0][1]
                jo_raw = jo_raw_list[0][1]

                fig1 = plot_stacked_profile(jo)
                fig2 = plot_per_order_stat(jo_raw, jo)

                save_fig_list(obsset, oi, [fig1, fig2])

            else:
                info("{band}/{obsid:04d} - unsupported OBJTYPE:{objtype}"
                     .format(band=b, obsid=oi, objtype=ot))
                continue

            index_db.save_hexdigest("quicklook", b, oi, "",
                                    dict(obstype=ot, frametype=ft))


def create_argh_command_quicklook():

    func = wrap_multi(quicklook_func, driver_args)
    func = argh.decorators.named("quicklook")(func)

    return func


def create_argh_command_noise_guard():

    func = wrap_multi(noise_guard_func, driver_args)
    func = argh.decorators.named("noise-guard")(func)

    return func
