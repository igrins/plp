from ..pipeline.driver import get_obsset as _get_obsset

from ..pipeline.argh_helper import argh, arg, wrap_multi


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

    keys = ["OBSID", "FRAMETYPE", "OBJTYPE"]
    m = df[keys].set_index("OBSID").to_dict(orient="index")

    if obsids is None:
        if (objtypes is not None) or (frametypes is not None):
            raise ValueError("objtypes and frametypes should not be None when obsids is None")

        obsids = m.keys()
        obsids.sort()

    if objtypes is None:
        objtypes = [m[o]["OBJTYPE"] for o in obsids]

    if frametypes is None:
        frametypes = [m[o]["FRAMETYPE"] for o in obsids]

    return zip(obsids, objtypes, frametypes)


def do_ql_flat(obsset):
    from ..quicklook import ql_flat

    hdus = obsset.get_hdus()
    jo_list = []
    for hdu, oi, ft in zip(hdus, obsset.obsids, obsset.frametypes):
        jo = ql_flat.do_ql_flat(hdus[0], ft)
        jo_list.append((oi, jo))

    return jo_list


def save_jo_list(obsset, jo_list):
    item_desc = ("OUTDATA_PATH", "{basename}{postfix}.quicklook.json")
    for oi, jo in jo_list:
        obsset.rs.store(str(oi), item_desc, jo)


def quicklook_func(obsdate, obsids=None, objtypes=None, frametypes=None,
                   bands="HK", **kwargs):
    import os
    from ..igrins_libs.igrins_config import IGRINSConfig

    print(obsdate, obsids)
    config_file = kwargs.pop("config_file", None)
    if config_file is not None:
        config = IGRINSConfig(config_file)
    else:
        config = IGRINSConfig("recipe.config")

    fn0 = config.get_value('INDATA_PATH', obsdate)

    if not os.path.exists(fn0):
        raise RuntimeError("directory {} does not exist.".format(fn0))

    if isinstance(obsids, str):
        obsids = map(int, obsids.split(","))

    oi_ot_ft_list = _get_obsid_obstype_frametype_list(config, obsdate,
                                                      obsids, objtypes,
                                                      frametypes)

    for b in bands:
        for oi, ot, ft in oi_ot_ft_list:
            obsset = get_obsset(obsdate, "quicklook", b,
                                obsids=[oi], frametypes=[ft],
                                config_file=config_file)
            print(obsset)
            obsset
            if ot == "FLAT":
                jo_list = do_ql_flat(obsset)
                save_jo_list(obsset, jo_list)

            elif ot in ["TAR", "STD"]:
                pass
            else:
                pass


def create_argh_command_quicklook():

    func = wrap_multi(quicklook_func, driver_args)
    func = argh.decorators.named("quicklook")(func)

    return func
