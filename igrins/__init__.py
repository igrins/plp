import os

### NEW

from .igrins_libs.resource_helper_igrins import ResourceHelper

from .igrins_libs.storage_descriptions import load_descriptions

DESCS = load_descriptions()


###

# def get_caldb(config_name, utdate, ensure_dir=False):
#     from .libs.recipe_helper import RecipeHelper
#     helper = RecipeHelper(config_name, utdate, ensure_dir=ensure_dir)
#     caldb = helper.get_caldb()
#     return caldb

# def get_obsset(caldb, band, recipe_name, obsids, frametypes):
#     from .libs.obs_set import ObsSet
#     obsset = ObsSet(caldb, band, recipe_name, obsids, frametypes)
#     return obsset

# def get_obsset_from_log(caldb, band, log_entry):
#     recipe_name, obsids, frametypes = (log_entry["recipe"],
#                                        log_entry["obsids"],
#                                        log_entry["frametypes"])
                                       
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)
#     return obsset

# def get_recipe_log(caldb):
#     config = caldb.get_config()
#     fn0 = config.get_value('RECIPE_LOG_PATH', caldb.utdate)
#     fn = os.path.join(config.root_dir, fn0)
#     from igrins.libs.recipes import RecipeLog
#     recipe_log = RecipeLog(fn)

#     return recipe_log

# def get_obsset_from_log(caldb, band, log_entry):
#     recipe_name, obsids, frametypes = (log_entry["recipe"],
#                                        log_entry["obsids"],
#                                        log_entry["frametypes"])
                                       
#     obsset = get_obsset(caldb, band, recipe_name, obsids, frametypes)
#     return obsset


def load_recipe_log(obsdate, config_file=None):
    from .igrins_libs.igrins_config import IGRINSConfig
    config = IGRINSConfig(config_file=config_file)
    fn = os.path.join(os.path.dirname(config.config_file),
                      config.get_value('RECIPE_LOG_PATH', obsdate))

    from .igrins_libs.recipes import RecipeLog
    recipe_log = RecipeLog(fn)

    return recipe_log


def get_obsset(obsdate, band, recipe_name_or_entry,
               obsids=None, frametypes=None,
               groupname=None, recipe_entry=None,
               config_file=None):
    if isinstance(recipe_name_or_entry, str):
        recipe_name = recipe_name_or_entry
    else:
        r = recipe_name_or_entry
        recipe_name = r["recipe"]
        obsids = r["obsids"]
        frametypes = r["frametypes"]
        groupname = r["group1"]

    from .pipeline.driver import get_obsset as _get_obsset
    obsset = _get_obsset(obsdate, recipe_name, band, obsids, frametypes,
                         groupname, recipe_entry,
                         config_file=config_file)
    return obsset


def get_obsset_helper(obsset):
    return ResourceHelper(obsset)
