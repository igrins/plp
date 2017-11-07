import os



### NEW

from .libs.storage_descriptions import load_descriptions

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

