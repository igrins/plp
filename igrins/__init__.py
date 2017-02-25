

def get_caldb(config_name, utdate):
    from .libs.recipe_helper import RecipeHelper
    helper = RecipeHelper(config_name, utdate)
    caldb = helper.get_caldb()
    return caldb

def get_obsset(caldb, recipe_name, band, obsids, frametypes):
    from .libs.obs_set import ObsSet
    obsset = ObsSet(caldb, recipe_name, band, obsids, frametypes)
    return obsset
